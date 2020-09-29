#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <string_view>
#include <iostream>
#include <fstream> 
#include <sstream>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm> 
#include <chrono>
#include <utility>
#include <limits>

#include "gmcp.hpp"
#include "utils.hpp"
#include "video.hpp"

#include <opencv2/opencv.hpp>

void get_parameters(int argc, char **argv, int& segment_size, std::string& in_video, std::string& out_video, 
                    std::string& detector, std::string& detector_cfg, std::string& tmp_folder)
{
    int opt;
    while((opt = getopt(argc, argv, "s:i:o:d:c:f:h")) != -1)
    {
        switch (opt)
        {
            case 's':
                segment_size = std::stoi(optarg);
                break;
            case 'i':
                in_video = optarg;
                break;
            case 'o':
                out_video = optarg;
                break;
            case 'd':
                detector = optarg;
                break;
            case 'c':
                detector_cfg = optarg;
                break;
            case 'f':
                tmp_folder = optarg;
                break;
            case 'h':
                print_usage_info();
                exit(0);
            default:
                std::cout << "Unsupported parameter passed to the script. Aborting." << std::endl;
                print_usage_info();
                abort();
        }
    }
}

void verify_parameters(int segment_size, std::string in_video, std::string out_video, 
                       std::string detector, std::string detector_cfg, std::string tmp_folder)
{
    if (segment_size < 3)
    {
        std::cout << "Segment size has to be at least 3. Aborting.";
        exit(0);
    }
    if (in_video == "")
    {
        std::cout << "Please specify input video path. Aborting.";
        exit(0);
    }
    if (out_video == "")
    {
        std::cout << "Please specify output video path. Aborting.";
        exit(0);
    }
    if (detector != "yolo" && detector != "ssd")
    {
        std::cout << "Unsupported detector. Aborting.";
        exit(0);
    }
    if (detector_cfg == "")
    {
        std::cout << "Please specify detector config path. Aborting.";
        exit(0);
    }
    if (tmp_folder == "")
    {
        std::cout << "Please specify path where tmp files should be stored. Aborting.";
        exit(0);
    }
}

std::string get_frame_path(int frame, std::string tmp_folder)
{
    std::stringstream ss;
    ss << tmp_folder << "/img/frame" << std::to_string(frame) << ".jpeg";
    std::string path = ss.str();
    return path;
}

void detect(std::string detector, std::string detector_cfg, int frame_cnt, std::string video, std::string tmp_folder) 
{
    // initiates object detections on selected frames of the trimmed video
    // the detections are then stored in csv files
    std::stringstream ss;
    ss << "python3 ../detectors/detect.py --detector " << detector 
       << " --cfg " << detector_cfg
       << " --video " << video
       << " --tmp_folder " << tmp_folder
       << " --frame_cnt " << std::to_string(frame_cnt);
    std::string detect_command = ss.str();
    std::cout << "Executing: " << detect_command << std::endl;
    system(detect_command.c_str());
}

vector<Detection> load_frame_detections(std::string csv_file) 
{
    vector<Detection> detections;
    std::ifstream f(csv_file);
    std::string line, colname;
    int val;

    if(!f.is_open()) 
         throw std::runtime_error("Could not open file");

    // one bounding box a line
    int id = 0;
    int x_min, y_min, x_max, y_max, x_center, y_center;
    while(std::getline(f, line))
    {
        std::stringstream ss(line); 
        int coords[4];
        constexpr int const coord_cnt = 4; 
        for (int i = 0; i < coord_cnt; i++)
        {
            std::string substr;
            getline(ss, substr, ',');
            coords[i] = std::stoi(substr);
        }
        x_min = coords[0];
        y_min = coords[1];
        x_max = coords[2];
        y_max = coords[3];
        x_center = (x_min + x_max) / 2;
        y_center = (y_min + y_max) / 2;
        Detection det = {
            .x = x_center,
            .y = y_center,
            .id = id,
            .x_min = x_min,
            .y_min = y_min,
            .x_max = x_max,
            .y_max = y_max
        };
        id++;
        detections.push_back(det);
    }
    f.close();
    return detections;
}

vector2d<Detection> load_detections(int frame_cnt, std::string tmp_folder) 
{
    // loads a vector of Detections for each frame
    vector2d<Detection> detections(frame_cnt, vector<Detection>());
    for (int i = 0; i < frame_cnt; i += 1) 
    {
        std::stringstream ss;
        ss << tmp_folder << "/csv/frame" << i << ".csv";
        std::string csv_path = ss.str();
        detections[i] = load_frame_detections(csv_path);
    }
    print_detections(detections);
    return detections;
}

int get_max_detections_per_frame(const vector2d<Detection> &detections)
{
    // checks what is the biggest number of detections in a single frame
    int maxi = 0;
    for(auto const& d : detections)
        if (d.size() > maxi)
            maxi = d.size();
    std::cout << "Max number of detections per frame is: " << maxi << std::endl;
    return maxi;
}

auto get_detection_histograms(const vector2d<Detection> &detections,
                              int frame_cnt, int segment_cnt, int segment_size,
                              const std::string &tmp_folder)
{
    // vector of Detections for each frame
    
    // vector2d<Detection> centers(frame_cnt, vector<Detection>());
    
    // vector of histogram matrices of detection images for each frame
    // detection of id==0 has its histogram at the start of the vector and so on and on
    vector2d<cv::Mat> histograms(frame_cnt, vector<cv::Mat>()); 
    constexpr int channels[3] = {0, 1, 2};
    constexpr float range[2] = {0, 256};
    const float * ranges[3] = {range, range, range};
    constexpr int histSize[3] = {8, 8, 8};
    for (int i = 0; i < segment_cnt; i++) 
    {
        int start_frame = segment_size * i;
        for (int j = start_frame; j < start_frame + segment_size; j++)
        {
            cv::Mat frame = cv::imread(get_frame_path(j, tmp_folder));
            for(auto const& d: detections[j])
            {
                int width = d.x_max - d.x_min;
                int height = d.y_max - d.y_min;                
                cv::Mat hist;
                cv::Mat detection = frame(cv::Rect(d.x, d.y, width, height));
                cv::calcHist(&detection, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
                histograms[j].push_back(hist); // 
            }
        }
    }
    return histograms;
}

auto get_net_cost(int frame_cnt, const vector2d<cv::Mat> &histograms)
{  
    vector2d<HistInterKernel> net_cost(frame_cnt, vector<HistInterKernel>());
    for (int i = 0; i < frame_cnt - 1; i++) // from which frame
        for (int k = 0; k < histograms[i].size(); k++) // from which detection
            for (int l = 0; l < histograms[i+1].size(); l++) // to which detection
            {
                double histogram_intersection_kernel = cv::compareHist(histograms[i][k], 
                                                                       histograms[i+1][l],
                                                                       3); // CV_COMP_INTERSECT
                HistInterKernel hik = {
                    .detection_id1 = k,
                    .detection_id2 = l,
                    .value = histogram_intersection_kernel
                };
                net_cost[i].push_back(hik);
            }
    return net_cost;
}

HistInterKernel get_cheapest(const vector<HistInterKernel> &hiks)
{
    double maxi = std::numeric_limits<double>::max();
    HistInterKernel mini = {
        .detection_id1 = -1,
        .detection_id2 = -1,
        .value = maxi
    };
    for(auto hik: hiks)
    {
        if (hik.value < mini.value)
        {
            mini = hik;
        }
    }
    return mini;
}

HistInterKernel get_cheapest(const vector<HistInterKernel> &hiks, int detection_id1)
{
    double maxi = std::numeric_limits<double>::max();
    HistInterKernel mini = {
        .detection_id1 = -1,
        .detection_id2 = -1,
        .value = maxi
    };
    for(auto hik: hiks)
    {   
        if (hik.detection_id1 == detection_id1 && hik.value < mini.value)
        {
            mini = hik;
        }
    }
    return mini;
}

vector<int> get_initial_detection_path(const vector2d<HistInterKernel> &net_cost,
                                       int seg_size, int seg_counter)
{
    // returns ids of detections in subsequent frames that form a brute-force solution
    vector<int> detection_ids;
    detection_ids.reserve(seg_size); // one detection from each frame in a segment

    int start = seg_counter * seg_size;
    auto first_hik = get_cheapest(net_cost[start]);
    detection_ids.push_back(first_hik.detection_id1);
    detection_ids.push_back(first_hik.detection_id2);

    for (int i = 1; i < seg_size - 1; i++)
    {
        auto hik = get_cheapest(net_cost[start+i], detection_ids.back());
        detection_ids.push_back(hik.detection_id2);
    }
    return detection_ids;
}

double get_appearance_cost(const vector2d<HistInterKernel> &net_cost,
                           const vector<int> &detection_ids, int seg_counter)
{
    double cost = 0;
    int component_counter = 0;
    int start = seg_counter * detection_ids.size();
    for (int i = 0; i < detection_ids.size() - 1; i++)
    {
        for(auto hik: net_cost[start+i])
        {
            if (hik.detection_id1 == detection_ids[i] && hik.detection_id2 == detection_ids[i+1])
            {
                cost += hik.value;
                component_counter++;
                break;
            }
        }
    }
    std::cout << "Appearance cost = " << cost 
              << ", based on " << component_counter << " components" << std::endl;
    return cost;
}

vector<Detection> get_detection_path(const vector2d<Detection> &detections,
                                     const vector<int> &detection_ids, int seg_counter)
{
    vector<Detection> path;
    int start = seg_counter * detection_ids.size();
    for (int i = 0; i < detection_ids.size(); i++) // for each detection id == for each frame
    {
        // find Detection with desired id and push it back to the resulting vector
        for(auto detection: detections[start+i])
        {
            if (detection.id == detection_ids[i])
                path.push_back(detection);
        }
    }
    return path;
}

double get_motion_cost(const vector<Detection> &path,
                       const vector<int> &detection_ids, int seg_counter)
{
    double cost = 0;
    vector<int> x_diffs, y_diffs, sums;
    int diff_size = detection_ids.size() - 1;
    sums.reserve(diff_size);
    for (int i = 0; i < diff_size; i++)
    {
        sums.push_back(pow(path[i].x - path[i+1].x, 2) + pow(path[i].y - path[i+1].y, 2));
    }
    for (int i = 0; i < diff_size; i++)
        cost += sums[i];
    cost = sqrt(cost);

    std::cout << "Motion cost     = " << cost << std::endl;
    return cost;
}

void remove_path(vector2d<Detection> &detections, const vector<int> &detection_ids, int seg_counter)
{
    int start = seg_counter * detection_ids.size();
    for(int i = 0; i < detection_ids.size(); i++)
    {
        for(int j = 0; j < detections[start+i].size(); j++)
        {
            if (detections[start+i][j].id == detection_ids[i])
                detections[start+i].erase(detections[start+i].begin() + j);
        }
    }   
}

void remove_path(vector2d<HistInterKernel> &net_cost, const vector<int> &detection_ids, int seg_counter)
{
    int start = seg_counter * detection_ids.size();
    // for each frame except the last one in the segment
    for (int i = 0; i < detection_ids.size() - 1; i++)
    {
        vector<HistInterKernel> tmp;
        for (int j = 0; j < net_cost[start+i].size(); j++)
        {
            if (net_cost[start+i][j].detection_id1 != detection_ids[i] // remove hiks starting at used detection
                && net_cost[start+i][j].detection_id2 != detection_ids[i+1]) // remove hiks leading to used detection
                tmp.push_back(net_cost[start+i][j]);
        }
        net_cost[i] = tmp;
    }   
}

bool is_any_frame_without_detections(vector2d<Detection> &detections, int seg_counter, int seg_size)
{
    int start = seg_counter * seg_size;
    for (int i = 0; i < seg_size; i++)
    {
        if (detections[start+i].empty())
        {
            std::cout << "Not enough detections in this segment for more paths" << std::endl;
            return true;
        }
    }
    return false;
}

auto track(vector2d<Detection> &detections,
           vector2d<HistInterKernel> &net_cost, 
           int segment_cnt, int segment_size, int max_detections_per_frame)
{
    vector3d<Detection> tracklets(segment_cnt, vector2d<Detection>(max_detections_per_frame, vector<Detection>()));

    for (int i = 0; i < segment_cnt; i++)
    {
        std::cout << std::endl << "Tracking in segment " << i+1 << "/" << segment_cnt << std::endl;
        print_detections_left_cnt(detections, i, segment_size);
        int j = 0;
        while (j < max_detections_per_frame)
        {
            print_net_cost(net_cost);
            if (is_any_frame_without_detections(detections, i, segment_size))
                break;
            std::cout << std::endl << "Tracking object number " << j+1 << "/" << max_detections_per_frame << std::endl;
            auto detection_ids = get_initial_detection_path(net_cost, segment_size, i);
            auto path = get_detection_path(detections, detection_ids, i);

            auto app_cost = get_appearance_cost(net_cost, detection_ids, i);
            auto motion_cost = get_motion_cost(path, detection_ids, i);

            tracklets[i][j] = path;
            remove_path(detections, detection_ids, i);
            remove_path(net_cost, detection_ids, i);
            print_detections_left_cnt(detections, i, segment_size);
            print_detections_left_ids(detections, i, segment_size);
            j++;
        }
    }
    print_tracklets(tracklets, segment_cnt);
    return tracklets;
}

int main(int argc, char **argv) {
    auto begin = std::chrono::steady_clock::now();
    int segment_size = 0;
    std::string in_video, out_video, detector, detector_cfg, tmp_folder;
    
    get_parameters(argc, argv, segment_size, in_video, out_video, detector, detector_cfg, tmp_folder);
    verify_parameters(segment_size, in_video, out_video, detector, detector_cfg, tmp_folder);
    print_parameters(segment_size, in_video, out_video, detector, detector_cfg, tmp_folder);
    make_tmp_dirs(tmp_folder);

    cv::VideoCapture in_cap(in_video);
    const int frame_cnt = get_trimmed_frame_cnt(in_cap, segment_size);
    const int segment_cnt = frame_cnt / segment_size;
    const std::string tmp_video = tmp_folder + "/input.mp4";
    prepare_tmp_video(in_cap, frame_cnt, tmp_folder, in_video, tmp_video);

    auto detect_begin = std::chrono::steady_clock::now();
    detect(detector, detector_cfg, frame_cnt - 1, tmp_video, tmp_folder);
    auto detect_end = std::chrono::steady_clock::now();
    auto detections = load_detections(frame_cnt, tmp_folder);
    // auto max_detections_per_frame = get_max_detections_per_frame(detections);
    // auto colors = get_colors(max_detections_per_frame);
    
    
    // auto histograms = get_detection_histograms(detections, 
    //                                            frame_cnt, 
    //                                            segment_cnt, segment_size,
    //                                            tmp_folder);
    // auto net_cost = get_net_cost(frame_cnt, histograms);

    // auto tracklets = track(detections, net_cost, segment_cnt, segment_size, max_detections_per_frame);
    // draw detections on frames
    // create a video out of tmp frames

    clear_tmp(tmp_folder);
    auto end = std::chrono::steady_clock::now();
    print_exec_time(begin, end);
    print_detect_time(detect_begin, detect_end);
    return 0;
}
