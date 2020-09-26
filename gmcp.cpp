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

#include "gmcp.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>


void print_usage_info()
{
    const char* usage_info =
    "\n    -s, --segment_size   frames in a segment\n"
    "    -i, --input_video      input video path\n"
    "    -o, --output_video     output video path\n"
    "    -d, --detector         object detector (ssd or yolo)\n"
    "    -c, --detector_cfg     path to detector cfg\n"
    "    -f, --tmp_fixtures     path to folder where temporary files will be stored\n"
    "    -h, --help             show this help msg";
    std::cout << usage_info << std::endl;
}

void verify_parameters(int segment_size, std::string input_video, std::string output_video, 
                       std::string detector, std::string detector_cfg, std::string tmp_fixtures)
{
    if (segment_size == 0)
    {
        std::cout << "Please specify segment size. Aborting.";
        exit(0);
    }
    if (input_video == "")
    {
        std::cout << "Please specify input video path. Aborting.";
        exit(0);
    }
    if (output_video == "")
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
    if (tmp_fixtures == "")
    {
        std::cout << "Please specify path where tmp files should be stored. Aborting.";
        exit(0);
    }
}

void print_parameters(int segment_size, std::string input_video, std::string output_video, 
                      std::string detector, std::string detector_cfg, std::string tmp_fixtures)
{
    printf("Segment size set to %d\n", segment_size);
    printf("Input video path set to %s\n", input_video.c_str());
    printf("Output video path set to %s\n", output_video.c_str());
    printf("Detector set to %s\n", detector.c_str());
    printf("Detector cfg path set to %s\n", detector_cfg.c_str());
    printf("Temporary files will be stored in %s\n", tmp_fixtures.c_str());
}

void print_exec_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
    std::cout << "Exec time = " << std::chrono::duration_cast<std::chrono::minutes>(end - begin).count() 
            << "[min] (" << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()  
            << "[s])" << std::endl;
}

int get_video_capture_frame_cnt(const cv::VideoCapture& cap)
{
    return cap.get(cv::CAP_PROP_FRAME_COUNT);
}

int get_trimmed_frame_cnt(const cv::VideoCapture& cap, int frames_in_segment) 
{
    int video_in_frame_cnt = get_video_capture_frame_cnt(cap);
    std::cout << "Input video frames count: " << video_in_frame_cnt << std::endl;
    return video_in_frame_cnt / frames_in_segment * frames_in_segment;
}

std::string get_frame_path(int frame, std::string tmp_folder)
{
    std::stringstream ss;
    ss << tmp_folder << "/img/frame" << std::to_string(frame) << ".jpeg";
    std::string path = ss.str();
    return path;
}

void trim_video(std::string video_in, std::string video_out, int frame_cnt) 
{
    std::stringstream ss;
    ss << "ffmpeg -i " << video_in << " -vframes " << std::to_string(frame_cnt) 
       << " -acodec copy -vcodec copy " << video_out << " -y";
    std::string trim_command = ss.str();
    std::cout << "Executing: " << trim_command << std::endl;
    system(trim_command.c_str());
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

std::vector<BoundingBox> load_frame_detections(std::string csv_file) 
{
    std::vector<BoundingBox> boxes;
    std::ifstream f(csv_file);
    std::string line, colname;
    int val;

    std::cout << "Loading detections from " << csv_file << std::endl;

    if(!f.is_open()) 
         throw std::runtime_error("Could not open file");

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
        BoundingBox box = {
            .x_min = coords[0],
            .y_min = coords[1],
            .x_max = coords[2],
            .y_max = coords[3]
        };
        boxes.push_back(box);
    }
    f.close();
    return boxes;
}

std::vector<std::vector<BoundingBox>> load_detections(int frame_cnt, std::string tmp_folder) 
{
    std::vector<std::vector<BoundingBox>> detections(frame_cnt, std::vector<BoundingBox>());
    for (int i = 0; i < frame_cnt; i += 1) 
    {
        std::cout << "Loading detections of frame " << i << std::endl;
        std::stringstream ss;
        ss << tmp_folder << "/csv/frame" << i << ".csv";
        std::string csv_path = ss.str();
        detections[i] = load_frame_detections(csv_path);
    }
    return detections;
}

unsigned int get_max_detections_per_frame(const std::vector<std::vector<BoundingBox>> &detections)
{
    unsigned int maxi = 0;
    for(auto const& d : detections)
        if (d.size() > maxi)
            maxi = d.size();
    std::cout << "Max number of detections per frame is: " << maxi << std::endl;
    return maxi;
}

std::vector<Color> get_colors(int vec_len) // mv to util
{    
    std::vector<Color> colors;
    colors.reserve(vec_len);
    for (int i = 0; i < vec_len; i++)
    {
        colors.push_back(Color());
    }
    return colors;
}

int main(int argc, char **argv) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int segment_size = 0;
    std::string input_video;
    std::string output_video;
    std::string detector;
    std::string detector_cfg;
    std::string tmp_fixtures;

    int opt;
    while((opt = getopt(argc, argv, "s:i:o:d:c:f:h")) != -1)
    {
        switch (opt)
        {
            case 's':
                segment_size = std::stoi(optarg);
                break;
            case 'i':
                input_video = optarg;
                break;
            case 'o':
                output_video = optarg;
                break;
            case 'd':
                detector = optarg;
                break;
            case 'c':
                detector_cfg = optarg;
                break;
            case 'f':
                tmp_fixtures = optarg;
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

    verify_parameters(segment_size, input_video, output_video, detector, detector_cfg, tmp_fixtures);
    print_parameters(segment_size, input_video, output_video, detector, detector_cfg, tmp_fixtures);
    make_tmp_dirs(tmp_fixtures);

    cv::VideoCapture in_cap(input_video);
    const int video_in_frame_cnt = get_video_capture_frame_cnt(in_cap);
    const int trimmed_video_frame_cnt = get_trimmed_frame_cnt(in_cap, segment_size);
    const std::string tmp_video = tmp_fixtures + "/input.mp4";
    if (video_in_frame_cnt != trimmed_video_frame_cnt)
    {
        std::string const trimmed_video = tmp_fixtures + "/trim.mp4";
        trim_video(input_video, trimmed_video, trimmed_video_frame_cnt);
        std::cout << "Input video frames count cut to: " << trimmed_video_frame_cnt << std::endl;
        mv(trimmed_video, tmp_video);
    }
    else
    {
        cp(input_video, tmp_video);
    }

    detect(detector, detector_cfg, trimmed_video_frame_cnt - 1, tmp_video, tmp_fixtures);
    auto detections = load_detections(trimmed_video_frame_cnt, tmp_fixtures);
    unsigned int max_detections_per_frame = get_max_detections_per_frame(detections);

    const int segment_cnt = trimmed_video_frame_cnt / segment_size;
    std::vector<std::vector<Location>> centers(trimmed_video_frame_cnt, std::vector<Location>()); 
    std::vector<std::vector<cv::Mat>> histograms(trimmed_video_frame_cnt, std::vector<cv::Mat>());
    std::vector<HistInterKernel> net_cost[trimmed_video_frame_cnt][trimmed_video_frame_cnt];

    // std::vector<std::vector<Location>> centers(trimmed_video_frame_cnt, std::vector<Location>()); = get_detection_centers()
    // std::vector<std::vector<cv::Mat>> histograms(trimmed_video_frame_cnt, std::vector<cv::Mat>()); = get_detection_histograms()
    constexpr int channels[3] = {0, 1, 2};
    constexpr float range[2] = {0, 256};
    const float * ranges[3] = {range, range, range};
    constexpr int histSize[3] = {8, 8, 8};
    for (int i = 0; i < segment_cnt; i++) 
    {
        std::cout << "Analyzing detection centers and histograms for segment: " << i+1 
                  << "/" << segment_cnt << std::endl;
        int start_frame = segment_size * i;
        for (int j = start_frame; j < start_frame + segment_size; j++)
        {
            std::cout << "Analyzing detection centers and histograms for frame: " << j+1 
                      << "/" << trimmed_video_frame_cnt << std::endl;
            cv::Mat frame = cv::imread(get_frame_path(j, tmp_fixtures));

            for(auto const& d : detections[j])
            {
                int width = d.x_max - d.x_min;
                int height = d.y_max - d.y_min;
                int x_center = (d.x_min + d.x_max) / 2;
                int y_center = (d.y_min + d.y_max) / 2;
                Location loc = {
                    .x = x_center,
                    .y = y_center
                };
                centers[j].push_back(loc);

                cv::Mat hist;
                cv::Mat detection = frame(cv::Rect(x_center, y_center, width, height));
                cv::calcHist(&detection, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
                histograms[j].push_back(hist);
            }
        }
    }
    
    // std::vector<HistInterKernel> net_cost[trimmed_video_frame_cnt][trimmed_video_frame_cnt]; = get_net_cost();
    for (int i = 0; i < trimmed_video_frame_cnt; i++)
        for (int k = 0; k < histograms[i].size(); k++)
            for (int j = 0; j < trimmed_video_frame_cnt; j++)
                for (int l = 0; l < histograms[j].size(); l++)
                {
                    double histogram_intersection_kernel = cv::compareHist(histograms[i][k], 
                                                                           histograms[j][l], 
                                                                           3); // CV_COMP_INTERSECT
                    HistInterKernel hik = {
                        .detection_idx1 = k,
                        .detection_idx2 = l,
                        .value = histogram_intersection_kernel
                    };
                    net_cost[i][j].push_back(hik);
                }

    std::vector<Color> colors = get_colors(max_detections_per_frame);

    // for (int i = 0; i < segment_cnt; i++)
    // {
    //     int j = 0;
    //     while (j < max_detections_per_frame)
    //     {
    //         // once a pedestrian is tracked, take corresponding detections out of net_cost matrix
    //         // remove_done_frames można zrobić na końcu tej pętli
             
    //         ;
    //     }
    // }

    clear_tmp(tmp_fixtures);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    print_exec_time(begin, end);
    return 0;
}
