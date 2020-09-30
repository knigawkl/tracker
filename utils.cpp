#include <sstream>
#include <iostream>
#include <vector>

#include "utils.hpp"

void make_tmp_dirs(std::string tmp_folder) 
{
    std::stringstream ss;
    ss << "mkdir " << tmp_folder << "/img " << tmp_folder << "/csv";
    std::string mkdir_command = ss.str();
    std::cout << "Executing: " << mkdir_command << std::endl;
    system(mkdir_command.c_str());
}

void clear_tmp(std::string tmp_folder) 
{
    std::stringstream ss;
    ss << "exec rm -r " << tmp_folder << "/*";
    std::string del_command = ss.str();
    std::cout << "Executing: " << del_command << std::endl;
    system(del_command.c_str());
}

void mv(std::string what, std::string where)
{
    std::stringstream ss;
    ss << "mv " << what << " " << where;
    std::string mv_command = ss.str();
    std::cout << "Executing: " << mv_command << std::endl;
    system(mv_command.c_str());
}

void cp(std::string what, std::string where)
{
    std::stringstream ss;
    ss << "cp " << what << " " << where;
    std::string cp_command = ss.str();
    std::cout << "Executing: " << cp_command << std::endl;
    system(cp_command.c_str());
}

std::vector<cv::Scalar> get_colors(int vec_len)
{    
    std::vector<cv::Scalar> colors;
    colors.reserve(vec_len);
    uint8_t r, g, b;
    for (int i = 0; i < vec_len; i++)
    {
        b = rand() % 256;
        r = rand() % 256;
        g = rand() % 256;
        colors.push_back(cv::Scalar(b, g, r));
    }
    return colors;
}

// all printing functions here
void print_usage_info()
{
    const char* usage_info =
    "\n    -s, --segment_size   frames in a segment\n"
    "    -i, --in_video         input video path\n"
    "    -o, --out_video        output video path\n"
    "    -d, --detector         object detector (ssd or yolo)\n"
    "    -c, --detector_cfg     path to detector cfg\n"
    "    -f, --tmp_folder       path to folder where temporary files will be stored\n"
    "    -h, --help             show this help msg";
    std::cout << usage_info << std::endl;
}

void print_detection_path(const std::vector<Detection> &path)
{
    if (path.size())
    {
        for (int i = 0; i < path.size()-1; i++)
        {
            std::cout << "(" << path[i].x << "," << path[i].y << ")->";
        }
        std::cout << "(" << path.back().x << "," << path.back().y << ")" << std::endl;
    } else {
        std::cout << "Not enough detections to form a tracklet" << std::endl;
    }
}

void print_detections_left_cnt(const std::vector<std::vector<Detection>> &centers, int seg_counter, int seg_size)
{
    int result = 0;
    int start = seg_counter * seg_size;
    for (int i = 0; i < seg_size; i++)
        result += centers[start+i].size();
    std::cout << "Detections left in segment: " << result << std::endl;
}

void print_detections_left_ids(const std::vector<std::vector<Detection>> &centers, int seg_counter, int seg_size)
{
    int start = seg_counter * seg_size;
    for (int i = 0; i < seg_size; i++)
    {
        std::cout << "ids of detections left from frame " << i << ": ";
        for (int j = 0; j < centers[start+i].size(); j++)
            std::cout << centers[start+i][j].id << " ";
        std::cout << std::endl;
    }
}

void print_parameters(int segment_size, std::string in_video, std::string out_video, 
                      std::string detector, std::string detector_cfg, std::string tmp_folder)
{
    printf("Segment size set to %d\n", segment_size);
    printf("Input video path set to %s\n", in_video.c_str());
    printf("Output video path set to %s\n", out_video.c_str());
    printf("Detector set to %s\n", detector.c_str());
    printf("Detector cfg path set to %s\n", detector_cfg.c_str());
    printf("Temporary files will be stored in %s\n", tmp_folder.c_str());
}

void print_exec_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
    std::cout << "Execution time = " << std::chrono::duration_cast<std::chrono::minutes>(end - begin).count() 
            << "[min] (" << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()  
            << "[s])" << std::endl;
}

void print_detect_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end)
{
    std::cout << "Detection took = " << std::chrono::duration_cast<std::chrono::minutes>(end - begin).count() 
            << "[min] (" << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()  
            << "[s])" << std::endl;
}

void print_detections(const vector2d<Detection> &detections)
{
    for (int i = 0; i < detections.size(); i++)
    {
        std::cout << detections[i].size() << " detections in frame " << i << std::endl;
        for (int j = 0; j < detections[i].size(); j++)
        {
            detections[i][j].print();
        }
    }
}

void print_net_cost(const std::vector<std::vector<HistInterKernel>> &net_cost)
{
    // for debug purposes only
    // for (auto frame: net_cost)
    for (int i = 0; i < net_cost.size(); i++)
    {
        std::cout << "========== net cost frame " << i << " ==========" << std::endl;
        for (auto hik: net_cost[i])
            hik.print();
    }

    std::cout << std::endl;
}

void print_tracklets(const vector3d<Detection> &tracklets, int segment_cnt)
{
    std::cout << std::endl;
    for (int i = 0; i < segment_cnt; i++)
    {
        std::cout << "Tracklets found in segment " << i+1 << "/" << segment_cnt << std::endl;
        for (int j = 0; j < tracklets[i].size(); j++)
        {
            std::cout << "Tracklet " << j+1 << "/" << tracklets[i].size() << std::endl;
            print_detection_path(tracklets[i][j]);
        }
    }
}
