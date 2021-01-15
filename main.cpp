#include <string>
#include <iostream>
#include "detector.hpp"
#include "tracker.hpp"


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
                utils::printing::print_usage_info();
                exit(0);
            default:
                std::cout << "Unsupported parameter passed to the script. Aborting." << std::endl;
                utils::printing::print_usage_info();
                abort();
        }
    }
}

void verify_parameters(int segment_size, std::string in_video, std::string out_video, 
                       std::string detector, std::string detector_cfg, std::string tmp_folder)
{
    if (segment_size < 2)
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

int main(int argc, char **argv) {
    auto begin = std::chrono::steady_clock::now();
    int segment_size = 0;
    std::string in_video, out_video, detector_type, detector_cfg, tmp_folder;
    get_parameters(argc, argv, segment_size, in_video, out_video, detector_type, detector_cfg, tmp_folder);
    verify_parameters(segment_size, in_video, out_video, detector_type, detector_cfg, tmp_folder);
    utils::printing::print_parameters(segment_size, in_video, out_video, detector_type, detector_cfg, tmp_folder);
    utils::sys::clear_tmp(tmp_folder);
    utils::sys::make_tmp_dirs(tmp_folder);
    cv::VideoCapture in_cap(in_video);
    video::info video_info = video::get_video_info(in_cap, segment_size, tmp_folder);
    video::prepare_tmp_video(in_cap, video_info, in_video);

    auto detector = Detector(detector_type, detector_cfg);
    auto detect_begin = std::chrono::steady_clock::now();
    detector.detect(video_info);
    auto detect_end = std::chrono::steady_clock::now();

    auto tracker = Tracker(video_info);
    tracker.track(out_video);

    utils::sys::clear_tmp(tmp_folder);
    auto end = std::chrono::steady_clock::now();
    utils::printing::print_exec_time(begin, end);
    utils::printing::print_detect_time(detect_begin, detect_end);
    return 0;
}