#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <string_view>
#include <iostream>
#include <sstream>
#include <cstdlib>

#include "generate_input.hpp"

#include <opencv2/opencv.hpp>


int get_trimmed_frame_cnt(const cv::VideoCapture& cap, int frames_in_segment) {
    // calculates max length of video in frames if it has to be a multiple of frames_in_segment
    int video_in_frame_cnt = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Input video frames count: " << video_in_frame_cnt << std::endl;
    return video_in_frame_cnt / frames_in_segment * frames_in_segment;
}

void trim_video(std::string video_in, std::string video_out, int frame_cnt) {
    // trims video stored at video_in path to frame_cnt frames
    std::stringstream ss;
    ss << "ffmpeg -i " << video_in << " -vframes " << std::to_string(frame_cnt) << " -acodec copy -vcodec copy " << video_out << " -y";
    std::string trim_command = ss.str();
    system(trim_command.c_str());
}

void detect(std::string detector, std::string detector_cfg, int segment_size, int frame_cnt, std::string video, std::string tmp_folder) {
    // initiates object detections on selected frames of the trimmed video
    std::stringstream ss;
    ss << "python3 ../detectors/detect.py --detector " << detector 
       << " --cfg " << detector_cfg
       << " --segment_size " << std::to_string(segment_size) 
       << " --video " << video
       << " --tmp_folder " << tmp_folder
       << " --frame_cnt " << std::to_string(frame_cnt);
    std::string detect_command = ss.str();
    system(detect_command.c_str());
}

// void load_detections() {

// }

// void create_net_cost_matrix() {

// }

// void save_input() {

// }

int main(int argc, char **argv) {
    int segment_size = 0;
    std::string input_video;
    std::string output_video;
    std::string detector;
    std::string detector_cfg;
    std::string tmp_fixtures;

    const char* usage_info =
    "\n    -s, --segment_size   frames in a segment\n"
    "    -i, --input_video      input video path\n"
    "    -o, --output_video     output video path\n"
    "    -d, --detector         object detector (ssd or yolo)\n"
    "    -c, --detector_cfg     path to detector cfg\n"
    "    -f, --tmp_fixtures     path to folder where temporary files will be stored\n"
    "    -h, --help             show this help msg";

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
                std::cout << "Please provide the following arguments:\n" << usage_info;
                exit(0);
            default:
                std::cout << "Unsupported parameter passed to the script. Aborting.";
                std::cout << usage_info;
                abort();
        }
    }

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

    printf("Segment size set to %d\n", segment_size);
    printf("Input video path set to %s\n", input_video.c_str());
    printf("Output video path set to %s\n", output_video.c_str());
    printf("Detector set to %s\n", detector.c_str());
    printf("Detector cfg path set to %s\n", detector_cfg.c_str());
    printf("Temporary files will be stored in %s\n", tmp_fixtures.c_str());

    std::string const trimmed_video = tmp_fixtures + "tmp.mp4";

    cv::VideoCapture in_cap(input_video);
    auto trimmed_video_frame_cnt = get_trimmed_frame_cnt(in_cap, segment_size);
    trim_video(input_video, trimmed_video, trimmed_video_frame_cnt);
    std::cout << "Input video frames count cut to: " << trimmed_video_frame_cnt << std::endl;

    detect(detector, detector_cfg, segment_size, trimmed_video_frame_cnt, trimmed_video, tmp_fixtures);

    return 0;
}
