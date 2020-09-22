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


int get_trimmed_frames_cnt(const cv::VideoCapture& cap, uint8_t frames_in_segment) {
    int video_in_frame_cnt = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Input video frames count: " << video_in_frame_cnt << std::endl;
    return video_in_frame_cnt / frames_in_segment * frames_in_segment;
}

void trim_video(std::string video_in, std::string video_out, int frames_cnt) {	
    std::stringstream ss;
    ss << "ffmpeg -i " << video_in << " -vframes " << std::to_string(frames_cnt) << " -acodec copy -vcodec copy " << video_out << " -y";
    std::string trim_command = ss.str();
    system(trim_command.c_str());
}

void create_net_cost_matrix() {

}

void save_input() {

}

int main(int argc, char **argv) {
    uint8_t segment_size;
    uint8_t segment_cnt;
    std::string input_video;
    std::string output_video;

    const char* usage_info =
    "    -s, --segment_size   frames in a segment\n"
    "    -c, --segment_cnt    number of segments\n"
    "    -i, --input_video    frames in a segment\n"
    "    -o, --output_video   frames in a segment\n"
    "    -h, --help           show this help msg";

    int opt;
    while((opt = getopt(argc, argv, "s:c:i:o:h")) != -1)
    {
        switch (opt)
        {
            case 's':
                segment_size = std::stoi(optarg);
                break;
            case 'c':
                segment_cnt = std::stoi(optarg);
                break;
            case 'i':
                input_video = optarg;
                break;
            case 'o':
                output_video = optarg;
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

    if (segment_cnt == 0)
        segment_cnt = 18;
    if (segment_size == 0)
        segment_size = 10;
    if (input_video == "")
        input_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/input/pets3s.mp4";
    if (output_video == "")
        output_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/output/output.mp4";

    printf("Running with configuration:\nsegment_size = %d\nsegment_cnt = %d\ninput_video = %s\noutput_video = %s\n", segment_size, segment_cnt, input_video.c_str(), output_video.c_str());

    std::string const trimmed_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/tmp/tmp.mp4";

    cv::VideoCapture in_cap(input_video);
    auto trimmed_video_frames_cnt = get_trimmed_frames_cnt(in_cap, segment_size);
    trim_video(input_video, trimmed_video, trimmed_video_frames_cnt);
    std::cout << "Input video frames count cut to: " << trimmed_video_frames_cnt << std::endl;

    return 0;
}
