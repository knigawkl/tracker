#include <stdint.h>
#include <string>
#include <string_view>
#include <iostream>
#include <sstream>

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

int main() {
    // todo: move configuration to separate file
    constexpr uint8_t const segment_size = 50;
    constexpr uint8_t const segment_cnt = 18;
    std::string const input_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/input/pets3s.mp4";
    std::string const output_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/output/output.mp4";

    // this configuration may be local
    std::string const trimmed_video = "/home/lk/Desktop/praca-inzynierska/gmcp-tracker-cpp-python/fixtures/tmp/tmp.mp4";

    cv::VideoCapture in_cap(input_video);
    auto trimmed_video_frames_cnt = get_trimmed_frames_cnt(in_cap, segment_size);
    trim_video(input_video, trimmed_video, trimmed_video_frames_cnt);
    std::cout << "Input video frames count cut to: " << trimmed_video_frames_cnt << std::endl;

    return 0;
}

