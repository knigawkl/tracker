#pragma once

#include <string_view>

#include <opencv2/opencv.hpp>


namespace video
{
    struct vidinfo 
    {
    public:
        int frame_cnt;
        int segment_size;
        int segment_cnt;
        int width;
        int height;
        double fps;
        std::string tmp_dir;
        std::string tmp_video;
    };
    vidinfo get_video_info(const cv::VideoCapture& cap, int segment_size, const std::string& tmp_folder);
    int get_video_capture_frame_cnt(const cv::VideoCapture& cap);
    int get_trimmed_frame_cnt(const cv::VideoCapture& cap, int frames_in_segment);
    void trim_video(std::string video_in, std::string video_out, int frame_cnt);
    void prepare_tmp_video(const cv::VideoCapture& in_cap, const vidinfo& video_info, const std::string& in_video);
    void merge_frames(std::string_view out_video, const video::vidinfo& video_info);
}
