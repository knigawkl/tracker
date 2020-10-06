#pragma once

#include <opencv2/opencv.hpp>

int get_video_capture_frame_cnt(const cv::VideoCapture& cap);

int get_trimmed_frame_cnt(const cv::VideoCapture& cap, int frames_in_segment);

void trim_video(std::string video_in, std::string video_out, int frame_cnt);

void prepare_tmp_video(const cv::VideoCapture& in_cap, int desired_frame_cnt, 
                       std::string tmp_folder, std::string in_video, std::string tmp_video);

void merge_frames(std::string tmp_folder, std::string out_video);
