#pragma once

#include <stdlib.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "tracker.hpp"


void make_tmp_dirs(std::string tmp_folder);
void clear_tmp(std::string tmp_folder);
void mv(std::string what, std::string where);
void cp(std::string what, std::string where);

vector<cv::Scalar> get_colors(int vec_len);
std::string get_frame_path(int frame, std::string tmp_folder);

void print_usage_info();
void print_parameters(int segment_size, std::string in_video, std::string out_video, 
                      std::string detector, std::string detector_cfg, std::string tmp_folder);
void print_exec_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
void print_detect_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
