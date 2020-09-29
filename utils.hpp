#pragma once

#include <stdlib.h>
#include <vector>
#include <chrono>

#include "gmcp.hpp"

void make_tmp_dirs(std::string tmp_folder);

void clear_tmp(std::string tmp_folder);

void mv(std::string what, std::string where);

void cp(std::string what, std::string where);

struct Color
{
    int r, g, b;
    Color()
    {
        r = rand() % 256;
        g = rand() % 256;
        b = rand() % 256;
    }
};

std::vector<Color> get_colors(int vec_len);

void print_usage_info();
void print_detection_path(const std::vector<Detection> &path);
void print_detections_left_cnt(const std::vector<std::vector<Detection>> &centers, int seg_counter, int seg_size);
void print_detections_left_ids(const std::vector<std::vector<Detection>> &centers, int seg_counter, int seg_size);
void print_parameters(int segment_size, std::string in_video, std::string out_video, 
                      std::string detector, std::string detector_cfg, std::string tmp_folder);
void print_exec_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
void print_detect_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
void print_boxes(const std::vector<std::vector<BoundingBox>> &boxes);
void print_centers(const std::vector<std::vector<Detection>> &centers);
