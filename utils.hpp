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
void print_detections_left_cnt(const vector2d<Detection> &centers, int seg_counter, int seg_size);
void print_detections_left_ids(const vector2d<Detection> &centers, int seg_counter, int seg_size);
void print_parameters(int segment_size, std::string in_video, std::string out_video, 
                      std::string detector, std::string detector_cfg, std::string tmp_folder);
void print_exec_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
void print_detect_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
void print_boxes(const vector2d<BoundingBox> &boxes);
void print_centers(const vector2d<Detection> &centers);
void print_net_cost(const vector2d<HistInterKernel> &net_cost);
void print_tracklets(const vector3d<Detection> &tracklets, int segment_cnt);
