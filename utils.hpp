#pragma once

#include <stdlib.h>
#include <vector>
#include <chrono>

#include "gmcp.hpp"
#include "tracklet.hpp"
#include "node.hpp"

#include <opencv2/opencv.hpp>

void make_tmp_dirs(std::string tmp_folder);
void clear_tmp(std::string tmp_folder);
void mv(std::string what, std::string where);
void cp(std::string what, std::string where);

std::vector<cv::Scalar> get_colors(int vec_len);
std::string get_frame_path(int frame, std::string tmp_folder);

void print_usage_info();
void print_detection_path(const std::vector<Node> &path);
// void print_detections_left_cnt(const vector2d<Detection> &centers, int seg_counter, int seg_size);
// void print_detections_left_ids(const vector2d<Detection> &centers, int seg_counter, int seg_size);
void print_parameters(int segment_size, std::string in_video, std::string out_video, 
                      std::string detector, std::string detector_cfg, std::string tmp_folder);
void print_exec_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
void print_detect_time(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end);
void print_nodes(const vector2d<Node> &nodes);
// void print_net_cost(const vector2d<HistInterKernel> &net_cost);
void print_tracklets(const vector2d<Tracklet> &tracklets, int segment_cnt);
// void print_tracklet_center(const Location &center, int segment_ctr, int tracklet_ctr);
// void print_tracklets_net_costs(const vector2d<Tracklet> &tracklets, int segment_cnt);
