#pragma once

#include <opencv2/opencv.hpp>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <thread>  
#include <fstream> 
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <algorithm> 
#include <chrono>
#include <utility>
#include <limits>

#include "edge.hpp"
#include "clique.hpp"
#include "iou.hpp"
#include "node.hpp"
#include "tracklet.hpp"
#include "utils.hpp"
#include "video.hpp"
#include "templates.hpp"


class Tracker
{
public:
    video::info video_info;
    Tracker(const video::info& video_info): video_info(video_info) {};
    void track(const std::string& out_video);
private:
    vector<Node> load_cluster_nodes(std::string csv_file, const cv::Mat &frame, int frame_id);
    vector2d<Node> load_nodes();
    int get_min_detections_in_segment_cnt(const vector2d<Node> &nodes, int seg_counter, int start);
    vector2d<Tracklet> get_tracklets(vector2d<Node> &nodes);
    void merge_tracklets(vector2d<Tracklet> &tracklets);
};
