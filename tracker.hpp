#pragma once

#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <fstream> 
#include <chrono>
#include <string_view>

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
    video::vidinfo video_info;
    Tracker(const video::vidinfo& video_info): video_info(video_info) {};
    void track(std::string_view out_video);
private:
    vector<Node> load_cluster_nodes(std::string csv_file, const cv::Mat &frame, int frame_id);
    vector2d<Node> load_nodes();
    int get_min_detections_in_segment_cnt(const vector2d<Node> &nodes, int seg_counter, int start);
    vector2d<Tracklet> get_tracklets(vector2d<Node> &nodes);
    void merge_tracklets(vector2d<Tracklet> &tracklets);
};
