#pragma once

#include <vector>
#include <iostream>

#include "tracker.hpp"
#include "location.hpp"
#include "node.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>

class Tracklet
{
public:
    int video_w; 
    int video_h; 
    int video_frame_cnt;
    std::string tmp_folder;
    bool is_end_of_traj;
    bool is_start_of_traj;
    bool is_hypothetical = false;
    vector<Node> detection_track;
    Node centroid;
    cv::Scalar color;

    Tracklet(const vector<Node> &path, int wideo_w, int wideo_h, int wideo_frame_cnt, std::string tmp_dir)
    {
        video_w = wideo_w; 
        video_h = wideo_h; 
        video_frame_cnt = wideo_frame_cnt;
        tmp_folder = tmp_dir;
        detection_track = path;
        init_color();
        set_centroid();
        eliminate_outliers();
        is_end_of_traj =  is_end_of_trajectory(video_w, video_h, video_frame_cnt);
        is_start_of_traj =  is_start_of_trajectory(video_w, video_h);
        print();
    }
    void print() const;
    void draw();
    static void print_tracklets(const vector2d<Tracklet> &tracklets);

private:
    static constexpr std::size_t const outlier_coeff = 3;
    void set_middle_point();
    void set_histogram();
    void set_centroid();
    void eliminate_outliers();
    void init_color();
    bool is_end_of_trajectory(int video_w, int video_h, int video_frame_cnt);
    bool is_start_of_trajectory(int video_w, int video_h);
};
