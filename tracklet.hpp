#pragma once

#include <vector>
#include <iostream>

#include "templates.hpp"
#include "node.hpp"
#include "utils.hpp"
#include "video.hpp"


#include <opencv2/opencv.hpp>

class Tracklet
{
public:
    video::info video_info;
    bool is_end_of_traj;
    bool is_start_of_traj;
    bool is_hypothetical = false;
    vector<Node> detection_track;
    Node centroid;
    cv::Scalar color;

    Tracklet(const vector<Node> &path, const video::info &wideo_info)
    {
        video_info = wideo_info;
        detection_track = path;
        init_color();
        set_centroid();
        eliminate_outliers();
        is_end_of_traj =  is_end_of_trajectory();
        is_start_of_traj =  is_start_of_trajectory();
        print();
    }
    void print() const;
    void draw() const;
    static void print_tracklets(const vector2d<Tracklet> &tracklets);
    static void draw_tracklets(const vector2d<Tracklet> &tracklets);

private:
    static constexpr std::size_t const outlier_coeff = 3;
    void set_middle_point();
    void set_histogram();
    void set_centroid();
    void eliminate_outliers();
    void init_color();
    bool is_end_of_trajectory() const;
    bool is_start_of_trajectory() const;
};
