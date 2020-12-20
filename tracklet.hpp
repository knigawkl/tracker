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
    // int id;
    // int trajectory_id = -1;
    vector<Node> detection_track;
    Location center;
    cv::Mat histogram;

    Tracklet(const vector<Node> &path)
    {
        // id = tracklet_id;
        detection_track = path;
        set_middle_point();
        set_histogram();
        print();
    }
    void print() const;
    static void print_tracklets(const vector2d<Tracklet> &tracklets);

private:
    void set_middle_point();
    void set_histogram();
};
