#pragma once

#include <vector>
#include <iostream>

#include "gmcp.hpp"

#include <opencv2/opencv.hpp>

class Tracklet
{
public:
    vector<Detection> detections;
    Location center;
    cv::Mat histogram;
    vector<HistInterKernel> net_cost;

    Tracklet(const vector<Detection> &path, const vector2d<cv::Mat> &histograms, int seg_ctr, int seg_size)
    {
        detections = path;
        set_middle_point(path);
        set_histogram(histograms, seg_ctr, seg_size);
    }
    void calculate_net_cost();

private:
    void set_middle_point(const vector<Detection> &detection_track);
    void set_histogram(const vector2d<cv::Mat> &histograms, int seg_ctr, int seg_size);
    vector<int> get_detection_ids() const;
};
