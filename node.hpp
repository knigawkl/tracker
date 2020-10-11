#pragma once

#include <opencv2/opencv.hpp>

struct Detection // todo: maybe incorporate this into the Node class?
{
    int x, y, id, x_min, y_min, x_max, y_max, height, width;
    void print() const
    {
        std::cout << "x: " << x << ", y: " << y << ", id: " << id
                  << ", x_min: " << x_min << ", y_min: " << y_min 
                  << ", x_max: " << x_max << ", y_max: " << y_max 
                  << ", height: " << height << ", width: " << width << std::endl;
    }
};

class Node
{
// Each node represents an object detection
public:
    int node_id;
    int cluster_id;
    Detection coords;
    cv::Mat histogram;
    // add cost to any node in any frame except current

    Node(const Detection &d, const cv::Mat &frame, int detection_id, int frame_id): 
         coords(d), node_id(detection_id), cluster_id(frame_id)
    {
        set_histogram(frame);
    }
    void print() const;

private:
    void set_histogram(const cv::Mat &frame);
};
