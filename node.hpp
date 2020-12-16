#pragma once

#include <opencv2/opencv.hpp>

#include "gmcp.hpp"

struct Box
{
    int x, y, id, x_min, y_min, x_max, y_max, height, width;
    float area;

    void print() const
    {
        std::cout << "x: " << x << ", y: " << y << ", id: " << id
                  << ", x_min: " << x_min << ", y_min: " << y_min 
                  << ", x_max: " << x_max << ", y_max: " << y_max 
                  << ", height: " << height << ", width: " << width 
                  << ", area: " << area << std::endl;
    }

    float calc_iou(Box b)
    { 
        int inter_x_min = std::max(x_min, b.x_min);
        int inter_y_min = std::max(y_min, b.y_min);
        int inter_x_max = std::min(x_max, b.x_max);
        int inter_y_max = std::min(y_max, b.y_max);
        int inter_width = std::max(0, inter_x_max - inter_x_min);
        int inter_height = std::max(0, inter_y_max - inter_y_min);
        int intersection_area = inter_width * inter_height;
        float iou = intersection_area / (area + b.area - intersection_area);
        return iou;
    }
};

class Node
{
// Each node represents an object detection
public:
    int node_id;
    int cluster_id;
    int next_node_id = -1;
    int prev_node_id = -1;
    Box coords;
    cv::Mat histogram;

    Node(const Box &d, const cv::Mat &frame, int detection_id, int frame_id): 
         coords(d), node_id(detection_id), cluster_id(frame_id)
    {
        set_histogram(frame);
    }
    void print() const;

private:
    void set_histogram(const cv::Mat &frame);
};

int get_max_nodes_per_cluster(const vector2d<Node> &nodes);
