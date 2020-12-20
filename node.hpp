#pragma once

#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "tracker.hpp"

struct Box
{
    int x, y, id, x_min, y_min, x_max, y_max, height, width;
    float area;

    void print() const;

    float calc_iou(Box b);
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
    static int get_max_nodes_per_cluster(const vector2d<Node> &nodes);
    static void print_nodes(const vector2d<Node> &nodes);
    static void print_detection_path(const vector<Node> &path);

private:
    void set_histogram(const cv::Mat &frame);
};
