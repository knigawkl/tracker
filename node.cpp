#include "node.hpp"

void Node::set_histogram(const cv::Mat &frame)
{
    constexpr int channels[3] = {0, 1, 2};
    constexpr float range[2] = {0, 256};
    const float * ranges[3] = {range, range, range};
    constexpr int histSize[3] = {8, 8, 8};
    cv::Mat hist;
    cv::Mat detection = frame(cv::Rect(coords.x_min, coords.y_min, coords.width, coords.height));
    cv::calcHist(&detection, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
    histogram = hist;
}

void Node::print() const
{
    std::cout << "node_id: " << node_id
              << ", frame_id: " << cluster_id << ", next: " << next_node_id << ", prev: " << prev_node_id << ", ";
    coords.print();
}

int get_max_nodes_per_cluster(const vector2d<Node> &nodes)
{
    // checks what is the biggest number of detections in a single frame
    int maxi = 0;
    for (auto const& n : nodes)
        if (n.size() > maxi)
            maxi = n.size();
    std::cout << "Max number of detections per frame is: " << maxi << std::endl;
    return maxi;
}
