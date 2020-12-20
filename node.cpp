#include "node.hpp"
#include "tracker.hpp"

void Box::print() const
{
    std::cout << "x: " << x << ", y: " << y << ", id: " << id
                << ", x_min: " << x_min << ", y_min: " << y_min 
                << ", x_max: " << x_max << ", y_max: " << y_max 
                << ", height: " << height << ", width: " << width 
                << ", area: " << area << std::endl;
}

float Box::calc_iou(Box b)
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

int Node::get_max_nodes_per_cluster(const vector2d<Node> &nodes)
{
    // checks what is the biggest number of detections in a single frame
    int maxi = 0;
    for (auto const& n : nodes)
        if (n.size() > maxi)
            maxi = n.size();
    std::cout << "Max number of detections per frame is: " << maxi << std::endl;
    return maxi;
}

void Node::print_detection_path(const vector<Node> &path)
{
    if (path.size())
    {
        for (int i = 0; i < path.size()-1; i++)
        {
            std::cout << "(" << path[i].coords.x << "," << path[i].coords.y << ")->";
        }
        std::cout << "(" << path.back().coords.x << "," << path.back().coords.y << ")" << std::endl;
    } else {
        std::cout << "Not enough detections to form a tracklet" << std::endl;
    }
}

void Node::print_nodes(const vector2d<Node> &nodes)
{
    for (int i = 0; i < nodes.size(); i++)
    {
        std::cout << nodes[i].size() << " detections in frame " << i << std::endl;
        for (int j = 0; j < nodes[i].size(); j++)
        {
            nodes[i][j].print();
        }
    }
}
