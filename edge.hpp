#pragma once

#include <opencv2/opencv.hpp>

#include "node.hpp"

class Edge
{
public:
    Node start;
    Node end;
    double weight;
    Edge(const Node &start, const Node &end): start(start), end(end)
    {
        double appearance_cost = get_appearance_cost(start.histogram, end.histogram);
        double motion_cost = get_motion_cost(start.coords, end.coords);
        std::cout << "appearance_cost " << appearance_cost;
        std::cout << ", motion_cost " << motion_cost << std::endl;
        weight = appearance_cost + motion_cost;
    }
    void print() const;
    static bool edge_cmp(const Edge& a, const Edge& b);
private:
    static constexpr double const MOTION_COEFF = 0.01;
    static constexpr int const CV_COMP_INTERSECT = 3;
    double get_appearance_cost(const cv::Mat &start, const cv::Mat &end);
    double get_motion_cost(const Box &start, const Box &end);
};
