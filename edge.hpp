// #pragma once

// #include <opencv2/opencv.hpp>

// #include "node.hpp"

// class Edge
// {
// public:
//     int start_node_id;
//     int end_node_id;
//     int start_node_cluster_id;
//     int end_node_cluster_id;
//     double weight;

//     Edge(const Node &start, const Node &end)
//     {
//         start_node_id = start.node_id;
//         end_node_id = end.node_id;
//         start_node_cluster_id = start.cluster_id;
//         end_node_cluster_id = end.cluster_id;
//         double appearance_cost = get_appearance_cost(start.histogram, end.histogram);
//         double motion_cost = get_motion_cost(start.coords, end.coords);
//         std::cout << "appearance_cost " << appearance_cost;
//         std::cout << ", motion_cost " << motion_cost << std::endl;

//         weight = appearance_cost + motion_cost;
//     }
//     void print() const;

// private:
//     static constexpr double const MOTION_COEFF = 0.001;
//     static constexpr int const CV_COMP_INTERSECT = 3;
//     double get_appearance_cost(const cv::Mat &start, const cv::Mat &end);
//     double get_motion_cost(const Box &start, const Box &end);
// };
