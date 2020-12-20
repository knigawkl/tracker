// #include "edge.hpp"

// #include <cmath>

// double Edge::get_appearance_cost(const cv::Mat &start, const cv::Mat &end) 
// { 
//     return cv::compareHist(start, end, CV_COMP_INTERSECT);
// };

// double Edge::get_motion_cost(const Box &start, const Box &end)
// {
//     return MOTION_COEFF * sqrt(pow(start.x - end.x, 2) + pow(start.y - end.y, 2));
// }

// void Edge::print() const
// {
//      std::cout << "start_node_id: " << start_node_id
//                << ", end_node_id: " << end_node_id
//                << ", start_node_cluster_id: " << start_node_cluster_id
//                << ", end_node_cluster_id: " << end_node_cluster_id
//                << ", weight: " << weight << std::endl;
// }
