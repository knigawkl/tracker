// #include "tracklet.hpp"

// void Tracklet::set_middle_point(const vector<Detection> &detection_track)
// {
//     // spatial location of a tracklet is defined as the middle point of the tracklet
//     int x_sum = 0;
//     int y_sum = 0;
//     for (auto detection: detection_track)
//     {
//         x_sum += detection.x;
//         y_sum += detection.y;
//     }
//     int x_center = x_sum / detection_track.size();
//     int y_center = y_sum / detection_track.size();
//     Location middle_point = {
//         .x = x_center,
//         .y = y_center
//     };
//     center = middle_point; 
// }

// void Tracklet::set_histogram(const vector2d<cv::Mat> &histograms, 
//                              int seg_ctr, int seg_size)
// {
//     // tracklet's appearance feature is the mean of color histograms of detections from the tracklet
//     vector<int> detection_ids = get_detection_ids();

//     int start = seg_ctr * seg_size;
//     double anti_overflow_coeff = 1.0 / seg_size;
//     auto tracklet_histogram = histograms[start][detection_ids[0]] * anti_overflow_coeff;
    
//     for (int i = 1; i < detection_ids.size(); i++)
//         tracklet_histogram += histograms[start+i][detection_ids[i]] * anti_overflow_coeff;
//     histogram = tracklet_histogram;
// }

// vector<int> Tracklet::get_detection_ids() const
// {
//     vector<int> detection_ids;
//     for (auto detection: detections)
//     {
//         detection_ids.push_back(detection.id);
//     }
//     return detection_ids;
// }

// void Tracklet::print() const
// {
//     std::cout << "Tracklet: " << id << ", trajectory_id: " << trajectory_id << ", center: ";
//     center.print();
// }
