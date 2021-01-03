#include "tracklet.hpp"

void Tracklet::set_middle_point()
{
    // spatial location of a tracklet is defined as the middle point of the tracklet
    int x_sum = 0;
    int y_sum = 0;
    for (auto detection: detection_track)
    {
        x_sum += detection.coords.x;
        y_sum += detection.coords.y;
    }
    int x_center = x_sum / detection_track.size();
    int y_center = y_sum / detection_track.size();
    Location middle_point = {
        .x = x_center,
        .y = y_center
    };
    center = middle_point; 
}

void Tracklet::set_histogram()
{
    // tracklet's appearance feature is the mean of color histograms of detections from the tracklet
    int seg_size = detection_track.size();
    double anti_overflow_coeff = 1.0 / seg_size;
    auto tracklet_histogram = detection_track[0].histogram * anti_overflow_coeff;
    
    for (int i = 1; i < detection_track.size(); i++)
        tracklet_histogram += detection_track[i].histogram * anti_overflow_coeff;
    histogram = tracklet_histogram;
}

bool Tracklet::is_end_of_trajectory(int video_w, int video_h)
{
    int x_diff_sum = detection_track.back().coords.x - detection_track[0].coords.x;
    int y_diff_sum = detection_track.back().coords.y - detection_track[0].coords.y;
    int next_x_pred = detection_track.back().coords.x + x_diff_sum;  // x position prediction at the end of the next tracklet
    int next_y_pred = detection_track.back().coords.y + y_diff_sum;  // y position prediction at the end of the next tracklet
    std::cout << "Next tracklet predicted to end at: (" << next_x_pred << "," << next_y_pred << ")" << std::endl;
    if ((next_x_pred < 0) || (next_x_pred > video_w))
        return true;
    if ((next_y_pred < 0) || (next_y_pred > video_h))
        return true;
    return false;
}

bool Tracklet::is_start_of_trajectory(int video_w, int video_h)
{
    int x_diff_sum = detection_track.back().coords.x - detection_track[0].coords.x;
    int y_diff_sum = detection_track.back().coords.y - detection_track[0].coords.y;
    int prev_x_pred = detection_track.front().coords.x - x_diff_sum;  // x position prediction at the start of the prev tracklet
    int prev_y_pred = detection_track.front().coords.y - y_diff_sum;  // y position prediction at the start of the prev tracklet
    std::cout << "Previous tracklet predicted to start at: (" << prev_x_pred << "," << prev_y_pred << ")" << std::endl;
    if ((prev_x_pred < 0) || (prev_x_pred > video_w))
        return true;
    if ((prev_y_pred < 0) || (prev_y_pred > video_h))
        return true;
    return false;
}

void Tracklet::print() const
{
    std::cout << "Tracklet: " << "center: ";
    center.print();
}

void Tracklet::print_tracklets(const vector2d<Tracklet> &tracklets)
{
    std::cout << std::endl;
    for (int i = 0; i < tracklets.size(); i++)
    {
        std::cout << "Tracklets found in segment " << i+1 << "/" << tracklets.size() << std::endl;
        for (int j = 0; j < tracklets[i].size(); j++)
        {
            std::cout << "Tracklet " << j+1 << "/" << tracklets[i].size() << std::endl;
            Node::print_detection_path(tracklets[i][j].detection_track);
        }
    }
}