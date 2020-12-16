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

void Tracklet::print() const
{
    std::cout << "Tracklet: " << "center: ";
    center.print();
}
