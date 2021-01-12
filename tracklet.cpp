#include <numeric>

#include "tracklet.hpp"
#include "utils.hpp"

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
    centroid.coords.x = x_center;
    centroid.coords.y = y_center; 
}

void Tracklet::set_histogram()
{
    // tracklet's appearance feature is the mean of color histograms of detections from the tracklet
    int seg_size = detection_track.size();
    double anti_overflow_coeff = 1.0 / seg_size;
    auto tracklet_histogram = detection_track[0].histogram * anti_overflow_coeff;
    
    for (size_t i = 1; i < detection_track.size(); i++)
        tracklet_histogram += detection_track[i].histogram * anti_overflow_coeff;
    centroid.histogram = tracklet_histogram;
}

void Tracklet::set_centroid()
{
    set_middle_point();
    set_histogram();
}

void Tracklet::init_color()
{
    uint8_t r, g, b;
    b = rand() % 256;
    r = rand() % 256;
    g = rand() % 256;
    color = cv::Scalar(b, g, r);
}

void Tracklet::draw()
{
    constexpr int const line_thickness = 2; 
    for (const Node& node: detection_track)
    {
        auto path = utils::get_frame_path(node.cluster_id, tmp_folder);
        cv::Mat img = cv::imread(path);
        cv::Rect rect(node.coords.x_min, node.coords.y_min, node.coords.width, node.coords.height);
        cv::rectangle(img, rect, color, line_thickness);
        cv::imwrite(path, img);
    }
}

bool Tracklet::is_end_of_trajectory(int video_w, int video_h, int video_frame_cnt)
{
    if (detection_track.back().cluster_id == (video_frame_cnt - 1))
        return true;
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
    if (detection_track.front().cluster_id == 0)
        return true;
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

void Tracklet::eliminate_outliers()
{
    int len = detection_track.size();
    for (size_t i = 0; i < len; i++)  // for each detection in this tracklet
    {
        vector<double> x;  // x coordinates of detections in this tracklet apart from current
        vector<double> y;  // y coordinates of detections in this tracklet apart from current
        for (size_t j = 0; j < len; j++)
        {
            if (i != j)
            {
                x.push_back(double(detection_track[j].coords.x));
                y.push_back(double(detection_track[j].coords.y));
            }
        }
        double x_center = 0;
        double y_center = 0;
        x_center = std::accumulate(x.begin(), x.end(), x_center);
        y_center = std::accumulate(y.begin(), y.end(), y_center);
        x_center /= (len - 1);  // centroid x center
        y_center /= (len - 1);  // centroid y center

        vector<double> dists;  // distances from detection to centroid
        for (size_t j = 0; j < len; j++)
            if (i != j)
                dists.push_back(utils::regression::euclidean_dist(detection_track[j].coords.x, detection_track[j].coords.y, x_center, y_center));
        double avg_center_dist = 0;
        avg_center_dist = std::accumulate(dists.begin(), dists.end(), avg_center_dist);
        avg_center_dist /= (len - 1);
        double current_center_dist = utils::regression::euclidean_dist(detection_track[i].coords.x, detection_track[i].coords.y, x_center, y_center);

        std::cout << "current_center_dist\t" << current_center_dist << std::endl;
        std::cout << "avg_center_dist\t" << avg_center_dist << std::endl;
        std::cout << "outlier_coeff * avg_center_dist\t" << outlier_coeff * avg_center_dist << std::endl;

        if (current_center_dist > outlier_coeff * avg_center_dist)
        {
            std::pair<double, double> linear_fit = utils::regression::get_linear_fit(x, y, len - 1);
            std::cout << "a: " << linear_fit.first << std::endl;
            std::cout << "b: " << linear_fit.second << std::endl;
            if (i == 0) {
                detection_track[i].coords.x = detection_track[i + 1].coords.x - (detection_track[i + 2].coords.x - detection_track[i + 1].coords.x);
                // set bounding box props
                detection_track[i].coords.width = detection_track[i + 1].coords.width;
                detection_track[i].coords.height = detection_track[i + 1].coords.height;
                detection_track[i].histogram = detection_track[i + 1].histogram;
            }
            else if (i == (len - 1)) {
                detection_track[i].coords.x = detection_track[i - 1].coords.x + (detection_track[i - 1].coords.x - detection_track[i - 2].coords.x);
                // set bounding box props 
                detection_track[i].coords.width = detection_track[i - 1].coords.width;
                detection_track[i].coords.height = detection_track[i - 1].coords.height;
                detection_track[i].histogram = detection_track[i - 1].histogram;
            }
            else {
                detection_track[i].coords.x = (detection_track[i - 1].coords.x + detection_track[i + 1].coords.x) / 2;
                // set bounding box props
                detection_track[i].coords.width = detection_track[i - 1].coords.width;
                detection_track[i].coords.height = detection_track[i - 1].coords.height;
                detection_track[i].histogram = detection_track[i + 1].histogram;
            }
            detection_track[i].coords.y = linear_fit.first * detection_track[i].coords.x + linear_fit.second;

            detection_track[i].coords.x_max = detection_track[i].coords.x + detection_track[i].coords.width / 2;
            detection_track[i].coords.x_min = detection_track[i].coords.x - detection_track[i].coords.width / 2;
            detection_track[i].coords.y_max = detection_track[i].coords.y + detection_track[i].coords.height / 2;
            detection_track[i].coords.y_min = detection_track[i].coords.y - detection_track[i].coords.height / 2;

            set_centroid();
            is_end_of_traj =  is_end_of_trajectory(video_w, video_h, video_frame_cnt);
            is_start_of_traj =  is_start_of_trajectory(video_w, video_h);
            is_hypothetical = true;
        }
    }
}


void Tracklet::print() const
{
    std::cout << "Tracklet: " << "center: "
              << " x: " << centroid.coords.x 
              << ", y: " << centroid.coords.y << std::endl;
}

void Tracklet::print_tracklets(const vector2d<Tracklet> &tracklets)
{
    std::cout << std::endl;
    for (size_t i = 0; i < tracklets.size(); i++)
    {
        std::cout << "Tracklets found in segment " << i+1 << "/" << tracklets.size() << std::endl;
        for (size_t j = 0; j < tracklets[i].size(); j++)
        {
            std::cout << "Tracklet " << j+1 << "/" << tracklets[i].size() << std::endl;
            Node::print_detection_path(tracklets[i][j].detection_track);
            std::cout << "Is end of trajectory: " << tracklets[i][j].is_end_of_traj << std::endl;
            std::cout << "Is start of trajectory: " << tracklets[i][j].is_start_of_traj << std::endl;
            std::cout << "Is hypothetical: " << tracklets[i][j].is_hypothetical << std::endl;
        }
    }
}
