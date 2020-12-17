#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <string_view>
#include <iostream>
#include <thread>  
#include <fstream> 
#include <sstream>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm> 
#include <chrono>
#include <utility>
#include <limits>

#include "gmcp.hpp"
#include "utils.hpp"
#include "video.hpp"
#include "tracklet.hpp"
#include "node.hpp"
#include "iou.hpp"
#include "edge.hpp"

#include <opencv2/opencv.hpp>

void get_parameters(int argc, char **argv, int& segment_size, std::string& in_video, std::string& out_video, 
                    std::string& detector, std::string& detector_cfg, std::string& tmp_folder)
{
    int opt;
    while((opt = getopt(argc, argv, "s:i:o:d:c:f:h")) != -1)
    {
        switch (opt)
        {
            case 's':
                segment_size = std::stoi(optarg);
                break;
            case 'i':
                in_video = optarg;
                break;
            case 'o':
                out_video = optarg;
                break;
            case 'd':
                detector = optarg;
                break;
            case 'c':
                detector_cfg = optarg;
                break;
            case 'f':
                tmp_folder = optarg;
                break;
            case 'h':
                print_usage_info();
                exit(0);
            default:
                std::cout << "Unsupported parameter passed to the script. Aborting." << std::endl;
                print_usage_info();
                abort();
        }
    }
}

void verify_parameters(int segment_size, std::string in_video, std::string out_video, 
                       std::string detector, std::string detector_cfg, std::string tmp_folder)
{
    if (segment_size < 2)
    {
        std::cout << "Segment size has to be at least 3. Aborting.";
        exit(0);
    }
    if (in_video == "")
    {
        std::cout << "Please specify input video path. Aborting.";
        exit(0);
    }
    if (out_video == "")
    {
        std::cout << "Please specify output video path. Aborting.";
        exit(0);
    }
    if (detector != "yolo" && detector != "ssd")
    {
        std::cout << "Unsupported detector. Aborting.";
        exit(0);
    }
    if (detector_cfg == "")
    {
        std::cout << "Please specify detector config path. Aborting.";
        exit(0);
    }
    if (tmp_folder == "")
    {
        std::cout << "Please specify path where tmp files should be stored. Aborting.";
        exit(0);
    }
}

void detect(std::string detector, std::string detector_cfg, int frame_cnt, std::string video, std::string tmp_folder) 
{
    std::stringstream ss;
    ss << "python3 ../detectors/detect.py --detector " << detector 
       << " --cfg " << detector_cfg
       << " --video " << video
       << " --tmp_folder " << tmp_folder
       << " --frame_cnt " << std::to_string(frame_cnt);
    std::string detect_command = ss.str();
    std::cout << "Executing: " << detect_command << std::endl;
    system(detect_command.c_str());
}

vector<Node> load_cluster_nodes(std::string csv_file, int video_w, int video_h, 
                                const cv::Mat &frame, int frame_id) 
{
    vector<Node> nodes;
    std::ifstream f(csv_file);
    std::string line;

    if(!f.is_open()) 
         throw std::runtime_error("Could not open file");

    // one bounding box a line
    int id = 0;
    int x_min, y_min, x_max, y_max, x_center, y_center, height, width;
    float area;
    while(std::getline(f, line))
    {
        std::stringstream ss(line); 
        int coords[4];
        constexpr int const coord_cnt = 4; 
        for (int i = 0; i < coord_cnt; i++)
        {
            std::string substr;
            getline(ss, substr, ',');
            coords[i] = std::stoi(substr);
        }
        x_min = coords[0];
        y_min = coords[1];
        x_max = coords[2];
        y_max = coords[3];
        if (x_min < 0)
            x_min = 0;
        if (y_min < 0)
            y_min = 0;
        if (x_max > video_w)
            x_max = video_w;
        if (y_max > video_h)
            y_max = video_h;         
        x_center = (x_min + x_max) / 2;  // todo mv this to Node
        y_center = (y_min + y_max) / 2;  // todo mv this to Node
        height = y_max - y_min;
        width = x_max - x_min;
        area = height * width;
        Box d = {
            .x = x_center,
            .y = y_center,
            .id = id,
            .x_min = x_min,
            .y_min = y_min,
            .x_max = x_max,
            .y_max = y_max,
            .height = height,
            .width = width,
            .area = area
        };
        nodes.push_back(Node(d, frame, id, frame_id));
        id++; 
    }
    f.close();
    return nodes;
}

vector2d<Node> load_nodes(int frame_cnt, std::string tmp_folder, int video_w, int video_h) 
{
    // loads a vector of Node for each frame
    vector2d<Node> nodes(frame_cnt, vector<Node>());
    for (int i = 0; i < frame_cnt; i += 1) 
    {
        std::stringstream ss;
        ss << tmp_folder << "/csv/frame" << i << ".csv";
        std::string csv_path = ss.str();
        cv::Mat frame = cv::imread(get_frame_path(i, tmp_folder));
        nodes[i] = load_cluster_nodes(csv_path, video_w, video_h, frame, i);
    }
    print_nodes(nodes);
    return nodes;
}

// bool is_any_frame_without_detections(vector2d<Detection> &detections, int seg_counter, int seg_size)
// {
//     int start = seg_counter * seg_size;
//     for (int i = 0; i < seg_size; i++)
//     {
//         if (detections[start+i].empty())
//         {
//             std::cout << "Not enough detections in this segment for more paths" << std::endl;
//             return true;
//         }
//     }
//     return false;
// }

// auto track(vector2d<Detection> &detections,
//            vector2d<HistInterKernel> &net_cost,
//            const vector2d<cv::Mat> &histograms, 
//            int segment_cnt, int segment_size, int max_detections_per_frame)
// {
//     vector2d<Tracklet> tracklets(segment_cnt, vector<Tracklet>());

//     for (int i = 0; i < segment_cnt; i++)
//     {
//         std::cout << std::endl << "Tracking in segment " << i+1 << "/" << segment_cnt << std::endl;
//         print_detections_left_cnt(detections, i, segment_size);
//         int j = 0;
//         while (j < max_detections_per_frame)
//         {
//             // print_net_cost(net_cost);
//             std::cout << std::endl << "Tracking object number " << j+1 
//                       << "/" << max_detections_per_frame << std::endl;
//             if (is_any_frame_without_detections(detections, i, segment_size))
//                 break;
//             auto detection_ids = get_initial_detection_path(net_cost, segment_size, i);
//             auto path = get_detection_path(detections, detection_ids, i);

//             auto app_cost = get_appearance_cost(net_cost, detection_ids, i);
//             auto motion_cost = get_motion_cost(path, segment_size, i);

//             // there the best possible path should be chosen, based on both app and motion cost
//             // current solution is just too greedy
//             tracklets[i].push_back(Tracklet(path, histograms, i, segment_size, j));
//             remove_path(detections, detection_ids, i);
//             remove_path(net_cost, detection_ids, i);
//             print_detections_left_cnt(detections, i, segment_size);
//             print_detections_left_ids(detections, i, segment_size);
//             j++;
//         }
//     }
//     print_tracklets(tracklets, segment_cnt);
//     return tracklets;
// }

void draw_rectangle(const Box &d, const cv::Scalar &color, cv::Mat &img)
{
    constexpr int const line_thickness = 2; 
    cv::Rect rect(d.x_min, d.y_min, d.width, d.height);
    cv::rectangle(img, rect, color, line_thickness);
}

// void set_tracklets_net_costs(vector2d<Tracklet> &tracklets, int segment_cnt)
// {
//     std::cout << std::endl;
//     if (segment_cnt < 2)
//         return;
//     for (int i = 0; i < tracklets.size() - 1; i++) // which segment
//     {
//         std::cout << "Setting tracklet net costs for segment " << i << std::endl;
//         for (int j = 0; j < tracklets[i].size(); j++) // from which tracklet in this segment
//         {
//             std::cout << "Setting tracklet net costs for tracklet " << j << std::endl;
//             tracklets[i][j].print();

//             for (int k = 0; k < tracklets[i+1].size(); k++) // id of the tracklet in the next segment
//             {
//                 std::cout << "Calculating HIK with ";
//                 tracklets[i+1][k].print();
                
//                 double hik_val = cv::compareHist(tracklets[i][j].histogram, 
//                                                  tracklets[i+1][k].histogram,
//                                                  3); // CV_COMP_INTERSECT
//                 HistInterKernel hik = {
//                     .id1 = j,
//                     .id2 = k,
//                     .value = hik_val
//                 };
//                 hik.print();
//                 tracklets[i][j].net_cost.push_back(hik);
//             }
//         }
//     }
//     print_tracklets_net_costs(tracklets, tracklets.size());
// }

// int get_trajectory_cnt(vector2d<Tracklet> &tracklets)
// {
//     int min = tracklets[0].size();
//     for (int i = 1; i < tracklets.size(); i++)
//         if (tracklets[i].size() < min)
//             min = tracklets[i].size();
//     std::cout << "Looking for " << min << " trajectories" << std::endl;
//     return min;
// }

// void assign_trajectory_ids(vector2d<Tracklet> &tracklets, int segment_cnt, int trajectory_cnt)
// {
//     if (segment_cnt < 2)
//         return;
//     // this is a greedy PoC
//     // first of all find the cheapest way from any node from the first segment to any node in the second segment
//     vector2d<int> used(segment_cnt, vector<int>());
//     for (int traj_id = 0; traj_id < trajectory_cnt; traj_id++) 
//     {
//         HistInterKernel first;
//         vector<HistInterKernel> first_seg_cheapest;
//         for (int i = 0; i < tracklets[0].size(); i++)
//         {
//             first_seg_cheapest.push_back(get_cheapest(tracklets[0][i].net_cost, used[0])); 
//         }
//         HistInterKernel first_hik = get_cheapest(first_seg_cheapest, used[0]);
//         std::cout << "First hik in trajectory: ";
//         first_hik.print();

//         tracklets[0][first_hik.id1].trajectory_id = traj_id;
//         tracklets[1][first_hik.id2].trajectory_id = traj_id;
//         tracklets[0][first_hik.id1].net_cost.clear();
//         used[0].push_back(first_hik.id2);
                
//         int from_hik = first_hik.id2;
//         for (int i = 1; i < segment_cnt - 1; i++)
//         {
//             HistInterKernel next_hik = get_cheapest(tracklets[i][from_hik].net_cost, from_hik, used[i]);
//             next_hik.print();
//             tracklets[i+1][next_hik.id2].trajectory_id = traj_id;
//             tracklets[i][next_hik.id1].net_cost.clear();
//             from_hik = next_hik.id2;
//             used[i].push_back(first_hik.id2);
//         }
//     }
// }

// vector2d<Detection> form_trajectories(vector2d<Tracklet> tracklets, int trajectory_cnt, int segment_cnt)
// {
//     vector2d<Detection> trajectories(trajectory_cnt, vector<Detection>());
//     if (segment_cnt == 1)
//     {
//         for (int i = 0; i < tracklets[0].size(); i++)
//             for (auto d: tracklets[0][i].detections)
//                 trajectories[i].push_back(d);
//         return trajectories;
//     }

//     for (int i = 0; i < trajectory_cnt; i++)
//     {
//         for (int j = 0; j < segment_cnt; j++)
//         {
//             for (int k = 0; k < tracklets[j].size(); k++)
//             {
//                 if (tracklets[j][k].id == i)
//                 {
//                     for (auto d: tracklets[j][k].detections)
//                         trajectories[i].push_back(d);
//                     std::cout << "Adding to trajectory " << i << " ";
//                     print_detection_path(tracklets[j][k].detections);
//                 }
//             }
//         }
//     }
//     return trajectories;
// }

// void draw_bounding_boxes(const vector2d<Detection> &trajectories, int frame_cnt,    // todo: maybe a vector2d of Node?
//                          std::string tmp_folder, vector<cv::Scalar> colors)
// {
//     for (int i = 0; i < frame_cnt; i++)
//     {
//         auto path = get_frame_path(i, tmp_folder);
//         cv::Mat img = cv::imread(path);
//         for (int j = 0; j < trajectories.size(); j++)
//         {
//             draw_rectangle(trajectories[j][i], colors[j], img);
//         }
//         cv::imwrite(path, img);
//     }
// }

int get_min_detections_in_segment_cnt(const vector2d<Node> &nodes, int segment_size, int seg_counter, int segment_cnt, int start)
{
    int min_detections_in_segment_cnt = 1000;
    for (int i = 0; i < segment_size; i++)
    {
        int detections_in_frame_cnt = nodes[start+i].size();
        if (detections_in_frame_cnt < min_detections_in_segment_cnt)
            min_detections_in_segment_cnt = detections_in_frame_cnt;
    }
    std::cout << "Min detections in segment " << seg_counter+1 << "/" << segment_cnt << ": " << min_detections_in_segment_cnt << std::endl;
    return min_detections_in_segment_cnt;
}

bool iou_cmp(const IOU& a, const IOU& b)
{
    return a.value > b.value;
}

bool hik_cmp(const HistInterKernel& a, const HistInterKernel& b)
{
    return a.value < b.value;
}

int main(int argc, char **argv) {
    auto begin = std::chrono::steady_clock::now();
    int segment_size = 0;
    std::string in_video, out_video, detector, detector_cfg, tmp_folder;
    
    get_parameters(argc, argv, segment_size, in_video, out_video, detector, detector_cfg, tmp_folder);
    verify_parameters(segment_size, in_video, out_video, detector, detector_cfg, tmp_folder);
    print_parameters(segment_size, in_video, out_video, detector, detector_cfg, tmp_folder);
    clear_tmp(tmp_folder);
    make_tmp_dirs(tmp_folder);

    cv::VideoCapture in_cap(in_video);
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    int video_w = in_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int video_h = in_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const int frame_cnt = get_trimmed_frame_cnt(in_cap, segment_size);
    const int segment_cnt = frame_cnt / segment_size;
    const std::string tmp_video = tmp_folder + "/input.mp4";
    prepare_tmp_video(in_cap, frame_cnt, tmp_folder, in_video, tmp_video);
    // in_cap.release();  // this should be automatic

    auto detect_begin = std::chrono::steady_clock::now();
    detect(detector, detector_cfg, frame_cnt - 1, tmp_video, tmp_folder);
    auto detect_end = std::chrono::steady_clock::now();

    vector2d<Node> nodes = load_nodes(frame_cnt, tmp_folder, video_w, video_h); // vector of Nodes (detections) for each frame
    int max_nodes_per_cluster = get_max_nodes_per_cluster(nodes); // max detections found in one frame
    // auto colors = get_colors(max_nodes_per_cluster);

    vector2d<Tracklet> tracklets(segment_size, vector<Tracklet>());
   
    for (int seg_counter = 0; seg_counter < segment_cnt; seg_counter++)
    {
        std::cout << std::endl << "Tracking in segment " << seg_counter+1 << "/" << segment_cnt << std::endl;
        // first we build a graph per segment
        // the graph has n clusters, where n is the number of frames in one segment
        // firstly, all nodes that are not from the same cluster are connected
        // but then we use IOU to determine which nodes from consequent frames must be the same object
        // then we can get rid of a lot of edges that are not needed anymore
        // if we struggled to find the solution using iou, we calculate the hik cost and try to minimize cost

        // at this stage there is a strong need of fighting with occlusions

        int start = seg_counter * segment_size;
        int min_detections_in_segment_cnt = get_min_detections_in_segment_cnt(nodes, segment_size, seg_counter, segment_cnt, start);

        vector<IOU> ious;
        // get the ious of subsequent frame detections
        for (int i = 0; i < segment_size - 1; i++)
        {
            for (int j = 0; j < nodes[start + i].size(); j++)  // detection id in the current frame
            {
                for (int k = 0; k < nodes[start + i + 1].size(); k++)  // detection id in the next frame
                {
                    
                    float val = nodes[start + i][j].coords.calc_iou(nodes[start + i + 1][k].coords);
                    if (val > 0)  // this should be some threshold rather than zero
                    { 
                        IOU iou = {
                            .frame = start + i,
                            .id1 = j,
                            .id2 = k,
                            .value = val
                        };
                        ious.push_back(iou);
                    }
                }
            }
        }
        // sorting ious in descending order
        std::sort(ious.begin(), ious.end(), iou_cmp);
        // connecting nodes with high ious
        for(auto const& iou: ious)
        {
            if (nodes[iou.frame][iou.id1].next_node_id == -1 && nodes[iou.frame + 1][iou.id2].prev_node_id == -1)  // if next node not set as yet
            {
                iou.print();
                nodes[iou.frame][iou.id1].next_node_id = iou.id2;
                nodes[iou.frame + 1][iou.id2].prev_node_id = iou.id1;
            }
        }

        // now we have to connect the nodes that still lack prev/next pointers 
        for (int i = 0; i < segment_size - 1; i++)  // for each frame in the segment except for the last one
        {
            int detections_with_next_in_frame = 0;
            vector<HistInterKernel> frame_hiks;
            for (int j = 0; j < nodes[start + i].size(); j++)  // for each detection in the frame
            {
                if (nodes[start + i][j].next_node_id > -1)  // if the detection already has next pointer go to the next detection
                    detections_with_next_in_frame++;
                else
                {
                    // if the detection does not have next pointer, then we calculate its hiks with detections from next frame that lack prev pointer
                    for (int k = 0; k < nodes[start + i + 1].size(); k++)
                    {
                        if (nodes[start + i + 1][k].prev_node_id == -1)
                        {
                            double hik_val = cv::compareHist(nodes[start + i][j].histogram, 
                                                             nodes[start + i + 1][k].histogram,
                                                             3); // CV_COMP_INTERSECT
                            int frame = start + i;
                            HistInterKernel hik = {
                                .id1 = j,
                                .id2 = k,
                                .frame = frame,
                                .value = hik_val
                            };
                            frame_hiks.push_back(hik);
                        }
                    }
                }
            }
            int connections_needed = min_detections_in_segment_cnt - detections_with_next_in_frame;
            if (!connections_needed)
                continue;
            std::sort(frame_hiks.begin(), frame_hiks.end(), hik_cmp);
            for (auto hik: frame_hiks)
            {
                if (nodes[start + i][hik.id1].next_node_id == -1 && nodes[start + i + 1][hik.id2].prev_node_id == -1)
                {
                    nodes[start + i][hik.id1].next_node_id = hik.id2;
                    nodes[start + i + 1][hik.id2].prev_node_id = hik.id1;
                    connections_needed--;
                    if (!connections_needed)
                        break;
                }
            } 
        }

        // get segment's tracklets
        for (int i = 0; i < nodes[start].size(); i++)  // for each detection in the first frame of the segment
        {
            vector<int> tracklet_ids;
            Node node = nodes[start][i];
            tracklet_ids.push_back(i);
            int next = node.next_node_id;
            if (next == -1)
                continue;

            for (int j = 1; j < segment_size; j++)
            {
                node = nodes[start + j][next];
                tracklet_ids.push_back(node.node_id);
                next = node.next_node_id;
                if (next == -1)
                    break;
            }

            if (tracklet_ids.size() != segment_size)
                continue;

            vector<Node> tracklet_nodes;
            for (int j = 0; j < segment_size; j++)
            {
                tracklet_nodes.push_back(nodes[start + j][tracklet_ids[j]]);
            }
            // tracklets[seg_counter].push_back(Tracklet(tracklet_nodes));
        }
    }
    // powinienem mieć możliwość narysowania dowolnego trackletu


    // vector2d<Edge> edges = get_edges()


    // vector2d<HistInterKernel> net_cost = get_net_cost(frame_cnt, histograms); // stop using net_cost naming; perform appearance_cost calculation after all nodes are initialised

// how to solve gmcp?
// 1. Begin with an initial solution. Mark it as current solution.
// 2. Find all neighbor solutions to the current solution.
// 3. If neighbor solution induces a lower cost:
//         - replace current solution with that solution and goto 2
//    Otherwise return current solution as the final solution





    // vector2d<Tracklet> tracklets = track(detections, net_cost, histograms, 
    //                                      segment_cnt, segment_size, max_detections_per_frame);
    
    // int trajectory_cnt = get_trajectory_cnt(tracklets);
    // set_tracklets_net_costs(tracklets, segment_cnt);
    // assign_trajectory_ids(tracklets, segment_cnt, trajectory_cnt);
    // vector2d<Detection> trajectories = form_trajectories(tracklets, trajectory_cnt, segment_cnt);

    // draw_bounding_boxes(trajectories, frame_cnt, tmp_folder, colors);
    // merge_frames(tmp_folder, out_video);

    clear_tmp(tmp_folder);
    auto end = std::chrono::steady_clock::now();
    print_exec_time(begin, end);
    print_detect_time(detect_begin, detect_end);
    return 0;
}
