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

#include <opencv2/opencv.hpp>

#include "tracker.hpp"
#include "edge.hpp"
#include "iou.hpp"
#include "hik.hpp"
#include "location.hpp"
#include "node.hpp"
#include "tracklet.hpp"
#include "utils.hpp"
#include "video.hpp"

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
    Node::print_nodes(nodes);
    return nodes;
}

void draw_rectangle(const Box &d, const cv::Scalar &color, cv::Mat &img)
{
    constexpr int const line_thickness = 2; 
    cv::Rect rect(d.x_min, d.y_min, d.width, d.height);
    cv::rectangle(img, rect, color, line_thickness);
}

void draw_trajectory(const vector<Node> &trajectory, std::string tmp_folder, cv::Scalar color)
{
    for (Node node: trajectory)
    {
        auto path = get_frame_path(node.cluster_id, tmp_folder);
        cv::Mat img = cv::imread(path);
        draw_rectangle(node.coords, color, img);
        cv::imwrite(path, img);
    }
}

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

vector2d<Tracklet> get_tracklets(vector2d<Node> &nodes, int segment_size, int segment_cnt, int video_w, int video_h, int video_frame_cnt)
{
    vector2d<Tracklet> tracklets(segment_cnt, vector<Tracklet>());

    for (int seg_counter = 0; seg_counter < segment_cnt; seg_counter++)
    {
        std::cout << std::endl << "Tracking in segment " << seg_counter+1 << "/" << segment_cnt << std::endl;
        // first we build a graph per segment
        // the graph has n clusters, where n is the number of frames in one segment
        // firstly, all nodes that are not from the same cluster are connected
        // but then we use IOU to determine which nodes from consequent frames must be the same object
        // then we can get rid of a lot of edges that are not needed anymore
        // if we struggled to find the solution using iou, we calculate the hik cost and try to minimize cost

        // at this stage there is a strong need for fighting with occlusions

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
        std::sort(ious.begin(), ious.end(), IOU::iou_cmp);
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
            // vector<HistInterKernel> frame_hiks;
            vector<Edge> frame_edges;
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
                            frame_edges.push_back(Edge(nodes[start + i][j], nodes[start + i + 1][k]));
                        }
                    }
                }
            }
            int connections_needed = min_detections_in_segment_cnt - detections_with_next_in_frame;
            if (!connections_needed)
                continue;
            std::sort(frame_edges.begin(), frame_edges.end(), Edge::edge_cmp);
            for (auto e: frame_edges)
            {
                if (nodes[start + i][e.start_node_id].next_node_id == -1 && nodes[start + i + 1][e.end_node_id].prev_node_id == -1)
                {
                    nodes[start + i][e.start_node_id].next_node_id = e.end_node_id;
                    nodes[start + i + 1][e.end_node_id].prev_node_id = e.start_node_id;
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
            tracklets[seg_counter].push_back(Tracklet(tracklet_nodes, video_w, video_h, video_frame_cnt));
        }
    }
    return tracklets;
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
    int max_nodes_per_cluster = Node::get_max_nodes_per_cluster(nodes); // max detections found in one frame
    // auto colors = get_colors(max_nodes_per_cluster);

    // vector of tracklets for each segment
    vector2d<Tracklet> tracklets = get_tracklets(nodes, segment_size, segment_cnt, video_w, video_h, frame_cnt);
    Tracklet::print_tracklets(tracklets);

    for (int i = 0; i < tracklets.size(); i++)
    {
        for (int j = 0; j < tracklets[i].size(); j++)
        {
            uint8_t r, g, b;
            b = rand() % 256;
            r = rand() % 256;
            g = rand() % 256;
            draw_trajectory(tracklets[i][j].detection_track, tmp_folder, cv::Scalar(b, g, r));
        }
    }

    // draw_bounding_boxes(trajectories, frame_cnt, tmp_folder, colors);
    merge_frames(tmp_folder, out_video);

    clear_tmp(tmp_folder);
    auto end = std::chrono::steady_clock::now();
    print_exec_time(begin, end);
    print_detect_time(detect_begin, detect_end);
    return 0;
}
