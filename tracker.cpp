#include "tracker.hpp"


vector<Node> Tracker::load_cluster_nodes(std::string csv_file, const cv::Mat &frame, int frame_id) 
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
        for (size_t i = 0; i < coord_cnt; i++)
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
        if (x_max > video_info.width)
            x_max = video_info.width;
        if (y_max > video_info.height)
            y_max = video_info.height;         
        x_center = (x_min + x_max) / 2;  // todo mv this to Node
        y_center = (y_min + y_max) / 2;  // todo mv this to Node
        height = y_max - y_min;
        width = x_max - x_min;
        area = height * width;
        Box d = {
            .x = x_center,
            .y = y_center,
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

vector2d<Node> Tracker::load_nodes()
{
    // loads a vector of Node for each frame
    vector2d<Node> nodes(video_info.frame_cnt, vector<Node>());
    for (size_t i = 0; i < video_info.frame_cnt; i += 1) 
    {
        std::stringstream ss;
        ss << video_info.tmp_dir << "/csv/frame" << i << ".csv";
        std::string csv_path = ss.str();
        cv::Mat frame = cv::imread(utils::get_frame_path(i, video_info.tmp_dir));
        nodes[i] = load_cluster_nodes(csv_path, frame, i);
    }
    Node::print_nodes(nodes);
    return nodes;
}

int Tracker::get_min_detections_in_segment_cnt(const vector2d<Node> &nodes, int segment_idx, int start_frame_idx)
{
    int min_detections_in_segment_cnt = 1000;
    for (size_t i = 0; i < video_info.segment_size; i++)
    {
        int detections_in_frame_cnt = nodes[start_frame_idx+i].size();
        if (detections_in_frame_cnt < min_detections_in_segment_cnt)
            min_detections_in_segment_cnt = detections_in_frame_cnt;
    }
    std::cout << "Min detections in segment " << segment_idx+1 << "/" << video_info.segment_cnt << ": " << min_detections_in_segment_cnt << std::endl;
    return min_detections_in_segment_cnt;
}

vector2d<Tracklet> Tracker::get_tracklets(vector2d<Node> &nodes)
{
    // returns a vector of tracklets for each segment
    vector2d<Tracklet> tracklets(video_info.segment_cnt, vector<Tracklet>());

    for (size_t seg_counter = 0; seg_counter < video_info.segment_cnt; seg_counter++)
    {
        std::cout << std::endl << "Tracking in segment " << seg_counter+1 << "/" << video_info.segment_cnt << std::endl;
        // first we build a graph per segment
        // the graph has n clusters, where n is the number of frames in one segment
        // firstly, all nodes that are not from the same cluster are connected
        // but then we use IOU to determine which nodes from consequent frames must be the same object
        // then we can get rid of a lot of edges that are not needed anymore
        // if we struggled to find the solution using iou, we calculate the edge cost and try to minimize it

        // at this stage there is a strong need for fighting with occlusions

        int start = seg_counter * video_info.segment_size;
        int min_detections_in_segment_cnt = get_min_detections_in_segment_cnt(nodes, seg_counter, start);

        vector<IOU> ious;
        // get the ious of subsequent frame detections
        for (int i = 0; i < video_info.segment_size - 1; i++)
        {
            for (int j = 0; j < nodes[start + i].size(); j++)  // detection id in the current frame
            {
                for (int k = 0; k < nodes[start + i + 1].size(); k++)  // detection id in the next frame
                {
                    
                    float val = nodes[start + i][j].coords.calc_iou(nodes[start + i + 1][k].coords);
                    if (val > 0)  // this should be some threshold rather than zero
                    { 
                        IOU iou = {
                            .start_frame = start + i,
                            .detection_id1 = j,
                            .detection_id2 = k,
                            .value = val
                        };
                        ious.push_back(iou);
                    }
                }
            }
        }
        std::sort(ious.begin(), ious.end(), IOU::iou_cmp);
        for(auto const& iou: ious)
        {
            if (nodes[iou.start_frame][iou.detection_id1].next_node_id == -1 && nodes[iou.start_frame + 1][iou.detection_id2].prev_node_id == -1)  // if next node not set as yet
            {
                iou.print();
                nodes[iou.start_frame][iou.detection_id1].next_node_id = iou.detection_id2;
                nodes[iou.start_frame + 1][iou.detection_id2].prev_node_id = iou.detection_id1;
            }
        }

        // now we have to connect the nodes that still lack prev/next pointers 
        for (size_t i = 0; i < video_info.segment_size - 1; i++)  // for each frame in the segment except for the last one
        {
            int detections_with_next_in_frame = 0;
            vector<Edge> frame_edges;
            for (size_t j = 0; j < nodes[start + i].size(); j++)  // for each detection in the frame
            {
                if (nodes[start + i][j].next_node_id > -1)  // if the detection already has next pointer go to the next detection
                    detections_with_next_in_frame++;
                else
                {
                    // if the detection does not have next pointer, then we calculate its hiks with detections from next frame that lack prev pointer
                    for (size_t k = 0; k < nodes[start + i + 1].size(); k++)
                    {
                        if (nodes[start + i + 1][k].prev_node_id == -1)
                            frame_edges.push_back(Edge(nodes[start + i][j], nodes[start + i + 1][k]));
                    }
                }
            }
            int connections_needed = min_detections_in_segment_cnt - detections_with_next_in_frame;
            if (!connections_needed)
                continue;
            std::sort(frame_edges.begin(), frame_edges.end(), Edge::edge_cmp);
            for (auto e: frame_edges)
            {
                if (nodes[start + i][e.start.id].next_node_id == -1 && nodes[start + i + 1][e.end.id].prev_node_id == -1)
                {
                    nodes[start + i][e.start.id].next_node_id = e.end.id;
                    nodes[start + i + 1][e.end.id].prev_node_id = e.start.id;
                    connections_needed--;
                    if (!connections_needed)
                        break;
                }
            } 
        }
        for (size_t i = 0; i < nodes[start].size(); i++)
        {
            vector<int> tracklet_ids;
            Node node = nodes[start][i];
            tracklet_ids.push_back(i);
            int next = node.next_node_id;
            if (next == -1)
                continue;
            for (size_t j = 1; j < video_info.segment_size; j++)
            {
                node = nodes[start + j][next];
                tracklet_ids.push_back(node.id);
                next = node.next_node_id;
                if (next == -1)
                    break;
            }
            if (tracklet_ids.size() != video_info.segment_size)
                continue;
            vector<Node> tracklet_nodes;
            for (size_t j = 0; j < video_info.segment_size; j++)
                tracklet_nodes.push_back(nodes[start + j][tracklet_ids[j]]);
            tracklets[seg_counter].push_back(Tracklet(tracklet_nodes, video_info));
        }
    }
    Tracklet::print_tracklets(tracklets);
    return tracklets;
}

void Tracker::merge_tracklets(vector2d<Tracklet> &tracklets)
{
    for (size_t i = 2; i < video_info.segment_cnt; i += 2)
    {
        std::cout << "SMCP in segment: " << i << std::endl;
        int minimum = std::min({tracklets[i - 2].size(), tracklets[i - 1].size(), tracklets[i].size()});
        vector<Clique> cliques;

        for (size_t j = 0; j < tracklets[i].size(); j++)
        {
            for (size_t k = 0; k < tracklets[i - 1].size(); k++)
            {
                for (size_t l = 0; l < tracklets[i - 2].size(); l++)
                {
                    bool is_new = true;
                    for (Clique const& c: cliques)
                        if (c.tracklet_idx1 == l && c.tracklet_idx2 == k && c.tracklet_idx3 == j)
                            is_new = false;
                    if (is_new)
                    {
                        Edge e1 = Edge(tracklets[i][j].centroid, tracklets[i - 1][k].centroid);
                        Edge e2 = Edge(tracklets[i - 1][k].centroid, tracklets[i - 2][l].centroid);
                        Edge e3 = Edge(tracklets[i - 2][l].centroid, tracklets[i][j].centroid);
                        cliques.push_back(Clique(e1, e2, e3, l, k, j));
                    }
                }
            }
        }
        std::sort(cliques.begin(), cliques.end(), Clique::clique_cmp);
        vector<Clique> solution;
        for (int i = 0; i < minimum; i++)
        {
            Clique min_clique = cliques.front();
            solution.push_back(min_clique);
            vector<Clique> slimmed_cliques;
            // na tym etapie trzeba już wykrywaź odchylenia w trackletach -> jeśli dodajemy tracklet hipotetyczny, to nie slimujemy, więc ify będą trzy
            // trzeba wymyślić sposób na faworyzowanie rozwiązania odnalezionego nad hipotetycznym
            for (const Clique& c: cliques)
                if (c.tracklet_idx1 != min_clique.tracklet_idx1 && c.tracklet_idx2 != min_clique.tracklet_idx2 && c.tracklet_idx3 != min_clique.tracklet_idx3)
                    slimmed_cliques.push_back(c);
            cliques = slimmed_cliques;
        }
        std::cout << "Solution: " << std::endl;
        for (const Clique& s: solution)
        {
            s.print();
            cv::Scalar clique_color = tracklets[i-2][s.tracklet_idx1].color;
            tracklets[i-1][s.tracklet_idx2].color = clique_color;
            tracklets[i][s.tracklet_idx3].color = clique_color;
        }
    }
}

void Tracker::track(const std::string& out_video)
{
    vector2d<Node> nodes = load_nodes();
    vector2d<Tracklet> tracklets = get_tracklets(nodes);
    merge_tracklets(tracklets);
    Tracklet::draw_tracklets(tracklets);
    video::merge_frames(out_video, video_info);
}