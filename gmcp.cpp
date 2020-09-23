#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <string_view>
#include <iostream>
#include <fstream> 
#include <sstream>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm> 

#include "gmcp.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>


int get_trimmed_frame_cnt(const cv::VideoCapture& cap, int frames_in_segment) {
    // calculates max length of video in frames if it has to be a multiple of frames_in_segment
    int video_in_frame_cnt = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Input video frames count: " << video_in_frame_cnt << std::endl;
    return video_in_frame_cnt / frames_in_segment * frames_in_segment;
}

void trim_video(std::string video_in, std::string video_out, int frame_cnt) {
    // trims video stored at video_in path to frame_cnt frames
    std::stringstream ss;
    ss << "ffmpeg -i " << video_in << " -vframes " << std::to_string(frame_cnt) << " -acodec copy -vcodec copy " << video_out << " -y";
    std::string trim_command = ss.str();
    std::cout << "Executing: " << trim_command << std::endl;
    system(trim_command.c_str());
}

void detect(std::string detector, std::string detector_cfg, int segment_size, int frame_cnt, std::string video, std::string tmp_folder) {
    // initiates object detections on selected frames of the trimmed video
    // the detections are then stored in csv files
    std::stringstream ss;
    ss << "python3 ../detectors/detect.py --detector " << detector 
       << " --cfg " << detector_cfg
       << " --segment_size " << std::to_string(segment_size) 
       << " --video " << video
       << " --tmp_folder " << tmp_folder
       << " --frame_cnt " << std::to_string(frame_cnt);
    std::string detect_command = ss.str();
    std::cout << "Executing: " << detect_command << std::endl;
    system(detect_command.c_str());
}

std::vector<BoundingBox> load_detections(std::string csv_file) {
    std::vector<BoundingBox> boxes;
    std::ifstream f(csv_file);
    std::string line, colname;
    int val;

    std::cout << "Loading detections from " << csv_file << std::endl;

    if(!f.is_open()) 
         throw std::runtime_error("Could not open file");

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
        BoundingBox box = {
            .x_min = coords[0],
            .y_min = coords[1],
            .x_max = coords[2],
            .y_max = coords[3]
        };
        boxes.push_back(box);
    }
    f.close();
    return boxes;
}

auto get_invariant_histogram(std::string img_path) {
    constexpr int const depth = 32;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat normalized_img;
    cv::Mat channel1, channel2, channel3;
    std::vector<cv::Mat> channels(3);
    cv::Mat fx, gx, fy, gy;
    cv::Mat gim, fim;
    if(img.empty())
    {
        std::cout << "Could not read the image: " << img_path << std::endl;
        abort();
    }
    img.convertTo(normalized_img, CV_32FC3, 1.f/255);
    int width = normalized_img.cols;
    int height = normalized_img.rows;

    float kernel_data[] = {-1, 0, 1};
    cv::Mat kernel(1, 3, CV_32F, kernel_data); // [-1, 0, 1]
    cv::Mat kernel_trans(3, 1, CV_32F, kernel_data);

    split(normalized_img, channels);
    channel1 = channels[0];
    channel2 = channels[1];
    channel3 = channels[2];

    gim = 0.5 * channels[2] + 0.5 * channels[1];
    fim = channels[0];

    cv::filter2D(fim, fx, CV_32F, kernel);
    cv::filter2D(gim, gx, CV_32F, kernel);
    cv::filter2D(fim, fy, CV_32F, kernel_trans);
    cv::filter2D(gim, gy, CV_32F, kernel_trans);

    float histo[depth][depth][depth] = {0};
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int bin1 = std::max(1, int(ceil(depth * channel1.at<float>(i,j))));
            int bin2 = std::max(1, int(ceil(depth * channel2.at<float>(i,j))));
            int bin3 = std::max(1, int(ceil(depth * channel3.at<float>(i,j))));

            float weight = abs(fx.at<float>(i,j) * gy.at<float>(i,j) - fy.at<float>(i,j) * gx.at<float>(i,j));
            histo[bin1][bin2][bin3] = histo[bin1][bin2][bin3] + weight;
        }
    }
    auto *histo_ptr = histo;
    return histo_ptr;
}

void get_detection_centers(std::vector<BoundingBox> detections) {
    
}



// void get_net_cost_matrix() {

// }

int main(int argc, char **argv) {
    int segment_size = 0;
    std::string input_video;
    std::string output_video;
    std::string detector;
    std::string detector_cfg;
    std::string tmp_fixtures;

    const char* usage_info =
    "\n    -s, --segment_size   frames in a segment\n"
    "    -i, --input_video      input video path\n"
    "    -o, --output_video     output video path\n"
    "    -d, --detector         object detector (ssd or yolo)\n"
    "    -c, --detector_cfg     path to detector cfg\n"
    "    -f, --tmp_fixtures     path to folder where temporary files will be stored\n"
    "    -h, --help             show this help msg";

    int opt;
    while((opt = getopt(argc, argv, "s:i:o:d:c:f:h")) != -1)
    {
        switch (opt)
        {
            case 's':
                segment_size = std::stoi(optarg);
                break;
            case 'i':
                input_video = optarg;
                break;
            case 'o':
                output_video = optarg;
                break;
            case 'd':
                detector = optarg;
                break;
            case 'c':
                detector_cfg = optarg;
                break;
            case 'f':
                tmp_fixtures = optarg;
                break;
            case 'h':
                std::cout << "Please provide the following arguments:\n" << usage_info << std::endl;
                exit(0);
            default:
                std::cout << "Unsupported parameter passed to the script. Aborting." << std::endl;
                std::cout << usage_info << std::endl;
                abort();
        }
    }

    if (segment_size == 0)
    {
        std::cout << "Please specify segment size. Aborting.";
        exit(0);
    }
    if (input_video == "")
    {
        std::cout << "Please specify input video path. Aborting.";
        exit(0);
    }
    if (output_video == "")
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
    if (tmp_fixtures == "")
    {
        std::cout << "Please specify path where tmp files should be stored. Aborting.";
        exit(0);
    }

    printf("Segment size set to %d\n", segment_size);
    printf("Input video path set to %s\n", input_video.c_str());
    printf("Output video path set to %s\n", output_video.c_str());
    printf("Detector set to %s\n", detector.c_str());
    printf("Detector cfg path set to %s\n", detector_cfg.c_str());
    printf("Temporary files will be stored in %s\n", tmp_fixtures.c_str());

    make_tmp_dirs(tmp_fixtures);
    std::string const trimmed_video = tmp_fixtures + "tmp.mp4";

    cv::VideoCapture in_cap(input_video);
    auto trimmed_video_frame_cnt = get_trimmed_frame_cnt(in_cap, segment_size);
    trim_video(input_video, trimmed_video, trimmed_video_frame_cnt);
    std::cout << "Input video frames count cut to: " << trimmed_video_frame_cnt << std::endl;

    detect(detector, detector_cfg, segment_size, trimmed_video_frame_cnt, trimmed_video, tmp_fixtures);

    int segment_cnt = trimmed_video_frame_cnt / segment_size;
    std::vector<BoundingBox> detections[segment_cnt];
    for (int i = 0; i < trimmed_video_frame_cnt; i += segment_size) 
    {
        std::cout << "Loading detections of frame " << i << std::endl;
        std::stringstream ss;
        ss << tmp_fixtures << "/csv/frame" << i << ".csv";
        std::string csv_path = ss.str();
        detections[i/segment_size] = load_detections(csv_path);
    }

    unsigned int max_detections_per_frame = 0;
    for(auto const& d : detections)
        if (d.size() > max_detections_per_frame)
            max_detections_per_frame = d.size();
    std::cout << "Max number of detections per frame is: " << max_detections_per_frame << std::endl; 
    for (int i = 0; i < segment_cnt; i++) {
        std::cout << "Analyzing segment: " << i+1 << "/" << segment_cnt << std::endl;
        get_detection_centers(detections[i]);

        // get histo_root
        
        
        // get net_cost_mat
    }

    clear_tmp(tmp_fixtures);
    return 0;
}
