#include <string>
#include <sstream>  
#include <iostream>
#include "detector.hpp"


void Detector::detect(const video::info& video_info) 
{
    std::stringstream ss;
    ss << "python3 ../detectors/detect.py --detector " << type
       << " --cfg " << cfg
       << " --video " << video_info.tmp_video
       << " --tmp_folder " << video_info.tmp_folder
       << " --frame_cnt " << std::to_string(video_info.frame_cnt - 1);
    std::string detect_command = ss.str();
    std::cout << "Executing: " << detect_command << std::endl;
    system(detect_command.c_str());
}
