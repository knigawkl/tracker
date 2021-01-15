#pragma once

#include "video.hpp"


class Detector
{
public:
    std::string type;
    std::string cfg;
    Detector(const std::string& detector, const std::string& detector_cfg): type(detector), cfg(detector_cfg) {};
    void detect(const video::info& video_info);
};
