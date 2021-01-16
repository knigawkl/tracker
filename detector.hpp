#pragma once

#include "video.hpp"


class Detector
{
public:
    Detector(const std::string& type, const std::string& cfg): type(type), cfg(cfg) {};
    void detect(const video::vidinfo& video_info);
private:
    const std::string type;
    const std::string cfg;
};
