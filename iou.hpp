#pragma once

#include "gmcp.hpp"

struct IOU
{
    int frame;
    int id1;
    int id2;
    float value;

    void print() const
    {
        std::cout << "frame: " << frame << ", id1: " << id1 << ", id2: " << id2
                  << ", value: " << value << std::endl;
    }
};
