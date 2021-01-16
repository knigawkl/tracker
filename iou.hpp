#pragma once

#include <iostream>
#include <vector>

struct IOU
{
public:
    int start_frame;
    int detection_id1;
    int detection_id2;
    float value;
    void print() const;
    static bool iou_cmp(const IOU& a, const IOU& b);
};
