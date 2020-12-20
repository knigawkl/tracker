#pragma once

#include <iostream>
#include <vector>

struct IOU
{
    int frame;
    int id1;
    int id2;
    float value;

    void print() const;
    static bool iou_cmp(const IOU& a, const IOU& b);
};
