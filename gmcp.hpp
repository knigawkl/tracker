#pragma once

struct Location
{
    int x, y;
};

struct BoundingBox
{
    int x_min, y_min, x_max, y_max;
};

struct HistInterKernel
{
    int detection_id1, detection_id2;
    double value;
};
