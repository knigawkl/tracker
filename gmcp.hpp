#pragma once

struct Detection
{
    int x, y, id;
    void print()
    {
        std::cout << "x: " << x << ", y: " << y << ", id: " << id << std::endl;
    }
};

struct BoundingBox
{
    int x_min, y_min, x_max, y_max;
    void print() const
    {
        std::cout << "x_min: " << x_min << ", y_min: " << y_min 
                  << ", x_max: " << x_max << ", y_max: " << y_max << std::endl;
    }
};

struct HistInterKernel
{
    int detection_id1, detection_id2;
    double value;
};
