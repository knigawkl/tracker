#pragma once

template <typename T>
using vector = std::vector<T>;

template <typename T>
using vector2d = std::vector<std::vector<T>>;

template <typename T>
using vector3d = std::vector<std::vector<std::vector<T>>>;

struct Detection
{
    int x, y, id, x_min, y_min, x_max, y_max;
    void print() const
    {
        std::cout << "x: " << x << ", y: " << y << ", id: " << id
                  << ", x_min: " << x_min << ", y_min: " << y_min 
                  << ", x_max: " << x_max << ", y_max: " << y_max << std::endl;
    }
};

struct HistInterKernel
{
    int detection_id1, detection_id2;
    double value;
    void print() const
    {
        std::cout << "detection_id1: " << detection_id1 
                  << ", detection_id2: " << detection_id2 
                  << ", value: " << value << std::endl;
    }
};
