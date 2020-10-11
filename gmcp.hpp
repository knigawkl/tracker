#pragma once

template <typename T>
using vector = std::vector<T>;

template <typename T>
using vector2d = vector<vector<T>>;

struct HistInterKernel
{
    int id1, id2;
    double value;
    void print() const
    {
        std::cout << "id1: " << id1 
                  << ", id2: " << id2 
                  << ", value: " << value << std::endl;
    }
};

struct Location
{
    int x, y;
    void print() const
    {
        std::cout << "x: " << x << ", y: " << y << std::endl;
    }
};
