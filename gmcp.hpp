#pragma once

template <typename T>
using vector = std::vector<T>;

template <typename T>
using vector2d = vector<vector<T>>;

struct Location
{
    int x, y;
    void print() const
    {
        std::cout << "x: " << x << ", y: " << y << std::endl;
    }
};
