#pragma once

#include <stdlib.h>
#include <vector>

void make_tmp_dirs(std::string tmp_folder);

void clear_tmp(std::string tmp_folder);

void mv(std::string what, std::string where);

void cp(std::string what, std::string where);

struct Color
{
    int r, g, b;
    Color()
    {
        r = rand() % 256;
        g = rand() % 256;
        b = rand() % 256;
    }
};

std::vector<Color> get_colors(int vec_len);
