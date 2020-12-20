#pragma once

struct HistInterKernel
{
    int id1, id2;
    int frame;
    double value;
    void print() const;
    static bool hik_cmp(const HistInterKernel& a, const HistInterKernel& b);
};
