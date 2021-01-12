#pragma once

#include "edge.hpp"


class Clique
{
public:
    int tracklet_idx1 = -1;
    int tracklet_idx2 = -1;
    int tracklet_idx3 = -1;
    double cost;
    void print() const;
    static bool clique_cmp(const Clique& a, const Clique& b);

    Clique(Edge e1, Edge e2, Edge e3, int tid1, int tid2, int tid3)
    {
        tracklet_idx1 = tid1;
        tracklet_idx2 = tid2;
        tracklet_idx3 = tid3;
        cost = e1.weight + e2.weight + e3.weight;
        print();
    }
};
