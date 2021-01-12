#include "clique.hpp"


bool Clique::clique_cmp(const Clique& a, const Clique& b)
{
    return a.cost < b.cost;
}

void Clique::print() const
{
     std::cout << "tracklet_idx1: " << tracklet_idx1
               << ", tracklet_idx2: " << tracklet_idx2
               << ", tracklet_idx3: " << tracklet_idx3
               << ", cost: " << cost << std::endl;
}
