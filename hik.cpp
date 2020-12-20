#include <iostream>

#include "hik.hpp"


void HistInterKernel::print() const
{
    std::cout << "id1: " << id1 
              << ", id2: " << id2  
              << ", frame: " << frame
              << ", value: " << value << std::endl;
}

bool HistInterKernel::hik_cmp(const HistInterKernel& a, const HistInterKernel& b)
{
    return a.value < b.value;
}
