#include "iou.hpp"

bool IOU::iou_cmp(const IOU& a, const IOU& b)
{
    return a.value > b.value;
}

void IOU::print() const
{
    std::cout << "frame: " << frame << ", id1: " << id1 << ", id2: " << id2
              << ", value: " << value << std::endl;
}
