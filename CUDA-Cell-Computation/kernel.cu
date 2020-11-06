#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Elysium.h>

__global__ void calculatePosition(Elysium::Vector2* positions, Elysium::Vector2 offset, float size,
    unsigned int numberOfCellsX, unsigned int numberOfCells)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < numberOfCells)
    {
        positions[i].x = ((float)(i % numberOfCellsX) - offset.x) * size;
        positions[i].y = ((float)(i / numberOfCellsX) - offset.y) * size;
    }
}