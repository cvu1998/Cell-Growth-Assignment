__kernel void calculate_positions(__global float* result, float2 offset, float cellSize, int numberOfCell_X) {

    int i = get_global_id(0);

    float2 position = (float2)(((float)(i % numberOfCell_X) - offset.x) * cellSize, ((float)(i / numberOfCell_X) - offset.y) * cellSize);
    vstore2(position, i, result);
}

__kernel void update_cells(__global int* readCells, __global int* writeCells*) {

    int i = get_global_id(0);
}

__kernel void update_cells(__global int* readCells, __global int* cellTypes, __global float* cellColors) {

    int i = get_global_id(0);
    if (readCells[i] == 1)
    {
        
    }
}