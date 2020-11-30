typedef struct {
    int Valid;
    int PreviousType;
    int Offset;
} MedecineCell;

__kernel void calculate_positions(__global float* result, float2 offset, float cellSize, int numberOfCell_X) 
{
    int i = get_global_id(0);

    float2 position = (float2)(((float)(i % numberOfCell_X) - offset.x) * cellSize, ((float)(i / numberOfCell_X) - offset.y) * cellSize);
    vstore2(position, i, result);
}

__kernel void set_cells(__global int* readCells, __global int* cellTypes, __global float* cellColor)
{
    int i = get_global_id(0);

    cellTypes[i] = 1;
    float4 color = (float4)(0.0f, 1.0f, 0.0f, 1.0f);

    vstore4(color, i, cellColor);

    if (readCells[i] == 1)
    {
        cellTypes[i] = 0;
        color = (float4)(0.75f, 0.0f, 0.0f, 1.0f);

        vstore4(color, i, cellColor);
    }
}

__kernel void update_healthy_cancer_cells(__constant int* numberOfCellsPerPartition, __global int* readCells, __global float4* readColors,
    __constant int* indexes, __constant int* neighbors,
    __global int* cellTypes, __global float* cellColor,
    __global int* updatedCells)
{
    int i = get_global_id(0);

    cellTypes[i] = readCells[i];
    float4 color = readColors[i];

    int index = indexes[i % *numberOfCellsPerPartition];
    int neighbor = i + neighbors[index];

    if (readCells[i] == 1)
    {
        int count = 0;
        while (neighbors[index] != 0)
        {
            int neighbor = i + neighbors[index++];
            if (readCells[neighbor] == 0)
                count++;
        }

        if (count >= 6)
        {
            cellTypes[i] = 0;
            color = (float4)(0.75f, 0.0f, 0.0f, 1.0f);
        }
    }
    else if (readCells[i] == 0)
    {
        int count = 0;
        int medecines[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
        while (neighbors[index] != 0)
        {
            int neighbor = i + neighbors[index++];;
            if (readCells[neighbor] == 2)
                medecines[count++] = neighbor;
        }

        if (count >= 6)
        {
            cellTypes[i] = 1;
            color = (float4)(0.0f, 1.0f, 0.0f, 1.0f);

            for (int j = 0; j < 8; j++)
            {
                if (medecines[j] >= 0)
                    updatedCells[medecines[j]] = 1;
            }
        }
    }
    vstore4(color, i, cellColor);
}

__kernel void update_neighbor_cancer_cells(__global int* readCells, __global float4* readColors,
    __global int* cellTypes, __global float* cellColor,
    __global int* updatedCells,
    __global MedecineCell* readMedecineCells, __global MedecineCell* medecineCells)
{
    int i = get_global_id(0);

    cellTypes[i] = readCells[i];
    float4 color = readColors[i];

    medecineCells[i] = readMedecineCells[i];
    if (updatedCells[i] == 1)
    {
        cellTypes[i] = 1;
        color = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
        medecineCells[i].Valid = 0;
    }
    vstore4(color, i, cellColor);
}

__kernel void update_medecine_cells(int numberOfCellInX, int numberOfCellInY,
    __global int* readCells, __global float4* readColors,
    __global int* cellTypes, __global float* cellColor,
    __global MedecineCell* readMedecineCells, __global float4* readMedecineColors,
    __global MedecineCell* medecineCells, __global float* medecineColors)
{
    int i = get_global_id(0);

    cellTypes[i] = readCells[i];
    float4 color = readColors[i];

    if (readMedecineCells[i].Valid > 0)
    {
        cellTypes[i] = readMedecineCells[i].PreviousType;
        color = readMedecineColors[i];
        int yOffset = readMedecineCells[i].Offset / (numberOfCellInX - 1);
        int xOffset = readMedecineCells[i].Offset - (yOffset * numberOfCellInX);
        int y = (i / numberOfCellInX) + yOffset;
        int x = (i % numberOfCellInX) + xOffset;
        if (x >= 0 && x < numberOfCellInX && y >= 0 && y < numberOfCellInY)
        {
            if (readCells[i + readMedecineCells[i].Offset] != 2)
            {
                medecineCells[i + readMedecineCells[i].Offset].Valid = 1;
                medecineCells[i + readMedecineCells[i].Offset].Offset = readMedecineCells[i].Offset;
            }
        }
    }
    vstore4(color, i, cellColor);
}

__kernel void move_medecine_cells(__global int* readCells, __global float4* readColors,
    __global int* cellTypes, __global float* cellColor,
    __global MedecineCell* readMedecineCells,
    __global MedecineCell* medecineCells, __global float* medecineColors)
{
    int i = get_global_id(0);

    cellTypes[i] = readCells[i];
    float4 color = readColors[i];

    if (readMedecineCells[i].Valid > 0)
    {
        medecineCells[i].Valid = 1;
        medecineCells[i].PreviousType = readCells[i];
        medecineCells[i].Offset = readMedecineCells[i].Offset;
        vstore4(color, i, medecineColors);

        cellTypes[i] = 2;
        color = (float4)(1.0f, 1.0f, 0.0f, 1.0f);
    }
    vstore4(color, i, cellColor);
}

__kernel void count_cells(__global int* readCells, __global int* result, int numberOfCell, int numberOfPartitions)
{
    int i = get_global_id(0);
    int startIndex = (numberOfCell / numberOfPartitions) * i;
    int endIndex = (numberOfCell / numberOfPartitions) * (i + 1);
    int outputIndex = i * 3;

    result[outputIndex] = 0;
    result[outputIndex + 1] = 0;
    result[outputIndex + 2] = 0;
    for (int j = startIndex; j < endIndex; j++)
    {
        switch (readCells[j])
        {
        case 0:
            ++result[outputIndex];
            break;
        case 1:
            ++result[outputIndex + 1];
            break;
        case 2:
            ++result[outputIndex + 2];
            break;
        }
    }
}