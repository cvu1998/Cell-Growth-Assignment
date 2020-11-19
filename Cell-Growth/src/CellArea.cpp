#include "CellArea.h"

#define MAX_SOURCE_SIZE (0x100000)

CellArea::CellArea(Elysium::Vector2 offset)
{
    //-----------OpenCL----------//

    cl_uint platformCount = 0;
    cl_int ret = clGetPlatformIDs(0, NULL, &platformCount);
    ELY_INFO("Number of platforms found: {0}", (unsigned int)platformCount);
    m_Platforms = (cl_platform_id*)malloc(platformCount * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(platformCount, m_Platforms, NULL);

    for (unsigned int i = 0; i < platformCount; i++)
    {
        cl_device_id device;
        m_CPU = clGetDeviceIDs(m_Platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL) == CL_SUCCESS ? device : m_CPU;
        m_GPU = clGetDeviceIDs(m_Platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL) == CL_SUCCESS ? device : m_GPU;
    }

    char* value;
    size_t valueSize;
    clGetDeviceInfo(m_CPU, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(m_CPU, CL_DEVICE_NAME, valueSize, value, NULL);
    ELY_INFO("CPU device: {0}", value);
    free(value);

    clGetDeviceInfo(m_GPU, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(m_GPU, CL_DEVICE_NAME, valueSize, value, NULL);
    ELY_INFO("GPU device: {0}", value);
    free(value);

    clProgram programSource = getProgramSoure("res/cl/cell_kernel.cl");

    // Create OpenCL contexts
    m_CPUContext = clCreateContext(NULL, 1, &m_CPU, NULL, NULL, &ret);
    m_GPUContext = clCreateContext(NULL, 1, &m_GPU, NULL, NULL, &ret);

    // Create command queues
    m_CPUCommandQueue = clCreateCommandQueue(m_CPUContext, m_CPU, 0, &ret);
    m_GPUCommandQueue = clCreateCommandQueue(m_GPUContext, m_GPU, 0, &ret);

    cl_mem positions_mem_obj = clCreateBuffer(m_GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector2), NULL, &ret);

    // Create programs from the kernel source
    m_CPUProgram = clCreateProgramWithSource(m_CPUContext, 1,
        (const char**)&programSource.sourceStr, (const size_t*)&programSource.sourceSize, &ret);
    m_GPUProgram = clCreateProgramWithSource(m_GPUContext, 1,
        (const char**)&programSource.sourceStr, (const size_t*)&programSource.sourceSize, &ret);

    // Build programs
    ret = clBuildProgram(m_CPUProgram, 1, &m_CPU, NULL, NULL, NULL);
    ret = clBuildProgram(m_GPUProgram, 1, &m_GPU, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(m_GPUProgram, "calculate_positions", &ret);

    // Set the arguments of the kernel
    int numberOfCell_X = (int)NumberOfCell_X;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&positions_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(Elysium::Vector2), (void*)&offset);
    ret = clSetKernelArg(kernel, 2, sizeof(float), (void*)&m_CellSize);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&numberOfCell_X);

    // Execute the OpenCL kernel on the list
    size_t local_item_size = 64; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(m_GPUCommandQueue, kernel, 1, NULL,
        &NumberOfCell, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer
    ret = clEnqueueReadBuffer(m_GPUCommandQueue, positions_mem_obj, CL_TRUE, 0,
        NumberOfCell * sizeof(Elysium::Vector2), Positions.data(), 0, NULL, NULL);

    ret = clReleaseKernel(kernel);
    ret = clReleaseMemObject(positions_mem_obj);

    //-----------OpenCL----------//

    float percentage = Random::Float();
    constexpr size_t MinimumNumberOfCancerCell = NumberOfCell / 4;
    NumberOfCancerCells = (unsigned int)(percentage * MinimumNumberOfCancerCell) + MinimumNumberOfCancerCell;
    unsigned int counter = 0;
    std::unordered_set<size_t> indexes;
    while (counter < NumberOfCancerCells)
    {
        size_t index = (size_t)Random::Integer(0, NumberOfCell);
        while (indexes.find(index) != indexes.end())
            index = (size_t)Random::Integer(0, NumberOfCell);
        indexes.insert(index);
        counter++;
    }

    for (size_t i = 0; i < NumberOfCell; i++)
    {
        if (i < s_NumberOfCellsPerPartition)
            setNeighbor((int)i);

        if (indexes.find(i) != indexes.end())
        {
            Colors[i] = s_ColorRed;
            m_Types[i] = CellType::CANCER;
        }
        else
        {
            Colors[i] = s_ColorGreen;
            m_Types[i] = CellType::HEALTHY;
            NumberOfHealthyCells++;
        }
    }

    Elysium::Renderer2D::setPointSize(m_CellSize);

    ELY_INFO("Number of cell per partition: {0}", s_NumberOfCellsPerPartition);
    ELY_INFO("Number of threads: {0}", std::thread::hardware_concurrency());
}

CellArea::~CellArea()
{
    clFlush(m_CPUCommandQueue);
    clFinish(m_CPUCommandQueue);
    clReleaseProgram(m_CPUProgram);
    clReleaseCommandQueue(m_CPUCommandQueue);
    clReleaseContext(m_CPUContext);

    clFlush(m_GPUCommandQueue);
    clFinish(m_GPUCommandQueue);
    clReleaseProgram(m_GPUProgram);
    clReleaseCommandQueue(m_GPUCommandQueue);
    clReleaseContext(m_GPUContext);

    free(m_Platforms);
}

void CellArea::onUpdate(Elysium::Timestep ts)
{
    m_CurrentTime += ts;
    constexpr float UpdateTime = 1.0f / 30.0f;
    if (m_CurrentTime >= UpdateTime)
    {
        m_CurrentTime -= UpdateTime;

        NumberOfCancerCells = 0;
        NumberOfHealthyCells = 0;
        NumberOfMedecineCells = 0;

        if (!m_InputBuffer.empty())
        {
            for (size_t i : m_InputBuffer)
            {
                int counter = 0;
                int numberOfCells = Random::Integer(1, 8);
                for (int j : m_Neighbors[i % s_NumberOfCellsPerPartition])
                {
                    counter++;
                    if (counter >= numberOfCells)
                        break;

                    size_t index = j + ((i + 1) / s_NumberOfCellsPerPartition) * s_NumberOfCellsPerPartition;
                    if (m_MedecineCells.find(index) == m_MedecineCells.end())
                    {
                        m_MedecineCells.insert({ index, { m_Types[index], (int)index - (int)i } });
                        m_Types[index] = CellType::MEDECINE;
                    }
                }
            }
            m_InputBuffer.clear();
        }

        std::thread threads[s_NumberOfThreads];
        CellType* partitions[s_NumberOfThreads];
        PartitionStats stats[s_NumberOfThreads];
        std::unordered_map<size_t, MedecineCell> medecineMap[s_NumberOfThreads];
        for (size_t i = 0; i < s_NumberOfThreads; i++)
        {
            partitions[i] = new CellType[s_NumberOfCellsPerPartition];
            std::copy_n(m_Types.begin() + s_NumberOfCellsPerPartition * i, s_NumberOfCellsPerPartition, partitions[i]);
            threads[i] = std::thread(&CellArea::updateCellsInPartition, this, partitions[i], std::ref(stats[i]), std::ref(medecineMap[i]), s_NumberOfCellsPerPartition * i);
        }
        for (size_t i = 0; i < s_NumberOfThreads; i++)
        {
            threads[i].join();
        }
        m_MedecineCells.clear();

        for (size_t i = 0; i < s_NumberOfThreads; i++)
        {
            m_MedecineCells.insert(medecineMap[i].begin(), medecineMap[i].end());
            NumberOfCancerCells += stats[i].NumberOfCancerCells;
            NumberOfHealthyCells += stats[i].NumberOfHealthyCells;
            NumberOfMedecineCells += stats[i].NumberOfMedecineCells;
            delete[] partitions[i];
        }
    }
}

void CellArea::updateCellsInPartition(CellType* partition, PartitionStats& stats, 
    std::unordered_map<size_t, MedecineCell>& medecineMap,
    size_t min)
{
    std::unordered_set<size_t> updatedMedecine;
    for (size_t i = 0; i < s_NumberOfCellsPerPartition; i++)
    {
        switch (partition[i])
        {
        case CellType::CANCER:
        {
            size_t counter = 0;
            size_t medecineCells[8];
            for (int j : m_Neighbors[i])
            {
                if (partition[j] == CellType::MEDECINE)
                    medecineCells[counter++] = (size_t)j + min;
            }

            if (counter >= 6)
            {
                m_Types[i + min] = CellType::HEALTHY;
                for (size_t j = 0; j < counter; j++)
                {
                    m_Types[medecineCells[j]] = CellType::HEALTHY;
                    updatedMedecine.insert(medecineCells[j]);
                }
            }
        }
            break;
        case CellType::HEALTHY:
        {
            uint8_t cancerCount = 0;
            for (int j : m_Neighbors[i])
                cancerCount += partition[j] == CellType::CANCER ? 1 : 0;

            if (cancerCount >= 6)
                m_Types[i + min] = CellType::CANCER;
        }
            break;
        }
    }

    for (size_t i = 0; i < s_NumberOfCellsPerPartition; i++)
    {
        switch (m_Types[i + min])
        {
        case CellType::CANCER:
            Colors[i + min] = s_ColorRed;
            stats.NumberOfCancerCells++;
            break;
        case CellType::HEALTHY:
            Colors[i + min] = s_ColorGreen;
            stats.NumberOfHealthyCells++;
            break;
        case CellType::MEDECINE:
            moveMedecineCells(i + min, medecineMap, updatedMedecine);
            Colors[i + min] = s_ColorYellow;
            stats.NumberOfMedecineCells++;
            break;
        }
    }
}

void CellArea::moveMedecineCells(size_t cellIndex,
    std::unordered_map<size_t, MedecineCell>& medecineMap,
    std::unordered_set<size_t>& updatedMedecine)
{
    if (updatedMedecine.find(cellIndex) == updatedMedecine.end())
    {
        auto iterators = m_MedecineCells.equal_range(cellIndex);
        for (auto it = iterators.first; it != iterators.second; it++)
        {
            if (it->second.PreviousType != CellType::MEDECINE)
                m_Types[cellIndex] = it->second.PreviousType;

            int index = (int)cellIndex + it->second.offset;
            int xOffset = (it->second.offset < 0) ? it->second.offset + (int)NumberOfCell_X : it->second.offset - (int)NumberOfCell_X;
            xOffset = abs(it->second.offset) == 1 ? it->second.offset : xOffset;
            int yOffset = (it->second.offset < 0) ? (it->second.offset - 1) / (int)NumberOfCell_Y : (it->second.offset + 1) / (int)NumberOfCell_Y;
            int x = (int)cellIndex % NumberOfCell_X;
            int y = (int)cellIndex / NumberOfCell_X;
            if (x + xOffset >= 0 && (size_t)(x + xOffset) < NumberOfCell_X && y + yOffset >= 0 && (size_t)(y + yOffset) < NumberOfCell_Y &&
                index >= 0 && index < NumberOfCell)
            {
                updatedMedecine.insert({ cellIndex });
                medecineMap.insert({ (size_t)index, { m_Types[index], it->second.offset } });
                m_Types[index] = CellType::MEDECINE;
            }
        }
    }
}

clProgram CellArea::getProgramSoure(const char* filepath)
{
    FILE* fp;
    clProgram kernel;

    fp = fopen(filepath, "r");
    if (!fp)
    {
        ELY_ERROR("Failed to load kernel from {0}!", filepath);
        return kernel;
    }

    kernel.sourceStr = (char*)malloc(MAX_SOURCE_SIZE);
    kernel.sourceSize = fread(kernel.sourceStr, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    return kernel;
}

void CellArea::setNeighbor(int index)
{
    for (size_t i = 0; i < 8; i++)
    {
        int x1 = index % (int)NumberOfCell_X;
        int y1 = index / (int)NumberOfCell_X;
        int partitionIndex = s_NeighborIndexes[i] + index;
        int x2 = partitionIndex % (int)NumberOfCell_X;
        int y2 = partitionIndex / (int)NumberOfCell_X;
        int xOffset = (s_NeighborIndexes[i] < 0) ? s_NeighborIndexes[i] + (int)NumberOfCell_X : s_NeighborIndexes[i] - (int)NumberOfCell_X;
        xOffset = abs(s_NeighborIndexes[i]) == 1 ? s_NeighborIndexes[i] : xOffset;
        int yOffset = (s_NeighborIndexes[i] < 0) ? (s_NeighborIndexes[i] - 1) / (int)NumberOfCell_Y : (s_NeighborIndexes[i] + 1) / (int)NumberOfCell_Y;
        if (x1 + xOffset == x2 && y1 + yOffset == y2 && partitionIndex >= 0 && partitionIndex < s_NumberOfCellsPerPartition)
            m_Neighbors[index].push_back(partitionIndex);
    }
}

size_t CellArea::getIndex(const Elysium::Vector2& position)
{
    size_t x = 0;
    for (size_t i = 0; i < NumberOfCell_X; i++)
    {
        if (position.x <= Positions[i].x && i == 0)
        {
            x = 0;
            break;
        }
        else if (position.x <= Positions[i].x)
        {
            x = i - 1;
            break;
        }
        else if (position.x > Positions[i].x && i == NumberOfCell_X - 1)
        {
            x = NumberOfCell_X - 1;
            break;
        }
    }
    int y = -1;
    for (size_t i = 0; i < NumberOfCell; i += NumberOfCell_X)
    {
        y++;
        if (position.y <= Positions[i].y)
            break;
        else if (position.y > Positions[i].y && i == NumberOfCell_Y - 1)
            break;
    }
    return x + (size_t)y * NumberOfCell_X;
}

void CellArea::injectMedecine(const Elysium::Vector2& position)
{
    m_InputBuffer.insert(getIndex(position));
}