#include "CellArea.h"

CellArea::CellArea(Elysium::Vector2 offset)
{
    int ret = 0;
    float percentage = Random::Float();
    constexpr size_t MinimumNumberOfCancerCell = NumberOfCell / 4;
    NumberOfCancerCells = (unsigned int)(percentage * MinimumNumberOfCancerCell) + MinimumNumberOfCancerCell;
    NumberOfHealthyCells = NumberOfCell - NumberOfCancerCells;
    unsigned int counter = 0;
    std::unordered_set<size_t> indexes;
    int* cancerCells = new int[NumberOfCell];
    memset(cancerCells, 0, NumberOfCell);
    while (counter < NumberOfCancerCells)
    {
        size_t index = (size_t)Random::Integer(0, NumberOfCell - 1);
        while (indexes.find(index) != indexes.end())
            index = (size_t)Random::Integer(0, NumberOfCell - 1);
        indexes.insert(index);
        cancerCells[index] = 1;
        counter++;
    }

    m_CLWrapper.Init("res/cl/cell_kernel.cl");

    // Create the OpenCL kernel
    cl_kernel kernel_positions = clCreateKernel(m_CLWrapper.GPUProgram, "calculate_positions", NULL);
    cl_kernel kernel_cells_info = clCreateKernel(m_CLWrapper.GPUProgram, "set_cells", NULL);

    cl_mem positions_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector2), NULL, NULL);

    cl_mem cancer_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, cancer_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(int), cancerCells, 0, NULL, NULL);

    cl_mem type_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
    cl_mem color_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);

    // Set the arguments of the kernel
    int numberOfCell_X = (int)NumberOfCell_X;
    CL_ASSERT(clSetKernelArg(kernel_positions, 0, sizeof(cl_mem), (void*)&positions_mem_obj));
    CL_ASSERT(clSetKernelArg(kernel_positions, 1, sizeof(Elysium::Vector2), (void*)&offset));
    CL_ASSERT(clSetKernelArg(kernel_positions, 2, sizeof(float), (void*)&m_CellSize));
    CL_ASSERT(clSetKernelArg(kernel_positions, 3, sizeof(int), (void*)&numberOfCell_X));

    CL_ASSERT(clSetKernelArg(kernel_cells_info, 0, sizeof(cl_mem), (void*)&cancer_cells_mem_obj));
    CL_ASSERT(clSetKernelArg(kernel_cells_info, 1, sizeof(cl_mem), (void*)&type_mem_obj));
    CL_ASSERT(clSetKernelArg(kernel_cells_info, 2, sizeof(cl_mem), (void*)&color_mem_obj));

    // Execute the OpenCL kernel on the list
    CL_ASSERT(clEnqueueNDRangeKernel(m_CLWrapper.GPUCommandQueue, kernel_positions, 1, NULL,
        &NumberOfCell, nullptr, 0, NULL, NULL));

    CL_ASSERT(clEnqueueNDRangeKernel(m_CLWrapper.GPUCommandQueue, kernel_cells_info, 1, NULL,
        &NumberOfCell, nullptr, 0, NULL, NULL));

    for (int i = 0; i < (int)s_NumberOfCellsPerPartition; i++)
        setNeighbor(i);

    Elysium::Renderer2D::setPointSize(m_CellSize);

    ELY_INFO("Number of cell per partition: {0}", s_NumberOfCellsPerPartition);

    // Read the memory buffer
    CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, positions_mem_obj, CL_TRUE, 0,
        NumberOfCell * sizeof(Elysium::Vector2), Positions.data(), 0, NULL, NULL));

    CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, type_mem_obj, CL_TRUE, 0,
        NumberOfCell * sizeof(CellType), m_Types.data(), 0, NULL, NULL));
    CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, color_mem_obj, CL_TRUE, 0,
        NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL));

    CL_ASSERT(clReleaseKernel(kernel_positions));
    CL_ASSERT(clReleaseMemObject(positions_mem_obj));

    CL_ASSERT(clReleaseKernel(kernel_cells_info));
    CL_ASSERT(clReleaseMemObject(cancer_cells_mem_obj));
    CL_ASSERT(clReleaseMemObject(type_mem_obj));
    CL_ASSERT(clReleaseMemObject(color_mem_obj));

    delete[] cancerCells;
}

CellArea::~CellArea()
{
    m_CLWrapper.Shutdown();
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
                int j = m_Indexes[i % s_NumberOfCellsPerPartition];
                while (m_Neighbors[j] != 0)
                {
                    counter++;
                    if (counter >= numberOfCells)
                        break;

                    size_t index = i + m_Neighbors[j];
                    if (m_Types[index] != CellType::MEDECINE)
                    {
                        m_MedecineCells[index] = { 1, m_Types[index], m_Neighbors[j] };
                        m_MedecineColors[index] = Colors[index];
                        m_Types[index] = CellType::MEDECINE;
                        Colors[index] = s_ColorYellow;
                    }
                    j++;
                }
            }
            m_InputBuffer.clear();
        }

        updateHealthyAndCancerCells();
        updateMedecineCells();
        countCells();
    }
}

void CellArea::setNeighbor(int index)
{
    m_Indexes[index] = (int)m_Neighbors.size();
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
            m_Neighbors.push_back(s_NeighborIndexes[i]);
    }
    m_Neighbors.push_back(0);
}

void CellArea::updateHealthyAndCancerCells()
{
    int* updatedCells = new int[NumberOfCell];
    memset(updatedCells, 0, NumberOfCell);
    {
        cl_kernel kernel_cells_update = clCreateKernel(m_CLWrapper.GPUProgram, "update_healthy_cancer_cells", NULL);

        cl_mem cell_per_partition_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
        cl_mem past_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem past_colors_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        cl_mem indexes_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, m_Indexes.size() * sizeof(int), NULL, NULL);
        cl_mem neighbors_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, m_Neighbors.size() * sizeof(int), NULL, NULL);

        int numberOfCellsPerPartition = (int)s_NumberOfCellsPerPartition;
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, cell_per_partition_obj, CL_TRUE, 0, sizeof(int), &numberOfCellsPerPartition, 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(int), m_Types.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_colors_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, indexes_mem_obj, CL_TRUE, 0, m_Indexes.size() * sizeof(int), m_Indexes.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, neighbors_mem_obj, CL_TRUE, 0, m_Neighbors.size() * sizeof(int), m_Neighbors.data(), 0, NULL, NULL);

        cl_mem type_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem color_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        cl_mem updated_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(int), NULL, NULL);

        CL_ASSERT(clSetKernelArg(kernel_cells_update, 0, sizeof(cl_mem), (void*)&cell_per_partition_obj));
        CL_ASSERT(clSetKernelArg(kernel_cells_update, 1, sizeof(cl_mem), (void*)&past_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_cells_update, 2, sizeof(cl_mem), (void*)&past_colors_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_cells_update, 3, sizeof(cl_mem), (void*)&indexes_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_cells_update, 4, sizeof(cl_mem), (void*)&neighbors_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_cells_update, 5, sizeof(cl_mem), (void*)&type_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_cells_update, 6, sizeof(cl_mem), (void*)&color_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_cells_update, 7, sizeof(cl_mem), (void*)&updated_cells_mem_obj));

        CL_ASSERT(clEnqueueNDRangeKernel(m_CLWrapper.GPUCommandQueue, kernel_cells_update, 1, NULL,
            &NumberOfCell, nullptr, 0, NULL, NULL));

        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, type_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(CellType), m_Types.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, color_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, updated_cells_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(int), updatedCells, 0, NULL, NULL));

        CL_ASSERT(clReleaseKernel(kernel_cells_update));
        CL_ASSERT(clReleaseMemObject(cell_per_partition_obj));
        CL_ASSERT(clReleaseMemObject(past_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(past_colors_mem_obj));
        CL_ASSERT(clReleaseMemObject(indexes_mem_obj));
        CL_ASSERT(clReleaseMemObject(neighbors_mem_obj));
        CL_ASSERT(clReleaseMemObject(type_mem_obj));
        CL_ASSERT(clReleaseMemObject(color_mem_obj));
        CL_ASSERT(clReleaseMemObject(updated_cells_mem_obj));
    }

    {
        cl_kernel kernel_neighbors_update = clCreateKernel(m_CLWrapper.GPUProgram, "update_neighbor_cancer_cells", NULL);

        cl_mem past_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem past_colors_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        cl_mem updated_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem past_medecine_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(MedecineCell), NULL, NULL);

        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(int), m_Types.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_colors_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, updated_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(int), updatedCells, 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_medecine_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(MedecineCell), m_MedecineCells.data(), 0, NULL, NULL);

        cl_mem type_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem color_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        cl_mem medecine_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(MedecineCell), NULL, NULL);

        CL_ASSERT(clSetKernelArg(kernel_neighbors_update, 0, sizeof(cl_mem), (void*)&past_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_neighbors_update, 1, sizeof(cl_mem), (void*)&past_colors_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_neighbors_update, 2, sizeof(cl_mem), (void*)&type_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_neighbors_update, 3, sizeof(cl_mem), (void*)&color_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_neighbors_update, 4, sizeof(cl_mem), (void*)&updated_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_neighbors_update, 5, sizeof(cl_mem), (void*)&past_medecine_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_neighbors_update, 6, sizeof(cl_mem), (void*)&medecine_cells_mem_obj));

        CL_ASSERT(clEnqueueNDRangeKernel(m_CLWrapper.GPUCommandQueue, kernel_neighbors_update, 1, NULL,
            &NumberOfCell, nullptr, 0, NULL, NULL));

        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, type_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(CellType), m_Types.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, color_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, medecine_cells_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(MedecineCell), m_MedecineCells.data(), 0, NULL, NULL));

        CL_ASSERT(clReleaseKernel(kernel_neighbors_update));
        CL_ASSERT(clReleaseMemObject(past_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(past_colors_mem_obj));
        CL_ASSERT(clReleaseMemObject(type_mem_obj));
        CL_ASSERT(clReleaseMemObject(color_mem_obj));
        CL_ASSERT(clReleaseMemObject(updated_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(past_medecine_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(medecine_cells_mem_obj));
    }
    delete[] updatedCells;
}

void CellArea::updateMedecineCells()
{
    {
        cl_kernel kernel_medecine_update = clCreateKernel(m_CLWrapper.GPUProgram, "update_medecine_cells", NULL);

        cl_mem past_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem past_colors_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        cl_mem past_medecine_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(MedecineCell), NULL, NULL);
        cl_mem past_medecine_colors_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);

        int numberOfCellInX = (int)NumberOfCell_X;
        int numberOfCellInY = (int)NumberOfCell_Y;
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(int), m_Types.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_colors_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_medecine_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(MedecineCell), m_MedecineCells.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_medecine_colors_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(Elysium::Vector4), m_MedecineColors.data(), 0, NULL, NULL);

        cl_mem type_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem color_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        cl_mem medecine_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(MedecineCell), NULL, NULL);
        cl_mem medecine_colors_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 0, sizeof(int), (void*)&numberOfCellInX));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 1, sizeof(int), (void*)&numberOfCellInY));;
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 2, sizeof(cl_mem), (void*)&past_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 3, sizeof(cl_mem), (void*)&past_colors_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 4, sizeof(cl_mem), (void*)&type_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 5, sizeof(cl_mem), (void*)&color_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 6, sizeof(cl_mem), (void*)&past_medecine_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 7, sizeof(cl_mem), (void*)&past_medecine_colors_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 8, sizeof(cl_mem), (void*)&medecine_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 9, sizeof(cl_mem), (void*)&medecine_colors_mem_obj));

        CL_ASSERT(clEnqueueNDRangeKernel(m_CLWrapper.GPUCommandQueue, kernel_medecine_update, 1, NULL,
            &NumberOfCell, nullptr, 0, NULL, NULL));

        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, type_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(CellType), m_Types.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, color_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, medecine_cells_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(MedecineCell), m_MedecineCells.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, medecine_colors_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(Elysium::Vector4), m_MedecineColors.data(), 0, NULL, NULL));

        CL_ASSERT(clReleaseKernel(kernel_medecine_update));
        CL_ASSERT(clReleaseMemObject(past_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(past_colors_mem_obj));
        CL_ASSERT(clReleaseMemObject(past_medecine_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(past_medecine_colors_mem_obj));
        CL_ASSERT(clReleaseMemObject(type_mem_obj));
        CL_ASSERT(clReleaseMemObject(color_mem_obj));
        CL_ASSERT(clReleaseMemObject(medecine_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(medecine_colors_mem_obj));
    }

    {
        cl_kernel kernel_medecine_update = clCreateKernel(m_CLWrapper.GPUProgram, "move_medecine_cells", NULL);

        cl_mem past_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem past_colors_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        cl_mem past_medecine_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(MedecineCell), NULL, NULL);

        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(int), m_Types.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_colors_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_CLWrapper.GPUCommandQueue, past_medecine_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(MedecineCell), m_MedecineCells.data(), 0, NULL, NULL);

        cl_mem type_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(int), NULL, NULL);
        cl_mem color_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);
        cl_mem medecine_cells_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(MedecineCell), NULL, NULL);
        cl_mem medecine_colors_mem_obj = clCreateBuffer(m_CLWrapper.GPUContext, CL_MEM_WRITE_ONLY, NumberOfCell * sizeof(Elysium::Vector4), NULL, NULL);

        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 0, sizeof(cl_mem), (void*)&past_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 1, sizeof(cl_mem), (void*)&past_colors_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 2, sizeof(cl_mem), (void*)&type_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 3, sizeof(cl_mem), (void*)&color_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 4, sizeof(cl_mem), (void*)&past_medecine_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 5, sizeof(cl_mem), (void*)&medecine_cells_mem_obj));
        CL_ASSERT(clSetKernelArg(kernel_medecine_update, 6, sizeof(cl_mem), (void*)&medecine_colors_mem_obj));

        CL_ASSERT(clEnqueueNDRangeKernel(m_CLWrapper.GPUCommandQueue, kernel_medecine_update, 1, NULL,
            &NumberOfCell, nullptr, 0, NULL, NULL));

        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, type_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(CellType), m_Types.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, color_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(Elysium::Vector4), Colors.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, medecine_cells_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(MedecineCell), m_MedecineCells.data(), 0, NULL, NULL));
        CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.GPUCommandQueue, medecine_colors_mem_obj, CL_TRUE, 0,
            NumberOfCell * sizeof(Elysium::Vector4), m_MedecineColors.data(), 0, NULL, NULL));

        CL_ASSERT(clReleaseKernel(kernel_medecine_update));
        CL_ASSERT(clReleaseMemObject(past_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(past_colors_mem_obj));
        CL_ASSERT(clReleaseMemObject(type_mem_obj));
        CL_ASSERT(clReleaseMemObject(color_mem_obj));
        CL_ASSERT(clReleaseMemObject(past_medecine_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(medecine_cells_mem_obj));
        CL_ASSERT(clReleaseMemObject(medecine_colors_mem_obj));
    }
}

void CellArea::countCells()
{
    int cellCountBuffer[3 * s_NumberOfThreads] = { 0 };
    cl_kernel kernel_cells_update = clCreateKernel(m_CLWrapper.CPUProgram, "count_cells", NULL);

    cl_mem past_cells_mem_obj = clCreateBuffer(m_CLWrapper.CPUContext, CL_MEM_READ_ONLY, NumberOfCell * sizeof(int), NULL, NULL);

    int numberOfCell = (int)NumberOfCell;
    int numberOfCellsPerPartition = (int)s_NumberOfThreads;
    clEnqueueWriteBuffer(m_CLWrapper.CPUCommandQueue, past_cells_mem_obj, CL_TRUE, 0, NumberOfCell * sizeof(int), m_Types.data(), 0, NULL, NULL);

    cl_mem count_mem_obj = clCreateBuffer(m_CLWrapper.CPUContext, CL_MEM_WRITE_ONLY, sizeof(cellCountBuffer), NULL, NULL);

    CL_ASSERT(clSetKernelArg(kernel_cells_update, 0, sizeof(cl_mem), (void*)&past_cells_mem_obj));
    CL_ASSERT(clSetKernelArg(kernel_cells_update, 1, sizeof(cl_mem), (void*)&count_mem_obj));
    CL_ASSERT(clSetKernelArg(kernel_cells_update, 2, sizeof(int), (void*)&numberOfCell));
    CL_ASSERT(clSetKernelArg(kernel_cells_update, 3, sizeof(int), (void*)&numberOfCellsPerPartition));

    CL_ASSERT(clEnqueueNDRangeKernel(m_CLWrapper.CPUCommandQueue, kernel_cells_update, 1, NULL,
        &s_NumberOfThreads, nullptr, 0, NULL, NULL));

    CL_ASSERT(clEnqueueReadBuffer(m_CLWrapper.CPUCommandQueue, count_mem_obj, CL_TRUE, 0,
        sizeof(cellCountBuffer), cellCountBuffer, 0, NULL, NULL));

    CL_ASSERT(clReleaseKernel(kernel_cells_update));
    CL_ASSERT(clReleaseMemObject(past_cells_mem_obj));
    CL_ASSERT(clReleaseMemObject(count_mem_obj));

    for (int i = 0; i < (int)s_NumberOfThreads; i++)
    {
        NumberOfCancerCells += cellCountBuffer[i * 3];
        NumberOfHealthyCells += cellCountBuffer[(i * 3) + 1];
        NumberOfMedecineCells += cellCountBuffer[(i * 3) + 2];
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