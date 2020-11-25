#pragma once

#include <unordered_set>

#include "OpenCLWrapper.h"

class CellArea
{
public:
#ifdef _DEBUG
    static constexpr size_t NumberOfCell_X = 200;
    static constexpr size_t NumberOfCell_Y = 200;
#else
    static constexpr size_t NumberOfCell_X = 400;
    static constexpr size_t NumberOfCell_Y = 400;
#endif

    static constexpr size_t NumberOfCell = NumberOfCell_X * NumberOfCell_Y;

private:
    OpenCLWrapper m_CLWrapper;

    enum class CellType
    {
        NONE = -1,
        CANCER = 0,
        HEALTHY = 1,
        MEDECINE = 2
    };

    struct PartitionStats
    {
        unsigned int NumberOfCancerCells = 0;
        unsigned int NumberOfHealthyCells = 0;
        unsigned int NumberOfMedecineCells = 0;
    };

    struct MedecineCell
    {
        int Valid = 0;
        CellType PreviousType = CellType::NONE;
        int Offset = 0;
    };

    static constexpr size_t s_NumberOfThreads = 4;
    static constexpr size_t s_NumberOfCellsPerPartition = NumberOfCell / s_NumberOfThreads;
    static constexpr size_t s_NumberOfCellsPerPartition_Y = NumberOfCell_Y / s_NumberOfThreads;

    static constexpr Elysium::Vector4 s_ColorGreen = { 0.0f, 1.0f, 0.0f, 1.0f };
    static constexpr Elysium::Vector4 s_ColorRed = { 0.75f, 0.0f, 0.0f, 1.0f };
    static constexpr Elysium::Vector4 s_ColorYellow = { 1.0f, 1.0f, 0.0f, 1.0f };

    static constexpr int s_NeighborIndexes[8] = { -(int)NumberOfCell_X - 1, -(int)NumberOfCell_X , -(int)NumberOfCell_X + 1,
        -1, 1,
        (int)NumberOfCell_X - 1, (int)NumberOfCell_X, (int)NumberOfCell_X + 1 };

    float m_CellSize = 2.5f;
    float m_CurrentTime = 0.0f;

    std::unordered_set<size_t>m_InputBuffer;

    std::array<MedecineCell, NumberOfCell> m_MedecineCells;
    std::array<Elysium::Vector4, NumberOfCell> m_MedecineColors;
    std::array<CellType, NumberOfCell> m_Types = { CellType::HEALTHY };

    std::array<int, s_NumberOfCellsPerPartition> m_Indexes = { 0 };
    std::vector<int> m_Neighbors;

public:
    std::array<Elysium::Vector2, NumberOfCell> Positions;
    std::array<Elysium::Vector4, NumberOfCell> Colors;

    unsigned int NumberOfCancerCells = 0;
    unsigned int NumberOfHealthyCells = 0;
    unsigned int NumberOfMedecineCells = 0;

private:
    void setNeighbor(int index);

    void updateHealthyCells();
    void updateCancerCells();
    void updateMedecineCells();

public:
    CellArea(Elysium::Vector2 offset);
    ~CellArea();

    void onUpdate(Elysium::Timestep ts);
    size_t getIndex(const Elysium::Vector2& position);
    void injectMedecine(const Elysium::Vector2& position);
};