#pragma once

#include <array>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Elysium/Math.h>
#include <Elysium/Timestep.h>
#include <Elysium/Random.h>

class InjectMedecineTask;

class CellArea
{
    friend class InjectMedecineTask;

public:
#ifdef _DEBUG
    static constexpr size_t NumberOfCell_X = 200;
    static constexpr size_t NumberOfCell_Y = 200;
#else
    static constexpr size_t NumberOfCell_X = 1024;
    static constexpr size_t NumberOfCell_Y = 768;
#endif

    static constexpr size_t NumberOfCells = NumberOfCell_X * NumberOfCell_Y;

private:
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
        CellType PreviousType = CellType::NONE;
        int offset = 0;
    };

    static constexpr size_t s_NumberOfThreads = 4;
    static constexpr size_t s_NumberOfCellsPerPartition = NumberOfCells / s_NumberOfThreads;
    static constexpr size_t s_NumberOfCellsPerPartition_Y = NumberOfCell_Y / s_NumberOfThreads;

    //static constexpr Elysium::Vector4 s_ColorGreen = { 0.0f, 1.0f, 0.0f, 1.0f };
    //static constexpr Elysium::Vector4 s_ColorRed = { 0.75f, 0.0f, 0.0f, 1.0f };
    //static constexpr Elysium::Vector4 s_ColorYellow = { 1.0f, 1.0f, 0.0f, 1.0f };

    static constexpr int s_NeighborIndexes[8] = { -(int)NumberOfCell_X - 1, -(int)NumberOfCell_X , -(int)NumberOfCell_X + 1,
        -1, 1,
        (int)NumberOfCell_X - 1, (int)NumberOfCell_X, (int)NumberOfCell_X + 1 };

    float m_CellSize = 2.5f;
    float m_CurrentTime = 0.0f;

    std::unordered_set<size_t>m_InputBuffer;
    std::unordered_map<size_t, MedecineCell> m_MedecineCells;

    std::array<CellType, NumberOfCells> m_Types = { CellType::HEALTHY };
    std::array<std::vector<int>, s_NumberOfCellsPerPartition> m_Neighbors;

public:
    std::array<Elysium::Vector2, NumberOfCells> Positions;
    std::array<Elysium::Vector4, NumberOfCells> Colors;

    unsigned int NumberOfCancerCells = 0;
    unsigned int NumberOfHealthyCells = 0;
    unsigned int NumberOfMedecineCells = 0;

private:
    void setNeighbor(int index);

    void updateCellsInPartition(CellType* partition, PartitionStats& stats,
        std::unordered_map<size_t, MedecineCell>& medecineMap,
        size_t min);

    void moveMedecineCells(size_t cellIndex,
        std::unordered_map<size_t, MedecineCell>& medecineMap,
        std::unordered_set<size_t>& updatedMedecine);

public:
    CellArea(Elysium::Vector2 offset);

    void onUpdate(Elysium::Timestep ts);
    size_t getIndex(const Elysium::Vector2& position);
    void injectMedecine(const Elysium::Vector2& position);
};

class InjectMedecineTask
{
private:
    std::unordered_map<size_t, CellArea::MedecineCell>* m_MedecineCells;
    std::array<std::vector<int>, CellArea::s_NumberOfCellsPerPartition>* m_Neighbors;
    std::array<CellArea::CellType, CellArea::NumberOfCells>* m_Types;
    std::mutex* m_Lock;

    size_t m_Index = 0;

public:
    InjectMedecineTask(std::unordered_map<size_t, CellArea::MedecineCell>* medecineMap,
        std::array<std::vector<int>, CellArea::s_NumberOfCellsPerPartition>* neighbors,
        std::array<CellArea::CellType, CellArea::NumberOfCells>* types,
        std::mutex* lock,
        size_t index) :
        m_MedecineCells(medecineMap),
        m_Neighbors(neighbors),
        m_Types(types),
        m_Lock(lock),
        m_Index(index)
    {
    }

    void operator()() const
    {
        int counter = 0;
        int numberOfCells = Random::Integer(1, 8);
        for (int j : m_Neighbors->operator[](m_Index% CellArea::s_NumberOfCellsPerPartition))
        {
            counter++;
            if (counter >= numberOfCells)
                break;

            size_t index = j + ((m_Index + 1) / CellArea::s_NumberOfCellsPerPartition) * CellArea::s_NumberOfCellsPerPartition;
            if (m_MedecineCells->find(index) == m_MedecineCells->end())
            {
                const std::lock_guard<std::mutex>(*m_Lock);
                m_MedecineCells->insert({ index, { m_Types->operator[](index), (int)index - (int)m_Index } });
                m_Types->operator[](index) = CellArea::CellType::MEDECINE;
            }
        }
    }
};