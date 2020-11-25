#pragma once

#include <CL/cl.h>

#include <Elysium.h>

struct clProgram
{
    std::string sourceStr;
    size_t sourceSize = 0;
};

class OpenCLWrapper
{
private:
    cl_platform_id* m_Platforms = nullptr;
    cl_device_id m_CPU;
    cl_device_id m_GPU;

private:
    clProgram getProgramSoure(const char* filepath);

    static const char* getCLError(int ret);

public:
    cl_context CPUContext;
    cl_context GPUContext;
    cl_command_queue CPUCommandQueue;
    cl_command_queue GPUCommandQueue;
    cl_program CPUProgram;
    cl_program GPUProgram;

public:
    void Init(const char* kernelPath);
    void Shutdown();

    static void logCLError(int ret, const char* file, int line);
};

#define CL_ASSERT(x) OpenCLWrapper::logCLError(x, __FILE__, __LINE__) 