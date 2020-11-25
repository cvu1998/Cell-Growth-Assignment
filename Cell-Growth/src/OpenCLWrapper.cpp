#include "OpenCLWrapper.h"

#include <streambuf>

#define MAX_SOURCE_SIZE (0x100000)

void OpenCLWrapper::Init(const char* kernelPath)
{
    cl_uint platformCount = 0;
    cl_int ret = 0;
    CL_ASSERT(clGetPlatformIDs(0, NULL, &platformCount));
    ELY_INFO("Number of platforms found: {0}", (unsigned int)platformCount);
    m_Platforms = (cl_platform_id*)malloc(platformCount * sizeof(cl_platform_id));
    CL_ASSERT(clGetPlatformIDs(platformCount, m_Platforms, NULL));

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
    if (value)
        ELY_INFO("CPU device: {0}", value);
    else
        ELY_INFO("CPU device not found!");
    free(value);

    clGetDeviceInfo(m_GPU, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(m_GPU, CL_DEVICE_NAME, valueSize, value, NULL);
    if (value)
        ELY_INFO("GPU device: {0}", value);
    else
        ELY_INFO("GPU device not found!");
    free(value);

    size_t maxWorkGroupSize;
    clGetDeviceInfo(m_GPU, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 0, NULL, &valueSize);
    clGetDeviceInfo(m_GPU, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    ELY_INFO("GPU device work group size: {0}", maxWorkGroupSize);

    clProgram programSource = getProgramSoure(kernelPath);

    const char* programSourceStr = programSource.sourceStr.c_str();
    if (m_CPU)
    {
        // Create OpenCL contexts
        CPUContext = clCreateContext(NULL, 1, &m_CPU, NULL, NULL, &ret);
        CL_ASSERT(ret);
        // Create command queues
        CPUCommandQueue = clCreateCommandQueue(CPUContext, m_CPU, 0, &ret);
        CL_ASSERT(ret);
        // Create programs from the kernel source
        CPUProgram = clCreateProgramWithSource(CPUContext, 1,
            (const char**)&programSourceStr, (const size_t*)&programSource.sourceSize, &ret);
        CL_ASSERT(ret);
        // Build programs
        CL_ASSERT(ret = clBuildProgram(CPUProgram, 1, &m_CPU, NULL, NULL, NULL));
        if (ret != CL_SUCCESS)
        {
            size_t len;
            char* buffer;
            clGetProgramBuildInfo(CPUProgram, m_CPU, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            buffer = new char[len];
            clGetProgramBuildInfo(CPUProgram, m_CPU, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
            ELY_ERROR("Build error: {0}", buffer);
            delete[] buffer;
        }
    }
    if (m_GPU)
    {
        GPUContext = clCreateContext(NULL, 1, &m_GPU, NULL, NULL, &ret);
        CL_ASSERT(ret);
        GPUCommandQueue = clCreateCommandQueue(GPUContext, m_GPU, 0, &ret);
        CL_ASSERT(ret);
        GPUProgram = clCreateProgramWithSource(GPUContext, 1,
            (const char**)&programSourceStr, (const size_t*)&programSource.sourceSize, &ret);
        CL_ASSERT(ret);
        CL_ASSERT(clBuildProgram(GPUProgram, 1, &m_GPU, NULL, NULL, NULL));
    }
}

void OpenCLWrapper::Shutdown()
{
    CL_ASSERT(clFlush(CPUCommandQueue));
    CL_ASSERT(clFinish(CPUCommandQueue));
    CL_ASSERT(clReleaseProgram(CPUProgram));
    CL_ASSERT(clReleaseCommandQueue(CPUCommandQueue));
    CL_ASSERT(clReleaseContext(CPUContext));

    CL_ASSERT(clFlush(GPUCommandQueue));
    CL_ASSERT(clFinish(GPUCommandQueue));
    CL_ASSERT(clReleaseProgram(GPUProgram));
    CL_ASSERT(clReleaseCommandQueue(GPUCommandQueue));
    CL_ASSERT(clReleaseContext(GPUContext));

    free(m_Platforms);
}

clProgram OpenCLWrapper::getProgramSoure(const char* filepath)
{

    std::ifstream file(filepath);
    clProgram kernel;

    if (!file.is_open())
    {
        ELY_ERROR("Cannot open kernel: {0}", filepath);
        return kernel;
    }

    file.seekg(0, std::ios::end);
    kernel.sourceStr.reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    kernel.sourceStr.assign((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    file.close();
    return kernel;
}

const char* OpenCLWrapper::getCLError(int ret)
{
    switch (ret)
    {
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return nullptr;
    }
}

void OpenCLWrapper::logCLError(int ret, const char* file, int line)
{
    const char* errorMessage = getCLError(ret);
    if (errorMessage)
        ELY_ERROR("{0} in file {1}: {2}", errorMessage, file, line);
}