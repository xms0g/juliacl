#include <math.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WIDTH 1280
#define HEIGHT 720

#define CHECK_ERROR(err) if (err != CL_SUCCESS) { printf("%s\n", clErrorString(err)); release(); return 1; }

static cl_event kernelEvent;
static cl_kernel kernel;
static cl_program program;
static cl_context context;
static cl_command_queue queue;
static cl_mem outputBuffer;
static uint8_t* imageData;
static char* source;

static char* loadProgramSource(const char* filename);
static void release();
static const char* clErrorString(cl_int err);

int main(int argc, char** argv) {
    cl_int err = 0;
    cl_device_id device;
    cl_platform_id platform;
    cl_uint platformCount;
    cl_uint deviceCount;

    // Get platform and device IDs
    err = clGetPlatformIDs(1, &platform, &platformCount);
    CHECK_ERROR(err)

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &deviceCount);
    CHECK_ERROR(err)

    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err)
    // Create Command Queue
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err)

    // Create output buffer
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, WIDTH * HEIGHT * sizeof(cl_uchar4), NULL, &err);
    CHECK_ERROR(err)

    source = loadProgramSource("../src/julia.cl");
    const size_t source_size = strlen(source);

    program = clCreateProgramWithSource(context, 1, &source, &source_size, &err);
    CHECK_ERROR(err)

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate memory for the log
        char* log = malloc(log_size + 1);
        // Get the log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, (void*) log, NULL);
        // Print the log
        printf("%s\n", log);
        free(log);
        release();
        return 1;
    }

    kernel = clCreateKernel(program, "julia", &err);
    CHECK_ERROR(err)
    // Set Kernel Arguments
    const int width = WIDTH;
    const int height = HEIGHT;
    const float cRe = -0.8f;
    const float cIm = 0.156f;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &width);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &height);
    clSetKernelArg(kernel, 3, sizeof(cl_float), &cRe);
    clSetKernelArg(kernel, 4, sizeof(cl_float), &cIm);

    // Execute Kernel
    size_t maxGroupSize;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxGroupSize, NULL);

    const size_t side = (size_t)sqrt((double)maxGroupSize);
    const size_t localWorkSize[2] = {side, side};

    const size_t globalWorkSize[2] = {
        (WIDTH + localWorkSize[0] - 1) / localWorkSize[0] * localWorkSize[0],
        (HEIGHT + localWorkSize[1] - 1) / localWorkSize[1] * localWorkSize[1]
    };

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernelEvent);
    CHECK_ERROR(err)

    clWaitForEvents(1, &kernelEvent);

    // Read output buffer
    imageData = malloc(WIDTH * HEIGHT * 4 * sizeof(uint8_t));
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, WIDTH * HEIGHT * sizeof(cl_uchar4), imageData, 0, NULL, NULL);
    CHECK_ERROR(err)

    stbi_write_png("julia.png", WIDTH, HEIGHT, 4, imageData, WIDTH * 4);

    release();
    return 0;
}

static char* loadProgramSource(const char* filename) {
    FILE* kernelFile = fopen(filename, "r");

    fseek(kernelFile, 0, SEEK_END);
    const size_t source_size = ftell(kernelFile);
    fseek(kernelFile, 0, SEEK_SET);

    char* source = malloc(source_size + 1);
    fread(source, 1, source_size, kernelFile);
    source[source_size] = '\0';
    fclose(kernelFile);

    return source;
}

void release() {
    clReleaseEvent(kernelEvent);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(outputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source);
    free(imageData);
}

static const char* clErrorString(const cl_int err) {
    const char* errStr = "Unknown OpenCL error";

    switch (err) {
        case CL_DEVICE_NOT_FOUND:                         errStr = "CL_DEVICE_NOT_FOUND"; break;
        case CL_DEVICE_NOT_AVAILABLE:                     errStr = "CL_DEVICE_NOT_AVAILABLE"; break;
        case CL_COMPILER_NOT_AVAILABLE:                   errStr = "CL_COMPILER_NOT_AVAILABLE"; break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:            errStr = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
        case CL_OUT_OF_RESOURCES:                         errStr = "CL_OUT_OF_RESOURCES"; break;
        case CL_OUT_OF_HOST_MEMORY:                       errStr = "CL_OUT_OF_HOST_MEMORY"; break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:             errStr = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
        case CL_MEM_COPY_OVERLAP:                         errStr = "CL_MEM_COPY_OVERLAP"; break;
        case CL_IMAGE_FORMAT_MISMATCH:                    errStr = "CL_IMAGE_FORMAT_MISMATCH"; break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:               errStr = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
        case CL_BUILD_PROGRAM_FAILURE:                    errStr = "CL_BUILD_PROGRAM_FAILURE"; break;
        case CL_MAP_FAILURE:                              errStr = "CL_MAP_FAILURE"; break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:             errStr = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:errStr = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
        case CL_COMPILE_PROGRAM_FAILURE:                  errStr = "CL_COMPILE_PROGRAM_FAILURE"; break;
        case CL_LINKER_NOT_AVAILABLE:                     errStr = "CL_LINKER_NOT_AVAILABLE"; break;
        case CL_LINK_PROGRAM_FAILURE:                     errStr = "CL_LINK_PROGRAM_FAILURE"; break;
        case CL_DEVICE_PARTITION_FAILED:                  errStr = "CL_DEVICE_PARTITION_FAILED"; break;
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:            errStr = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;
        case CL_INVALID_VALUE:                            errStr = "CL_INVALID_VALUE"; break;
        case CL_INVALID_DEVICE_TYPE:                      errStr = "CL_INVALID_DEVICE_TYPE"; break;
        case CL_INVALID_PLATFORM:                         errStr = "CL_INVALID_PLATFORM"; break;
        case CL_INVALID_DEVICE:                           errStr = "CL_INVALID_DEVICE"; break;
        case CL_INVALID_CONTEXT:                          errStr = "CL_INVALID_CONTEXT"; break;
        case CL_INVALID_QUEUE_PROPERTIES:                 errStr = "CL_INVALID_QUEUE_PROPERTIES"; break;
        case CL_INVALID_COMMAND_QUEUE:                    errStr = "CL_INVALID_COMMAND_QUEUE"; break;
        case CL_INVALID_HOST_PTR:                         errStr = "CL_INVALID_HOST_PTR"; break;
        case CL_INVALID_MEM_OBJECT:                       errStr = "CL_INVALID_MEM_OBJECT"; break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:          errStr = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
        case CL_INVALID_IMAGE_SIZE:                       errStr = "CL_INVALID_IMAGE_SIZE"; break;
        case CL_INVALID_SAMPLER:                          errStr = "CL_INVALID_SAMPLER"; break;
        case CL_INVALID_BINARY:                           errStr = "CL_INVALID_BINARY"; break;
        case CL_INVALID_BUILD_OPTIONS:                    errStr = "CL_INVALID_BUILD_OPTIONS"; break;
        case CL_INVALID_PROGRAM:                          errStr = "CL_INVALID_PROGRAM"; break;
        case CL_INVALID_PROGRAM_EXECUTABLE:               errStr = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
        case CL_INVALID_KERNEL_NAME:                      errStr = "CL_INVALID_KERNEL_NAME"; break;
        case CL_INVALID_KERNEL_DEFINITION:                errStr = "CL_INVALID_KERNEL_DEFINITION"; break;
        case CL_INVALID_KERNEL:                           errStr = "CL_INVALID_KERNEL"; break;
        case CL_INVALID_ARG_INDEX:                        errStr = "CL_INVALID_ARG_INDEX"; break;
        case CL_INVALID_ARG_VALUE:                        errStr = "CL_INVALID_ARG_VALUE"; break;
        case CL_INVALID_ARG_SIZE:                         errStr = "CL_INVALID_ARG_SIZE"; break;
        case CL_INVALID_KERNEL_ARGS:                      errStr = "CL_INVALID_KERNEL_ARGS"; break;
        case CL_INVALID_WORK_DIMENSION:                   errStr = "CL_INVALID_WORK_DIMENSION"; break;
        case CL_INVALID_WORK_GROUP_SIZE:                  errStr = "CL_INVALID_WORK_GROUP_SIZE"; break;
        case CL_INVALID_WORK_ITEM_SIZE:                   errStr = "CL_INVALID_WORK_ITEM_SIZE"; break;
        case CL_INVALID_GLOBAL_OFFSET:                    errStr = "CL_INVALID_GLOBAL_OFFSET"; break;
        case CL_INVALID_EVENT_WAIT_LIST:                  errStr = "CL_INVALID_EVENT_WAIT_LIST"; break;
        case CL_INVALID_EVENT:                            errStr = "CL_INVALID_EVENT"; break;
        case CL_INVALID_OPERATION:                        errStr = "CL_INVALID_OPERATION"; break;
        case CL_INVALID_GL_OBJECT:                        errStr = "CL_INVALID_GL_OBJECT"; break;
        case CL_INVALID_BUFFER_SIZE:                      errStr = "CL_INVALID_BUFFER_SIZE"; break;
        case CL_INVALID_MIP_LEVEL:                        errStr = "CL_INVALID_MIP_LEVEL"; break;
        case CL_INVALID_GLOBAL_WORK_SIZE:                 errStr = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
        case CL_INVALID_PROPERTY:                         errStr = "CL_INVALID_PROPERTY"; break;
        case CL_INVALID_IMAGE_DESCRIPTOR:                 errStr = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
        case CL_INVALID_COMPILER_OPTIONS:                 errStr = "CL_INVALID_COMPILER_OPTIONS"; break;
        case CL_INVALID_LINKER_OPTIONS:                   errStr = "CL_INVALID_LINKER_OPTIONS"; break;
        case CL_INVALID_DEVICE_PARTITION_COUNT:           errStr = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;
        default:                                          break;
    }

    return errStr;
}
