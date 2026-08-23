#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do {\
        cudaError_t err_ = (call);\
        if (err_ != cudaSuccess){\
            fprintf(stderr, "CUDA error %s at %s:%d: %s", cudaGetErrorName(err_), __FILE__, __LINE__, cudaGetErrorString(err_));\
            exit(1);\
        }\
    } while(0)
#endif

#ifndef CUDA_CHECK_KERNEL
#define CUDA_CHECK_KERNEL()\
    do{\
        CUDA_CHECK(cudaGetLastError());\
        CUDA_CHECK(cudaDeviceSynchronize());\
    }while(0)
#endif

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

__global__ void saxpy(const float *A, float *B, const int n){
    for (int idx=blockIdx.x * blockDim.x + threadIdx.x; idx<n; idx+=blockDim.x * gridDim.x) B[idx] += 2.0f * A[idx];
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx<n){
    //     B[idx] += 2.0f * A[idx];
    // }
}

int main(int argc, char **argv){
    assert(argc==2);
    int n = atoi(argv[1]);
    if (n==0){
        printf("SUM=0\n");
        exit(0);
    }
    float *x, *y;
    
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(&x, bytes));
    CUDA_CHECK(cudaMallocManaged(&y, bytes));
    for (int i=0; i<n; ++i){
        x[i] = ((i % 2048) - 1024) * 0.5f;
        y[i] = (i % 1024) - 512;
    }
    GpuTimer gt = GpuTimer();
    
    int block = 1024;
    int grid = (n+block-1)/block;
    gt.start();
    saxpy<<<64, 256>>>(x, y, n);
    float t = gt.stop_ms();

    CUDA_CHECK_KERNEL();

    double s = 0;
    for (int i=0; i<n; ++i){
        s += y[i];
    }
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    printf("SUM=%.0f, n=%d, time=%f\n", s, n, t);
    return 0;
}