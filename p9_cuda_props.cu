// Program 9: GPU properties + kernel launch with different blocks/threads
#include <stdio.h>

__global__ void hello()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;   // global thread id
    printf("  Block %d, Thread %d, Global ID %d\n", blockIdx.x, threadIdx.x, id);
}

int main()
{
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("GPU Name             : %s\n", p.name);
    printf("Compute Capability   : %d.%d\n", p.major, p.minor);
    printf("Global Memory        : %zu MB\n", p.totalGlobalMem / (1024 * 1024));
    printf("Multiprocessors (SM) : %d\n", p.multiProcessorCount);
    printf("Warp Size            : %d\n", p.warpSize);
    printf("Max Threads/Block    : %d\n", p.maxThreadsPerBlock);
    printf("Max Block Dimensions : %d x %d x %d\n",
           p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("Max Grid Dimensions  : %d x %d x %d\n",
           p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

    printf("\n<<<1 block, 4 threads>>>\n");
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();          // wait for GPU so its printf output appears

    printf("\n<<<2 blocks, 4 threads>>>\n");
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();

    printf("\n<<<4 blocks, 2 threads>>>\n");
    hello<<<4, 2>>>();
    cudaDeviceSynchronize();

    return 0;
}
