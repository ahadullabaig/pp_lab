// Program 10: Vector addition - CPU vs GPU execution time
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000

__global__ void add(float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread id
    if (i < N)
        c[i] = a[i] + b[i];
}

int main()
{
    int size = N * sizeof(float);

    float *a = (float*)malloc(size);
    float *b = (float*)malloc(size);
    float *c = (float*)malloc(size);      // CPU result
    float *g = (float*)malloc(size);      // GPU result
    
    float *d_a, *d_b, *d_c;

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // CPU
    clock_t start = clock();
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
    double cpu = (double)(clock() - start) / CLOCKS_PER_SEC;

    // GPU
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = N / threads + 1;  // enough blocks to cover N

    start = clock();
    add<<<blocks, threads>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();                   // wait for GPU to finish
    double gpu = (double)(clock() - start) / CLOCKS_PER_SEC;

    cudaMemcpy(g, d_c, size, cudaMemcpyDeviceToHost);

    printf("Results:\n");
    printf("CPU time = %f s\n", cpu);
    printf("GPU time = %f s\n", gpu);
    printf("Speedup  = %.2f\n", cpu / gpu);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a); free(b); free(c); free(g);
    return 0;
}
