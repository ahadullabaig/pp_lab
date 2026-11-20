// Program 1: Addition of two arrays using CUDA
#include <stdio.h>

#define N 10

__global__ void add(int *a, int *b, int *c)
{
    int i = threadIdx.x;              // each thread adds one element
    if (i < N)
        c[i] = a[i] + b[i];
}

int main()
{
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;             // d_ = device (GPU) memory
    int size = N * sizeof(int);

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    add<<<1, N>>>(d_a, d_b, d_c);     // 1 block, N threads

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
