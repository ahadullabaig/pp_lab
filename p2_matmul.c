// Program 2: Parallel matrix multiplication using OpenMP
#include <stdio.h>
#include <omp.h>

#define N 500

int A[N][N], B[N][N], C[N][N];

void multiply()
{
    #pragma omp parallel for          // rows of C are split among threads
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;

            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];

            C[i][j] = sum;
        }
    }
}

int main()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) 
        {
            A[i][j] = 1;
            B[i][j] = 2;
        }
    }

    for (int t = 1; t <= 8; t *= 2) 
    {
        omp_set_num_threads(t);

        double start = omp_get_wtime();
        multiply();
        double time = omp_get_wtime() - start;

        printf("Threads = %d   Time = %f s\n", t, time);
    }

    printf("C[0][0] = %d (expected %d)\n", C[0][0], 2 * N);

    return 0;
}
