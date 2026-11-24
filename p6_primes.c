// Program 6: Parallel prime number generation using OpenMP
#include <stdio.h>
#include <omp.h>

#define N 10000

int isPrime(int n)
{
    if (n <= 1) return 0;

    int count = 0;

    for (int i = 2; i <= n; i++)
    {
        if (n % i == 0) count++;
    }
    
    return (count == 1);
}

int main()
{
    // Sequential
    int count = 0;

    double start = omp_get_wtime();
    for (int i = 2; i <= N; i++)
        count += isPrime(i);
    double seq = omp_get_wtime() - start;
    
    printf("Sequential: primes up to %d = %d, time = %f s\n\n", N, count, seq);

    // Parallel
    for (int t = 1; t <= 8; t *= 2)
    {
        count = 0;

        omp_set_num_threads(t);

        start = omp_get_wtime();
        #pragma omp parallel for reduction(+:count)
        for (int i = 2; i <= N; i++)
            count += isPrime(i);
        double time = omp_get_wtime() - start;

        double speedup = seq / time;

        printf("Threads: %d\tTime: %f\tSpeedup: %.2f\tEfficiency: %.2f\n", t, time, speedup, speedup / t);
    }

    printf("\nPrimes found (parallel) = %d\n", count);
    
    return 0;
}
