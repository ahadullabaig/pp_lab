// Program 3: Parallel merge sort using OpenMP tasks
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 2000000

int orig[N], a[N], tmp[N];

void merge(int left, int mid, int right)
{
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right)
    {
        if (a[i] <= a[j])
        {
            tmp[k] = a[i];
            i++;
        }
        else
        {
            tmp[k] = a[j];
            j++;
        }

        k++;
    }
    
    while (i <= mid)
    {
        tmp[k] = a[i];
        i++; k++;
    }
    
    while (j <= right)
    {
        tmp[k] = a[j];
        j++; k++;
    }
    
    for (i = left; i <= right; i++) a[i] = tmp[i];
}

void mergeSort(int left, int right)
{
    if (left >= right) return;

    int mid = (left + right) / 2;

    #pragma omp task      // each half becomes a parallel task
    mergeSort(left, mid);

    #pragma omp task
    mergeSort(mid + 1, right);

    #pragma omp taskwait  // wait for both halves
    merge(left, mid, right);
}

void reset()                             // give every run the same unsorted data
{
    for (int i = 0; i < N; i++) a[i] = orig[i];
}

int main()
{
    for (int i = 0; i < N; i++) orig[i] = rand();

    // Sequential: outside a parallel region there is only 1 thread
    reset();
    double start = omp_get_wtime();
    mergeSort(0, N - 1);
    double seq = omp_get_wtime() - start;
    printf("Sequential time = %f s\n", seq);

    // Parallel
    reset();
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single               // one thread starts the recursion
        mergeSort(0, N - 1);
    }
    double par = omp_get_wtime() - start;
    
    printf("Parallel time   = %f s\n", par);
    printf("Speedup         = %.2f\n", seq / par);

    return 0;
}
