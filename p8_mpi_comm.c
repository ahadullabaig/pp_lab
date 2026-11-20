// Program 8: Point-to-point (MPI_Send/MPI_Recv) and broadcast (MPI_Bcast)
#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    char msg[20];
    int arr[5] = {0, 0, 0, 0, 0};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Point-to-point: process 0 sends a string to process 1
    if (rank == 0) {
        strcpy(msg, "Hello from P0");
        MPI_Send(msg, 20, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent: %s\n", msg);
    } else if (rank == 1) {
        MPI_Recv(msg, 20, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received: %s\n", msg);
    }

    // 2. Broadcast: process 0 sends an array to ALL processes
    if (rank == 0)
        for (int i = 0; i < 5; i++) arr[i] = (i + 1) * 10;

    MPI_Bcast(arr, 5, MPI_INT, 0, MPI_COMM_WORLD);   // called by every process

    printf("Process %d after Bcast: %d %d %d %d %d\n",
           rank, arr[0], arr[1], arr[2], arr[3], arr[4]);

    MPI_Finalize();
    return 0;
}
