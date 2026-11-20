# Parallel Programming Lab (23AIL57)

| # | File | Topic |
|---|---|---|
| 1 | `p1_array_add.cu` | CUDA: addition of two arrays |
| 2 | `p2_matmul.c` | OpenMP: matrix multiplication, time vs. thread count |
| 3 | `p3_mergesort.c` | OpenMP: parallel merge sort, speedup vs. sequential |
| 4 | `p4_threadprivate.c` | OpenMP: `threadprivate` and `copyin` |
| 5 | `p5_schedule.c` | OpenMP: `static` / `dynamic` / `guided` scheduling |
| 6 | `p6_primes.c` | OpenMP: prime generation, time / speedup / efficiency |
| 7 | `p7_mpi_sum.c` | MPI: array sum with `MPI_Scatter` + `MPI_Reduce` |
| 8 | `p8_mpi_comm.c` | MPI: `MPI_Send` / `MPI_Recv` + `MPI_Bcast` |
| 9 | `p9_cuda_props.cu` | CUDA: GPU properties + varying blocks/threads |
| 10 | `p10_vec_add_time.cu` | CUDA: vector addition, CPU vs. GPU time |
| 11 | `p11_linreg.py` | PyTorch: linear regression (gradients + SGD) |
| 12 | `p12_ffnn.py` | PyTorch: feedforward neural network on random data |

All 12 programs were compiled and run (GCC 16 / OpenMP, MPICH, CUDA 13.5 on an RTX 3050, PyTorch 2.15 on CPU). The real output is in `sample_outputs.txt`. Timings will differ on other machines, and the first CUDA run after compiling can be slower because the GPU starts cold.

### Setup on this machine
Only `gcc` (OpenMP) is installed system-wide. MPI, CUDA and PyTorch come from the project venv in `.venv/`, which git ignores.

One-time setup:
```sh
python3 -m venv .venv
.venv/bin/pip install mpich nvidia-cuda-nvcc nvidia-cuda-runtime numpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Each new terminal, from inside `pp_lab/`:
```sh
source .venv/bin/activate                                                     # gives mpicc, mpirun, python with torch
export PATH=$VIRTUAL_ENV/lib/python3.14/site-packages/nvidia/cu13/bin:$PATH   # gives nvcc
```
On the lab machines `mpicc`, `nvcc` and `torch` are usually preinstalled, so the commands below work as-is there.

## How to compile and run

**OpenMP (2–6)**
```sh
gcc -fopenmp p2_matmul.c -o p2
./p2
```

**MPI (7–8)**: program 8 needs at least 2 processes
```sh
mpicc p7_mpi_sum.c -o p7
mpirun -np 4 ./p7
```

**CUDA (1, 9, 10)**
```sh
nvcc p1_array_add.cu -o p1
./p1
```

**PyTorch (11–12)**
```sh
python3 p11_linreg.py
```

## The patterns to memorize

**OpenMP timing loop (programs 2, 3, 6)**
```c
for (int t = 1; t <= 8; t *= 2) {
    omp_set_num_threads(t);
    double start = omp_get_wtime();
    /* parallel work */
    double time = omp_get_wtime() - start;
    printf(... seq / time ...);              // speedup; efficiency = speedup / t
}
```

**MPI skeleton (programs 7, 8)**
```c
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
/* work */
MPI_Finalize();
```

**CUDA skeleton (programs 1, 9, 10)**
```
cudaMalloc  ->  cudaMemcpy (HostToDevice)  ->  kernel<<<blocks, threads>>>  ->  cudaMemcpy (DeviceToHost)  ->  cudaFree
```
Global thread id: `blockIdx.x * blockDim.x + threadIdx.x`

**PyTorch training loop (programs 11, 12)**
```python
pred = model(x)
loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
