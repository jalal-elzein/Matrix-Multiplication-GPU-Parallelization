Lebanese American University - School of Arts & Sciences  
CSC447 - Parallel Programming for Multicore and Cluster Systems  
Instructor: Dr. Hamdan Abdellatef  
Jalal El Zein - 202105966  
Due : 19 / 05 / 2023  

# Table of Contents
- [Table of Contents](#table-of-contents)
- [I - Introduction](#i---introduction)
  - [1. What is a Matrix?](#1-what-is-a-matrix)
  - [2. What is Matrix Multiplication?](#2-what-is-matrix-multiplication)
    - [a. When Can We Multiply Matrices?](#a-when-can-we-multiply-matrices)
    - [b. Product of Matrix Multiplication](#b-product-of-matrix-multiplication)
  - [3. Why Parallelize Matrix Multiplication](#3-why-parallelize-matrix-multiplication)
- [II - Parallelization](#ii---parallelization)
  - [1. Sequential Implementation](#1-sequential-implementation)
  - [2. General Strategy](#2-general-strategy)
  - [3. Standard CUDA Implementation](#3-standard-cuda-implementation)
  - [4. Tiled CUDA Implementation](#4-tiled-cuda-implementation)
- [III - Code](#iii---code)
- [IV - Performance \& Evaluation](#iv---performance--evaluation)
  - [1. Environment](#1-environment)
  - [2. Runtime](#2-runtime)
    - [Serial Time](#serial-time)
    - [CUDA](#cuda)
    - [Tiled CUDA](#tiled-cuda)
  - [3. Speedup](#3-speedup)
- [V - Comparison](#v---comparison)
- [VI - Discussion \& Conclusion](#vi---discussion--conclusion)


# I - Introduction

## 1. What is a Matrix?
Matrices are two-dimensional arrays of numbers, usually organized in *rows* and *columns*.  

They are very important and have applications in Computer Graphics, Image Processing, Machine Learning, Physics et cetera.

Example:
$$X = \begin{bmatrix}1 & 2 & 3\\
3 & 4 & 5\\
6 & 7 & 8
\end{bmatrix}$$

## 2. What is Matrix Multiplication?
Multiplication is one of the most important operations to be performed on a matrix.  

### a. When Can We Multiply Matrices?
Matrix Multiplication is only possible when the number of rows $r$ of the first matrix is equal to the number of columns $c$ in the other matrix.

### b. Product of Matrix Multiplication
Let $A$ be an $n × k$ matrix, and $B$ be an $k × m$ matrix.  
The product $A × B$ is an $n × m$ matrix, such that:  
The $(i, j)^{th}$  entry of $A×B$ is $[c ij]$ where $c_{ij} = (a_{i1}×b_{1j}) + (a_{i2}×b_{2j}) + …$

## 3. Why Parallelize Matrix Multiplication
We observe that matrix multiplication is often used in computationally intensive tasks, that also often deal with massive data.  

Additionally, the best sequential implementation of Matrix Multiplication runs in $O(n^3)$ asymptotic time complexity, which scales horribly with large data.

Thus, we need to take advantage of parallel techniques to reduce the time needed to perform these operations.

# II - Parallelization
## 1. Sequential Implementation
The sequential implementation of matrix multiplication will be used as a benchmark for the evaluation of the parallel implementations discussed later.

The sequential implementation used here will be the standard implementation in $O(n^3)$ asymptotic time complexity.

Here is the general pseudo-code followed:
```
for every row in C do
    for every column in C do
        for every row in A do
            entry C += entry A * entry B
```

## 2. General Strategy 
First, I compressed the matrices into 1-dimensional arrays for optimization. But more generally, I mostly focused on Thread-Level Parallelism and relied on the computing power of the GPU!

I used Thread-Level Parallelism by implementing the solution in such a way that each element in the resulting matrix is computed by a thread that handles information from the two input matrices.

## 3. Standard CUDA Implementation
For this implementation, I followed closely the strategy described in point (2) of this section. I used standard CUDA functionality:
- Memory allocation with `cudaMalloc()`, `cudaMallocManaged()` and `cudaFree()`

## 4. Tiled CUDA Implementation
For this implementation, I try to load in the important parts of the program data into shared memory for faster memory access and lower overhead, as shared memory is faster than global device memory and unified memory.

The approach is to divide the matrices into strips that will be assigned to thread-blocks, inside which each thread will handle a single row and column to compute a single value in the output matrix.

# III - Code
The code for all the implementations can be found [here](https://github.com/jalal-elzein/Matrix-Multiplication-GPU-Parallelization)

# IV - Performance & Evaluation
## 1. Environment
All code was ran on Google machines using [Google Colab](https://colab.research.google.com) 

The code ran with a T4 GPU runtime

## 2. Runtime
### Serial Time
Run # | Time (ms)
---- | ----
1 | 19644 
2 | 18703 
3 | 19494 
4 | 18823 
5 | 18868 
6 | 19472 
7 | 19940 
8 | 19009 
9 | 20172 
10 | 20132 
Average | 18425.7
### CUDA
Run # | Time (ms)
---- | ----
1 | 12.2305
2 | 12.624 
3 | 11.9378 
4 | 12.4621 
5 | 11.9173 
6 | 12.2007 
7 | 12.6774     
8 | 12.1572 
9 | 11.8863 
10 | 11.9902 
Average | 12.20835
### Tiled CUDA
Run # | Time (ms)
---- | ----
1 | 9.55638
2 | 9.28685
3 | 9.5935 
4 | 9.28096 
5 | 9.48227 
6 | 9.6695 
7 | 9.68077 
8 | 9.22266 
9 | 9.40848 
10 | 9.53949 
Average | 9.472086

![](3way.png)

## 3. Speedup
Speedup Factor $S(p) = \frac{t_s}{t_p}$  

For the regular CUDA implementation, $S(p) = \frac{18425.7}{12.20835} = 1,509.27$  

For the Tiled CUDA implementation, $S(p) = \frac{18425.7}{9.472086} = 1,945.263$  

# V - Comparison
The tiled version of the matrix multiplication algorithm demonstrated improved performance compared to the non-tiled version. By dividing the matrices into smaller tiles and performing computations on these tiles, the tiled version optimized memory access patterns and reduced memory latency. This optimization strategy allowed for better data reuse, as each thread in the block accessed a small portion of the matrices repeatedly, reducing the need to fetch data from global memory. Additionally, the use of shared memory in the tiled version further improved performance by providing a fast on-chip memory space that could be accessed by all threads in a block. By leveraging shared memory, the tiled version minimized memory access conflicts and improved overall memory throughput. As a result, the tiled version achieved a higher speedup factor and demonstrated better scalability, making it a more efficient and effective approach for matrix multiplication on CUDA-enabled GPUs.
![](tiledvsno.png)

# VI - Discussion & Conclusion
In this assignment, we implemented a matrix multiplication algorithm using CUDA and compared its performance with a serial implementation. We also introduced a tiled version of the CUDA implementation to further optimize the execution.

Our benchmarking results showed significant speedup factors for both the CUDA implementation and the tiled version compared to the serial implementation. The CUDA implementation achieved a speedup factor of 1500, while the tiled version achieved a speedup factor of 1900. These results indicate the effectiveness of parallelization using CUDA in accelerating matrix multiplication.

In conclusion, the CUDA implementation and the tiled version demonstrated significant speedup factors compared to the serial implementation, showcasing the power of parallel computing for matrix multiplication. These results highlight the potential for GPU acceleration in compute-intensive applications and underline the importance of optimizing memory access patterns and utilizing shared memory for improved performance. The assignment provided valuable hands-on experience with parallel programming using CUDA and showcased the performance gains achievable through parallelization.
