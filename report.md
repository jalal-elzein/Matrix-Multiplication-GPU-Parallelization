Lebanese American University - School of Arts & Sciences  
CSC447 - Parallel Programming for Multicore and Cluster Systems  
Instructor: Dr. Hamdan Abdellatef  
Jalal El Zein - 202105966  
Due : 19 / 05 / 2023  

# Table of Contents

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
The code for all the implementations can be found [here]()

# Performance & Evaluation
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
## 3. Speedup

## 4. Efficiency

# IV - Comparison

# V - Discussion & Conclusion

