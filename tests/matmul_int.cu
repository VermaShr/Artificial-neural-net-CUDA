#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>

#define NUM_THREADS 1024
#define EPSILON 0.0001
//#define EPSILON 0.00001 --> error is too small

// KERNEL: x*A = B
__global__ void MatMul(int* x, int* A, int* B, int height, int width)
{
    // index into flattened weights matrix
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // index into the input vector
    int row = i / width;

    // index into the output vector
    int col = i % width;

    // zero out resultant vector B
    if (i < width) B[i] = 0.0;

    __syncthreads();

    if ((i < height * width) && (row < height))
    {
        // TODO: atomicAdd to local, shared output vectors --> atomicAdd to global
        atomicAdd(&B[col], x[row] * A[i]);
    }
}

// HOST
int main(int argc, char** argv)
{
    // Variables
    int *h_x, *h_A, *h_B, *d_x, *d_A, *d_B;
    int height = 256;
    int width = 100;

    // Allocate vectors and matrices in host memory and device memory
    h_x = (int*)malloc(height*sizeof(int));
    h_A = (int*)malloc(height*width*sizeof(int));
    h_B = (int*)malloc(width*sizeof(int));
    cudaMalloc((void**)&d_x, height*sizeof(int));
    cudaMalloc((void**)&d_A, height*width*sizeof(int));
    cudaMalloc((void**)&d_B, width*sizeof(int));

    // Initialize input vector x
    for (int i = 0; i < height; ++i)
    {
        h_x[i] = rand() / (int)RAND_MAX;
    }

    // Initialize input matrix A
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            h_A[i*width + j] = rand() / (int)RAND_MAX;
        }
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_x, h_x, height*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, height*width*sizeof(int), cudaMemcpyHostToDevice);

    // FILL IN KERNEL SETUP AND INVOCATION

    int blocks = (height*width) / NUM_THREADS;
    if ((height*width) % NUM_THREADS != 0) blocks++;

    MatMul <<< blocks, NUM_THREADS  >>> (d_x, d_A, d_B, height, width);

    cudaDeviceSynchronize();

    // Copy result from device memory to host memory
    cudaMemcpy(h_B, d_B, width*sizeof(int), cudaMemcpyDeviceToHost);

    bool correct = true;

    // Calculate solution on the host and compare
    int* result = (int*)malloc(width*sizeof(int));

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // zero out result elements
            if (i == 0) result[j] = 0.0;

            result[j] += h_x[i] * h_A[i*width + j];
        }
    }

    for (int j = 0; j < width; j++)
    {
        //if (fabs(h_B[j] - result[j]) > EPSILON)
        if (h_B[j] != result[j])
        {
            printf("ERROR: expected h_B[%i] = %f but received %f\n", j, result[j], h_B[j]);
            correct = false;
            break;
        }
    }

    if (correct) printf("---PASSED---\n");

    // Free host and device memory
    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B);
    free(h_x); free(h_A); free(h_B); free(result);
}
