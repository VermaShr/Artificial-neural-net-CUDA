#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>

#define NUM_THREADS 1024
//#define EPSILON 0.0001
#define EPSILON 0.00001 //--> error is too small

#define height 256
#define width 10

#define UPPER 0.01
#define LOWER -0.01

// KERNEL: x*A = B
__global__ void MatMul(float* x, float* A, float* B)
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
        __syncthreads();

        if (i < width && B[i] < 0.0) B[i] = 0.0;
    }
}

// HOST
int main(int argc, char** argv)
{
    // Variables
    float *h_x, *h_A, *h_B, *d_x, *d_A, *d_B;

    // Allocate vectors and matrices in host memory and device memory
    h_x = (float*)malloc(height*sizeof(float));
    h_A = (float*)malloc(height*width*sizeof(float));
    h_B = (float*)malloc(width*sizeof(float));
    cudaMalloc((void**)&d_x, height*sizeof(float));
    cudaMalloc((void**)&d_A, height*width*sizeof(float));
    cudaMalloc((void**)&d_B, width*sizeof(float));

    // Initialize input vector x
    for (int i = 0; i < height; ++i)
    {
        h_x[i] = rand() / (float)RAND_MAX - 0.5;
    }

    // Initialize input matrix A
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // initialize weights matrix values to be between LOWER and UPPER
            h_A[i*width + j] = (rand() / (float)RAND_MAX)*(UPPER - LOWER) + LOWER;
        }
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_x, h_x, height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, height*width*sizeof(float), cudaMemcpyHostToDevice);

    // FILL IN KERNEL SETUP AND INVOCATION

    int blocks = (height*width) / NUM_THREADS;
    if ((height*width) % NUM_THREADS != 0) blocks++;

    MatMul <<< blocks, NUM_THREADS  >>> (d_x, d_A, d_B);

    cudaDeviceSynchronize();

    // Copy result from device memory to host memory
    cudaMemcpy(h_B, d_B, width*sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;

    // Calculate solution on the host and compare
    float* result = (float*)malloc(width*sizeof(float));

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
        if (result[j] < 0.0)
        {
            printf("--old result[%i]: %f--", j, result[j]);
            result[j] = 0.0;
        }

        if (fabs(h_B[j] - result[j]) > EPSILON)
        {
            printf("ERROR: expected h_B[%i] = %f but received %f\n", j, result[j], h_B[j]);
            correct = false;
            //break;
        }
        else
        {
            printf("result[j]: %f\th_B[j]: %f\n", result[j], h_B[j]);
        }        
    }

    if (correct) printf("---PASSED---\n");

    // Free host and device memory
    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B);
    free(h_x); free(h_A); free(h_B); free(result);
}
