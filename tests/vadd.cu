#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>

// KERNEL
__global__ void Vadd(float* A, float* B, float* C, int N)
{
/*    int tx = threadIdx.x;
    int i = 4*(blockIdx.x * blockDim.x) + tx;
    int count = 0;

    while (i < N && count < 4)
    {
        C[i] = A[i] + B[i];
        i += blockDim.x;
        count ++;
    }
*/
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

// HOST
int main(int argc, char** argv)
{
    // Variables
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    int N = 5000000; // Number of floats per vector
    size_t size = N * sizeof(float);
    
    // Allocate vectors in host memory and device memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // Initialize input vectors
    for (int i = 0; i < N; ++i){
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    
    // FILL IN KERNEL SETUP AND INVOCATION

//    int blocks = N / (4*threadsPerBlock);
//    if (N % (4*threadsPerBlock) != 0) blocks++;

    int blocks = N / threadsPerBlock;
    if (N % threadsPerBlock != 0) blocks++;

    Vadd <<< blocks, threadsPerBlock  >>> (d_A, d_B, d_C, N);    
    
    
    cudaDeviceSynchronize();
    
    // Copy result from device memory to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
   
    bool correct = true;

    // Calculate solution on the host and compare
    for (int i = 0; i < N; i++)
    {
        if (h_C[i] != (h_A[i] + h_B[i]))
        {
            printf("ERROR: expected h_C[%i] = %f but received %f\n", i, h_A[i] + h_B[i], h_C[i]);
            correct = false;
            break;
        }
    }

    if (correct) printf("---PASSED---\n");

    // Free host and device memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}
