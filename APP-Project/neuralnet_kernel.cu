#ifndef _NEURALNET_KERNEL_H_
#define _NEURALNET_KERNEL_H_

#include <stdio.h>
#include "neuralnet.h"

// TODO: place each data vector (row of input_d + bias term) within constant memory

// Neural net relu matrix multiplication kernel
//  - height x width = HIDDEN_UNITS x FEATURES+1
//  - input   = 1 x width
//  - weights = flattened matrix = height x width --> NEED TO TRANSPOSE
//  - returns output = relu(input * transpose(weights)) = 1 x height
__global__ void ReluMulKernel(float *input, float *weights, float *output)
{
  // TODO: input vector --> constant memory

  // local const values
  const int height = HIDDEN_UNITS;
  const int width = FEATURES + 1;

  // index into flattened weights matrix
  int tx = threadIdx.x;
  int i = blockDim.x * blockIdx.x + tx;

  // index into the input vector
  int row = i / width;

  // index into the output vector
  int col = i % width;

  // local products vector
  __shared__ float local_output[height];

  if ((i < height * width) && (row < height))
  {
    // local_output = input * transpose(weights)
    atomicAdd(&local_output[row], input[col] * weights[i]);
    __syncthreads(); // wait for everyone to add to local_output

    if (tx < height)
    {
      // integrate the local products with global
      atomicAdd(&output[tx], local_output[tx]);
    }
    __syncthreads(); // wait for local products -> global

    // apply relu function
    if ((i < height) && (output[i] < 0.0)) output[i] = 0.0; // TODO: try to reduce divergence here?

  } // END if within weights
}

// Neural net softmax matrix multiplication kernel
//  - height x width = HIDDEN_UNITS+1 x NO_OF_CLASSES
//  - input   = 1 x (height-1)
//  - weights = flattened matrix = height x width
//  - returns output = softmax(input * weights) = 1 x width
__global__ void SoftmaxMulKernel(float *input, float *weights, float *output, int start)
{

  // local const values
  const int height = HIDDEN_UNITS + 1;
  const int width = NO_OF_CLASSES;

  // index into flattened weights matrix
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // index into the input vector
  int row = i / width;

  // index into the output vector
  int col = i % width;

  // local products vector
  __shared__ float local_output[width];


  if ((i < height * width) && (row < height))
  {

    if (row == 0)
    {
      // apply bias
      atomicAdd(&local_output[col], BIAS * weights[i]);
    }
    else
    {
      // adjust index into input since input has one less element
      atomicAdd(&local_output[col], input[row-1] * weights[i]);
    }
    __syncthreads(); // wait for everyone to add to local_output

    if (threadIdx.x < width)
    {
      // integrate the local products with global
      atomicAdd(&output[start + i], local_output[i]);
      __syncthreads();
    }

    // apply softmax function
    if (i < width)
    {
      __shared__ float sum;

      // 1. store the exp() of each output value
      __shared__ float exp_vector[width];
      exp_vector[i] = expf(output[start + i]);

      // 2. calculate the sum of all the exponent values
      //  --> width < BLOCK_SIZE, so this will only be in the first block
      if (threadIdx.x == 0) sum = 0.0;
      __syncthreads(); // wait for sum to be zeroed

      atomicAdd(&sum, exp_vector[i]);

      // 3. store new output value
      output[start + i] = exp_vector[i] / sum;
    } // END if within output
  } // END if within weights

}

// output = delta1
__global__ void BackPropMulKernel1(float *A, float *z, float *output)
{
  //delta v = transpose(z) * A
  int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < (HIDDEN_UNITS+1)) {
    for(int k = 0; k < NO_OF_CLASSES; k++) {
      if(i == 0)
        output[i*NO_OF_CLASSES + k] = A[k];
      else
        output[i*NO_OF_CLASSES + k] = z[i-1] * A[k];
		}
	}
}

__global__ void BackPropMulKernel2(float *A, float *data, float *v, float *deltaw, float *w, float *z)
{
	int h = threadIdx.x + blockDim.x * blockIdx.x;
	if(h < HIDDEN_UNITS){
		float check = 0;
		for(int i = 0 ; i < (FEATURES+1) ; i++){
			check += data[i]*w[h*(FEATURES+1)+i];
		}
		if(check > 0) {
      float sum = 0;
      for(int k = 0 ; k < NO_OF_CLASSES ; k++){
        sum += A[k]*v[h*NO_OF_CLASSES + k];
      }
      float temp = eta*sum*z[h]*(1-z[h]);
        for(int j = 0 ; j < (FEATURES+1) ; j++){
          deltaw[h*(FEATURES+1) + j] = temp*data[j];
        }
		} else {
      for(int j = 0 ; j < (FEATURES+1) ; j++){
        deltaw[h*(FEATURES+1) + j] = 0;
      }
		}
	}
}

// vectorAddition: matrix = matrix + deltaMatrix
//  - matrix = flattened with length N
__global__ void vectorAddition(float *matrix, float *deltaMatrix, int N)
{
  // index into the vectors

  // initial index using thread coarsening
  int i = COURSENESS*(blockDim.x * blockIdx.x) + threadIdx.x;
  int count = 0;

  while (i < N && count < COURSENESS)
  {
    matrix[i] += deltaMatrix[i];
    // move to next block
    i += blockDim.x;
    count++;
  }

}

__global__ void vectorSubtraction(float* input, float* output, float* delta, int N, int start)
{
  // index into the vectors
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) output[i] = input[start + i] - delta[start + i];
}

void getAccuracy(float *input_d, float *onehotR_d, float *R_d, float *w, float *v, float *y)
{

  //initialize deltaw
  float *deltaw;
  cudaMalloc((void**)&deltaw, (FEATURES+1)*HIDDEN_UNITS*sizeof(float));
  cudaMemset(deltaw, 0, (FEATURES+1)*HIDDEN_UNITS*sizeof(float));


  //initialize deltav
  float *deltav;
  cudaMalloc((void**)&deltav, (HIDDEN_UNITS+1)*NO_OF_CLASSES*sizeof(float));
  cudaMemset(deltav, 0, (HIDDEN_UNITS+1)*NO_OF_CLASSES*sizeof(float));

  //initialize z
  float *z;
  cudaMalloc((void**)&z, HIDDEN_UNITS*sizeof(float));
  cudaMemset(z, 0, HIDDEN_UNITS*sizeof(float));

  float *A;
  cudaMalloc((void**)&A, NO_OF_CLASSES*sizeof(float));
  cudaMemset(A, 0, NO_OF_CLASSES*sizeof(float));

  int i = 0;

  while(i<EPOCH)
  {
    //iterate through the data
    for (int k = 0; k<NUM_ELEMENTS; k++)
    {
      // point to portion of the input array (within device global memory)
      float *data = &input_d[k*(FEATURES+1)];
      int blocks = 0;

      //kernel call for z = relu(x * w^T) --> data = x, w = flattened matrix
      // divide w up into blocks -- one thread per matrix element
      blocks = ((FEATURES+1)*HIDDEN_UNITS) / BLOCK_SIZE;
      if (((FEATURES+1)*HIDDEN_UNITS) % BLOCK_SIZE != 0) blocks++;

      ReluMulKernel<<< blocks, BLOCK_SIZE >>>(data, w, z);
      cudaDeviceSynchronize();

      //kernel call for y = softmax(z * v) --> v = flattened matrix
      // divide v up into blocks -- one thread per matrix element
      blocks = ((HIDDEN_UNITS+1)*NO_OF_CLASSES) / BLOCK_SIZE;
      if (((HIDDEN_UNITS+1)*NO_OF_CLASSES) % BLOCK_SIZE != 0) blocks++;
      SoftmaxMulKernel<<< blocks, BLOCK_SIZE >>>(z, v, y, k*NO_OF_CLASSES);
      cudaDeviceSynchronize();

      //calculate A = eta*(one hot R - one hot Y) -- no kernel call for this, will be a vector
      // (input, output, delta, N, start)
      vectorSubtraction<<<1,BLOCK_SIZE>>>(onehotR_d, A, y, NO_OF_CLASSES, k*NO_OF_CLASSES);
      cudaDeviceSynchronize();

      //kernel call for delta v
      int kernel1_grid_size = (int)ceil((float)(HIDDEN_UNITS+1)/BLOCK_SIZE);

      BackPropMulKernel1<<<kernel1_grid_size,BLOCK_SIZE>>>(A, z, deltav);
      cudaDeviceSynchronize();


      //kernel call for delta w
      int kernel2_grid_size = (int)ceil((float)HIDDEN_UNITS/BLOCK_SIZE);

      BackPropMulKernel2<<<kernel2_grid_size,BLOCK_SIZE>>>(A, data, v, deltaw,w,z);
      cudaDeviceSynchronize();

      //kernel call for updating v values
      // using thread coarsening
      blocks = ((HIDDEN_UNITS+1)*NO_OF_CLASSES) / (BLOCK_SIZE*COURSENESS);
      if (((HIDDEN_UNITS+1)*NO_OF_CLASSES) % (BLOCK_SIZE*COURSENESS) != 0) blocks++;

      vectorAddition <<< blocks , BLOCK_SIZE >>> (v, deltav, (HIDDEN_UNITS+1)*NO_OF_CLASSES);
      cudaDeviceSynchronize();

      //kernel call for updating w values
      // using thread coarsening
      blocks = ((FEATURES+1)*HIDDEN_UNITS) / (BLOCK_SIZE*COURSENESS);
      if (((FEATURES+1)*HIDDEN_UNITS) % (BLOCK_SIZE*COURSENESS) != 0) blocks++;

      vectorAddition <<< blocks , BLOCK_SIZE >>> (w, deltaw, (FEATURES+1)*HIDDEN_UNITS);
      cudaDeviceSynchronize();

    }

    i++;
  }

}


float calculateErrorRate(float * r, float *y_h)
{
  int count = 0;
  for(int i = 0; i < NUM_ELEMENTS; i++){
    if(r[i]==(float)predictY(&y_h[i * NO_OF_CLASSES])){
      count++;
    }
  }
  return (float)count / NUM_ELEMENTS;
}

int predictY(float *y){
  int maxindex = 0;
  float max = y[0];
  for(int j = 1; j<NO_OF_CLASSES; j++){
    if(y[j]>max){
      max = y[j];
      maxindex = j;
    }
  }
  return maxindex;
}

void allocateDeviceArray(float **deviceArray, unsigned int size)
{
  cudaMalloc((void**)deviceArray, size*sizeof(float));
}

void copyDataHostToDevice(float *deviceArray, float *hostArray, unsigned int size)
{
	cudaMemcpy(deviceArray, hostArray , size*sizeof(float), cudaMemcpyHostToDevice);
}
void copyDataDeviceToHost(float *deviceArray, float *hostArray, unsigned int size)
{
	cudaMemcpy(hostArray, deviceArray , size*sizeof(float), cudaMemcpyDeviceToHost);
}
void freeMemory(float *deviceArray)
{
	cudaFree(&deviceArray);
}


#endif // #ifndef _NEURALNET_KERNEL_H_
