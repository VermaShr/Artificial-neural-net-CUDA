#ifndef _NEURALNET_H_
#define _NEURALNET_H_

#define NUM_ELEMENTS 60000
#define FEATURES 784
#define EPOCH 10
#define BLOCK_SIZE 256
#define HIDDEN_UNITS 10
#define NO_OF_CLASSES 10
#define eta 0.01

#define COURSENESS 4
#define BIAS 1.0
//Add code here


// NeuralNet Kernel function
void getAccuracy(float *input_d, float *onehotR_d, float *R_d);

/* Include below the function headers of any other functions that you implement */
void allocateDeviceArray(float **deviceArray, unsigned int size);
void copyDataHostToDevice(float *deviceArray, float *hostArray, unsigned int size);
void copyDataDeviceToHost(float *deviceArray, float *hostArray, unsigned int size);
void freeMemory(float *device);
float calculateErrorRate(float* r, float* y);
int predictY(float *y);


#endif // _NEURALNET_H_
