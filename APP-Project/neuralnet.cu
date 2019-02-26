/* Neural Net.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <errno.h>
#include "util.h"

// includes, kernels
#include "neuralnet_kernel.cu"
#include "neuralnet_gold.h"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
//void computeGold();
float* readInputs(char * filename, int rows, int columns);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    TIME_IT("Sequential Code",1 , computeGold();)
    unsigned int size = NUM_ELEMENTS*(FEATURES+1);//some random number for now. Will change it to be the size of the dataset
    float* input_h = readInputs(argv[1], NUM_ELEMENTS, FEATURES+1);
    float* onehotR_h = readInputs(argv[2], NUM_ELEMENTS, NO_OF_CLASSES);
    float* train_labels_h = readInputs(argv[3], NUM_ELEMENTS, 1);
    float* w_h = (float*)malloc(NUM_ELEMENTS*(FEATURES+1)*sizeof(float));
    float* v_h = (float*)malloc((HIDDEN_UNITS+1)*(NO_OF_CLASSES)*sizeof(float));
    float* input_d;
    float* onehotR_d;
    float* train_labels_d;
    float* w_d;
    float* v_d;

    //Initializing the seed
    srand(-123);
    float a = -0.01;
    float b = 0.01;

    float *y_d;
    allocateDeviceArray(&y_d, NUM_ELEMENTS*NO_OF_CLASSES);
    cudaMemset(y_d, 0, NUM_ELEMENTS*NO_OF_CLASSES*sizeof(float));

    //read one hot label r and original label r into host variables and then transfer into device array
    allocateDeviceArray(&input_d, size);
    allocateDeviceArray(&onehotR_d, NUM_ELEMENTS*NO_OF_CLASSES);
    allocateDeviceArray(&train_labels_d, NUM_ELEMENTS);
    allocateDeviceArray(&w_d, NUM_ELEMENTS*(FEATURES+1));
    allocateDeviceArray(&v_d, (HIDDEN_UNITS+1)*NO_OF_CLASSES);

    for(int x=0; x<HIDDEN_UNITS; x++)
    {
      for(int y=0; y<FEATURES+1; y++)
      {
        w_h[x*HIDDEN_UNITS + y] = ((float)rand()/RAND_MAX) * (b - a) + a;
      }
    }

    for(int x=0; x<NO_OF_CLASSES; x++)
    {
      for(int y=0; y<HIDDEN_UNITS+1; y++)
      {
        v_h[x*NO_OF_CLASSES + y] = ((float)rand()/RAND_MAX) * (b - a) + a;
      }
    }

    copyDataHostToDevice(input_d, input_h , size);
    copyDataHostToDevice(onehotR_d, onehotR_h , NUM_ELEMENTS*NO_OF_CLASSES);
    copyDataHostToDevice(train_labels_d, train_labels_h , NUM_ELEMENTS);
    copyDataHostToDevice(w_d, w_h , NUM_ELEMENTS*(FEATURES+1));
    copyDataHostToDevice(v_d, v_h , (HIDDEN_UNITS+1)*NO_OF_CLASSES);
    /* This is the call you will use to time your parallel implementation */
    printf("parallel code\n");
    TIME_IT("getAccuracy",
            1,
            getAccuracy(input_d, onehotR_d, train_labels_d, w_d, v_d, y_d);)


  float* y_h= (float*)malloc(NUM_ELEMENTS*NO_OF_CLASSES*sizeof(float));
  copyDataDeviceToHost(y_h, y_d, NUM_ELEMENTS*NO_OF_CLASSES);
  float accuracy = calculateErrorRate(train_labels_h, y_h);
	printf("ACCURACY: %f\n",accuracy);
    //free memory
    freeMemory(input_d);
    freeMemory(onehotR_d);
    freeMemory(train_labels_d);
    freeMemory(w_d);
    freeMemory(v_d);

    return 0;
}

float* readInputs(char * filename, int rows, int columns)
{
  char buff[100000];
  FILE *fp;
  char *record,*line;
  int i=0,j=0;
  float* data = (float*)malloc(rows * columns * sizeof(float *));

  //Opening train_data csv file
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("\n file opening failed ");
    printf("Error: %s\n", strerror(errno));
    return (float*)(-1) ;
  }

  //iterating over each line of train_data.csv
  while((line = fgets(buff,sizeof(buff),fp)) != NULL)
  {
    //Splitting line into pixel values based on comma
    record = strtok(line,",");
    while(record != NULL)
    {
      data[i * columns + j] = atof(record) ;
      ++j;
      record = strtok(NULL,",");
    }
    ++i ;
    j=0;
  }
  fclose(fp);
  free(record);
  free(line);
  return data;
}
