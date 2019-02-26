#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <math.h>
////////////////////////////////////////////////////////////////////////////////
extern "C"
#include "neuralnet.h"
#include "neuralnet_gold.h"

//***** Declarations forward *****//
int ** readInputs(char * filename, int rows, int columns);
double ** double_readInputs(char * filename, int rows, int columns);
void transpose(double **A, double **B, int N, int M);
void MLP_train(double** train, int** train_labels_onehot, double **w, double **wTranspose, double **delta_w, double **v, double **vTranspose, double **delta_v, double **z, double **y);
int** predict_labels(double **y);
double get_accuracy(int **predicted_labels, int **train_labels);
void free_2dArray_double(double **data, int rows);
void free_2dArray_int(int **data, int rows);


void computeGold()
{
    printf("The problem started with sequential coding\n");
    printf("Reading inputs...\n");
    char train_name[] = "train_data.csv";
    double **train = double_readInputs(train_name, NUM_ELEMENTS, FEATURES+1);
    char train_labels_onehot_name[] = "train_labels_onehotencoded.csv";
    int **train_labels_onehot = readInputs(train_labels_onehot_name, NUM_ELEMENTS, NO_OF_CLASSES);
    char train_labels_name[] = "train_labels.csv";
    int **train_labels = readInputs(train_labels_name, NUM_ELEMENTS, 1);

    printf("Successs!!\n");

    // for(int i=0; i<NUM_ELEMENTS; i++)
    // {
    //   for(int j=0; j<FEATURES+1; j++)
    //   {
    //     printf("%f ", train[i][j]);
    //   }
    //   printf("\n");
    // }

    printf("Initializing variables\n");
    // Initializing variables
    //setting seed for random number genrator
    srand(123);
    double a = -0.01;
    double b = 0.01;

    //Malloc for W and delta_W
    double **w = (double **)malloc(HIDDEN_UNITS * sizeof(double *));
    double **delta_w = (double **)malloc(HIDDEN_UNITS * sizeof(double *));
    for (int i=0; i<HIDDEN_UNITS; i++)
    {
         w[i] = (double *)malloc((FEATURES+1)* sizeof(double));
         delta_w[i] = (double *)malloc((FEATURES+1)* sizeof(double));
    }

    //Initializing W : size (m, d), m: number of hidden units , d: number of features
    for(int x=0; x<HIDDEN_UNITS; x++)
    {
      for(int y=0; y<FEATURES+1; y++)
      {
        // w[x][y] = ((double)rand()/RAND_MAX) * (b - a) + a;
        w[x][y] = 0.01;
        delta_w[x][y] = 0.0;
      }
    }

    //transpose W
    double **wTranspose = (double **)malloc((FEATURES+1) * sizeof(double *));
    for (int i=0; i<(FEATURES+1); i++)
         wTranspose[i] = (double *)malloc(HIDDEN_UNITS* sizeof(double));
    transpose(w, wTranspose, HIDDEN_UNITS, FEATURES+1);

    //Malloc for V and delta_v
    double **v = (double **)malloc(NO_OF_CLASSES * sizeof(double *));
    double **delta_v = (double **)malloc(NO_OF_CLASSES * sizeof(double *));
    for (int i=0; i<NO_OF_CLASSES; i++)
    {
      v[i] = (double *)malloc((HIDDEN_UNITS+1)* sizeof(double));
      delta_v[i] = (double *)malloc((HIDDEN_UNITS+1)* sizeof(double));
    }

    //Initializing V : size (k, m), m: number of hidden units , k: number of classes
    for(int x=0; x<NO_OF_CLASSES; x++)
    {
      for(int y=0; y<HIDDEN_UNITS+1; y++)
      {
        // v[x][y] = ((double)rand()/RAND_MAX) * (b - a) + a;
        v[x][y] = 0.01;
        delta_v[x][y] = 0.0;
      }
    }

    //transpose v
    double **vTranspose = (double **)malloc((HIDDEN_UNITS+1) * sizeof(double *));
    for (int i=0; i<(HIDDEN_UNITS+1); i++)
         vTranspose[i] = (double *)malloc(NO_OF_CLASSES* sizeof(double));
    transpose(v, vTranspose, NO_OF_CLASSES, HIDDEN_UNITS+1);

    //Malloc for z and y
    double **z = (double **)malloc(NUM_ELEMENTS * sizeof(double *));
    double **y = (double **)malloc(NUM_ELEMENTS * sizeof(double *));
    for (int i=0; i<NUM_ELEMENTS; i++)
    {
         z[i] = (double *)malloc(HIDDEN_UNITS* sizeof(double));
         y[i] = (double *)malloc(NO_OF_CLASSES* sizeof(double));
    }
    //assigning intitial values to z
    for(int x=0; x<NUM_ELEMENTS; x++)
    {
      for(int y=0; y<HIDDEN_UNITS; y++)
      {
        z[x][y] = 0.0;
      }
    }

    //assigning initial values for y
    for(int i=0; i<NUM_ELEMENTS; i++)
    {
    for(int j=0; j<NO_OF_CLASSES; j++)
      {
        y[i][j] = 0.0;
      }
    }

    printf("Successs!!\n");
    printf("MLP Training\n");
    MLP_train(train, train_labels_onehot, w, wTranspose, delta_w, v, vTranspose, delta_v, z, y);
    printf("After MLP call\n");

    // for(int n=0; n<ITERATIONS; n++)
    // {
    //   for(int i=0; i <NO_OF_CLASSES; i++)
    //   {
    //       printf("%f ", y[n][i]);
    //   }
    //   printf("\n");
    // }

    int **predicted_labels = predict_labels(y);
    double accuracy = get_accuracy(predicted_labels, train_labels);

    printf("Accuracy: %f\n", accuracy);

    //Free-ing all the weights
    free_2dArray_double(w, HIDDEN_UNITS);
    free_2dArray_double(delta_w, HIDDEN_UNITS);
    free_2dArray_double(wTranspose, FEATURES+1);
    free_2dArray_double(v, NO_OF_CLASSES);
    free_2dArray_double(delta_v, NO_OF_CLASSES);
    free_2dArray_double(vTranspose, HIDDEN_UNITS+1);
    free_2dArray_double(z, NUM_ELEMENTS);
    free_2dArray_double(y, NUM_ELEMENTS);
    free_2dArray_int(predicted_labels, ITERATIONS);

    //free-ing the data array
    free_2dArray_double(train, NUM_ELEMENTS);
    free_2dArray_int(train_labels, NUM_ELEMENTS);
    free_2dArray_int(train_labels_onehot, NUM_ELEMENTS);
}


int ** readInputs(char * filename, int rows, int columns)
{
  char buff[100000];
  FILE *fp;
  char *record,*line;
  int i=0,j=0;
  int **data = (int **)malloc(rows * sizeof(int *));
  for (int i=0; i<rows; i++)
       data[i] = (int *)malloc(columns* sizeof(int));

  //Opening train_data csv file
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("\n file opening failed ");
    printf("Error: %s\n", strerror(errno));
    return (int **)(-1) ;
  }

  //iterating over each line of train_data.csv
  while((line = fgets(buff,sizeof(buff),fp)) != NULL)
  {
    //Splitting line into pixel values based on comma
    //printf("Line %d: %s\n",i, buff);
    record = strtok(line,",");
    while(record != NULL)
    {
      data[i][j] = atoi(record) ;
      ++j;
      record = strtok(NULL,",");
    }
    ++i ;
    j=0;
  }
  free(record);
  free(line);
  fclose(fp);
  return data;
}

double ** double_readInputs(char * filename, int rows, int columns)
{
  char buff[100000];
  FILE *fp;
  char *record,*line;
  int i=0,j=0;
  double **data = (double **)malloc(rows * sizeof(double *));
  for (int i=0; i<rows; i++)
       data[i] = (double *)malloc(columns* sizeof(double));

  //Opening train_data csv file
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("\n file opening failed ");
    printf("Error: %s\n", strerror(errno));
    return (double **)(-1) ;
  }

  //iterating over each line of train_data.csv
  while((line = fgets(buff,sizeof(buff),fp)) != NULL)
  {
    //Splitting line into pixel values based on comma
    //printf("Line %d: %s\n",i, buff);
    record = strtok(line,",");
    while(record != NULL)
    {
      data[i][j] = atof(record) ;
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

void free_2dArray_double(double **data, int rows)
{
  for (int i=0; i<rows; i++)
  {
       free(data[i]);
  }
  free(data);
}

void free_2dArray_int(int **data, int rows)
{
  for (int i=0; i<rows; i++)
  {
       free(data[i]);
  }
  free(data);
}

void MLP_train(double** train, int** train_labels_onehot, double **w, double **wTranspose, double **delta_w, double **v,
  double **vTranspose, double **delta_v, double **z, double **y)
{
  int ctr = 0;
  //Initializing matrix o for output at output layer
  double **out = (double **)malloc(NUM_ELEMENTS * sizeof(double *));
  for (int i=0; i<NUM_ELEMENTS; i++)
       out[i] = (double *)malloc(NO_OF_CLASSES * sizeof(double));

  //Initializing out to zero's
  for(int i=0; i<NUM_ELEMENTS; i++)
  {
    for(int j=0; j<NO_OF_CLASSES; j++)
    {
      out[i][j] = 0.0;
    }
  }

  //training for a fixed number of iterations
  while(ctr < EPOCH)
  {
    //For each row(image) in the training data set
    for(int i = 0; i < ITERATIONS; i++)
    {
      //Forward propagation
      //computing output of hidden layer z - applying ReLu as activation function
      for(int j = 0; j < HIDDEN_UNITS; j++)
      {
        double input = 0.0;
        for(int mul_i=0; mul_i < FEATURES+1; mul_i++)
        {
          //printf("%f ", wTranspose[mul_i][j]);
          input += (double)train[i][mul_i] * wTranspose[mul_i][j];
        }

        if(input < 0){
            z[i][j] = 0;
        }
        else{
          z[i][j] = input;
        }
        // printf("%f \n",z[i][j] );
      }
      // printf("After z, %d\n", i);
      //computing output for last layer
      for(int k=0; k<NO_OF_CLASSES;k++)
      {
        double sum = 0;
        for(int m=1; m<HIDDEN_UNITS+1;m++)
        {
          //printf("%f\n", vTranspose[m][k]);
          sum += z[i][m-1] * vTranspose[m][k];
        }
        out[i][k] = sum + vTranspose[0][k];
        // printf("%f\n", out[i][k]);
      }
      // printf("Heree noww done with output %d\n", i);
      //applying softmax
      double sum_y = 0.0;
      for(int k = 0; k<NO_OF_CLASSES; k++)
      {
        y[i][k] = exp((double)out[i][k]);
        sum_y += y[i][k];
      }
      for(int k = 0; k<NO_OF_CLASSES; k++)
      {
        y[i][k] = y[i][k]/sum_y;
        // printf("%f\n", y[i][k]);
      }
      // printf("After softmax %d\n",i);

      //Backward propagation
      //malloc for delta_v
      double **diff = (double **)malloc(1 * sizeof(double *));
      double **diff_transpose = (double **)malloc(NO_OF_CLASSES * sizeof(double *));
      diff[0] = (double *)malloc(NO_OF_CLASSES* sizeof(double));
      for(int a = 0; a < NO_OF_CLASSES; a++) {
        diff_transpose[a] = (double *)malloc(1* sizeof(double));
      }
      //computing delta_v
      for(int k=0; k<NO_OF_CLASSES; k++){
        diff[0][k] = (train_labels_onehot[i][k] - y[i][k]);
        // printf("%f \n",diff[0][k]);
      }
      // printf("After diff %d\n",i);

      transpose(diff, diff_transpose, 1, NO_OF_CLASSES);
      // for(int k=0; k<NO_OF_CLASSES; k++)
      // {
      //   printf("%f \n",diff_transpose[k][0]);
      // }
      // printf("After diff_transpose %d\n",i);

      for (int k = 0; k < NO_OF_CLASSES; k++)
      {
        for (int m = 0; m < HIDDEN_UNITS+1; m++)
        {
          double sum_v = 0.0;
          if (m == 0){
            sum_v += diff_transpose[k][0] * 1;
          }
          else{
            sum_v += diff_transpose[k][0]*z[i][m-1];
          }
          delta_v[k][m] = eta * sum_v;
        }
      }
      // printf("Delta V %d\n", i);
      // for (int k = 0; k < NO_OF_CLASSES; k++)
      // {
      //   for (int m = 0; m < HIDDEN_UNITS+1; m++)
      //   {
      //     printf("%f", delta_v[k][m]);
      //   }
      //   printf("\n");
      // }

      //Checking if input is less than zero for ReLu
      for(int j = 0; j < HIDDEN_UNITS; j++)
      {
        double input = 0.0;
        for(int mul_i=0; mul_i < FEATURES+1; mul_i++)
        {
          input += (double)train[i][mul_i] * wTranspose[mul_i][j];
        }
        //computing delta_w
        //computing sum
        double sum_w = 0.0;
        for(int k = 0; k < NO_OF_CLASSES; k++){
          sum_w += diff_transpose[k][0] * v[k][j+1];
        }
        // printf("%f ",sum_w);

        for(int n=0; n < FEATURES+1; n++)
        {
          if (input < 0){
            delta_w[j][n] = 0;
          }
          else{
            delta_w[j][n] = eta * sum_w * z[i][j] * (1-z[i][j]) * train[i][n];
            // printf("%lf ", delta_w[j][n] );
          }
        }
        // printf("\n");
      }

      // for(int x=0; x<HIDDEN_UNITS; x++)
      // {
      //   for(int y=0; y<FEATURES+1; y++)
      //   {
      //     printf("%f ",delta_w[x][y]);
      //   }
      //   printf("\n");
      // }
      // printf("After delta w %d\n",i);

       //Updating w, v and wTranspose
       for(int x = 0; x < HIDDEN_UNITS; x++){
        for(int y = 0; y < FEATURES+1; y++){
          w[x][y] += delta_w[x][y];
        }}

      for(int x = 0; x < NO_OF_CLASSES; x++){
       for(int y = 0; y < HIDDEN_UNITS+1; y++){
         v[x][y] += delta_v[x][y];
       }}

      transpose(w, wTranspose, HIDDEN_UNITS, FEATURES+1);
      transpose(v, vTranspose, NO_OF_CLASSES, HIDDEN_UNITS+1);

      free_2dArray_double(diff, 1);
      free_2dArray_double(diff_transpose, NO_OF_CLASSES);
    }
    ctr++;
  }
  free_2dArray_double(out, NUM_ELEMENTS);

}

void transpose(double **A, double **B, int N, int M)
{
    int i, j;

    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            B[i][j] = A[j][i];
}

int** predict_labels(double **y)
{
  int **predicted_labels = (int **)malloc(ITERATIONS * sizeof(int *));
  for(int a = 0; a < ITERATIONS; a++)
    predicted_labels[a] = (int *)malloc(1* sizeof(int));

  int max_index;
  double max;
  //finding max probability for every row and assigning the index as label
  for(int i=0; i<ITERATIONS; i++)
  {
    max_index = 0;
    max = y[i][0];
    for(int j=1; j<NO_OF_CLASSES; j++)
    {
      if(y[i][j] > max)
      {
        max_index = j;
        max = y[i][j];
      }
    }
    predicted_labels[i][0] = max_index;
    // printf("Index: %d", max_index);
    // printf("pred: %d\n", predicted_labels[i][0]);
  }

  return predicted_labels;
}

double get_accuracy(int **predicted_labels, int **train_labels)
{
  int count = 0;

  for(int i = 0; i<ITERATIONS; i++)
  {
    printf("Train label: %d", train_labels[i][0]);
    printf("Predicted label: %d\n", predicted_labels[i][0]);
    if(predicted_labels[i][0] == train_labels[i][0])
      count ++;
  }
  return ((double)count/ITERATIONS);
}
