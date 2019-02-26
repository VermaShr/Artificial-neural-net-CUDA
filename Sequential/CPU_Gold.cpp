#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#define FEATURES  784
#define ROWS 60000
#define HIDDEN_UNITS 10
#define NO_OF_CLASSES 10
#define eta 0.001
#define EPOCH 10

//***** Declarations forward *****//
int ** readInputs(char * filename, int rows, int columns);
void transpose(float **A, float **B, int N, int M);
void MLP_train(int** train, int** train_labels_onehot, float **w, float **wTranspose, float **delta_w, float **v, float **vTranspose, float **delta_v, float **z, float **y);
int** predict_labels(float **y);
float get_accuracy(int **predicted_labels, int **train_labels);

int main(int argc, char** argv)
{
  printf("Reading inputs...\n");
  char train_name[] = "train_data.csv";
  int **train = readInputs(train_name, ROWS, FEATURES);
  char train_labels_onehot_name[] = "train_labels_onehotencoded.csv";
  int **train_labels_onehot = readInputs(train_labels_onehot_name, ROWS, NO_OF_CLASSES);
  char train_labels_name[] = "train_labels.csv";
  int **train_labels = readInputs(train_labels_name, ROWS, 1);

  printf("Successs!!\n");

  printf("Initializing variables\n");
  // Initializing variables
  //setting seed for random number genrator
  srand(-123);
  float a = -0.01;
  float b = 0.01;

  //Malloc for W and delta_W
  float **w = (float **)malloc(HIDDEN_UNITS * sizeof(float *));
  float **delta_w = (float **)malloc(HIDDEN_UNITS * sizeof(float *));
  for (int i=0; i<HIDDEN_UNITS; i++)
  {
       w[i] = (float *)malloc((FEATURES+1)* sizeof(float));
       delta_w[i] = (float *)malloc((FEATURES+1)* sizeof(float));
  }

  //Initializing W : size (m, d), m: number of hidden units , d: number of features
  for(int x=0; x<HIDDEN_UNITS; x++)
  {
    for(int y=0; y<FEATURES+1; y++)
    {
      w[x][y] = ((float)rand()/RAND_MAX) * (b - a) + a;
      delta_w[x][y] = 0.0;
    }
  }

  // for(int x=0; x<HIDDEN_UNITS; x++)
  // {
  //   for(int y=0; y<FEATURES+1; y++)
  //   {
  //     printf("%f \n", w[x][y]);
  //   }
  //   //printf("\n");
  // }

  //transpose W
  float **wTranspose = (float **)malloc((FEATURES+1) * sizeof(float *));
  for (int i=0; i<(FEATURES+1); i++)
       wTranspose[i] = (float *)malloc(HIDDEN_UNITS* sizeof(float));
  transpose(w, wTranspose, HIDDEN_UNITS, FEATURES+1);

  //Malloc for V and delta_v
  float **v = (float **)malloc(NO_OF_CLASSES * sizeof(float *));
  float **delta_v = (float **)malloc(NO_OF_CLASSES * sizeof(float *));
  for (int i=0; i<NO_OF_CLASSES; i++)
  {
    v[i] = (float *)malloc((HIDDEN_UNITS+1)* sizeof(float));
    delta_v[i] = (float *)malloc((HIDDEN_UNITS+1)* sizeof(float));
  }

  //Initializing V : size (k, m), m: number of hidden units , k: number of classes
  for(int x=0; x<NO_OF_CLASSES; x++)
  {
    for(int y=0; y<HIDDEN_UNITS+1; y++)
    {
      v[x][y] = ((float)rand()/RAND_MAX) * (b - a) + a;
      delta_v[x][y] = 0.0;
    }
  }
  //transpose v
  float **vTranspose = (float **)malloc((HIDDEN_UNITS+1) * sizeof(float *));
  for (int i=0; i<(HIDDEN_UNITS+1); i++)
       vTranspose[i] = (float *)malloc(NO_OF_CLASSES* sizeof(float));
  transpose(v, vTranspose, NO_OF_CLASSES, HIDDEN_UNITS+1);

  //Malloc for z and y
  float **z = (float **)malloc(ROWS * sizeof(float *));
  float **y = (float **)malloc(ROWS * sizeof(float *));
  for (int i=0; i<ROWS; i++)
  {
       z[i] = (float *)malloc(HIDDEN_UNITS* sizeof(float));
       y[i] = (float *)malloc(NO_OF_CLASSES* sizeof(float));
  }
  //assigning intitial values to z
  for(int x=0; x<ROWS; x++)
  {
    for(int y=0; y<HIDDEN_UNITS; y++)
    {
      z[x][y] = 0.0;
    }
  }

  //assigning initial values for y
  for(int i=0; i<ROWS; i++)
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

  // for(int n=0; n<4; n++)
  // {
  //   for(int i=0; i <NO_OF_CLASSES; i++)
  //   {
  //       printf("%f ", y[n][i]);
  //   }
  //   printf("\n");
  // }

  int **predicted_labels = predict_labels(y);
  float accuracy = get_accuracy(predicted_labels, train_labels);

  printf("Accuracy: %f\n", accuracy);
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
  return data;
}

void MLP_train(int** train, int** train_labels_onehot, float **w, float **wTranspose, float **delta_w, float **v,
  float **vTranspose, float **delta_v, float **z, float **y)
{
  int ctr = 0;
  //Initializing matrix o for output at output layer
  float **out = (float **)malloc(ROWS * sizeof(float *));
  for (int i=0; i<ROWS; i++)
       out[i] = (float *)malloc(NO_OF_CLASSES * sizeof(float));

  //training for a fixed number of iterations
  while(ctr < EPOCH)
  {
    //For each row(image) in the training data set
    for(int i = 0; i < ROWS; i++)
    {
      //Forward propagation
      //computing output of hidden layer z - applying ReLu as activation function
      for(int j = 0; j < HIDDEN_UNITS; j++)
      {
        float input = 0.0;
        for(int mul_i=0; mul_i < FEATURES+1; mul_i++)
        {
          //printf("%f ", wTranspose[mul_i][j]);
          input += (float)train[i][mul_i] * wTranspose[mul_i][j];
        }

        if(input < 0){
            z[i][j] = 0;
        }
        else{
          z[i][j] = input;
        }
        //printf("%f\n",z[i][j] );
      }
      //printf("After z\n");
      //computing output for last layer
      for(int k=0; k<NO_OF_CLASSES;k++)
      {
        float sum = 0;
        for(int m=1; m<HIDDEN_UNITS+1;m++)
        {
          //printf("%f\n", vTranspose[m][k]);
          sum += z[i][m-1] * vTranspose[m][k];
        }
        out[i][k] = sum + vTranspose[0][k];
        //printf("%f\n", out[i][k]);
      }
      //printf("Heree noww done with output\n");
      //applying softmax
      float sum_y = 0.0;
      for(int k = 0; k<NO_OF_CLASSES; k++)
      {
        y[i][k] = exp((double)out[i][k]);
        sum_y += y[i][k];
      }
      for(int k = 0; k<NO_OF_CLASSES; k++)
      {
        y[i][k] = y[i][k]/sum_y;
        //printf("%f\n", y[i][k]);
      }

      //Backward propagation
      //malloc for delta_v
      float **diff = (float **)malloc(1 * sizeof(float *));
      float **diff_transpose = (float **)malloc(NO_OF_CLASSES * sizeof(float *));
      diff[0] = (float *)malloc(NO_OF_CLASSES* sizeof(float));
      for(int a = 0; a < NO_OF_CLASSES; a++) {
        diff_transpose[a] = (float *)malloc(1* sizeof(float));
      }
      //computing delta_v
      for(int k=0; k<NO_OF_CLASSES; k++){
        diff[0][k] = (train_labels_onehot[i][k] - y[i][k]);
      }

      transpose(diff, diff_transpose, 1, NO_OF_CLASSES);

      for (int k = 0; k < NO_OF_CLASSES; k++)
      {
        for (int m = 0; m < HIDDEN_UNITS+1; m++)
        {
          float sum_v = 0.0;
          if (m == 0){
            sum_v += diff_transpose[k][0] * 1;
          }
          else{
            sum_v += diff_transpose[k][0]*z[i][m-1];
          }
          delta_v[k][m] = eta * sum_v;
        }
      }
      //printf("Delta V\n");
      // for (int k = 0; k < NO_OF_CLASSES; k++)
      // {
      //   for (int m = 0; m < HIDDEN_UNITS+1; m++)
      //   {
      //     printf("%f\n", delta_v[k][m]);
      //   }
      //   printf("\n");
      // }

      //Checking if input is less than zero for ReLu
      for(int j = 0; j < HIDDEN_UNITS; j++)
      {
        float input = 0.0;
        for(int mul_i=0; mul_i < FEATURES+1; mul_i++)
        {
          input += (float)train[i][mul_i] * wTranspose[mul_i][j];
        }
        //computing delta_w
        //computing sum
        float sum_w = 0.0;
        for(int k = 0; k < NO_OF_CLASSES; k++){
          sum_w += diff_transpose[k][0] * v[k][j+1];
        }

        for(int n=0; n < FEATURES+1; n++)
        {
          if (input < 0){
            delta_w[j][n] = 0;
          }
          else{
            delta_w[j][n] = eta * sum_w * z[i][j] * (1-z[i][j]) * train[i][n];
          }
        }
      }

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
    }
    ctr++;
  }
}

void transpose(float **A, float **B, int N, int M)
{
    int i, j;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            B[i][j] = A[j][i];
}

int** predict_labels(float **y)
{
  int **predicted_labels = (int **)malloc(ROWS * sizeof(int *));
  for(int a = 0; a < ROWS; a++)
    predicted_labels[a] = (int *)malloc(1* sizeof(int));

  int max_index;
  float max;
  //finding max probability for every row and assigning the index as label
  for(int i=0; i<ROWS; i++)
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

float get_accuracy(int **predicted_labels, int **train_labels)
{
  int count = 0;

  for(int i = 0; i<ROWS; i++)
  {
    printf("Train label: %d", train_labels[i][0]);
    printf("Predicted label: %d\n", predicted_labels[i][0]);
    if(predicted_labels[i][0] == train_labels[i][0])
      count ++;
  }
  return ((float)count/ROWS);
}
