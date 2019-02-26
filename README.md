# Artificial-neural-net-CUDA
Classification using Multilayer Perceptron implemented in CUDA
Final Project for EE 5351: Applied Parallel Programming

Dataset: MNIST Digits  
Drive: https://drive.google.com/drive/u/1/folders/0AM8p7TLEES-zUk9PVA  
Size of Datasets:  
  1. Train: 60,000 * 784  
  2. Test: 10,000 * 784  
  3. Train_labels: 60,000 * 1  
  4. Test_labels: 10,000 * 1  
  5. Train_one_hot_encoded_labels: 60,000 * 10  
  6. Test_one_hot_encoded_labels: 10,000 * 10  

The Data format: First column represents the Label, remaining 784 columns represent the pixels of the image (image is 28 * 28)  

To run the code
type 'make' to build the executable.
type './neuralnet train_data.csv train_labels_onehotencoded.csv train_labels.csv' to run the program
