# Neural-Computation
This is my code for Neural Computation course project.

## Project 1. Deep Clustering
### Project description
The goal of this project is to get you familiar with one deep learning platform and use it to solve one of the most fundamental problems, data clustering, with the help of Autoencoders to map the data into a latent space so that clustering purity can be improved.. 

### Data set. 
1.	Toy data set: mixture of three Gaussians data set. 
2.	MNIST data sets. 
3.	If you are interested, you can download other data sets used in the reading list papers (bonus).

### Tasks 
0.	Implement the k-means algorithm on the toy data set. Plot the loss function curve over epochs, and also the cluster centers (like following)
 
1.	Implement differentiable k-means clustering with k = 10 directly on the 784-dimensional images (training set of MNIST 60K images). Plot the evolution of the clustering accuracy over epochs, and report the average clustering accuracy over 10 random initialized runs. 
a)	There are a number of variants of k-means clustering as has been discussed in the class, such as how to choose the variables to optimize, relations between variables if there are multiple of them, how to choose bandwidth, loss function, etc. please compare at least two variants and report the better one.
b)	You need to learn how to perform gradient updates and choose different optimization schemes using deep learning platforms (basically, just how to call related functions)
2.	Deep clustering: Perform k-means clustering on the bottleneck layer of an auto-encoder.
a)	You need to test with different choices of: number of layers in AE; nonlinear activation functions, the dimension of each latent layers, and report the best configuration and clustering accuracies. 
b)	You need to consider the scale of the bottleneck layer, and see whether there is a need to prevent it from diminishing.
c)	Use t-SNE to visualize the clusters you have obtained and see how they look like. (tSNE package https://lvdmaaten.github.io/tsne/   and useful likes  https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1)
You can google relevant posts on t-SNE freely.
3.	Bonus task (can be very challenging, but optional):
a)	Bi-clustering. How to perform bi-clustering of the input data matrix. For MNIST, that means both samples and pixels will be clustered. (Hint: you may want to reformulate AE).
b)	Irregular shaped clusters. How to handle clusters of various shapes. (Hint. Revise the loss function) 

## Project 2. Sequence Classification in Brain functional networks
### Project description
The goal of this project is to apply several deep learning techniques to solve problems in the diagnosis of psychiatric diseases, based on the multi-dimensional time series through fMRI. Related methods include attention, encode-decoder model, and graph neural networks. 

### Data set. 
1.	Training data: 246 samples; testing data 20 samples.
2.	Each sample is a csv file, with 240 rows (each row is a time point) and 94 columns (each column is a brain region). Namely the csv file is a multi-dimensional time series. 
3.	The 94 brain regions are the same for each sample. The time points, though numbered from 1 to 240 for every sample, do not have any correspondence across different samples. 

### Task: Choose one of the following methods.
1.	Use attention mechanism to perform sequence classification
2.	Use GNN (graph neural networks) to perform classification 

### Report
1.	Details of your algorithm design and performance curve on the training data.
2.	Submit the predicted labels of your algorithm for the 20 testing data points (1/0). 
