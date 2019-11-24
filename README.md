# Image Classification for Hand Written Digits Recognition.

the goal is to implement a Deep Learning-based classifier made of fully
connected layers.  The classifier should be able to correctly classify the content
of a set of images containing handwritten digits.

# Requirements
Matlab r2017b or higher
Image Processing Toolbox
Sufficient Ram for deeper layers.

# Dataset
Load the DigitDataset made of 28 × 28 pixels digit images.

Simply open file.m and run it. It will load dataset from repository.

# Network
The network is being trained using stochastic gradient descent and softmax as loss
function, the batch size is initially set to 200, the learning rate to 0.01 and max epochs to 30 

The parameters can be further tuned to acheive desired resutls keeping in mind to avoid overfitting.

# Results

The used parameters acheives 99.48% Validation Accuracy.

#Detailed Report


https://medium.com/@neelchawla/classification-of-handwritten-digits-using-matlab-cnn-37ad45c32057
