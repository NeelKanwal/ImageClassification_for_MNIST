clc;
close all
clear all

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(digitData.Files{perm(i)});
% end
% 
% digitData.countEachLabel

trainNumFiles = 750;
[trainDigitData,valDigitData] = splitEachLabel(digitData,trainNumFiles,'randomize');

myNet = [
    imageInputLayer([28 28 1])

    convolution2dLayer([8 8],8)
     %batchNormalizationLayer
    reluLayer();
    
    %maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer([4 4],8)
     %batchNormalizationLayer
    reluLayer();
    
    %maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(10)
    softmaxLayer();
    classificationLayer()
    ];

options = trainingOptions('sgdm','MaxEpochs',20,'ValidationData',valDigitData,'ValidationFrequency',30,'Verbose',false,...
    'Plots','training-progress','ValidationPatience',Inf,'MiniBatchSize',200,'InitialLearnRate',0.15);
  
trainedNet = trainNetwork(trainDigitData,myNet,options);


% Learning Filters
%%
weight_conv1 = trainedNet.Layers(2,1).Weights;
weight_conv1 = reshape(weight_conv1,[16,16,1,8])
figure,montage(mat2gray(weight_conv1))


weight_conv2 = trainedNet.Layers(4,1).Weights;
weight_conv2 = reshape(weight_conv2,[8,8,1,64])
figure,montage(mat2gray(weight_conv2))

%%
% Activation

activconv1 = activations(trainedNet ,imread(valDigitData.Files{25}),'conv_1','OutputAs','channels');
A_activconv1 = reshape(activconv1,[25,25,1,8]);
figure,montage(mat2gray(A_activconv1))

activconv2 = activations(trainedNet ,imread(valDigitData.Files{25}),'conv_2','OutputAs','channels');
A_activconv2 = reshape(activconv2,[24,24,1,8]);
figure,montage(mat2gray(A_activconv2))

%% Analysis

% By repeating the same like before


%% Normalization Layer
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(digitData.Files{perm(i)});
% end
% 
% digitData.countEachLabel

trainNumFiles = 750;
[trainDigitData,valDigitData] = splitEachLabel(digitData,trainNumFiles,'randomize');

myNet = [
    imageInputLayer([28 28 1])

    convolution2dLayer([4 4],8)
    batchNormalizationLayer
    reluLayer();

    %maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer([2 2],8)
    batchNormalizationLayer
    reluLayer();

    %maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(10)
    softmaxLayer();
    classificationLayer()
    ];

options = trainingOptions('sgdm','MaxEpochs',20,'ValidationData',valDigitData,'ValidationFrequency',30,'Verbose',false,...
    'Plots','training-progress','ValidationPatience',Inf,'MiniBatchSize',200,'InitialLearnRate',0.01);
  
trainedNet = trainNetwork(trainDigitData,myNet,options);
