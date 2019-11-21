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

    fullyConnectedLayer(100)
    reluLayer();

    %maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(100)
    reluLayer();

    %maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(100)
    reluLayer();

    fullyConnectedLayer(10)
    softmaxLayer();
    classificationLayer()];

options = trainingOptions('sgdm','MaxEpochs',30,'ValidationData',valDigitData,'ValidationFrequency',30,'Verbose',false,...
    'Plots','training-progress','ValidationPatience',Inf,'MiniBatchSize',200,'InitialLearnRate',0.01);
  
trainedNet = trainNetwork(trainDigitData,myNet,options);


% Testing of Network
%%


perm = randperm(10000,1);
imshow(digitData.Files{perm});
im=imread(digitData.Files{perm});
% imshow(valDigitData.Files{perm})
% im=imread(valDigitData.Files{perm});
YPred = classify(trainedNet,im)
%%
% Validation Accuracy
YTest = classify(trainedNet,valDigitData);
TTest = valDigitData.Labels;
Val_accuracy = sum(YTest == TTest)/numel(TTest)

%%
%Overfiting the network
% By training it over reduced training data or increasing the number of
% hidden layers

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

trainNumFiles = 100;
[trainDigitData,valDigitData] = splitEachLabel(digitData,trainNumFiles,'randomize');

myNet = [
    imageInputLayer([28 28 1])

    fullyConnectedLayer(100)
    reluLayer();

    %maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(100)
    reluLayer();

    %maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(100)
    reluLayer();

    fullyConnectedLayer(10)
    softmaxLayer();
    classificationLayer()];

options = trainingOptions('sgdm','MaxEpochs',30,'ValidationData',valDigitData,'ValidationFrequency',30,'Verbose',false,...
    'Plots','training-progress','ValidationPatience',Inf,'MiniBatchSize',200,'InitialLearnRate',0.01);
  
OverfitNet = trainNetwork(trainDigitData,myNet,options);

% Testing overfitted 
%%
figure;
perm = randperm(9000,1);

imshow(valDigitData.Files{perm})
im=imread(valDigitData.Files{perm});
YPred = classify(OverfitNet,im)