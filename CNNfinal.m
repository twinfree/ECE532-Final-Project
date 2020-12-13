%read in images
[img,label] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
[imgTEST,labelTEST] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);
%%
%define CNN architecture
layers = [
    imageInputLayer([20 20 1])
    
    convolution2dLayer(5,20,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,40,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,80,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
       
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%% Train
Xtrain = img(:,:,1:50000);
XVal = img(:,:,50001:end);
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...%'ValidationData',{reshape(XVal, [20 20 1 10000]), categorical(label(50001:end))}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
out = trainNetwork(reshape(img, [20 20 1 60000]),categorical(label), layers, options);

%% predict
Ypred = classify(out, reshape(imgTEST, [20 20 1 10000]));
acc = mean(Ypred == categorical(labelTEST));


