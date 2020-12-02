function [Xtrain,Xtest] = preProcessingMain(img, imgtest)
%[img,label] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
d = dir('filters'); files = {d.name}; files = files(3:end); % read in filters
filters = cell(1,length(files));
%normalizationConstants = zeros(1,length(files));
for f = 1:length(filters)
    ff = files{f};
    temp = open(['filters\' ff]);
    temp = struct2cell(temp);
    filters{f} = temp{1};
    %normalizationConstants(f) = sum(temp{1}, 'all'); %max output of convolution with filter is sum of all values in filter
    %normalizationConstants = [1 1 1 1 1];
end

for imgset = 1:2

if imgset == 2
    img = imgtest;
end
    
%apply processing to each image to extract features
X = zeros(size(img,3), (length(filters)+2) * 8 * 8);
for i = 1:size(img,3)
    digit = img(:,:,i);
    filterOutput = zeros(8,8, length(filters) + 2); %num structure correlation filters + corner detection + dilation/erosion
    for filt = 1:length(filters)
        thisFilt = filters{filt};
        output = conv2(digit, thisFilt, 'same');%/normalizationConstants(filt); 
        %output(output<.75) = 0;
        output = imresize(output, [8 8]);
        filterOutput(:,:,filt) = output;%subsample
    end
    R = cornerDetection(digit, 11);
    R = R / max(R(:));
    R(R<.75) = 0; %threshold
    R = imresize(R, [8 8]);%subsample
    filterOutput(:,:, filt+1) = R;
    
    DE = dilateErode(digit, 5); % apply dilation/erosion function
    DE = imresize(DE, [8 8]); %subsample
    filterOutput(:,:, filt+2) = DE;
    
    featureVector = reshape(filterOutput, [1 numel(filterOutput)]);
    X(i,:) = featureVector;
    
end

if imgset == 1
    Xtrain = X;
else
    Xtest = X;
end

end
%normalize
%X = normalizeX(X, 17);
[Xtrain, Xtest] = normalizeFeat(Xtrain, Xtest); %trim zero columns and normalize each non zero column
end



