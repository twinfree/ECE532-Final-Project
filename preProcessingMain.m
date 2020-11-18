function X = preProcessingMain(img)
%takes the set 60000 MNIST images as input and outputs a matrix X, whose ith row corresponds to the ith images feature vector
%[img,label] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
d = dir('filters'); files = {d.name}; files = files(3:end); % read in filters
filters = cell(1,length(files));
%normalizationConstants = zeros(1,length(files));
for f = 1:length(filters) % read in filters
    ff = files{f};
    temp = open(['filters\' ff]);
    temp = struct2cell(temp);
    filters{f} = temp{1};
    %normalizationConstants(f) = sum(temp{1}, 'all'); %max output of convolution with filter is sum of all values in filter
    %normalizationConstants = [1 1 1 1 1];
end

%apply processing to each image to extract features
X = zeros(60000, (length(filters)+2) * 17 * 17);
for i = 1:60000
    digit = img(:,:,i);
    filterOutput = zeros(17,17, length(filters) + 2); %num structure correlation filters + corner detection + dilation/erosion
    for filt = 1:length(filters)
        thisFilt = filters{filt};
        output = conv2(digit, thisFilt, 'same');% convolve pattern/template matching filter with image
        %output(output<.75) = 0;
        output = imresize(output, [17 17]); %subsample
        filterOutput(:,:,filt) = output;
    end
    R = cornerDetection(digit, 7); %apply corner/junction detection
    R = R / max(R(:)); %normalize
    R(R<.75) = 0; %threshold
    R = imresize(R, [17 17]);%subsample
    filterOutput(:,:, filt+1) = R;
    
    DE = dilateErode(digit, 5); % apply dilation/erosion function
    DE = imresize(DE, [17 17]); %subsample
    filterOutput(:,:, filt+2) = DE;
    
    featureVector = reshape(filterOutput, [1 numel(filterOutput)]);
    X(i,:) = featureVector;
    
end

X = normalizeFeat(X); %trim zero columns and normalize each non zero column
end



