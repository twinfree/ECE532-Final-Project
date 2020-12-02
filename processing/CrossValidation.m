function err=CrossValidation(X, y, iterations, testSize, varargin)
%varargin: cell array in which first element is the algorithm followed by
%parameters specific to that algorithm: {'KNN' k}, {'OvO', 'ridge' or 'lasso', lamarray}
err = zeros(1,iterations);
for iter = 1:iterations
    idx = randperm(length(y));
    Xtrain = X(idx(1:end-testSize),:);
    Xtest = X(idx(end-testSize+1:end),:);
    ytrain = y(idx(1:end-testSize));
    ytest = y(idx(end-testSize+1:end));

    if strcmp(varargin{1},'KNN')
        prediction = KNN(Xtrain, ytrain, Xtest, varargin{2});
        err(iter) = mean(prediction' == ytest);
    elseif strcmp(varargin{1}, 'OvO')
        nck = nchoosek(0:9, 2);
        lamarray = varargin{3};
        accuracy = zeros(length(lamarray), size(nck,1));
        for i = 1:length(lamarray)
            if strcmp(varargin{2}, 'ridge')
                W = trainOvO(Xtrain, ytrain, 'ridge', lamarray(i));
            else
                W = trainOvO(Xtrain, ytrain, 'lasso', lamarray(i));
            end
            prediction = sign(Xtest * W);
            %evaluate accuracy of each pairwise classifier
            for pair = 1:size(nck,1)
                n1 = nck(pair, 1);
                n2 = nck(pair,2);
                idx2 = (ytest == n1) | (ytest == n2);
                y1v1 = ytest(idx2);
                y1v1(y1v1 == n1) = -1;
                y1v1(y1v1 == n2) = 1;
                accuracy(i, pair) = mean(y1v1 == prediction(idx2,pair));
            end
        end
    end
    
    
end
if exist('accuracy', 'var')
    err = accuracy;
else
    err = mean(err);
end
end