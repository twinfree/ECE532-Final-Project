function [Xtrain,Xtest]=normalizeFeat(Xtrain, Xtest)
Xtest(:, find(sum(Xtest) == 0)) = []; % eliminate columns with only zeros
Xtrain(:, find(sum(Xtrain) == 0)) = []; % eliminate columns with only zeros

Xtest = Xtest - repmat(mean(Xtest), [size(Xtest,1) 1]); %subtract mean from test data
Xtest = Xtest ./ repmat(max(abs(Xtest)), [size(Xtest,1) 1]); %normalize test data s.t it has a max of 1

Xtrain = Xtrain - repmat(mean(Xtrain), [size(Xtrain,1) 1]); % subtract mean
Xtrain = Xtrain ./ repmat(max(abs(Xtrain)), [size(Xtrain,1) 1]); % normalize s.t max value of 1

% SD = std(Xtrain);
% [~,I] = sort(SD, 'descend');
% Xtrain(:, I(401:end)) = []; % eliminate features with low variance across training samples
% Xtest(:, I(401:end)) = []; % perform same reduction for test data
[Xtrain, V] = PCA(Xtrain);

% Xtest = Xtest - repmat(mean(Xtest), [size(Xtest,1) 1]); %subtract mean from test data
% Xtest = Xtest ./ repmat(max(abs(Xtest)), [size(Xtest,1) 1]); %normalize test data s.t it has a max of 1

Xtest = Xtest * V;
end