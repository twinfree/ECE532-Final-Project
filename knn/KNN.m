function predictedLabels = KNN(Xtrain, ytrain,xtest, k, varargin)
%xtest: matrix of feature vectors of data points to be classified
%Xtrain: matrix of data points with known labels
%ytrain: labels corresponding to the rows of Xtrain
%k: number of nearest neighbors contributing to the vote
%varargin: if varargin is not empty, this function computes KNN with a
%weight function of 1/d
predictedLabels = zeros(1, size(xtest,1));
for j = 1:size(xtest,1)
    d = (Xtrain - repmat(xtest(j,:), [size(Xtrain,1) 1])) .^ 2;
    d = sum(d,2) ;%.^ .5;
    [~, I] = sort(d);
    if isempty(varargin)
        votes = ytrain(I(1:k));
        [modeV, f1] = mode(votes);
        votes(votes == modeV) = [];
        [~,f2] = mode(votes);
        if f1 ~= f2
            predictedLabels(j) = modeV;
        else             %matlabs mode function returns the lowest value if there is a tie. This step ensures the nearest label is used in a tie.
            votes = ytrain(I(1:k))';
            weight = 1 ./ d(I(1:k)); weight = weight / max(weight(:));
            u = unique(votes, 'stable');
            t = u' == votes;
            t = sum(t .* repmat(weight', [length(u) 1]), 2);
            vote = u(t == max(t));
            predictedLabels(j) = vote;
        end
    else
        votes = ytrain(I(1:k))';
        weight = 1 ./ d(I(1:k)); weight = weight / max(weight(:));
        u = unique(votes, 'stable');
        t = u' == votes;
        t = sum(t .* repmat(weight', [length(u) 1]), 2);
        vote = u(t == max(t));
        predictedLabels(j) = vote;
    end
end
end