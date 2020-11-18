function predictedLabels = LeastSquaresTree(X, label, train,test)
W = cell(1,9);
for i = 0:8
    iX = X(train,:);
    ilabel = label(train);
    iX(find(ilabel<i),:) = [];
    ilabel(ilabel<i) = [];
    ilabel = double(ilabel == i);
    ilabel(ilabel == 0) = -1;
    
    W{i+1} = leastSquares(iX, ilabel, '', 0);
end

%%
%evaluate on holdout data
predictedLabels = zeros(1, length(test));
for im = test
    for i = 1:9
        prediction = sign(X(im,:)*W{i});
        if prediction == 1
            predictedLabels(im) = i-1;
            break
        elseif i==9
            predictedLabels(im) = 9;
        end
    end
end
end
            
