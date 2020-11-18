W = cell(1,9);
for i = 0:8
    iX = X(1:50000,:);
    ilabel = label(1:50000);
    iX(find(ilabel<i),:) = [];
    ilabel(ilabel<i) = [];
    ilabel = double(ilabel == i);
    ilabel(ilabel == 0) = -1;
    
    W{i+1} = leastSquares(iX, ilabel, '', 0);
end

%%
%evaluate on holdout data
predictedLabels = zeros(1, 10000);
for im = 50001:60000
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
            