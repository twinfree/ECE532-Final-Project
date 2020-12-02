function votes = predictOvO(Xtest, W)
%Predict
nck = nchoosek(0:9, 2);
confidence = (Xtest * W);
predictions = sign(confidence);
votes = zeros(size(predictions,1),1);
for row = 1:size(predictions,1)
    t = predictions(row,:);
    t2 = [nck(t<0,1);nck(t>0,2)];
    M = betterMode(t2);
    if length(M) == 1
        votes(row) = M;
    else
        %use average distance from decision boundary to settle a tie
        c = confidence(row,:);
        c = [c(t>0) c(t<0)];
        csum = abs(c) * (M' == t2')';
        votes(row) = M(csum == max(csum));
    end
end
end