function W = trainOvO(Xtrain, label,str,lams)
%Train 45 sets of weights for each pair
nck = nchoosek(0:9, 2);
W = zeros(size(Xtrain,2), size(nck,1));
for i = 1:size(nck,1)
    n1 = nck(i,1);
    n2 = nck(i,2);
    W(:,i) = OneVsOne(n1, n2, Xtrain, label, str,lams(i));
end
end