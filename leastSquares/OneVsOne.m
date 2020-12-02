function w = OneVsOne(n1,n2, Xtrain, y, str, lam)
idx = (y == n1) | (y == n2);
X1v1 = Xtrain(idx, :);
y1v1 = y(idx);
y1v1(y1v1 == n1) = -1;
y1v1(y1v1 == n2) = 1;
%PCA?
w = leastSquares(X1v1, y1v1, str, lam);
end