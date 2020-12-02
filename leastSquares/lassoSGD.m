function w = lassoSGD(X, y, lambda)
% A = feature matrix
% y = label array
% lambda = lasso coefficient
max_iter = 1e5;
tol = 1e-8;
%tau = 1 / norm(X,2)^2;
tau = 1e-4;
N = size(X,1);
w = zeros(size(X,2),1);
for iter = 1:max_iter
    i = randperm(size(X,1),1);
    Xi = X(i,:);

    yi = y(i);
    wold = w;
    w = w - tau*(-2*(yi - Xi*w)*Xi' + lambda/N * sign(w));
    if norm(w-wold) < tol
        break
    end
end
end