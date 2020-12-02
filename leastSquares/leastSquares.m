function w = leastSquares(A, y, str, lambda)
if strcmp(str, '')
    if size(A,2) <= size(A,1)
        w = ((A' * A) \ A') * y;
        %w = pinv(A' * A, 1e-5) * A' * y;
        %w = LSSGD(A, y, 1);
    else
        w = (A' \ (A * A')) * y;
        
    end
elseif strcmp(str, 'ridge')
    if size(A,2) <= size(A,1)
        w = (A' * A + lambda*eye(size(A,2))) \ A' * y;
    else
        w = A' * (A * A' + lambda * eye(size(A,1))) \ y;
    end
elseif strcmp(str, 'lasso')
    %w = lassoGD(A, y, lambda);
    w = lassoSGD(A,y,lambda);
else
    disp('error')
    w = [];
    return
end