function [Xout,Vr]=PCA(Xms)
%Xms: mean subtracted matrix
[U,S,V] = svd(Xms,'econ');
s = diag(S);
cums = cumsum(s); cums = cums/max(cums(:));
r = find(cums < .9, 1, 'last');
Vr = V(:,1:r);
Xout = U(:, 1:r) * S(1:r,1:r);
% r = find(s>t, 1, 'last');
% Xout = (U(:,1:r) * S(1:r,1:r) * V(:,1:r)') ;
end