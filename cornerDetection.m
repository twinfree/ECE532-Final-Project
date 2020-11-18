function R = cornerDetection(A, w)
% w = scalar number ; dimension of window, must be odd 
if mod(w,2) == 0
    disp('use odd window')
    return
end
% A = image
HSx = [-1 -2 -1; 0 0 0; 1 2 1];  % Sobol x gradient filter
HSy = [-1 0 1; -2 0 2; -1 0 1];  % Sobol y gradient filter
R = zeros(size(A)); % initialize Harris Score
Apad = padarray(A, [floor(w/2) floor(w/2)]);
window = logical(zeros(size(Apad)));
window(1:w,1:w) = 1;
Ix = conv2(Apad, HSx, 'same');
Iy = conv2(Apad,HSy, 'same');
Ix2 = Ix .^ 2;
Iy2 = Iy .^ 2;
Ixy = Ix .* Iy;

for row = 1:size(A,1)
    for col = 1:size(A,2)
        %Ixt = Ix(logical(window));
        %Iyt = Iy(logical(window));
        M = zeros(2,2);
        M(1) = sum(Ix2(window), 'all');
        M(2) = sum(Ixy(window), 'all'); M(3) = M(2);
        M(4) = sum( Iy2(window), 'all');
        EIG = eig(M);
        R(row,col) = det(M) - .04*trace(M)^2;
        window = circshift(window, [0 1]);
    end
    window = circshift(window, [1 w-1]);
end

end