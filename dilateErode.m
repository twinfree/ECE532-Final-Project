function Error = dilateErode(A, SE)
% A = image
% SE = binary image - dimension of square structuring element
SE = ones(SE,SE); 
A = imbinarize(A);
B = imdilate(A, SE);
B = imerode(B, SE);
Error = A - B; Error = Error .^ 2;
end