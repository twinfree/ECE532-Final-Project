function out=betterMode(v)

M = [];
while true
    [m,f] = mode(v);
    if ~isempty(M)
        if fold ~= f
            break
        end
    end
    fold = f;
    v(v==m) = [];
    M = [M m];
end
if length(M) == 1
    out = M(1);
else
    out = M;
end
end
