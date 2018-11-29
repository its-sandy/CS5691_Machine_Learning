function phi = get_design_matrix(X, M)

if M>2
    N = size(X, 1);
    d = size(X, 2);
    D = nchoosek(M+d, d);
    phi = zeros(N, D);

    temp = num2cell(repmat((0:M)',1,d), 1);

    ncart = cell(1, d);
    [ncart{:}] = ndgrid(temp{:});
    ncart = cell2mat(cellfun(@(x) x(:), ncart, 'UniformOutput', false));
    ncart(sum(ncart, 2)>M, :) = [];

    for i=1:D
        phi(:,i) = prod(X.^ncart(i,:), 2);
    end
elseif M==2
    N = size(X, 1);
    d = size(X, 2);
    D = nchoosek(M+d, d);
    phi = [ones(N, 1), X, zeros(N, D-d-1)];
    ctr = d+2;
    for i=2:d+1
        for j=i:d+1
            phi(:, ctr) = phi(:, i).*phi(:, j);
            ctr = ctr+1;
        end
    end
else
    phi = [ones(size(X, 1), 1), X];
end
end