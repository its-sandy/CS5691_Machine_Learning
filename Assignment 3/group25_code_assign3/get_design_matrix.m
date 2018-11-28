function phi = get_design_matrix(X, M)

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