ds = dataset('File','Team21-Team30/team30/bivariate_group30/bivariateData/train.txt', 'Delimiter', ' ', 'VarNames', {'x1', 'x2', 'y'}, 'ReadVarNames', false);

k = 40;

X = double(ds(:,1:end-1));
Y = double(ds(:,end));
N = size(X, 1);
d = size(X, 2);
phi = get_gaussian_design_matrix(X, k);

w = pinv(phi)*Y;
scatter3(X(:, 1), X(:, 2), Y, 'b'); hold on; scatter3(X(:, 1), X(:, 2), phi*w, 'r')
