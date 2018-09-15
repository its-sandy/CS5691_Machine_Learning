ds = dataset('File','Team21-Team30/team30/bivariate_group30/bivariateData/train.txt', 'Delimiter', ' ', 'ReadVarNames', false);

M = 2;
lambda = 0;

X = double(ds(:,1:end-1));
Y = double(ds(:,end));
N = size(X, 1);
d = size(X, 2);
phi = get_design_matrix(X, M);
D = size(phi, 2);

w = ((phi.')*phi+lambda*eye(D))\((phi.')*Y);
scatter3(X(:, 1), X(:, 2), Y, 'b'); hold on; scatter3(X(:, 1), X(:, 2), phi*w, 'r')