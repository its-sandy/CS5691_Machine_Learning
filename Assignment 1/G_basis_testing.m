rng(0);
ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train.txt'));
%ds = ds(1:100,:);
k = 50;
var = 500;
lambda = 0.001;

N = size(ds, 1);

[train_ind, test_ind, validation_ind] = dividerand(N, 0.7, 0.2, 0.1);

X = double(ds(:,1:end-1));
X_train = X(train_ind, :);
X_test = X(test_ind, :);
X_validation = X(validation_ind, :);
Y = double(ds(:,end));
Y_train = Y(train_ind, :);
Y_test = Y(test_ind, :);
Y_validation = Y(validation_ind, :);
d = size(X, 2);
[means, covs, ~] = k_means(X_train,k);
for i=1:k
   covs(:,:,i) = var*eye(d); 
end
phi = get_gaussian_design_matrix(X_train, k, means, covs);
phi_tilda = get_gaussian_design_matrix(means, k, means, covs); %k*k

w = pinv(phi'*phi + lambda*phi_tilda)*(phi')*Y_train;

phi_test = get_gaussian_design_matrix(X_test, k, means, covs);
phi_validation = get_gaussian_design_matrix(X_validation, k, means, covs);

RMSE(Y_test, phi_test*w)
RMSE(Y_validation, phi_validation*w)

get_roughness(w, phi_tilda, lambda)

figure('Name', sprintf('Gaussian Curve Fitting %d',N), 'NumberTitle', 'off');
subplot(1,2,1);
title(sprintf("1"));

scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_gaussian_design_matrix(Xsurf, k, means, covs);
surfplot = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot.EdgeColor = 'none';


subplot(1,2,2);
title(sprintf("2"));

scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_gaussian_design_matrix(Xsurf, k, means, covs);
surfplot = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot.EdgeColor = 'none';