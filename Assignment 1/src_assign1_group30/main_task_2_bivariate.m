ds = table2array(readtable('../data_assign1_group30/Team21-Team30/team30/bivariate_group30/bivariateData/train.txt'));
rng(0); %setting random number generator seed
%Hyperparameters for polynomial basis linear regression
M = 3;
lambda = 250;

X = double(ds(:,1:end-1));
Y = double(ds(:,end));
N = size(X, 1);
d = size(X, 2);

[train_ind, test_ind, validation_ind] = dividerand(N, 0.7, 0.2, 0.1);

X_train = X(train_ind, :);
X_test = X(test_ind, :);
X_validation = X(validation_ind, :);
Y_train = Y(train_ind, :);
Y_test = Y(test_ind, :);
Y_validation = Y(validation_ind, :);

phi_train = get_design_matrix(X_train, M);
phi_test = get_design_matrix(X_test, M);
phi_validation = get_design_matrix(X_validation, M);
D = size(phi_train, 2);

w = ((phi_train.')*phi_train+lambda*eye(D))\((phi_train.')*Y_train);
train_RMSE = RMSE(Y_train, phi_train*w)
validation_RMSE = RMSE(Y_validation, phi_validation*w)
test_RMSE = RMSE(Y_test, phi_test*w)
%figure; 
%subplot(1,2,1); scatter(Y_train, phi_train*w); xlabel('Target Output'); ylabel('Model Output');
%subplot(1,2,2); scatter(Y_test, phi_test*w); xlabel('Target Output'); ylabel('Model Output');
%figure; scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'b'); hold on; scatter3(X_train(:, 1), X_train(:, 2), phi_train*w, 'r')

%Hyperparameters for linear regression using Gaussian basis functions
k = 30;
var = 750;
lambda_q = 0;%exp(-10);
lambda_t = 0;%exp(-5);

d = size(X, 2);
[means, ~] = k_means(X_train,k);
covs = zeros(d,d,k);
for i=1:k
   covs(:,:,i) = var*eye(d); 
end
phi_train = get_gaussian_design_matrix(X_train, k, means, covs);
phi_tilda = get_gaussian_design_matrix(means, k, means, covs); %k*k

w_q = pinv(phi_train'*phi_train + lambda_q*eye(k))*(phi_train')*Y_train;  %quadratic regularization
w_t = pinv(phi_train'*phi_train + lambda_t*phi_tilda)*(phi_train')*Y_train;   %Tikhonov regularization

phi_test = get_gaussian_design_matrix(X_test, k, means, covs);
phi_validation = get_gaussian_design_matrix(X_validation, k, means, covs);

r_qtrain = RMSE(Y_train, phi_train*w_q)
r_qval = RMSE(Y_validation, phi_validation*w_q)
r_qtest = RMSE(Y_test, phi_test*w_q)
r_ttrain = RMSE(Y_train, phi_train*w_t)
r_tval = RMSE(Y_validation, phi_validation*w_t)
r_ttest = RMSE(Y_test, phi_test*w_t)

figure; scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'b'); 
hold on; 
scatter3(X_train(:, 1), X_train(:, 2), phi_train*w_q, 'r')

figure; scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'b'); 
hold on; 
scatter3(X_train(:, 1), X_train(:, 2), phi_train*w_t, 'r')