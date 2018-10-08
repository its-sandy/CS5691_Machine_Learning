ds = table2array(readtable('../data_assign1_group30/Team21-Team30/team30/Housing_Dataset/housing-data.txt'));
rng(0);

%Hyperparameters for linear regression using Gaussian basis functions
k = 350;
var = 55000;
lambda = 1e-5;

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
[means, ~] = k_means(X_train,k);
covs = zeros(d,d,k);
for i=1:k
   covs(:,:,i) = var*eye(d); 
end
phi_train = get_gaussian_design_matrix(X_train, k, means, covs);
phi_tilda = get_gaussian_design_matrix(means, k, means, covs); %k*k

w_q = pinv(phi_train'*phi_train + lambda*eye(k))*(phi_train')*Y_train;
w_t = pinv(phi_train'*phi_train + lambda*phi_tilda)*(phi_train')*Y_train;

phi_test = get_gaussian_design_matrix(X_test, k, means, covs);
phi_validation = get_gaussian_design_matrix(X_validation, k, means, covs);

r_qtrain = RMSE(Y_train, phi_train*w_q)
r_qval = RMSE(Y_validation, phi_validation*w_q)
r_qtest = RMSE(Y_test, phi_test*w_q)
r_ttrain = RMSE(Y_train, phi_train*w_t)
r_tval = RMSE(Y_validation, phi_validation*w_t)
r_ttest = RMSE(Y_test, phi_test*w_t)