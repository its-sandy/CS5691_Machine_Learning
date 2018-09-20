rng(0);
ds_train = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train.txt'));
ds_test = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/test.txt'));
ds_validation = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/val.txt'));
k = 350;
var = 1000;
lambda = 0.00001;

X_train = double(ds_train(:,1:end-1));
X_test = double(ds_test(:,1:end-1));
X_validation = double(ds_validation(:,1:end-1));
Y_train = double(ds_train(:,end));
Y_test = double(ds_test(:,end));
Y_validation = double(ds_validation(:,end));
d = size(X_train, 2);

[means, covs, ~] = k_means(X_train,k);
for i=1:k
  covs(:,:,i) = var*eye(d); 
end
phi = get_gaussian_design_matrix(X_train, k, means, covs);  %phi_train
phi_tilda = get_gaussian_design_matrix(means, k, means, covs); %k*k
%phi_tilda = eye(k);

w = pinv(phi'*phi + lambda*phi_tilda)*(phi')*Y_train;

phi_test = get_gaussian_design_matrix(X_test, k, means, covs);
phi_validation = get_gaussian_design_matrix(X_validation, k, means, covs);

RMSE(Y_train, phi*w)
RMSE(Y_validation, phi_validation*w)
RMSE(Y_test, phi_test*w)

figure('Name', sprintf('Variation w.r.t Size of Training Data'), 'NumberTitle', 'off');
subplot(1,2,1);
scatter(Y_train, phi*w);
xlabel('target output');
ylabel('model output');
title(sprintf('training data'));
subplot(1,2,2);
scatter(Y_test, phi_test*w);
xlabel('target output');
ylabel('model output');
title(sprintf('test data'));

%get_roughness(w, phi_tilda, lambda)
%scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'b');
