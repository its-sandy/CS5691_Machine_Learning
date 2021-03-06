%List of hyperparameters
N = 100;
M = 9;
lambda = exp(-20);

fprintf('N=%d, M=%d, \x03bb=%e:\n', N, M, lambda);

rng(0);
X = sort(rand(N, 1));
Y = exp(sin(2*pi*X))+0.1*randn(N,1);

[train_ind, test_ind, validation_ind] = dividerand(N, 0.7, 0.2, 0.1);

X_train = X(train_ind, :);
X_test = X(test_ind, :);
X_validation = X(validation_ind, :);
Y_train = Y(train_ind, :);
Y_test = Y(test_ind, :);
Y_validation = Y(validation_ind, :);

plot(X_train, Y_train, 'bo', 'LineWidth', 1); hold on

phi_train = zeros(size(X_train,1), M+1);
phi_validation = zeros(size(X_validation,1), M+1);
phi_test = zeros(size(X_test,1), M+1);
a = zeros(M+1, M+1);

for i=1:M+1
    phi_train(:,i) = X_train.^(i-1);
    for j=1:M+1
        a(i,j) = sum(X_train.^(i+j-2));
    end
    phi_validation(:,i) = X_validation.^(i-1);
    phi_test(:,i) = X_test.^(i-1);
end

c = phi_train'*Y_train;
a = a+lambda*eye(M+1);
w = inv(a)*c;
plot(0:0.01:1, polyval(flipud(w), 0:0.01:1), 'r-', 'LineWidth', 1); hold on
plot(0:0.01:1, exp(sin(2*pi*(0:0.01:1))), 'g-', 'LineWidth', 1);

train_RMSE = sqrt(mean((Y_train-phi_train*w).^2))
validation_RMSE = sqrt(mean((Y_validation-phi_validation*w).^2))
test_RMSE = sqrt(mean((Y_test-phi_test*w).^2))