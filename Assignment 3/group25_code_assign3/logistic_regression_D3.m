rng(1);

pg = 0; %0 for polynomial basis, 1 for Gaussian basis
k = 250;  %Number of clusters for Gaussian basis
%k = 250 seems to work well
M = 2;  %Model complexity for polynomial basis
eta = 1e-5; %1e-5 for poly, 5e-1 for Gaussian

files = dir('../group25_data_assign3/image_features/forest/*.jpg_color_edh_entropy');
ctr = 1;
X_1 = zeros(size(files, 1), 36, 23);
for file = files'
    X_1(ctr, :, :) = table2array(readtable(fullfile('..','group25_data_assign3','image_features','forest',file.name), 'FileType', 'text', 'Delimiter', ' '));
    fprintf("%d\n", ctr);
    ctr = ctr+1;
end
[train_ind, test_ind, val_ind] = dividerand(size(X_1, 1), 0.7, 0.2, 0.1);
X_train1 = reshape(X_1(train_ind, :, :), [size(train_ind, 2)*36, 23]);
X_test1 = X_1(test_ind, :, :);
X_val1 = X_1(val_ind, :, :);

files = dir('../group25_data_assign3/image_features/highway/*.jpg_color_edh_entropy');
ctr = 1;
X_2 = zeros(size(files, 1), 36, 23);
for file = files'
    X_2(ctr, :, :) = table2array(readtable(fullfile('..','group25_data_assign3','image_features','highway',file.name), 'FileType', 'text', 'Delimiter', ' '));
    fprintf("%d\n", ctr);
    ctr = ctr+1;
end
[train_ind, test_ind, val_ind] = dividerand(size(X_2, 1), 0.7, 0.2, 0.1);
X_train2 = reshape(X_2(train_ind, :, :), [size(train_ind, 2)*36, 23]);
X_test2 = X_2(test_ind, :, :);
X_val2 = X_2(val_ind, :, :);

files = dir('../group25_data_assign3/image_features/insidecity/*.jpg_color_edh_entropy');
ctr = 1;
X_3 = zeros(size(files, 1), 36, 23);
for file = files'
    X_3(ctr, :, :) = table2array(readtable(fullfile('..','group25_data_assign3','image_features','insidecity',file.name), 'FileType', 'text', 'Delimiter', ' '));
    fprintf("%d\n", ctr);
    ctr = ctr+1;
end
[train_ind, test_ind, val_ind] = dividerand(size(X_3, 1), 0.7, 0.2, 0.1);
X_train3 = reshape(X_3(train_ind, :, :), [size(train_ind, 2)*36, 23]);
X_test3 = X_3(test_ind, :, :);
X_val3 = X_3(val_ind, :, :);

if pg==0
    phi_train1 = get_design_matrix(X_train1, M);
    
    phi_train2 = get_design_matrix(X_train2, M);
    
    phi_train3 = get_design_matrix(X_train3, M);
else
    X_train = [X_train1; X_train2; X_train3];
    [means, covs, R] = k_means(X_train, k);
    
    phi_train1 = get_gaussian_design_matrix(X_train1, k, means, covs);
    
    phi_train2 = get_gaussian_design_matrix(X_train2, k, means, covs);
    
    phi_train3 = get_gaussian_design_matrix(X_train3, k, means, covs);
end

rangeval = max([phi_train1; phi_train2; phi_train3]) - min([phi_train1; phi_train2; phi_train3]);
minval = min([phi_train1; phi_train2; phi_train3]);

if pg==0
    rangeval(1) = 1;
    minval(1) = 0;
end
phi_train1 = (phi_train1-minval)./rangeval;
phi_train2 = (phi_train2-minval)./rangeval;
phi_train3 = (phi_train3-minval)./rangeval;
    
w = randn(3, size(phi_train1, 2));
wold = zeros(3, size(phi_train1, 2));

ctr = 0;
num_iters = 5000;
tot = 0;
preverror = 0;

while ctr<num_iters
    wold = w;
    yn_1 = exp(w*phi_train1');
    yn_1 = yn_1./sum(yn_1, 1);
    ytn_1 = yn_1 - [1; 0; 0];
    yn_2 = exp(w*phi_train2');
    yn_2 = yn_2./sum(yn_2, 1);
    ytn_2 = yn_2 - [0; 1; 0];
    yn_3 = exp(w*phi_train3');
    yn_3 = yn_3./sum(yn_3, 1);
    ytn_3 = yn_3 - [0; 0; 1];
    w = w - eta*(ytn_1*phi_train1+ytn_2*phi_train2+ytn_3*phi_train3);
    currerror = -sum(log(yn_1(1, :)), 2)-sum(log(yn_2(2, :)), 2)-sum(log(yn_3(3, :)), 2);
    ctr = ctr+1;
    fprintf('Iteration %d over: %f\n', ctr, currerror);
    if currerror>preverror
        eta = 0.995*eta;
    else
        eta = 1.001*eta;
    end
    preverror = currerror;
end

conf_matrix_val = zeros(3, 3);
conf_matrix_test = zeros(3, 3);

for i = 1:size(X_val1, 1)
    curr = squeeze(X_val1(i, :, :));
    if pg==0
        currphi = get_design_matrix(curr, M);
    else
        currphi = get_gaussian_design_matrix(curr, k, means, covs);
    end
    currphi = (currphi-minval)./rangeval;
    p = exp(w*currphi');
    p = sum(log(p./sum(p, 1)), 2);
    [~, predicted_class] = max(p);
    conf_matrix_val(1, predicted_class) = conf_matrix_val(1, predicted_class)+1;
end

for i = 1:size(X_val2, 1)
    curr = squeeze(X_val2(i, :, :));
    if pg==0
        currphi = get_design_matrix(curr, M);
    else
        currphi = get_gaussian_design_matrix(curr, k, means, covs);
    end
    currphi = (currphi-minval)./rangeval;
    p = exp(w*currphi');
    p = sum(log(p./sum(p, 1)), 2);
    [~, predicted_class] = max(p);
    conf_matrix_val(2, predicted_class) = conf_matrix_val(2, predicted_class)+1;
end

for i = 1:size(X_val3, 1)
    curr = squeeze(X_val3(i, :, :));
    if pg==0
        currphi = get_design_matrix(curr, M);
    else
        currphi = get_gaussian_design_matrix(curr, k, means, covs);
    end
    currphi = (currphi-minval)./rangeval;
    p = exp(w*currphi');
    p = sum(log(p./sum(p, 1)), 2);
    [~, predicted_class] = max(p);
    conf_matrix_val(3, predicted_class) = conf_matrix_val(3, predicted_class)+1;
end

for i = 1:size(X_test1, 1)
    curr = squeeze(X_test1(i, :, :));
    if pg==0
        currphi = get_design_matrix(curr, M);
    else
        currphi = get_gaussian_design_matrix(curr, k, means, covs);
    end
    currphi = (currphi-minval)./rangeval;
    p = exp(w*currphi');
    p = sum(log(p./sum(p, 1)), 2);
    [~, predicted_class] = max(p);
    conf_matrix_test(1, predicted_class) = conf_matrix_test(1, predicted_class)+1;
end

for i = 1:size(X_test2, 1)
    curr = squeeze(X_test2(i, :, :));
    if pg==0
        currphi = get_design_matrix(curr, M);
    else
        currphi = get_gaussian_design_matrix(curr, k, means, covs);
    end
    currphi = (currphi-minval)./rangeval;
    p = exp(w*currphi');
    p = sum(log(p./sum(p, 1)), 2);
    [~, predicted_class] = max(p);
    conf_matrix_test(2, predicted_class) = conf_matrix_test(2, predicted_class)+1;
end

for i = 1:size(X_test3, 1)
    curr = squeeze(X_test3(i, :, :));
    if pg==0
        currphi = get_design_matrix(curr, M);
    else
        currphi = get_gaussian_design_matrix(curr, k, means, covs);
    end
    currphi = (currphi-minval)./rangeval;
    p = exp(w*currphi');
    p = sum(log(p./sum(p, 1)), 2);
    [~, predicted_class] = max(p);
    conf_matrix_test(3, predicted_class) = conf_matrix_test(3, predicted_class)+1;
end