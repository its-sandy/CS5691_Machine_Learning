Q1 = 1;
Q2 = 1;
Q3 = 1;

XY = table2array(readtable('..\data_assign2_group25\wine.data.txt'));

X_1 = XY(XY(:, 1)==1, 2:end);
X_2 = XY(XY(:, 1)==2, 2:end);
X_3 = XY(XY(:, 1)==3, 2:end);

[train_ind, test_ind, val_ind] = dividerand(size(X_1, 1), 0.7, 0.2, 0.1);

X_train1 = X_1(train_ind, :);
X_test1 = X_1(test_ind, :);
X_val1 = X_1(val_ind, :);

[train_ind, test_ind, val_ind] = dividerand(size(X_2, 1), 0.7, 0.2, 0.1);

X_train2 = X_2(train_ind, :);
X_test2 = X_2(test_ind, :);
X_val2 = X_2(val_ind, :);

[train_ind, test_ind, val_ind] = dividerand(size(X_3, 1), 0.7, 0.2, 0.1);

X_train3 = X_3(train_ind, :);
X_test3 = X_3(test_ind, :);
X_val3 = X_3(val_ind, :);

ccprob = [size(X_train1, 1), size(X_train2, 1), size(X_train3, 1)];
ccprob = ccprob/sum(ccprob);

conf_matrix_val = zeros(3, 3);
conf_matrix_test = zeros(3, 3);

[w1, mu1, C1] = trainGMM(X_train1, Q1, 0);
[w2, mu2, C2] = trainGMM(X_train2, Q2, 0);
[w3, mu3, C3] = trainGMM(X_train3, Q3, 0);

p1 = zeros(size(X_val1, 1), 1);
p2 = zeros(size(X_val1, 1), 1);
p3 = zeros(size(X_val1, 1), 1);

for i = 1:size(X_val1, 1)
    p1(i) = ccprob(1)*evalGMM(X_val1(i, :), w1, mu1, C1);
    p2(i) = ccprob(2)*evalGMM(X_val1(i, :), w2, mu2, C2);
    p3(i) = ccprob(3)*evalGMM(X_val1(i, :), w3, mu3, C3);
    
    if max([p1(i) p2(i) p3(i)])==p1(i)
       conf_matrix_val(1, 1) = conf_matrix_val(1, 1)+1;
    elseif max([p1(i) p2(i) p3(i)])==p2(i)
       conf_matrix_val(1, 2) = conf_matrix_val(1, 2)+1;
    elseif max([p1(i) p2(i) p3(i)])==p3(i)
       conf_matrix_val(1, 3) = conf_matrix_val(1, 3)+1;
    end
end

p1 = zeros(size(X_val2, 1), 1);
p2 = zeros(size(X_val2, 1), 1);
p3 = zeros(size(X_val2, 1), 1);

for i = 1:size(X_val2, 1)
    p1(i) = ccprob(1)*evalGMM(X_val2(i, :), w1, mu1, C1);
    p2(i) = ccprob(2)*evalGMM(X_val2(i, :), w2, mu2, C2);
    p3(i) = ccprob(3)*evalGMM(X_val2(i, :), w3, mu3, C3);
    
    if max([p1(i) p2(i) p3(i)])==p1(i)
       conf_matrix_val(2, 1) = conf_matrix_val(2, 1)+1;
    elseif max([p1(i) p2(i) p3(i)])==p2(i)
       conf_matrix_val(2, 2) = conf_matrix_val(2, 2)+1;
    elseif max([p1(i) p2(i) p3(i)])==p3(i)
       conf_matrix_val(2, 3) = conf_matrix_val(2, 3)+1;
    end
end

p1 = zeros(size(X_val3, 1), 1);
p2 = zeros(size(X_val3, 1), 1);
p3 = zeros(size(X_val3, 1), 1);

for i = 1:size(X_val3, 1)
    p1(i) = ccprob(1)*evalGMM(X_val3(i, :), w1, mu1, C1);
    p2(i) = ccprob(2)*evalGMM(X_val3(i, :), w2, mu2, C2);
    p3(i) = ccprob(3)*evalGMM(X_val3(i, :), w3, mu3, C3);
    
    if max([p1(i) p2(i) p3(i)])==p1(i)
       conf_matrix_val(3, 1) = conf_matrix_val(3, 1)+1;
    elseif max([p1(i) p2(i) p3(i)])==p2(i)
       conf_matrix_val(3, 2) = conf_matrix_val(3, 2)+1;
    elseif max([p1(i) p2(i) p3(i)])==p3(i)
       conf_matrix_val(3, 3) = conf_matrix_val(3, 3)+1;
    end
end

p1 = zeros(size(X_test1, 1), 1);
p2 = zeros(size(X_test1, 1), 1);
p3 = zeros(size(X_test1, 1), 1);

for i = 1:size(X_test1, 1)
    p1(i) = ccprob(1)*evalGMM(X_test1(i, :), w1, mu1, C1);
    p2(i) = ccprob(2)*evalGMM(X_test1(i, :), w2, mu2, C2);
    p3(i) = ccprob(3)*evalGMM(X_test1(i, :), w3, mu3, C3);
    
    if max([p1(i) p2(i) p3(i)])==p1(i)
       conf_matrix_test(1, 1) = conf_matrix_test(1, 1)+1;
    elseif max([p1(i) p2(i) p3(i)])==p2(i)
       conf_matrix_test(1, 2) = conf_matrix_test(1, 2)+1;
    elseif max([p1(i) p2(i) p3(i)])==p3(i)
       conf_matrix_test(1, 3) = conf_matrix_test(1, 3)+1;
    end
end

p1 = zeros(size(X_test2, 1), 1);
p2 = zeros(size(X_test2, 1), 1);
p3 = zeros(size(X_test2, 1), 1);

for i = 1:size(X_test2, 1)
    p1(i) = ccprob(1)*evalGMM(X_test2(i, :), w1, mu1, C1);
    p2(i) = ccprob(2)*evalGMM(X_test2(i, :), w2, mu2, C2);
    p3(i) = ccprob(3)*evalGMM(X_test2(i, :), w3, mu3, C3);
    
    if max([p1(i) p2(i) p3(i)])==p1(i)
       conf_matrix_test(2, 1) = conf_matrix_test(2, 1)+1;
    elseif max([p1(i) p2(i) p3(i)])==p2(i)
       conf_matrix_test(2, 2) = conf_matrix_test(2, 2)+1;
    elseif max([p1(i) p2(i) p3(i)])==p3(i)
       conf_matrix_test(2, 3) = conf_matrix_test(2, 3)+1;
    end
end

p1 = zeros(size(X_test1, 1), 1);
p2 = zeros(size(X_test1, 1), 1);
p3 = zeros(size(X_test1, 1), 1);

for i = 1:size(X_test3, 1)
    p1(i) = ccprob(1)*evalGMM(X_test3(i, :), w1, mu1, C1);
    p2(i) = ccprob(2)*evalGMM(X_test3(i, :), w2, mu2, C2);
    p3(i) = ccprob(3)*evalGMM(X_test3(i, :), w3, mu3, C3);
    
    if max([p1(i) p2(i) p3(i)])==p1(i)
       conf_matrix_test(3, 1) = conf_matrix_test(3, 1)+1;
    elseif max([p1(i) p2(i) p3(i)])==p2(i)
       conf_matrix_test(3, 2) = conf_matrix_test(3, 2)+1;
    elseif max([p1(i) p2(i) p3(i)])==p3(i)
       conf_matrix_test(3, 3) = conf_matrix_test(3, 3)+1;
    end
end

fprintf("Validation accuracy: %f%%\n", sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
fprintf("Test accuracy: %f%%\n", sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));