Q1 = 2;
Q2 = 2;
Q3 = 2;
Q4 = 2;
Q5 = 2;

n1 = 149;
X_1 = cell(n1, 1);

for i=1:n1
    try
    X_1{i} = table2array(readtable(fullfile('..','data_assign2_group25','features_SURF','feats_surf','158.penguin',sprintf('158_%04d.jpg.txt', i)))); 
    catch
    end
end

X_1 = X_1(~cellfun('isempty', X_1));
n1 = size(X_1, 1);

[train_ind1, test_ind1, val_ind1] = dividerand(n1, 0.7, 0.2, 0.1);

X_train1 = cell2mat(X_1(train_ind1));

n2 = 800;
X_2 = cell(n2, 1);

for i=1:n2
    try
    X_2{i} = table2array(readtable(fullfile('..','data_assign2_group25','features_SURF','feats_surf','251.airplanes-101',sprintf('251_%04d.jpg.txt', i)))); 
    catch
    end
end

X_2 = X_2(~cellfun('isempty', X_2));
n2 = size(X_2, 1);

[train_ind2, test_ind2, val_ind2] = dividerand(n2, 0.7, 0.2, 0.1);

X_train2 = cell2mat(X_2(train_ind2));

n3 = 358;
X_3 = cell(n3, 1);

for i=1:n3
    try
    X_3{i} = table2array(readtable(fullfile('..','data_assign2_group25','features_SURF','feats_surf','232.t-shirt',sprintf('232_%04d.jpg.txt', i)))); 
    catch
    end
end

X_3 = X_3(~cellfun('isempty', X_3));
n3 = size(X_3, 1);

[train_ind3, test_ind3, val_ind3] = dividerand(n3, 0.7, 0.2, 0.1);

X_train3 = cell2mat(X_3(train_ind3));

n4 = 190;
X_4 = cell(n4, 1);

for i=1:n4
    try
    X_4{i} = table2array(readtable(fullfile('..','data_assign2_group25','features_SURF','feats_surf','132.light-house',sprintf('132_%04d.jpg.txt', i)))); 
    catch
    end
end

X_4 = X_4(~cellfun('isempty', X_4));
n4 = size(X_4, 1);

[train_ind4, test_ind4, val_ind4] = dividerand(n4, 0.7, 0.2, 0.1);

X_train4 = cell2mat(X_4(train_ind4));

n5 = 192;
X_5 = cell(n5, 1);

for i=1:n5
    try
    X_5{i} = table2array(readtable(fullfile('..','data_assign2_group25','features_SURF','feats_surf','138.mattress',sprintf('138_%04d.jpg.txt', i)))); 
    catch
    end
end

X_5 = X_5(~cellfun('isempty', X_5));
n5 = size(X_5, 1);

[train_ind5, test_ind5, val_ind5] = dividerand(n5, 0.7, 0.2, 0.1);

X_train5 = cell2mat(X_5(train_ind5));

%ccprob = [size(X_train1, 1), size(X_train2, 1), size(X_train3, 1), size(X_train4, 1), size(X_train5, 1)];
ccprob = [n1 n2 n3 n4 n5];
ccprob = ccprob/sum(ccprob);

conf_matrix_val = zeros(5, 5);
conf_matrix_test = zeros(5, 5);

[w1, mu1, C1] = trainGMM(X_train1, Q1, 0);
[w2, mu2, C2] = trainGMM(X_train2, Q2, 0);
[w3, mu3, C3] = trainGMM(X_train3, Q3, 0);
[w4, mu4, C4] = trainGMM(X_train4, Q4, 0);
[w5, mu5, C5] = trainGMM(X_train5, Q5, 0);

for i = val_ind1
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_1{i},1)
         p1 = p1 + log(evalGMM(X_1{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_1{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_1{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_1{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_1{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_val(1, 1) = conf_matrix_val(1, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_val(1, 2) = conf_matrix_val(1, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_val(1, 3) = conf_matrix_val(1, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_val(1, 4) = conf_matrix_val(1, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_val(1, 5) = conf_matrix_val(1, 5)+1;
    end
end

for i = val_ind2
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_2{i},1)
         p1 = p1 + log(evalGMM(X_2{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_2{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_2{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_2{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_2{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_val(2, 1) = conf_matrix_val(2, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_val(2, 2) = conf_matrix_val(2, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_val(2, 3) = conf_matrix_val(2, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_val(2, 4) = conf_matrix_val(2, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_val(2, 5) = conf_matrix_val(2, 5)+1;
    end
end

for i = val_ind3
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_3{i},1)
         p1 = p1 + log(evalGMM(X_3{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_3{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_3{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_3{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_3{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_val(3, 1) = conf_matrix_val(3, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_val(3, 2) = conf_matrix_val(3, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_val(3, 3) = conf_matrix_val(3, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_val(3, 4) = conf_matrix_val(3, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_val(3, 5) = conf_matrix_val(3, 5)+1;
    end
end

for i = val_ind4
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_4{i},1)
         p1 = p1 + log(evalGMM(X_4{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_4{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_4{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_4{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_4{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_val(4, 1) = conf_matrix_val(4, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_val(4, 2) = conf_matrix_val(4, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_val(4, 3) = conf_matrix_val(4, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_val(4, 4) = conf_matrix_val(4, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_val(4, 5) = conf_matrix_val(4, 5)+1;
    end
end

for i = val_ind5
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_5{i},1)
         p1 = p1 + log(evalGMM(X_5{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_5{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_5{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_5{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_5{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_val(5, 1) = conf_matrix_val(5, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_val(5, 2) = conf_matrix_val(5, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_val(5, 3) = conf_matrix_val(5, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_val(5, 4) = conf_matrix_val(5, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_val(5, 5) = conf_matrix_val(5, 5)+1;
    end
end

for i = test_ind1
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_1{i},1)
         p1 = p1 + log(evalGMM(X_1{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_1{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_1{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_1{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_1{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_test(1, 1) = conf_matrix_test(1, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_test(1, 2) = conf_matrix_test(1, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_test(1, 3) = conf_matrix_test(1, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_test(1, 4) = conf_matrix_test(1, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_test(1, 5) = conf_matrix_test(1, 5)+1;
    end
end

for i = test_ind2
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_2{i},1)
         p1 = p1 + log(evalGMM(X_2{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_2{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_2{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_2{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_2{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_test(2, 1) = conf_matrix_test(2, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_test(2, 2) = conf_matrix_test(2, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_test(2, 3) = conf_matrix_test(2, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_test(2, 4) = conf_matrix_test(2, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_test(2, 5) = conf_matrix_test(2, 5)+1;
    end
end

for i = test_ind3
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_3{i},1)
         p1 = p1 + log(evalGMM(X_3{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_3{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_3{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_3{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_3{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_test(3, 1) = conf_matrix_test(3, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_test(3, 2) = conf_matrix_test(3, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_test(3, 3) = conf_matrix_test(3, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_test(3, 4) = conf_matrix_test(3, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_test(3, 5) = conf_matrix_test(3, 5)+1;
    end
end

for i = test_ind4
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_4{i},1)
         p1 = p1 + log(evalGMM(X_4{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_4{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_4{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_4{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_4{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_test(4, 1) = conf_matrix_test(4, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_test(4, 2) = conf_matrix_test(4, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_test(4, 3) = conf_matrix_test(4, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_test(4, 4) = conf_matrix_test(4, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_test(4, 5) = conf_matrix_test(4, 5)+1;
    end
end

for i = test_ind5
    p1 = log(ccprob(1));
    p2 = log(ccprob(2));
    p3 = log(ccprob(3));
    p4 = log(ccprob(4));
    p5 = log(ccprob(5));
    
    for j = 1:size(X_5{i},1)
         p1 = p1 + log(evalGMM(X_5{i}(j, :), w1, mu1, C1));
         p2 = p2 + log(evalGMM(X_5{i}(j, :), w2, mu2, C2));
         p3 = p3 + log(evalGMM(X_5{i}(j, :), w3, mu3, C3));
         p4 = p4 + log(evalGMM(X_5{i}(j, :), w4, mu4, C4));
         p5 = p5 + log(evalGMM(X_5{i}(j, :), w5, mu5, C5));
    end
    
    if max([p1 p2 p3 p4 p5])==p1
       conf_matrix_test(5, 1) = conf_matrix_test(5, 1)+1;
    elseif max([p1 p2 p3 p4 p5])==p2
       conf_matrix_test(5, 2) = conf_matrix_test(5, 2)+1;
    elseif max([p1 p2 p3 p4 p5])==p3
       conf_matrix_test(5, 3) = conf_matrix_test(5, 3)+1;
    elseif max([p1 p2 p3 p4 p5])==p4
       conf_matrix_test(5, 4) = conf_matrix_test(5, 4)+1;
    elseif max([p1 p2 p3 p4 p5])==p5
       conf_matrix_test(5, 5) = conf_matrix_test(5, 5)+1;
    end
end

fprintf("Validation accuracy: %f%%\n", sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
fprintf("Test accuracy: %f%%\n", sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));