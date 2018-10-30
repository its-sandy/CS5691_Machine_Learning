k = 10;

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

conf_matrix_val = zeros(5, 5);
conf_matrix_test = zeros(5, 5);

for i = val_ind1
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_1{i},1)
         p1 = p1 - log(getKNNRadius(X_1{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_1{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_1{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_1{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_1{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_2{i},1)
         p1 = p1 - log(getKNNRadius(X_2{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_2{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_2{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_2{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_2{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_3{i},1)
         p1 = p1 - log(getKNNRadius(X_3{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_3{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_3{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_3{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_3{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_4{i},1)
         p1 = p1 - log(getKNNRadius(X_4{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_4{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_4{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_4{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_4{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_5{i},1)
         p1 = p1 - log(getKNNRadius(X_5{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_5{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_5{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_5{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_5{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_1{i},1)
         p1 = p1 - log(getKNNRadius(X_1{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_1{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_1{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_1{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_1{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_2{i},1)
         p1 = p1 - log(getKNNRadius(X_2{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_2{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_2{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_2{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_2{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_3{i},1)
         p1 = p1 - log(getKNNRadius(X_3{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_3{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_3{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_3{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_3{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_4{i},1)
         p1 = p1 - log(getKNNRadius(X_4{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_4{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_4{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_4{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_4{i}(j, :), X_train5, k));
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
    p1 = 0;p2 = 0;p3 = 0;p4 = 0;p5 = 0;
    
    for j = 1:size(X_5{i},1)
         p1 = p1 - log(getKNNRadius(X_5{i}(j, :), X_train1, k));
         p2 = p2 - log(getKNNRadius(X_5{i}(j, :), X_train2, k));
         p3 = p3 - log(getKNNRadius(X_5{i}(j, :), X_train3, k));
         p4 = p4 - log(getKNNRadius(X_5{i}(j, :), X_train4, k));
         p5 = p5 - log(getKNNRadius(X_5{i}(j, :), X_train5, k));
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