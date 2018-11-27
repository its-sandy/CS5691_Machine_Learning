k = 10;
batch_size = 10;

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
fprintf("Read Class 1\n");

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
fprintf("Read Class 2\n");

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
fprintf("Read Class 3\n");

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
fprintf("Read Class 4\n");

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
fprintf("Read Class 5\n");

All_train = cell(5, 1);
All_train{1} = X_train1;clear X_train1;
All_train{2} = X_train2;clear X_train2;
All_train{3} = X_train3;clear X_train3;
All_train{4} = X_train4;clear X_train4;
All_train{5} = X_train5;clear X_train5;

val_ind = cell(5, 1);
val_ind{1} = val_ind1;clear val_ind1;
val_ind{2} = val_ind2;clear val_ind2;
val_ind{3} = val_ind3;clear val_ind3;
val_ind{4} = val_ind4;clear val_ind4;
val_ind{5} = val_ind5;clear val_ind5;

test_ind = cell(5, 1);
test_ind{1} = test_ind1;clear test_ind1;
test_ind{2} = test_ind2;clear test_ind2;
test_ind{3} = test_ind3;clear test_ind3;
test_ind{4} = test_ind4;clear test_ind4;
test_ind{5} = test_ind5;clear test_ind5;

X = cell(5, 1);
X{1} = X_1;clear X_1;
X{2} = X_2;clear X_2;
X{3} = X_3;clear X_3;
X{4} = X_4;clear X_4;
X{5} = X_5;clear X_5;

accuracies = zeros(10,2);
for hyper = 1:4
  k = hyper*5;
  conf_matrix_val = zeros(5, 5);
  conf_matrix_test = zeros(5, 5);

  for l = 1:5
    num_batches = ceil(size(val_ind{l},2)/batch_size);
    fprintf("Class %d Validation Data - Number of Batches = %d\n", l, num_batches);
    for batch_no = 0 : (num_batches-1)
      pred = VaryingLengthKNN(X{l}(val_ind{l}( (batch_no*batch_size + 1) : min(batch_no*batch_size + batch_size, size(val_ind{l},2)) )), All_train, k);
      for i = 1:5
        conf_matrix_val(l,i) = conf_matrix_val(l,i) + sum(pred == i);
      end
      fprintf("Validation Data Class %d Batch %d Done\n", l, batch_no+1);
    end
  end

  for l = 1:5
    num_batches = ceil(size(test_ind{l},2)/batch_size);
    fprintf("Class %d Test Data - Number of Batches = %d\n", l, num_batches);
    for batch_no = 0 : (num_batches-1)
      pred = VaryingLengthKNN(X{l}(test_ind{l}( (batch_no*batch_size + 1) : min(batch_no*batch_size + batch_size, size(test_ind{l},2)) )), All_train, k);
      for i = 1:5
        conf_matrix_test(l,i) = conf_matrix_test(l,i) + sum(pred == i);
      end
      fprintf("Test Data Class %d Batch %d Done\n", l, batch_no+1);
    end
  end

  fprintf("Validation accuracy (k=%d): %f%%\n", k, sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
  fprintf("Test accuracy (k=%d): %f%%\n", k, sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));
  accuracies(hyper, 1) = sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1));
  accuracies(hyper, 2) = sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1));
end
save('BayesKNN_2b_accuracies.mat','accuracies');