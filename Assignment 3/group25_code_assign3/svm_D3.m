rng(1);

opstr='-s 0 -t 1 -d 2  -r 1 -g 3 -c 1 -h 0 -e 0.001';
%let s=0 always => C-SVM
%-t kernel_type : set type of kernel function (default 2)
%	0 -- linear: u'*v
%	1 -- polynomial: (gamma*u'*v + coef0)^degree
%	2 -- radial basis function: exp(-gamma*|u-v|^2)
%-g gamma : set gamma in kernel function (default 1/num_features) 
%-r coef0 : set coef0 in kernel function (default 0)
%-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)


files = dir('../group25_data_assign3/image_features/forest/*.jpg_color_edh_entropy');
ctr = 1;
X_1 = zeros(size(files, 1), 36, 23);
for file = files'
    X_1(ctr, :, :) = table2array(readtable(fullfile('..','group25_data_assign3','image_features','forest',file.name), 'FileType', 'text', 'Delimiter', ' '));

    ctr = ctr+1;
end
[train_ind, test_ind, val_ind] = dividerand(size(X_1, 1), 0.7, 0.2, 0.1);
X_train1nn = X_1(train_ind, :, :);
X_train1 = reshape(X_1(train_ind, :, :), [size(train_ind, 2)*36, 23]);
X_test1 = X_1(test_ind, :, :);
X_val1 = X_1(val_ind, :, :);
fprintf("Forest data Read\n");

files = dir('../group25_data_assign3/image_features/highway/*.jpg_color_edh_entropy');
ctr = 1;
X_2 = zeros(size(files, 1), 36, 23);
for file = files'
    X_2(ctr, :, :) = table2array(readtable(fullfile('..','group25_data_assign3','image_features','highway',file.name), 'FileType', 'text', 'Delimiter', ' '));
    ctr = ctr+1;
end
[train_ind, test_ind, val_ind] = dividerand(size(X_2, 1), 0.7, 0.2, 0.1);
X_train2nn = X_2(train_ind, :, :);
X_train2 = reshape(X_2(train_ind, :, :), [size(train_ind, 2)*36, 23]);
X_test2 = X_2(test_ind, :, :);
X_val2 = X_2(val_ind, :, :);
fprintf("Highway data Read\n");

files = dir('../group25_data_assign3/image_features/insidecity/*.jpg_color_edh_entropy');
ctr = 1;
X_3 = zeros(size(files, 1), 36, 23);
for file = files'
    X_3(ctr, :, :) = table2array(readtable(fullfile('..','group25_data_assign3','image_features','insidecity',file.name), 'FileType', 'text', 'Delimiter', ' '));
    ctr = ctr+1;
end
[train_ind, test_ind, val_ind] = dividerand(size(X_3, 1), 0.7, 0.2, 0.1);
X_train3nn = X_3(train_ind, :, :);
X_train3 = reshape(X_3(train_ind, :, :), [size(train_ind, 2)*36, 23]);
X_test3 = X_3(test_ind, :, :);
X_val3 = X_3(val_ind, :, :);
fprintf("Insidecity data Read\n");

minval =   min([X_train1; X_train2; X_train3]);
rangeval = max([X_train1; X_train2; X_train3]) - minval;

X_train1 = (X_train1-minval)./rangeval;
X_train2 = (X_train2-minval)./rangeval;
X_train3 = (X_train3-minval)./rangeval;

alltrain = [X_train1; X_train2; X_train3];
classlb = [ones(size(X_train1,1),1);2*ones(size(X_train2,1),1);3*ones(size(X_train3,1),1)];
model = svmtrain(classlb,alltrain,opstr);

conf_matrix_test = zeros(3, 3);
conf_matrix_train = zeros(3, 3);
conf_matrix_val = zeros(3, 3);
for i = 1:size(X_test1, 1)
    test = (squeeze(X_test1(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(1*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_test(1, predicted_class) = conf_matrix_test(1, predicted_class)+1;
end
for i = 1:size(X_test2, 1)
    test = (squeeze(X_test2(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(2*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_test(2, predicted_class) = conf_matrix_test(2, predicted_class)+1;
end
for i = 1:size(X_test3, 1)
    test = (squeeze(X_test3(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(3*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_test(3, predicted_class) = conf_matrix_test(3, predicted_class)+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(X_val1, 1)
    test = (squeeze(X_val1(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(1*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_val(1, predicted_class) = conf_matrix_val(1, predicted_class)+1;
end
for i = 1:size(X_val2, 1)
    test = (squeeze(X_val2(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(2*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_val(2, predicted_class) = conf_matrix_val(2, predicted_class)+1;
end
for i = 1:size(X_val3, 1)
    test = (squeeze(X_val3(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(3*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_val(3, predicted_class) = conf_matrix_val(3, predicted_class)+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(X_train1nn, 1)
    test = (squeeze(X_train1nn(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(1*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_train(1, predicted_class) = conf_matrix_train(1, predicted_class)+1;
end
for i = 1:size(X_train2nn, 1)
    test = (squeeze(X_train2nn(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(2*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_train(2, predicted_class) = conf_matrix_train(2, predicted_class)+1;
end
for i = 1:size(X_train3nn, 1)
    test = (squeeze(X_train3nn(i,:,:))-minval)./rangeval;    
    [lb] = svmpredict(3*ones(36,1), test,model,'-q');
    predicted_class = mode(lb);
    conf_matrix_train(3, predicted_class) = conf_matrix_train(3, predicted_class)+1;
end

conf_matrix_train = conf_matrix_train 
sumc = sum(conf_matrix_train,2);
conf_percentage_train=100*(conf_matrix_train./sumc)

conf_matrix_test =conf_matrix_test
sumc = sum(conf_matrix_test,2);
conf_percentage_test=100*(conf_matrix_test./sumc)

conf_matrix_val = conf_matrix_val
sumc = sum(conf_matrix_val,2);
conf_percentage_val=100*(conf_matrix_val./sumc)

conf_matrix_train = reshape(conf_matrix_train,1,[]);
conf_percentage_train = reshape(conf_percentage_train,1,[]);

conf_matrix_test = reshape(conf_matrix_test,1,[]);
conf_percentage_test = reshape(conf_percentage_test,1,[]);

conf_matrix_val = reshape(conf_matrix_val,1,[]);
conf_percentage_val = reshape(conf_percentage_val,1,[]);