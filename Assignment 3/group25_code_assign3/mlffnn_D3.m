rng(0);

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

targets_train = full(ind2vec([ones(1, size(X_train1,1)), ones(1, size(X_train2,1))*2, ones(1, size(X_train3,1))*3]));
targets_val = full(ind2vec([ones(1, size(X_val1,1)), ones(1, size(X_val2,1))*2, ones(1, size(X_val3,1))*3]));
targets_test = full(ind2vec([ones(1, size(X_test1,1)), ones(1, size(X_test2,1))*2, ones(1, size(X_test3,1))*3]));
rangeval = max([X_train1; X_train2; X_train3]) - min([X_train1; X_train2; X_train3]);
minval = min([X_train1; X_train2; X_train3]);

Xn_train1 = (X_train1-minval)./rangeval;
Xn_train2 = (X_train2-minval)./rangeval;
Xn_train3 = (X_train3-minval)./rangeval;
X_train = [X_train1; X_train2; X_train3];
Xn_train = [Xn_train1; Xn_train2; Xn_train3];

nnet = patternnet([60, 60], 'traingdm');
nnet.trainParam.lr = 0.005;
nnet.trainParam.epochs = 2000;
nnet = train(nnet, Xn_train', targets_train);

outputs_val = zeros(3, size(targets_val,2));
outputs_test = zeros(3, size(targets_test, 2));

ctr = 1;

for i = 1:size(X_val1, 1)
    curr = squeeze(X_val1(i, :, :));
    curr = (curr-minval)./rangeval;
    outputs_val(:, ctr) = sum(log(nnet(curr')), 2);
    ctr = ctr+1;
end

for i = 1:size(X_val2, 1)
    curr = squeeze(X_val2(i, :, :));
    curr = (curr-minval)./rangeval;
    outputs_val(:, ctr) = sum(log(nnet(curr')), 2);
    ctr = ctr+1;
end

for i = 1:size(X_val3, 1)
    curr = squeeze(X_val3(i, :, :));
    curr = (curr-minval)./rangeval;
    outputs_val(:, ctr) = sum(log(nnet(curr')), 2);
    ctr = ctr+1;
end

ctr = 1;

for i = 1:size(X_test1, 1)
    curr = squeeze(X_test1(i, :, :));
    curr = (curr-minval)./rangeval;
    outputs_test(:, ctr) = sum(log(nnet(curr')), 2);
    ctr = ctr+1;
end

for i = 1:size(X_test2, 1)
    curr = squeeze(X_test2(i, :, :));
    curr = (curr-minval)./rangeval;
    outputs_test(:, ctr) = sum(log(nnet(curr')), 2);
    ctr = ctr+1;
end

for i = 1:size(X_test3, 1)
    curr = squeeze(X_test3(i, :, :));
    curr = (curr-minval)./rangeval;
    outputs_test(:, ctr) = sum(log(nnet(curr')), 2);
    ctr = ctr+1;
end

figure;
plotconfusion(outputs_val, targets_val);
figure;
plotconfusion(outputs_test, targets_test);