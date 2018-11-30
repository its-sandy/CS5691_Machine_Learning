rng(0);

file = 'linearly_separable'; %'linearly_separable' or 'nonlinearly_separable'

X_train1 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class1_train.txt')));
X_val1 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class1_val.txt')));
X_test1 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class1_test.txt')));

X_train2 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class2_train.txt')));
X_val2 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class2_val.txt')));
X_test2 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class2_test.txt')));

if strcmp(file, 'linearly_separable')
    X_train3 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class3_train.txt')));
    X_val3 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class3_val.txt')));
    X_test3 = table2array(readtable(fullfile('..','group25_data_assign3',file,'class3_test.txt')));
end

%%%Initializing plotting%%%
xrange = [-5 20];
yrange = [-15 20];

if strcmp(file, 'nonlinearly_separable')
    xrange = [-2 3];
    yrange = [-2 2];
end

inc = 0.01;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(file, 'linearly_separable')
    targets_train = full(ind2vec([ones(1, size(X_train1,1)), ones(1, size(X_train2,1))*2, ones(1, size(X_train3,1))*3]));
    targets_val = full(ind2vec([ones(1, size(X_val1,1)), ones(1, size(X_val2,1))*2, ones(1, size(X_val3,1))*3]));
    targets_test = full(ind2vec([ones(1, size(X_test1,1)), ones(1, size(X_test2,1))*2, ones(1, size(X_test3,1))*3]));
    rangeval = max([X_train1; X_train2; X_train3]) - min([X_train1; X_train2; X_train3]);
    minval = min([X_train1; X_train2; X_train3]);
else
    targets_train = full(ind2vec([ones(1, size(X_train1,1)), ones(1, size(X_train2,1))*2]));
    targets_val = full(ind2vec([ones(1, size(X_val1,1)), ones(1, size(X_val2,1))*2]));
    targets_test = full(ind2vec([ones(1, size(X_test1,1)), ones(1, size(X_test2,1))*2]));
    rangeval = max([X_train1; X_train2]) - min([X_train1; X_train2]);
    minval = min([X_train1; X_train2]);
end

Xn_train1 = (X_train1-minval)./rangeval;
Xn_train2 = (X_train2-minval)./rangeval;
Xn_val1 = (X_val1-minval)./rangeval;
Xn_val2 = (X_val2-minval)./rangeval;
Xn_test1 = (X_test1-minval)./rangeval;
Xn_test2 = (X_test2-minval)./rangeval;
X_train = [X_train1; X_train2];
Xn_train = [Xn_train1; Xn_train2];
Xn_val = [Xn_val1; Xn_val2];
Xn_test = [Xn_test1; Xn_test2];
if strcmp(file, 'linearly_separable')
    Xn_train3 = (X_train3-minval)./rangeval;
    Xn_val3 = (X_val3-minval)./rangeval;
    Xn_test3 = (X_test3-minval)./rangeval;
    X_train = [X_train; X_train3];
    Xn_train = [Xn_train; Xn_train3];
    Xn_val = [Xn_val; Xn_val3];
    Xn_test = [Xn_test; Xn_test3];
end

xyn = (xy-minval)./rangeval;

nnet = patternnet([6, 5], 'traingdm');
nnet.trainParam.epochs = 15000;
nnet = train(nnet, Xn_train', targets_train);

%%%Getting points to plot%%%

predicted_class = vec2ind(nnet(xyn'));

decisionmap = reshape(predicted_class, image_size);

subplot(3, 3, 4);
imagesc(xrange,yrange,decisionmap);
title('MLFFNN');
hold on;
set(gca,'ydir','normal');

if strcmp(file, 'linearly_separable')
    X_label = [ones(size(X_train1,1),1); ones(size(X_train2,1),1)*2; ones(size(X_train3,1),1)*3];
    cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
else
    X_label = [ones(size(X_train1,1),1); ones(size(X_train2,1),1)*2];
    cmap = [1 0.8 0.8; 0.95 1 0.95];
end
colormap(cmap);
gscatter(X_train(:,1), X_train(:,2), X_label, 'rgb', 'sod');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
plotconfusion(targets_val, nnet(Xn_val'));
figure;
plotconfusion(targets_test, nnet(Xn_test'));