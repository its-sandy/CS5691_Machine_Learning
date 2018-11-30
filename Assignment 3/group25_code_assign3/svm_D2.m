rng(0);

file = 'nonlinearly_separable'; %'linearly_separable' or 'nonlinearly_separable'
opstr='-s 0 -t 2 -d 3 -r 1 -c 5';
%let s=0 always => C-SVM
%-t kernel_type : set type of kernel function (default 2)
%	0 -- linear: u'*v
%	1 -- polynomial: (gamma*u'*v + coef0)^degree
%	2 -- radial basis function: exp(-gamma*|u-v|^2)
%-d degree : set degree in kernel function (default 3)
%-g gamma : set gamma in kernel function (default 1/num_features) 
%-r coef0 : set coef0 in kernel function (default 0)
%-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)

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
 
if strcmp(file, 'linearly_separable')
    train_nn=[X_train1; X_train2; X_train3];
    minval = min([X_train1; X_train2; X_train3]);
    maxval = max([X_train1; X_train2; X_train3]);
    rangeval = max([X_train1; X_train2; X_train3]) - minval;
else
    train_nn=[X_train1; X_train2;];
    minval = min([X_train1; X_train2]);
    maxval = max([X_train1; X_train2]);
    rangeval = max([X_train1; X_train2]) - minval;
end

%%%Initializing plotting%%%
xrange = [(minval(1)-0.5) (maxval(1)+0.5)];
yrange = [(minval(2)-0.5) (maxval(2)+0.5)];
%xrange = [0 1];
%yrange = [0 1];
inc = 0.01;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_test1 = (X_test1-minval)./rangeval;
X_test2 = (X_test2-minval)./rangeval;

X_val1 = (X_val1-minval)./rangeval;
X_val2 = (X_val2-minval)./rangeval;

xy = (xy-minval)./rangeval;

if strcmp(file, 'linearly_separable')
    X_test3 = (X_test3-minval)./rangeval;
    X_val3 = (X_val3-minval)./rangeval;
    
    
    all_train = ([X_train1; X_train2; X_train3]-minval)./rangeval;
    all_test = [X_test1; X_test2;X_test3];
    
    all_train_lb = [ones(size(X_train1,1),1);2*ones(size(X_train2,1),1);3*ones(size(X_train3,1),1)];
    all_test_lb = [ones(size(X_test1,1),1);2*ones(size(X_test2,1),1);3*ones(size(X_test3,1),1)];
    cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
else
    all_train = ([X_train1; X_train2]-minval)./rangeval;
    all_test = [X_test1; X_test2];
    
    all_train_lb = [ones(size(X_train1,1),1);2*ones(size(X_train2,1),1)];
    all_test_lb = [ones(size(X_test1,1),1);2*ones(size(X_test2,1),1)];
    cmap = [1 0.8 0.8; 0.95 1 0.95];
end

model = svmtrain(all_train_lb,all_train,opstr);

[predicted_class] = svmpredict(1*ones(size(xy,1),1), xy ,model);
decisionmap = reshape(predicted_class, image_size);

imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');

colormap(cmap);
gscatter(train_nn(:,1), train_nn(:,2), all_train_lb, 'rgb', 'sod');
%gscatter(all_train(:,1), all_train(:,2), all_train_lb, 'rgb', 'sod');
