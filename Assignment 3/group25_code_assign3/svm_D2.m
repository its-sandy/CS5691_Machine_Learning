rng(0);
c=5;

file = 'linearly_separable'; %'linearly_separable' or 'nonlinearly_separable'
opstr=sprintf('-s 0 -t 2 -d 2 -g 2 -r 1 -c %f', c);
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

X_test1 = (X_test1-minval)./rangeval;
X_test2 = (X_test2-minval)./rangeval;

X_val1 = (X_val1-minval)./rangeval;
X_val2 = (X_val2-minval)./rangeval;



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
xyn = (xy-minval)./rangeval;
[predicted_class] = svmpredict(1*ones(size(xyn,1),1), xyn ,model,'-q');
decisionmap = reshape(predicted_class, image_size);

subplot(3, 3, 8);
imagesc(xrange,yrange,decisionmap);
title('C-SVM with Gaussian Kernel');
hold on;
set(gca,'ydir','normal');

X_train = [X_train1; X_train2; X_train3];
colormap(cmap);
%gscatter(train_nn(:,1), train_nn(:,2), all_train_lb, 'rgb', 'sod');
gscatter(X_train(:,1), X_train(:,2), all_train_lb, 'rgb', 'sod');
hold on;
sv = full(model.SVs);
coefs = model.sv_coef;
bsv = ((sv(abs(sum(coefs, 2))'==c, :)).*rangeval)+minval;
usv = (sv(abs(sum(coefs, 2))'~=c, :).*rangeval)+minval;
plot(usv(:,1),usv(:,2),'ko','MarkerSize',10);
plot(bsv(:,1),bsv(:,2),'ks','MarkerSize',10);
%legend('Location','northwest');
if strcmp(file, 'linearly_separable')
    conf_matrix_test = zeros(3, 3);
    lgd=legend('Class1','Class2','Class3','Unbounded Support Vector','Bounded Support Vector');
else
    conf_matrix_test = zeros(2, 2);
    lgd=legend('Class1','Class2','Unbounded Support Vector','Bounded Support Vector');
end
lgd.FontSize = 15;
set(lgd, 'Position', [0.8,0.5,0,0]);
 hold off;
[predicted_class] = svmpredict(1*ones(size(X_test1,1),1), X_test1 ,model,'-q');
for i = 1:size(conf_matrix_test,1)
    conf_matrix_test(1,i) = sum(predicted_class==i);
end

[predicted_class] = svmpredict(2*ones(size(X_test2,1),1), X_test2 ,model,'-q');
for i = 1:size(conf_matrix_test,1)
    conf_matrix_test(2,i) = sum(predicted_class==i);
end
if strcmp(file, 'linearly_separable')
    [predicted_class] = svmpredict(3*ones(size(X_test3,1),1), X_test3 ,model,'-q');
    for i = 1:size(conf_matrix_test,1)
        conf_matrix_test(3,i) = sum(predicted_class==i);
    end    
end
conf_matrix_test = conf_matrix_test
sumc = sum(conf_matrix_test,2);
conf_percentage=100*(conf_matrix_test./sumc)
