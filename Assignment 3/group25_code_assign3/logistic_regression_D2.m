rng(0);

pg = 1; %0 for polynomial basis, 1 for Gaussian basis
file = 'nonlinearly_separable'; %'linearly_separable' or 'nonlinearly_separable'
k = 6;  %Number of clusters for Gaussian basis
M = 3;  %Model complexity for polynomial basis
eta = 1e-1;

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

if pg==0
    phi_train1 = get_design_matrix(X_train1, M);
    phi_val1 = get_design_matrix(X_val1, M);
    phi_test1 = get_design_matrix(X_test1, M);
    
    phi_train2 = get_design_matrix(X_train2, M);
    phi_val2 = get_design_matrix(X_val2, M);
    phi_test2 = get_design_matrix(X_test2, M);
    
    if strcmp(file, 'linearly_separable')
        phi_train3 = get_design_matrix(X_train3, M);
        phi_val3 = get_design_matrix(X_val3, M);
        phi_test3 = get_design_matrix(X_test3, M);
    end
    
    xy = get_design_matrix(xy, M);
else
    if strcmp(file, 'linearly_separable')
        X_train = [X_train1; X_train2; X_train3];
    else
        X_train = [X_train1; X_train2];
    end
    [means, covs, R] = k_means(X_train, k);
    
    phi_train1 = get_gaussian_design_matrix(X_train1, k, means, covs);
    phi_val1 = get_gaussian_design_matrix(X_val1, k, means, covs);
    phi_test1 = get_gaussian_design_matrix(X_test1, k, means, covs);
    
    phi_train2 = get_gaussian_design_matrix(X_train2, k, means, covs);
    phi_val2 = get_gaussian_design_matrix(X_val2, k, means, covs);
    phi_test2 = get_gaussian_design_matrix(X_test2, k, means, covs);
    
    if strcmp(file, 'linearly_separable')
        phi_train3 = get_gaussian_design_matrix(X_train3, k, means, covs);
        phi_val3 = get_gaussian_design_matrix(X_val3, k, means, covs);
        phi_test3 = get_gaussian_design_matrix(X_test3, k, means, covs);
    end
    
    xy = get_gaussian_design_matrix(xy, k, means, covs);
end

if strcmp(file, 'linearly_separable')
    rangeval = max([phi_train1; phi_train2; phi_train3]) - min([phi_train1; phi_train2; phi_train3]);
    minval = min([phi_train1; phi_train2; phi_train3]);
else
    rangeval = max([phi_train1; phi_train2]) - min([phi_train1; phi_train2]);
    minval = min([phi_train1; phi_train2]);
end

if pg==0
    rangeval(1) = 1;
    minval(1) = 0;
end
phi_train1 = (phi_train1-minval)./rangeval;
phi_train2 = (phi_train2-minval)./rangeval;
phi_val1 = (phi_val1-minval)./rangeval;
phi_val2 = (phi_val2-minval)./rangeval;
phi_test1 = (phi_test1-minval)./rangeval;
phi_test2 = (phi_test2-minval)./rangeval;
if strcmp(file, 'linearly_separable')
    phi_train3 = (phi_train3-minval)./rangeval;
    phi_val3 = (phi_val3-minval)./rangeval;
    phi_test3 = (phi_test3-minval)./rangeval;
end

xy = (xy-minval)./rangeval;

if strcmp(file, 'linearly_separable')
    w = randn(3, size(phi_train1, 2));
    wold = zeros(3, size(phi_train1, 2));
else
    w = randn(2, size(phi_train1, 2));
    wold = zeros(2, size(phi_train1, 2));
end

ctr = 0;
num_iters = 10000;

if strcmp(file, 'linearly_separable')
    while ctr<num_iters
        wold = w;
        yn_1 = exp(w*phi_train1');
        yn_1 = yn_1./sum(yn_1, 1) - [1; 0; 0];
        yn_2 = exp(w*phi_train2');
        yn_2 = yn_2./sum(yn_2, 1) - [0; 1; 0];
        yn_3 = exp(w*phi_train3');
        yn_3 = yn_3./sum(yn_3, 1) - [0; 0; 1];
        w = w - eta*(yn_1*phi_train1+yn_2*phi_train2+yn_3*phi_train3);
        ctr = ctr+1;
    end
else
    while ctr<num_iters
        wold = w;
        yn_1 = exp(w*phi_train1');
        yn_1 = yn_1./sum(yn_1, 1) - [1; 0];
        yn_2 = exp(w*phi_train2');
        yn_2 = yn_2./sum(yn_2, 1) - [0; 1];
        w = w - eta*(yn_1*phi_train1+yn_2*phi_train2);
        ctr = ctr+1;
    end
end

%%%Getting points to plot%%%

p = exp(w*xy');
p = p./sum(p, 1);
[~, predicted_class] = max(p);

decisionmap = reshape(predicted_class, image_size);

imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');

if strcmp(file, 'linearly_separable')
    X_train = [X_train1; X_train2; X_train3];
    X_label = [ones(size(X_train1,1),1); ones(size(X_train2,1),1)*2; ones(size(X_train3,1),1)*3];
    cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
else
    X_train = [X_train1; X_train2];
    X_label = [ones(size(X_train1,1),1); ones(size(X_train2,1),1)*2];
    cmap = [1 0.8 0.8; 0.95 1 0.95];
end
colormap(cmap);
gscatter(X_train(:,1), X_train(:,2), X_label, 'rgb', 'sod');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%