Q1 = 5;
Q2 = 5;
Q3 = 5;


X_train1 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class1_train.txt')));
X_val1 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class1_val.txt')));
X_test1 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class1_test.txt')));

X_train2 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class2_train.txt')));
X_val2 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class2_val.txt')));
X_test2 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class2_test.txt')));

X_train3 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class3_train.txt')));
X_val3 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class3_val.txt')));
X_test3 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class3_test.txt')));

%%%Initializing plotting%%%
xrange = [-5 15];
yrange = [-10 10];
inc = 0.05;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
predicted_class = zeros(size(xy,1),1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('default');
rng(1);

ccprob = ones(3, 1)/3;

conf_matrix_val = zeros(3, 3);
conf_matrix_test = zeros(3, 3);

likelihood = zeros(3,50);

for Q = 2:50
  fprintf("class 1 Q = %d\n",Q);
  try
    [w, mu, C, likelihood(1,Q)] = trainGMM_retLikelihood(X_train1, Q, 1);
  catch
    likelihood(1,Q) = 0;
    fprintf("Q = %d didn't work",Q);
  end
end
for Q = 2:50
  fprintf("class 2 Q = %d\n",Q);
  try
    [w, mu, C, likelihood(2,Q)] = trainGMM_retLikelihood(X_train2, Q, 1);
  catch
    likelihood(2,Q) = 0;
    fprintf("Q = %d didn't work",Q);
  end
end
for Q = 2:50
  fprintf("class 3 Q = %d\n",Q);
  try
    [w, mu, C, likelihood(3,Q)] = trainGMM_retLikelihood(X_train3, Q, 1);
  catch
    likelihood(3,Q) = 0;
    fprintf("Q = %d didn't work",Q);
  end
end

[w1, mu1, C1] = trainGMM(X_train1, Q1, 1);
[w2, mu2, C2] = trainGMM(X_train2, Q2, 1);
[w3, mu3, C3] = trainGMM(X_train3, Q3, 1);

%%%Getting points to plot%%%

p1 = zeros(size(xy, 1), 1);
p2 = zeros(size(xy, 1), 1);
p3 = zeros(size(xy, 1), 1);

for i = 1:size(xy, 1)
   p1(i) = ccprob(1)*evalGMM(xy(i, :), w1, mu1, C1);
   p2(i) = ccprob(2)*evalGMM(xy(i, :), w2, mu2, C2);
   p3(i) = ccprob(3)*evalGMM(xy(i, :), w3, mu3, C3);    
    
   if max([p1(i) p2(i) p3(i)])==p1(i)
       predicted_class(i) = 1;
   elseif max([p1(i) p2(i) p3(i)])==p2(i)
       predicted_class(i) = 2;
   elseif max([p1(i) p2(i) p3(i)])==p3(i)
       predicted_class(i) = 3;
   end
end

figure;
decisionmap = reshape(predicted_class, image_size);

imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
colormap(cmap);

X_train = [X_train1; X_train2; X_train3];
X_label = [ones(size(X_train1,1),1); ones(size(X_train2,1),1)*2; ones(size(X_train3,1),1)*3];
gscatter(X_train(:,1), X_train(:,2), X_label, 'rgb', 'sod');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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