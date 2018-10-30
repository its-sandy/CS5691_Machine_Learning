sigma = 1;

X_train1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class1_train.txt'));
X_val1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class1_val.txt'));
X_test1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class1_test.txt'));

X_train2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class2_train.txt'));
X_val2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class2_val.txt'));
X_test2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class2_test.txt'));

%%%Initializing plotting%%%
xrange = [-2 3];
yrange = [-2 2];
inc = 0.01;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
predicted_class = zeros(size(xy,1),1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

means = zeros(2, size(X_train1,2));
means(1, :) = mean(X_train1, 1);
means(2, :) = mean(X_train2, 1);
ccprob = ones(2, 1)/2;
    
C = sigma*sigma*eye(size(X_train1, 2));

conf_matrix_train = zeros(2, 2);
conf_matrix_val = zeros(2, 2);
conf_matrix_test = zeros(2, 2);

%%%Getting points to plot%%%

p1 = mvnpdf(xy, means(1, :), C)*ccprob(1);
p2 = mvnpdf(xy, means(2, :), C)*ccprob(2);

for i = 1:size(xy, 1)
   if max([p1(i) p2(i)])==p1(i)
       predicted_class(i) = 1;
   elseif max([p1(i) p2(i)])==p2(i)
       predicted_class(i) = 2;
   end
end

subplot(1, 3, 1);
decisionmap = reshape(predicted_class, image_size);

imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap = [1 0.8 0.8; 0.95 1 0.95];
colormap(cmap);

X_train = [X_train1; X_train2];
X_label = [ones(size(X_train1,1),1); ones(size(X_train2,1),1)*2];
gscatter(X_train(:,1), X_train(:,2), X_label, 'rg', 'so');

title('C_{1} = C_{2} = C = \sigma^{2} I');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p1 = mvnpdf(X_train1, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_train1, means(2, :), C)*ccprob(2);

for i = 1:size(X_train1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_train(1, 1) = conf_matrix_train(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_train(1, 2) = conf_matrix_train(1, 2)+1;
   end
end

p1 = mvnpdf(X_train2, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_train2, means(2, :), C)*ccprob(2);

for i = 1:size(X_train2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_train(2, 1) = conf_matrix_train(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_train(2, 2) = conf_matrix_train(2, 2)+1;
   end
end

p1 = mvnpdf(X_val1, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_val1, means(2, :), C)*ccprob(2);

for i = 1:size(X_val1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_val(1, 1) = conf_matrix_val(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_val(1, 2) = conf_matrix_val(1, 2)+1;
   end
end

p1 = mvnpdf(X_val2, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_val2, means(2, :), C)*ccprob(2);

for i = 1:size(X_val2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_val(2, 1) = conf_matrix_val(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_val(2, 2) = conf_matrix_val(2, 2)+1;
   end
end

p1 = mvnpdf(X_test1, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_test1, means(2, :), C)*ccprob(2);

for i = 1:size(X_test1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_test(1, 1) = conf_matrix_test(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_test(1, 2) = conf_matrix_test(1, 2)+1;
   end
end

p1 = mvnpdf(X_test2, means(1, :), C1)*ccprob(1);
p2 = mvnpdf(X_test2, means(2, :), C2)*ccprob(2);

for i = 1:size(X_test2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_test(2, 1) = conf_matrix_test(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_test(2, 2) = conf_matrix_test(2, 2)+1;
   end
end

fprintf("Training accuracy: %f%%\n", sum(sum(diag(conf_matrix_train), 1))*100/sum(sum(conf_matrix_train, 1)));
fprintf("Validation accuracy: %f%%\n", sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
fprintf("Test accuracy: %f%%\n", sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));

disp(conf_matrix_train);
disp(conf_matrix_test);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C1 = cov_matrix(X_train1);
C2 = cov_matrix(X_train2);

C1(1,2) = 0; C1(2,1) = 0;
C2(1,2) = 0; C2(2,1) = 0;

C = (C1+C2)/2;

conf_matrix_train = zeros(2, 2);
conf_matrix_val = zeros(2, 2);
conf_matrix_test = zeros(2, 2);

%%%Getting points to plot%%%

p1 = mvnpdf(xy, means(1, :), C)*ccprob(1);
p2 = mvnpdf(xy, means(2, :), C)*ccprob(2);

for i = 1:size(xy, 1)
   if max([p1(i) p2(i)])==p1(i)
       predicted_class(i) = 1;
   elseif max([p1(i) p2(i)])==p2(i)
       predicted_class(i) = 2;
   end
end

subplot(1, 3, 2);
decisionmap = reshape(predicted_class, image_size);

imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap = [1 0.8 0.8; 0.95 1 0.95];
colormap(cmap);

X_train = [X_train1; X_train2];
X_label = [ones(size(X_train1,1),1); ones(size(X_train2,1),1)*2];
gscatter(X_train(:,1), X_train(:,2), X_label, 'rg', 'so');

title('C_{1} = C_{2} = C');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p1 = mvnpdf(X_train1, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_train1, means(2, :), C)*ccprob(2);

for i = 1:size(X_train1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_train(1, 1) = conf_matrix_train(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_train(1, 2) = conf_matrix_train(1, 2)+1;
   end
end

p1 = mvnpdf(X_train2, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_train2, means(2, :), C)*ccprob(2);

for i = 1:size(X_train2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_train(2, 1) = conf_matrix_train(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_train(2, 2) = conf_matrix_train(2, 2)+1;
   end
end

p1 = mvnpdf(X_val1, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_val1, means(2, :), C)*ccprob(2);

for i = 1:size(X_val1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_val(1, 1) = conf_matrix_val(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_val(1, 2) = conf_matrix_val(1, 2)+1;
   end
end

p1 = mvnpdf(X_val2, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_val2, means(2, :), C)*ccprob(2);

for i = 1:size(X_val2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_val(2, 1) = conf_matrix_val(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_val(2, 2) = conf_matrix_val(2, 2)+1;
   end
end

p1 = mvnpdf(X_test1, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_test1, means(2, :), C)*ccprob(2);

for i = 1:size(X_test1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_test(1, 1) = conf_matrix_test(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_test(1, 2) = conf_matrix_test(1, 2)+1;
   end
end

p1 = mvnpdf(X_test2, means(1, :), C1)*ccprob(1);
p2 = mvnpdf(X_test2, means(2, :), C2)*ccprob(2);

for i = 1:size(X_test2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_test(2, 1) = conf_matrix_test(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_test(2, 2) = conf_matrix_test(2, 2)+1;
   end
end

fprintf("Training accuracy: %f%%\n", sum(sum(diag(conf_matrix_train), 1))*100/sum(sum(conf_matrix_train, 1)));
fprintf("Validation accuracy: %f%%\n", sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
fprintf("Test accuracy: %f%%\n", sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));

disp(conf_matrix_train);
disp(conf_matrix_test);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conf_matrix_train = zeros(2, 2);
conf_matrix_val = zeros(2, 2);
conf_matrix_test = zeros(2, 2);

%%%Getting points to plot%%%

p1 = mvnpdf(xy, means(1, :), C1)*ccprob(1);
p2 = mvnpdf(xy, means(2, :), C2)*ccprob(2);

for i = 1:size(xy, 1)
   if max([p1(i) p2(i)])==p1(i)
       predicted_class(i) = 1;
   elseif max([p1(i) p2(i)])==p2(i)
       predicted_class(i) = 2;
   end
end

subplot(1, 3, 3);
decisionmap = reshape(predicted_class, image_size);

imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
cmap = [1 0.8 0.8; 0.95 1 0.95];
colormap(cmap);

X_train = [X_train1; X_train2];
X_label = [ones(size(X_train1,1),1); ones(size(X_train2,1),1)*2];
gscatter(X_train(:,1), X_train(:,2), X_label, 'rg', 'so');

title('C_{1} \neq C_{2}');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p1 = mvnpdf(X_train1, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_train1, means(2, :), C)*ccprob(2);

for i = 1:size(X_train1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_train(1, 1) = conf_matrix_train(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_train(1, 2) = conf_matrix_train(1, 2)+1;
   end
end

p1 = mvnpdf(X_train2, means(1, :), C)*ccprob(1);
p2 = mvnpdf(X_train2, means(2, :), C)*ccprob(2);

for i = 1:size(X_train2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_train(2, 1) = conf_matrix_train(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_train(2, 2) = conf_matrix_train(2, 2)+1;
   end
end

p1 = mvnpdf(X_val1, means(1, :), C1)*ccprob(1);
p2 = mvnpdf(X_val1, means(2, :), C2)*ccprob(2);

for i = 1:size(X_val1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_val(1, 1) = conf_matrix_val(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_val(1, 2) = conf_matrix_val(1, 2)+1;
   end
end

p1 = mvnpdf(X_val2, means(1, :), C1)*ccprob(1);
p2 = mvnpdf(X_val2, means(2, :), C2)*ccprob(2);

for i = 1:size(X_val2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_val(2, 1) = conf_matrix_val(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_val(2, 2) = conf_matrix_val(2, 2)+1;
   end
end

p1 = mvnpdf(X_test1, means(1, :), C1)*ccprob(1);
p2 = mvnpdf(X_test1, means(2, :), C2)*ccprob(2);

for i = 1:size(X_test1, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_test(1, 1) = conf_matrix_test(1, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_test(1, 2) = conf_matrix_test(1, 2)+1;
   end
end

p1 = mvnpdf(X_test2, means(1, :), C1)*ccprob(1);
p2 = mvnpdf(X_test2, means(2, :), C2)*ccprob(2);

for i = 1:size(X_test2, 1)
   if max([p1(i) p2(i)])==p1(i)
       conf_matrix_test(2, 1) = conf_matrix_test(2, 1)+1;
   elseif max([p1(i) p2(i)])==p2(i)
       conf_matrix_test(2, 2) = conf_matrix_test(2, 2)+1;
   end
end

fprintf("Training accuracy: %f%%\n", sum(sum(diag(conf_matrix_train), 1))*100/sum(sum(conf_matrix_train, 1)));
fprintf("Validation accuracy: %f%%\n", sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
fprintf("Test accuracy: %f%%\n", sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));

disp(conf_matrix_train);
disp(conf_matrix_test);