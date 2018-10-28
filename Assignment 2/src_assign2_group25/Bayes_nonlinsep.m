X_train1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class1_train.txt'));
X_val1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class1_val.txt'));
X_test1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class1_test.txt'));

X_train2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class2_train.txt'));
X_val2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class2_val.txt'));
X_test2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\nonlinearly_separable\class2_test.txt'));

means = zeros(2, size(X_train1,2));
means(1, :) = mean(X_train1, 1);
means(2, :) = mean(X_train2, 1);
ccprob = ones(2, 1)/2;
    
C1 = cov_matrix(X_train1);
C2 = cov_matrix(X_train2);

C = (C1+C2)/2;

conf_matrix_val = zeros(2, 2);
conf_matrix_test = zeros(2, 2);

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

fprintf("Validation accuracy: %f%%\n", sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
fprintf("Test accuracy: %f%%\n", sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conf_matrix_val = zeros(2, 2);
conf_matrix_test = zeros(2, 2);

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

fprintf("Validation accuracy: %f%%\n", sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
fprintf("Test accuracy: %f%%\n", sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));
