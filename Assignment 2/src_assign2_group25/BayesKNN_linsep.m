k = 1;

X_train1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class1_train.txt'));
X_val1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class1_val.txt'));
X_test1 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class1_test.txt'));

X_train2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class2_train.txt'));
X_val2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class2_val.txt'));
X_test2 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class2_test.txt'));

X_train3 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class3_train.txt'));
X_val3 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class3_val.txt'));
X_test3 = table2array(readtable('..\data_assign2_group25\datasets 1  2\datasets 1 _ 2\group25\linearly_separable\class3_test.txt'));

ccprob = ones(3, 1)/3;

conf_matrix_val = zeros(3, 3);
conf_matrix_test = zeros(3, 3);

p1 = zeros(size(X_val1, 1), 1);
p2 = zeros(size(X_val1, 1), 1);
p3 = zeros(size(X_val1, 1), 1);

for i = 1:size(X_val1, 1)
    closest = zeros(size(X_train1, 1), 1);
    for j = 1:size(X_train1, 1)
        closest(j) = sum((X_train1(j, :)-X_val1(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p1(i) = (3*k)*ccprob(1)/(size(X_train1,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train2, 1), 1);
    for j = 1:size(X_train2, 1)
        closest(j) = sum((X_train2(j, :)-X_val1(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p2(i) = (3*k)*ccprob(2)/(size(X_train2,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train3, 1), 1);
    for j = 1:size(X_train3, 1)
        closest(j) = sum((X_train3(j, :)-X_val1(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p3(i) = (3*k)*ccprob(3)/(size(X_train3,1)*4*pi*(closest(k)^3));
    
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
    closest = zeros(size(X_train1, 1), 1);
    for j = 1:size(X_train1, 1)
        closest(j) = sum((X_train1(j, :)-X_val2(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p1(i) = (3*k)*ccprob(1)/(size(X_train1,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train2, 1), 1);
    for j = 1:size(X_train2, 1)
        closest(j) = sum((X_train2(j, :)-X_val2(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p2(i) = (3*k)*ccprob(2)/(size(X_train2,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train3, 1), 1);
    for j = 1:size(X_train3, 1)
        closest(j) = sum((X_train3(j, :)-X_val2(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p3(i) = (3*k)*ccprob(3)/(size(X_train3,1)*4*pi*(closest(k)^3));
    
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
    closest = zeros(size(X_train1, 1), 1);
    for j = 1:size(X_train1, 1)
        closest(j) = sum((X_train1(j, :)-X_val3(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p1(i) = (3*k)*ccprob(1)/(size(X_train1,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train2, 1), 1);
    for j = 1:size(X_train2, 1)
        closest(j) = sum((X_train2(j, :)-X_val3(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p2(i) = (3*k)*ccprob(2)/(size(X_train2,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train3, 1), 1);
    for j = 1:size(X_train3, 1)
        closest(j) = sum((X_train3(j, :)-X_val3(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p3(i) = (3*k)*ccprob(3)/(size(X_train3,1)*4*pi*(closest(k)^3));
    
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
    closest = zeros(size(X_train1, 1), 1);
    for j = 1:size(X_train1, 1)
        closest(j) = sum((X_train1(j, :)-X_test1(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p1(i) = (3*k)*ccprob(1)/(size(X_train1,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train2, 1), 1);
    for j = 1:size(X_train2, 1)
        closest(j) = sum((X_train2(j, :)-X_test1(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p2(i) = (3*k)*ccprob(2)/(size(X_train2,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train3, 1), 1);
    for j = 1:size(X_train3, 1)
        closest(j) = sum((X_train3(j, :)-X_test1(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p3(i) = (3*k)*ccprob(3)/(size(X_train3,1)*4*pi*(closest(k)^3));
    
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
    closest = zeros(size(X_train1, 1), 1);
    for j = 1:size(X_train1, 1)
        closest(j) = sum((X_train1(j, :)-X_test2(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p1(i) = (3*k)*ccprob(1)/(size(X_train1,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train2, 1), 1);
    for j = 1:size(X_train2, 1)
        closest(j) = sum((X_train2(j, :)-X_test2(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p2(i) = (3*k)*ccprob(2)/(size(X_train2,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train3, 1), 1);
    for j = 1:size(X_train3, 1)
        closest(j) = sum((X_train3(j, :)-X_test2(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p3(i) = (3*k)*ccprob(3)/(size(X_train3,1)*4*pi*(closest(k)^3));
    
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
    closest = zeros(size(X_train1, 1), 1);
    for j = 1:size(X_train1, 1)
        closest(j) = sum((X_train1(j, :)-X_test3(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p1(i) = (3*k)*ccprob(1)/(size(X_train1,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train2, 1), 1);
    for j = 1:size(X_train2, 1)
        closest(j) = sum((X_train2(j, :)-X_test3(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p2(i) = (3*k)*ccprob(2)/(size(X_train2,1)*4*pi*(closest(k)^3));
    
    closest = zeros(size(X_train3, 1), 1);
    for j = 1:size(X_train3, 1)
        closest(j) = sum((X_train3(j, :)-X_test3(i,:)).^2);
    end
    closest = sortrows(closest, 1);
    p3(i) = (3*k)*ccprob(3)/(size(X_train3,1)*4*pi*(closest(k)^3));
    
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