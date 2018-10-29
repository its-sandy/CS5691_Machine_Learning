k = 30;

X_train1 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class1_train.txt')));
X_val1 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class1_val.txt')));
X_test1 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class1_test.txt')));

X_train2 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class2_train.txt')));
X_val2 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class2_val.txt')));
X_test2 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class2_test.txt')));

X_train3 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class3_train.txt')));
X_val3 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class3_val.txt')));
X_test3 = table2array(readtable(fullfile('..','data_assign2_group25','datasets 1  2','datasets 1 _ 2','group25','overlapping','class3_test.txt')));

conf_matrix_val = zeros(3, 3);
conf_matrix_test = zeros(3, 3);

for i = 1:size(X_val1, 1)
    closest = zeros(size(X_train1, 1)+size(X_train2, 1)+size(X_train3, 1), 2);
    ctr = 1;
    for j = 1:size(X_train1, 1)
        closest(ctr, 1) = sum((X_train1(j, :)-X_val1(i,:)).^2);
        closest(ctr, 2) = 1;
        ctr = ctr+1;
    end
    for j = 1:size(X_train2, 1)
        closest(ctr, 1) = sum((X_train2(j, :)-X_val1(i,:)).^2);
        closest(ctr, 2) = 2;
        ctr = ctr+1;
    end
    for j = 1:size(X_train3, 1)
        closest(ctr, 1) = sum((X_train3(j, :)-X_val1(i,:)).^2);
        closest(ctr, 2) = 3;
        ctr = ctr+1;
    end
    closest = sortrows(closest, 1);
    closest = closest(1:k, :);
    conf_matrix_val(1, mode(closest(:, 2))) = conf_matrix_val(1, mode(closest(:, 2)))+1;
end

for i = 1:size(X_val2, 1)
    closest = zeros(size(X_train1, 1)+size(X_train2, 1)+size(X_train3, 1), 2);
    ctr = 1;
    for j = 1:size(X_train1, 1)
        closest(ctr, 1) = sum((X_train1(j, :)-X_val2(i,:)).^2);
        closest(ctr, 2) = 1;
        ctr = ctr+1;
    end
    for j = 1:size(X_train2, 1)
        closest(ctr, 1) = sum((X_train2(j, :)-X_val2(i,:)).^2);
        closest(ctr, 2) = 2;
        ctr = ctr+1;
    end
    for j = 1:size(X_train3, 1)
        closest(ctr, 1) = sum((X_train3(j, :)-X_val2(i,:)).^2);
        closest(ctr, 2) = 3;
        ctr = ctr+1;
    end
    closest = sortrows(closest, 1);
    closest = closest(1:k, :);
    conf_matrix_val(2, mode(closest(:, 2))) = conf_matrix_val(2, mode(closest(:, 2)))+1;
end

for i = 1:size(X_val3, 1)
    closest = zeros(size(X_train1, 1)+size(X_train2, 1)+size(X_train3, 1), 2);
    ctr = 1;
    for j = 1:size(X_train1, 1)
        closest(ctr, 1) = sum((X_train1(j, :)-X_val3(i,:)).^2);
        closest(ctr, 2) = 1;
        ctr = ctr+1;
    end
    for j = 1:size(X_train2, 1)
        closest(ctr, 1) = sum((X_train2(j, :)-X_val3(i,:)).^2);
        closest(ctr, 2) = 2;
        ctr = ctr+1;
    end
    for j = 1:size(X_train3, 1)
        closest(ctr, 1) = sum((X_train3(j, :)-X_val3(i,:)).^2);
        closest(ctr, 2) = 3;
        ctr = ctr+1;
    end
    closest = sortrows(closest, 1);
    closest = closest(1:k, :);
    conf_matrix_val(3, mode(closest(:, 2))) = conf_matrix_val(3, mode(closest(:, 2)))+1;
end

for i = 1:size(X_test1, 1)
    closest = zeros(size(X_train1, 1)+size(X_train2, 1)+size(X_train3, 1), 2);
    ctr = 1;
    for j = 1:size(X_train1, 1)
        closest(ctr, 1) = sum((X_train1(j, :)-X_test1(i,:)).^2);
        closest(ctr, 2) = 1;
        ctr = ctr+1;
    end
    for j = 1:size(X_train2, 1)
        closest(ctr, 1) = sum((X_train2(j, :)-X_test1(i,:)).^2);
        closest(ctr, 2) = 2;
        ctr = ctr+1;
    end
    for j = 1:size(X_train3, 1)
        closest(ctr, 1) = sum((X_train3(j, :)-X_test1(i,:)).^2);
        closest(ctr, 2) = 3;
        ctr = ctr+1;
    end
    closest = sortrows(closest, 1);
    closest = closest(1:k, :);
    conf_matrix_test(1, mode(closest(:, 2))) = conf_matrix_test(1, mode(closest(:, 2)))+1;
end

for i = 1:size(X_test2, 1)
    closest = zeros(size(X_train1, 1)+size(X_train2, 1)+size(X_train3, 1), 2);
    ctr = 1;
    for j = 1:size(X_train1, 1)
        closest(ctr, 1) = sum((X_train1(j, :)-X_test2(i,:)).^2);
        closest(ctr, 2) = 1;
        ctr = ctr+1;
    end
    for j = 1:size(X_train2, 1)
        closest(ctr, 1) = sum((X_train2(j, :)-X_test2(i,:)).^2);
        closest(ctr, 2) = 2;
        ctr = ctr+1;
    end
    for j = 1:size(X_train3, 1)
        closest(ctr, 1) = sum((X_train3(j, :)-X_test2(i,:)).^2);
        closest(ctr, 2) = 3;
        ctr = ctr+1;
    end
    closest = sortrows(closest, 1);
    closest = closest(1:k, :);
    conf_matrix_test(2, mode(closest(:, 2))) = conf_matrix_test(2, mode(closest(:, 2)))+1;
end

for i = 1:size(X_test3, 1)
    closest = zeros(size(X_train1, 1)+size(X_train2, 1)+size(X_train3, 1), 2);
    ctr = 1;
    for j = 1:size(X_train1, 1)
        closest(ctr, 1) = sum((X_train1(j, :)-X_test3(i,:)).^2);
        closest(ctr, 2) = 1;
        ctr = ctr+1;
    end
    for j = 1:size(X_train2, 1)
        closest(ctr, 1) = sum((X_train2(j, :)-X_test3(i,:)).^2);
        closest(ctr, 2) = 2;
        ctr = ctr+1;
    end
    for j = 1:size(X_train3, 1)
        closest(ctr, 1) = sum((X_train3(j, :)-X_test3(i,:)).^2);
        closest(ctr, 2) = 3;
        ctr = ctr+1;
    end
    closest = sortrows(closest, 1);
    closest = closest(1:k, :);
    conf_matrix_test(3, mode(closest(:, 2))) = conf_matrix_test(3, mode(closest(:, 2)))+1;
end

fprintf("Validation accuracy: %f%%\n", sum(sum(diag(conf_matrix_val), 1))*100/sum(sum(conf_matrix_val, 1)));
fprintf("Test accuracy: %f%%\n", sum(sum(diag(conf_matrix_test), 1))*100/sum(sum(conf_matrix_test, 1)));