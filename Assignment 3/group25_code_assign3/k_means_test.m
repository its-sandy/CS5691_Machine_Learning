function I = k_means_test(X, centers)
%assumes each row corresponds to a data point and each column corresponds
%to an attribute size(x) = N*d
distances = dist(X, centers');
[~,I] = min(distances, [], 2);
