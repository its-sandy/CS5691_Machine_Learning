function [centers, R] = k_means(X, k)
%assumes each row corresponds to a data point and each column corresponds
%to an attribute size(x) = N*d
d = size(X, 2);
N = size(X, 1);
centers = X(randperm(N,k),:);
R = zeros(k, N);
Rold = ones(k, N);
ctr = 0;

while ctr < 1000 && sum(sum(abs(R - Rold))) ~= 0
    Rold = R;
    distances = dist(X, centers');
    [~,I] = min(distances, [], 2);
    R = ind2vec(I');    %k*N
    centers = R*X; %k*d
    centers = centers./sum(R,2);
    centers(isnan(centers)) = 0;
    ctr = ctr+1;
end
%covs = zeros(d,d,k);
%for i = 1:k
%    temp = (X - centers(i,:)).*(R(i,:)'); %N*d
%    covs(:,:,i) = (temp'*temp)./sum(R(i,:)); %d*d
%end
%covs(isnan(covs)) = 0;
centers = full(centers);
end