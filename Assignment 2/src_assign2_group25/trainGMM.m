function [w, mu, C] = trainGMM(X, Q, isNaive)

N = size(X, 1);
d = size(X, 2);

[centers, covs, R] = k_means(X, Q);

w = full(sum(R~=0, 2))/nnz(R);
disp(w);
mu = centers;
C = covs;
gamma = zeros(N, Q);

lhood_old = 0;
temp = zeros(N, Q);
for i=1:Q
   C(:, :, i) = C(:, :, i) + (1e-3)*eye(size(C(:, :, i), 1));
   if isNaive==1
       C(:, :, i) = diag(diag(C(:, :, i)));
   end
   temp(:, i) = mvnpdf(X, mu(i, :), C(:, :, i)); 
end
lhood_new = 0;
for i=1:N
    lhood_new = lhood_new+log(temp(i, :)*w);
end

ctr = 0;
threshold = 2000;

while ctr==0 || abs(lhood_new-lhood_old)>threshold
    gamma = temp;
    for i=1:N
        gamma(i, :) = gamma(i, :)/sum(gamma(i, :));
    end
    for i=1:Q
        mu(i, :) = sum(gamma(:,i).*X, 1)/sum(gamma(:,i));
    end
    for i=1:Q
        C(:, :, i) = (gamma(:,i).*(X-mu(i, :)))'*(X-mu(i, :))/sum(gamma(:,i));
    end
    
    lhood_old = lhood_new;
    
    for i=1:Q
        temp(:, i) = mvnpdf(X, mu(i, :), C(:, :, i));
        if isNaive==1
            C(:, :, i) = diag(diag(C(:, :, i)));
        end
    end
    lhood_new = 0;
    for i=1:N
        lhood_new = lhood_new+log(temp(i, :)*w);
    end
    ctr = ctr+1;
    fprintf("%f %f\n", lhood_old, lhood_new);
end
fprintf("\n");
end