function loglik = getLogLikelihood(X, w, mu, C)
N = size(X,1);
Q = size(mu,1);
temp = zeros(N, Q);
for i=1:Q
   temp(:, i) = mvnpdf(X, mu(i, :), C(:, :, i)); 
end
loglik = 0;
for i=1:N
    loglik = loglik+log(temp(i, :)*w);
end