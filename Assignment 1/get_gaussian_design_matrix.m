function phi = get_gaussian_design_matrix(X,k,means,covs)

phi = zeros(size(X,1),k);

for i = 1:k
    cov_inv = inv(covs(:,:,i));
    phi(:,i) = diag(((X - means(i,:))*cov_inv)*((X-means(i,:))'));
end
phi(isnan(phi)) = 0;
phi = exp((-0.5)*phi);
phi(isinf(phi)) = 0.001;
end


