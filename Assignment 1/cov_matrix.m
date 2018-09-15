function C = cov_matrix(inp_mat)
%Returns covariance matrix of input matrix 
%(assuming each row is a variable and each column is an observation)
num_obs = size(inp_mat, 2);
mean_val = mean(inp_mat, 2);
C = 0;

for i = 1:num_obs
    C = C + (inp_mat(:,i)-mean_val)*(inp_mat(:,i)-mean_val)';
end

end