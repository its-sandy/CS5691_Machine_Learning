function C = cov_matrix(inp_mat)
%Returns covariance matrix of input matrix 
%(assuming each column is a variable and each row is an observation)
num_obs = size(inp_mat, 1);
mean_val = mean(inp_mat, 1);
inp_mat = inp_mat - mean_val;
C = (inp_mat'*inp_mat)./num_obs;

end
