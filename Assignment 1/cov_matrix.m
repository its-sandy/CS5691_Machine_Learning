function C = cov_matrix(inp_mat)
%Returns covariance matrix of input matrix 
%(assuming each row is a variable and each column is an observation)
num_obs = size(inp_mat, 2);
mean_val = mean(inp_mat, 2);

C = ((inp_mat-mean_val)*(inp_mat-mean_val)')./num_obs;

end
