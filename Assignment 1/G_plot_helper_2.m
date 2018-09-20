rng(0);
figure('Name', sprintf('Variation w.r.t Size of Training Data'), 'NumberTitle', 'off');

k = 18;
var = 500;
lambda = 0.001;


ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train20.txt'));
X_train = double(ds(:,1:end-1));
Y_train = double(ds(:,end));
d = size(X_train, 2);
[means, covs, ~] = k_means(X_train,k);
for i=1:k
   covs(:,:,i) = var*eye(d); 
end
phi = get_gaussian_design_matrix(X_train, k, means, covs);
phi_tilda = get_gaussian_design_matrix(means, k, means, covs); %k*k
phi_tilda = eye(k);

w = pinv(phi'*phi + lambda*phi_tilda)*(phi')*Y_train;

subplot(2,2,1);
scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_gaussian_design_matrix(Xsurf, k, means, covs);
surfplot5 = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot5.EdgeColor = 'none';
title(sprintf('20 data points'));

%%%%%%%%%%
k = 18;
var = 500;
lambda = 0.001;


ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train100.txt'));
X_train = double(ds(:,1:end-1));
Y_train = double(ds(:,end));
d = size(X_train, 2);
[means, covs, ~] = k_means(X_train,k);
for i=1:k
   covs(:,:,i) = var*eye(d); 
end
phi = get_gaussian_design_matrix(X_train, k, means, covs);
phi_tilda = get_gaussian_design_matrix(means, k, means, covs); %k*k
phi_tilda = eye(k);
w = pinv(phi'*phi + lambda*phi_tilda)*(phi')*Y_train;

subplot(2,2,2);
scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_gaussian_design_matrix(Xsurf, k, means, covs);
surfplot5 = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot5.EdgeColor = 'none';
title(sprintf('100 data points'));

%%%%%%%%%%%%%
k = 18;
var = 500;
lambda = 0.001;


ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train1000.txt'));
X_train = double(ds(:,1:end-1));
Y_train = double(ds(:,end));
d = size(X_train, 2);
[means, covs, ~] = k_means(X_train,k);
for i=1:k
   covs(:,:,i) = var*eye(d); 
end
phi = get_gaussian_design_matrix(X_train, k, means, covs);
phi_tilda = get_gaussian_design_matrix(means, k, means, covs); %k*k
phi_tilda = eye(k);
w = pinv(phi'*phi + lambda*phi_tilda)*(phi')*Y_train;

subplot(2,2,3);
scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_gaussian_design_matrix(Xsurf, k, means, covs);
surfplot5 = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot5.EdgeColor = 'none';
title(sprintf('1000 data points'));

%%%%%%%%%%

k = 18;
var = 500;
lambda = 0.001;


ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train.txt'));
X_train = double(ds(:,1:end-1));
Y_train = double(ds(:,end));
d = size(X_train, 2);
[means, covs, ~] = k_means(X_train,k);
for i=1:k
   covs(:,:,i) = var*eye(d); 
end
phi = get_gaussian_design_matrix(X_train, k, means, covs);
phi_tilda = get_gaussian_design_matrix(means, k, means, covs); %k*k
phi_tilda = eye(k);
w = pinv(phi'*phi + lambda*phi_tilda)*(phi')*Y_train;

subplot(2,2,4);
scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_gaussian_design_matrix(Xsurf, k, means, covs);
surfplot5 = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot5.EdgeColor = 'none';
title(sprintf('2000 data points'));
