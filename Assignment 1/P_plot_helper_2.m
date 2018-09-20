rng(0);
figure('Name', sprintf('Variation w.r.t Size of Training Data'), 'NumberTitle', 'off');

M = 2;
lambda = 1;


ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train100.txt'));
X_train = double(ds(:,1:end-1));
Y_train = double(ds(:,end));
d = size(X_train, 2);
phi = get_design_matrix(X_train, M);
D = size(phi, 2);
w = ((phi.')*phi+lambda*eye(D))\((phi.')*Y_train);

subplot(2,2,1);
scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_design_matrix(Xsurf, M);
surfplot5 = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot5.EdgeColor = 'none';
title(sprintf('lambda = 1'));

%%%%%%%%%%
M = 2;
lambda = 100;


ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train100.txt'));
X_train = double(ds(:,1:end-1));
Y_train = double(ds(:,end));
d = size(X_train, 2);
phi = get_design_matrix(X_train, M);
D = size(phi, 2);
w = ((phi.')*phi+lambda*eye(D))\((phi.')*Y_train);

subplot(2,2,2);
scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_design_matrix(Xsurf, M);
surfplot5 = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot5.EdgeColor = 'none';
title(sprintf('lambda = 100'));

%%%%%%%%%%%%%
M = 2;
lambda = 10000;


ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train100.txt'));
X_train = double(ds(:,1:end-1));
Y_train = double(ds(:,end));
d = size(X_train, 2);
phi = get_design_matrix(X_train, M);
D = size(phi, 2);
w = ((phi.')*phi+lambda*eye(D))\((phi.')*Y_train);

subplot(2,2,3);
scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_design_matrix(Xsurf, M);
surfplot5 = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot5.EdgeColor = 'none';
title(sprintf('lambda = 10000'));

%%%%%%%%%%

M = 2;
lambda = 1000000;


ds = table2array(readtable('Team21-Team30/team30/bivariate_group30/bivariateData/train100.txt'));
X_train = double(ds(:,1:end-1));
Y_train = double(ds(:,end));
d = size(X_train, 2);
phi = get_design_matrix(X_train, M);
D = size(phi, 2);
w = ((phi.')*phi+lambda*eye(D))\((phi.')*Y_train);

subplot(2,2,4);
scatter3(X_train(:, 1), X_train(:, 2), Y_train, 'r'); hold on; %scatter3(X_train(:, 1), X_train(:, 2), phi*w, 'r');

[x1surf,x2surf] = meshgrid(-10:0.25:10, -10:0.25:10);
Xsurf = horzcat(reshape(x1surf,81*81,1), reshape(x2surf,81*81,1));
phisurf = get_design_matrix(Xsurf, M);
surfplot5 = surf(x1surf, x2surf, reshape(phisurf*w,81,81), 'FaceAlpha', 0.7);
surfplot5.EdgeColor = 'none';
title(sprintf('lambda = 1000000'));
