mat = zeros(4096, 150);

ctr = 1;
for i = [2, 10, 14, 19, 20, 21, 23, 25, 26, 28, 31, 32, 33, 34, 37]
    for j = 1:10
        mat(:, ctr) = double(reshape(imread(sprintf('EigenAnalysis_Data/Datasets1/64x64/%d/%d.pgm', i, j)), 4096, 1));
        ctr = ctr + 1;
    end
end
C = cov_matrix(mat);
[V, D] = eig(C);
[d,ind] = sort(abs(diag(D)));
V = V(:,ind);

for img_num = randperm(10, 5)
    
    figure('Name', sprintf('Image number: %d', img_num), 'NumberTitle', 'off');
    orig_img = double(reshape(imread(sprintf('EigenAnalysis_Data/Datasets1/64x64/37/%d.pgm', img_num)), 4096, 1));
    ctr = 1;
    for i = [1, 10, 20, 40, 80, 160, 320, 640]
        recon_img = zeros(4096, 1);
        for j = 1:i
            recon_img = recon_img + (orig_img'*V(:, end-j+1))*V(:, end-j+1);
        end
        recon_img = uint8(reshape(recon_img, 64, 64));
        subplot(2, 4, ctr), imshow(recon_img);
        ctr = ctr + 1;
    end
end