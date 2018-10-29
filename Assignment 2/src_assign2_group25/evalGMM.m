function ret = evalGMM(x, w, mu, C)

ret=0;
for i=1:size(w)
    ret = ret + w(i)*mvnpdf(x, mu(i, :), C(:, :, i));

end