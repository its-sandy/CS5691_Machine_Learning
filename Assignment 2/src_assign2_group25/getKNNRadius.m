function r = getKNNRadius(x, train, k)

d = size(train,2);
closest = zeros(size(train, 1), 1);
for j = 1:size(train, 1)
    closest(j) = sum((train(j, :)-x).^2);
end
closest = sortrows(closest, 1);

k = min(k, size(closest,1));
r = sqrt(closest(k));
