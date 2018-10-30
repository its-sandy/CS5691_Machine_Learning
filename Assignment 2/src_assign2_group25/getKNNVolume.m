function V = getKNNVolume(x, train, k)

d = size(train,2);
closest = zeros(size(train, 1), 1);
for j = 1:size(train, 1)
    closest(j) = sum((train(j, :)-x).^2);
end
closest = sortrows(closest, 1);

k = min(k, size(closest,1));
r = sqrt(closest(k));

if mod(d,2) == 0
	V = ((pi^(d/2))/factorial(d/2))*(r^d);
else
	V = (2^d)*(pi^((d-1)/2))*(factorial((d-1)/2)/factorial(d))*(r^d);
end
