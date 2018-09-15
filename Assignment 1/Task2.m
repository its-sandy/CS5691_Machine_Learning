N = 30;
M = 20;
lambda = 1e-5;

rng(0);
X = sort(rand(N, 1));
Y = awgn(exp(sin(2*pi*X)), 10);
plot(X, Y, 'bo'); hold on

phi = zeros(N, M+1);
a = zeros(M+1, M+1);

for i=1:M+1
    phi(:,i) = X.^(i-1);
    for j=1:M+1
        a(i,j) = sum(X.^(i+j-2));
    end
end

c = phi'*Y;

w = linsolve(a+lambda*eye(M+1), c);
plot(0:0.01:1, polyval(flipud(w), 0:0.01:1), 'r-'); hold on
plot(0:0.01:1, exp(sin(2*pi*(0:0.01:1))), 'g-');
