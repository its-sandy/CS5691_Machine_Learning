rng(0);
bN=25;
bM=1;
bL=1e-2;
err=100;
Ermstest=zeros(3,4);
Ermstrain=zeros(3,4);
for N = [20,40,100]
    X = sort(rand(N, 1));
    X2 = sort(rand(N/5, 1));
    Y = awgn(exp(sin(2*pi*X)), 10);
    Y2 = awgn(exp(sin(2*pi*X2)), 10);
    ctr = 1;
    i1=0;
    figure('Name', sprintf('Polynomial Curve Fitting %d',N), 'NumberTitle', 'off');
    for lambda = [1e-2,1e-5,1e-10]
        i1=i1+1;
        i2=0;
        for M=[1,10,20,30]
            i2=i2+1;
            phi = zeros(N, M+1);
            phi2 = zeros(N/5, M+1);
            a = zeros(M+1, M+1);

            for i=1:M+1
                phi(:,i) = X.^(i-1);
                phi2(:,i) = X2.^(i-1);
                for j=1:M+1
                    a(i,j) = sum(X.^(i+j-2));
                end
            end
            c = phi'*Y;
            w = linsolve(a+lambda*eye(M+1), c);
            %means w= pinv(a+lambda*eye(M+1))*c;
            subplot(3, 4, ctr);
            plot(X, Y, 'b.'); hold on
            plot(X2, Y2, 'm.'); hold on
            plot(0:0.01:1, polyval(flipud(w), 0:0.01:1), 'r-'); hold on
            plot(0:0.01:1, exp(sin(2*pi*(0:0.01:1))), 'g-');
            ylim([-1,3.5]);
            title(sprintf("N: %d M: %d\nlambda: %0.1e",N, M,lambda));
            if N==100
                newerrtrn=(lambda*w'*w+sum((phi*w)-Y).^2);
                newerrtst=(lambda*w'*w+sum((phi2*w)-Y2).^2);  
                
                Ermstrain(i1,i2)=sqrt(newerrtrn/100);
                Ermstest(i1,i2)=sqrt(newerrtst/100);
                
                %Erms(i,j)=sqrt(2*newerr/100);
                if newerrtst < err
                    err=newerrtst;
                    bM=M;
                    bL=lambda;
                    btrainX=X;
                    btrainY=Y;
                    btestX=X2;
                    btestY=Y2;
                    bw=w;
                end
            end
            ctr=ctr + 1;
        end
    end
end
figure('Name', sprintf('Polynomial Curve Fitting %d',N), 'NumberTitle', 'off');
plot(btrainX, btrainY, 'b.'); hold on
plot(btestX, btestY, 'm.'); hold on
plot(0:0.01:1, polyval(flipud(bw), 0:0.01:1), 'r-'); hold on
plot(0:0.01:1, exp(sin(2*pi*(0:0.01:1))), 'g-');
Ermstest
Ermstrain



