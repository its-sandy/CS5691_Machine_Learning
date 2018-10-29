function retvec = mvn(X, mu, C)
    retvec = ones(size(X, 1), 1)/((2*pi)^(size(X,2)/2));
    retvec = retvec/(det(C)^0.5);
    X = X-mu;
    for i=1:size(X, 1)
        retvec(i) = retvec(i)*exp(-0.5*(X(i,:)/C)*X(i,:)');
    end
end