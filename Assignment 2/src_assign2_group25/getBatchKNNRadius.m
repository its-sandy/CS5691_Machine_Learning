function R = getBatchKNNRadius(X, train, k)

distMat = pdist2(X, train);
distMat = mink(distMat, k, 2);
size(distMat)
R = distMat(:,k);
