function pred = VaryingLengthKNN(XCells, TrainCells, k)
%Each cell in XCells is a test example containing a varying number of feature vectors
%Each cell in TrainCells is the set of feature vectors of a particular class
for j = 1:size(XCells, 1)
	XCells{j} = [repmat(j,size(XCells{j},1),1) XCells{j}];
end
X_test = cell2mat(XCells); 

likelihood = zeros(size(XCells,1), size(TrainCells,1));

for i = 1:size(TrainCells,1)
	ptemp = getBatchKNNRadius(X_test(:,2:end), TrainCells{i}, k);
	ptemp = -log(ptemp);

	for j = 1:size(XCells,1)
		likelihood(i,j) = sum(ptemp(X_test(:,1) == j));
	end
end
[~,pred] = max(likelihood,[],2);
