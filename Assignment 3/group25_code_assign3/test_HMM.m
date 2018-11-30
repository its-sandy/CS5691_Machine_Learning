function alpha_out = test_HMM(num_examples, test_file, model_file)
%assumes each row corresponds to a data point and each column corresponds
%to an attribute size(x) = N*d
if ispc
    system(sprintf('bash -c \"hmm-1.04/hmm-1.04/test_hmm %s %s\"', test_file, model_file));
else
    system(sprintf("%s %s %s",fullfile('hmm-1.04','hmm-1.04','./test_hmm'), test_file, model_file));
end
in_file = fopen(fullfile('alphaout'), 'r');
[alpha_out, ~] = fscanf(in_file, '%f', num_examples);
fclose(in_file);
