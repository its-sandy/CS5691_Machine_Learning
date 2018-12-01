data_conn = zeros(0,39);

in_file = fopen(fullfile('..','group25_data_assign3','connected','test1','25','49oa.mfcc'), 'r');
if (in_file == -1) 
    error('oops, file can''t be read'); 
end 
fprintf("opened file %s\n", fullfile('..','group25_data_assign3','connected','test1','25','49oa.mfcc'));
[d, count] = fscanf(in_file, '%d', 1);
if (count == -1)
    fprintf("empty file\n");
end
[num_feature_vectors, ~] = fscanf(in_file, '%d', 1);

for j = 1:num_feature_vectors
    [vec, ~] = fscanf(in_file, '%f', d);
    data_conn = [data_conn; [vec' i]];
end
fclose(in_file);

conn_sequence = [(k_means_test(data_conn(:,1:38),centers)-1) data_conn(:,39)];

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','conn_sequence.seq'), 'w');
for i = 1:size(conn_sequence,1)
    for j = 1:i
        fprintf(out_file,"%d ",conn_sequence(j, 1));   
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

alpha_4 = test_HMM(size(conn_sequence, 1), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
alpha_9 = test_HMM(size(conn_sequence, 1), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
alpha_o = test_HMM(size(conn_sequence, 1), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');