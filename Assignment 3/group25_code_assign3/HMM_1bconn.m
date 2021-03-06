data_conn = zeros(0,39);
file = '94a.mfcc';
in_file = fopen(fullfile('..','group25_data_assign3','connected','test1','25',file), 'r');
if (in_file == -1) 
    error('oops, file can''t be read'); 
end 
fprintf("opened file %s\n", fullfile('..','group25_data_assign3','connected','test1','25',file));
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

best = -inf;
i1best = -1;
i2best = -1;
strbest = '';
mapch = ['4', '9', 'o'];

tot = 0;
out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','conn_sequence.seq'), 'w');
for i1 = 1:size(conn_sequence,1)-2
    for i2 = i1+1:size(conn_sequence,1)-1
        for j = 1:size(conn_sequence,1)
            fprintf(out_file,"%d ",conn_sequence(j, 1));
            if j==i1 || j==i2 || j==size(conn_sequence,1)
                fprintf(out_file,"\n");
                tot = tot+1;
            end
        end
    end
end
fclose(out_file);

alpha_4 = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
alpha_9 = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
alpha_o = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');

ctr = 1;
for i1 = 1:size(conn_sequence,1)-2
    for i2 = i1+1:size(conn_sequence,1)-1
        [val, ind] = max([alpha_4(ctr:ctr+2), alpha_9(ctr:ctr+2), alpha_o(ctr:ctr+2)], [], 2);
        if sum(val)>best
            i1best = i1;
            i2best = i2;
            strbest = mapch(ind);
            best = sum(val);
        end
        ctr = ctr+3;
    end
end

best3 = best;
strbest3 = strbest;
i1best3 = i1best;
i2best3 = i2best;
%%%%%%%%%%%%%%%%
best = -inf;
i1best = -1;
i2best = -1;
strbest = '';

tot = 0;
out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','conn_sequence.seq'), 'w');
for i1 = 1:size(conn_sequence,1)-1
    for j = 1:size(conn_sequence,1)
        fprintf(out_file,"%d ",conn_sequence(j, 1));
        if j==i1 || j==size(conn_sequence,1)
            fprintf(out_file,"\n");
            tot = tot+1;
        end
    end
end
fclose(out_file);

alpha_4 = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
alpha_9 = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
alpha_o = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');

ctr = 1;
for i1 = 1:size(conn_sequence,1)-1
    [val, ind] = max([alpha_4(ctr:ctr+1), alpha_9(ctr:ctr+1), alpha_o(ctr:ctr+1)], [], 2);
    if sum(val)>best
        i1best = i1;
        i2best = -1;
        strbest = mapch(ind);
        best = sum(val);
    end
    ctr = ctr+2;
end

best2 = best;
strbest2 = strbest;
i1best2 = i1best;
i2best2 = i2best;

% fprintf('Best string is %d, %d: %s (%f)\n', i1best, i2best, strbest, best);
fprintf('Best 2 digit string is %d, %d: %s (%f)\n', i1best2, i2best2, strbest2, best2);
fprintf('Best 3 digit string is %d, %d: %s (%f)\n', i1best3, i2best3, strbest3, best3);