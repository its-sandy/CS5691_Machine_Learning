
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

data_conn = zeros(0,38);
for j = 1:num_feature_vectors
    [vec, ~] = fscanf(in_file, '%f', d);
    data_conn = [data_conn; vec'];
end
fclose(in_file);

conn_sequence = [k_means_test(data_conn,centers)-1];
mapch = ["4"; "9"; "o"];
%%%%%%%%%%%%%%%%
% 1 digit
tot = 0;
out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','conn_sequence.seq'), 'w');
for i1 = 1:num_feature_vectors
    for i2 = i1:num_feature_vectors
        tot = tot+1;
        for j = i1:i2
            fprintf(out_file,"%d ",conn_sequence(j));
        end
        fprintf(out_file,"\n");
    end
end
fclose(out_file);

alpha_4 = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
alpha_9 = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
alpha_o = test_HMM(tot, 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/conn_sequence.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');

alpha_1 = zeros(num_feature_vectors);
word_1 = string(zeros(num_feature_vectors));
tot = 0;
for i1 = 1:num_feature_vectors
    for i2 = i1:num_feature_vectors
        tot = tot+1;
        [val, ind] = max([alpha_4(tot), alpha_9(tot), alpha_o(tot)], [], 2);
        word_1(i1,i2) = mapch(ind);
        alpha_1(i1,i2) = val;
    end
end

%%%%%%%%%%%%%%%%
% 2 digit

alpha_2 = zeros(num_feature_vectors);
word_2 = string(zeros(num_feature_vectors));

for i1 = 1:num_feature_vectors-1
    for i2 = i1+1:num_feature_vectors
        best = -inf;
        strbest = "0";
        for j = i1:i2-1
            val = alpha_1(i1,j) + alpha_1(j+1,i2);
            if val > best
                strbest = word_1(i1,j) + word_1(j+1,i2);
                best = val;
            end
        end
        word_2(i1,i2) = strbest;
        alpha_2(i1,i2) = best;
    end
end

%%%%%%%%%%%%%%%%
% 3 digit

alpha_3 = zeros(num_feature_vectors);
word_3 = string(zeros(num_feature_vectors));

for i1 = 1:num_feature_vectors-2
    for i2 = i1+2:num_feature_vectors
        best = -inf;
        strbest = "0";
        for j = i1+1:i2-1
            val = alpha_2(i1,j) + alpha_1(j+1,i2);
            if val > best
                strbest = word_2(i1,j) + word_1(j+1,i2);
                best = val;
            end
        end
        word_3(i1,i2) = strbest;
        alpha_3(i1,i2) = best;
    end
end

%%%%%%%%%%%%%%%%
% 4 digit

alpha_4 = zeros(num_feature_vectors);
word_4 = string(zeros(num_feature_vectors));

for i1 = 1:num_feature_vectors-3
    for i2 = i1+3:num_feature_vectors
        best = -inf;
        strbest = "0";
        for j = i1+2:i2-1
            val = alpha_3(i1,j) + alpha_1(j+1,i2);
            if val > best
                strbest = word_3(i1,j) + word_1(j+1,i2);
                best = val;
            end
        end
        word_4(i1,i2) = strbest;
        alpha_4(i1,i2) = best;
    end
end

%%%%%%%%%%%%%%%

fprintf('Best 1 digit string is : %s (%f)\n', word_1(1,num_feature_vectors), alpha_1(1,num_feature_vectors));
fprintf('Best 2 digit string is : %s (%f)\n', word_2(1,num_feature_vectors), alpha_2(1,num_feature_vectors));
fprintf('Best 3 digit string is : %s (%f)\n', word_3(1,num_feature_vectors), alpha_3(1,num_feature_vectors));
fprintf('Best 4 digit string is : %s (%f)\n', word_4(1,num_feature_vectors), alpha_4(1,num_feature_vectors));