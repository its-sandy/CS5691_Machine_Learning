l = 5; % max string length



file = '949a.mfcc';
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

alpha_1dig = zeros(num_feature_vectors);
word_1dig = string(zeros(num_feature_vectors));
tot = 0;
for i1 = 1:num_feature_vectors
    for i2 = i1:num_feature_vectors
        tot = tot+1;
        [val, ind] = max([alpha_4(tot), alpha_9(tot), alpha_o(tot)], [], 2);
        word_1dig(i1,i2) = mapch(ind);
        alpha_1dig(i1,i2) = val;
    end
end

%%%%%%%%%%%%%%%%
%we check till l

alphas = zeros(num_feature_vectors,l);
words = string(zeros(num_feature_vectors,l));
cut_indices = zeros(num_feature_vectors,l);

for i = 1:num_feature_vectors
    alphas(i,1) = alpha_1dig(1,i);
    words(i,1) = word_1dig(1,i);
    cut_indices(i,1) = 1;
end

for dig = 2:l
    for i = dig:num_feature_vectors
        best = -inf;
        strbest = "0";
        for j = dig-1:i-1
            val = alphas(j,dig-1) + alpha_1dig(j+1,i);
            if val > best
                strbest = words(j,dig-1) + word_1dig(j+1,i);
                best = val;
                cut_indices(i,dig) = j+1;
            end
        end
        alphas(i,dig) = best;
        words(i,dig) = strbest;
    end
end
%%%%%%%%%%%%%%%

fprintf("sequence length = %d\n", num_feature_vectors);
for dig = 1:l
    fprintf('Best %d digit string is : %s (%f)\n', dig, words(num_feature_vectors,dig), alphas(num_feature_vectors,dig));
    fprintf("The subsequences start at the following indices:- ");
    v = num_feature_vectors;
    cuts = zeros(dig,1);
    for i = dig:-1:1
        cuts(i) = cut_indices(v,i);
        v = cuts(i) - 1;
    end
    for i = 1:dig
        fprintf("%d ",cuts(i));
    end
    fprintf("\n");
end

%%%%%%%%%%%%%%%
%finding value for 949
% 
n = num_feature_vectors;
index_matrix = zeros(n);
tot = 0;
for i1 = 1:n
    for i2 = i1:n
        tot = tot+1;
        index_matrix(i1,i2) = tot;
    end
end

best_alpha = -inf;
for i1 = 1:n-2
    for i2 = i1+1:n-1
        val = alpha_9(index_matrix(1,i1)) + alpha_4(index_matrix(i1+1,i2)) + alpha_9(index_matrix(i2+1,n));
        if val > best_alpha
            best_alpha = val;
        end
    end
end
fprintf("alpha for 949 = %f\n",best_alpha);

%%%%%%%%%%%%%%%%%%%%