% since prior probabilites are same, we directly compare posterior
% probabilities

rng(1);

num_observation_symbols = 25;
num_states = 4; %might have to be number of phonemes
min_delta_psum = 0.01;
seed = 1234;

%%%%%%%%%%%%%%%
%class 4
data_4 = zeros(0,39); % last column has example number
files = dir(fullfile('..','group25_data_assign3','isolated','25','4','*.mfcc'));

i=0;
for m = 1:size(files,1)
    i = i+1;
    in_file = fopen(fullfile('..','group25_data_assign3','isolated','25','4',files(m).name), 'r');
    if (in_file == -1) 
        error('oops, file can''t be read'); 
    end 
    fprintf("opened file %s\n", fullfile('..','group25_data_assign3','isolated','25','4',files(m).name));
    [d, count] = fscanf(in_file, '%d', 1);
    if (count == -1)
        fprintf("empty file\n");
    end
    [num_feature_vectors, ~] = fscanf(in_file, '%d', 1);
    
    for j = 1:num_feature_vectors
        [vec, ~] = fscanf(in_file, '%f', d);
        data_4 = [data_4; [vec' i]];
    end
    fclose(in_file);
end
N_4 = i;

%%%%%%%%%%%%%%%
%class 9
data_9 = zeros(0,39); % last column has example number
files = dir(fullfile('..','group25_data_assign3','isolated','25','9','*.mfcc'));

i=0;
for m = 1:size(files,1)
    i = i+1;
    in_file = fopen(fullfile('..','group25_data_assign3','isolated','25','9',files(m).name), 'r');
    if (in_file == -1) 
        error('oops, file can''t be read'); 
    end 
    fprintf("opened file %s\n", fullfile('..','group25_data_assign3','isolated','25','9',files(m).name));
    [d, count] = fscanf(in_file, '%d', 1);
    if (count == -1)
        fprintf("empty file\n");
    end
    [num_feature_vectors, ~] = fscanf(in_file, '%d', 1);
    
    for j = 1:num_feature_vectors
        [vec, ~] = fscanf(in_file, '%f', d);
        data_9 = [data_9; [vec' i]];
    end
    fclose(in_file);
end
N_9 = i;

%%%%%%%%%%%%%%%
%class o
data_o = zeros(0,39); % last column has example number
files = dir(fullfile('..','group25_data_assign3','isolated','25','o','*.mfcc'));

i=0;
for m = 1:size(files,1)
    i = i+1;
    in_file = fopen(fullfile('..','group25_data_assign3','isolated','25','o',files(m).name), 'r');
    if (in_file == -1) 
        error('oops, file can''t be read'); 
    end 
    fprintf("opened file %s\n", fullfile('..','group25_data_assign3','isolated','25','o',files(m).name));
    [d, count] = fscanf(in_file, '%d', 1);
    if (count == -1)
        fprintf("empty file\n");
    end
    [num_feature_vectors, ~] = fscanf(in_file, '%d', 1);
    
    for j = 1:num_feature_vectors
        [vec, ~] = fscanf(in_file, '%f', d);
        data_o = [data_o; [vec' i]];
    end
    fclose(in_file);
end
N_o = i;

%%%%%%%%%%%%%%%%%%%%%%

fprintf("read input\n");
%%%%%%%%%%%%%%%%%%%%%

[train_ind_4, ~, test_ind_4] = dividerand(N_4, 0.7, 0, 0.3);
train_4 = zeros(0,39);
for i = train_ind_4
    train_4 = [train_4; data_4(data_4(:,39) == i,:)];
end
test_4 = zeros(0,39);
for i = test_ind_4
    test_4 = [test_4; data_4(data_4(:,39) == i,:)];
end
%%%%%%%%%%%%%%%%%%%%%
[train_ind_9, ~, test_ind_9] = dividerand(N_9, 0.7, 0, 0.3);
train_9 = zeros(0,39);
for i = train_ind_9
    train_9 = [train_9; data_9(data_9(:,39) == i,:)];
end
test_9 = zeros(0,39);
for i = test_ind_9
    test_9 = [test_9; data_9(data_9(:,39) == i,:)];
end
%%%%%%%%%%%%%%%%%%%%%
[train_ind_o, ~, test_ind_o] = dividerand(N_o, 0.7, 0, 0.3);
train_o = zeros(0,39);
for i = train_ind_o
    train_o = [train_o; data_o(data_o(:,39) == i,:)];
end
test_o = zeros(0,39);
for i = test_ind_o
    test_o = [test_o; data_o(data_o(:,39) == i,:)];
end
%%%%%%%%%%%%%%%%%%%%%

fprintf("split data\n");
%%%%%%%%%%%%%%%%%%%%%
clustering_train = [train_4(:,1:38); train_9(:,1:38); train_o(:,1:38)];
[centers,~,~] = k_means(clustering_train,num_observation_symbols); %try 100

%sequences have to be zero indexed
train_sequence_4 = [(k_means_test(train_4(:,1:38),centers)-1) train_4(:,39)];
test_sequence_4 = [(k_means_test(test_4(:,1:38),centers)-1) test_4(:,39)];

train_sequence_9 = [(k_means_test(train_9(:,1:38),centers)-1) train_9(:,39)];
test_sequence_9 = [(k_means_test(test_9(:,1:38),centers)-1) test_9(:,39)];

train_sequence_o = [(k_means_test(train_o(:,1:38),centers)-1) train_o(:,39)];
test_sequence_o = [(k_means_test(test_o(:,1:38),centers)-1) test_o(:,39)];
%%%%%%%%%%%%%%%%%%%%%

fprintf("generated observation sequences\n");
%%%%%%%%%%%%%%%%%%%%%
out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq'), 'w');
for i = train_ind_4
    seq = train_sequence_4(train_sequence_4(:,2) == i, 1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_4.seq'), 'w');
for i = test_ind_4
    seq = test_sequence_4(test_sequence_4(:,2) == i,1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq'), 'w');
for i = train_ind_9
    seq = train_sequence_9(train_sequence_9(:,2) == i, 1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_9.seq'), 'w');
for i = test_ind_9
    seq = test_sequence_9(test_sequence_9(:,2) == i,1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq'), 'w');
for i = train_ind_o
    seq = train_sequence_o(train_sequence_o(:,2) == i, 1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_o.seq'), 'w');
for i = test_ind_o
    seq = test_sequence_o(test_sequence_o(:,2) == i,1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("written sequences to files\n");
%%%%%%%%%%%%%%%%%%%%%

if ispc
    system(sprintf('bash -c \"hmm-1.04/hmm-1.04/train_hmm hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq %d %d %d %f\"',seed, num_states, num_observation_symbols, min_delta_psum));
    system(sprintf('bash -c \"hmm-1.04/hmm-1.04/train_hmm hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq %d %d %d %f\"',seed, num_states, num_observation_symbols, min_delta_psum));
    system(sprintf('bash -c \"hmm-1.04/hmm-1.04/train_hmm hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq %d %d %d %f\"',seed, num_states, num_observation_symbols, min_delta_psum));
else
    system(sprintf("%s %s %d %d %d %f",fullfile('hmm-1.04','hmm-1.04','./train_hmm'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq'), seed, num_states, num_observation_symbols, min_delta_psum));
    system(sprintf("%s %s %d %d %d %f",fullfile('hmm-1.04','hmm-1.04','./train_hmm'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq'), seed, num_states, num_observation_symbols, min_delta_psum));
    system(sprintf("%s %s %d %d %d %f",fullfile('hmm-1.04','hmm-1.04','./train_hmm'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq'), seed, num_states, num_observation_symbols, min_delta_psum));
end
fprintf("models trained\n");

%%%%%%%%%%%%%%%%%%%%%%
test_confusion_matrix = zeros(3,3); % 1-4, 2-9, 3-o ; rows correspond to actual class

if ispc
    alpha_4 = test_HMM(length(test_ind_4), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_4.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
    alpha_9 = test_HMM(length(test_ind_4), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_4.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
    alpha_o = test_HMM(length(test_ind_4), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_4.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');
else
    alpha_4 = test_HMM(length(test_ind_4), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_4.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq.hmm'));
    alpha_9 = test_HMM(length(test_ind_4), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_4.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq.hmm'));
    alpha_o = test_HMM(length(test_ind_4), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_4.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq.hmm'));
end
[~,pred_class] = max([alpha_4 alpha_9 alpha_o], [], 2);
test_confusion_matrix(1,1) = test_confusion_matrix(1,1) + sum(pred_class == 1);
test_confusion_matrix(1,2) = test_confusion_matrix(1,2) + sum(pred_class == 2);
test_confusion_matrix(1,3) = test_confusion_matrix(1,3) + sum(pred_class == 3);

if ispc
    alpha_4 = test_HMM(length(test_ind_9), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_9.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
    alpha_9 = test_HMM(length(test_ind_9), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_9.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
    alpha_o = test_HMM(length(test_ind_9), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_9.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');
else
    alpha_4 = test_HMM(length(test_ind_9), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_9.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq.hmm'));
    alpha_9 = test_HMM(length(test_ind_9), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_9.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq.hmm'));
    alpha_o = test_HMM(length(test_ind_9), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_9.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq.hmm'));
end
[~,pred_class] = max([alpha_4 alpha_9 alpha_o], [], 2);
test_confusion_matrix(2,1) = test_confusion_matrix(2,1) + sum(pred_class == 1);
test_confusion_matrix(2,2) = test_confusion_matrix(2,2) + sum(pred_class == 2);
test_confusion_matrix(2,3) = test_confusion_matrix(2,3) + sum(pred_class == 3);

if ispc
    alpha_4 = test_HMM(length(test_ind_o), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_o.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
    alpha_9 = test_HMM(length(test_ind_o), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_o.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
    alpha_o = test_HMM(length(test_ind_o), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/test_sequence_o.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');
else
    alpha_4 = test_HMM(length(test_ind_o), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_o.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq.hmm'));
    alpha_9 = test_HMM(length(test_ind_o), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_o.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq.hmm'));
    alpha_o = test_HMM(length(test_ind_o), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','test_sequence_o.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq.hmm'));
end
[~,pred_class] = max([alpha_4 alpha_9 alpha_o], [], 2);
test_confusion_matrix(3,1) = test_confusion_matrix(3,1) + sum(pred_class == 1);
test_confusion_matrix(3,2) = test_confusion_matrix(3,2) + sum(pred_class == 2);
test_confusion_matrix(3,3) = test_confusion_matrix(3,3) + sum(pred_class == 3);

%%%%%%%%%%%%%%%%%%%%%%
train_confusion_matrix = zeros(3,3); % 1-4, 2-9, 3-o ; rows correspond to actual class

if ispc
    alpha_4 = test_HMM(length(train_ind_4), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
    alpha_9 = test_HMM(length(train_ind_4), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
    alpha_o = test_HMM(length(train_ind_4), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');
else
    alpha_4 = test_HMM(length(train_ind_4), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq.hmm'));
    alpha_9 = test_HMM(length(train_ind_4), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq.hmm'));
    alpha_o = test_HMM(length(train_ind_4), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq.hmm'));
end
[~,pred_class] = max([alpha_4 alpha_9 alpha_o], [], 2);
train_confusion_matrix(1,1) = train_confusion_matrix(1,1) + sum(pred_class == 1);
train_confusion_matrix(1,2) = train_confusion_matrix(1,2) + sum(pred_class == 2);
train_confusion_matrix(1,3) = train_confusion_matrix(1,3) + sum(pred_class == 3);

if ispc
    alpha_4 = test_HMM(length(train_ind_9), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
    alpha_9 = test_HMM(length(train_ind_9), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
    alpha_o = test_HMM(length(train_ind_9), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');
else
    alpha_4 = test_HMM(length(train_ind_9), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq.hmm'));
    alpha_9 = test_HMM(length(train_ind_9), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq.hmm'));
    alpha_o = test_HMM(length(train_ind_9), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq.hmm'));
end
[~,pred_class] = max([alpha_4 alpha_9 alpha_o], [], 2);
train_confusion_matrix(2,1) = train_confusion_matrix(2,1) + sum(pred_class == 1);
train_confusion_matrix(2,2) = train_confusion_matrix(2,2) + sum(pred_class == 2);
train_confusion_matrix(2,3) = train_confusion_matrix(2,3) + sum(pred_class == 3);

if ispc
    alpha_4 = test_HMM(length(train_ind_o), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_4.seq.hmm');
    alpha_9 = test_HMM(length(train_ind_o), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_9.seq.hmm');
    alpha_o = test_HMM(length(train_ind_o), 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq', 'hmm-1.04/hmm-1.04/GeneratedSequences_1b/train_sequence_o.seq.hmm');
else
    alpha_4 = test_HMM(length(train_ind_o), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_4.seq.hmm'));
    alpha_9 = test_HMM(length(train_ind_o), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_9.seq.hmm'));
    alpha_o = test_HMM(length(train_ind_o), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1b','train_sequence_o.seq.hmm'));
end
[~,pred_class] = max([alpha_4 alpha_9 alpha_o], [], 2);
train_confusion_matrix(3,1) = train_confusion_matrix(3,1) + sum(pred_class == 1);
train_confusion_matrix(3,2) = train_confusion_matrix(3,2) + sum(pred_class == 2);
train_confusion_matrix(3,3) = test_confusion_matrix(3,3) + sum(pred_class == 3);

%%%%%%%%%%%%%%%%%%%%
fprintf("finished testing:-\n");
fprintf("train_confusion_matrix:-\n");
display(train_confusion_matrix);
fprintf("test_confusion_matrix:-\n");
display(test_confusion_matrix);




