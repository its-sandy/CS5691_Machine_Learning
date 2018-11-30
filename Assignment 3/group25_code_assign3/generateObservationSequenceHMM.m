rng(1);

num_observation_symbols = 25;
num_states = 15;
min_delta_psum = 0.01;
seed = 1234;

%%%%%%%%%%%%%%%
in_file = fopen(fullfile('..','group25_data_assign3','ocr_data','HandWritten_data','DATA','FeaturesHW','a.ldf'), 'r');
if (in_file == -1) 
    error('oops, file can''t be read'); 
end 

fprintf("opened file\n");
data_a = zeros(0,3);
i=0;
[num_strokes, count] = fscanf(in_file, '%d', 1); 
while (count == 1) % while we have read a number 
%     fprintf('We just read %d\n', number); 
    i=i+1;
    [class, ~] = fscanf(in_file, '%s', 1);
    fprintf("num_strokes = %d, class = %s\n", num_strokes, class);
    [time_length, ~] = fscanf(in_file, '%d', 1); % attempt to read the next number
    
    fprintf("time_length = %d\n", time_length);
    for j = 1:time_length
        [xx, ~] = fscanf(in_file, '%f', 1);
        [yy, ~] = fscanf(in_file, '%f', 1);
        data_a = [data_a; [xx yy i]];
        fprintf("x = %f, y = %f, i = %d\n", xx,yy,i);
    end
    [num_strokes, count] = fscanf(in_file, '%d', 1);
end
fclose(in_file);
N_a = i;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in_file = fopen(fullfile('..','group25_data_assign3','ocr_data','HandWritten_data','DATA','FeaturesHW','ai.ldf'), 'r');
if (in_file == -1) 
    error('oops, file can''t be read'); 
end 

fprintf("opened file\n");
data_ai = zeros(0,3);
i=0;
[num_strokes, count] = fscanf(in_file, '%d', 1); 
while (count == 1) % while we have read a number 
%     fprintf('We just read %d\n', number); 
    i=i+1;
    [class, ~] = fscanf(in_file, '%s', 1);
    fprintf("num_strokes = %d, class = %s\n", num_strokes, class);
    [time_length, ~] = fscanf(in_file, '%d', 1); % attempt to read the next number
    
    fprintf("time_length = %d\n", time_length);
    for j = 1:time_length
        [xx, ~] = fscanf(in_file, '%f', 1);
        [yy, ~] = fscanf(in_file, '%f', 1);
        data_ai = [data_ai; [xx yy i]];
        fprintf("x = %f, y = %f, i = %d\n", xx,yy,i);
    end
    [num_strokes, count] = fscanf(in_file, '%d', 1);
end 
fclose(in_file);
N_ai = i;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in_file = fopen(fullfile('..','group25_data_assign3','ocr_data','HandWritten_data','DATA','FeaturesHW','dA.ldf'), 'r');
if (in_file == -1) 
    error('oops, file can''t be read'); 
end 

fprintf("opened file\n");
data_da = zeros(0,3);
i=0;
[num_strokes, count] = fscanf(in_file, '%d', 1); 
while (count == 1) % while we have read a number 
%     fprintf('We just read %d\n', number); 
    i=i+1;
    [class, ~] = fscanf(in_file, '%s', 1);
    fprintf("num_strokes = %d, class = %s\n", num_strokes, class);
    [time_length, ~] = fscanf(in_file, '%d', 1); % attempt to read the next number
    
    fprintf("time_length = %d\n", time_length);
    for j = 1:time_length
        [xx, ~] = fscanf(in_file, '%f', 1);
        [yy, ~] = fscanf(in_file, '%f', 1);
        data_da = [data_da; [xx yy i]];
        fprintf("x = %f, y = %f, i = %d\n", xx,yy,i);
    end
    [num_strokes, count] = fscanf(in_file, '%d', 1);
end 
fclose(in_file);
N_da = i;
%%%%%%%%%%%%%%%%%%%%%%

fprintf("read input\n");
%%%%%%%%%%%%%%%%%%%%%
[train_ind_a, test_ind_a] = dividerand(N_a, 0.7, 0.3);
train_a = zeros(0,3);
for i = train_ind_a
    train_a = [train_a; data_a(data_a(:,3) == i,:)];
end
test_a = zeros(0,3);
for i = test_ind_a
    test_a = [test_a; data_a(data_a(:,3) == i,:)];
end
%%%%%%%%%%%%%%%%%%%%%
[train_ind_ai, test_ind_ai] = dividerand(N_ai, 0.7, 0.3);
train_ai = zeros(0,3);
for i = train_ind_ai
    train_ai = [train_ai; data_ai(data_ai(:,3) == i,:)];
end
test_ai = zeros(0,3);
for i = test_ind_ai
    test_ai = [test_ai; data_ai(data_ai(:,3) == i,:)];
end
%%%%%%%%%%%%%%%%%%%%%
[train_ind_da, test_ind_da] = dividerand(N_da, 0.7, 0.3);
train_da = zeros(0,3);
for i = train_ind_da
    train_da = [train_da; data_da(data_da(:,3) == i,:)];
end
test_da = zeros(0,3);
for i = test_ind_da
    test_da = [test_da; data_da(data_da(:,3) == i,:)];
end
%%%%%%%%%%%%%%%%%%%%%

fprintf("split data\n");
%%%%%%%%%%%%%%%%%%%%%
clustering_train = [train_a(:,1:2); train_ai(:,1:2); train_da(:,1:2)];
[centers,~,~] = k_means(clustering_train,num_observation_symbols); %try 100

%sequences have to be zero indexed
train_sequence_a = [(k_means_test(train_a(:,1:2),centers)-1) train_a(:,3)];
test_sequence_a = [(k_means_test(test_a(:,1:2),centers)-1) test_a(:,3)];

train_sequence_ai = [(k_means_test(train_ai(:,1:2),centers)-1) train_ai(:,3)];
test_sequence_ai = [(k_means_test(test_ai(:,1:2),centers)-1) test_ai(:,3)];

train_sequence_da = [(k_means_test(train_da(:,1:2),centers)-1) train_da(:,3)];
test_sequence_da = [(k_means_test(test_da(:,1:2),centers)-1) test_da(:,3)];
%%%%%%%%%%%%%%%%%%%%%

fprintf("generated observation sequences\n");
%%%%%%%%%%%%%%%%%%%%%
out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq'), 'w');
for i = train_ind_a
    seq = train_sequence_a(train_sequence_a(:,2) == i, 1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_a.seq'), 'w');
for i = test_ind_a
    seq = test_sequence_a(test_sequence_a(:,2) == i,1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq'), 'w');
for i = train_ind_ai
    seq = train_sequence_ai(train_sequence_ai(:,2) == i, 1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_ai.seq'), 'w');
for i = test_ind_ai
    seq = test_sequence_ai(test_sequence_ai(:,2) == i,1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq'), 'w');
for i = train_ind_da
    seq = train_sequence_da(train_sequence_da(:,2) == i, 1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

out_file = fopen(fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_da.seq'), 'w');
for i = test_ind_da
    seq = test_sequence_da(test_sequence_da(:,2) == i,1);
    for s = seq
        fprintf(out_file,"%d ",s);
    end
    fprintf(out_file,"\n");
end
fclose(out_file);

%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("written sequences to files\n");
%%%%%%%%%%%%%%%%%%%%%


system(sprintf("%s %s %d %d %d %f",fullfile('hmm-1.04','hmm-1.04','./train_hmm'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq'), seed, num_states, num_observation_symbols, min_delta_psum));
system(sprintf("%s %s %d %d %d %f",fullfile('hmm-1.04','hmm-1.04','./train_hmm'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq'), seed, num_states, num_observation_symbols, min_delta_psum));
system(sprintf("%s %s %d %d %d %f",fullfile('hmm-1.04','hmm-1.04','./train_hmm'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq'), seed, num_states, num_observation_symbols, min_delta_psum));
fprintf("models trained\n");

%%%%%%%%%%%%%%%%%%%%%%
test_confusion_matrix = zeros(3,3); % 1-a, 2-ai, 3-da ; rows correspond to actual class

alpha_a = test_HMM(length(test_ind_a), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_a.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq.hmm'));
alpha_ai = test_HMM(length(test_ind_a), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_a.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq.hmm'));
alpha_da = test_HMM(length(test_ind_a), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_a.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq.hmm'));
[~,pred_class] = max([alpha_a alpha_ai alpha_da], [], 2);
test_confusion_matrix(1,1) = test_confusion_matrix(1,1) + sum(pred_class == 1);
test_confusion_matrix(1,2) = test_confusion_matrix(1,2) + sum(pred_class == 2);
test_confusion_matrix(1,3) = test_confusion_matrix(1,3) + sum(pred_class == 3);

alpha_a = test_HMM(length(test_ind_ai), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_ai.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq.hmm'));
alpha_ai = test_HMM(length(test_ind_ai), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_ai.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq.hmm'));
alpha_da = test_HMM(length(test_ind_ai), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_ai.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq.hmm'));
[~,pred_class] = max([alpha_a alpha_ai alpha_da], [], 2);
test_confusion_matrix(2,1) = test_confusion_matrix(2,1) + sum(pred_class == 1);
test_confusion_matrix(2,2) = test_confusion_matrix(2,2) + sum(pred_class == 2);
test_confusion_matrix(2,3) = test_confusion_matrix(2,3) + sum(pred_class == 3);

alpha_a = test_HMM(length(test_ind_da), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_da.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq.hmm'));
alpha_ai = test_HMM(length(test_ind_da), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_da.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq.hmm'));
alpha_da = test_HMM(length(test_ind_da), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','test_sequence_da.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq.hmm'));
[~,pred_class] = max([alpha_a alpha_ai alpha_da], [], 2);
test_confusion_matrix(3,1) = test_confusion_matrix(3,1) + sum(pred_class == 1);
test_confusion_matrix(3,2) = test_confusion_matrix(3,2) + sum(pred_class == 2);
test_confusion_matrix(3,3) = test_confusion_matrix(3,3) + sum(pred_class == 3);

%%%%%%%%%%%%%%%%%%%%%%
train_confusion_matrix = zeros(3,3); % 1-a, 2-ai, 3-da ; rows correspond to actual class

alpha_a = test_HMM(length(train_ind_a), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq.hmm'));
alpha_ai = test_HMM(length(train_ind_a), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq.hmm'));
alpha_da = test_HMM(length(train_ind_a), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq.hmm'));
[~,pred_class] = max([alpha_a alpha_ai alpha_da], [], 2);
train_confusion_matrix(1,1) = train_confusion_matrix(1,1) + sum(pred_class == 1);
train_confusion_matrix(1,2) = train_confusion_matrix(1,2) + sum(pred_class == 2);
train_confusion_matrix(1,3) = train_confusion_matrix(1,3) + sum(pred_class == 3);

alpha_a = test_HMM(length(train_ind_ai), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq.hmm'));
alpha_ai = test_HMM(length(train_ind_ai), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq.hmm'));
alpha_da = test_HMM(length(train_ind_ai), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq.hmm'));
[~,pred_class] = max([alpha_a alpha_ai alpha_da], [], 2);
train_confusion_matrix(2,1) = train_confusion_matrix(2,1) + sum(pred_class == 1);
train_confusion_matrix(2,2) = train_confusion_matrix(2,2) + sum(pred_class == 2);
train_confusion_matrix(2,3) = train_confusion_matrix(2,3) + sum(pred_class == 3);

alpha_a = test_HMM(length(train_ind_da), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_a.seq.hmm'));
alpha_ai = test_HMM(length(train_ind_da), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_ai.seq.hmm'));
alpha_da = test_HMM(length(train_ind_da), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq'), fullfile('hmm-1.04','hmm-1.04','GeneratedSequences_1a','train_sequence_da.seq.hmm'));
[~,pred_class] = max([alpha_a alpha_ai alpha_da], [], 2);
train_confusion_matrix(3,1) = train_confusion_matrix(3,1) + sum(pred_class == 1);
train_confusion_matrix(3,2) = train_confusion_matrix(3,2) + sum(pred_class == 2);
train_confusion_matrix(3,3) = test_confusion_matrix(3,3) + sum(pred_class == 3);

%%%%%%%%%%%%%%%%%%%%
fprintf("finished testing:-\n");
fprintf("train_confusion_matric:-\n");
display(train_confusion_matrix);
fprintf("test_confusion_matric:-\n");
display(test_confusion_matrix);

