clear;
clc;

% model bit length
nbits =12;

% train test video, still image num: PB
test_set_num = 10495;
test_still_num = 10495;
train_set_num = 2415;
train_still_num = 7245;

% train test video , still image num: YTC
% test_set_num = 3101;
% test_still_num = 3101;
% train_set_num = 7190;
% train_still_num = 21570;

% train test video, still image num: UMDFaces
% test_sample_num = 3422;
% test_still_num = 3422;
% train_set_num = 6614;
% train_still_num = 19842;

% 
fid1 = fopen('test_feature_still_pb_12','rb');
test_still_feature = fread(fid1,[nbits test_still_num],'float');
fclose(fid1);
fid2 = fopen('test_label_still_pb','rb');
test_still_label = fread(fid2,[1 test_still_num],'float');
test_still_label = test_still_label';
fclose(fid2);
fid3 = fopen('test_feature_pb_12','rb');
test_set_fea = fread(fid3,[nbits test_set_num],'float');
fclose(fid3);
fid4=fopen('test_label_pb','rb');
test_set_label = fread(fid4,[1 test_set_num],'float');
test_set_label = test_set_label';
fclose(fid4);
% 
fid5 = fopen('train_feature_still_pb_12','rb');
train_still_feature = fread(fid5,[nbits train_still_num],'float');
fclose(fid5);
fid6 = fopen('train_label_still_pb','rb');
train_still_label = fread(fid6,[1 train_still_num],'float');
train_still_label = train_still_label';
fclose(fid6);

fid7 = fopen('train_feature_pb_12','rb');
train_set_fea = fread(fid7,[nbits train_set_num],'float');
fclose(fid7);
fid8 = fopen('train_label_pb','rb');
train_set_label = fread(fid8,[1 train_set_num],'float');
train_set_label = train_set_label';
fclose(fid8);

test_still_binary = zeros(nbits,test_still_num);
test_still_binary(test_still_feature>=0.5) = 1;
test_still_binary(test_still_feature<0.5) = 0;
test_set_binary = zeros(nbits,test_set_num);
test_set_binary(test_set_fea>=0.5) = 1;
test_set_binary(test_set_fea<0.5)= 0;

train_still_binary = zeros(nbits,train_still_num);
train_still_binary(train_still_feature>=0.5) = 1;
train_still_binary(train_still_feature<0.5) = 0;
train_set_binary = zeros(nbits,train_set_num);
train_set_binary(train_set_fea>=0.5) = 1;
train_set_binary(train_set_fea<0.5) = 0;

% YTC evalutation
% s_v_dis_mtx = pdist2(test_still_binary',train_set_binary', 'hamming');
% s_v_map = compute_map(s_v_dis_mtx',test_still_label,train_set_label)
% v_s_dis_mtx = pdist2(test_set_binary',train_still_binary');
% v_s_map = compute_map(v_s_dis_mtx',test_set_label,train_still_label)

% save(['cross_s_v_dis_mtx_ytc_DHH_',num2str(nbits)],'s_v_dis_mtx');
% save(['cross_v_s_dis_mtx_ytc_DHH_',num2str(nbits)],'v_s_dis_mtx');

% PB evaluation
s_v_dis_mtx = pdist2(train_still_binary',test_set_binary', 'hamming');
s_v_map = compute_map(s_v_dis_mtx',train_still_label,test_set_label)
v_s_dis_mtx = pdist2(train_set_binary',test_still_binary', 'hamming');
v_s_map = compute_map(v_s_dis_mtx',train_set_label,test_still_label)

% save(['cross_s_v_dis_mtx_pb_DHH_',num2str(nbits)],'s_v_dis_mtx');
% save(['cross_v_s_dis_mtx_pb_DHH_',num2str(nbits)],'v_s_dis_mtx');

% UMDFaces evaluation
% s_v_dis_mtx = pdist2(test_still_binary',train_set_binary','hamming');
% s_v_map = compute_map(s_v_dis_mtx',test_still_lab',train_set_lab')
% v_s_dis_mtx = pdist2(test_set_binary',train_still_binary','hamming');
% v_s_map = compute_map(v_s_dis_mtx',test_set_lab',train_still_lab')

% save(['cross_s_v_dis_mtx_umdface_DHH_',num2str(nbits)],'s_v_dis_mtx');
% save(['cross_v_s_dis_mtx_umdface_DHH_',num2str(nbits)],'v_s_dis_mtx');

