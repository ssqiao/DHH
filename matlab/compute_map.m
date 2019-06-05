function [map] = compute_map(dis_mtx, query_label, database_label)
tic;

q_num = length(query_label);%查询的次数
d_num = length(database_label);%数据库的规模
map = zeros(q_num, 1);%每一个查询对应一个AP

database_label_mtx = repmat(database_label, 1, q_num);%每个数据库中的样本的label
sorted_database_label_mtx = database_label_mtx;%用于记录每个检索排序后每个样本的label

[mtx idx_mtx] = sort(dis_mtx, 1);%对每次检索的结果进行排序（按列进行）

for q = 1 : q_num
    sorted_database_label_mtx(:, q) = database_label_mtx(idx_mtx(:, q), q);%label重新排序
end

result_mtx = (sorted_database_label_mtx == repmat(query_label', d_num, 1));%考察检索结果每个位置的样本是否和查询标签一致，0/1

for q = 1 : q_num
    Qi = sum(result_mtx(:, q));%检索结果中一共有多少个真正和查询相关的结果
    map(q) = sum( ([1:Qi]') ./ (find(result_mtx(:, q) == 1)) ) / Qi;%计算所有与查询相关的位置的Precision，然后累加求和，最后除以总数，得到AP
end

map = mean(map);
toc;
end