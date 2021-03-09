function [new_data,new_label] = FS_clust(data,data_labels,FS_percent,sample_percent) 

%% 选择部分样本(已单独作为一个函数)
% data(:,size(data,2)+1) = data_labels;       %标签数据并入特征数据中    
% data = data(randperm(size(data,1)),:) ;     %样本随机排序
% data = data(1: ceil( size(data,1)*sample_percent ),:);  %选取前面的一部分样本
% data_labels = data(:,size(data,2));         %标签
% data(:,size(data,2)) = [];                  %data中去掉标签

%% Fisher Score 手动设置保留特征数
%[redu,FS,List] = fsFisher(data,data_labels,FS_percent);     %redu：保留特征

%% 最大类间方差——自适应保留特征数
[redu,FS,~] = fsFisher(data,data_labels,1);     %redu：保留特征
FS_sort = sort(FS,'descend');
for i = 1:length(redu)
    MSE_FS(i) = mse((FS-FS_sort(i))) ;
end
[~,idx] = min(MSE_FS);
FS_percent = idx/length(FS_sort);       %自适应保留特征百分比
[redu,FS,List] = fsFisher(data,data_labels,FS_percent); 

%% 去除多余特征
new_data = zeros(size(data,1),length(redu));
new_FS =zeros(length(redu),1);

for ifs = 1:length(redu)
    new_data(:,ifs) = data(:,redu(ifs) );
    new_FS(ifs) = FS(redu(ifs));
end
new_label = data_labels;
