function [data,data_labels]= sample_select(data,data_labels,sample_percent)
%% 选择部分样本
data(:,size(data,2)+1) = data_labels;       %标签数据并入特征数据中    
data = data(randperm(size(data,1)),:) ;     %样本随机排序
data = data(1: ceil( size(data,1)*sample_percent ),:);  %选取前面的一部分样本
data_labels = data(:,size(data,2));         %标签
data(:,size(data,2)) = [];                  %data中去掉标签

%% 选择部分标签
% data(:,size(data,2)+1) = data_labels;       %标签数据并入特征数据中    
% data = data(randperm(size(data,1)),:) ;     %样本随机排序
% data_labels = data(:,size(data,2));         %标签
% data(:,size(data,2)) = [];                  %data中去掉标签
% data_labels = data_labels(1: ceil(length(data_labels)*sample_percent)  );         