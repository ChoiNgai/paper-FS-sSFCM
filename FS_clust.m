function [new_data,new_label] = FS_clust(data,data_labels,FS_percent,sample_percent) 

%% ѡ�񲿷�����(�ѵ�����Ϊһ������)
% data(:,size(data,2)+1) = data_labels;       %��ǩ���ݲ�������������    
% data = data(randperm(size(data,1)),:) ;     %�����������
% data = data(1: ceil( size(data,1)*sample_percent ),:);  %ѡȡǰ���һ��������
% data_labels = data(:,size(data,2));         %��ǩ
% data(:,size(data,2)) = [];                  %data��ȥ����ǩ

%% Fisher Score �ֶ����ñ���������
%[redu,FS,List] = fsFisher(data,data_labels,FS_percent);     %redu����������

%% �����䷽�������Ӧ����������
[redu,FS,~] = fsFisher(data,data_labels,1);     %redu����������
FS_sort = sort(FS,'descend');
for i = 1:length(redu)
    MSE_FS(i) = mse((FS-FS_sort(i))) ;
end
[~,idx] = min(MSE_FS);
FS_percent = idx/length(FS_sort);       %����Ӧ���������ٷֱ�
[redu,FS,List] = fsFisher(data,data_labels,FS_percent); 

%% ȥ����������
new_data = zeros(size(data,1),length(redu));
new_FS =zeros(length(redu),1);

for ifs = 1:length(redu)
    new_data(:,ifs) = data(:,redu(ifs) );
    new_FS(ifs) = FS(redu(ifs));
end
new_label = data_labels;
