function [data,data_labels]= sample_select(data,data_labels,sample_percent)
%% ѡ�񲿷�����
data(:,size(data,2)+1) = data_labels;       %��ǩ���ݲ�������������    
data = data(randperm(size(data,1)),:) ;     %�����������
data = data(1: ceil( size(data,1)*sample_percent ),:);  %ѡȡǰ���һ��������
data_labels = data(:,size(data,2));         %��ǩ
data(:,size(data,2)) = [];                  %data��ȥ����ǩ

%% ѡ�񲿷ֱ�ǩ
% data(:,size(data,2)+1) = data_labels;       %��ǩ���ݲ�������������    
% data = data(randperm(size(data,1)),:) ;     %�����������
% data_labels = data(:,size(data,2));         %��ǩ
% data(:,size(data,2)) = [];                  %data��ȥ����ǩ
% data_labels = data_labels(1: ceil(length(data_labels)*sample_percent)  );         