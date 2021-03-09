function [V_pc,V_pe,V_xb] = FCMclust(data, cluster_n)
%% matlab�Դ�fcm(���������ȶ���ÿ�ζ�����õ����)
%[center, U] = fcm(data, cluster_n);

%% �����ʼ������FCM������ȽϷ����㷨ʵ�������
data_n = size(data, 1); % ���data�ĵ�һά(rows)��,���������� [center, U, obj_fcn]= FCMClust(data, cluster_n)
in_n = size(data, 2);   % ���data�ĵڶ�ά(columns)����������ֵ����
% Ĭ�ϲ�������
options = [2; % �����Ⱦ���U��ָ��
    1000;                % ����������
    1e-5;               % ��������С�仯��,������ֹ����
    1];                 % ÿ�ε����Ƿ������Ϣ��־
  
%��options �еķ����ֱ�ֵ���ĸ�����;
expo = options(1);          % �����Ⱦ���U��ָ��
max_iter = options(2);  % ����������
min_impro = options(3);  % ��������С�仯��,������ֹ����
display = options(4);  % ÿ�ε����Ƿ������Ϣ��־
 
obj_fcn = zeros(max_iter, 1); % ��ʼ���������obj_fcn
 
U = initfcm(cluster_n, data_n);     % ��ʼ��ģ���������,ʹU�����������Ϊ1,cluster_n=2,�û�����ȥ��������c=cluster_n
% Main loop  ��Ҫѭ��
for i = 1:max_iter
    %�ڵ�k��ѭ���иı��������ceneter,�ͷ��亯��U��������ֵ;
    [U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo);
    if display
       fprintf('FCM:Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
    end
 % ��ֹ�����б�
    if i>1
      if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro
            break;
      end
    end
end
 
iter_n = i; % ʵ�ʵ�������
%obj_fcn(iter_n+1:max_iter) = [];
%% ����ָ��
[V_pc,~,V_pe,V_xb] = V_pcpexb(U,data,center)
end

 
%% �Ӻ���
function U = initfcm(cluster_n, data_n)
% ��ʼ��fcm�������Ⱥ�������
% ����:
%   cluster_n   ---- �������ĸ���
%   data_n      ---- ��������
% �����
%   U           ---- ��ʼ���������Ⱦ���
U = rand(cluster_n, data_n);
col_sum = sum(U);
U = U./col_sum(ones(cluster_n, 1), :);%��һ��
end
 
 
%% �Ӻ���
function [U_new, center, obj_fcn] = stepfcm(data, U, cluster_n, expo)
% ģ��C��ֵ����ʱ������һ��
% ���룺
%   data        ---- nxm����,��ʾn������,ÿ����������m��ά����ֵ
%   U           ---- �����Ⱦ���
%   cluster_n   ---- ����,��ʾ�ۺ�������Ŀ,�������
%   expo        ---- �����Ⱦ���U��ָ��                     
% �����
%   U_new       ---- ������������µ������Ⱦ���
%   center      ---- ������������µľ�������
%   obj_fcn     ---- Ŀ�꺯��ֵ
mf = U.^expo;       % �����Ⱦ������ָ��������
center = mf*data./((ones(size(data, 2), 1)*sum(mf'))'); % �¾�������(5.4)ʽ
dist = distfcm(center, data);       % ����������
obj_fcn = sum(sum((dist.^2).*mf));  % ����Ŀ�꺯��ֵ (5.1)ʽ
tmp = dist.^(-2/(expo-1));    
U_new = tmp./(ones(cluster_n, 1)*sum(tmp));  % �����µ������Ⱦ��� (5.3)ʽ
 
end
 
 
%% �Ӻ���
function out = distfcm(center, data)
% �������������������ĵľ���
% ���룺
%   center     ---- ��������
%   data       ---- ������
% �����
%   out        ---- ����
out = zeros(size(center, 1), size(data, 1));
  for k = 1:size(center, 1) % ��ÿһ����������
    % ÿһ��ѭ��������������㵽һ���������ĵľ���
    out(k, :) = sqrt(sum(((data-ones(size(data,1),1)*center(k,:)).^2)',1));
  end
end
