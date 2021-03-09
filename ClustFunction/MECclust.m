function [V_pc,V_pe,V_xb] = MECclust(data, cluster_n)
%% matlab自带fcm(算出来结果稳定，每次都是最好的情况)
%[center, U] = fcm(data, cluster_n);

%% 随机初始隶属度FCM（这个比较符合算法实际情况）
data_n = size(data, 1); % 求出data的第一维(rows)数,即样本个数 [center, U, obj_fcn]= FCMClust(data, cluster_n)
in_n = size(data, 2);   % 求出data的第二维(columns)数，即特征值长度
% 默认操作参数
options = [2; % 隶属度矩阵U的指数
    1000;                % 最大迭代次数
    1e-5;               % 隶属度最小变化量,迭代终止条件
    1];                 % 每次迭代是否输出信息标志
lamda=0.1;  
%将options 中的分量分别赋值给四个变量;
expo = options(1);          % 隶属度矩阵U的指数
max_iter = options(2);  % 最大迭代次数
min_impro = options(3);  % 隶属度最小变化量,迭代终止条件
display = options(4);  % 每次迭代是否输出信息标志
 
obj_fcn = zeros(max_iter, 1); % 初始化输出参数obj_fcn
 
U = initfcm(cluster_n, data_n);     % 初始化模糊分配矩阵,使U满足列上相加为1,cluster_n=2,用户填上去的种类数c=cluster_n
% Main loop  主要循环
for i = 1:max_iter,
    %在第k步循环中改变聚类中心ceneter,和分配函数U的隶属度值;
    [U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo,lamda);
    if display
       fprintf('MEC:Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
    end
 % 终止条件判别
    if i>1
      if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro
            break;
      end
    end
end
 
iter_n = i; % 实际迭代次数
%obj_fcn(iter_n+1:max_iter) = [];
%% 评价指标
[V_pc,~,V_pe,V_xb] = V_pcpexb(U,data,center);
end

 
%% 子函数
function U = initfcm(cluster_n, data_n)
% 初始化fcm的隶属度函数矩阵
% 输入:
%   cluster_n   ---- 聚类中心个数
%   data_n      ---- 样本点数
% 输出：
%   U           ---- 初始化的隶属度矩阵
U = rand(cluster_n, data_n);
col_sum = sum(U);
U = U./col_sum(ones(cluster_n, 1), :);%归一化
end
 
 
%% 子函数
function [U_new, center, obj_fcn] = stepfcm(data, U, cluster_n, expo,lamda)
% 模糊C均值聚类时迭代的一步
% 输入：
%   data        ---- nxm矩阵,表示n个样本,每个样本具有m的维特征值
%   U           ---- 隶属度矩阵
%   cluster_n   ---- 标量,表示聚合中心数目,即类别数
%   expo        ---- 隶属度矩阵U的指数                     
% 输出：
%   U_new       ---- 迭代计算出的新的隶属度矩阵
%   center      ---- 迭代计算出的新的聚类中心
%   obj_fcn     ---- 目标函数值


%distance_Xj_Vk = sum(center)'*ones(1,size(data,1));
%U_new = exp(-(dist.^2)./lamda) ./ exp( - distance_Xj_Vk.^2 ./lamda ) ; %隶属度矩阵

center = U*data./(sum(U,2)*ones(1,size(data,2))); %聚类中心
dist = distfcm(center, data);       %距离矩阵
tmp = exp( -(dist.^2)./lamda);  
U_new = tmp./ (ones(cluster_n, 1)*sum(tmp));
obj_fcn = sum(sum((dist.^2).*U_new))+lamda*( sum (sum( U_new.* log(U_new) )  ) );  % 计算目标函数值 
end
 
 
%% 子函数
function out = distfcm(center, data)
% 计算样本点距离聚类中心的距离
% 输入：
%   center     ---- 聚类中心
%   data       ---- 样本点
% 输出：
%   out        ---- 距离
out = zeros(size(center, 1), size(data, 1));
  for k = 1:size(center, 1) % 对每一个聚类中心
    % 每一次循环求得所有样本点到一个聚类中心的距离
    out(k, :) = sqrt(sum(((data-ones(size(data,1),1)*center(k,:)).^2)',1));
  end
end
