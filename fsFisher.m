
function [redu,W,List] = fsFisher(X,Y,selectrate)

%Fisher Score, use the N var formulation
%input:   X, the data, each row is an instance
%            Y, the label in 1 2 3 ... format
%           selectrate， 0.1 0.2 0.3...0.9选择的属性数目占总属性数目的比例
%output
%      redu %if selectrate=0.5, redu为top 50% 的属性
%     W%各个特征的Fisher score得分
%     List%各个特征按照Fisher score得分排序
%调用格式 [W,List]=fsFisher(c,d)
numC = max(Y);%类别数目
[~, numF] = size(X);%特征总数numF
m=ceil(selectrate*numF);%最终选择的属性的数目
W = zeros(1,numF);

% statistic for classes
cIDX = cell(numC,1);%cIDX存储属于某一类的样本
n_i = zeros(numC,1);%n_i存储每一类的样本数
for j = 1:numC
    cIDX{j} = find(Y(:)==j);
    n_i(j) = length(cIDX{j});
end

% calculate score for each features
for i = 1:numF%
    temp1 = 0;
    temp2 = 0;
    f_i = X(:,i);
    u_i = mean(f_i);%每一个特征的均值
    
    for j = 1:numC%类别数numC
        u_cj = mean(f_i(cIDX{j}));
        var_cj = var(f_i(cIDX{j}),1);
        temp1 = temp1 + n_i(j) * (u_cj-u_i)^2;
        temp2 = temp2 + n_i(j) * var_cj;
    end
    
%     if temp1 == 0
%         out.W(i) = 0;
%     else
        if temp2 == 0
            W(i) = 1000000;%分母为0，应为正无穷，用一个很大的数代替
        else
            W(i) = temp1/temp2;
        end
%     end
end
[~, List] = sort(W, 'descend');%各个特征的排序
redu=List(1:m);
% redu=[];
% for i=1:m
%     redu=[redu find(List==i)];
% end
end
