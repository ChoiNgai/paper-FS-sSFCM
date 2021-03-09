function [Vpc,Vpe,Vxb] = FCM(data,c)  

[t,k]=size(data);

Maxiter=1000;     %�趨����������
n=c;           
% c1=0.01;
% c2=0.02;         %�趨���徭��ϵ����Ⱥ�徭��ϵ��
% w=0.3;          %�趨����ϵ��
% vmax=0.6;       %�趨����ٶ�
%c=3;         %�趨���ӣ��������ģ���Ŀ
e=10^(-5);         %�趨��ֵ
ref=2;          %�趨fcm��ϵ��(��Ȩָ����)
result=zeros(c-1,1);
u=cell(c,n);
vit=cell(c,n);
particle=cell(c,n);
dist=cell(c,n);
obj=zeros(c,n);
pbest=cell(c,1);
pbest_pos=cell(c,n);
gbest=zeros(c,1);
gbest_index=zeros(c,1);
gbest_pos=cell(c,1);


u_new=zeros(t,c);
for i=1:n
    x=randperm(t);
    for j=1:c
        particle{c,i}(j,:)=data(x(j),:);%�����ʼ��������
    end
    
    u{c,i}=zeros(t,c);
    dist{c,i}=distfcm(particle{c,i},data);
    dist{c,i}=dist{c,i}+0.01;
    tmp=dist{c,i}.^(-2/(ref-1));
    u{c,i}=tmp./(ones(c,1)*sum(tmp));
    %���������Ⱦ��󣨳�ʼ��

    [u_new,particle{c,i},obj(c,i)]=stepfcm(data,u{c,i},c,ref);
    u{c,i}=u_new;
    %���������Ⱦ���fcm��
end
iter=1;

fit(c)=1e+9;
% while(iter<=Maxiter&fit(c)>e)

    for i=1:n
        [u{c,i},particle{c,i},obj(c,i)]=stepfcm(data,u{c,i},c,ref);
    end
%     fit(c)=min(obj(c,:));
%     iter
%     iter=iter+1;
% end

[Vpc,~,Vpe,Vxb]=V_pcpexb(cell2mat(u(c)),data, cell2mat(particle(c)))

            
        
        
