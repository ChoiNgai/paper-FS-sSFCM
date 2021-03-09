close all
clear
clc

%% Add path, import data
addpath data
addpath ClustFunction
load seed.txt
load seedlabel.txt
load iris.csv
load irislabel.csv
load wine.csv
load winelabel.csv
load breast.csv
load breastlabel.csv
load digital.csv
load digitallabel.csv
load signdata     %Gesture data set
load signlabel   %Gesture label

%% select data
data = zscore(double(iris));
label = double(irislabel);
datasetname = 'iris';
cluster_num = max(label);

lou = 0.3;

%% Adaptive preserved feature number
sample_percent=0.1:0.1:1;               %(labeled)number of sample
for j = 1:10                            %number of Sample(percent)
    FS_percent=0.1:0.1:1;               %number of feature（percent）
    
    for i = 1:10                        %Iterations (percentage of selected features)
        %%  Original clustering algorithm
        [select_data,select_label]= sample_select(data,label,1);            %Some samples were randomly selected according to the percentage of samples
        select_label(ceil( length(select_label)*sample_percent(j)):length(select_label))=[];   %According to the number of selected samples (rounded up), remove some labels (semi supervised)

        t=cputime;
        [FCM_Vpc(j,i),FCM_Vpe(j,i),FCM_Vxb(j,i)] = FCM(select_data, cluster_num);   %FCM
        time_FCM(j,i)=cputime-t;
        
        t=cputime;
        [SFCM_Vpc(j,i),SFCM_Vpe(j,i),SFCM_Vxb(j,i)] = SFCMclust(select_data, cluster_num,select_label);
        time_SFCM(j,i)=cputime-t;
        
        t=cputime;
        [sSFCM_Vpc(j,i),sSFCM_Vpe(j,i),sSFCM_Vxb(j,i)] = sSFCMclust(select_data, cluster_num,select_label);
        time_sSFCM(j,i)=cputime-t;

        t=cputime;
        [eSFCM_Vpc(j,i),eSFCM_Vpe(j,i),eSFCM_Vxb(j,i)] = eSFCMclust(select_data, cluster_num,select_label);
        time_eSFCM(j,i)=cputime-t;
      
        %% Clustering algorithm based on fisher-score
        [new_data,new_label] = FS_clust(select_data,select_label,FS_percent(i),sample_percent(j));  %特征选择        %这个FS_percent(1)是未使用的参数 
        FS_select(j,i)=cputime;
        
        [FS_FCM_Vpc(j,i),FS_FCM_Vpe(j,i),FS_FCM_Vxb(j,i)] = FCMclust(new_data, cluster_num);
        time_FS_FCM(j,i)=cputime-FS_select(j,i);
        
        T=cputime;
        [FS_SFCM_Vpc(j,i),FS_SFCM_Vpe(j,i),FS_SFCM_Vxb(j,i)] = SFCMclust(new_data,cluster_num,new_label);
        time_FS_SFCM(j,i)=cputime-T;
        
        T=cputime;
        [FS_sSFCM_Vpc(j,i),FS_sSFCM_Vpe(j,i),FS_sSFCM_Vxb(j,i)] = sSFCMclust(new_data,cluster_num,new_label);
        time_FS_sSFCM(j,i)=cputime-T;
        
        T=cputime;
        [FS_eSFCM_Vpc(j,i),FS_eSFCM_Vpe(j,i),FS_eSFCM_Vxb(j,i)] = eSFCMclust(new_data,cluster_num,new_label);
        time_FS_eSFCM(j,i)=cputime-T;

    end
end


%% Vpc with different sample numbers
figure(1)
hold on
plot(sample_percent,mean(SFCM_Vpc'),'-o','LineWidth',2)
plot(sample_percent,mean(sSFCM_Vpc'),'-s','LineWidth',2)
plot(sample_percent,mean(eSFCM_Vpc'),'-*','LineWidth',2)
plot(sample_percent,mean(FS_SFCM_Vpc'),'-h','LineWidth',2)
plot(sample_percent,mean(FS_sSFCM_Vpc'),'-d','LineWidth',2)
plot(sample_percent,mean(FS_eSFCM_Vpc'),'-p','LineWidth',2)

legend('SFCM','sSFCM','eSFCM','FS-SFCM','FS-sSFCM','FS-eSFCM','location','southeast')
set(gca,'XLim',[sample_percent(1) sample_percent(length(sample_percent))]);
%set(gca,'YLim',[0 max(ymax)*1.5]);
xlabel('Percentage of labeled samples')     %标记样本百分比
ylabel('V_{pc}')
title(datasetname)

%% Vpe with different sample numbers
figure(2)
hold on
plot(sample_percent,mean(SFCM_Vpe'),'-o','LineWidth',2)
plot(sample_percent,mean(sSFCM_Vpe'),'-s','LineWidth',2)
plot(sample_percent,mean(eSFCM_Vpe'),'-*','LineWidth',2)

plot(sample_percent,mean(FS_SFCM_Vpe'),'-h','LineWidth',2)
plot(sample_percent,mean(FS_sSFCM_Vpe'),'-d','LineWidth',2)
plot(sample_percent,mean(FS_eSFCM_Vpe'),'-p','LineWidth',2)
ymax = [ max(SFCM_Vpe),max(sSFCM_Vpe), max(FS_SFCM_Vpe), max(FS_sSFCM_Vpe) ];
legend('SFCM','sSFCM','eSFCM','FS-SFCM','FS-sSFCM','FS-eSFCM','location','northeast')
%set(gca,'XLim',[sample_percent(1) sample_percent(length(sample_percent))]);
set(gca,'YLim',[0 max(ymax)*1.5]);
xlabel('Percentage of labeled samples')
ylabel('V_{pe}')
title(datasetname)


%% Vpe with different Rho
figure(3)
hold on
plot(sample_percent,mean(SFCM_Vxb'),'-o','LineWidth',2)
plot(sample_percent,mean(sSFCM_Vxb'),'-s','LineWidth',2)
plot(sample_percent,mean(eSFCM_Vxb'),'-*','LineWidth',2)

plot(sample_percent,mean(FS_SFCM_Vxb'),'-h','LineWidth',2)
plot(sample_percent,mean(FS_sSFCM_Vxb'),'-d','LineWidth',2)
plot(sample_percent,mean(FS_eSFCM_Vxb'),'-p','LineWidth',2)
ymax = [ max(SFCM_Vpe),max(sSFCM_Vpe), max(FS_SFCM_Vpe), max(FS_sSFCM_Vpe) ];
legend('SFCM','sSFCM','eSFCM','FS-SFCM','FS-sSFCM','FS-eSFCM','location','northeast')
set(gca,'XLim',[sample_percent(1) sample_percent(length(sample_percent))]);
%set(gca,'YLim',[0 max(ymax)*1.5]);
xlabel('Percentage of labeled samples')
ylabel('V_{xb}')
title(datasetname)

%% Time consuming of different algorithms
figure(4)
hold on
plot(sample_percent,mean(time_SFCM'),'-o','LineWidth',2)
plot(sample_percent,mean(time_sSFCM'),'-s','LineWidth',2)
plot(sample_percent,mean(time_eSFCM'),'-*','LineWidth',2)

plot(sample_percent,mean(time_FS_SFCM'),'-h','LineWidth',2)
plot(sample_percent,mean(time_FS_sSFCM'),'-d','LineWidth',2)
plot(sample_percent,mean(time_FS_eSFCM'),'-p','LineWidth',2)
ymax = [ max(time_SFCM),max(time_sSFCM), max(time_FS_SFCM), max(time_FS_sSFCM) ];
legend('SFCM','sSFCM','eSFCM','FS-SFCM','FS-sSFCM','FS-eSFCM','location','northeast')
set(gca,'XLim',[sample_percent(1) sample_percent(length(sample_percent))]);
%set(gca,'YLim',[0 max(ymax)*1.5]);
xlabel('Percentage of labeled samples')
ylabel('time(s)')
title(datasetname)
