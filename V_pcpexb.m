function [V_pc,V_pe_10,V_pe_e,V_xb]=V_pcpexb(u,data,center)
%���ۺ���ָ�� ����ϵ��V_pc��������V_pe

%% u�������Ⱥ���
[m,n]=size(u);
%% ����ϵ��V_pc
V_pc = sum(sum(u.^2))/n;

%% ������V_pe
V_pe_10=-sum(sum(u.*log10(u)))/n;
%V_pe_e=-sum(sum(u.*log(u)))/n;
taiyingle = u.*log(u);      %��Щ��������1���㲻����
[i,j] = find(isnan(taiyingle));
taiyingle(i,j) = 0;
V_pe_e=-sum(sum(taiyingle))/n;
%% V_xb
% u_ij * x_n-v_i
%distfcm(center, data);   %x_n-v_i
% u_ij * ||x_n-v_i|| = U.*distfcm(center, data)

%v_i-v_j
for i = 1:size(center,1)
    for j = size(center,2)
            distanceV_center(i,j) = sum( sum( (center(i,:) - center(:,j)).^2 ) );
    end
end
distanceV_center(find(distanceV_center==0))=[];

%V_xb= sum(sum( u.^2  )) * x_n_v_i / (n* min( abs(distanceV_center) ) );
V_xb=  sum(sum( u.*distfcm(center, data) ))/ (n* min( abs(distanceV_center) ) );