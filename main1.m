tic;
close all;
clear all
clc;

%% Dataset

% load('bupaliver.txt');
% A= bupaliver(:,1:end-1,:);
% d= bupaliver(:,end);
% d(find(d==2))=-1;

%   load('australian.txt')
%   A= australian(:,1:end-1,:);
%   d= australian(:,end);
%   d(find(d==0))=-1;

% load('diabetes_data.mat')
% load('diabetes_label.mat')
% A= data1;
% d= label;
% d(find(d==0))=-1;

%pima_indians_diabetes

%   load('heart.dat');
%   A=heart(:,1:end-1,:);
%   d= heart(:,end);
%   d(find(d~=1))=-1;
%   

%  load('pima-indians-diabetes.data')
%  A= pima_indians_diabetes(:,1:end-1,:);
%  d = pima_indians_diabetes(:,end);
%  d(find(d==0))=-1;

%   load('ionosphere_label.mat');
%   load('ionosphere_data.mat');
%   A= data;
%   d=y2;
%   d(find(d==2))=-1;

 

%   load('wdbc_data.mat')
%   load('wdbc_label.mat');
%   A= wdbc_data;
%   d=wdbc_label;
%   d(find(d==2))=-1;



% load('crossplane.mat')
% load('crossplane1.mat')
% A =fvtrain;
% d=ctrain;
% d(find(d==2))=-1;
%
%%
   %load ('sonar.txt') 
%   A=data;
%    d=y1;
%    d(find(d~=1))=-1;
%%
% load('echocardiogram_data.mat');
% load('echocardiogram_label.mat');
% A=x;
% d=y;
% d(find(d~=1))=-1;

%%
  load('iris.dat')
  data=iris(1:100,1:4);
  label=iris(1:100,5);
   A= data;
   label(find(label==2))=-1;
%   label(find(label==3))=-1;
    d=label;
  %%  
% 
%  load('Haberman_dataset.mat');
%  A= Haberman_data(:,1:end-1);
%  d= Haberman_data(:,end);
%  d(find(d==2))=-1;

%  for i=1:size(A,2)
%      A(:,i)= zscore(A(:,i));
%  end
%  
%  load('soybean_data.mat')
%  A= soybean_dataset(:,1:end-1);
%  d= soybean_dataset(:,end);
%  d(find(d~=1))=-1;

% load('votes.mat');
% A= votes(:,2:end);
% d=votes(:,1);
%  d(find(d~=1))=-1;


% load('german.txt')
% A=german(:,end-1);
% d= german(:,end);
% d(find(d~=1))=-1;

% load('ecoli_data.mat')
% A=ecoli_data(:,end-1);
%  d= ecoli_data(:,end);
%  d(find(d~=1))=-1;


% load('Teaching_eval uation_data.mat')
% A= Teaching_Evaluation(:,1:end-1);
% d = Teaching_Evaluation(:,end);
% d(find(d~=3))=-1;
% d(find(d==3))=1;
% d(find(d~=1))=-1; 
%  for i=1:size(A,2)
%   A(:,i)=zscore(A(:,i));    
%  end  
%% Normalizing data
 A= svdatanorm(A,'s8pline');
%% Parameter setting
kernel=1; % kernel =1  is linear kernel
tau=0.1;
c1= 8;
p =0.2; % kerenl parameter It works when  kernel =2 .
%%
k=10;
[m,n]=size(A);
m=floor(m/k)*k;
rng(1);   
r=[randperm(m)]';
d=d(r,:);
A=A(r,:);
%% if k= folds
indx = [0:k];
indx = floor(m*indx/k);   
time=0;
% split trainining set from test set
for i = 1:k
    disp(i);
    Ctest = []; dtest = []; Ctrain = []; dtrain = [];
    Ctest = A((indx(i)+1:indx(i+1)),:);
    dtest = d(indx(i)+1:indx(i+1));
    Ctrain = A(1:indx(i),:);
    Ctrain = [Ctrain;A(indx(i+1)+1:m,:)];
    dtrain = [d(1:indx(i));d(indx(i+1)+1:m,:)];    
   %%
    [pdtest, run_time,spars(i)] = l1norm_pin_svm(Ctrain, dtrain, Ctest, dtest, kernel, tau, c1,p);
     acc(i)= length(find(pdtest==dtest))*100/length(dtest);
end
 time= time+toc;
fprintf('\nAccuracy=%3.2f+%3.2f',mean(acc),std(acc));
  fprintf('\n Sparsity=     %3.2f+%3.2f',mean(spars),std(spars));
 fprintf('\n time=    %3.2f \n',time);
 