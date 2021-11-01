clc
clear all
close all
cd('Train Data');
imagefiles = dir('*.png');
k=1;
n = length(imagefiles); % Number of files found

%% Found out maximum values of the whole train_images

h_train=zeros(60,2);
for i=1:n
currentfilename = imagefiles(i).name;
im_train = imread(currentfilename);
imgs_train=im2bw(im_train,0.01);
img_train= double(imgs_train);
[p,q]=size(img_train);
h_train(i,:)=[p,q];
end
sum_train=sum(h_train(:,:));
r1=sum_train/n;

%% Found out maximum values of the whole test_images

cd('..');
cd('Test Data');
imagefiles = dir('*.png');
l = length(imagefiles); % Number of files found
h_test=zeros(30,2);
for i=1:l
currentfilename = imagefiles(i).name;
im_test = imread(currentfilename);
imgs_test=im2bw(im_test,0.01);
img_test= double(imgs_test);
[p,q]=size(img_test);
h_test(i,:)=[p,q];
end
sum_test=sum(h_test(:,:));
r2=sum_test/30;
%%
r=ceil(r1+r2);

%% Changing sizes of train_images 

s=zeros(998,622);
S=zeros(60,998*622);
cd('..');
cd('Train Data');
imagefiles = dir('*.png');
for i=1:n
currentfilename = imagefiles(i).name;
im_train = imread(currentfilename);
imgs=im2bw(im_train,0.01);
s(:,:)=imresize(imgs,[998,622]);
imgs_train=im2bw(s,0.01);
img_train= double(imgs_train);
ss=reshape(img_train,1,620756);
ss(ss==0)= -1;
S(k,:)=ss(:,:);
noise_train_1=imnoise(s,'salt & pepper',0.1); % 10% nosie
noise_train_2=imnoise(s,'salt & pepper',0.2); % 20% noise
ss_noise_1=reshape(noise_train_1,1,620756);
ss_noise_2=reshape(noise_train_2,1,620756);
ss_noise_1(ss_noise_1==0)=-1;
ss_noise_2(ss_noise_2==0)=-1;
S_noise_10(k,:)=ss_noise_1(:,:);
S_noise_20(k,:)=ss_noise_2(:,:);
k=k+1;
end
S=S';
S_noise_10=S_noise_10';
S_noise_20=S_noise_20';
bin = [0,0,1;0,1,0;0,1,1;1,0,0;1,0,1;0,0,0];
nu = 0;
for i = 1:6
    b = bin(i,:);
    for j= 1:10
        nu = nu+1;
        T(nu,:)= b;
    end
end
T(T==0)=-1;
T=T';

%% Changing sizes of test_images 

cd('..');
cd('Test Data');
imagefiles = dir('*.png');

st=zeros(998,622);
sst=zeros(30,998*622);
k=1;
for i=1:l
currentfilename = imagefiles(i).name;
im_test = imread(currentfilename);
imgst=im2bw(im_test,0.01);
st(:,:)=imresize(imgst,[998,622]);
imgs_test=im2bw(st,0.01);
img_test= double(imgs_test);

sst=reshape(img_test,1,620756);
sst(sst==0)= -1;
S_test(k,:)=sst(:,:);
k=k+1;
end
S_test=S_test';
bin = [1,0,0;1,0,1;0,1,1;0,0,1;0,0,0;0,1,0];
nu = 0;
for i = 1:6
    b = bin(i,:);
    for j= 1:5
        nu = nu+1;
        T_test(nu,:)= b;
    end
end
T_test(T_test==0)=-1;
T_test=T_test';

%% Training the Network

net = newp(S,T,'hardlims');
net.trainParam.epochs = 35;
net = train(net,S,T);
Y = sim(net,S);

%% Applying test data and getting output

Out=sim(net,S_test);

disp(' ')


%% Applying 10% noisy data and getting output

Out_noisy_10=sim(net,S_noise_10);

disp(' ')


%% Applying 20% noisy data and getting output

Out_noisy_20=sim(net,S_noise_20);

disp(' ')


%% Calculating the acuracy of training data

for i=1:60
    if Y(:,i)==T(:,i)
        A(1,i)=1;
    else
        A(1,i)=0;
    end
end
Acuuracy_traint=mean(A)*100

%% Calculating the acuracy of training data

for i=1:30
    if Out(:,i)==T_test(:,i)
        Ac(1,i)=1;
    else
        Ac(1,i)=0;
    end
end
Acuuracy_test=mean(Ac)*100

%% Calculating the acuracy of 10% noisy data

for i=1:60
    if Out_noisy_10(:,i)==T(:,i)
        Ac_noisy_10(1,i)=1;
    else
        Ac_noisy_10(1,i)=0;
    end
end
Acuuracy_noisy_10=mean(Ac_noisy_10)*100

%% Calculating the acuracy of 20% noisy data

for i=1:60
    if Out_noisy_20(:,i)==T(:,i)
        Ac_noisy_20(1,i)=1;
    else
        Ac_noisy_20(1,i)=0;
    end
end
Acuuracy_noisy_20=mean(Ac_noisy_20)*100
