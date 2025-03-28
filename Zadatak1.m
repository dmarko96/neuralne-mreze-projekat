clc, clear, close all

load dataset3.mat

ulaz= pod(:,1:2);
izlaz=pod(:,3);

izlazOH= zeros(length(izlaz),3);
izlazOH(izlaz==1,1)=1;
izlazOH(izlaz==2,2)=1;
izlazOH(izlaz==3,3)=1;

K1=ulaz(izlaz==1,:);
K2=ulaz(izlaz==2,:);
K3=ulaz(izlaz==3,:);

figure, hold all
plot( K1(:,1), K1(:,2), 'o')
plot( K2(:,1), K2(:,2), 'o')
plot( K3(:,1), K3(:,2), 'o')
title("Klase");

ulaz=ulaz';
izlazOH=izlazOH';

rng(200);
n=length(izlaz);
ind=randperm(n);
indTrening=ind(1:0.8*n);
indTest=ind(0.8*n+1:n);

ulazTrening = ulaz(:,indTrening);
izlazTrening = izlazOH(:,indTrening);
 
ulazTest = ulaz(:,indTest);
izlazTest = izlazOH(:,indTest);
%%
%
%Underfit  Neuralna mreza
%
rng(200);
layersUnderfit= [3,3];
netUnderfit= patternnet(layersUnderfit);
netUnderfit.divideFcn='';

for i=1:length(layersUnderfit)
   netUnderfit.layers{i}.transferFcn='tansig';
end
netUnderfit.layers{i+1}.transferFcn='softmax';

netUnderfit.trainParam.epochs=2000;
netUnderfit.trainParam.goal= 1e-5;
netUnderfit.trainParam.min_grad=1e-6;

rng(200);
netUnderfit = train(netUnderfit, ulazTrening, izlazTrening);

predUnderfit=sim(netUnderfit,ulazTest);
figure
plotconfusion(izlazTest,predUnderfit)
title("Underfit Test Confusion Matrix");

Ntest=500;
ulazGO=[];
x1=linspace(-5,5,Ntest);
x2=linspace(-5,5,Ntest);

for x11=x1
    pom = [x11*ones(1,Ntest);x2];
    ulazGO= [ulazGO,pom];
end

predGO=sim(netUnderfit,ulazGO);
[vr,klasa]=max(predGO);

K1go=ulazGO(:,klasa==1);
K2go=ulazGO(:,klasa==2);
K3go=ulazGO(:,klasa==3);

figure, hold all
plot(K1go(1,:),K1go(2,:),'.')
plot(K2go(1,:),K2go(2,:),'.')
plot(K3go(1,:),K3go(2,:),'.')
plot(K1(:,1),K1(:,2), 'bo')
plot(K2(:,1),K2(:,2), 'ro')
plot(K3(:,1),K3(:,2), 'yo')
title("Underfit granica odlucivanja");
%%
%
%Overfit  Neuralna mreza
%
rng(200);
layersOverfit= [20,20,20,20];
netOverfit= patternnet(layersOverfit);
netOverfit.divideFcn='';

for i=1:length(layersOverfit)
   netOverfit.layers{i}.transferFcn='tansig';
end
netOverfit.layers{i+1}.transferFcn='softmax';

netOverfit.trainParam.epochs=2000;
netOverfit.trainParam.goal= 1e-5;
netOverfit.trainParam.min_grad=1e-6;

rng(200);
netOverfit = train(netOverfit, ulazTrening, izlazTrening);

predOverfit=sim(netOverfit,ulazTest);
figure
plotconfusion(izlazTest,predOverfit)
title("Overfit Test Confusion Matrix");

Ntest=500;
ulazGO=[];
x1=linspace(-5,5,Ntest);
x2=linspace(-5,5,Ntest);

for x11=x1
    pom = [x11*ones(1,Ntest);x2];
    ulazGO= [ulazGO,pom];
end

predGO=sim(netOverfit,ulazGO);
[vr,klasa]=max(predGO);

K1go=ulazGO(:,klasa==1);
K2go=ulazGO(:,klasa==2);
K3go=ulazGO(:,klasa==3);

figure, hold all
plot(K1go(1,:),K1go(2,:),'.')
plot(K2go(1,:),K2go(2,:),'.')
plot(K3go(1,:),K3go(2,:),'.')
plot(K1(:,1),K1(:,2), 'bo')
plot(K2(:,1),K2(:,2), 'ro')
plot(K3(:,1),K3(:,2), 'yo')
title("Overfit granica odlucivanja");
%%
%
%Optimal  Neuralna mreza
%
rng(200);
layersOptimal= [3,3,3];
netOptimal= patternnet(layersOptimal);
netOptimal.divideFcn='';

for i=1:length(layersOptimal)
   netOptimal.layers{i}.transferFcn='tansig';
end
netOptimal.layers{i+1}.transferFcn='softmax';

netOptimal.trainParam.epochs=2000;
netOptimal.trainParam.goal= 1e-5;
netOptimal.trainParam.min_grad=1e-6;

rng(200);
netOptimal = train(netOptimal, ulazTrening, izlazTrening);

predOptimal=sim(netOptimal,ulazTest);
figure
plotconfusion(izlazTest,predOptimal)
title("Optimal Test Confusion");

Ntest=500;
ulazGO=[];
x1=linspace(-5,5,Ntest);
x2=linspace(-5,5,Ntest);

for x11=x1
    pom = [x11*ones(1,Ntest);x2];
    ulazGO= [ulazGO,pom];
end

predGO=sim(netOptimal,ulazGO);
[vr,klasa]=max(predGO);

K1go=ulazGO(:,klasa==1);
K2go=ulazGO(:,klasa==2);
K3go=ulazGO(:,klasa==3);

figure, hold all
plot(K1go(1,:),K1go(2,:),'.')
plot(K2go(1,:),K2go(2,:),'.')
plot(K3go(1,:),K3go(2,:),'.')
plot(K1(:,1),K1(:,2), 'bo')
plot(K2(:,1),K2(:,2), 'ro')
plot(K3(:,1),K3(:,2), 'yo')
title("Optimal granica odlucivanja");