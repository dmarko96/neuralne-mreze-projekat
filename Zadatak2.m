clc, clear, close all

pod= readtable('Genres.csv');

ulaz = pod(:,1:11);
ulaz=table2array(ulaz);
izlaz = pod.genre;

genres={'Rap','Pop','RnB'};
genrevalue=[1,2,3];
[tf, idx] = ismember(izlaz(:), genres);
izlaz = genrevalue( idx(tf) ) ;

izlazOH= zeros(3,length(izlaz));
izlazOH(1,izlaz==1)=1;
izlazOH(2,izlaz==2)=1;
izlazOH(3,izlaz==3)=1;

ulaz=ulaz';

figure,histogram(izlaz)

K1= ulaz(:,izlaz==1);
I1= izlazOH(:,izlaz==1);
rng(200);
n=length(K1);
ind=randperm(n);
K1= K1(:,ind);
I1=I1(:,ind);


K2= ulaz(:,izlaz==2);
I2=izlazOH(:,izlaz==2);
rng(200);
n=length(K2);
ind=randperm(n);
K2= K2(:,ind);
I2=I2(:,ind);

K3= ulaz(:,izlaz==3);
I3=izlazOH(:,izlaz==3);
rng(200);
n=length(K3);
ind=randperm(n);
K3= K3(:,ind);
I3=I3(:,ind);

rng(200);

divTren=0.6;
divTest=0.8;

N1=length(K1);
K1trening = K1(:, 1 : floor(divTren*N1));
K1test = K1(:, floor(divTren*N1)+1 : floor(divTest*N1));
K1val  = K1(:, floor(divTest*N1)+1 : N1);
I1trening = I1(:, 1 : floor(divTren*N1));
I1test = I1(:, floor(divTren*N1)+1 : floor(divTest*N1));
I1val  = I1(:, floor(divTest*N1)+1 : N1);

N2=length(K2);
K2trening = K2(:, 1 : floor(divTren*N2));
K2test = K2(:, floor(divTren*N2)+1 : floor(divTest*N2));
K2val  = K2(:, floor(divTest*N2)+1 : N2);
I2trening = I2(:, 1 : floor(divTren*N2));
I2test = I2(:, floor(divTren*N2)+1 : floor(divTest*N2));
I2val  = I2(:, floor(divTest*N2)+1 : N2);

N3=length(K3);
K3trening = K3(:, 1 : floor(divTren*N3));
K3test = K3(:, floor(divTren*N3)+1 : floor(divTest*N3));
K3val  = K3(:, floor(divTest*N3)+1 : N3);
I3trening = I3(:, 1 : floor(divTren*N3));
I3test = I3(:, floor(divTren*N3)+1 : floor(divTest*N3));
I3val  = I3(:, floor(divTest*N3)+1 : N3);

ulazTrening = [K1trening,K2trening,K3trening];
izlazTrening= [I1trening,I2trening,I3trening];

ulazTest= [K1test,K2test,K3test];
izlazTest = [I1test,I2test,I3test];

rng(200);
n=length(ulazTrening);
ind=randperm(n);
ulazTrening= ulazTrening(:,ind);
izlazTrening=izlazTrening(:,ind);


ulazVal=[K1val,K2val,K3val];
izlazVal=[I1val,I2val,I3val];

ulazSve=[ulazTrening,ulazVal];
izlazSve=[izlazTrening,izlazVal];

%%
arhitektura= {[5],[8],[5,5],[8,8],[10,10],[8,6,4],[5,5,5],[10,10,10],[3,3,3]};

Amax=0;

for lr=[0.5,0.25,0.1,0.01,0.001]
for arh=1:length(arhitektura)
for w=[1,2,4,8]
    rng(200);
    net=patternnet(arhitektura{arh}); %crossvalidacija
   
    net.divideFcn= 'divideind';
    net.divideParam.trainInd= 1:length(ulazTrening);
    net.divideParam.valInd= length(ulazTrening)+1:length(ulazSve);
    net.divideParam.testInd=[];

    net.trainFcn='trainrp';
    net.trainParam.lr=lr;
    for i=1:length(arhitektura{arh})
        net.layers{i}.transferFcn='tansig';
    end
    net.layers{i+1}.transferFcn='softmax';

    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-2;
    net.trainParam.min_grad=1e-3;

    net.trainParam.showWindow=false;
    
    weight= izlazSve;
    weight(2,izlazSve(2,:)==1)=w;

    %train
    [net,info]=train(net,ulazSve,izlazSve,[],[],weight);
    %

    pred=round(sim(net,ulazVal));
    [e,cm]=confusion(izlazVal,pred);
    A=sum(trace(cm))/sum(sum(cm))*100;

    A

    if A>Amax
        %postavi max param
        best_ep=info.best_epoch;
    
        best_arh=arhitektura{arh};
        best_lr=lr;
        best_w=w;

        Amax=A;
    end
end
end
end
%%
rng(200);
best_arh
best_lr
best_w
net=patternnet(best_arh);
net.divideFcn= '';

net.trainFcn='trainrp';
net.trainParam.lr=best_lr;
weight= izlazSve;
weight(2,izlazSve(2,:)==1)=best_w;

for i=1:length(best_arh)
    net.layers{i}.transferFcn='tansig';
end
net.layers{i+1}.transferFcn='softmax';

net.trainParam.epochs = best_ep;
net.trainParam.goal = 1e-2;
net.trainParam.min_grad=1e-3;

net=train(net,ulazSve,izlazSve,[],[],weight);

pred=sim(net,ulazTest);
figure,plotconfusion(izlazTest,pred);

