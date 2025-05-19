%% Lab 1 - Buck DC microgrid | Grupo 13

%DC_DC_buck_Microgrid
clear all
clc

% Data
U=100;
ri=0.01;
DeltaU=2*0.15;
Vo=48;
DeltaVo=2*0.01;
fpwm=10000;
Po=1500;
Pomin=100;

% Calculations
Umin=U*(1-DeltaU/2); % Umin = 85 V
Umax=U*(1+DeltaU/2); % Umax=115 V
IoAVin=Pomin/Vo;     % IoAVin=2.08 A
deltamin=Vo/U;       % deltamin=0.417
deltaN=Vo/U;         % deltamax=0.565
deltamax=Vo/Umin;

% LC filter
Lf=1.01*Vo*(1-deltamin)/(2*IoAVin*fpwm); % Lf=6e-4 H
Cf=2*IoAVin/(8*fpwm*DeltaVo*Vo);         % Cf=5.42e-5 F
% C for point-of-load

% Load
Rmax=Vo^2/Pomin; % Rmax=23.04 Ω
Ios=Po/2/Vo;     % Ios=15.625 A
Iop=Po/7/Vo;     % Iop=4.464 A

% Semiconductors
Rdson=0.1;
Rs=1e4;
Cs=1e-7;
Ron=0.1;
Vd=0.8;


