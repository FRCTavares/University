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
deltaN=Vo/U;         
deltamax=Vo/Umin;    % deltamax=0.565

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
Rs=1e5;
Cs=1e-7;
Ron=0.1;
Vd=0.8;

pr=0.61;

% aumentar o Td permite baixar o pr

% Po/Vo a 10%: pr instavel a 0.58 e "semi" instavel a 0.59
% Po/Vo a 30%: pr instavel a 0.59 e "semi" instavel a 0.6
% Po/Vo a 50%: pr instavel a 0.6 e "semi" instavel a 0.61
% Po/Vo a 70%: pr instavel a 0.61 e "semi" instavel a 0.62
% Po/Vo a 90%: pr instavel a 0.61 e "semi" instavel a 0.62

% Td = 8/(2*fpwm)
% pr instavel a 0.43 e "semi" instavel a 0.44