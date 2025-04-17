%DC_DC
clear all

% DATA
U=100;
ri=0.015;
DeltaU=2*0.15
Vo=48;
DeltaVo=2*0.01;
fpwm=10000;
Po=1500;
Pomin=100;

%Calc
Umin=U*(1-DeltaU/2)
Umax=U*(1+DeltaU/2)
IoAVmin=Pomin/Vo;
deltamin=Vo/Umax
deltaN=Vo/U
deltamax=Vo/Umin

%LC filter
Lf=1.01*Vo*(1-deltamin)/(2*IoAVmin*fpwm)
Cf=2*IoAVmin/ (8*fpwm*DeltaVo*Vo)
%C for point-of-load
%Load
Rmax=Vo^2/Pomin
Ios=Po/2/Vo

%Semiconductors
Rdson=0.1;
Rs=1e4;
Cs=1e-7;
Ron=0.1;
Vd=0.8;

