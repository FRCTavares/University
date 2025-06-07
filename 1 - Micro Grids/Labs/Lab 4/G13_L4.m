%% Lab - Buck DC microgrid | Grupo 13
clear all
clc

%% Dados do sistema
U=100;           % Tensão de entrada nominal [V]
ri=0.01;         % Resistência série de entrada [Ohm]
DeltaU=2*0.15;   % Variação de tensão ±15%
Vo=48;           % Tensão de saída [V]
DeltaVo=2*0.01;  % Variação de tensão ±1%
fpwm=20e3;       % Frequência de PWM [Hz]
Po=1500;         % Potência nominal [W]
Pomin=100;       % Potência mínima [W]

%% Cálculos de tensão e corrente
Umin = U * (1 - DeltaU / 2);
Umax = U * (1 + DeltaU / 2);
IoAVin = Pomin / Vo;

deltamin = Vo / Umax;
deltaN = Vo / U;
deltamax = Vo / Umin;

%% Filtro LC
Lf = 1.01 * Vo * (1 - deltamin) / (2 * IoAVin * fpwm);
Cf = 2 * IoAVin / (8 * fpwm * DeltaVo * Vo);

%% Cargas
Rmax = Vo^2 / Pomin;
Ios = Po / 2 / Vo; 
Iop = Po / 7 / Vo;

%% Semicondutores
Rdson = 0.1;
Rs = 1e5;
Cs = 1e-7;
Ron = 0.1;
Vd = 0.8;
