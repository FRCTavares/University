%% Lab - Buck DC microgrid | Grupo 13
clear all
clc

%% Dados do sistema
U = 100;           % Tensão de entrada nominal [V]
ri = 0.01;         % Resistência série de entrada [Ohm]
DeltaU = 2 * 0.15; % Variação de tensão ±15%
Vo = 48;           % Tensão de saída [V]
DeltaVo = 2 * 0.01; % Variação de tensão ±1%
fpwm = 10e3;       % Frequência de PWM [Hz]
Po = 1500;         % Potência nominal [W]
Pomin = 100;       % Potência mínima [W]

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
Ios = 0; %Po*0.10/Vo; %Po / 2 / Vo;
Iop = Po / 7 / Vo;

%% Semicondutores
Rdson = 0.1;
Rs = 1e5;
Cs = 1e-7;
Ron = 0.1;
Vd = 0.8;

%% Parâmetro de razão de potência para carga de potência constante
% pr = Po_cargaPot / Po_total
% Este parâmetro é essencial para estudar a estabilidade em função da percentagem de carga de potência.

%% Exercício 3 i)

%pr = 0.454; % -> Limite de Semi-Estabilidade
%pr = 0.598; % -> Limite de Estabilidade

% Agora para Ios =  0
% Ios = 0;
%pr = 0.48; % -> Limite de Semi-Estabilidade
%pr = 0.598; % -> Limite de Estabilidade

% Agora para Ios = Po*0.10 / Vo
%Ios = Po*0.10/Vo;
%pr = 0.48; % -> Limite de Semi-Estabilidade
%pr = 0.598; % -> Limite de Estabilidade

%% Exercício 4. f)
%Para K = Cf*kv = 2
%pr = 0.46; % -> Limite de Semi-Estabilidade
%pr = 0.48; % -> Limite de Estabilidade
%Para K = Cf*kv = 5
%pr = 0.508; % -> Limite de Estabilidade e Semi-Estabilidade


% Agora para Ios =  0
% Ios = 0;
%Para K = Cf*kv = 2
%pr = 0.44; % -> Limite de Semi-Estabilidade
%pr = 0.49; % -> Limite de Estabilidade
%Para K = Cf*kv = 5
pr = 0.48; % -> Limite de Estabilidade e Semi-Estabilidade

% Agora para Ios = Po*0.10 / Vo
%Ios = Po*0.10/Vo;
%Para K = Cf*kv = 2
%pr = 0.43; % -> Limite de Semi-Estabilidade
%pr = 0.46 ; % -> Limite de Estabilidade
%Para K = Cf*kv = 5
%pr = 0.50; % -> Limite de Estabilidade e Semi-Estabilidade

%% Ganhos dos controladores PI

% Constantes úteis
ucmax = 1;
KM = U / ucmax;
%Td = 1 / (2 * fpwm);         % Td = T/2
Td = 8 / (2 * fpwm);
Req = Vo^2 / Po;             % Resistência equivalente de carga
qsi_i = 0.707;               % Amortecimento da corrente
qsi_v = 0.85;                % Amortecimento da tensão
alfav = 1;                   % Ganho de medição de tensão

%% Controlador de Corrente (Kp, Ki)
Tzi = Lf / Req;
ki = 1;                      % Ganho integrador base (arbitrário para dimensionamento)
Tp = 4 * qsi_i^2 * ki * KM * Td / Req;
Kp = Tzi / Tp;
Ki = 1 / Tp;

%% Controlador de Tensão (Kpv, Kiv)
Tzv = Cf * Req;
Tpv = 8 * qsi_v^2 * Req * alfav * Td;
Kpv = Tzv / Tpv;
Kiv = 1 / Tpv;

%% Mostrar Resultados
fprintf('Controlador de corrente:\n');
fprintf('Kp  = %.4f\n', Kp);
fprintf('Ki  = %.2f\n\n', Ki);

fprintf('Controlador de tensão:\n');
fprintf('Kpv = %.4f\n', Kpv);
fprintf('Kiv = %.1f\n', Kiv);
