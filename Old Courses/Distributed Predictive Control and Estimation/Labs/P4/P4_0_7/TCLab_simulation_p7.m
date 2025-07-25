% Simulation of the TCLab linear model previously identified
%
% Loads the model identified in the TCLab_identification script, creates
% the h1 and T1C functions that mimick the TCLab interface, and performs a
% simulation starting at ambient temperature.
% You will be developing and testing your MPC controller and Kalman filter
% in this simulation environment. 
%
% Afonso Botelho and J. Miranda Lemos, IST, May 2023
%__________________________________________________________________________

% Initialization
clear all
close all
clc

% Load model
load('singleheater_model.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);
e_std = sqrt(e_var); % input disturbance standard deviation
% Build the functions for applying the control and reading the temperature,
% mimicking the TCLab interface
x_ss = [eye(n)-A; C]\[B*u_ss; y_ss];
c1 = ((eye(n)-A)*x_ss - B*u_ss)*1.1;
c2 = (y_ss - C*x_ss);
h1 = @(x,u) A*x + B*u + Ke*e_std*randn + c1; % apply control
T1C = @(x) C*x + e_std*randn + c2; % read temperature

% Simulation parameters
T = 4000; % Experiment duration [s]
N = T/Ts; % Number of samples to collect

% Open-loop control profile
%u = zeros(1,N);
%u(:,1:200)   = u_ss;
%u(:,201:400) = u_ss+5;
%u(:,401:600) = u_ss-5;
%u(:,601:800) = u_ss;

% Parâmetros MPC
H = 20;    % horizonte de predição
R = 0.01;  % penalização do controlo

u = nan(1,N); % vetor de controlo (agora vai ser calculado a cada instante)
% Initial conditions (start at ambient temperature, i.e. equilibrium for u = 0)
Dx0Dy0 = [eye(n)-A, zeros(n,1); C, -1]\[-B*u_ss; 0];
Dx0 = Dx0Dy0(1:n);

% Initialize signals
t = nan(1,N);
x = nan(n,N);
y = nan(1,N);
Dy = nan(1,N);
Du = nan(1,N);
Dx = nan(n,N+1);
eta_hist = nan(1, N);  % now a row vector

yhat_inc  = nan(1,N);
yhat_abs   = nan(1,N);
dhat_hist =nan(1,N);
du=0;            
x(:,1) = Dx0 + x_ss;

k_chang = N/4; 
k_change = N/2; 
k_change2 = 3*N/4; 
% Simulate incremental model
fprintf('Running simulation...')
r = nan(1,N);      % absolute reference temperature
Dr=50-y_ss;
d=0;
de=1;
Ad = [A,  B;
     zeros(1,n), 1];

Bd = [B; 0];

Cd = [C,  0];

Q_E = Ke * e_var * Ke';
QEd = blkdiag(Q_E, de);
L = dlqe(Ad,eye(n+1),Cd,QEd,e_var);
xdhat   = zeros(n+1,1);    % [Δx̂; d̂] starts at zero

xdhat(1:n) = pinv(C) * 5;      
du_prev = 0;               % last Δu

y_max = 55;   % [°C] upper bound
for k = 1:N
   
    % Computes analog time
    t(k) = (k-1)*Ts;
    
    if k==k_chang
        Dr=40-y_ss;
    end
    if k==k_change
        Dr=60-y_ss;
    end
    if k==k_change2
        Dr=45-y_ss;
    end
    
    r(k) = y_ss + Dr;

    % Reads the sensor temperature
    y(:,k) = T1C(x(:,k));

    % Compute incremental variables
    Dy(:,k) = y(:,k) - y_ss;
    dy=Dy(:,k)-Dr;
    Dx(:,k) = x(:,k) - x_ss;
    
    %Kalman filter steps
    if k == 1
        xdhat = pinv(Cd) * (Dy(:,k) + 5);  % inject extra +5°C error
        Dx_est = xdhat(1:n); 
        d   = xdhat(n+1); 
    else
        xdhat = Ad*xdhat + Bd*du_prev;
        xdhat  = xdhat + L*(Dy(:,k) - Cd*xdhat);
    
        Dx_est = xdhat(1:n);  
        d   = xdhat(n+1);  
    end
    %for plot
    yhat_inc(k)   = Cd * xdhat;      % incremental output prediction
    yhat_abs(k)   = y_ss + yhat_inc(k);  % absolute output prediction
    dhat_hist(k)  = xdhat(end);      % disturbance estimate
    
    % equaçoes fixes
    Du_ss=pinv(C*((eye(3)-A)\B))*Dr;
    Dx_ss=((eye(3)-A)\B)*Du_ss;
    Du_ss=Du_ss-d; %real Du_ss
    dx=Dx_est-Dx_ss;

    % Calcula controlo com MPC
    [du, eta_now] = mpc_solvep6_7(dx, H, R, A, B, C,(u_ss+Du_ss),y_ss,Dr);
    eta_hist(k) = sum(eta_now);
    Du(:,k)=Du_ss+du;
    du_prev=Du(:,k);
    u(:,k) = u_ss + Du(:,k);
    
    
    % Aplica controlo
    x(:,k+1) = h1(x(:,k), u(:,k));

end
fprintf(' Done.\n');

err_rel = abs(y - yhat_abs) ./ y * 100;

%% Plots

%% Plot 1: measured vs. estimated output
figure; hold on; grid on
plot(t, y,         'b-',  'LineWidth',1.5, 'DisplayName','Measured y');
plot(t, yhat_abs,  'r--', 'LineWidth',1.5, 'DisplayName','Estimated ŷ');
plot(t, r,         'k:',  'LineWidth',1.5, 'DisplayName','Reference y^*');
yline(y_max, 'm--', 'y_{max}', 'LineWidth',1.5, ...
      'DisplayName','y_{max}');
xlabel('Time [s]');
ylabel('Temperature [°C]');
title('Measured vs. Estimated Output with Reference');
legend('Location','best');

%% Figure X: Output‐error and disturbance estimate
figure('Units','normalized','Position',[0.1 0.1 0.35 0.35])

% 1) Relative error
subplot(2,1,1), hold on, grid on
plot(t, err_rel, '.', 'MarkerSize',5, 'DisplayName','|y - ŷ|/y [%]')
xlabel('Time [s]')
ylabel('|y - ŷ| / y [%]')
title('Relative Output Error')
legend('Location','best')

% 2) Disturbance estimate + bound
subplot(2,1,2), hold on, grid on
plot(t, dhat_hist, '.', 'MarkerSize',5, 'DisplayName','Estimated \deltâ')
xlabel('Time [s]')
ylabel('\deltâ [%]')
title('Disturbance Estimate')
legend('Location','best')

figure; hold on; grid on
scatter(t, eta_hist, 15, 'filled')
grid on;
xlabel('Time [s]')
ylabel('\eta_i (Constraint Violation)')
title('Constraint Relaxation Over Time')
ylim([0 max(eta_hist)*1.1]) % Add a bit of headroom
