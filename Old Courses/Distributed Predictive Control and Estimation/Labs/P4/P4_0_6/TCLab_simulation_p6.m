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
Dr=50-y_ss;
d=0;
de=1;
Ad = [A,  B;
     zeros(1,n), 1];

Bd = [B; 0];

Cd = [C,  0];

Q_E = Ke * e_std * Ke';
QEd = blkdiag(Q_E, de);
L = dlqe(Ad,eye(n+1),Cd,QEd,e_std);
xdhat   = zeros(n+1,1);    % [Δx̂; d̂] starts at zero

xdhat(1:n) = pinv(C) * 5;      
du_prev = 0;               % last Δu

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
    %du = mpc_solvep6_7(dx, H, R, A, B, C,(u_ss+Du_ss),y_ss,Dr);
    Du(:,k)=Du_ss+du;
    Du(:,k)=0;
    du_prev=Du(:,k);
    u(:,k) = u_ss;
    
    
    % Aplica controlo
    x(:,k+1) = h1(x(:,k), u(:,k));

end
fprintf(' Done.\n');

%% Plot 1: measured vs. estimated output
figure; hold on; grid on
plot(t, y,           'b-',  'LineWidth',1.5, 'DisplayName','Measured y');
plot(t, yhat_abs,    'r--', 'LineWidth',1.5, 'DisplayName','Estimated ŷ');
xlabel('Time [s]');
ylabel('Temperature [°C]');
title('Measured vs. Estimated Output');
legend('Location','best');

%% Plot 2: estimated disturbance
figure; hold on; grid on
  plot(t, dhat_hist, 'k-',  'LineWidth',1.5, 'DisplayName','Estimated $\hat d$');

xlabel('Time [s]');
ylabel('Disturbance Estimate');
title('Kalman‐Filter Estimated Input Disturbance');
legend('Interpreter','latex','Location','best');

T=500;
N = T/Ts; % Number of samples to collect

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

yhat_inc  = nan(1,N);
yhat_abs   = nan(1,N);
dhat_hist =nan(1,N);
du=0;            
x(:,1) = Dx0 + x_ss;

k_chang = N/4; 
k_change = N/2; 
k_change2 = 3*N/4; 
deltaE_values = [0.1, 1, 10] * e_std;
dhat_all = nan(3,N);  % to store all 3 runs

for v = 1:3
    delta_e = deltaE_values(v);

    % Reset system state
    x(:,1) = Dx0 + x_ss;
    xdhat = zeros(n+1,1);
    xdhat(1:n) = pinv(C) * 5;
    du_prev = 0;
    Dr = 50 - y_ss;
    
    for k = 1:N
        t(k) = (k-1)*Ts;

        % Reference switching
        if k==k_chang, Dr = 40 - y_ss; end
        if k==k_change, Dr = 60 - y_ss; end
        if k==k_change2, Dr = 45 - y_ss; end

        y(:,k) = T1C(x(:,k));
        Dy(:,k) = y(:,k) - y_ss;
        Dx(:,k) = x(:,k) - x_ss;

        % Recompute L with current delta_e
        Q_E = Ke * e_std * Ke';
        QEd = blkdiag(Q_E, delta_e);
        L = dlqe(Ad, eye(n+1), Cd, QEd, e_std);

        if k == 1
            xdhat = pinv(Cd) * (Dy(:,k) + 5);
        else
            xdhat = Ad * xdhat + Bd * du_prev;
            xdhat = xdhat + L * (Dy(:,k) - Cd * xdhat);
        end

        dhat_all(v,k) = xdhat(end);  % store disturbance

        % Fixed open-loop u = u_ss
        u(:,k) = u_ss;
        du_prev = 0;

        x(:,k+1) = h1(x(:,k), u(:,k));
    end
end

fprintf(' Done.\n');

%% Plot comparison of disturbance estimates
figure; hold on; grid on
plot(t, dhat_all(1,:), 'b-',  'DisplayName', '$\delta_E = 0.1 \cdot e_{\mathrm{std}}$');
plot(t, dhat_all(2,:), 'k-',  'DisplayName', '$\delta_E = e_{\mathrm{std}}$');
plot(t, dhat_all(3,:), 'r-',  'DisplayName', '$\delta_E = 10 \cdot e_{\mathrm{std}}$');
xlabel('Time [s]');
ylabel('Estimated Disturbance');
title('Effect of $\delta_E$ on Kalman Disturbance Estimation','Interpreter','latex');
legend('Interpreter','latex','Location','best');
