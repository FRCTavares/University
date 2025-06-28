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
clc
clear
close all

% Load model
load('singleheater_model.mat', 'A', 'B', 'C', 'Ke', 'e_var', 'y_ss', 'u_ss', 'Ts');
n = size(A, 1);
e_std = sqrt(e_var); % Input disturbance standard deviation

% Build the functions for applying the control and reading the temperature,
% mimicking the TCLab interface
x_ss = [eye(n) - A; C] \ [B * u_ss; y_ss];
c1 = (eye(n) - A) * x_ss - B * u_ss;
c2 = y_ss - C * x_ss;
h1 = @(x, u) A * x + B * u + Ke * e_std * randn + c1; % Apply control
T1C = @(x) C * x + e_std * randn + c2; % Read temperature

% Simulation parameters
T = 4000; % Experiment duration [s]
N = T / Ts; % Number of samples to collect

% Initial conditions (start at ambient temperature, i.e. equilibrium for u = 0)
Dx0Dy0 = [eye(n) - A, zeros(n, 1); C, -1] \ [-B * u_ss; 0];
Dx0 = Dx0Dy0(1:n);

% Initialize signals
t = nan(1, N);
x = nan(n, N);
y = nan(1, N);
Dy = nan(1, N);
Du = nan(1, N);
Dx = nan(n, N + 1);
x(:, 1) = Dx0 + x_ss;

% MPC parameters
H = 50;
R = 0.05;
beta = 50;

%% Kalman filter design
% Define the covariance matrices
QE = Ke * e_var * Ke';
RE = e_var;

% Define the disturbance tunning parameter
dE = 2 * e_std;

% Compute augmented matrices
Ad = [A B; zeros(1, n) 1];
Bd = [B; 0];
Cd = [C 0];
QEd = [[QE zeros(n, 1)]; zeros(1, n) dE];

% Kalman gain
L = dlqe(Ad, eye(n + 1), Cd, QEd, RE);

% Reference
r = zeros(N, 1);
r(1:150)    = 50;
r(151: 300) = 40;
r(301: 450) = 60;
r(451: 800) = 45;
Dr = r - y_ss;

% Initialize signals
u = zeros(1, N);
xd_est = nan(n + 1, N);
xd_est(:, 1) = [Dx0 + Ke * e_std * randn; 0]; % Kalman filter initialization

%% Simulate incremental model
fprintf('Running simulation...')
for k = 1:N
    % Computes analog time
    t(k) = (k - 1) * Ts;

    % Reads the sensor temperature
    y(:, k) = T1C(x(:, k));

    % Compute incremental variables
    Dy(:, k) = y(:, k) - y_ss;
    Dx(:, k) = x(:, k) - x_ss;
    Du(:, k) = u(:, k) - u_ss;

    % Kalman filter correction step
    xd_est(:, k) = xd_est(:, k) + L * (Dy(:, k) - Cd * xd_est(:, k));

    % Compute the steady-state disturbance and associated state and control
    d_ss = xd_est(n + 1, k);
    Du_ss = (C / (eye(n) - A) * B) \ Dr(k) - d_ss;
    Dx_ss = (eye(n) - A) \ B * (Du_ss + d_ss);

    % Computes the control variable to apply
    dx = xd_est(1:n, k) - Dx_ss;
    du = mpc_solve(dx, u_ss, Du_ss, y_ss, Dr(k), H, R, A, B, C, beta);
    Du(:, k) = du + Du_ss;
    u(:, k) = Du(:, k) + u_ss;

    % Kalman filter prediction step
    xd_est(:, k + 1) = Ad * xd_est(:, k) + Bd * Du(:, k);

    % Applies the control variable to the plant
    x(:, k + 1) = h1(x(:, k), u(:, k));
end
fprintf(' Done.\n');

%% Plots
% Plot absolute variables
figure('Units','normalized','Position',[0.2 0.5 0.3 0.4])
subplot(2,1,1), hold on, grid on   
title('Absolute input/output')
plot(t,y,'.','MarkerSize',5)
yl=yline(y_ss,'k--');
xlabel('Time [s]')
ylabel('y [°C]')
legend(yl,'$\bar{y}$','Interpreter','latex','Location','best')
subplot(2,1,2), hold on, grid on   
stairs(t,u,'LineWidth',2)
yl=yline(u_ss,'k--');
yline(0,'r--')
yline(100,'r--')
xlabel('Time [s]')
ylabel('u [%]')
legend(yl,'$\bar{u}$','Interpreter','latex','Location','best');

% Plot incremental variables
figure('Units','normalized','Position',[0.5 0.5 0.3 0.4])
subplot(2,1,1), hold on, grid on   
title('Incremental input/output')
plot(t,Dy,'.','MarkerSize',5)
xlabel('Time [s]')
ylabel('\Delta{y} [°C]')
subplot(2,1,2), hold on, grid on   
stairs(t,Du,'LineWidth',2)
yline(-u_ss,'r--')
yline(100-u_ss,'r--')
xlabel('Time [s]')
ylabel('\Delta{u} [%]')

%--------------------------------------------------------------------------
% End of File