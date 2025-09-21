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
e_std = sqrt(e_var); % input disturbance standard deviation
e_std = 0;

% Build the functions for applying the control and reading the temperature,
% mimicking the TCLab interface
x_ss = [eye(n) - A; C] \ [B * u_ss; y_ss];
c1 = ((eye(n) - A) * x_ss - B * u_ss);
c2 = (y_ss - C * x_ss);
h1 = @(x,u) A * x + B * u + Ke * e_std * randn + c1; % apply control
T1C = @(x) C * x + e_std * randn + c2; % read temperature

% Simulation parameters
T = 4000; % Experiment duration [s]
N = T / Ts; % Number of samples to collect

% Initial conditions (start at ambient temperature, i.e. equilibrium for u = 0)
Dx0Dy0 = [eye(n) - A, zeros(n, 1); C, -1] \ [-B * u_ss; 0];
Dx0 = Dx0Dy0(1:n);

% Closed-loop control profile -> Added for P4
u = zeros(1, N); % Control
H = 50; % Horizon 
R = 0.01;

% Initialize signals
t = nan(1, N);
x = nan(n, N);
y = nan(1, N);
Dy = nan(1, N);
Du = nan(1, N);
Dx = nan(n, N + 1);
x(:, 1) = Dx0 + x_ss;

% Simulate incremental model
fprintf('Running simulation...')
for k = 1:N
    % Computes analog time
    t(k) = (k - 1) * Ts;

    % Reads the sensor temperature
    y(:, k) = T1C(x(:, k));

    % Compute incremental variables
    Dy(:, k) = y(:, k) - y_ss;
    Dx(:, k) = x(:, k) - x_ss;
    % Du(:, k) = u(:, k) - u_ss;
    Du(:, k) = mpc_solve_2(Dx(:, k), H, R, A, B, C); % Added for P4

    % Compute control -> Added for P4
    u(:, k) = Du(:, k) + u_ss;

    % Applies the control variable to the plant
    x(:, k + 1) = h1(x(:, k), u(:, k));
end
fprintf(' Done.\n');

%% Plots
% Plot absolute variables
figure('Units', 'normalized', 'Position', [0.2 0.5 0.3 0.4])
subplot(2, 1, 1), hold on, grid on   
title('Absolute input/output')
plot(t, y, '.', 'MarkerSize', 5)
yl = yline(y_ss, 'k--');
xlabel('Time [s]')
ylabel('y [°C]')
legend(yl, '$\bar{y}$', 'Interpreter', 'latex', 'Location', 'best')
subplot(2, 1, 2), hold on, grid on   
stairs(t, u, 'LineWidth', 2)
yl = yline(u_ss, 'k--');
yline(0, 'r--')
yline(100, 'r--')
xlabel('Time [s]')
ylabel('u [%]')
legend(yl, '$\bar{u}$', 'Interpreter', 'latex', 'Location', 'best');

% Plot incremental variables
figure('Units', 'normalized', 'Position', [0.5 0.5 0.3 0.4])
subplot(2, 1, 1), hold on, grid on   
title('Incremental input/output')
plot(t, Dy, '.', 'MarkerSize', 5)
xlabel('Time [s]')
ylabel('\Delta{y} [°C]')
subplot(2, 1, 2), hold on, grid on   
stairs(t, Du, 'LineWidth', 2)
yline(-u_ss, 'r--')
yline(100 - u_ss, 'r--')
xlabel('Time [s]')
ylabel('\Delta{u} [%]')