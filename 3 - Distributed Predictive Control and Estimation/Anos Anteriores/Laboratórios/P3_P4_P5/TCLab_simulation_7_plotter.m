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

alpha = 50;

% Define covariance matrices -> Added for P4
QE = Ke * e_var * Ke';
RE = e_var;
delta_e = 2 * sqrt(e_var);

% Extend model matrices -> Added for P4
Ad = [A B; zeros(1, n) 1];
Bd = [B; 0];
Cd = [C 0];
QEd = [[QE zeros(n, 1)]; zeros(1, n) delta_e];

% Build the functions for applying the control and reading the temperature,
% mimicking the TCLab interface
x_ss = [eye(n) - A; C] \ [B * u_ss; y_ss];
c1 = ((eye(n) - A) * x_ss - B * u_ss);
c1 = c1 * 1.1;
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
d = zeros(1, N);
H = 50; % Horizon 
R = 0.05;

% Initialize signals
t = nan(1, N);
x = nan(n, N);
y = nan(1, N);
y_hat = nan(1,N);
Dy = nan(1,N);
Du = nan(1,N);
x(:, 1) = Dx0 + x_ss;
disturbance_estimated=nan(1,N);
% Added for P6
x_hat = nan(n,N);
x_hat_d = nan(n+1, N+1);

dx_hat = nan(n, N + 1);
du_hat = nan(1, N);
eta = nan(1, N);

% Reference Variation:
reference = zeros(N,1);
reference(1:150) = 50;
reference(151: 300) = 40;
reference(301: 450) = 60;
reference(451: 800) = 45;
Dr = reference - y_ss;

% Initialize estimate of Dx to be the real Dx + error
x_hat_d(:, 1) = [Dx0 + Ke * e_std * randn; 0];
x_hat(:, 1) = Dx0 + Ke * e_std * randn + x_ss;

% Kalman gain
L = dlqe(Ad, eye(n + 1), Cd, QEd, RE);

% Simulate incremental model
fprintf('Running simulation...')
for k = 1:N
    % Computes analog time
    t(k) = (k - 1) * Ts;

    % Reads the sensor temperature
    % Simulated
    y(:, k) = T1C(x(:, k));
    % Estimated
    y_hat(:,k) = T1C(x_hat(:, k));

    % Compute incremental variables
    Dy(:, k) = y(:, k) - y_ss;
    Du(:, k) = u(:, k) - u_ss;

    % Kalman filter correction step
    x_hat_d(:, k) = x_hat_d(:, k) + L * (Dy(:,k) - Cd * x_hat_d(:, k));

    d_bar = x_hat_d(4, k);
    Du_bar = (C / (eye(n) - A) * B) \ Dr(k) - d_bar;
    Dx_bar = (eye(n) - A) \ B * (Du_bar + d_bar);
    
    % Computes the control variable to apply
    [du_hat(:, k), eta(:, k)] = mpc_solve_5_2(x_hat_d(1:3, k) - Dx_bar, u_ss, Du_bar, Dr(k), y_ss, H, R, A, B, C, alpha);
    Du(:, k) = du_hat(:, k) + Du_bar;

    % Compute control
    u(:, k) = Du(:, k) + u_ss;

    % Kalman filter prediction step
    x_hat_d(:, k + 1) = Ad * x_hat_d(:, k) + Bd * Du(:, k);  

    % Applies the control variable to the plant
    % Simulated
    x(:, k+1) = h1(x(:, k), u(:, k));
    % Estimated
    x_hat(:, k+1) = x_hat_d(1:3, k) + x_ss;
    disturbance_estimated(:,k)= x_hat_d(4,k);
end
fprintf(' Done.\n');

%% Plots
figure(1)
subplot(2, 1, 1)
hold on
set(gca, 'ColorOrderIndex', 2)
stairs(t, reference, 'LineWidth', 2)
set(gca, 'ColorOrderIndex', 1)
plot(t, y, '.', 'MarkerSize', 10)
yline(55, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$[^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('$r$', '$y$', '$y_\mathrm{max}$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
%xlim([0 800])
grid on

subplot(2, 1, 2)
hold on
set(gca, 'ColorOrderIndex', 2)
stairs(t, reference, 'LineWidth', 2)
set(gca, 'ColorOrderIndex', 1)
plot(t, y_hat, '.', 'MarkerSize', 10)
yline(55, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$[^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('$r$', '$\hat{y}$', '$y_\mathrm{max}$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
%xlim([0 800])
grid on

figure(2)
subplot(2, 1, 1)
hold on
plot(t, abs(y - y_hat) ./ y * 100, '.', 'MarkerSize', 10, 'HandleVisibility', 'off')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\frac{|y - \hat{y}|}{y}$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
%xlim([0 800])
ylim([0 5])
grid on

subplot(2, 1, 2)
hold on
yline(1 ./ ((c1 - c1 / 1.1) \ B), 'k--')
plot(t, disturbance_estimated, '.', 'MarkerSize', 10, 'HandleVisibility', 'off')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$d$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
legend('$Bd = 0.1c_1$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
%xlim([0 800])
grid on

f = figure(3);
f.Position = [616 574 560 210];
hold on
plot(t, eta, '.', 'MarkerSize', 10)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\eta_1$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
%xlim([0 800])
grid on