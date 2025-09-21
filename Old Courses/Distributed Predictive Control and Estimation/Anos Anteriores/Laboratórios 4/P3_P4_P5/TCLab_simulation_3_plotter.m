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
u_rev = zeros(1, N); % Control
H = 50; % Horizon 
R = 0.01;

% Initialize signals
t = nan(1, N);
x = nan(n, N);
x_rev = nan(n, N);
y = nan(1, N);
y_rev = nan(1, N);
Dy = nan(1, N);
Dy_rev = nan(1, N);
Du = nan(1, N);
Du_rev = nan(1, N);
Dx = nan(n, N + 1);
Dx_rev = nan(n, N + 1);
x(:, 1) = Dx0 + x_ss;
x_rev(:, 1) = Dx0 + x_ss;

% Simulate incremental model
fprintf('Running simulation...')
for k = 1:N
    % Computes analog time
    t(k) = (k - 1) * Ts;

    % Reads the sensor temperature
    y(:, k) = T1C(x(:, k));
    y_rev(:, k) = T1C(x_rev(:, k));

    % Compute incremental variables
    Dy(:, k) = y(:, k) - y_ss;
    Dy_rev(:, k) = y_rev(:, k) - y_ss;
    Dx(:, k) = x(:, k) - x_ss;
    Dx_rev(:, k) = x_rev(:, k) - x_ss;
    % Du(:, k) = u(:, k) - u_ss;
    Du(:, k) = mpc_solve_2(Dx(:, k), H, R, A, B, C); % Added for P4
    Du_rev(:, k) = mpc_solve_3(Dx_rev(:, k), u_ss, H, R, A, B, C); % Added for P4

    % Compute control -> Added for P4
    u(:, k) = Du(:, k) + u_ss;
    u_rev(:, k) = Du_rev(:, k) + u_ss;

    % Applies the control variable to the plant
    x(:, k + 1) = h1(x(:, k), u(:, k));
    x_rev(:, k + 1) = h1(x_rev(:, k), u_rev(:, k));
end
fprintf(' Done.\n');

%% Plots
% Plot absolute variables
f = figure(1);
f.Position = [616 574 1120 210];
subplot(1, 2, 1)
hold on
plot(t, y, '.', 'MarkerSize', 10)
plot(t, y_rev, '.', 'MarkerSize', 10)
yline(y_ss, 'k--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('No control limits', 'With control limits', '$\bar{y}$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
xlim([0 150])
yyaxis right
plot(t, abs(y - y_rev), '.', 'MarkerSize', 10, 'HandleVisibility', 'off')
ylabel('$y$ error [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
ylim([0 4])
grid on

subplot(1, 2, 2)
hold on
stairs(t, u, 'LineWidth', 2)
stairs(t, u_rev, 'LineWidth', 2)
yline(u_ss, 'k--')
yline(0, 'r--')
yline(100, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$u$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
legend('No control limits', 'With control limits', '$\bar{u}$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
xlim([0 150])
grid on

% % Plot incremental variables
% figure(2)
% subplot(2, 1, 1)
% hold on
% plot(t, Dy, '.', 'MarkerSize', 10)
% plot(t, Dy_rev, '.', 'MarkerSize', 10)
% yline(0, 'k--')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$\Delta y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
% legend('No control limits', 'With control limits', '$\Delta y = 0$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
% xlim([0 150])
% yyaxis right
% plot(t, abs(Dy - Dy_rev), '.', 'MarkerSize', 10, 'HandleVisibility', 'off')
% ylabel('$\Delta y$ error [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
% ylim([0 4])
% grid on
% 
% subplot(2, 1, 2)
% hold on
% stairs(t, Du, 'LineWidth', 2)
% stairs(t, Du_rev, 'LineWidth', 2)
% yline(0, 'k--')
% yline(-u_ss, 'r--')
% yline(100 - u_ss, 'r--')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$\Delta u$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
% legend('No control limits', 'With control limits', '$\Delta u = 0$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
% xlim([0 150])
% grid on