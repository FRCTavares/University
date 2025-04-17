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
h1_dist = @(x,u) A * x + B * u + Ke * e_std * randn + 1.1 * c1; % apply control
T1C = @(x) C * x + e_std * randn + c2; % read temperature

% Simulation parameters
T = 4000; % Experiment duration [s]
N = T / Ts; % Number of samples to collect

% Initial conditions (start at ambient temperature, i.e. equilibrium for u = 0)
Dx0Dy0 = [eye(n) - A, zeros(n, 1); C, -1] \ [-B * u_ss; 0];
Dx0 = Dx0Dy0(1:n);

% Closed-loop control profile -> Added for P4
u = zeros(1, N); % Control
u_dist = zeros(1, N); % Control
H = 50; % Horizon 
R = 0.05;

% Initialize signals
t = nan(1, N);
x = nan(n, N);
y = nan(1, N);
Dy = nan(1, N);
Du = nan(1, N);
Dx = nan(n, N + 1);
x(:, 1) = Dx0 + x_ss;

x_dist = nan(n, N);
y_dist = nan(1, N);
Dy_dist = nan(1, N);
Du_dist = nan(1, N);
Dx_dist = nan(n, N + 1);
x_dist(:, 1) = Dx0 + x_ss;

% Added for P4
dx_hat_dist = nan(n, N + 1);
du_hat_dist = nan(1, N);
dx_hat = nan(n, N + 1);
du_hat = nan(1, N);
Dr = 0;
Du_bar = (C / (eye(n) - A) * B) \ Dr;
Dx_bar = (eye(n) - A) \ B * Du_bar;

% Simulate incremental model
fprintf('Running simulation...')
for k = 1:N
    % Computes analog time
    t(k) = (k - 1) * Ts;

    % Reads the sensor temperature
    y(:, k) = T1C(x(:, k));
    y_dist(:, k) = T1C(x_dist(:, k));

    % Compute incremental variables
    Dy(:, k) = y(:, k) - y_ss;
    Dy_dist(:, k) = y_dist(:, k) - y_ss;
    Dx(:, k) = x(:, k) - x_ss;
    Dx_dist(:, k) = x_dist(:, k) - x_ss;
    Du(:, k) = u(:, k) - u_ss;
    Du_dist(:, k) = u_dist(:, k) - u_ss;
    dx_hat(:, k) = Dx(:, k) - Dx_bar;
    dx_hat_dist(:, k) = Dx_dist(:, k) - Dx_bar;
    % DENSE FORMULATION
    du_hat(:, k) = mpc_solve_4(dx_hat(:, k), u_ss, Du_bar, H, R, A, B, C); % Added for P4
    du_hat_dist(:, k) = mpc_solve_4(dx_hat_dist(:, k), u_ss, Du_bar, H, R, A, B, C); % Added for P4

    Du(:, k) = du_hat(:, k) + Du_bar;
    Du_dist(:, k) = du_hat_dist(:, k) + Du_bar;

    % Compute control -> Added for P4
    u(:, k) = Du(:, k) + u_ss;
    u_dist(:, k) = Du_dist(:, k) + u_ss;

    % Applies the control variable to the plant
    x(:, k + 1) = h1(x(:, k), u(:, k));
    x_dist(:, k + 1) = h1_dist(x_dist(:, k), u_dist(:, k));
end
fprintf(' Done.\n');

%% Plots
close all
% Plot absolute variables
%f = figure(1);
%f.Position = [616 574 1120 210];
% figure(1)
% subplot(2, 1, 1)
% hold on
% plot(t, y, '.', 'MarkerSize', 10)
% plot(t, y_dist, '.', 'MarkerSize', 10)
% yline(y_ss + Dr, 'k--')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
% legend('original \texttt{c1}', '$10 \%$ increased \texttt{c1}', '$r = \bar{y} + \Delta r$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
% xlim([0 400])
% ylim([25 55])
% yyaxis right
% plot(t, abs(y - y_dist), '.', 'MarkerSize', 10, 'HandleVisibility', 'off')
% ylabel('$y$ error [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
% ylim([0 4])
% grid on
% 
% subplot(2, 1, 2)
% hold on
% stairs(t, u, 'LineWidth', 2)
% stairs(t, u_dist, 'LineWidth', 2)
% yline(u_ss + Du_bar, 'k--')
% yline(0, 'r--')
% yline(100, 'r--')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$u$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
% legend('original \texttt{c1}', '$10 \%$ increased \texttt{c1}', '$\bar{u} + \Delta \bar{u}$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
% xlim([0 400])
% yyaxis right
% stairs(t, abs(u - u_dist), 'LineWidth', 2, 'HandleVisibility', 'off')
% ylabel('$u$ error [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
% ylim([0 10])
% grid on

% Plot incremental variables
figure(2)
subplot(2, 1, 1)
hold on
plot(t, Dy - Dr, '.', 'MarkerSize', 10)
plot(t, Dy_dist - Dr, '.', 'MarkerSize', 10)
yline(0, 'k--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\delta y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('original \texttt{c1}', '$10 \%$ increased \texttt{c1}', '$\delta y = 0$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
xlim([0 400])
ylim([-20 5])
yyaxis right
plot(t, abs(Dy - Dy_dist), '.', 'MarkerSize', 10, 'HandleVisibility', 'off')
ylabel('$\delta y$ error [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
ylim([0 4])
grid on

subplot(2, 1, 2)
hold on
stairs(t, du_hat, 'LineWidth', 2)
stairs(t, du_hat_dist, 'LineWidth', 2)
yline(0, 'k--')
yline(- u_ss - Du_bar, 'r--')
yline(100 - u_ss - Du_bar, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\delta u$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
legend('original \texttt{c1}', '$10 \%$ increased \texttt{c1}', '$\delta u = 0$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
xlim([0 400])
ylim([-10 50])
yyaxis right
stairs(t, abs(du_hat - du_hat_dist), 'LineWidth', 2, 'HandleVisibility', 'off')
ylabel('$\delta u$ error [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
ylim([0 10])
grid on

% load('last_opt_4.mat')
% 
% figure(3)
% subplot(2, 1, 1)
% hold on
% plot(Y_aux, '.', 'MarkerSize', 10)
% yline(0, 'k--')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$\delta y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
% xlim([1 50])
% grid on
% 
% subplot(2, 1, 2)
% hold on
% stairs(U_aux, 'LineWidth', 2)
% yline(0, 'k--')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$\delta u$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
% xlim([1 50])
% grid on