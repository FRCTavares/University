% Initialization
clear
close all
clc

% Load model
load('singleheater_model.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);

% Load results
load('openloop_data_delta_e_2_e_std.mat')

% Experiment parameters
T = 4000; % experiment duration [s]
% T = 500; % experiment duration [s]
N = T/Ts; % number of samples to collect

% Extend model matrices
Ad = [A B; zeros(1, n) 1];
Bd = [B; 0];
Cd = [C 0];

% Reference
r = zeros(N,1);
r(1:150) = 50;
r(151: 300) = 40;
r(301: 450) = 60;
r(451: 800) = 45;
Dr = r - y_ss;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shared_axes = [];

%% Plots
figure(1)
subplot(2, 1, 1)
hold on
set(gca, 'ColorOrderIndex', 2)
stairs(t, r, 'LineWidth', 2)
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
stairs(t, u, 'LineWidth', 2)
yline(100, 'r--')
yline(0, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$u$ [\%]', 'Interpreter', 'latex', 'FontSize', 18)
%xlim([0 800])
grid on

figure(2)
subplot(2, 1, 1)
hold on
plot(t, abs(y - (Cd * xd_est(:,1:N) + y_ss)) ./ y * 100, '.', 'MarkerSize', 10, 'HandleVisibility', 'off')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\frac{|y - \hat{y}|}{y}$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
%xlim([0 800])
% ylim([0 5])
grid on

subplot(2, 1, 2)
hold on
plot(t, xd_est(4,1:N), '.', 'MarkerSize', 10)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$d$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
%xlim([0 800])
grid on