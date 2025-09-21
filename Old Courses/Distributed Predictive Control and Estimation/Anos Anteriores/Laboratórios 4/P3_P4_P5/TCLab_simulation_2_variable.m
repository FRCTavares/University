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
% e_std = sqrt(e_var); % input disturbance standard deviation
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

%% Variable R
% % Closed-loop control profile -> Added for P4
% R_values = [0.02 0.05 0.1 0.5 1]; 
% H = 50; % Horizon
% 
% f = figure(1);
% f.Position = [616 574 1120 210];
% hold on
% for ii = 1:length(R_values)
%     fprintf('Running simulation for R = %f...', R_values(ii))
%     % Closed-loop control profile -> Added for P4
%     u = zeros(1, N); % Control
% 
%     % Initialize signals
%     t = nan(1, N);
%     x = nan(n, N);
%     y = nan(1, N);
%     Dy = nan(1, N);
%     Du = nan(1, N);
%     Dx = nan(n, N + 1);
%     x(:, 1) = Dx0 + x_ss;
%     
%     % Added for P4
%     dx_hat = nan(n, N + 1);
%     du_hat = nan(1, N);
%     
%     % Simulate incremental model
%     for k = 1:N
%         % Computes analog time
%         t(k) = (k - 1) * Ts;
%     
%         % Reads the sensor temperature
%         y(:, k) = T1C(x(:, k));
%     
%         % Compute incremental variables
%         Dy(:, k) = y(:, k) - y_ss;
%         Dx(:, k) = x(:, k) - x_ss;
%         % Du(:, k) = u(:, k) - u_ss;
%         Du(:, k) = mpc_solve_2(Dx(:, k), H, R_values(ii), A, B, C); % Added for P4
%     
%         % Compute control -> Added for P4
%         u(:, k) = Du(:, k) + u_ss;
%     
%         % Applies the control variable to the plant
%         x(:, k + 1) = h1(x(:, k), u(:, k));
%     end
%     fprintf(' Done.\n');
% 
%     subplot(1, 2, 1)
%     hold on
%     plot(t, y, '.', 'MarkerSize', 10);
% 
%     subplot(1, 2, 2)
%     hold on
%     stairs(t, u, 'LineWidth', 2);
% end
% 
% subplot(1, 2, 1)
% hold on
% yline(y_ss, 'k--')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
% legend_labels = arrayfun(@(val) ['$R =$ ' num2str(val)], R_values, 'UniformOutput', false);
% legend_labels = [legend_labels, '$\bar{y}$'];
% legend(legend_labels, 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
% % title('Variable $R$ for $H =$ ' + sprintf("%d", H), 'Interpreter', 'latex', 'FontSize', 18)
% xlim([0 400])
% grid on
% 
% subplot(1, 2, 2)
% hold on
% yline(u_ss, 'k--')
% yline(0, 'r--')
% yline(100, 'r--')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('$u$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
% legend_labels = arrayfun(@(val) ['$R =$ ' num2str(val)], R_values, 'UniformOutput', false);
% legend_labels = [legend_labels, '$\bar{u}$'];
% legend(legend_labels, 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
% % title('Variable $R$ for $H =$ ' + sprintf("%d", H), 'Interpreter', 'latex', 'FontSize', 18)
% xlim([0 400])
% grid on

%% Variable H
% Closed-loop control profile -> Added for P4
R = 0.05; 
%H_values = [1 20:20:200]; % Horizon for runtime plot
H_values = [10 20 50 70]; % Horizon for response plot
Dy_50_70 = zeros(1, N);
Du_50_70 = zeros(1, N);

runtimes = zeros(length(H_values), 1);

f2 = figure(2);
f2.Position = [616 574 1120 210];
hold on
for ii = 1:length(H_values)
    tic
    fprintf('Running simulation for H = %d...', H_values(ii))
    % Closed-loop control profile -> Added for P4
    u = zeros(1, N); % Control

    % Initialize signals
    t = nan(1, N);
    x = nan(n, N);
    y = nan(1, N);
    Dy = nan(1, N);
    Du = nan(1, N);
    Dx = nan(n, N + 1);
    x(:, 1) = Dx0 + x_ss;
    
    % Added for P4
    dx_hat = nan(n, N + 1);
    du_hat = nan(1, N);
    
    % Simulate incremental model
    for k = 1:N
        % Computes analog time
        t(k) = (k - 1) * Ts;
    
        % Reads the sensor temperature
        y(:, k) = T1C(x(:, k));
    
        % Compute incremental variables
        Dy(:, k) = y(:, k) - y_ss;
        Dx(:, k) = x(:, k) - x_ss;
        % Du(:, k) = u(:, k) - u_ss;
        Du(:,k) = mpc_solve_2(Dx(:, k), H_values(ii), R, A, B, C); % Added for P4
    
        % Compute control -> Added for P4
        u(:, k) = Du(:, k) + u_ss;
    
        % Applies the control variable to the plant
        x(:, k + 1) = h1(x(:, k), u(:, k));

        if H_values(ii) == 50
            Dy_50_70(1, k) = y(1, k);
            Du_50_70(1, k) = u(1, k);
        elseif H_values(ii) == 70
            Dy_50_70(1, k) = Dy_50_70(1, k) - y(1, k);
            Du_50_70(1, k) = Du_50_70(1, k) - u(1, k);
        end
    end
    runtimes(ii) = toc;
    fprintf(' Done.\n');
    
    subplot(1, 2, 1)
    hold on
    plot(t, y, '.', 'MarkerSize', 10)

    subplot(1, 2, 2)
    hold on
    stairs(t, u, 'LineWidth', 2)
end

subplot(1, 2, 1)
hold on
yline(y_ss, 'k--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend_labels = arrayfun(@(val) ['$H =$ ' num2str(val)], H_values, 'UniformOutput', false);
legend_labels = [legend_labels, '$\bar{y}$'];
legend(legend_labels, 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
% title('Variable $H$ for $R =$ ' + sprintf("%d", R), 'Interpreter', 'latex', 'FontSize', 18)
xlim([0 500])
yyaxis right
plot(t, abs(Dy_50_70), '.', 'MarkerSize', 10, 'HandleVisibility', 'off');
ylim([0 0.015])
ylabel('$|y_{H = 70} - y_{H = 50}|$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
grid on

subplot(1, 2, 2)
hold on
yline(u_ss, 'k--')
yline(0, 'r--')
yline(100, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$u$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
legend_labels = arrayfun(@(val) ['$H =$ ' num2str(val)], H_values, 'UniformOutput', false);
legend_labels = [legend_labels, '$\bar{u}$', ''];
legend(legend_labels, 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
% title('Variable $H$ for $R =$ ' + sprintf("%d", R), 'Interpreter', 'latex', 'FontSize', 18)
xlim([0 500])
%ylim([20 80])
yyaxis right
stairs(t, abs(Du_50_70), 'LineWidth', 2, 'HandleVisibility', 'off');
ylim([0 0.2])
ylabel('$|u_{H = 70} - u_{H = 50}|$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
grid on

% %% Runtime as function of H
% figure(3)
% scatter(H_values, runtimes / T * 100, 'o', 'filled')
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
% xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 18)
% ylabel('Runtime [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
% grid on