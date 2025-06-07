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
%e_std=0;
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
R = 0.05;
alpha = 50;  % maior alpha mais perto fica do ymax(55 graus)
             % mas gera um pouco de oscilacao do inicio
             % menor alpha tem propriedade inversa
% Initialize signals
t = nan(1, N);
x = nan(n, N);
y = nan(1, N);
eta = nan(1, N); 
Dy = nan(1, N);
Du = nan(1, N);
Dx = nan(n, N + 1);
x(:, 1) = Dx0 + x_ss;

% Added for P4
dx_hat = nan(n, N + 1);
du_hat = nan(1, N);
reference = 60; % Needs to be above 55 ºC
Dr = reference - y_ss;
Du_bar = (C / (eye(n) - A) * B) \ Dr;
Dx_bar = (eye(n) - A) \ B * Du_bar;

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
    Du(:, k) = u(:, k) - u_ss;
    dx_hat(:, k) = Dx(:, k) - Dx_bar;
    % Dense Formulation with hard constraints
    %du_hat(:, k) = mpc_solve_5(dx_hat(:, k), u_ss, Du_bar, Dr, y_ss, H, R, A, B, C); % Added for P4
    % Dense Formulation with soft constraints
    [du_hat(:, k), til] = mpc_solve_5_1(dx_hat(:, k), u_ss, Du_bar, Dr, y_ss, H, R, A, B, C, alpha); % Added for P4
    eta(:, k) = til;

    Du(:, k) = du_hat(:, k) + Du_bar;

    % Compute control -> Added for P4
    u(:, k) = Du(:, k) + u_ss;

    % Applies the control variable to the plant
    x(:, k + 1) = h1(x(:, k), u(:, k));
end
fprintf(' Done.\n');

%% Plots
% Plot absolute variables
f = figure(1);
f.Position = [616 574 1680 210];
subplot(1, 3, 1)
hold on
plot(t, y, '.', 'MarkerSize', 10)

subplot(1, 3, 2)
hold on
stairs(t, u, 'LineWidth', 2)

subplot(1, 3, 3)
hold on
plot(t, eta, '.', 'MarkerSize', 10)

%% With out noise
% Load model
load('singleheater_model.mat', 'A', 'B', 'C', 'Ke', 'e_var', 'y_ss', 'u_ss', 'Ts');
n = size(A, 1);
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
R = 0.05;
alpha = 50;  % maior alpha mais perto fica do ymax(55 graus)
             % mas gera um pouco de oscilacao do inicio
             % menor alpha tem propriedade inversa
% Initialize signals
t = nan(1, N);
x = nan(n, N);
y = nan(1, N);
eta = nan(1, N); 
Dy = nan(1, N);
Du = nan(1, N);
Dx = nan(n, N + 1);
x(:, 1) = Dx0 + x_ss;

% Added for P4
dx_hat = nan(n, N + 1);
du_hat = nan(1, N);
reference = 60; % Needs to be above 55 ºC
Dr = reference - y_ss;
Du_bar = (C / (eye(n) - A) * B) \ Dr;
Dx_bar = (eye(n) - A) \ B * Du_bar;

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
    Du(:, k) = u(:, k) - u_ss;
    dx_hat(:, k) = Dx(:, k) - Dx_bar;
    % Dense Formulation with hard constraints
    %du_hat(:, k) = mpc_solve_5(dx_hat(:, k), u_ss, Du_bar, Dr, y_ss, H, R, A, B, C); % Added for P4
    % Dense Formulation with soft constraints
    [du_hat(:, k), til] = mpc_solve_5_1(dx_hat(:, k), u_ss, Du_bar, Dr, y_ss, H, R, A, B, C, alpha); % Added for P4
    eta(:, k) = til;

    Du(:, k) = du_hat(:, k) + Du_bar;

    % Compute control -> Added for P4
    u(:, k) = Du(:, k) + u_ss;

    % Applies the control variable to the plant
    x(:, k + 1) = h1(x(:, k), u(:, k));
end
fprintf(' Done.\n');

%% Plots
% Plot absolute variables
f = figure(1);
f.Position = [616 574 1680 210];
subplot(1, 3, 1)
hold on
plot(t, y, '.', 'MarkerSize', 10)
yline(reference, 'k--')
yline(55, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('\texttt{e\char`_std}$ = 0.1283$', '\texttt{e\char`_std}$ = 0$', '$r$', '$y_\mathrm{max}$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
xlim([0 600])
grid on

subplot(1, 3, 2)
hold on
stairs(t, u, 'LineWidth', 2)
yline(0, 'r--')
yline(100, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$u$ [$\%$]', 'Interpreter', 'latex', 'FontSize', 18)
legend('\texttt{e\char`_std} $= 0.1283$', '\texttt{e\char`_std} $= 0$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
xlim([0 600])
grid on

subplot(1, 3, 3)
hold on
plot(t, eta, '.', 'MarkerSize', 10)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\eta_1$', 'Interpreter', 'latex', 'FontSize', 18)
legend('\texttt{e\char`_std} $= 0.1283$', '\texttt{e\char`_std} $= 0$', 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
xlim([0 600])
grid on