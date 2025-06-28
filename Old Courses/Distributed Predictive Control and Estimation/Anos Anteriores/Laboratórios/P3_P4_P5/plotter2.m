clc
clear
close all

% Load data and select the output/input for the first heater only
load('openloop_data_1.mat','y','u','t');
u = u(1,:);
y = y(1,:);

% Choose interval for initial equilibrium
k_ss_begin = 201; % initial sample
k_ss_end = 400; % final sample

% Compute steady-state output/input from initial equilibrium
y_ss = mean(y(:,k_ss_begin:k_ss_end),2);
u_ss = u(:,k_ss_begin);

% Truncate initial transient
t = t(k_ss_begin:end-1);
u = u(:,k_ss_begin:end-1);
y = y(:,k_ss_begin:end-1);

% Compute incremental output/input
Dy = y - y_ss;
Du = u - u_ss;

% Identify state-space system for incremental dynamics
n = 3;

Ts = t(2) - t(1);
sys = ssest(Du',Dy',n,'Ts',Ts);
[A,B,C,~,Ke] = idssdata(sys);
e_var = sys.NoiseVariance;

% Test on dataset 1, with which the model was identified
% Initializations
N = length(t);
Dy_sim = nan(1,N);
Dx_sim = nan(n,N);

% Find initial incremental state that best fits the data given the identified model
Dx0 = findstates(sys,iddata(Dy',Du',Ts));

% Set initial conditions
Dy_sim(:,1) = Dy(:,1);
Dx_sim(:,1) = Dx0;

% Propagate model
for k = 1:N-1
    Dx_sim(:,k+1) = A*Dx_sim(:,k) + B*Du(:,k);
    Dy_sim(:,k+1) = C*Dx_sim(:,k+1);
end

%Plot results
figure(1)

subplot(2, 1, 1)
hold on
plot(t, Dy, '.')
plot(t, Dy_sim, 'LineWidth', 2)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\Delta y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('Experimental data', 'Model', 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 14)
ylim([-7 7])
grid on

subplot(2, 1, 2)
hold on
stairs(t, Du, 'LineWidth', 2)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\Delta u$ [\%]', 'Interpreter', 'latex', 'FontSize', 18)
grid on

% Test on dataset 2, with which the model was not identified
% Load data and select the output/input for the first heater only
load('openloop_data_2.mat','y','u','t');
u = u(1,:);
y = y(1,:);

% Compute incremental output/input
Dy2 = y - y_ss;
Du2 = u - u_ss;

% Initializations
N = length(t);
Dy2_sim = nan(1,N);
Dx2_sim = nan(n,N);

% Find initial incremental state that best fits the data given the identified model
Dx02 = findstates(sys,iddata(Dy2',Du2',Ts));

% Set initial conditions
Dy2_sim(:,1) = Dy2(:,1);
Dx2_sim(:,1) = Dx02;

% Propagate model
for k = 1:N-1
    Dx2_sim(:,k+1) = A*Dx2_sim(:,k) + B*Du2(:,k);
    Dy2_sim(:,k+1) = C*Dx2_sim(:,k+1);
end

%Plot results
figure(2)

subplot(2, 1, 1)
hold on
plot(t, Dy2, '.')
plot(t, Dy2_sim, 'LineWidth', 2)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\Delta y$ [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('Experimental data', 'Model', 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 14)
grid on

subplot(2, 1, 2)
hold on
stairs(t, Dy2 - Dy2_sim, 'LineWidth', 2)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\Delta y$ error [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
grid on