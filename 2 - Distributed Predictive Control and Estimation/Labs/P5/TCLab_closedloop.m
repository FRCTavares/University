% TCLab_closedloop.m
% Closed-loop experiment for data collection in TCLab (P5.1)
% A. Botelho & J. Miranda Lemos, IST, 2025
%__________________________________________________________________________

%% 1) Initialization
clear all %#ok<CLALL>
close all
clc

% Connect to the TCLab hardware (defines h1, h2, T1C, T2C, led)
tclab;

% Load the identified linear model
load('singleheater_model.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);

%% 2) Experiment parameters
T  = 4000;            % total experiment duration [s]
N  = T/Ts;            % number of samples
k1 = round(N/4);      % first set-point switch
k2 = round(N/2);      % second
k3 = round(3*N/4);    % third

%% 3) Kalman-filter design
% Augmented model to estimate Δx and disturbance d
Ad = [A, B;
      zeros(1,n), 1];
Bd = [B; 0];
Cd = [C, 0];

% Noise covariances (tuning)
Q_E = Ke * e_var * Ke';  % process noise on x
de  = 1;                 % trial STD for d
QEd = blkdiag(Q_E, de^2);
R   = e_var;             % measurement noise variance

% Steady-state Kalman gain
L = dlqe(Ad, eye(n+1), Cd, QEd, R);

%% 4) Reference profile (as in P4.7)
Dr = 50 - y_ss;           % initial offset
r  = nan(1,N);

%% 5) Visualization flag
rt_plot = true;

%% 6) Pre-allocate memory
t      = nan(1, N);
y      = nan(1, N);
Dy     = nan(1, N);
u      = nan(1, N);
Du     = nan(1, N);
dhat   = nan(1, N);
xd_est = nan(n+1, N+1);

% Compute initial Δx for the filter (model equilibrium)
tmp        = [eye(n)-A, zeros(n,1); C, -1]\[-B*u_ss; 0];
Dx0        = tmp(1:n);
xd_est(:,1) = [Dx0; 0];

%% 7) Start experiment
led(1);
disp('Closed-loop experiment started.');

%% 8) Main sampling loop
for k = 1:N
    tic
    t(k) = (k-1)*Ts;
    
    % update reference offsets
    if      k==1,    Dr = 50 - y_ss;
    elseif  k==k1,   Dr = 40 - y_ss;
    elseif  k==k2,   Dr = 60 - y_ss;
    elseif  k==k3,   Dr = 45 - y_ss;
    end
    r(k) = y_ss + Dr;
    
    % measurement
    y(k)  = T1C();
    Dy(k) = y(k) - y_ss;
    
    % Kalman-correct
    xd_est(:,k) = xd_est(:,k) + L*( Dy(k) - Cd*xd_est(:,k) );
    dhat(k)     = xd_est(end,k);
    
    % compute steady-state increments
    Du_ss = pinv( C * ((eye(n)-A)\B) ) * Dr - xd_est(end,k);
    Dx_ss = (eye(n)-A)\B * Du_ss;
    
    % form error for MPC
    dx = xd_est(1:n,k) - Dx_ss;
    
    % solve MPC (first Δu)
    H    = 20;
    Rmpc = 0.01;
    du   = mpc_solve(dx, H, Rmpc, A, B, C, u_ss + Du_ss, y_ss, Dr);
    Du(k) = Du_ss + du;
    u(k)  = u_ss + Du(k);
    
    % Kalman-predict for next step
    xd_est(:,k+1) = Ad * xd_est(:,k) + Bd * Du(k);
    
    % apply control to plant
    h1(u(k));
    
    % real-time plotting
    if rt_plot
        clf
        subplot(2,1,1), hold on, grid on
        plot(t(1:k), y(1:k),   '.','MarkerSize',8)
        plot(t(1:k), r(1:k), 'r--','LineWidth',1.5)
        xlabel('Time [s]'), ylabel('Temperature [°C]')
        title('Measured Temp vs. Reference')
        subplot(2,1,2), hold on, grid on
        stairs(t(1:k), u(1:k),'LineWidth',2)
        xlabel('Time [s]'), ylabel('Heater [%]'), ylim([0 100])
        drawnow
    else
        fprintf('t=%.1f  y=%.2f  ref=%.2f  u=%.2f  d̂=%.2f\n', ...
                t(k), y(k), r(k), u(k), dhat(k));
    end
    
    % timing check
    if toc > Ts
        warning('Overrun by %.3f s at k=%d', toc-Ts, k)
    end
    pause(max(0, Ts-toc));
end

%% 9) Shutdown and cleanup
h1(0); 
h2(0);
led(0);
disp('Experiment complete.');

%% 10) Final plots if not real-time
if ~rt_plot
    figure
    subplot(2,1,1), hold on, grid on
    plot(t, y, '.', 'MarkerSize',8), plot(t, r, 'r--','LineWidth',1.5)
    xlabel('Time [s]'), ylabel('Temp [°C]'), title('Temp vs Reference')
    subplot(2,1,2), hold on, grid on
    stairs(t, u, 'LineWidth',2)
    xlabel('Time [s]'), ylabel('Heater [%]'), ylim([0 100])
end

%% 11) Save data and figure
ts = char(datetime('now','Format','yyMMdd_HHmmSS'));
exportgraphics(gcf, ['closedloop_plot_' ts '.png'], 'Resolution',300);
save(['closedloop_data_' ts '.mat'], 't','y','r','u','dhat','xd_est','Du');
