% Closed-loop experiment for data collection in TCLab
%
% Copy of TCLab_openloop with some placeholder suggestions for where to
% place controller and state observer.
%
% If you see the warning 'Computation time exceeded sampling time by x
% seconds at sample k', it is because the computation time in a given
% loop took more than the sampling period Ts. Try to disable the rt_plot
% flag to fix it.
%
% Functions called: tclab.
%
% J. Miranda Lemos and Afonso Botelho, IST, May 2023
%__________________________________________________________________________

% Initialization
clear all %#ok<CLALL> 
close all
clc
tclab;

% Load model
load('singleheater_model.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);

% Experiment parameters
T = 300; % experiment duration [s]
N = T/Ts; % number of samples to collect

% Kalman filter design
% TO DO: Compute augmented matrices Ad, Bd, Cd
% TO DO: L = dlqe(...);

% Parâmetros MPC
H = 50;    % horizonte de predição
R = 0.01;  % penalização do controlo


Dr=5;%!!!!!!!!!!!!!!!!!!!!!!!!!!
d=0;
de=0.5;
Ad = [A,  B;
     zeros(1,n), 1];

Bd = [B; 0];

Cd = [C,  0];

Q_E = Ke * e_var * Ke';
QEd = blkdiag(Q_E, de);
L = dlqe(Ad,eye(n+1),Cd,QEd,e_var);
xdhat   = zeros(n+1,1);    % [Δx̂; d̂] starts at zero
du_prev = 0;               % last Δu

% Initial conditions (start at ambient temperature, i.e. equilibrium for u = 0)
Dx0Dy0 = [eye(n)-A, zeros(n,1); C, -1]\[-B*u_ss; 0];
Dx0 = Dx0Dy0(1:n); % to initialize filter

x_ss = [eye(n)-A; C]\[B*u_ss; y_ss];
yhat_inc  = nan(1,N);
yhat_abs   = nan(1,N);
dhat_hist =nan(1,N);
du=0;
x = nan(n,N);
x(:,1) = Dx0 + x_ss;

% Reference
%Dr = zeros(1,N);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%r = Dr + y_ss;
r = ones(1,N)*(Dr + y_ss);      % absolute reference temperature

% Real-time plot flag. If true, plots the input and measured temperature in
% real time. If false, only plots at the end of the experiment and instead
% prints the results in the command window.
rt_plot = true;

% Initialize figure and signals
if rt_plot
    figure
    drawnow;
end
t_cycle=nan(1,N);
t = nan(1,N);
u = zeros(1,N);
y = zeros(1,N);
Dy = nan(1,N);
Du = nan(1,N);
Dx = nan(n,N+1);
xd_est = nan(n+1,N);
xd_est(:,1) = [Dx0; 0]; % Kalman filter initialization

% String with date for saving results
timestr = char(datetime('now','Format','yyMMdd_HHmmSS'));

% Signals the start of the experiment by lighting the LED
led(1)
disp('Temperature test started.')

for k=1:N
    tic;
    if ( k >= N/2)
        Dr=-5;
        r(k) = (Dr + y_ss);
    end
    % Computes analog time
    t(k) = (k-1)*Ts;

    % Reads the sensor temperatures
    y(1,k) = T1C();

    % Compute incremental variables
    Dy(:,k) = y(:,k) - y_ss;
    dy=Dy(:,k)-Dr;
    Dx(:,k) = x(:,k) - x_ss;

    % Kalman filter correction step
    % TO DO: xd_est(:,k) = xd_est(:,k) + L*(Dy(:,k) - Cd*xd_est(:,k));

    %Kalman filter steps
    xd_est(:,k)  = xd_est(:,k) + L*(Dy(:,k) - Cd*xd_est(:,k));
    
    xdhat=xd_est(:,k);
    %xdhat = Ad*xdhat + Bd*du_prev;
    %xdhat  = xdhat + L*(Dy(:,k) - Cd*xdhat);
    Dx_est = xdhat(1:n);  
    d   = xdhat(n+1);  

    % Computes the control variable to apply
    % TO DO: [...] = mpc_solve(...)+
    %for plot
    yhat_inc(k)   = Cd * xdhat;      % incremental output prediction
    yhat_abs(k)   = y_ss + yhat_inc(k);  % absolute output prediction
    dhat_hist(k)  = xdhat(end);      % disturbance estimate
    
    % equaçoes fixes
    Du_ss=pinv(C*((eye(3)-A)\B))*Dr;
    Dx_ss=((eye(3)-A)\B)*Du_ss;
    Du_ss=Du_ss-d; %real Du_ss
    dx=Dx_est-Dx_ss;

    % Calcula controlo com MPC
    du = mpc_solve(dx, H, R, A, B, C,(u_ss+Du_ss),y_ss,Dr);
    Du(:,k)=Du_ss+du;
    %Du(:,k)=0;
    du_prev=Du(:,k);
    u(:,k) = Du(:,k) + u_ss;

    % Kalman filter prediction step
    % TO DO: xd_est(:,k+1) = Ad*xd_est(:,k) + Bd*Du(:,k);
    %xdhat = Ad*xdhat + Bd*du_prev;
    xd_est(:,k+1) = Ad*xd_est(:,k) + Bd*Du(:,k);
    %xdhat=xd_est;
    
    % Applies the control variable to the plant
    h1(u(1,k));

    if rt_plot
        % Plots results
        clf
        subplot(2,1,1), hold on, grid on   
        plot(t(1:k),y(1,1:k),'.','MarkerSize',10)
        stairs(t,r,'g--')
        xlabel('Time [s]')
        ylabel('y [°C]')
        subplot(2,1,2), hold on, grid on   
        stairs(t(1:k),u(1,1:k),'LineWidth',2)
        xlabel('Time [s]')
        ylabel('u [%]')
        ylim([0 100]);
        drawnow;
    else
        fprintf('t = %d, y1 = %.1f C, y2 = %.1f C, u1 = %.1f, u2 = %.1f\n',t(k),y(1,k),y(2,k),u(1,k),u(2,k)) %#ok<UNRCH> 
    end

    % Check if computation time did not exceed sampling time
    if toc > Ts
        t_cycle(k)=toc;
        warning('Computation time exceeded sampling time by %.2f s at sample %d.',toc-Ts,k)
    end
    % Waits for the begining of the new sampling interval
    pause(max(0,Ts-toc));
end

% Turns off both heaters at the end of the experiment
h1(0);
h2(0);

% Signals the end of the experiment by shutting off the LED
led(0)

disp('Temperature test complete.')

if ~rt_plot
    figure
    subplot(2,1,1), hold on, grid on   
    plot(t(1:k),y(1,1:k),'.','MarkerSize',10)
    stairs(t,r,'g--')
    xlabel('Time [s]')
    ylabel('y [°C]')
    subplot(2,1,2), hold on, grid on   
    stairs(t(1:k),u(1,1:k),'LineWidth',2)
    xlabel('Time [s]')
    ylabel('u [%]')
    ylim([0 100]);
end

%--------------------------------------------------------------------------

% Save figure and experiment data to file
exportgraphics(gcf,['openloop_plot_',timestr,'.png'],'Resolution',300)
save(['openloop_data_',timestr,'.mat'],'y','u','t');

%--------------------------------------------------------------------------
% End of File


