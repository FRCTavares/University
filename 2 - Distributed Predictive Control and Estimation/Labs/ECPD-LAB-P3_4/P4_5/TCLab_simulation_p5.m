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
clear all
close all
clc

% Load model
load('singleheater_model.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);
e_std = sqrt(e_var); % input disturbance standard deviation
% Build the functions for applying the control and reading the temperature,
% mimicking the TCLab interface
x_ss = [eye(n)-A; C]\[B*u_ss; y_ss];
c1 = ((eye(n)-A)*x_ss - B*u_ss)*1;
c2 = (y_ss - C*x_ss);
h1 = @(x,u) A*x + B*u + Ke*e_std*randn + c1; % apply control
T1C = @(x) C*x + e_std*randn + c2; % read temperature

% Simulation parameters
T = 1000; % Experiment duration [s]
N = T/Ts; % Number of samples to collect


% Parâmetros MPC
H = 20;    % horizonte de predição
R = 0.01;  % penalização do controlo

u = nan(1,N); % vetor de controlo (agora vai ser calculado a cada instante)
% Initial conditions (start at ambient temperature, i.e. equilibrium for u = 0)
Dx0Dy0 = [eye(n)-A, zeros(n,1); C, -1]\[-B*u_ss; 0];
Dx0 = Dx0Dy0(1:n);

% Initialize signals
t = nan(1,N);
x = nan(n,N);
y = nan(1,N);
Dy = nan(1,N);
Du = nan(1,N);
Dx = nan(n,N+1);

du=0;            
x(:,1) = Dx0 + x_ss;

k_change = N/2; 
% Simulate incremental model
fprintf('Running simulation...')
Dr=0;
for k = 1:N
   
    % Computes analog time
    t(k) = (k-1)*Ts;
    
    if k==k_change
        Dr=20;
    end


    % Reads the sensor temperature
    y(:,k) = T1C(x(:,k));

    % Compute incremental variables
    Dy(:,k) = y(:,k) - y_ss;
    Dx(:,k) = x(:,k) - x_ss;
    
    dy=Dy(:,k)-Dr;
    Du_ss=pinv(C*((eye(3)-A)\B))*Dr;
    Dx_ss=((eye(3)-A)\B)*Du_ss;
    dx=Dx(:,k)-Dx_ss;
    % Calcula controlo com MPC
    du = mpc_solve_p5(dx, H, R, A, B, C,(u_ss+Du_ss),y_ss,Dr,dy);
    Du(:,k)=Du_ss+du;
    u(:,k) = u_ss + Du(:,k);
    
    % Aplica controlo
    x(:,k+1) = h1(x(:,k), u(:,k));
end
fprintf(' Done.\n');

%% Plots
% Plot absolute variables
figure('Units','normalized','Position',[0.2 0.5 0.3 0.4])

subplot(2,1,1), hold on, grid on   
title('Absolute input/output')
plot(t,y,'.','MarkerSize',10)

% Reference line: y_ref = y_ss + Dr (after k_change)
y_ref = ones(1,N) * y_ss;
y_ref(k_change:end) = y_ss + Dr;
plot(t, y_ref, 'b--', 'LineWidth', 1.5, 'DisplayName', '$y_{ref}$')

% Equilibrium and max temp lines
yline(55, 'm--', 'y_{max}', 'LineWidth',1.5)
yl = yline(y_ss,'k--');

xlabel('Time [s]')
ylabel('y [°C]')
legend({'Measured $y$', '$y_{ref}$', '$y_{max}$', '$\bar{y}$'}, ...
    'Interpreter','latex','Location','best')
subplot(2,1,2), hold on, grid on   
stairs(t,u,'LineWidth',2)
yl=yline(u_ss,'k--');
yline(0,'r--')
yline(100,'r--')
xlabel('Time [s]')
ylabel('u [%]')
legend(yl,'$\bar{u}$','Interpreter','latex','Location','best');


%--------------------------------------------------------------------------
% End of File