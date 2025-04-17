% Open-loop experiment for data collection in TCLab
%
% Initializes TCLab, applies a sequence of open-loop controls and records
% the corresponding temperature.
%
% If you see the warning 'Computation time exceeded sampling time by x
% seconds at sample k', it is because the computation time in a given
% loop took more than the sampling period Ts. Try to disable the rt_plot
% flag to fix it or increase Ts.
%
% Functions called: tclab.
%
% J. Miranda Lemos and Afonso Botelho, IST, May 2023
%__________________________________________________________________________

% Initialization
clear all
close all
clc
tclab;

% Experiment parameters
T = 8000; % experiment duration [s]
Ts = 5; % sampling period [s]
N = T/Ts; % number of samples to collect
N_transient = 1000/Ts; % samples in transient period
u_eq=30; % equilibrium control value
% Open-loop profile
u = zeros(2,N);
u(1,2:N_transient*2) = u_eq; % equilibrium point
u(1,N_transient*2+1:N_transient*3) = u_eq+5; 
u(1,N_transient*3+1:N_transient*4) = u_eq;
u(1,N_transient*4+1:N_transient*5) = u_eq-5;
u(1,N_transient*5+1:N_transient*6) = u_eq+10;
u(1,N_transient*6+1:N_transient*7) = u_eq-10;
u(1,N_transient*7+1:N) = u_eq;

% Real-time plot flag. If true, plots the input and measured temperature in
% real time. If false, only plots at the end of the experiment and instead
% prints the results in the command window.
rt_plot = true;

% Initialize figure and signals
if rt_plot
    figure
    drawnow;
end
t = nan(1,N);
y = nan(2,N);

% String with date for saving results
timestr = char(datetime('now','Format','yyMMdd_HHmmSS'));

% Signals the start of the experiment by lighting the LED
led(1)
disp('Temperature test started.')

for k=1:N
    tic;

    % Computes analog time
    t(k) = (k-1)*Ts;

    % Reads the sensor temperatures
    y(1,k) = T1C();
    y(2,k) = T2C();

    % Applies the control variables to the plant
    h1(u(1,k));
    h2(u(2,k));

    if rt_plot
        % Plots results
        clf
        subplot(2,1,1), hold on, grid on   
        plot(t(1:k),y(1,1:k),'.','MarkerSize',10)
        plot(t(1:k),y(2,1:k),'.','MarkerSize',10)
        legend('Temperature 1','Temperature 2','Location','northwest')
        xlabel('Time [s]')
        ylabel('Temperature [°C]')
        subplot(2,1,2), hold on, grid on   
        stairs(t(1:k),u(1,1:k),'LineWidth',2)
        stairs(t(1:k),u(2,1:k),'LineWidth',2)
        legend('Heater 1','Heater 2','Location','northwest')
        xlabel('Time [s]')
        ylabel('Heater [%]')
        ylim([0 100]);
        drawnow;
    else
        fprintf('t = %d, y1 = %.1f C, y2 = %.1f C, u1 = %.1f, u2 = %.1f\n',t(k),y(1,k),y(2,k),u(1,k),u(2,k))
    end

    % Check if computation time did not exceed sampling time
    if toc > Ts
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
    plot(t,y(1,:),'.','MarkerSize',10)
    plot(t,y(2,:),'.','MarkerSize',10)
    legend('Temperature 1','Temperature 2','Location','best')
    xlabel('Time [s]')
    ylabel('Temperature [°C]')
    subplot(2,1,2), hold on, grid on   
    stairs(t,u(1,:),'LineWidth',2)
    stairs(t,u(2,:),'LineWidth',2)
    legend('Heater 1','Heater 2','Location','best')
    xlabel('Time [s]')
    ylabel('Heater control [%]')
    ylim([0 100]);
end

%--------------------------------------------------------------------------

% Save figure and experiment data to file
exportgraphics(gcf,['openloop_plot_',timestr,'.png'],'Resolution',300)
save(['openloop_data_',timestr,'.mat'],'y','u','t');

%--------------------------------------------------------------------------
% End of File


