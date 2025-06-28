% Simulation of the TCLab linear model previously identified
% Modified to run multiple H values and plot results

clear all
close all
clc

load('singleheater_model.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);
e_std = 0;
%e_std = sqrt(e_var); % input disturbance standard deviation
x_ss = [eye(n)-A; C]\[B*u_ss; y_ss];
c1 = ((eye(n)-A)*x_ss - B*u_ss);
c2 = (y_ss - C*x_ss);
h1 = @(x,u) A*x + B*u + Ke*e_std*randn + c1;
T1C = @(x) C*x + e_std*randn + c2;

T = 4000;
N = T/Ts;
H_values = [2, 3, 5, 10, 20,100];
colors = lines(length(H_values));

results = struct('t',[],'y',[],'u',[],'Dy',[],'Du',[]);

fprintf('Running simulation for multiple horizons...\n')

for idx = 1:length(H_values)
    H = H_values(idx);
    R = 0.1;

    u = nan(1,N);
    t = nan(1,N);
    x = nan(n,N);
    y = nan(1,N);
    Dy = nan(1,N);
    Du = nan(1,N);
    Dx = nan(n,N+1);

    Dx0Dy0 = [eye(n)-A, zeros(n,1); C, -1]\[-B*u_ss; 0];
    Dx0 = Dx0Dy0(1:n);

    du = 0;
    x(:,1) = Dx0 + x_ss;

    k_change = N/2;
    k_change2 = 3*N/4;
    Dr = 0;

    for k = 1:N
        t(k) = (k-1)*Ts;
        if k == k_change
            Dr = 0;
        elseif k == k_change2
            Dr = 0;
        end

        y(:,k) = T1C(x(:,k));
        Dy(:,k) = y(:,k) - y_ss;
        Dx(:,k) = x(:,k) - x_ss;

        dy = Dy(:,k) - Dr;
        Du_ss = pinv(C*((eye(3)-A)\B))*Dr;
        Dx_ss = ((eye(3)-A)\B)*Du_ss;
        dx = Dx(:,k) - Dx_ss;

        du = mpc_solve(dx, H, R, A, B, C, (u_ss + Du_ss), y(:,k), dy);
        Du(:,k) = Du_ss + du;
        u(:,k) = u_ss + Du(:,k);

        x(:,k+1) = h1(x(:,k), u(:,k));
    end

    results(idx).t = t;
    results(idx).y = y;
    results(idx).u = u;
    results(idx).Dy = Dy;
    results(idx).Du = Du;
end

fprintf('All simulations complete.\n')

%% Plots
figure('Units','normalized','Position',[0.2 0.5 0.3 0.4])
subplot(2,1,1), hold on, grid on
for i = 1:length(H_values)
    plot(results(i).t, results(i).y, 'Color', colors(i,:), 'DisplayName', sprintf('H=%d', H_values(i)))
end
%yl = yline(y_ss,'k--');
title('Absolute input/output')
xlabel('Time [s]')
ylabel('y [°C]')
legend show

subplot(2,1,2), hold on, grid on
for i = 1:length(H_values)
    stairs(results(i).t, results(i).u, 'Color', colors(i,:), 'DisplayName', sprintf('H=%d', H_values(i)))
end
%yl = yline(u_ss,'k--');
%yline(0,'r--')
%yline(100,'r--')
xlabel('Time [s]')
ylabel('u [%]')
legend show

%% Incremental plots
figure('Units','normalized','Position',[0.5 0.5 0.3 0.4])
subplot(2,1,1), hold on, grid on
for i = 1:length(H_values)
    plot(results(i).t, results(i).Dy, 'Color', colors(i,:), 'DisplayName', sprintf('H=%d', H_values(i)))
end
title('Incremental output \Deltay')
xlabel('Time [s]')
ylabel('\Delta{y} [°C]')
legend show

subplot(2,1,2), hold on, grid on
for i = 1:length(H_values)
    stairs(results(i).t, results(i).Du, 'Color', colors(i,:), 'DisplayName', sprintf('H=%d', H_values(i)))
end
%yline(-u_ss,'r--')
%yline(100-u_ss,'r--')
title('Incremental input \Deltau')
xlabel('Time [s]')
ylabel('\Delta{u} [%]')
legend show

% End of File
