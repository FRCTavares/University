clear all
close all
clc

% Load model
load('singleheater_model.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);
e_std = 0;

% Simulation parameters
T = 500;
N = T/Ts;
H = 20;
R = 0.01;
Dr_step = 5;
k_change = N/2;

% Initialize storage for both runs
results = struct('u', [], 'y', [], 't', []);

for perturbed = [false true]
    % Compute steady-state x_ss
    x_ss = [eye(n)-A; C]\[B*u_ss; y_ss];

    % Set disturbance model residual (perturb c1 if needed)
    if perturbed
        c1 = ((eye(n)-A)*x_ss - B*u_ss)*1.5;  % 10% model mismatch
    else
        c1 = ((eye(n)-A)*x_ss - B*u_ss);      % nominal model
    end
    c2 = y_ss - C*x_ss;

    % Emulate plant
    h1 = @(x,u) A*x + B*u + Ke*e_std*randn + c1;
    T1C = @(x) C*x + e_std*randn + c2;

    % Initialize variables
    u = nan(1,N); y = nan(1,N); t = nan(1,N);
    x = nan(n,N); Dx = nan(n,N+1);

    Dx0Dy0 = [eye(n)-A, zeros(n,1); C, -1]\[-B*u_ss; 0];
    Dx0 = Dx0Dy0(1:n);
    x(:,1) = Dx0 + x_ss;

    Dr = 0;

    for k = 1:N
        t(k) = (k-1)*Ts;

        if k == k_change
            Dr = Dr_step;
        end

        y(:,k) = T1C(x(:,k));
        Dx(:,k) = x(:,k) - x_ss;

        Du_ss = pinv(C * ((eye(n)-A) \ B)) * Dr;
        Dx_ss = (eye(n) - A) \ B * Du_ss;
        dx = Dx(:,k) - Dx_ss;

        du = mpc_solve_p4(dx, H, R, A, B, C, u_ss + Du_ss, y_ss, Dr);
        Du = Du_ss + du;
        u(:,k) = u_ss + Du;

        x(:,k+1) = h1(x(:,k), u(:,k));
    end

    % Save results
    idx = 1 + perturbed;
    results(idx).u = u;
    results(idx).y = y;
    results(idx).t = t;
end

%% Calculate incremental variables and errors

t = results(1).t;
delta_y_nom = results(1).y - y_ss;
delta_y_pert = results(2).y - y_ss;
delta_u_nom = results(1).u - u_ss;
delta_u_pert = results(2).u - u_ss;

error_dy = abs(delta_y_pert - delta_y_nom);
error_du = abs(delta_u_pert - delta_u_nom);

%% INCREMENTAL PLOTS (Δy and Δu)

figure('Units','normalized','Position',[0.3 0.5 0.4 0.45])

% Δy plot with reference lines
subplot(2,1,1), hold on, grid on
title('Comparison of Incremental Output Δy(t)')
plot(t, delta_y_nom, 'b', 'DisplayName','Nominal')
plot(t, delta_y_pert, 'r--', 'DisplayName','Perturbed')
yline(Dr_step, 'k--', 'DisplayName','Δr = 5°C')
xline(t(k_change), 'k-.', 'DisplayName','Ref change','LineWidth', 1)
xlabel('Time [s]')
ylabel('Δy [°C]')
legend('Location','best')

% Δu plot
subplot(2,1,2), hold on, grid on
title('Comparison of Incremental Input Δu(t)')
stairs(t, delta_u_nom, 'b', 'DisplayName','Nominal')
stairs(t, delta_u_pert, 'r--', 'DisplayName','Perturbed')
xlabel('Time [s]')
ylabel('Δu [%]')
legend('Location','best')

%% STANDALONE ERROR PLOT (Δy and Δu errors)

figure('Units','normalized','Position',[0.35 0.5 0.4 0.3])
yyaxis left
plot(t, error_dy, 'Color', [0.129 0.294 0.625], 'LineWidth', 2)
ylabel('Δy error [°C]')
ylim([0 inf])
grid on

yyaxis right
plot(t, error_du, 'Color', [0.850 0.325 0.098], 'LineWidth', 2)
ylabel('Δu error [%]')
ylim([0 inf])

xlabel('Time [s]')
title('Tracking Error: Δy and Δu')
legend({'Δy error', 'Δu error'}, 'Location', 'best')
