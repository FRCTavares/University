% P2.m
%
% Description: This script computes the gains for the infinite horizon 
% Linear Quadratic (LQ) optimal control problem and for the Reciding 
% Horizon (RH) optimal control problem and the absolute value of the
% eigenvalues of A - bK as functions H, for several R
%
% Authors: 
% * Afonso Bispo Certo  (96134) - afonso.certo@tecnico.ulisboa.pt
% * Francisco Rodrigues (96210) - francisco.rodrigues.09@tecnico.ulisboa.pt
% * João Marafuz Gaspar (96240) - joao.marafuz.gaspar@tecnico.ulisboa.pt
% * Yandi Jiang         (96344) - yandijiang@tecnico.ulisboa.pt
%__________________________________________________________________________

clc
clear
close all

%% Parameters for P2.1, P2.2, P2.3 and P2.4
A = 1.2;
B = 1;
C = 1;
Q = C' * C;
R = 10;

%% P2_1
[KLQ, S, lambda] = dlqr(A, B, Q, R);
fprintf("P2.1: K_LQ = %f\n", KLQ);

%% P2.2
max_H = 15;
R_vector = [0 1 10 100];

H = 1:max_H;
K_RH = zeros(max_H, length(R_vector));
K_LQ = zeros(length(R_vector), 1);

% Calcultate RH gains and eigenvalues for diferent values of H and R
for ii = 1:length(R_vector)
    K_LQ(ii) = dlqr(A, B, Q, R_vector(ii));
    for jj = 1:max_H
        K_RH(jj, ii) = get_RH_gain(A,B,C, R_vector(ii), H(jj));
    end
end

figure(1)
hold on
color = lines(length(R_vector));
for ii = 1:length(R_vector)
    plot([0 max_H], [K_LQ(ii) K_LQ(ii)], "--", "Color", color(ii, :), 'LineWidth', 1.5);
    plot(H, K_RH(:, ii), 'o-', "DisplayName", sprintf("R = %.0f", R_vector(ii)), "Color", color(ii, :), 'LineWidth', 1.5);
end

% Set plot properties
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$K$', 'Interpreter', 'latex', 'FontSize', 18)
legend('', '$R = 0$', '', '$R = 1$', '', '$R = 10$', '', '$R = 100$', 'Interpreter', 'latex', 'FontSize', 14)
grid on


%% P2.3
eigenvalues = zeros(max_H, length(R_vector));

% Calcultate RH gains and eigenvalues for diferent values of H and R
for ii = 1:length(R_vector)
    K_LQ(ii) = dlqr(A, B, Q, R_vector(ii));
    for jj = 1:max_H
        eigenvalues(jj, ii) = eig(A - B * K_RH(jj, ii));
    end
end

figure(2)
hold on
color = lines(length(R_vector));
for ii = 1:length(R_vector)
    plot([0 max_H], eig(A - B * K_LQ(ii)) * [1 1], "--", "Color", color(ii, :), 'LineWidth', 1.5);
    plot(H, eigenvalues(:,ii), 'o-', "DisplayName", sprintf("R = %.0f", R_vector(ii)), "Color", color(ii, :), 'LineWidth', 1.5);
end

% Plot the stability boundary
hold on
yline(1, 'LineWidth', 2, 'Color', 'black');

% Set plot properties
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$|\lambda|, \ \left\{\lambda: \mathcal{C}_{A-bK}(\lambda) = 0\right\}$', 'Interpreter', 'latex', 'FontSize', 18)
legend('', '$R = 0$', '', '$R = 1$', '', '$R = 10$', '', '$R = 100$', 'Stability boundary', 'Interpreter', 'latex', 'FontSize', 14)
grid on

%% Parameters for P2.5
A = 0.8;
B = 1;
C = 1;
Q = C' * C;
R = 10;

%% P2.5
max_H = 15;
R_vector = [0 1 10 100];

H = 1:max_H;
K_RH = zeros(max_H, length(R_vector));
K_LQ = zeros(length(R_vector), 1);

% Calcultate RH gains and eigenvalues for diferent values of H and R
for ii = 1:length(R_vector)
    K_LQ(ii) = dlqr(A, B, Q, R_vector(ii));
    for jj = 1:max_H
        K_RH(jj, ii) = get_RH_gain(A,B,C, R_vector(ii), H(jj));
    end
end

figure(3)
hold on
color = lines(length(R_vector));
for ii = 1:length(R_vector)
    plot([0 max_H], [K_LQ(ii) K_LQ(ii)], "--", "Color", color(ii, :), 'LineWidth', 1.5);
    plot(H, K_RH(:, ii), 'o-', "DisplayName", sprintf("R = %.0f", R_vector(ii)), "Color", color(ii, :), 'LineWidth', 1.5);
end

% Set plot properties
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$K$', 'Interpreter', 'latex', 'FontSize', 18)
legend('', '$R = 0$', '', '$R = 1$', '', '$R = 10$', '', '$R = 100$', 'Interpreter', 'latex', 'FontSize', 14)
grid on

eigenvalues = zeros(max_H, length(R_vector));

% Calcultate RH gains and eigenvalues for diferent values of H and R
for ii = 1:length(R_vector)
    K_LQ(ii) = dlqr(A, B, Q, R_vector(ii));
    for jj = 1:max_H
        eigenvalues(jj, ii) = eig(A - B * K_RH(jj, ii));
    end
end

figure(4)
hold on
color = lines(length(R_vector));
for ii = 1:length(R_vector)
    plot([0 max_H], eig(A - B * K_LQ(ii)) * [1 1], "--", "Color", color(ii, :), 'LineWidth', 1.5);
    plot(H, eigenvalues(:,ii), 'o-', "DisplayName", sprintf("R = %.0f", R_vector(ii)), "Color", color(ii, :), 'LineWidth', 1.5);
end

% Plot the stability boundary
hold on
yline(1, 'LineWidth', 2, 'Color', 'black');

% Set plot properties
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$|\lambda|, \ \left\{\lambda: \mathcal{C}_{A-bK}(\lambda) = 0\right\}$', 'Interpreter', 'latex', 'FontSize', 18)
legend('', '$R = 0$', '', '$R = 1$', '', '$R = 10$', '', '$R = 100$', 'Stability boundary', 'Interpreter', 'latex', 'FontSize', 14)
grid on