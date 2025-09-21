% P1_1.m
%
% Description: Illustrates the use of the Matlab solvers fminunc and 
% fmincon with the Rosenbrock function.
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

% Rosenbrock function
f = @(x) 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2;

% Initial minimum estimate
x0 = [-1, 1];

% Computation of the unconstrained minimum 
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');
xoptunconstr = fminunc(f, x0, options);

% Computation of the constrained minimum 
A = [1, 0];
B = 0.5;
xoptconstr = fmincon(f, x0, A, B);

% Preparing the data to be ploted
N = 501;
x1_min = -2; x1_max = 2;
x2_min = -1; x2_max = 2;
x1 = linspace(x1_min, x1_max, N);
x2 = linspace(x2_min, x2_max, N);
[X1, X2] = meshgrid(x1, x2);

C = zeros(N, N);
for ii = 1:N
    for jj = 1:N
        C(ii, jj) = f([X1(ii, jj), X2(ii, jj)]);
    end
end

%% First figure: Plot the Rosenbrock function
% Plot the function
figure(1)
imagesc([x1_min x1_max], [x2_min x2_max], C)

% Set plot properties
axis([x1_min x1_max x2_min x2_max]), axis square
set(gca,'ColorScale', 'log', 'YDir', 'normal', 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 18)

% Create colorbar and set properties
cb = colorbar;
set(cb, 'TickLabelInterpreter', 'latex', 'fontsize', 14)
set(cb.Label, 'String', '$f(x_1, x_2)$', 'Interpreter', 'latex', 'Rotation', -90, 'Position', [4.5 25.0901 0], 'FontSize', 18)

% Set colormap and grid
colormap(jet(4096))
grid on

%% Second figure: Compute unconstrained and constrained minima and plot them
% Set contour levels
levels = logspace(-1, 3, 6);

% Plot the image and contours
figure(2)
imagesc([x1_min x1_max], [x2_min x2_max], C), hold on
contour(X1, X2, C, levels, 'LineWidth', 1.5, 'LineColor', 'white')

% Set plot properties
axis([x1_min x1_max x2_min x2_max]), axis square
set(gca, 'ColorScale', 'log', 'YDir', 'normal', 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 18)

% Create binary matrix for infeasible region
infeasible = (X1 > 0.5);
infeasible = double(infeasible);

% Set brightness for infeasible region
infeasible(infeasible == 1) = 0.4;

% Set alphadata to adjust brightness
h = imagesc(x1, x2, infeasible);
set(h, 'AlphaData', infeasible);

% Create colorbar and set properties
cb = colorbar;
set(cb, 'TickLabelInterpreter', 'latex', 'fontsize', 14)
set(cb.Label, 'String', '$f(x_1, x_2)$', 'Interpreter', 'latex', 'Rotation', -90, 'Position', [4.5 7.93 0], 'FontSize', 18)

% Set colormap and grid
colormap(jet(4096))
grid on

% Plot the minimums
plot(x0(1), x0(2), 'or', 'MarkerSize', 15, 'LineWidth', 2)
plot(xoptunconstr(1), xoptunconstr(2), 'xr', 'MarkerSize', 15, 'LineWidth', 1.5)
plot(xoptconstr(1), xoptconstr(2), '*r', 'MarkerSize', 15, 'LineWidth', 1.5)

xline(0.5, '--', 'LineWidth', 1.3)
text(-1.15, 1.8, 'Feasible Region', 'Interpreter', 'latex', 'FontSize', 14)
text(0.65, 1.8, 'Infeasible Region', 'Interpreter', 'latex', 'FontSize', 14)
xtick_vals = sort([get(gca,'xtick'), 0.5]);
xticks(xtick_vals);

% Display the results
clc
fprintf('Unconstrained solution:\n')
fprintf('xoptunconstr = [%.7f %.7f] => f(xoptunconstr) = %.7f\n', xoptunconstr(1), xoptunconstr(2), f(xoptunconstr))
fprintf('Constrained solution:\n')
fprintf('xoptconstr = [%.7f %.7f] => f(xoptconstr) = %.7f\n', xoptconstr(1), xoptconstr(2), f(xoptconstr))