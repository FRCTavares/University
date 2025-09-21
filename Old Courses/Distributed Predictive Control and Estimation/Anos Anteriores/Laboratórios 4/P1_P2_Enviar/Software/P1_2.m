% P1_2.m
%
% Description: This MATLAB code analyzes a mathematical function. It 
% includes the definition of the function, creation of plots to visualize 
% it, calculation of minima using optimization techniques, and 
% determination of attraction basins for one minimum. The results are 
% displayed in different figures, providing insights into the function's 
% behavior, minima, and attraction basins.
%
% Authors:
% * Afonso Bispo Certo (96134) - afonso.certo@tecnico.ulisboa.pt
% * Francisco Rodrigues (96210) - francisco.rodrigues.09@tecnico.ulisboa.pt
% * João Marafuz Gaspar (96240) - joao.marafuz.gaspar@tecnico.ulisboa.pt
% * Yandi Jiang (96344) - yandijiang@tecnico.ulisboa.pt
%__________________________________________________________________________

clc 
clear
close all

% Define the exercise function
f = @(x) x(1)^4 - 10 * x(1)^2 + x(2)^4 - 10 * x(2)^2;

%% First figure: Plot the function and minima
% Define the number of points and domain
N = 1001;
x1_min = -3; x1_max = 3;
x2_min = -3; x2_max = 3;
x1 = linspace(x1_min, x1_max, N);
x2 = linspace(x2_min, x2_max, N);
[X1, X2] = meshgrid(x1, x2);

C = zeros(N, N);
for ii = 1:N
    for jj = 1:N
        C(ii, jj) = f([X1(ii, jj), X2(ii, jj)]);
    end
end

% Compute the minima using fminunc
x0 = [2 2; 2 -2; -2 2; -2 -2];
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');

minima = zeros(4,2);
for ii = 1:4
    minima(ii, :) = fminunc(f, x0(ii, :), options);
end

% Plot the function
figure(1)
imagesc([x1_min x1_max], [x2_min x2_max], C)

% Set plot properties
axis([x1_min x1_max x2_min x2_max]), axis square
set(gca, 'ColorScale', 'linear', 'YDir', 'normal', 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 18)

% Create colorbar and set properties
cb = colorbar;
clim([-50, 0]);
set(cb, 'TickLabelInterpreter', 'latex', 'fontsize', 14)
set(cb.Label, 'String', '$f(x_1, x_2)$', 'Interpreter', 'latex', 'Rotation', -90, 'Position', [4.5 -25 0], 'FontSize', 18)

% Set colormap and grid
colormap(jet(4096))
grid on

% Plot the minima
hold on
plot(minima(:, 1), minima(:, 2), 'xr', 'MarkerSize', 15, 'LineWidth', 1.5)

%% Second figure: Calculate and plot attraction basins
x1_min = -6; x1_max = 250;
x2_min = -6; x2_max = 250;
N_grid_points = 100;
grid_resolution = abs(x2_max - x2_min) / N_grid_points;

% Get attraction basins using our approach
tStart1 = tic;
[our_approach_result] = get_boundary_DDA(f, 200, 1000, grid_resolution, 0.5, minima(1, :), options);
fprintf("Our Approach elapsed time: %.3f s\n", toc(tStart1))

% Get attraction basins using brute force
tStart2 = tic;
brute_force_result = get_boundary_brute_force(f, N_grid_points, options, minima(1,:), x1_min, x1_max, x2_min, x2_max, 0.5);
fprintf("Brute Force Approach elapsed time: %.3f s\n", toc(tStart2))

% Plot the resulting attraction basins
figure(2)

subplot(1, 2, 1)
imagesc([x1_min x1_max], [x2_min x2_max], brute_force_result), hold on
plot(our_approach_result(:, 1), our_approach_result(:, 2), 'LineWidth', 2, 'Color', 'red')
axis([x1_min x1_max, x2_min x2_max]), axis square
set(gca,'ColorScale', 'linear', 'YDir', 'normal', 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 18)

subplot(1, 2, 2)
imagesc([x1_min x1_max], [x2_min x2_max], brute_force_result), hold on
plot(our_approach_result(:, 1), our_approach_result(:, 2), 'LineWidth', 2, 'Color', 'red')
axis([x1_min x1_max / 5, x2_min x2_max / 5]), axis square
set(gca,'ColorScale', 'linear', 'YDir', 'normal', 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 18)
hold on
plot(minima(1, 1), minima(1, 2), 'xr', 'MarkerSize', 10, 'LineWidth', 1.5)