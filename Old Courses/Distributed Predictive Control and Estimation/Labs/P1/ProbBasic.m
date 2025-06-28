% ProbBasic.m
% Illustrates the use of the Matlab solver fminunc and fmincon to find
% the minimum of the Rosenbrock function, both unconstrained and constrained.
%
% Required Toolboxes:
%   - Optimization Toolbox
% Functions used:
%   - BasicFunction.m (user-defined)
%
% IST, MEEC, Distributed Predictive Control and Estimation
% Adaptado por Alexandre Leal, Diogo Sampaio, 
% Francisco Tavares e Marta Valente
%--------------------------------------------------------------------------

% Plot configuration
LW = 'linewidth'; FS = 'fontsize'; MS = 'markersize';

% Define meshgrid ranges
x1min = -2; x1max = 2;
x2min = -2; x2max = 2;
N1 = 400; N2 = 400;

xv1 = linspace(x1min, x1max, N1);
xv2 = linspace(x2min, x2max, N2);
[xx1, xx2] = meshgrid(xv1, xv2);

% Evaluate function on grid
ff = zeros(size(xx1));
for ii = 1:N1
    for jj = 1:N2
        x = [xx1(ii, jj); xx2(ii, jj)];
        ff(ii, jj) = BasicFunction(x);
    end
end

% Initial estimate
x0 = [-1; 1];

% Unconstrained minimization (quasi-Newton)
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton');
xopt = fminunc(@BasicFunction, x0, options);

% Constrained minimization: x1 <= 0.5
Ac = [1 0];
Bc = 0.5;
xoptconstr = fmincon(@BasicFunction, x0, Ac, Bc);


%% ===================== FIGURE 1 =====================
% Rosenbrock function with optimization points and constraint visualization
figure(1)
set(gcf, 'Position', [100 100 800 700]);  % Set consistent size

% Compute log-scale of the Rosenbrock function for visualization
Zlog = log10(ff);
contourf(xx1, xx2, Zlog, 100, 'LineColor', 'none');
colormap('jet')

% Configure logarithmic colorbar
cb = colorbar;
cb.Ticks = log10([1e-1 1e0 1e1 1e2 1e3]);
cb.TickLabels = {'10^{-1}', '10^{0}', '10^{1}', '10^{2}', '10^{3}'};
cb.Label.String = '$f(x_1, x_2)$';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 14;

% Set color limits for consistency across figures
clim(log10([0.1 1e3]));

hold on

% Add white logarithmic level curves
log_levels = [1e-1 1e0 1e1 1e2 1e3];
contour(xx1, xx2, ff, log_levels, 'LineColor', 'w', 'LineWidth', 0.8);

% Highlight infeasible region (x1 > 0.5) with transparent blue fill
xfill = [0.5, x1max, x1max, 0.5];
yfill = [x2min, x2min, x2max, x2max];
fill(xfill, yfill, [0 0.4 1], 'FaceAlpha', 0.45, 'EdgeColor', 'none');

% Plot optimization points
h1 = plot(x0(1), x0(2), 'o', 'Color', 'r', LW, 1.5, 'MarkerSize', 15);  % Initial point
h2 = plot(xopt(1), xopt(2), 'x', 'Color', 'r', LW, 1.5, 'MarkerSize', 15);  % Unconstrained minimum
h3 = plot(xoptconstr(1), xoptconstr(2), '*', 'Color', 'r', LW, 1.5, 'MarkerSize', 15);  % Constrained minimum

% Add dashed boundary line at x1 = 0.5
plot([0.5 0.5], [x2min x2max], 'b--', 'LineWidth', 2);  

% Add region labels
text(0.6, -1, 'Infeasible Region', 'Color', 'black', 'FontSize', 16)
text(-1.25, -1, 'Feasible Region', 'Color', 'black', 'FontSize', 16)

% Axis labels and legend
xlabel('$x_1$', FS, 18, 'Interpreter', 'latex');
ylabel('$x_2$', FS, 18, 'Interpreter', 'latex');
title('Rosenbrock Function with Optimization Points and Constraint', ...
    'Interpreter', 'latex', 'FontSize', 20);

lgd = legend([h1 h2 h3], ...
    {'Initial point', 'Unconstrained minimum', 'Constrained minimum'}, ...
    'Location', 'southeast');
set(lgd, 'FontSize', 12, 'TextColor', 'black');

axis square
grid on
hold off

%% ===================== FIGURE 2 =====================
% 3D surface view of the Rosenbrock function
figure(2)
set(gcf, 'Position', [100 100 800 700]);  % Set consistent size
surf(xx1, xx2, ff, 'EdgeColor', 'none')
colormap('jet')
xlabel('$x_1$', FS, 16, 'Interpreter', 'latex');
ylabel('$x_2$', FS, 16, 'Interpreter', 'latex');
zlabel('$f(x_1, x_2)$', FS, 16, 'Interpreter', 'latex');
title('3D Surface of the Rosenbrock Function', 'FontSize', 18, 'Interpreter', 'latex')
axis tight
grid on

%% ===================== FIGURE 3 =====================
% Heatmap of the Rosenbrock function without optimization points
figure(3)
set(gcf, 'Position', [100 100 800 700]);  % Set consistent size
Zlog = log10(ff);
contourf(xx1, xx2, Zlog, 100, 'LineColor', 'none');
colormap('jet')

% Logarithmic colorbar
cb = colorbar;
cb.Ticks = log10([1e-1 1e0 1e1 1e2 1e3]);
cb.TickLabels = {'10^{-1}', '10^{0}', '10^{1}', '10^{2}', '10^{3}'};
cb.Label.String = '$f(x_1, x_2)$';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 14;

% Set color range
clim(log10([0.1 1e3]));

% Overlay white level curves
hold on
log_levels = [1e-1 1e0 1e1 1e2 1e3];
contour(xx1, xx2, ff, log_levels, 'LineColor', 'w', 'LineWidth', 1.0);

% Axis labeling
xlabel('$x_1$', FS, 18, 'Interpreter', 'latex');
ylabel('$x_2$', FS, 18, 'Interpreter', 'latex');
title('Rosenbrock Function', ...
      'Interpreter', 'latex', 'FontSize', 20)

axis square
grid on
hold off
