clear; clc; close all;

%% ----------------------------------------------------
%% P2.1 – Compute the optimal LQ state feedback gain
%% ----------------------------------------------------
Q = 1;
R_set = [0, 1, 10, 100];

systems = struct( ...
    'unstable', struct('A', 1.2, 'B', 1), ...
    'stable',   struct('A', 0.8, 'B', 1)  ...
);

fprintf('\n--- P2.1: Optimal LQ Gains ---\n');
for sysName = fieldnames(systems)'
    A = systems.(sysName{1}).A;
    B = systems.(sysName{1}).B;
    for rIdx = 1:length(R_set)
        R = R_set(rIdx);
        [K_LQ, ~, ~] = dlqr(A, B, Q, R);
        fprintf('System: %s | R = %g | K_LQ = %.6f\n', sysName{1}, R, K_LQ);
    end
end

%% ----------------------------------------------------
%% P2.2 – Compute the RH gains for different H and plot
%% ----------------------------------------------------
H_vals = 1:15;
results = struct();

fprintf('\n--- P2.2: Receding Horizon Gains ---\n');
for sysName = fieldnames(systems)'
    A = systems.(sysName{1}).A;
    B = systems.(sysName{1}).B;

    K_LQ_all = zeros(size(R_set));
    K_RH_all = zeros(length(R_set), length(H_vals));
    eig_RH_all = zeros(size(K_RH_all));

    for rIdx = 1:length(R_set)
        R = R_set(rIdx);
        [K_LQ, ~, ~] = dlqr(A, B, Q, R);
        K_LQ_all(rIdx) = K_LQ;

        for hIdx = 1:length(H_vals)
            H = H_vals(hIdx);

            Theta = zeros(H, H);
            for r = 1:H
                for c = 1:r
                    Theta(r, c) = A^(r-c) * B;
                end
            end
            Pi = A.^(1:H)';

            M = Theta' * Theta * Q + R * eye(H);
            W = Theta' * Pi * Q;

            e1 = [1, zeros(1, H-1)];
            K_RH = e1 * (M \ W);

            fprintf('System: %s | R = %g | H = %d | K_RH = %.6f\n', sysName{1}, R, H, K_RH);

            K_RH_all(rIdx, hIdx) = K_RH;
            eig_RH_all(rIdx, hIdx) = abs(A - B * K_RH);
        end
    end

    results.(sysName{1}).A = A;
    results.(sysName{1}).B = B;
    results.(sysName{1}).K_LQ_all = K_LQ_all;
    results.(sysName{1}).K_RH_all = K_RH_all;
    results.(sysName{1}).eig_RH_all = eig_RH_all;
end

%% ----------------------------------------------------
%% P2.3 – Plot eigenvalues and compare with stability
%% ----------------------------------------------------
colors = lines(length(R_set));
markers = {'o', 'x', '^', 's'};
names = {'unstable', 'stable'};
titles = {'Unstable', 'Stable'};

for idx = 1:2
    name = names{idx};
    data = results.(name);
    A = data.A;
    B = data.B;
    K_LQ_all = data.K_LQ_all;
    K_RH_all = data.K_RH_all;
    eig_RH_all = data.eig_RH_all;

    %% P2.2 Plot – Gains vs H
    figure;
    hold on;
    for i = 1:length(R_set)
        plot(H_vals, K_RH_all(i,:), '-', ...
            'Color', colors(i,:), ...
            'LineWidth', 2, ...
            'Marker', markers{i}, ...
            'DisplayName', sprintf('$R = %g$', R_set(i)));
        y = yline(K_LQ_all(i), '--', 'Color', colors(i,:), 'LineWidth', 1.5);
        y.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    title(sprintf('P2.2: Gains K for %s System', titles{idx}), 'Interpreter', 'latex');
    xlabel('$H$', 'Interpreter', 'latex'); ylabel('$K$', 'Interpreter', 'latex');
    legend('Interpreter', 'latex', 'Location', 'best'); grid on;

    %% P2.3 Plot – Eigenvalue vs H
    figure;
    hold on;
    for i = 1:length(R_set)
        plot(H_vals, eig_RH_all(i,:), '-', ...
            'Color', colors(i,:), ...
            'LineWidth', 2, ...
            'Marker', markers{i}, ...
            'DisplayName', sprintf('$R = %g$', R_set(i)));
        y = yline(abs(A - B * K_LQ_all(i)), '--', 'Color', colors(i,:), 'LineWidth', 1.5);
        y.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    yline(1, '-', '$|\lambda| = 1$', ...
        'Color', 'k', 'LineWidth', 1.5, ...
        'Interpreter', 'latex', ...
        'DisplayName', '$|\lambda| = 1$');
    title(sprintf('P2.3: Eigenvalues |A - BK| for %s System', titles{idx}), 'Interpreter', 'latex');
    xlabel('$H$', 'Interpreter', 'latex'); ylabel('$|\lambda|$', 'Interpreter', 'latex');
    legend('Interpreter', 'latex', 'Location', 'best'); grid on;
end

%% ----------------------------------------------------
%% P2.4 & P2.5: To be discussed analytically using the plots
%% ----------------------------------------------------
disp('--- P2.4: Observe convergence of K_RH → K_LQ as H increases ---');
disp('--- P2.5: Compare effect of horizon H on stability between stable and unstable systems using plots ---');
