%% Optimal and Receding Horizon Gain Analysis – Stable and Unstable Systems

clear; clc; close all;

%% ---------------------------- PARAMETERS ----------------------------
Q = 1;
R_set = [0, 1, 10, 100];
H_vals = 1:15;

% Define systems
systems = struct( ...
    'unstable', struct('A', 1.2, 'B', 1), ...
    'stable',   struct('A', 0.8, 'B', 1)  ...
);

% Initialize result containers
results = struct();

%% ---------------------------- MAIN COMPUTATION LOOP ----------------------------
for sysName = fieldnames(systems)'
    sys = systems.(sysName{1});
    A = sys.A; B = sys.B;

    K_LQ_all = zeros(size(R_set));
    K_RH_all = zeros(length(R_set), length(H_vals));
    eig_RH_all = zeros(size(K_RH_all));

    for rIdx = 1:length(R_set)
        R = R_set(rIdx);
        [K_LQ, ~, ~] = dlqr(A, B, Q, R);
        K_LQ_all(rIdx) = K_LQ;

        for hIdx = 1:length(H_vals)
            H = H_vals(hIdx);

            % Construct Theta matrix
            Theta = zeros(H, H);
            for r = 1:H
                for c = 1:r
                    Theta(r, c) = A^(r-c) * B;
                end
            end

            % Construct Pi vector
            Pi = A.^(1:H)';

            % Cost matrices
            M = Theta' * Theta * Q + R * eye(H);
            W = Theta' * Pi * Q;

            % RH gain: K = e1 * (M \ W)
            e1 = [1, zeros(1, H-1)];
            K_RH = e1 * (M \ W);
            fprintf('System: %s | R = %g | H = %d | K_RH = %.6f\n', sysName{1}, R, H, K_RH);

            % Store results
            K_RH_all(rIdx, hIdx) = K_RH;
            eig_RH_all(rIdx, hIdx) = abs(A - B * K_RH);
        end
    end

    % Save for plotting
    results.(sysName{1}).K_LQ_all = K_LQ_all;
    results.(sysName{1}).K_RH_all = K_RH_all;
    results.(sysName{1}).eig_RH_all = eig_RH_all;
end

%% ---------------------------- PLOTTING ----------------------------

colors = lines(length(R_set));
markers = {'o', 'x', '^', 's'};
names = {'stable', 'unstable'};
titles = {'Stable', 'Unstable'};

for idx = 1:2
    name = names{idx};
    data = results.(name);
    A = systems.(name).A;
    B = systems.(name).B;
    K_LQ_all = data.K_LQ_all;
    K_RH_all = data.K_RH_all;
    eig_RH_all = data.eig_RH_all;

    % === Figure A: Gains K ===
    figure;
    hold on;
    for i = 1:length(R_set)
        plot(H_vals, K_RH_all(i,:), '-', ...
            'Color', colors(i,:), ...
            'LineWidth', 2, ...
            'Marker', markers{i}, ...
            'MarkerSize', 6, ...
            'DisplayName', sprintf('$R = %g$', R_set(i)));
        y = yline(K_LQ_all(i), '--', 'Color', colors(i,:), 'LineWidth', 1.5);
        y.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('$K$', 'Interpreter', 'latex', 'FontSize', 14);
    title(sprintf('Gains $K$ for %s System', titles{idx}), 'Interpreter', 'latex');
    ylim([0 1.3]);
    grid on;
    legend('Interpreter', 'latex', 'Location', 'best');
    set(gca, 'TickLabelInterpreter', 'latex');

    % === Figure B: Eigenvalues ===
    figure;
    hold on;
    for i = 1:length(R_set)
        plot(H_vals, eig_RH_all(i,:), '-', ...
            'Color', colors(i,:), ...
            'LineWidth', 2, ...
            'Marker', markers{i}, ...
            'MarkerSize', 6, ...
            'DisplayName', sprintf('$R = %g$', R_set(i)));
        y = yline(abs(A - B * K_LQ_all(i)), '--', 'Color', colors(i,:), 'LineWidth', 1.5);
        y.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    yline(1, '-', '$|\lambda| = 1$', ...
        'Color', 'k', 'LineWidth', 1.5, ...
        'Interpreter', 'latex', ...
        'DisplayName', '$|\lambda| = 1$');
    xlabel('$H$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('$|\lambda|$', 'Interpreter', 'latex', 'FontSize', 14);
    title(sprintf('Eigenvalues $|A - BK|$ for %s System', titles{idx}), 'Interpreter', 'latex');
    ylim([0 1.3]);
    grid on;
    legend('Interpreter', 'latex', 'Location', 'best');
    set(gca, 'TickLabelInterpreter', 'latex');
end
