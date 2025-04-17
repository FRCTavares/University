% mpc_solve_5(x0, u_ss, Du_bar, Dr, y_ss, H, R, A, B, C)
%
% Description: Computes the gain for the Reciding Horizon (RH) optimal 
% control problem, given the system parameters, A, B and C, the cost
% function parameter R and que horizon H
%
% Authors: 
% * Afonso Bispo Certo  (96134) - afonso.certo@tecnico.ulisboa.pt
% * Francisco Rodrigues (96210) - francisco.rodrigues.09@tecnico.ulisboa.pt
% * João Marafuz Gaspar (96240) - joao.marafuz.gaspar@tecnico.ulisboa.pt
% * Yandi Jiang         (96344) - yandijiang@tecnico.ulisboa.pt
%__________________________________________________________________________

function [u0, FVAL] = mpc_solve_5(x0, u_ss, Du_bar, Dr, y_ss, H, R, A, B, C)
    % Get model order
    n = size(A, 1);

    %  Matrices initialization
    W = zeros(H, H);
    PI = zeros(H, n);
    I = eye(H);

    % Build PI and W matrices
    for ii = 1:H
        PI(ii, :) = C * A^ii;
        for jj = 1:H
            if jj > ii
                break
            end 
            W(ii,jj) = C * A^(ii - jj) * B;
        end
    end

    M = W' * W + R * I;

    % Add lower and upper bounds contraints
    lb = zeros(H, 1) - u_ss - Du_bar; 
    ub = 100 * ones(H, 1) - u_ss - Du_bar;

    % Safety constraint for a maximum temperature
    y_max = 55;
    G = eye(H);
    g = (y_max - y_ss - Dr) * ones(H, 1);
    Aineq = G * W;
    bineq = g - G * PI * x0;

    % Perform minimization using quadprog
    opts = optimoptions(@quadprog, 'Display', 'off');
    [U, FVAL, exitflag, ~] = quadprog(2 * M, 2 * x0' * PI' * W, Aineq, bineq, [], [], lb, ub, x0, opts);

    % Check whether the MPC optimization is successful
    if exitflag ~= 1
        fprintf("FLAG != 1");
        disp(exitflag)
        u0 = nan;
        FVAL = nan;
        %opts = optimoptions(@quadprog, 'Display', 'off');
        %U = quadprog(2 * M, 2 * x0' * PI' * W, Aineq, bineq, [], [], [], [], x0, opts);
        %save('last_U.mat', 'U', 'x0', 'PI', 'G', 'W', 'g', 'lb', 'ub', "Aineq", 'bineq');
    else
        save('last_U.mat', 'U', 'x0', 'PI', 'G', 'W', 'g', 'lb', 'ub', "Aineq", 'bineq');
        % We only want the first control command
        u0 = U(1);
    end
end