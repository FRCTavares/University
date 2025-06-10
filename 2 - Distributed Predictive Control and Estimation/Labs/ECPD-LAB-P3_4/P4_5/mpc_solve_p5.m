function u0 = mpc_solve_p5(x0, H, R, A, B, C, u_ss, y_ss, Dr, dy)
    % mpc_solve - MPC with soft safety constraint y <= 55°C
    %   Implements a slack-variable softening of y <= y_max = 55 - y_ss - Dr - dy.

    n = size(A,1);  % número de estados
    m = size(B,2);  % número de entradas
    p = size(C,1);  % número de saídas

    % Build prediction matrices W and Pi
    W = zeros(H*p, H*m);
    Pi = zeros(H*p, n);
    for i = 1:H
        Pi((i-1)*p+1:i*p, :) = C * (A^i);
        for j = 1:i
            W((i-1)*p+1:i*p, (j-1)*m+1:j*m) = C * (A^(i-j)) * B;
        end
    end

    % Cost matrices: augment slack penalty
    Qy = eye(H*p);             % output tracking weight
    Ru = R * eye(H*m);         % control increment weight
    alpha = 1e4;               % slack variable penalty (large)
    H_eta = alpha * eye(H*p);

    % Build Hessian F_z  = 2 * [W'QyW + Ru,      0;
    %                          0,           H_eta]
    F = blkdiag(W' * Qy * W + Ru, H_eta);
    F = (F + F') / 2;  % ensure symmetry

    % Build linear term f_z = 2 * [W'QyPi*x0; zeros(H*p,1)]
    f = [2 * W' * Qy * Pi * x0; zeros(H*p,1)];

    % Determine safety limits and reference
    y_max = 55 - y_ss - Dr ;

    % Inequality: W*U + Pi*x0 <= y_max + eta
    % => [  W,  -I ] * [U; eta] <= y_max - Pi*x0
    A_ineq = [W, -eye(H*p)];
    b_ineq = y_max * ones(H*p,1) - Pi * x0;

    % Control bounds: Δu ∈ [ -u_ss, 100-u_ss ]
    ub_U = (100 - u_ss) * ones(H*m,1);
    lb_U = (0 - u_ss)   * ones(H*m,1);

    % Slack variables η >= 0
    lb_eta = zeros(H*p,1);
    ub_eta = inf(H*p,1);

    % Combine bounds
    lb = [lb_U; lb_eta];
    ub = [ub_U; ub_eta];

    % Solve QP via quadprog
    options = optimoptions('quadprog','Display','off');
    [z, ~, exitflag] = quadprog(F, f, A_ineq, b_ineq, [], [], lb, ub, [], options);
    if exitflag ~= 1
        warning('MPC optimization failed (exitflag = %d).', exitflag);
    end

    % Extract the first control increment
    Du_opt = z(1:m);
    u0 = Du_opt(1);
end
