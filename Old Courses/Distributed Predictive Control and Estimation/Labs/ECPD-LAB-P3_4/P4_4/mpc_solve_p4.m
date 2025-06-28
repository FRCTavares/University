function u0 = mpc_solve_p4(x0, H, R, A, B, C,u_ss,y_ss,Dr)
    % mpc_solve - calcula a primeira ação de controlo do MPC com restrições
    % usando formulação densa e quadprog

    n = size(A,1);  % número de estados
    m = size(B,2);  % número de entradas
    p = size(C,1);  % número de saídas

    % Construção das matrizes W e Pi
    W = zeros(H*p, H*m);
    Pi = zeros(H*p, n);
    for i = 1:H
        Pi((i-1)*p+1:i*p, :) = C * A^i;
        for j = 1:i
            W((i-1)*p+1:i*p, (j-1)*m+1:j*m) = C * A^(i-j) * B;
        end
    end

    % Matriz de custo
    Q = eye(H*p);
    Rbar = R * eye(H*m);
    F = 2 * (W' * Q * W + Rbar);
    F = (F + F') / 2;  % força simetria numérica
    f = 2 * (W' * Q * Pi * x0);

    % Restrições: garantir u ∈ [0, 100] ⇒ Δu ∈ [-u_ss, 100 - u_ss]
    lb = (0-u_ss) * ones(H*m, 1);
    ub = (100 - u_ss) * ones(H*m, 1);

    % Output constraint: y <= 55
    
    y_max = 55-y_ss-Dr;
    A_ineq = eye(H)*W;
    b_ineq = (y_max  - eye(H)* Pi * x0);
    
    % Resolver com quadprog
    options = optimoptions('quadprog','Display','off');
    %U = quadprog(F, f, A_ineq, b_ineq, [], [], lb, ub, [], options);
    [U, ~, exitflag] = quadprog(F, f, A_ineq, b_ineq, [], [], lb, ub, [], options);
    
    if exitflag ~= 1
        warning('MPC optimization failed (exitflag = %d).', exitflag);
    end
    % Devolver apenas a primeira ação de controlo (Δu(0))
    u0 = U(1);
end
