function u0 = mpc_solve(x0, H, R, A, B, C, u_ss)
% mpc_solve – MPC that enforces ONLY the actuator bound 0 ≤ u ≤ 100 %.
%
% Inputs
%   x0    – current state estimate
%   H     – prediction horizon
%   R     – control-weight scalar
%   A,B,C – incremental plant model
%   u_ss  – steady-state input (so Δu = u – u_ss)
%
% Output
%   u0    – absolute control to send to the plant at k=0  (u_ss + Δu_opt)
%
% -------------------------------------------------------------------------

n = size(A,1);               % number of states
m = size(B,2);               % number of inputs
p = size(C,1);               % number of outputs

%% Prediction matrices ----------------------------------------------------
W  = zeros(H*p, H*m);
Pi = zeros(H*p, n);
for i = 1:H
    Pi((i-1)*p+1:i*p, :) = C*A^i;
    for j = 1:i
        W((i-1)*p+1:i*p,(j-1)*m+1:j*m) = C*A^(i-j)*B;
    end
end

%% Quadratic cost ---------------------------------------------------------
Q     = eye(H*p);            % output-weight (tracking Δy→0)
Rbar  = R*eye(H*m);          % input-increment weight
F     = 2*(W'*Q*W + Rbar);
F     = (F+F')/2;            % numerical symmetry
f     = 2*W'*Q*Pi*x0;        % linear term

%% Pure actuator bounds (no output constraint) ----------------------------
% Remember: u = u_ss + Δu ⇒  Δu ∈ [−u_ss , 100−u_ss]
lb =  (0   - u_ss)*ones(H*m,1);
ub = (100 - u_ss)*ones(H*m,1);

%% Solve QP ---------------------------------------------------------------
opts      = optimoptions('quadprog','Display','off');
[U,~,flag] = quadprog(F, f, [], [], [], [], lb, ub, [], opts);

if flag ~= 1
    warning('MPC optimization did not converge (exitflag = %d).', flag);
end

%% Return first move as ABSOLUTE control ----------------------------------
Du_opt = U(1:m);             % first Δu
u0     = u_ss + Du_opt;      % absolute heater command (0–100 %)
end
