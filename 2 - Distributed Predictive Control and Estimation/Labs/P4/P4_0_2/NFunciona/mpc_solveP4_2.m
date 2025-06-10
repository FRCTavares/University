function u0 = mpc_solveP4_2(x0, H, R, A, B, C, u_ss)
% mpc_solveP4_2  First‐step unconstrained MPC (dense QP)
%
%   u0 = mpc_solveP4_1(x0, H, R, A, B, C, u_ss)
%
% Solves the finite‐horizon, unconstrained MPC using quadprog in the
% dense formulation.  Returns the first control increment u0 = Δu(0).
%
% Inputs:
%   x0   – current incremental state Δx(k)
%   H    – prediction horizon
%   R    – scalar input‐weight
%   A,B,C – plant matrices (Δx(k+1)=AΔx+BΔu, Δy=CΔx)
%   u_ss – steady‐state (feed‐forward) inputs
%
% Output:
%   u0   – first control increment Δu(0)

  % Dimensions
  [n, ~] = size(A);
  p = size(C,1);
  m = size(B,2);

  % Build dense prediction matrices W and Pi
  W  = zeros(H*p, H*m);
  Pi = zeros(H*p, n);
  for i = 1:H
    Pi((i-1)*p+1:i*p, :) = C * (A^i);
    for j = 1:i
      W((i-1)*p+1:i*p, (j-1)*m+1:j*m) = C * (A^(i-j) * B);
    end
  end

  % Cost matrices (output‐weight = I, input‐weight = R)
  F = 2*(W'*W + R*eye(H*m));
  F = (F + F')/2;                % ensure symmetry
  f = 2*(W'*Pi*x0);

  % Solve unconstrained QP: min ½ U'FU + f'U
  opts = optimoptions('quadprog','Display','off');
  U = quadprog(F, f, [], [], [], [], [], [], [], opts);

  % Warn if solver failed
  if isempty(U)
    warning('mpc_solveP4_1: quadprog returned empty solution.');
    u0 = 0;
  else
    u0 = U(1);
  end
end
