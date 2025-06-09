% TCLab_MPC_Tuning.m
% Closed‐loop unconstrained MPC tuning: horizon H & weight R sweeps
% Figures 10, 11, 12
%__________________________________________________________________________

%% Initialization
clear, close all, clc

% Load identified model
load('singleheater_model.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n     = size(A,1);
e_std = sqrt(e_var);

% Build TCLab‐style interface
x_ss = [eye(n)-A; C] \ [B*u_ss; y_ss];
c1   = (eye(n)-A)*x_ss - B*u_ss;
c2   = y_ss - C*x_ss;
h1   = @(x,u) A*x + B*u + Ke*e_std*randn + c1;  % plant step
T1C  = @(x) C*x + e_std*randn + c2;             % noisy measurement

% Compute Δx(0) at equilibrium
Dx0Dy0 = [eye(n)-A, zeros(n,1); C, -1] \ [-B*u_ss; 0];
Dx0    = Dx0Dy0(1:n);

% Simulation parameters
T     = 500;                  % duration [s]
N     = T/Ts;                 % number of steps
t_vec = (0:N-1)*Ts;           % time vector

% MPC tuning parameters
H_values = [10,20,50,70,100,200];
R_values = [0.02,0.05,0.1,0.5,1];
H_fixed  = 50;
R_fixed  = 0.05;


%% 1) Horizon sweep (R = 0.05) → Fig. 10
results = repmat(struct('H',[],'y_cl',[],'u_cl',[],'avg_solve',[]), ...
                 numel(H_values),1);

for idxH = 1:numel(H_values)
  H = H_values(idxH);

  % preallocate
  x          = nan(n,N+1);
  Dx         = nan(n,N);
  y_cl       = nan(1,N);
  u_cl       = nan(1,N);
  solve_time = nan(1,N);

  % initial state
  x(:,1) = Dx0 + x_ss;

  for k = 1:N
    % measure & form Δx
    y_cl(k) = T1C(x(:,k));
    Dx(:,k) = x(:,k) - x_ss;

    % solve MPC
    tic;
      Du     = mpc_solveP4_2(Dx(:,k), H, R_fixed, A, B, C, u_ss);
    solve_time(k) = toc;

    % apply
    u_cl(k)  = u_ss + Du;
    x(:,k+1) = h1(x(:,k), u_cl(k));
  end

  % store
  results(idxH).H         = H;
  results(idxH).y_cl      = y_cl;
  results(idxH).u_cl      = u_cl;
  results(idxH).avg_solve = mean(solve_time)/Ts*100;
end

% Figure 10: Avg solve time vs Horizon
figure(10);
plot([results.H],[results.avg_solve],'o-','LineWidth',1.5);
grid on;
xlabel('Horizon H');
ylabel('Avg solve time (% of T_s)');
title('Solver time vs Horizon');


%% 2) Closed-loop output & input for H={10,20,50,70} → Fig. 11
sel    = [10,20,50,70];
colors = lines(numel(sel));

figure(11);

% 11a) output y
subplot(2,1,1), hold on, grid on
for i = 1:numel(sel)
  idx = find([results.H]==sel(i),1);
  plot(t_vec, results(idx).y_cl, '-', ...
       'Color',colors(i,:), 'LineWidth',1.5, ...
       'DisplayName',['H=' num2str(sel(i))]);
end
% steady-state output (shown in legend)
yline(y_ss, 'k--', '$\bar y$', ...
      'Interpreter','latex', ...
      'LabelHorizontalAlignment','left', ...
      'DisplayName','$\bar y$');
xlabel('Time [s]'); ylabel('y [\circC]');
title('Closed‐loop Output for Various H');
legend('Interpreter','latex','Location','best');

% 11b) input u
subplot(2,1,2), hold on, grid on
for i = 1:numel(sel)
  idx = find([results.H]==sel(i),1);
  stairs(t_vec, results(idx).u_cl, '-', ...
         'Color',colors(i,:), 'LineWidth',1.5, ...
         'DisplayName',['H=' num2str(sel(i))]);
end
% steady-state input (shown in legend)
yline(u_ss, 'k--', '$\bar u$',  'Interpreter','latex', ...
      'LabelHorizontalAlignment','left', ...
      'DisplayName','$\bar u$');
% hide the 0% & 100% bounds from legend
yline(0,   'r--',    'Interpreter','latex', ...
      'LabelHorizontalAlignment','right','HandleVisibility','off');
yline(100, 'r--',   'Interpreter','latex', ...
      'LabelHorizontalAlignment','right','HandleVisibility','off');
xlabel('Time [s]'); ylabel('u [\%]');
title('Control Input for Various H');
legend('Interpreter','latex','Location','best');


%% 3) Closed-loop output & input for R = [0.02 0.05 0.1 0.5 1] → Fig. 12
% first build R_results
R_results = repmat(struct('R',[],'y_cl',[],'u_cl',[]), numel(R_values),1);

for iR = 1:numel(R_values)
  R = R_values(iR);

  % preallocate
  x    = nan(n,N+1);
  Dx   = nan(n,N);
  y_cl = nan(1,N);
  u_cl = nan(1,N);

  x(:,1) = Dx0 + x_ss;
  for k = 1:N
    y_cl(k)   = T1C(x(:,k));
    Dx(:,k)   = x(:,k) - x_ss;
    Du        = mpc_solveP4_2(Dx(:,k), H_fixed, R, A, B, C, u_ss);
    u_cl(k)   = u_ss + Du;
    x(:,k+1)  = h1(x(:,k), u_cl(k));
  end

  R_results(iR).R    = R;
  R_results(iR).y_cl = y_cl;
  R_results(iR).u_cl = u_cl;
end

% define cols for the R sweep
cols = lines(numel(R_values));

% now plot Figure 12
figure(12);

% 12a) output y
subplot(2,1,1), hold on, grid on
for i = 1:numel(R_values)
  plot(t_vec, R_results(i).y_cl, '-', ...
       'Color',cols(i,:), 'LineWidth',1.5, ...
       'DisplayName',['R=' num2str(R_results(i).R)]);
end
% steady-state output (shown in legend)
yline(y_ss, 'k--', '$\bar y$', ...
      'Interpreter','latex', ...
      'LabelHorizontalAlignment','left',...
      'DisplayName','$\bar y$');
xlabel('Time [s]'); ylabel('y [\circC]');
title('Closed‐loop Output for Various R');
legend('Interpreter','latex','Location','best');

% 12b) input u
subplot(2,1,2), hold on, grid on
for i = 1:numel(R_values)
  stairs(t_vec, R_results(i).u_cl, '-', ...
         'Color',cols(i,:), 'LineWidth',1.5, ...
         'DisplayName',['R=' num2str(R_results(i).R)]);
end
% steady-state input (shown in legend)
yline(u_ss, 'k--', '$\bar u$', ...
      'Interpreter','latex','LabelHorizontalAlignment','left', ...
      'DisplayName','$\bar u$');
% hide the 0% & 100% bounds from legend
yline(0,   'r--',   'Interpreter','latex', ...
      'LabelHorizontalAlignment','right','HandleVisibility','off');
yline(100, 'r--',  'Interpreter','latex', ...
      'LabelHorizontalAlignment','right','HandleVisibility','off');
xlabel('Time [s]'); ylabel('u [\%]');
title('Control Input for Various R');
legend('Interpreter','latex','Location','best');


%% Local MPC solver (dense, unconstrained)
function u0 = mpc_solveP4_2(x0,H,R,A,B,C,u_ss)
  p  = size(C,1);
  m  = size(B,2);
  W  = zeros(H*p,H*m);
  Pi = zeros(H*p,size(A,1));
  for i = 1:H
    Pi((i-1)*p+1:i*p,:) = C*(A^i);
    for j = 1:i
      W((i-1)*p+1:i*p,(j-1)*m+1:j*m) = C*(A^(i-j)*B);
    end
  end
  F = 2*(W'*W + R*eye(H*m));  F=(F+F')/2;
  f = 2*(W'*Pi*x0);
  opts = optimset('Display','off');
  U    = quadprog(F,f,[],[],[],[],[],[],[],opts);
  if isempty(U)
    warning('mpc_solveP4_2: no solution, returning zero.');
    u0 = 0;
  else
    u0 = U(1);
  end
end
