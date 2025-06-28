% Load the existing figure
figura = openfig('task5_final.fig');

% Find axes
axesHandles = findall(figura, 'Type', 'axes');
topAx = axesHandles(end);      % Top plot (temperature)
bottomAx = axesHandles(end-1); % Bottom plot (control input)

% --- Clear previous legends ---
legend(topAx, 'off');
legend(bottomAx, 'off');

% --- Add magenta safety line ---
yMaxHandle = yline(topAx, 55, 'm', 'LineWidth', 1.5);
set(yMaxHandle, 'DisplayName', 'y ≤ 55 °C');

% --- Identify 'y (measured)' as dotted blue line ---
measHandle = findobj(topAx, 'Type', 'Line', ...
    'LineStyle', 'none', ...
    'Marker', '.', ...
    'Color', [0 0.447 0.741]); % MATLAB default blue
set(measHandle, 'DisplayName', 'y (measured)');

% --- Identify 'r (reference)' as stair now black ---
stairsObjs = findobj(topAx, 'Type', 'Stair', 'LineStyle', '--');
refHandle = stairsObjs(1); % assume first stair is the reference
set(refHandle, 'Color', [0 0 0], 'DisplayName', 'r (reference)');

% --- Add top legend with confirmed handles ---
legend(topAx, [measHandle, refHandle, yMaxHandle], ...
       'Location', 'northwest');

% --- Control input 'u' ---
lineBottom = findobj(bottomAx, 'Type', 'Line');
if ~isempty(lineBottom)
    set(lineBottom(1), 'DisplayName', 'u (control input)');
    legend(bottomAx, lineBottom(1), 'Location', 'northeast');
end

% --- Add overall title ---
sgtitle(figura, 'MPC and Kalman Filter – Real System Response');

% Save updated figure
savefig(figura, 'task5_final.fig');
exportgraphics(figura, 'task5_final.png', 'Resolution', 300);
