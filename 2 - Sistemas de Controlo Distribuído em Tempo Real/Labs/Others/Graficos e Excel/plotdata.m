% single_plot_lux_duty_ignore_first_2s.m
%
% Reads a 4-column CSV (Time, MeasuredLux, DutyCycle, ReferenceLux),
% discards the first 2 seconds of data, and plots them in one figure.

% ------------------------------
% STEP 1: Read the CSV
% ------------------------------
data = readmatrix('output.csv');  % Adjust filename/path if needed

% ------------------------------
% STEP 2: Assign columns to variables
% ------------------------------
Time           = data(:,1);
MeasuredLux    = data(:,2);
DutyCycle      = data(:,3);
ReferenceLux   = data(:,4);  % 'SetpointLux' renamed for clarity

% ------------------------------
% STEP 3: Shift time to start at zero (optional)
% ------------------------------
TimeZero = Time - Time(1);
Time = Time/1000;

% ------------------------------
% STEP 4: Filter out first 2 seconds
% ------------------------------
idx           = TimeZero >= 2;       % indices where time >= 2 s
TimeFiltered  = TimeZero(idx) - 2;   % re-zero so the new start is 0
MeasFiltered  = MeasuredLux(idx);
DutyFiltered  = DutyCycle(idx);
RefFiltered   = ReferenceLux(idx);

% ------------------------------
% STEP 5: Create a single figure with 2 y-axes
% ------------------------------
figure('Name','Lux & DutyCycle','Position',[100 100 800 400]);

% --- Left y-axis: Reference Lux & Measured Lux ---
yyaxis left;
p1 = plot(TimeFiltered, RefFiltered, ...
    'LineWidth', 1.7, 'Color','b', 'DisplayName','Reference Lux');
hold on;
p2 = plot(TimeFiltered, MeasFiltered, ...
    'LineWidth', 1.7, 'Color','r', 'DisplayName','Measured Lux');
xlabel('Time [s]');
ylabel('Lux [lx]');
grid on;

% --- Right y-axis: Duty Cycle ---
yyaxis right;
p3 = plot(TimeFiltered, DutyFiltered, ...
    'LineWidth', 1.7, 'Color','k', 'DisplayName','Duty Cycle');
ylabel('Duty Cycle [0..1]');
title('Reference Lux, Measured Lux, and Duty Cycle vs. Time (Ignoring First 2 s)');

% Single legend for all three lines
legend([p1 p2 p3],'Location','NorthWest');
