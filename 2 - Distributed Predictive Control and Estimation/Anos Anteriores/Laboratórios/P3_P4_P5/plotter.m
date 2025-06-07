clc
clear 
close all

% Plot for the first experiment for model identification
load('openloop_data_1.mat')

figure(1)

subplot(2, 1, 1)
hold on
plot(t, y, '.')
set(gca,'ColorOrderIndex',1)
plot(t, y, 'LineWidth', 0.1)
xline(1e3, 'r--')
xline(2e3, 'r--')
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Temperature [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('Temperature 1', 'Temperature 2', 'Interpreter', 'latex', 'FontSize', 14, 'Position', [0.656606878553118, 0.621849471367444, 0.23446455001831, 0.090714284351894])
grid on

subplot(2, 1, 2)
hold on
stairs(t, u(1, :), 'LineWidth', 2)
stairs(t, u(2, :), 'LineWidth', 2)
ylim([0 50])
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Heater control [\%]', 'Interpreter', 'latex', 'FontSize', 18)
legend('Heater 1', 'Heater 2', 'Interpreter', 'latex', 'FontSize', 14)
grid on

% Plot for the second experiment for model identification
load('openloop_data_2.mat')

figure(2)

subplot(2, 1, 1)
hold on
plot(t, y, '.')
set(gca,'ColorOrderIndex',1)
plot(t, y, 'LineWidth', 0.1)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Temperature [$^\circ$C]', 'Interpreter', 'latex', 'FontSize', 18)
legend('Temperature 1', 'Temperature 2', 'Interpreter', 'latex', 'FontSize', 14, 'Position', [0.656606878553118, 0.621849471367444, 0.23446455001831, 0.090714284351894])
grid on

subplot(2, 1, 2)
hold on
stairs(t, u(1, :), 'LineWidth', 2)
stairs(t, u(2, :), 'LineWidth', 2)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14)
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Heater control [\%]', 'Interpreter', 'latex', 'FontSize', 18)
legend('Heater 1', 'Heater 2', 'Interpreter', 'latex', 'FontSize', 12, 'Position', [0.139226068769182, 0.364230414292413, 0.15363107408796, 0.079285714739845])
grid on