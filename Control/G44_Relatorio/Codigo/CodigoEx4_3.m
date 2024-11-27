%Expeiência com o controlador Proporcional Derivativo.
data_PD = run_experiment(4,0,27.328,0,53.044);
plot_PD = plot_experiment_results_v3(data_PD);

%Expeiência com o controlador Proporcional Integral Derivativo.
data_PID = run_experiment(4,0,23.922,28.001,67.962);
plot_PID = plot_experiment_results_v3(data_PID);