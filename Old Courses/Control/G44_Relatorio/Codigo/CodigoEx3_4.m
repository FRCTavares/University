data_P = run_experiment(3,0,16.69,0,0);
data_I = run_experiment(3,0,0,5.888,0);
data_PI = run_experiment(3,0,27.328,53.044,0);

plot_P = plot_experimemt_results_v2(data_P);
plot_I = plot_experimemt_results_v2(data_I);
plot_PI = plot_experimemt_results_v2(data_PI);
