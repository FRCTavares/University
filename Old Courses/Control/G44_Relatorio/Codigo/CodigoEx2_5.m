data_1 = run_experiment(0,0.222,0,0,0);
data_2 = run_experiment(0,0.444,0,0,0);
data_3 = run_experiment(0,0.666,0,0,0);
data_4 = run_experiment(0,1.11,0,0,0);
data_5 = run_experiment(0,2.22,0,0,0);
data_6 = run_experiment(0,3.33,0,0,0);
data_7 = run_experiment(0,4.44,0,0,0);
data_8 = run_experiment(0,5.55,0,0,0);
data_9 = run_experiment(0,6.66,0,0,0);
data_10 = run_experiment(0,7.77,0,0,0);
data_11 = run_experiment(0,8.88,0,0,0);
data_12 = run_experiment(0,9.99,0,0,0);
data_13 = run_experiment(0,11.1,0,0,0);

plot_data1 = plot_experiment_results(data_1);
plot_data2 = plot_experiment_results(data_2);
plot_data3 = plot_experiment_results(data_3);
plot_data4 = plot_experiment_results(data_4);
plot_data5 = plot_experiment_results(data_5);
plot_data6 = plot_experiment_results(data_6);
plot_data7 = plot_experiment_results(data_7);
plot_data8 = plot_experiment_results(data_8);
plot_data9 = plot_experiment_results(data_9);
plot_data10 = plot_experiment_results(data_10);
plot_data11 = plot_experiment_results(data_11);
plot_data12 = plot_experiment_results(data_12);
plot_data13 = plot_experiment_results(data_13);

%Array com os ganhos calculados no excel por observação dos gráficos.
x = [0.083307, 0.089381111, 0.083785556, 0.079056667, 0.062927778, 0.049782222, 0.041533333, 0.033853333, 0.029458889, 0.026502222,	0.021314444, 0.019868889, 0.017545556];

%Mudança para dB.
x = 20*log10(x);