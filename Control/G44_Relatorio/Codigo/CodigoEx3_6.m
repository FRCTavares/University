%Controlador Proporcional.
data_kw1669 = run_experiment(3,0,16.69,0,0);
data_kw17 = run_experiment(3,0,17,0,0);
data_kw19 = run_experiment(3,0,19,0,0);
data_kw25 = run_experiment(3,0,25,0,0);

kw1669_plot=plot_experiment_results_v2(data_kw1669);
kw17_plot=plot_experiment_results_v2(data_kw17);
kw19_plot=plot_experiment_results_v2(data_kw19);
kw25_plot=plot_experiment_results_v2(data_kw25);

%Controlador Integral.
data_ki5888=run_experiment(3,0,0,5.888,0);
data_ki10=run_experiment(3,0,0,10,0);
data_ki15=run_experiment(3,0,0,15,0);
data_ki20=run_experiment(3,0,0,20,0);

ki5888_plot=plot_experiment_results_v2(data_ki5888);
ki10_plot=plot_experiment_results_v2(data_ki10);
ki15_plot=plot_experiment_results_v2(data_ki15);
ki20_plot=plot_experiment_results_v2(data_ki20);

%Controlador Prporcional Integral.
data_kw27328ki54033=run_experiment(3,0,27.328,54.033,0);
data_kw10ki90=run_experiment(3,0,10,90,0);
data_kw30ki20=run_experiment(3,0,30,20,0);
data_kw50ki20=run_experiment(3,0,50,20,0);

kw27328ki54033_plot=plot_experiment_results_v2(data_kw27328ki54033);
kw10ki90_plot=plot_experiment_results_v2(data_kw10ki90);
kw30ki20_plot=plot_experiment_results_v2(data_kw30ki20);
kw50ki20_plot=plot_experiment_results_v2(data_kw50ki20);

%Plot das diferentes repostas sobrepostas.
%Controlador Proporcional.
x = kw1669_plot.t_unit_step;
y = kw1669_plot.y_unit_step;
plot(x,y);
hold on;
x = kw17_plot.t_unit_step;
y = kw17_plot.y_unit_step;
plot(x,y);
hold on;
x = kw19_plot.t_unit_step;
y = kw19_plot.y_unit_step;
plot(x,y);
hold on;
x = kw25_plot.t_unit_step;
y = kw25_plot.y_unit_step;
plot(x,y);
hold on;
xlabel('Tempo (s)');
ylabel('[rad/s]');
title('Resposta ao Degrau com Diferentes Valores de kw');
legend('kw=16.69','kw=17','kw=19','kw=25');

%Controlador Integral.
x = ki5888_plot.t_unit_step;
y = ki5888_plot.y_unit_step;
plot(x,y);
hold on;
x = ki10_plot.t_unit_step;
y = ki10_plot.y_unit_step;
plot(x,y);
hold on;
x = ki15_plot.t_unit_step;
y = ki15_plot.y_unit_step;
plot(x,y);
hold on;
x = ki20_plot.t_unit_step;
y = ki20_plot.y_unit_step;
plot(x,y);
hold on;
xlabel('Tempo (s)');
ylabel('[rad/s]');
title('Resposta ao Degrau com Diferentes Valores de ki');
legend('ki=5.888','ki=10','ki=15','ki=20');

%Controlador Prporcional Integral.
x = kw27328ki54033_plot.t_unit_step;
y = kw27328ki54033_plot.y_unit_step;
plot(x,y);
hold on;
x = kw10ki90_plot.t_unit_step;
y = kw10ki90_plot.y_unit_step;
plot(x,y);
hold on;
x = kw30ki20_plot.t_unit_step;
y = kw30ki20_plot.y_unit_step;
plot(x,y);
hold on;
x = kw50ki20_plot.t_unit_step;
y = kw50ki20_plot.y_unit_step;
plot(x,y);
hold on;
xlabel('Tempo (s)');
ylabel('[rad/s]');
title('Resposta ao Degrau com Diferentes Valores de kw e ki');
legend('kw=27.328 e ki=53.044','kw=10 e ki=90','kw=30 e ki=20','kw=50 e ki=20');

%Plot dos diferentes atuadores sobrepostos.
%Controlador Proporcional.
x = kw1669_plot.time;
y = kw1669_plot.u;
plot(x,y);
hold on;
x = kw17_plot.time;
y = kw17_plot.u;
plot(x,y);
hold on;
x = kw19_plot.time;
y = kw19_plot.u;
plot(x,y);
hold on;
x = kw25_plot.time;
y = kw25_plot.u;
plot(x,y);
hold on;
xlabel('Tempo (s)');
ylabel('[rad/s]');
title('Atuador com Diferentes Valores de kw');
legend('kw=16.69','kw=17','kw=19','kw=25');

%Controlador Integral.
x = ki5888_plot.time;
y = ki5888_plot.u;
plot(x,y);
hold on;
x = ki10_plot.time;
y = ki10_plot.u;
plot(x,y);
hold on;
x = ki15_plot.time;
y = ki15_plot.u;
plot(x,y);
hold on;
x = ki20_plot.time;
y = ki20_plot.u;
plot(x,y);
hold on;
xlabel('Tempo (s)');
ylabel('[rad/s]');
title('Atuador com Diferentes Valores de ki');
legend('ki=5.888','ki=10','ki=15','ki=20');

%Controlador Prporcional Integral.
x = kw27328ki54033_plot.time;
y = kw27328ki54033_plot.u;
plot(x,y);
hold on;
x = kw10ki90_plot.time;
y = kw10ki90_plot.u;
plot(x,y);
hold on;
x = kw30ki20_plot.time;
y = kw30ki20_plot.u;
plot(x,y);
hold on;
x = kw50ki20_plot.time;
y = kw50ki20_plot.u;
plot(x,y);
hold on;
xlabel('Tempo (s)');
ylabel('[rad/s]');
title('Atuador com Diferentes Valores de kw e ki');
legend('kw=27.328 e ki=53.044','kw=10 e ki=90','kw=30 e ki=20','kw=50 e ki=20');