%Diagrama de Bode Ideal para a Função a estudar.
K = 0.1036;
a = 2.22;
numerator = K * a;
denominator = [1, a];
sys = tf(numerator, denominator);
bode(sys);
hold on;

%Array com as frequências calculdas em 2.5.
x = [0.222,0.444,0.666,1.11,2.22,3.33,4.44,5.55,6.66,7.77,8.88,9.99,11.1];
%Array com os ganhos calculados no excel por observação dos gráficos.
y = [0.083307, 0.089381111, 0.083785556, 0.079056667, 0.062927778, 0.049782222, 0.041533333, 0.033853333, 0.029458889, 0.026502222,	0.021314444, 0.019868889, 0.017545556];

%Mudança para dB.
y = 20*log10(y);

%Plot dos pontos experimentais de Magnitude em Função da frequência
%sobrepostos ao diagrama ideal.
plot(x,y,'o');
hold on;
plot(x,y,'x');
