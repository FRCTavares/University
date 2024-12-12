%Parametros de g.
k0= 0.1036;
a=2.22;

%Função de Transferência em Ciclo Aberto sem o controlador.
s=tf('s');
g=(1/s)*k0*a/(s+a);

%Controlador a testar.
k2=27;
k=k2*(s-2.7639)*(s-7.2361)/s;

controlSystemDesigner(g,k);

%Plot da Resposta ao Degrau;
k1=67.9624;
k2=23.922;
k3=28.001;
k=k1+k2*s+k3/s;

H=k*g/(1+k*g);
step(H,6);