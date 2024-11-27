%Parametros de g.
k0= 0.1036;
a=2.22;
%Função de Transferência em Ciclo Aberto sem o controlador.
s=tf('s');
g=(1/s)*k0*a/(s+a);
%Controlador a testar.
k=53.044+27.328*s;

controlSystemDesigner(g,k);
%Plot da Resposta ao Degrau
H=k*g/(1+k*g);
step(H,1.8);