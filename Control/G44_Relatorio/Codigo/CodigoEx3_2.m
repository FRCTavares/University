%Parametros da Função G(s).
k0=0.1036;
a=2.22;

%Criação da Função G(s).
s=tf('s');
G=k0*a/(s+a);

%Definir um valor Arbitrário de ki.
ki=1;
%Criação da Função do Controlador Integral.
k=ki/s;
%Abrir Control System Designer.
controlSystemDesigner(g,k);