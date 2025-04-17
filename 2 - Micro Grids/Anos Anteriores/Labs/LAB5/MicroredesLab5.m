%DC_AC_Microgrid
clear
%pack
clc

%Three-Phase Power Source (phase to phase)
Vac=230;
fac=50;
Scc=20*1e6;
X_R=7;

%Three-Phase Parallel RLC load
Pac=200e3/3;
Fpi=0.9;
QLac=Pac*tan(acos(Fpi));
FpL=cos (atan (QLac/Pac)); 
Fp=0.94;
QCac=QLac- (Pac*tan (acos (Fp)));
Fpv = cos (atan ((QLac-QCac)/Pac));

%Universal Bridge with MOSFET/Diodes Cs=inf & Filtering inductors Lf
Ron=0.04;
Rs=1e6;
fpwm_inv= (4*50-1)*fac;
Udc=700;
Pdc=100e3;
Pdcmin=Pdc/10;
Rdcmax=Udc^2/Pdcmin;
Rdc=Udc^2/ (Pdc-Pdcmin);
DeltaUdc=0.02;
Cdc=2*Pdc/ (fac* (((1+DeltaUdc/2) *Udc)^2-((1-DeltaUdc/2) *Udc)^2));
Deltalac=0.1* ((Pdc/3)/Vac);
Lfac=Udc/ (6*fpwm_inv*Deltalac);

%Calculations
wac=2*pi*fac;
m_inv=Vac/ (Udc/(2*sqrt(2)));
XLf=wac*Lfac;
XLfs=XLf+Vac^2/(Scc/3);
Vinv=m_inv* (Udc/(2*sqrt (2)));
deltafi=-asin ((Pdc/3) *XLfs/ (Vac*Vinv));
Pacv=3*Vac*Vinv*sin (deltafi)/XLfs;

%Current Controller
alfai=1;
Tdi=1/(2*fpwm_inv);
Reqi=Vac^2/(Pdc/3);
Tzi=Lfac/Reqi;
qsii=0.707;
Tpi=2*alfai*Tdi/Reqi;
Kpi=Tzi/Tpi;
Kii=1/Tpi;

% DC voltage Controller 
Vdcref=Udc;
alfav=1;
Tdv=1/(2*fac);
Gi=Vac* sqrt (2) /Udc;
Kpvdc=-2.15*Cdc*alfai/(1.75^2*alfav*Gi*Tdv);
Kivdc=-Cdc*alfai/ (1.75^3*alfav*Gi*Tdv^2);