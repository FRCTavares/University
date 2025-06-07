%% DC‐AC Microgrid Parameter & Controller Computation
% This script computes:
%   1) Three‐phase parallel RLC load parameters (QLac, QCac)
%   2) Filter inductance Lfac for 10% current ripple
%   3) DC‐bus capacitance Cdc for 2% voltage ripple
%   4) PI gains for current control (Kpi, Kii)
%   5) PI gains for DC‐voltage control (Kpvdc, Kivdc)
%
% All formulas are taken from the lab image instructions.

%% 1) Clear the workspace and set up basic parameters
clear variables
clc

% 1.1 Three‐Phase AC Source (phase‐to‐neutral)
Vac   = 230;           % [V]    Phase‐to‐neutral RMS voltage
fac   = 50;            % [Hz]   AC fundamental frequency
Scc   = 20e6;          % [VA]   Three‐phase short‐circuit MVA
X_R   = 7;             % [–]    X/R ratio (unused later, but defined here)

% 1.2 Inverter & DC‐Microgrid Specs
Ron        = 0.04;                 % [Ω]   MOSFET on‐resistance
Rs         = 1e6;                  % [Ω]   Snubber resistance (≈ open circuit)
fpwm_inv   = (4*50-1) * fac;       % [Hz]  PWM carrier freq = (4·50 – 1)·50 = 49 950 Hz
Udc        = 700;                  % [V]   Nominal DC‐bus voltage
Pdc        = 100e3;                % [W]   DC‐side delivered power (100 kW)
Pdcmin     = Pdc / 10;             % [W]   Minimum DC power (10 kW)

% Resistive locks for DC‐load
Rdcmax     = Udc^2 / Pdcmin;       % [Ω]   Resistive path for Pdcmin (10 kW)
Rdc        = Udc^2 / (Pdc - Pdcmin);% [Ω]   Resistive path for remaining 90 kW

%% 2) Three‐Phase Parallel RLC Load (per‐phase)
Pac = 200e3 / 3;     % [W]    Per‐phase active power (200 kW total ÷ 3)
Fpi = 0.90;          % [–]    Desired RLC load PF before compensation
QLac = Pac * tan( acos(Fpi) );  
FpL = cos( atan( QLac / Pac ) );  
Fp_target = 0.94; 
QCac = QLac - ( Pac * tan( acos(Fp_target) ) );  
Fpv = cos( atan( (QLac - QCac) / Pac ) );  

%% 3) DC‐Bus Capacitance Cdc for 2% voltage ripple
DeltaUdc = 0.02;  % [pu]  ±1% around Udc → total ΔUdc = 0.02
Cdc = 2 * Pdc / ( fac * ( ((1 + DeltaUdc/2)*Udc)^2 - ((1 - DeltaUdc/2)*Udc)^2 ) );

%% 4) Inverter Filter Inductor Lfac for 10% current ripple
I_phase_nom = (Pdc/3) / Vac;  
DeltaIac    = 0.10 * I_phase_nom;  
Lfac = Udc / ( 6 * fpwm_inv * DeltaIac ); 


%% 5) Inverter‐to‐Grid Phasor Computations
wac   = 2 * pi * fac;                       
m_inv = Vac / ( Udc / (2*sqrt(2)) );  
XLf = wac * Lfac;  
XLfs = XLf + (Vac^2)/(Scc/3);  
Vinv = m_inv * (Udc / (2*sqrt(2)));  
deltafi = - asin( ( (Pdc/3) * XLfs ) / ( Vac * Vinv ) );  
Pacv = 3 * Vac * Vinv * sin(deltafi) / XLfs;  
deltafi_deg = deltafi * 180/pi;


%% 6) Current‐Controller PI Gains (d–q axis)
% 6.1 Define intermediate timing & resistance constants
alfai  = 1;                    % [–]  Damping coefficient for current loop
Tdi    = 1 / (2 * fpwm_inv);   % [s]  Delay constant ≈ half PWM period
Reqi   = (Vac^2) / (Pdc/3);    % [Ω]  Equivalent per‐phase resistance for Pdc/3
Tzi    = Lfac / Reqi;          % [s]  Zero time‐constant of the current‐loop TF
qsii   = 0.707;                % [–]  Damping factor (only used if you wanted a 2nd‐order design—here for completeness)

Tpi = (2 * alfai * Tdi) / Reqi;  
Kpi = Tzi / Tpi;  
Kii = 1 / Tpi;  


%% 7) DC‐Voltage Controller PI Gains
% 7.1 Intermediate constants for the DC loop
Vdcref = Udc;                 % [V]   DC‐voltage reference = 700 V
alfav  = 1;                   % [–]   Damping coefficient for voltage loop
Tdv    = 1 / (2 * fac);       % [s]   Delay constant ≈ half fundamental period (50 Hz)
Gi     = (Vac * sqrt(2)) / Udc;  
Kpvdc = -2.15 * Cdc * alfav / ( (1.75^2) * alfav * Gi * Tdv );  
Kivdc = -Cdc * alfav   / ( (1.75^3) * alfav * Gi * Tdv^2 );  

%% 8. Summary of All Selected Values
fprintf('===== FINAL PARAMETER SUMMARY =====\n');
fprintf('QLac    = %.3f kVAr/pi-phase\n', QLac/1e3);
fprintf('QCac    = %.3f kVAr/pi-phase\n', QCac/1e3);
fprintf('Lfac    = %.3f mH\n', Lfac * 1e3);
fprintf('Cdc     = %.4f F\n', Cdc);
fprintf('Kpi     = %.3f   | Kii = %.1f\n', Kpi, Kii);
fprintf('Kpvdc   = %.4f | Kivdc = %.4f\n\n', Kpvdc, Kivdc);
