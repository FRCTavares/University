% Makes the LED in the TCLab board blink every second

clear all

a=arduino('COM6','Leonardo');

disp('Blinking test started.');

Led=1;
N=10;
for k=1:N
    if Led==0
        Led=1;
        disp('LED on');
    else
        Led=0;
        disp('LED off');
    end
    writeDigitalPin(a,'D9',Led)
    pause(1);
end

% Turns the LED off at the end
Led=0;
writeDigitalPin(a,'D9',Led)

disp('Blinking test finished.')

%--------------------------------------------------------------------------
% End of File

