% get_RH_gain(A, B, C, R, H)
%
% Description: Computes the gain for the Reciding Horizon (RH) optimal 
% control problem, given the system parameters, A, B and C, the cost
% function parameter R and que horizon H
%
% Authors: 
% * Afonso Bispo Certo  (96134) - afonso.certo@tecnico.ulisboa.pt
% * Francisco Rodrigues (96210) - francisco.rodrigues.09@tecnico.ulisboa.pt
% * João Marafuz Gaspar (96240) - joao.marafuz.gaspar@tecnico.ulisboa.pt
% * Yandi Jiang         (96344) - yandijiang@tecnico.ulisboa.pt
%__________________________________________________________________________

function K_RH = get_RH_gain(A, B, C, R, H)

    W = zeros(H, H);
    P = zeros(H, 1);
    I = eye(H);
    e1 = zeros(1, H);
    e1(1, 1) = 1;
    
    for ii = 1:H
        P(ii, 1) = C * A^ii;
        for jj = 1:H
            if jj > ii
                break
            end 
            W(ii,jj) = C * A^(ii - jj) * B;
        end
    end
    
    M = W' * W + R * I;
    K_RH = e1 * (M \ W' * P);

end