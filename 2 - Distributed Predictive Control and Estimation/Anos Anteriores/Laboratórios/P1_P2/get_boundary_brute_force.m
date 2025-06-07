% get_boundary_brute_force
%
% Description: Calculates the attraction basin boundary using a brute-force 
% approach.
%
% Authors:
% * Afonso Bispo Certo (96134) - afonso.certo@tecnico.ulisboa.pt
% * Francisco Rodrigues (96210) - francisco.rodrigues.09@tecnico.ulisboa.pt
% * João Marafuz Gaspar (96240) - joao.marafuz.gaspar@tecnico.ulisboa.pt
% * Yandi Jiang (96344) - yandijiang@tecnico.ulisboa.pt
%__________________________________________________________________________

function [attraction_grid] = get_boundary_brute_force(f, grid_size, options, desired_minima, x1_min, x1_max, x2_min, x2_max, tolerance)

    x0_1 = linspace(x1_min, x1_max, grid_size);
    x0_2 = linspace(x2_min, x2_max, grid_size);
    attraction_grid = zeros(grid_size);
    
    % Calculate minima for each place in the grid
    for ii = 1:grid_size
        for jj = 1:grid_size
            calc_minima = fminunc(f, [x0_1(ii), x0_2(jj)], options);
            
            % If it converges to the correct minima, mark in the grid
            if norm(calc_minima - desired_minima) < tolerance
                attraction_grid(ii, jj) = 1;
            end
        end
    end
end