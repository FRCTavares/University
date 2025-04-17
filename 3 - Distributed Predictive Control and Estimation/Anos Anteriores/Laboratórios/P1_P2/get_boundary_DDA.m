% get_boundary_DDA(f, n_points, max_iterations, resolution, tolerance, minimizer, options)
%
% Description: This function calculates the boundary points of a function
% using the Digital Differential Analyzer (DDA) algorithm. It starts from
% the point minimizer and iteratively moves along the boundary in n_points
% directions by performing ray tracing. The algorithm stops when either the
% maximum number of iterations is reached or the distance between the current
% point and the starting point exceeds the specified tolerance.
%
% Authors:
% * Afonso Bispo Certo (96134) - afonso.certo@tecnico.ulisboa.pt
% * Francisco Rodrigues (96210) - francisco.rodrigues.09@tecnico.ulisboa.pt
% * João Marafuz Gaspar (96240) - joao.marafuz.gaspar@tecnico.ulisboa.pt
% * Yandi Jiang (96344) - yandijiang@tecnico.ulisboa.pt
%__________________________________________________________________________

function [points] = get_boundary_DDA(f, n_points, max_iterations, resolution, tolerance, minimizer, options)

    radius = zeros(1, n_points);
    angles = 2 * pi / n_points * (0:n_points - 1); % Compute the angles for each direction

    for jj = 1:n_points
        dx1 = 0; 
        dx2 = 0; 
        ray_x1 = 0; 
        ray_x2 = 0; 
        aux = 0; % Indicates if cos or sin is zero
        
        for ii = 1:max_iterations
            if cos(angles(jj)) > 0 && ray_x1 <= ray_x2 
                dx1 = dx1 + resolution;
            elseif cos(angles(jj)) < 0 && ray_x1 <= ray_x2
                dx1 = dx1 - resolution;
            end

            if sin(angles(jj)) > 0 && ray_x2 <= ray_x1 
                dx2 = dx2 + resolution;
            elseif sin(angles(jj)) < 0 && ray_x2 <= ray_x1 
                dx2 = dx2 - resolution;
            end

            if abs(sin(angles(jj))) < 1e-6 % Check if sin is close to zero
                radius(jj) = abs(dx1);
                aux = 1;
            elseif abs(cos(angles(jj))) < 1e-6 % Check if cos is close to zero
                radius(jj) = abs(dx2); 
                aux = 1; 
            elseif cos(angles(jj)) > 0 && sin(angles(jj)) > 0 % Check if in the first quadrant
                ray_x1 = abs(dx1 / cos(angles(jj)));
                ray_x2 = abs(dx2 / sin(angles(jj)));
            elseif cos(angles(jj)) < 0 && sin(angles(jj)) > 0 % Check if in the second quadrant
                ray_x1 = abs(dx1 / cos(pi - angles(jj)));
                ray_x2 = abs(dx2 / sin(pi - angles(jj)));
            elseif cos(angles(jj)) < 0 && sin(angles(jj)) < 0 % Check if in the third quadrant
                ray_x1 = abs(dx1 / cos(3 * pi / 2 - angles(jj)));
                ray_x2 = abs(dx2 / sin(3 * pi / 2 - angles(jj)));
            elseif cos(angles(jj)) > 0 && sin(angles(jj)) < 0 % Check if in the fourth quadrant
                ray_x1 = abs(dx1 / cos(2 * pi - angles(jj)));
                ray_x2 = abs(dx2 / sin(2 * pi - angles(jj)));
            end

            if ray_x1 <= ray_x2 && aux == 0
                radius(jj) = ray_x1; % Assign ray_x1 as the radius
            elseif ray_x2 < ray_x1 && aux == 0
                radius(jj) = ray_x2; % Assign ray_x2 as the radius
            end

            result = fminunc(f, minimizer + radius(jj) * [cos(angles(jj)) sin(angles(jj))], options); % Optimize the function

            if (norm(result - minimizer) > tolerance) % Check if the optimization result exceeds the tolerance
                break; % Break the inner loop
            end
        end
    end

    points = radius' .* [cos(angles') sin(angles')] + minimizer .* ones(n_points, 1); % Compute the boundary points
    points = [points; points(1, :)]; % Close the boundary by adding the first point to the end

end