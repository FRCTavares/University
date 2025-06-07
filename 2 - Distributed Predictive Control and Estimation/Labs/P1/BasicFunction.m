function f = BasicFunction(x)
% BasicFunction - Rosenbrock function for optimization testing
%
%   This function computes the Rosenbrock function value at a given 2D point.
%
%   Input:
%       x - a 2x1 vector [x1; x2]
%
%   Output:
%       f - scalar value of the function evaluated at x
%
%   Mathematical definition:
%       f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2
%
%   This function has a global minimum at [x1; x2] = [1; 1],
%   where f(x) = 0.
%
%   Used in optimization tasks (P1) for the course:
%   Estimação e Controlo Preditivo Distribuído (ECPD)
%
%   Instituto Superior Técnico, MEEC, 2024/2025
%   Autor original: J. Miranda Lemos
%   Comentado e adaptado por: Francisco Tavares

%--------------------------------------------------------------------------

% Extract components
x1 = x(1);
x2 = x(2);

% Compute Rosenbrock function
f = 100 * (x2 - x1^2)^2 + (1 - x1)^2;

end
%--------------------------------------------------------------------------
% End of file
