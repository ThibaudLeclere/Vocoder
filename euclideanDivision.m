function [quotient, remainder] = euclideanDivision(dividend,divisor)


% euclideanDivision  Performs Euclidean division
%   euclideanDivision(a,b) executes the division of a by b and returns the
%   quotient and remainder.
%
%   See also REM, MOD

%dividend = quotient*divisor + remainder
quotient = floor(dividend/divisor);
remainder = rem(dividend,divisor);