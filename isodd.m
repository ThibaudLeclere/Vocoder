function [out] = isodd(number)
% Array elements that are odd
%
%   Description:
%        out = isodd(number) returns an array the same size as number
%        containing logical 1 (true) where the elements of number are even
%        and logical 0 (false) when they are not.
%   Input : 
%        number : numeric scalar, vector or array to evaluate


if isnumeric(number)
    out = logical(mod(number,2));
    
else
    error('input argument should be numeric')
end