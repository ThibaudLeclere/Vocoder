% function pos = CochleaFreq2Pos(cf,k)
%
% This function relates frequency to position on basilar membrane based on
%   data given in Greenwood (1990).
%
% Input parameter:
%   cf is the frequency corresponding to pos.
%
% Output parameter:
%   pos is the position relative to the length of the basilar membrane in mm.
%
% Optional parameter:
%   k is an integration constant. Typical values are between 0.8-0.9. [0.88]
%
% Reference: Greenwood D. "A cochlear frequency-position for several
%   species - 29 years later", J. Acoust. Soc Am. 1990(87) pp2592-2605.
%
function pos = CochleaFreq2Pos(cf,k)

if ~exist('k','var')
    k = 0.88; % gives low frequency limit of 20 Hz
end

bmLength = 35; % length of basilar membrane
a = 2.1;
A = 165.4; % value for man
pos = log10((cf/A)+k)*bmLength/a;
