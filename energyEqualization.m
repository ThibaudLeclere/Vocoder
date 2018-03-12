function [equalizedSignal] = energyEqualization(signal, energyTarget, option, type, fs)

% energyEqualization : Equalizes a signal to a specific energy

% OUTPUT : 
% 
% - equalizedSignal : signal having the desired energy
% 
% INPUTS : 
% 
% - signal : column-vector you want to equalize in terms of energy. You can
%             equalize a multichannel signal by putting a matrix instead of a vector but
%             remind the temporal dimension is considered to be along the lines of the
%             matrix.
% - energyTarget : value of the desired energy you want to reach with your signal
% - type : '-t' computes the energy in the temporal domain
%          '-s' computes the energy in the frequency domain (between 20Hz and
%               fs/2 by default)
% 
% - option : 'mean' makes the eaqualization according to the mean energy of the different channels of your input signal (the level difference between the channels is preserved)
%            'each' applies the equalization to each channel independently
% 
% %
[lines, columns] = size(signal);
equalizedSignal = zeros(lines, columns);

if nargin < 4
    type = '-t';
end
if nargin < 3
    option = 'each';
end
if nargin < 2
    energyTarget = 1;
end


switch option
    case 'each'
        switch type
            case '-t'
                E = energy(signal);         
            case '-s'
                E = effectiveEnergyDFT(signal, 20, fs/2, fs);
        end
    case 'mean'
        switch type
            case '-t'
                E(1,1:columns) = mean(energy(signal));
            case '-s'
                E(1,1:columns) = mean(effectiveEnergyDFT(signal, 20, fs/2, fs));
        end
    otherwise
        disp('Wrong type chosen')
end

for k = 1:columns
    equalizedSignal(:,k) = (sqrt(energyTarget(k)./E(k))) * signal(:,k);

end

