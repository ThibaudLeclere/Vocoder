function [corners,CF,CFgeo,CFari,BW]=CalcGreenwoodCorners(lower,upper,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lower = lower frequency boundary in Hz (usually around 300)
% upper = lower frequency boundary in Hz (usually around 8000)
% N = number of channels
%
% Output is corner frequencies in Hz, center frequencies determined by the 
% geometric mean in Hz, and the bandwidth in Hz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lowPos = CochleaFreq2Pos(lower);
highPos = CochleaFreq2Pos(upper);
corners_mm = linspace(lowPos,highPos,N+1);
% corners_mm = CochleaFreq2Pos(200:1900:12000);N = length(corners_mm)-1;

corners = Pos2CochleaFreq(corners_mm);

for ii=1:N
    CFgeo(ii)=sqrt(corners(ii)*corners(ii+1));
    CFari(ii)=(corners(ii)+corners(ii+1))/2;
    CF(ii) = Pos2CochleaFreq((corners_mm(ii)+corners_mm(ii+1))/2);
    BW(ii)=corners(ii+1)-corners(ii);
end

corners=corners';
CF=CF';
CFgeo=CFgeo';
CFari=CFari';
BW=BW';