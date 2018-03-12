function displayCommandWindow(message,width)



% type : start/end

if nargin < 2
    fprintf([message '\n'])
else
    fprintf([message repmat('-',1,width-length(message))])
end


end