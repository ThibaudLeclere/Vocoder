classdef Vocoder
    properties
        SampleRate = 44100;
        
        % Analysis Parameters
        analysis = struct('LowF', 50 ...
                        ,'HighF', 8000 ...
                        ,'Nchannels',8 ...
                        ,'Method', 'Butterworth'...
                        ,'Order', 4 ...
                        ,'FreqAllocation','Greenwood'...
                        );
        
        % Synthesis Parameters
        synthesis = struct('CarrierType', 'Tone'...
                        ,'FrequencyShift_mm', 6 ...
                        ,'Routing',[]...
                        ,'Carriers_CF',[]...
                        ,'FrequencyShiftMethod', 'fix_mm' ... % 'Cochlear', 'Medel', 'AB', 'Custom' or 'Noshift'
                        ,'CarrierPhase', 0 ...
                        ,'Delay', [0 0] ... % Delay value (in seconds)
                        ,'DelayType', 'WAV'... % TFS (carrier), ENV (envelope), WAV (waveform)
                        ,'DelayChannels', 'LR'... % On what channels apply an ITD: 'HR' (High-rate), 'LR' (Low-rate), 'all' (All channels)
                        );
        
        % Envelope extraction parameters
        envelopeExtraction = struct('Method','Halfwave'...
                                   ,'LPcutoff', 400 ...
                                  );
        
        % Mixed Rates parameters
        mixedRates = struct('MixedRates', true ...
                            ,'Distribution','AllLow'...
                            ,'HighPulseRate', 1000 ...
                            ,'LowPulseRate', 100 ...
                            );
        
        
    end
    properties (SetAccess = private)
        AnalysisComputations = struct('Filters', []...
            ,'CenterFreq',[]...
            ,'GeoCF',[]...
            ,'AriCF',[]...
            ,'BW',[]...
            );
        
        SynthesisComputations = struct('Filters', []...
            ,'Carriers',[]...
            ,'Carriers_CF',[] ...
            );
        
        
        
        
    end
    
    properties (Hidden = true)
        methodDictionary = struct('Butterworth', 'butter'...
            ,'Chebychev1', 'cheby1'...
            ,'Chebychev2', 'cheby2'...
            ,'Equiripple', 'equiripple');
    end
    
    methods
        % ---- CONSTRUCTOR
        % ---------------
        function obj = Vocoder(varargin)
            displayCommandWindow('Creating new Vocoder Object', 70)
            if nargin == 0 % If no argument are passed in, create a default Vocoder (this occurs when initializing an array of Vocoder objects)
                obj = designAnalysisFilters(obj);
            end
            
            if nargin > 0 % Set Analysis structure
                enteredAnalysis = varargin{1};
                              
                % Check that analysis input is a structure
                if ~isstruct(enteredAnalysis)   
                    error('First argument must be a structure (analysis)')                    
                else     % If it is, assign values to vocoder object only if field names match.               
                    enteredFieldnames = fieldnames(enteredAnalysis);
                    s = obj.analysis;
                    for i = 1:length(enteredFieldnames)
                       if ~isfield(s,enteredFieldnames{i})
                           displayCommandWindow('Warning')
                           warning('The field ''%s'' does not exist in the analysis property of the Vocoder class. Assignment has been ignored for this field.', enteredFieldnames{i})
                       else
                           s.(enteredFieldnames{i}) = enteredAnalysis.(enteredFieldnames{i});
                       end                        
                    end
                    obj.analysis = s;
                    
                end
                
            end
            if nargin > 1 % Set synthesis structure
                obj.synthesis = varargin{2};
            end
            if nargin > 2 % Set envelope extraction structure
                obj.envelopeExtraction = varargin{3};
            end
            if nargin > 3 % Set mixed rates structure
                obj.mixedRates = varargin{4};
            end
            
           displayCommandWindow('Complete') 
        end
        
        function obj = set.SampleRate(obj, fs)
            obj.SampleRate = fs;
        end
        
        function obj = set.analysis(obj,analysisStruct)
            if ~isstruct(analysisStruct)
                % Erreur
            else
               obj.analysis = analysisStruct; 
            end 
            
            % Recompute Analysis Filters
            disp('Designing Analysis Filters')
            obj = designAnalysisFilters(obj);
        end
        
        function obj = set.synthesis(obj,synthesisStruct)
            if ~isstruct(synthesisStruct)
                % Erreur
            else
               obj.synthesis = synthesisStruct; 
            end  
            
        end
        
        
        
        % ------------ DESIGN ANALYSIS FILTERBANK
        % -------------------------------
        function VocoderObj = designAnalysisFilters(VocoderObj)
            N = VocoderObj.analysis.Nchannels;
            
            % Get frequency corners
            switch VocoderObj.analysis.FreqAllocation
                case 'Greenwood'
                    [corners,CFA,GeoCFA,AriCFA,bandwidth]=CalcGreenwoodCorners(VocoderObj.analysis.LowF,VocoderObj.analysis.HighF,N);
            end
            
            % Design bandpass analysis filters
            h = waitbar(0,'Designing Analysis Filters');            
            
            filterbank = [];
            
            for i = 1:N
                waitbar(i/N, h, sprintf('Designing Analysis Filters (%d/%d)', i, N))
                bpFilter = designfilt('bandpassiir',...
                    'FilterOrder', VocoderObj.analysis.Order,...
                    'HalfPowerFrequency1', corners(i),...
                    'HalfPowerFrequency2', corners(i+1),...
                    'SampleRate',VocoderObj.SampleRate,...
                    'DesignMethod', VocoderObj.methodDictionary.(VocoderObj.analysis.Method)...
                    );
                
%                 [Bcoeffs(:,i), Acoeffs(:,i)] = tf(bpFilter);
                filterbank = [filterbank bpFilter];
                
                
            end
            
            close(h)
            VocoderObj.AnalysisComputations.Filters = filterbank;
            VocoderObj.AnalysisComputations.CenterFreq = CFA;
            VocoderObj.AnalysisComputations.AriCF = AriCFA;
            VocoderObj.AnalysisComputations.GeoCF = GeoCFA;
            VocoderObj.AnalysisComputations.BW = bandwidth;
            
        end
        
        % ------------ EXTRACT ENVELOPES
        %------------------------------
        function [envelopes, energyByBand, filteredSignals] = extractEnvelopes(VocoderObj, audio)
            N = VocoderObj.analysis.Nchannels;
            L = length(audio);
            
            envelopes = zeros(L,N);
            filteredSignals = zeros(size(envelopes));
            energyByBand = zeros(1,N);
            r = (2*100e-3)/L;
            win = repmat(tukeywin(L,r),1,N);
            
            % Design LP filter
            LP_env = designfilt('lowpassiir',...
                                'StopbandFrequency', VocoderObj.envelopeExtraction.LPcutoff+200,...
                                'PassbandFrequency', VocoderObj.envelopeExtraction.LPcutoff,...
                                'StopbandAttenuation',40,...
                                'SampleRate',VocoderObj.SampleRate,...
                                'DesignMethod', 'butter');
                            
                          
            for i = 1:N % For each channel
                
                % Get filter coefficients
                [B, A] = tf(VocoderObj.AnalysisComputations.Filters(i));
                % Bandpass filter input audio
                filteredSig = FiltFiltM(B, A, audio);
                energyByBand(i) = energy(filteredSig);
                
                % Extract envelope in each channel
                switch lower(VocoderObj.envelopeExtraction.Method)
                    case  'halfwave'
                        env = 0.5*(abs(filteredSig) + filteredSig);
                    case 'fullwave'
                    case 'hilbert'
                        env = abs(hilbert(filteredSig));
                end
                
                envelopes(:,i) = filtfilt(LP_env, env);
                filteredSignals(:,i) = filteredSig;
                
                
                
                
            end
            
            % Windowing
            envelopes = envelopes.*win;
        end
        
        % ------------ COMPUTE CARRIERS CENTER FREQUENCIES
        % ------------------------------------------------
        function VocoderObj = computeCarriers_CF(VocoderObj)
            Nchannels = VocoderObj.analysis.Nchannels;
            
           switch lower(VocoderObj.synthesis.FrequencyShiftMethod)
               case 'cochlear' % 
                   Nelectrodes = 22; % 22 electrodes on the array                  
                   spacing = 0.75; % electrode spacing (mm)
                   
                   if isempty(VocoderObj.synthesis.Routing)
                       if mod(Nelectrodes-1,Nchannels-1) == 0 % If multiple of 21
                           routing = linspace(1,Nelectrodes,Nchannels);
                       elseif mod(Nelectrodes-2,Nchannels-1) == 0 % If multiple of 20
                           routing = linspace(1,Nelectrodes-1,Nchannels);
                       else % Otherwise                           
                           routing = round(linspace(1,Nelectrodes,Nchannels));
                       end
                   else
                       routing = VocoderObj.synthesis.Routing;
                   end
                   
                   electrodeLocations = 17:spacing:17+spacing*21; % From Apex
                   CF = Pos2CochleaFreq(electrodeLocations(routing));
               case 'medel'
                   
               case 'ab'
                   
               case 'custom'
                   CF = VocoderObj.synthesis.Carriers_CF;
                   
               case 'noshift' % Center frequency of carriers match those of the analysis filters
                   CF = VocoderObj.AnalysisComputations.CenterFreq;
                   
               case 'fix_mm'
                   shift_mm = VocoderObj.synthesis.FrequencyShift_mm;
                   shiftedPlaces = CochleaFreq2Pos(VocoderObj.AnalysisComputations.CenterFreq) + shift_mm; 
                   CF = Pos2CochleaFreq(shiftedPlaces);
           end
           
           
           VocoderObj.SynthesisComputations.Carriers_CF = CF;
           
            
        end
        
        % ------------ COMPUTE CARRIERS
        %------------------------------
        function VocoderObj = computeCarriers(VocoderObj, L)
            t = (0:L-1)/VocoderObj.SampleRate; % Time vector
            Nchannels = VocoderObj.analysis.Nchannels;
            carriers = zeros(L,Nchannels);
            carrierPhase = VocoderObj.synthesis.CarrierPhase;
            
            switch VocoderObj.synthesis.CarrierType
                case 'Tone' 
%                     frequencyShift = VocoderObj.shift_Freq(VocoderObj.AnalysisComputations.CenterFreq,VocoderObj.synthesis.FrequencyShift_mm);
                                        
                    cf = VocoderObj.SynthesisComputations.Carriers_CF;
                    
                    for i = 1:Nchannels
                        if ischar(carrierPhase) && strcmp(carrierPhase, 'random')
                            phase = 2*pi*rand(1);
                        else                            
                            phase = carrierPhase;
                        end      
                            carriers(:,i) = sin(2*pi*cf(i)*t+phase);
                    end
                    
                case 'Noise'
            end
            
            VocoderObj.SynthesisComputations.Carriers = carriers;
        end

        % ------------ DEGRADE ENVELOPES
        % ------------------------------
        function degradedEnvelopes = degradeEnvelopes(VocoderObj,envelopes)
            N = VocoderObj.analysis.Nchannels;
            L = size(envelopes,1);
            fs = VocoderObj.SampleRate;
            
            mixedRateDistribution = VocoderObj.create_MRdistribution(VocoderObj.mixedRates.Distribution, N);
            
            % Initialiazing Pulse trains
            lowPulseTrain = zeros(L,N);
            highPulseTrain = zeros(L,N);
            
            lowPulseRate = VocoderObj.mixedRates.LowPulseRate;
            highPulseRate = VocoderObj.mixedRates.HighPulseRate;
            
            % Sampling Envelopes
            lowPulseTrain(1:round(fs/lowPulseRate):L,~mixedRateDistribution) = envelopes(1:round(fs/lowPulseRate):L,~mixedRateDistribution);
            highPulseTrain(1:round(fs/highPulseRate):L,mixedRateDistribution) = envelopes(1:round(fs/highPulseRate):L,mixedRateDistribution);
                    
            
            % Reconstruction
            % Designing filters
            lpIIR_high = designfilt('lowpassiir',...
                'StopbandFrequency', highPulseRate/2 + 100,...
                'PassbandFrequency', highPulseRate/2,...
                'StopbandAttenuation',40,...
                'SampleRate',fs,...
                'DesignMethod', 'butter');
            
            lpIIR_low = designfilt('lowpassiir',...
                'StopbandFrequency', lowPulseRate/2 +20,...
                'PassbandFrequency', lowPulseRate/2,...
                'StopbandAttenuation',40,...
                'SampleRate',fs,...
                'DesignMethod', 'butter');

            highPulseTrain_reconstructed = filtfilt(lpIIR_high,highPulseTrain);
            lowPulseTrain_reconstructed = filtfilt(lpIIR_low,lowPulseTrain);
            
            degradedEnvelopes = lowPulseTrain_reconstructed + highPulseTrain_reconstructed;
        end
        
        % ------------ APPLY DELAY
        % ------------------------
        function [VocoderObj, delayedEnvelopes] = applyDelay(VocoderObj, delay, envelopes)
            disp('Applying Delay')

            delaySamples = round(delay*VocoderObj.SampleRate);
            delayChannels = VocoderObj.synthesis.DelayChannels;
            Nchannels = size(envelopes,2);
            channelsIdx = 1:Nchannels;
            carriers = VocoderObj.SynthesisComputations.Carriers;
            
            if VocoderObj.mixedRates.MixedRates
                MR = VocoderObj.create_MRdistribution(VocoderObj.mixedRates.Distribution, Nchannels);
                
                switch delayChannels
                    case 'HR'
                        channelsIdx = find(MR);
                    case 'LR'
                        channelsIdx = find(~MR);
                    case 'all'
                        channelsIdx = 1:Nchannels;
                end
            end
            
            
            
            % Delay coresponding channels while preserving the length
            delayedEnvelopes = envelopes;
            delayedCarriers = carriers;
            switch VocoderObj.synthesis.DelayType
                case 'ENV' % Apply delay on envelope only
                    delayedEnvelopes(:,channelsIdx) = [zeros(delaySamples,length(channelsIdx)); envelopes(1:end-delaySamples,channelsIdx)];
                    

                case 'TFS' % Apply delay on fine structure only
                    delayedCarriers(:,channelsIdx) = [zeros(delaySamples,length(channelsIdx)); carriers(1:end-delaySamples,channelsIdx)];
                    

                case 'WAV' % Apply delay on both envelope and fine structure
                    delayedEnvelopes(:,channelsIdx) = [zeros(delaySamples,length(channelsIdx)); envelopes(1:end-delaySamples,channelsIdx)];
                    delayedCarriers(:,channelsIdx) = [zeros(delaySamples,length(channelsIdx)); carriers(1:end-delaySamples,channelsIdx)];
            end
            VocoderObj.SynthesisComputations.Carriers = delayedCarriers;
        end
        
        % ----------- SYNTHESIZE
        %-----------------------
        function [vocodedAudio, modulatedCarriers, VocoderObj] = synthesize(VocoderObj,envelopes,L, energyByBand,k)
            
            % ---- Compute Carriers Center Frequencies
            if isempty(VocoderObj.SynthesisComputations.Carriers_CF)
                VocoderObj = VocoderObj.computeCarriers_CF();
            end
            
            % ---- Compute Carriers
            VocoderObj = VocoderObj.computeCarriers(L);
            
            
            % ---- Apply ITD
            delay = VocoderObj.synthesis.Delay(k);
            if delay ~= 0
                [VocoderObj, envDelayed] = applyDelay(VocoderObj, delay, envelopes);                           
            end
            
            % ---- Multiply envelopes by corresponding carriers
                if exist('envDelayed', 'var')
                    modulatedCarriers = envDelayed.*VocoderObj.SynthesisComputations.Carriers;
                else
                    modulatedCarriers = envelopes.*VocoderObj.SynthesisComputations.Carriers;
                end
                
            % ---- Equalize channels
            modulatedCarriers_Eq = energyEqualization(modulatedCarriers, energyByBand, 'each');
            % ---- Sum channels
            vocodedAudio = sum(modulatedCarriers_Eq,2);
            
            % ---- Temporal Windowing 
            onsetTime = 100e-3; % 50 ms
            r = 2*onsetTime*VocoderObj.SampleRate/length(vocodedAudio);
            win = tukeywin(length(vocodedAudio),r);
            vocodedAudio = win.*vocodedAudio;
            
            
        end
        
        % ----------- VOCODE AUDIO
        %------------------------
        function varargout = Vocode(VocoderObj,input, fs)
            if ischar(input) % If input is a path to an audio file
                [audio, fs] = audioread(input);
                Nsignals = size(audio,2);
                flag_Binaural_Input = false;
                NBinauralSignals = 1;
                
            elseif isnumeric(input) % If Input is a matrix
                flag_Binaural_Input = false;
                Nsignals = size(input,2);
                audio = input;
                NBinauralSignals = 1;
                
                % Preallocation
                output = zeros(size(audio));
                
            elseif isa(input, 'BinauralSignal') % If input is a BinauralSignal array
                
                Nsignals = 2;
                NBinauralSignals = numel(input);
                flag_Binaural_Input = true;                
                   
                % Preallocation
                output(NBinauralSignals) = BinauralSignal();
            end
            
            for n = 1:NBinauralSignals
                if flag_Binaural_Input
                    audio = horzcat(input(n).leftChannel, input(n).rightChannel);
                    fs = input(n).fs;
                    label = input(n).label;
                    if ~exist('b', 'var')
                        b = waitbar(0, 'Binaural Signal');
                    end
                    
                    flag_Diotic = input(n).isDiotic;
                    
                    waitbar(n/NBinauralSignals, b, sprintf('Binaural Signal n° %d/%d',n,NBinauralSignals));
                end
                if VocoderObj.SampleRate ~= fs
                    VocoderObj.SampleRate = fs;
                end
                
                % Preallocation
                vocodedAudio = zeros(size(audio));
                
                h = waitbar(0, 'Vocoding Signal');
                for k = 1:Nsignals
                    waitbar(k/Nsignals, h, sprintf('Vocoding Signal (%d/%d)',k, Nsignals))
                    if isempty(VocoderObj.AnalysisComputations.Filters) % If no analysis filters have not been computed
                        % ------ Design Analysis Filters
                        displayCommandWindow('Designing Filterbank', 50)
                        VocoderObj = VocoderObj.designAnalysisFilters;
                        displayCommandWindow('Done')
                    end
                    
                    
                    % ----- Envelope Extraction
                    displayCommandWindow('Extracting Envelopes', 50)
                    [extractedEnvelopes, energyByBand, filteredSignals] = VocoderObj.extractEnvelopes(audio(:,k));
                    displayCommandWindow('Done')
                    
                    if VocoderObj.mixedRates.MixedRates
                        env = VocoderObj.degradeEnvelopes(extractedEnvelopes);                        
                    else
                        env = extractedEnvelopes;
                    end
                    
                    % ----- Perform Synthesis   
                    [vocodedAudio(:,k), modulatedCarriers, VocoderObj] = VocoderObj.synthesize(env,size(audio,1),energyByBand,k);
                    
                end
                close(h)
                
                if flag_Binaural_Input
                    output(n) = BinauralSignal(vocodedAudio, fs, [label '_vocoded']);
                else
                    output = vocodedAudio;
                end
            end
            
            if flag_Binaural_Input
                close(b)
                output = reshape(output,size(input));
            end
            varargout = {output, modulatedCarriers, env, filteredSignals, VocoderObj};
        end
        
       
        
        % ------------ PLOTS
        % -------------------------
        function PlotAnalysisFilterbank(obj)
            % Plot Analysis Filters
            
            if ~isempty(obj.AnalysisComputations.Filters)
                % Get center frequencies of anlaysis filters
                cf = obj.AnalysisComputations.CenterFreq;
                geocf = obj.AnalysisComputations.GeoCF;
                aricf = obj.AnalysisComputations.AriCF;
%                 bw = obj.AnalysisComputations.BW;
                
                figure, hold on
                for n = 1:length(obj.AnalysisComputations.Filters)
                    [H, f] = freqz(obj.AnalysisComputations.Filters(n), 4096, obj.SampleRate);
                    
                    
                    plot(f,20*log10(abs(H)), 'linewidth', 2)
                    line([cf(n) cf(n)], [-100 0], 'color', 'k', 'linestyle', '--')                    
                    line([geocf(n) geocf(n)], [-100 0], 'color', 'b', 'linestyle', '--') 
                    line([aricf(n) aricf(n)], [-100 0], 'color', 'r', 'linestyle', '--') 
                    text(round(cf(n)),2,num2str(round(cf(n))))
                end
                
                line([obj.analysis.LowF obj.analysis.LowF], [-100 0], 'color', 'r', 'linestyle', '-')
                line([obj.analysis.HighF obj.analysis.HighF], [-100 0], 'color', 'r', 'linestyle', '-')
                line([obj.analysis.LowF obj.analysis.HighF], [-3 -3], 'color', 'k', 'linestyle', '--')
                
                set(gca, 'XScale', 'log'...
                       , 'Box', 'on'...
                       ,'YLim', [-50 10]... 
                       ,'XLim', [obj.analysis.LowF obj.analysis.HighF]...
                    );
                xlabel('Frequency (Hz)', 'FontSize', 18)
                ylabel('Magnitude (dB)', 'FontSize', 18)
                title('Analyis Filterbank', 'FontSize', 24)
            else
                disp('Analysis Filterbank not designed yet')
                
                
            end
        end
        
        function PlotCarriers(obj)
            % Plot computed Carriers
            if ~isempty(obj.SynthesisComputations.Carriers)
                [L, N] = size(obj.SynthesisComputations.Carriers);
                figure, hold on
                
                % Time Vector
                t = (0:L-1)/obj.SampleRate; 
                for n = 1:N
                    subplot(N,1,n)
                    plot(t,obj.SynthesisComputations.Carriers(:,n))
                    
                    
                end
                
            else
                disp('Carriers not computed yet')
            end
            
            
        end
        
        % ----------- Get Vocoder Parameters
        %-----------------------------------
        function vocParameters = printParameters(VocObj)
            
        vocParameters = sprintf('Sample Rate: %d', VocObj.SampleRate);
        vocParameters = horzcat(vocParameters, sprintf('\n\n-----ANALYSIS SETTINGS-------\n'));

        vocParameters = horzcat( vocParameters, sprintf('\nFirst Corner frequency: %d Hz', VocObj.analysis.LowF));
        vocParameters = horzcat(vocParameters, sprintf('\nLast corner frequency: %d Hz', VocObj.analysis.HighF));
        vocParameters = horzcat(vocParameters, sprintf('\nCenter frequencies (Hz): %s', array2str(VocObj.AnalysisComputations.CenterFreq,',')));
        vocParameters = horzcat(vocParameters, sprintf('\nNumber of Channels: %d', VocObj.analysis.Nchannels));
        vocParameters = horzcat(vocParameters, sprintf('\nFilter type: %s', VocObj.analysis.Method));
        vocParameters = horzcat(vocParameters, sprintf('\nFilter Order: %d', VocObj.analysis.Order));
        vocParameters = horzcat(vocParameters, sprintf('\nFrequency Allocation: %s', VocObj.analysis.FreqAllocation));

        vocParameters = horzcat(vocParameters, sprintf('\n\n-----SYNTHESIS SETTINGS-------\n'));

        vocParameters = horzcat( vocParameters, sprintf('\nCarrier Type: %s', VocObj.synthesis.CarrierType));
        if ~isempty(VocObj.SynthesisComputations.Carriers_CF)
            vocParameters = horzcat( vocParameters, sprintf('\nCarrier Center Frequencies (Hz): %s', array2str(VocObj.SynthesisComputations.Carriers_CF,',')));
        end
        if ischar(VocObj.synthesis.CarrierPhase)
            vocParameters = horzcat( vocParameters, sprintf('\nCarrier Phase: %s', VocObj.synthesis.CarrierPhase));
        else
            vocParameters = horzcat( vocParameters, sprintf('\nCarrier Phase: %d', VocObj.synthesis.CarrierPhase));
        end
        vocParameters = horzcat(vocParameters, sprintf('\nFrequency Shift Method: %s', VocObj.synthesis.FrequencyShiftMethod));
        if strcmp(VocObj.synthesis.FrequencyShiftMethod, 'fix_mm')
            vocParameters = horzcat(vocParameters, sprintf('\nFrequency Shift: %d mm', VocObj.synthesis.FrequencyShift_mm));
        end
        vocParameters = horzcat(vocParameters, sprintf('\nDelays on each audio channel: %.2eµs %.2eµs', VocObj.synthesis.Delay));
        vocParameters = horzcat(vocParameters, sprintf('\nDelay applied on: %s of %s channels', VocObj.synthesis.DelayType, VocObj.synthesis.DelayChannels));

        vocParameters = horzcat(vocParameters, sprintf('\n\n-----ENVELOPE EXTRACTION SETTINGS-------\n'));
        vocParameters = horzcat(vocParameters, sprintf('\nMethod: %s', VocObj.envelopeExtraction.Method));
        vocParameters = horzcat(vocParameters, sprintf('\nLow-pass cutoff: %d Hz', VocObj.envelopeExtraction.LPcutoff));
        
                   
        if VocObj.mixedRates.MixedRates
            vocParameters = horzcat(vocParameters, sprintf('\n\n-----MIXED RATES SETTINGS-------\n'));
            vocParameters = horzcat(vocParameters, sprintf('\nDistribution: %s', VocObj.mixedRates.Distribution));
            vocParameters = horzcat(vocParameters, sprintf('\nHigh Sample Rate: %d Hz', VocObj.mixedRates.HighPulseRate));
            vocParameters = horzcat(vocParameters, sprintf('\nLow Sample Rate: %d Hz', VocObj.mixedRates.LowPulseRate));
        
        end
        
        
        end % end printParameters
    end % end methods
    
    
    
    % ------------------------------ STATIC METHODS
    %----------------------------------------------
    methods (Static)
        
        % --------- GENERATE MIXED RATE DISTRIBUTION
        % ------------------------------------------
        function mixedRateDistribution = create_MRdistribution(distrib, Nchannels)
           % Channels represented as columns (1: Low Freq (Apex) --> N: High Freq (Base)) 
           switch strrep(lower(distrib),' ', '')
               case 'interleaved'
                   if iseven(Nchannels)
                       mixedRateDistribution = repmat([true false], 1, Nchannels/2);
                   end
               case 'apicalhigh'
                   if iseven(Nchannels)
                        mixedRateDistribution = [true(1,Nchannels/2) false(1,Nchannels/2) ];
                       
                   end
               case 'basalhigh'
                   if iseven(Nchannels)
                      mixedRateDistribution = [false(1,Nchannels/2) true(1,Nchannels/2)];
                   end
               case 'alllow'
                   mixedRateDistribution = false(1,Nchannels);
                   
               case 'allhigh'
                   mixedRateDistribution = true(1,Nchannels);
           end
            
        end
        
         function shift_Hz = shift_Freq(f, shift_mm)
            shift_Hz = Pos2CochleaFreq(CochleaFreq2Pos(f) + shift_mm);  
        end
        
        
        
    end
        
    
end
