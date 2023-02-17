function [regLabels,regIdx,fullR,regZeroFrames,zeromeanVc,U] = ridgeModel_returnDesignMatrix(cPath,Animal,Rec)
%Return the design matrix for ridge regression model described in Musall
%2019 on data from that paper. I have begun refactoring the code, but the
%overall functionality should not have changed.

%% Constants

%task
PIEZOLINE = 2;     % channel in the analog data that contains data from piezo sensor
STIMLINE = 6;      % channel in the analog data that contains stimulus trigger.
PARADIGM = 'SpatialDisc';

%data to fit
PRE_HANDLE_SECONDS = 2; %for each trial, fit to 2 seconds of wfield data preceding trial onset
POST_HANDLE_SECONDS = 3; %for each trial, fit to 3 seconds of wfield data after trial onset

%kernel lengths
MOTOR_PRE_SECONDS = 0.5; % precede motor events to capture preparatory activity for _ seconds
MOTOR_POST_SECONDS = 5; % follow motor events for _ seconds. (so in this case the motor kernels will be 5.5 seconds long)
FIRST_STIM_POST_SECONDS = 5; % follow first stim event for _ seconds
FIRST_STIM_PRE_SECONDS = 0; % don't fit stim driven activity before stimulus comes on
STIM_POST_SECONDS = 2; % follow all other stim events for _ seconds
STIM_PRE_SECONDS = 0; % don't fit stim driven activity before stimulus comes on
GAUSSIAN_SHIFT = 1; %leave as 1. I believe this is included for 2P and neuropixels data, but we don't need to do any gaussian smoothing for widefield.

%other
TAPDUR = 0.1;      % minimum time of lever contact, required to count as a proper grab.
LEVER_MOVE_SECONDS = 0.3; %duration of lever movement. this is used to orthogonalize video against lever movement.

BHV_DIM_COUNT = 200;    % number of dimensions from behavioral videos that are used as regressors.
DIMS = 200; %number of dimensions in wfield temporal components that are used in the model

%% general variables

cPath = [cPath Animal filesep PARADIGM filesep Rec filesep]; %Widefield data path

if ~strcmpi(cPath(end),filesep)
    cPath = [cPath filesep];
end

load([cPath 'opts.mat'], 'opts');         % get some options from imaging data
sRate = opts.frameRate;           % Sampling rate of imaging
if sRate == 30
    vcFile = 'rsVc.mat'; %use downsampled data here and change sampling frequency
    sRate = 15;
else
    vcFile = 'Vc.mat';
end


frames = round((PRE_HANDLE_SECONDS + POST_HANDLE_SECONDS) * sRate); %nr of frames per trial
trialDur = (frames * (1/sRate)); %duration of trial in seconds

mPreTime = ceil(MOTOR_PRE_SECONDS * sRate); % conversions to frames
mPostTime = ceil(MOTOR_POST_SECONDS * sRate);
fsPostTime = ceil(FIRST_STIM_POST_SECONDS * sRate);
sPostTime = ceil(STIM_POST_SECONDS * sRate);
leverMoveDur = ceil(LEVER_MOVE_SECONDS * sRate);

motorIdx = [-(mPreTime: -1 : 1) 0 (1:mPostTime)]; %index for design matrix to cover pre- and post motor action
shVal = sRate * opts.preStim  + 1; %expected position of stimulus onset in the imaging data (s).
maxStimShift = 1 * sRate; % maximal stimulus onset after handle grab. (default is 1s - this means that stimulus onset should not be more than 1s after handle grab. Cognitive regressors will have up to 1s of baseline because of stim jitter.)

%% load data
bhvFile = dir([cPath filesep Animal '_' PARADIGM '*.mat']);
load([cPath bhvFile(1).name],'SessionData'); %load behavior data
SessionData.TrialStartTime = SessionData.TrialStartTime * 86400; %convert trailstart timestamps to seconds

load([cPath vcFile], 'Vc', 'U', 'trials', 'bTrials')
Vc = Vc(1:DIMS,:,:);
U = U(:,:,1:DIMS);

% ensure there are not too many trials in Vc
ind = trials > SessionData.nTrials;
trials(ind) = [];
bTrials(ind) = [];
Vc(:,:,ind) = [];

bhv = selectBehaviorTrials(SessionData,bTrials); %only use completed trials that are in the Vc dataset


%equalize L/R choices, with secondary equalization to reward
useIdx = ~isnan(bhv.ResponseSide); %only use performed trials
choiceIdx = rateDisc_equalizeTrials(useIdx, bhv.ResponseSide == 1, bhv.Rewarded, inf, true);

trials = trials(choiceIdx);
bTrials = bTrials(choiceIdx);
Vc = Vc(:,:,choiceIdx);
bhv = selectBehaviorTrials(SessionData,bTrials); %only use completed trials that are in the Vc dataset
trialCnt = length(bTrials);


%% load behavior data
if exist([cPath 'BehaviorVideo' filesep 'SVD_CombinedSegments.mat'],'file') ~= 2 || ... %check if svd behavior exists on hdd and pull from server otherwise
        exist([cPath 'BehaviorVideo' filesep 'motionSVD_CombinedSegments.mat'],'file') ~= 2

    if ~exist([cPath 'BehaviorVideo' filesep], 'dir')
        mkdir([cPath 'BehaviorVideo' filesep]);
    end
    copyfile([sPath 'BehaviorVideo' filesep 'SVD_CombinedSegments.mat'],[cPath 'BehaviorVideo' filesep 'SVD_CombinedSegments']);
    copyfile([sPath 'BehaviorVideo' filesep 'motionSVD_CombinedSegments.mat'],[cPath 'BehaviorVideo' filesep 'motionSVD_CombinedSegments.mat']);
    copyfile([sPath 'BehaviorVideo' filesep 'FilteredPupil.mat'],[cPath 'BehaviorVideo' filesep 'FilteredPupil.mat']);
    copyfile([sPath 'BehaviorVideo' filesep 'segInd1.mat'],[cPath 'BehaviorVideo' filesep 'segInd1.mat']);
    copyfile([sPath 'BehaviorVideo' filesep 'segInd2.mat'],[cPath 'BehaviorVideo' filesep 'segInd2.mat']);

    movFiles = dir([sPath 'BehaviorVideo' filesep '*Video_*1.mj2']);
    copyfile([sPath 'BehaviorVideo' filesep movFiles(1).name],[cPath 'BehaviorVideo' filesep movFiles(1).name]);
    movFiles = dir([sPath 'BehaviorVideo' filesep '*Video_*2.mj2']);
    copyfile([sPath 'BehaviorVideo' filesep movFiles(1).name],[cPath 'BehaviorVideo' filesep movFiles(1).name]);

    svdFiles = dir([sPath 'BehaviorVideo' filesep '*SVD*-Seg*']);
    for iFiles = 1:length(svdFiles)
        copyfile([sPath 'BehaviorVideo' filesep svdFiles(iFiles).name],[cPath 'BehaviorVideo' filesep svdFiles(iFiles).name]);
    end
end

load([cPath 'BehaviorVideo' filesep 'SVD_CombinedSegments.mat'],'vidV'); %load behavior video data
V1 = vidV(:,1:BHV_DIM_COUNT); %behavioral video regressors
load([cPath 'BehaviorVideo' filesep 'motionSVD_CombinedSegments.mat'],'vidV'); %load abs motion video data
V2 = vidV(:,1:BHV_DIM_COUNT); % motion regressors

% check options that were used for dimensionality reduction and ensure that imaging and video data trials are equal length
load([cPath 'BehaviorVideo' filesep 'bhvOpts.mat'], 'bhvOpts'); %load abs motion video data
bhvRate = bhvOpts.targRate; %framerate of face camera
if (bhvOpts.preStimDur + bhvOpts.postStimDur) > (opts.preStim + opts.postStim) %if behavioral video trials are longer than imaging data
    V1 = reshape(V1, [], SessionData.nTrials, BHV_DIM_COUNT);
    V2 = reshape(V2, [], SessionData.nTrials, BHV_DIM_COUNT);
    if bhvOpts.preStimDur > opts.preStim
        frameDiff = ceil((bhvOpts.preStimDur - opts.preStim) * bhvRate); %overhead in behavioral frames that can be removed.
        V1 = V1(frameDiff+1:end, :, :); %cut data to size
        V2 = V2(frameDiff+1:end, :, :);
    end
    if bhvOpts.postStimDur > opts.postStim
        frameDiff = ceil((bhvOpts.postStimDur - opts.postStim) * bhvRate); %overhead in behavioral frames that can be removed.
        V1 = V1(1 : end - frameDiff, :, :); %cut data to size
        V2 = V2(1 : end - frameDiff, :, :);
    end
    V1 = reshape(V1, [], BHV_DIM_COUNT);
    V2 = reshape(V2, [], BHV_DIM_COUNT);
end

load([cPath 'BehaviorVideo' filesep 'FilteredPupil.mat'], 'pTime', 'fPupil', 'sPupil', 'whisker', 'faceM', 'bodyM', 'nose', 'bTime'); %load pupil data
%check if timestamps from pupil data are shifted against bhv data
timeCheck1 = (SessionData.TrialStartTime(1)) - (pTime{1}(1)); %time difference between first acquired frame and onset of first trial
timeCheck2 = (SessionData.TrialStartTime(1)) - (bTime{1}(1)); %time difference between first acquired frame and onset of first trial
if (timeCheck1 > 3590 && timeCheck1 < 3610) && (timeCheck2 > 3590 && timeCheck2 < 3610) %timeshift by one hour (+- 10seconds)
    warning('Behavioral and video timestamps are shifted by 1h. Will adjust timestamps in video data accordingly.')
    for iTrials = 1 : length(pTime)
        pTime{iTrials} = pTime{iTrials} + 3600; %add one hour
        bTime{iTrials} = bTime{iTrials} + 3600; %add one hour
    end
elseif timeCheck1 > 30 || timeCheck1 < -30 || timeCheck2 > 30 || timeCheck2 < -30
    error('Something wrong with timestamps in behavior and video data. Time difference is larger as 30 seconds.')
end

if any(bTrials > length(pTime))
    warning(['There are insufficient trials in the pupil data. Rejected the last ' num2str(sum(bTrials > length(pTime))) ' trial(s)']);
    bTrials(bTrials > length(pTime)) = [];
    trialCnt = length(bTrials);
end


%% find events in BPod time - All timestamps are relative to stimulus onset event to synchronize to imaging data later
% pre-allocate vectors
lickL = cell(1,trialCnt);
lickR = cell(1,trialCnt);
leverIn = NaN(1,trialCnt);
levGrabL = cell(1,trialCnt);
levGrabR = cell(1,trialCnt);
levReleaseL = cell(1,trialCnt);
levReleaseR = cell(1,trialCnt);
water = NaN(1,trialCnt);
handleSounds = cell(1,trialCnt);

tacStimL = cell(1,trialCnt);
tacStimR = cell(1,trialCnt);
audStimL = cell(1,trialCnt);
audStimR = cell(1,trialCnt);

for iTrials = 1:trialCnt

    leverTimes = [reshape(bhv.RawEvents.Trial{iTrials}.States.WaitForAnimal1',1,[]) ...
        reshape(bhv.RawEvents.Trial{iTrials}.States.WaitForAnimal2',1,[]) ...
        reshape(bhv.RawEvents.Trial{iTrials}.States.WaitForAnimal3',1,[])];

    try
        stimGrab(iTrials) = leverTimes(find(leverTimes == bhv.RawEvents.Trial{iTrials}.States.WaitForCam(1))-1); %find start of lever state that triggered stimulus onset
        handleSounds{iTrials} = leverTimes(1:2:end) - stimGrab(iTrials); %track indicator sound when animal is grabing both handles
        stimTime(iTrials) = bhv.RawEvents.Trial{iTrials}.Events.Wire3High - stimGrab(iTrials); %time of stimulus onset - measured from soundcard
        stimEndTime(iTrials) = bhv.RawEvents.Trial{iTrials}.States.DecisionWait(1) - stimGrab(iTrials); %end of stimulus period, relative to handle grab, from simons email
    catch
        stimTime(iTrials) = NaN;
        stimGrab(iTrials) = 0;
    end

    %check for spout motion
    if isfield(bhv.RawEvents.Trial{iTrials}.States,'MoveSpout')
        spoutTime(iTrials) = bhv.RawEvents.Trial{iTrials}.States.MoveSpout(1) - stimGrab(iTrials);

        %also get time when the other spout was moved out at
        if bhv.Rewarded(iTrials)
            spoutOutTime(iTrials) = bhv.RawEvents.Trial{iTrials}.States.Reward(1) - stimGrab(iTrials);
        else
            spoutOutTime(iTrials) = bhv.RawEvents.Trial{iTrials}.States.HardPunish(1) - stimGrab(iTrials);
        end
    else
        spoutTime(iTrials) = NaN;
        spoutOutTime(iTrials) = NaN;
    end

    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Port1In') %check for licks
        lickL{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Port1In;
        lickL{iTrials}(lickL{iTrials} < bhv.RawEvents.Trial{iTrials}.States.MoveSpout(1)) = []; %dont use false licks that occured before spouts were moved in
        lickL{iTrials} = lickL{iTrials} - stimGrab(iTrials);
    end
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Port3In') %check for right licks
        lickR{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Port3In;
        lickR{iTrials}(lickR{iTrials} < bhv.RawEvents.Trial{iTrials}.States.MoveSpout(1)) = []; %dont use false licks that occured before spouts were moved in
        lickR{iTrials} = lickR{iTrials} - stimGrab(iTrials);
    end

    % get stimulus events times
    audStimL{iTrials} = bhv.stimEvents{iTrials}{1} + stimTime(iTrials);
    audStimR{iTrials} = bhv.stimEvents{iTrials}{2} + stimTime(iTrials);
    tacStimL{iTrials} = bhv.stimEvents{iTrials}{5} + stimTime(iTrials);
    tacStimR{iTrials} = bhv.stimEvents{iTrials}{6} + stimTime(iTrials);

    leverIn(iTrials) = min(bhv.RawEvents.Trial{iTrials}.States.Reset(:)) - stimGrab(iTrials); %first reset state causes lever to move in

    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Wire2High') %check for left grabs
        levGrabL{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Wire2High - stimGrab(iTrials);
    end
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Wire1High') %check for right grabs
        levGrabR{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Wire1High - stimGrab(iTrials);
    end

    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Wire2Low') %check for left release
        levReleaseL{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Wire2Low - stimGrab(iTrials);
    end
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Wire1Low') %check for right release
        levReleaseR{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Wire1Low - stimGrab(iTrials);
    end

    if ~isnan(bhv.RawEvents.Trial{iTrials}.States.Reward(1)) %check for reward state
        water(iTrials) = bhv.RawEvents.Trial{iTrials}.States.Reward(1) - stimGrab(iTrials);
    end
end
maxSpoutRegs = length(min(round((PRE_HANDLE_SECONDS + spoutTime) * sRate)) : frames); %maximal number of required spout regressors

%% build regressors - create design matrix based on event times
%basic time regressors
timeR = logical(diag(ones(1,frames)));

lGrabR = cell(1,trialCnt);
lGrabRelR = cell(1,trialCnt);
rGrabR = cell(1,trialCnt);
rGrabRelR = cell(1,trialCnt);
lLickR = cell(1,trialCnt);
rLickR = cell(1,trialCnt);
leverInR = cell(1,trialCnt);

lfirstTacStimR = cell(1,trialCnt);
rfirstTacStimR = cell(1,trialCnt);
lfirstAudStimR = cell(1,trialCnt);
rfirstAudStimR = cell(1,trialCnt);

lTacStimR = cell(1,trialCnt);
rTacStimR = cell(1,trialCnt);
lAudStimR = cell(1,trialCnt);
rAudStimR = cell(1,trialCnt);

spoutR = cell(1,trialCnt);
spoutOutR = cell(1,trialCnt);

rewardR = cell(1,trialCnt);
ChoiceR = cell(1,trialCnt);



prevRewardR = cell(1,trialCnt);
prevChoiceR = cell(1,trialCnt);
prevStimR = cell(1,trialCnt);
nextChoiceR = cell(1,trialCnt);
repeatChoiceR = cell(1,trialCnt);

waterR = cell(1,trialCnt);
fastPupilR = cell(1,trialCnt);
slowPupilR = cell(1,trialCnt);

whiskR = cell(1,trialCnt);
noseR = cell(1,trialCnt);
piezoR = cell(1,trialCnt);
piezoMoveR = cell(1,trialCnt);
faceR = cell(1,trialCnt);
bodyR = cell(1,trialCnt);

handleSoundR = cell(1,trialCnt);

%%
tic
for iTrials = 1:trialCnt
    %% first tactile/auditory stimuli
    lfirstAudStimR{iTrials} = false(frames, fsPostTime);
    rfirstAudStimR{iTrials} = false(frames, fsPostTime);
    lfirstTacStimR{iTrials} = false(frames, fsPostTime);
    rfirstTacStimR{iTrials} = false(frames, fsPostTime);

    firstStim = NaN;
    if bhv.StimType(iTrials) == 2 || bhv.StimType(iTrials) == 6 %auditory or mixed stimulus
        if ~isempty(audStimL{iTrials}(~isnan(audStimL{iTrials})))
            firstStim = round((audStimL{iTrials}(1) + PRE_HANDLE_SECONDS) * sRate);
            stimEnd = firstStim - 1 + fsPostTime; stimEnd = min([frames stimEnd]);
            lfirstAudStimR{iTrials}(:,1 : stimEnd - firstStim + 1)  = timeR(:, firstStim : stimEnd);
        end
        if ~isempty(audStimR{iTrials}(~isnan(audStimR{iTrials})))
            firstStim = round((audStimR{iTrials}(1) + PRE_HANDLE_SECONDS) * sRate);
            stimEnd = firstStim - 1 + fsPostTime; stimEnd = min([frames stimEnd]);
            rfirstAudStimR{iTrials}(:,1 : stimEnd - firstStim + 1) = timeR(:, firstStim : stimEnd);
        end
    end

    if bhv.StimType(iTrials) == 4 || bhv.StimType(iTrials) == 6 %tactile or mixed stimulus
        if ~isempty(tacStimL{iTrials}(~isnan(tacStimL{iTrials})))
            firstStim = round((tacStimL{iTrials}(1) + PRE_HANDLE_SECONDS) * sRate);
            stimEnd = firstStim - 1 + fsPostTime; stimEnd = min([frames stimEnd]);
            lfirstTacStimR{iTrials}(:,1 : stimEnd - firstStim + 1) = timeR(:, firstStim : stimEnd);
        end
        if ~isempty(tacStimR{iTrials}(~isnan(tacStimR{iTrials})))
            firstStim = round((tacStimR{iTrials}(1) + PRE_HANDLE_SECONDS) * sRate);
            stimEnd = firstStim - 1 + fsPostTime; stimEnd = min([frames stimEnd]);
            rfirstTacStimR{iTrials}(:,1 : stimEnd - firstStim + 1) = timeR(:, firstStim : stimEnd);
        end
    end



    %% other tactile/auditory stimuli
    lAudStimR{iTrials} = false(frames, sPostTime);
    rAudStimR{iTrials} = false(frames, sPostTime);
    lTacStimR{iTrials} = false(frames, sPostTime);
    rTacStimR{iTrials} = false(frames, sPostTime);

    for iRegs = 0 : sPostTime - 1
        allStims = audStimL{iTrials}(2:end) + (iRegs * 1/sRate);
        lAudStimR{iTrials}(logical(histcounts(allStims,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)),iRegs+1) = 1;

        allStims = audStimR{iTrials}(2:end) + (iRegs * 1/sRate);
        rAudStimR{iTrials}(logical(histcounts(allStims,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)),iRegs+1) = 1;

        allStims = tacStimL{iTrials}(2:end) + (iRegs * 1/sRate);
        lTacStimR{iTrials}(logical(histcounts(allStims,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)),iRegs+1) = 1;

        allStims = tacStimR{iTrials}(2:end) + (iRegs * 1/sRate);
        rTacStimR{iTrials}(logical(histcounts(allStims,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)),iRegs+1) = 1;
    end



    %% spout regressors
    spoutIdx = round((PRE_HANDLE_SECONDS + spoutTime(iTrials)) * sRate) : round((PRE_HANDLE_SECONDS + POST_HANDLE_SECONDS) * sRate); %index for which part of the trial should be covered by spout regressors
    spoutR{iTrials} = false(frames, maxSpoutRegs);
    if ~isnan(spoutTime(iTrials))
        spoutR{iTrials}(:, 1:length(spoutIdx)) = timeR(:, spoutIdx);
    end

    spoutOutR{iTrials} = false(frames, 3);
    spoutOut = round((PRE_HANDLE_SECONDS + spoutOutTime(iTrials)) * sRate); %time when opposing spout moved out again
    if ~isnan(spoutOut) && spoutOut < (frames + 1)
        cInd = spoutOut : spoutOut + 2; cInd(cInd > frames) = [];
        temp = diag(ones(1,3));
        spoutOutR{iTrials}(cInd, :) = temp(1:length(cInd),:);
    end


    %% lick regressors
    lLickR{iTrials} = false(frames, length(motorIdx));
    rLickR{iTrials} = false(frames, length(motorIdx));

    for iRegs = 0 : length(motorIdx)-1
        licks = lickL{iTrials} - ((mPreTime/sRate) - (iRegs * 1/sRate));
        lLickR{iTrials}(logical(histcounts(licks,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)),iRegs+1) = 1;

        licks = lickR{iTrials} - ((mPreTime/sRate) - (iRegs * 1/sRate));
        rLickR{iTrials}(logical(histcounts(licks,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)),iRegs+1) = 1;
    end

    %% lever in
    leverInR{iTrials} = false(frames, leverMoveDur);
    leverShift = round((PRE_HANDLE_SECONDS + leverIn(iTrials))* sRate); %timepoint in frames when lever moved in, relative to lever grab

    if ~isnan(leverShift)
        if leverShift > 0 %lever moved in during the recorded trial
            leverInR{iTrials}(leverShift : leverShift + leverMoveDur -1, :) = diag(ones(1, leverMoveDur));
        elseif (leverShift + leverMoveDur) > 0  %lever was moving before data was recorded but still moving at trial onset
            leverInR{iTrials}(1 : leverMoveDur + leverShift, :) = [zeros(leverMoveDur + leverShift, abs(leverShift)) diag(ones(1, leverMoveDur + leverShift))];
        end
    end

    %% dual-handle indicator sound
    handleSoundR{iTrials} = false(frames, sPostTime);
    for iRegs = 0 : sPostTime - 1
        allStims = handleSounds{iTrials}(1:end) + (iRegs * 1/sRate);
        allStims = allStims(~isnan(allStims)) + 0.001;
        handleSoundR{iTrials}(logical(histcounts(allStims,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)),iRegs+1) = 1;
    end

    %% choice and reward
    stimTemp = false(frames, frames + maxStimShift);
    stimShift = round(stimTime(iTrials) * sRate); %amount of stimshift compared to possible maximum. move diagonal on x-axis accordingly.

    if (stimShift > maxStimShift) || isnan(stimTime(iTrials))
        stimTemp = NaN(frames, frames + maxStimShift); %don't use trial if stim onset is too late
    else
        stimTemp(:, end - stimShift - frames + 1 : end - stimShift) = timeR;
    end

    rewardR{iTrials} = false(size(stimTemp));

    if bhv.Rewarded(iTrials) %rewarded
        rewardR{iTrials} = stimTemp; %trial was rewarded
    end

    % get L/R choices as binary design matrix
    ChoiceR{iTrials} = false(size(stimTemp));
    if bhv.ResponseSide(iTrials) == 1 %IF LEFT CHOICE!!! Left choice is 1, right is 2
        ChoiceR{iTrials} = stimTemp;
    end


    % previous trial regressors
    if iTrials == 1 %don't use first trial
        prevRewardR{iTrials} = NaN(size(timeR(:,1:end-4)));
        prevChoiceR{iTrials} = NaN(size(timeR(:,1:end-4)));
        prevStimR{iTrials} = NaN(size(timeR(:,1:end-4)));

    else %for all subsequent trials
        % same as for regular choice regressors but for prevoious trial
        prevChoiceR{iTrials} = false(size(timeR(:,1:end-4)));
        if SessionData.ResponseSide(bTrials(iTrials)-1) == 1
            prevChoiceR{iTrials} = timeR(:,1:end-4);
        end

        prevStimR{iTrials} = false(size(timeR(:,1:end-4)));
        if SessionData.CorrectSide(bTrials(iTrials)-1) == 1 % if previous trial had a left target
            prevStimR{iTrials} = timeR(:,1:end-4);
        end

        prevRewardR{iTrials} = false(size(timeR(:,1:end-4)));
        if SessionData.Rewarded(bTrials(iTrials)-1) %last trial was rewarded
            prevRewardR{iTrials} = timeR(:,1:end-4);
        end
    end

    % subsequent trial regressors
    if iTrials == length(bTrials) %don't use lat trial
        nextChoiceR{iTrials} = NaN(size(timeR(:,1:end-4)));
        repeatChoiceR{iTrials} = NaN(size(timeR(:,1:end-4)));

    else %for all subsequent trials
        nextChoiceR{iTrials} = false(size(timeR(:,1:end-4)));
        if SessionData.ResponseSide(bTrials(iTrials)+1) == 1 %choice in next trial is left
            nextChoiceR{iTrials} = timeR(:,1:end-4);
        end

        repeatChoiceR{iTrials} = false(size(timeR(:,1:end-4)));
        if SessionData.ResponseSide(bTrials(iTrials)) == SessionData.ResponseSide(bTrials(iTrials)+1) %choice in next trial is similar to the current one
            repeatChoiceR{iTrials} = timeR(:,1:end-4);
        end
    end

    %determine timepoint of reward given
    waterR{iTrials} = false(frames, sRate * 2);
    if ~isnan(water(iTrials)) && ~isempty(water(iTrials))
        waterOn = round((PRE_HANDLE_SECONDS + water(iTrials)) * sRate); %timepoint in frames when reward was given
        if waterOn <= frames
            waterR{iTrials}(:, 1 : size(timeR,2) - waterOn + 1) = timeR(:, waterOn:end);
        end
    end

    %% lever grabs
    cGrabs = levGrabL{iTrials};
    cGrabs(cGrabs >= POST_HANDLE_SECONDS) = []; %remove grabs after end of imaging
    cGrabs(find(diff(cGrabs) < TAPDUR) + 1) = []; %remove grabs that are too close to one another
    lGrabR{iTrials} = histcounts(cGrabs,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)'; %convert to binary trace

    cGrabs = levGrabR{iTrials};
    cGrabs(cGrabs >= POST_HANDLE_SECONDS) = []; %remove grabs after end of imaging
    cGrabs(find(diff(cGrabs) < TAPDUR) + 1) = []; %remove grabs that are too close to one another
    rGrabR{iTrials} = histcounts(cGrabs,-PRE_HANDLE_SECONDS:1/sRate:POST_HANDLE_SECONDS)'; %convert to binary trace

    %% pupil / whisk / nose / face / body regressors
    bhvFrameRate = round(1/mean(diff(pTime{bTrials(iTrials)}))); %framerate of face camera
    trialOn = bhv.TrialStartTime(iTrials) + (stimGrab(iTrials) - PRE_HANDLE_SECONDS);
    trialTime = pTime{bTrials(iTrials)} - trialOn;
    rejIdx = trialTime < trialDur; %don't use late frames
    trialTime = trialTime(rejIdx);

    if isempty(trialTime) || trialTime(1) > 0 %check if there is missing time at the beginning of a trial
        warning(['Trial ' int2str(bTrials(iTrials)) ': Missing behavioral video frames at trial onset. Trial removed from analysis']);
        fastPupilR{iTrials} = NaN(frames, 1);
        slowPupilR{iTrials} = NaN(frames, 1);
        whiskR{iTrials} = NaN(frames, 1);
        noseR{iTrials} = NaN(frames, 1);
        faceR{iTrials} = NaN(frames, 1);
        bodyR{iTrials} = NaN(frames, 1);

    else
        timeLeft = trialDur - trialTime(end); %check if there is missing time at the end of a trial
        if (timeLeft < trialDur * 0.9) && (timeLeft > 0) %if there is some time missing to make a whole trial
            addTime = trialTime(end) + (1/bhvFrameRate : 1/bhvFrameRate : timeLeft + 1/bhvFrameRate); %add some dummy times to make complete trial
            trialTime = [trialTime' addTime];
        end

        fastPupilR{iTrials} = Behavior_vidResamp(fPupil{bTrials(iTrials)}(rejIdx), trialTime, sRate);
        fastPupilR{iTrials} = smooth(fastPupilR{iTrials}(end - frames + 1 : end), 'rlowess');

        slowPupilR{iTrials} = Behavior_vidResamp(sPupil{bTrials(iTrials)}(rejIdx), trialTime, sRate);
        slowPupilR{iTrials} =  smooth(slowPupilR{iTrials}(end - frames + 1 : end), 'rlowess');

        whiskR{iTrials} = Behavior_vidResamp(whisker{bTrials(iTrials)}(rejIdx), trialTime, sRate);
        whiskR{iTrials} = smooth(whiskR{iTrials}(end - frames + 1 : end), 'rlowess');

        noseR{iTrials} = Behavior_vidResamp(nose{bTrials(iTrials)}(rejIdx), trialTime, sRate);
        noseR{iTrials} = smooth(noseR{iTrials}(end - frames + 1 : end), 'rlowess');

        faceR{iTrials} = Behavior_vidResamp(faceM{bTrials(iTrials)}(rejIdx), trialTime, sRate);
        faceR{iTrials} = smooth(faceR{iTrials}(end - frames + 1 : end), 'rlowess');


        % body regressors
        bhvFrameRate = round(1/mean(diff(bTime{bTrials(iTrials)}))); %framerate of body camera
        trialTime = bTime{bTrials(iTrials)} - trialOn;
        rejIdx = trialTime < trialDur; %don't use late frames
        trialTime = trialTime(rejIdx);
        timeLeft = trialDur - trialTime(end); %check if there is missing time at the end of a trial

        if (timeLeft < trialDur * 0.9) && (timeLeft > 0) %if there is some time missing to make a whole trial
            addTime = trialTime(end) + (1/bhvFrameRate : 1/bhvFrameRate : timeLeft + 1/bhvFrameRate); %add some dummy times to make complete trial
            trialTime = [trialTime' addTime];
        end

        bodyR{iTrials} = Behavior_vidResamp(bodyM{bTrials(iTrials)}(rejIdx), trialTime, sRate);
        bodyR{iTrials} = smooth(bodyR{iTrials}(end - frames + 1 : end), 'rlowess');
    end

    %% piezo sensor information

    if exist([cPath 'Analog_'  num2str(trials(iTrials)) '.dat'],'file') ~= 2  %check if files exists on hdd and pull from server otherwise
        cFile = dir([sPath 'Analog_'  num2str(trials(iTrials)) '.dat']);
        copyfile([sPath 'Analog_'  num2str(trials(iTrials)) '.dat'],[cPath 'Analog_'  num2str(trials(iTrials)) '.dat']);
    end
    [~,Analog] = Widefield_LoadData([cPath 'Analog_'  num2str(trials(iTrials)) '.dat'],'Analog'); %load analog data
    stimOn = find(diff(double(Analog(STIMLINE,:)) > 1500) == 1); %find stimulus onset in current trial


    if ~isnan(stimTime(iTrials))
        try
            Analog(1,round(stimOn + ((POST_HANDLE_SECONDS-stimTime(iTrials)) * 1000) - 1)) = 0; %make sure there are enough datapoints in analog signal
            temp = Analog(PIEZOLINE,round(stimOn - ((PRE_HANDLE_SECONDS + stimTime(iTrials)) * 1000)) : round(stimOn + ((POST_HANDLE_SECONDS - stimTime(iTrials))* 1000) - 1)); % data from piezo sensor. Should encode animals hindlimb motion.
            temp = smooth(double(temp), sRate*5, 'lowess')'; %do some smoothing
            temp = [repmat(temp(1),1,1000) temp repmat(temp(end),1,1000)]; %add some padding on both sides to avoid edge effects when resampling
            temp = resample(double(temp), sRate, 1000); %resample to imaging rate
            piezoR{iTrials} = temp(sRate + 1 : end - sRate)'; %remove padds again
            piezoR{iTrials} = piezoR{iTrials}(end - frames + 1:end); %make sure, the length is correct

            temp = abs(hilbert(diff(piezoR{iTrials})));
            piezoMoveR{iTrials} = [temp(1); temp]; %keep differential motion signal
            clear temp
        catch
            piezoMoveR{iTrials} = NaN(frames, 1);
            piezoR{iTrials} = NaN(frames, 1);
        end
    else
        piezoMoveR{iTrials} = NaN(frames, 1);
        piezoR{iTrials} = NaN(frames, 1);
    end

    % give some feedback over progress
    if rem(iTrials,50) == 0
        fprintf(1, 'Current trial is %d out of %d\n', iTrials,trialCnt);
        toc
    end
end

%% get proper design matrices for handle grab
lGrabR = cat(1,lGrabR{:});
lGrabR = Widefield_analogToDesign(lGrabR, 0.5, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT); %get design matrix

rGrabR = cat(1,rGrabR{:});
rGrabR = Widefield_analogToDesign(rGrabR, 0.5, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT); %get design matrix

%% rebuild analog motor regressors to get proper design matrices
temp = double(cat(1,fastPupilR{:}));
temp = (temp - prctile(temp,1))./ nanstd(temp); %minimum values are at 0, signal in standard deviation units
[dMat, traceOut] = Widefield_analogToDesign(temp, median(temp), trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);
[dMat2, ~] = Widefield_analogToDesign(traceOut, prctile(traceOut,75), trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);

fastPupilAnalog = single(traceOut);
fastPupilDigital = cat(1,dMat{:});
fastPupilHiDigital = cat(1,dMat2{:});

[dMat, traceOut] = Widefield_analogToDesign(double(cat(1,whiskR{:})), 0.5, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);
[dMat2, ~] = Widefield_analogToDesign(double(cat(1,whiskR{:})), 2, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);

whiskAnalog = single(traceOut);
whiskDigital = cat(1,dMat{:});
whiskHiDigital = cat(1,dMat2{:});

[dMat, traceOut] = Widefield_analogToDesign(double(cat(1,noseR{:})), 0.5, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);
[dMat2, ~] = Widefield_analogToDesign(double(cat(1,noseR{:})), 2, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);

noseAnalog = single(traceOut);
noseDigital = cat(1,dMat{:});
noseHiDigital = cat(1,dMat2{:});

[dMat, traceOut] = Widefield_analogToDesign(double(cat(1,piezoR{:})), 2, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);
[dMat2, traceOut2] = Widefield_analogToDesign(double(cat(1,piezoMoveR{:})), 0.5, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);
[dMat3, ~] = Widefield_analogToDesign(double(cat(1,piezoMoveR{:})), 2, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);

piezoAnalog = traceOut;
piezoDigital = cat(1,dMat{:});
piezoMoveAnalog = traceOut2;
piezoMoveDigital = cat(1,dMat2{:});
piezoMoveHiDigital = cat(1,dMat3{:});

[dMat, traceOut] = Widefield_analogToDesign(double(cat(1,faceR{:})), 0.5, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);
[dMat2, ~] = Widefield_analogToDesign(double(cat(1,faceR{:})), 2, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);

faceAnalog = single(traceOut);
faceDigital = cat(1,dMat{:});
faceHiDigital = cat(1,dMat2{:});

[dMat, traceOut] = Widefield_analogToDesign(double(cat(1,bodyR{:})), 0.5, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);
[dMat2, ~] = Widefield_analogToDesign(double(cat(1,bodyR{:})), 2, trialCnt, sRate, sRate, motorIdx, GAUSSIAN_SHIFT);

bodyAnalog = single(traceOut);
bodyDigital = cat(1,dMat{:});
bodyHiDigital = cat(1,dMat2{:});

clear piezoR1 piezoR2 dMat dMat2 traceOut temp

%% re-align behavioral video data and Vc to lever grab instead of stimulus onset

iiSpikeFrames = findInterictalSpikes(U, Vc, 2, false); %find interictal spikes
Vc = interpOverInterictal(Vc, iiSpikeFrames); %interpolate over interictal spikes

V1 = reshape(V1, [], SessionData.nTrials, BHV_DIM_COUNT);
V2 = reshape(V2, [], SessionData.nTrials, BHV_DIM_COUNT);

%if video sampling rate is different from widefield, resample video data
if bhvOpts.targRate ~= sRate
    vidR = NaN(size(Vc,2), length(bTrials), size(V1,3), 'single');
    moveR = NaN(size(Vc,2), length(bTrials), size(V1,3), 'single');
    for iTrials = 1 : length(bTrials)

        temp1 = squeeze(V1(:,bTrials(iTrials),:));
        if ~any(isnan(temp1(:)))
            trialTime = 1/bhvRate : 1/bhvRate : size(Vc,2)/sRate;
            vidR(:, iTrials, :) = Behavior_vidResamp(double(temp1), trialTime, sRate);

            temp2 = squeeze(V2(:,bTrials(iTrials),:));
            trialTime = 1/bhvRate : 1/bhvRate : size(Vc,2)/sRate;
            moveR(:, iTrials, :) = Behavior_vidResamp(double(temp2), trialTime, sRate);
        end
    end
else
    vidR = V1(:,bTrials,:); clear V1 %get correct trials from behavioral video data.
    moveR = V2(:,bTrials,:); clear V2 %get correct trials from behavioral video data.
end

% re-align video data
temp1 = NaN(DIMS,frames,trialCnt);
temp2 = NaN(frames,trialCnt,BHV_DIM_COUNT);
temp3 = NaN(frames,trialCnt,BHV_DIM_COUNT);
temp4 = NaN(2,frames,trialCnt);
for x = 1 : size(vidR,2) %iterate thru trials
    try
        temp1(:,:,x) = Vc(:,(shVal - ceil(stimTime(x) / (1/sRate))) - (PRE_HANDLE_SECONDS * sRate) : (shVal - ceil(stimTime(x) / (1/sRate))) + (POST_HANDLE_SECONDS * sRate) - 1,x);
        temp2(:,x,:) = vidR((shVal - ceil(stimTime(x) / (1/sRate))) - (PRE_HANDLE_SECONDS * sRate) : (shVal - ceil(stimTime(x) / (1/sRate))) + (POST_HANDLE_SECONDS * sRate) - 1,x,:);
        temp3(:,x,:) = moveR((shVal - ceil(stimTime(x) / (1/sRate))) - (PRE_HANDLE_SECONDS * sRate) : (shVal - ceil(stimTime(x) / (1/sRate))) + (POST_HANDLE_SECONDS * sRate) - 1,x,:);
    catch
        fprintf(1,'Could not align trial %d. Relative stim time: %fs\n', x, stimTime(x));
    end
end
Vc = reshape(temp1,DIMS,[]); clear temp1
vidR = reshape(temp2,[],BHV_DIM_COUNT); clear temp2
moveR = reshape(temp3,[],BHV_DIM_COUNT); clear temp3

clear temp4


%% reshape regressors, make design matrix and indices for regressors that are used for the model
timeR = repmat(logical(diag(ones(1,frames))),trialCnt,1); %time regressor
timeR = timeR(:,1:end-4);

lGrabR = cat(1,lGrabR{:});
lGrabRelR = cat(1,lGrabRelR{:}); %not in the model currently
rGrabR = cat(1,rGrabR{:});
rGrabRelR = cat(1,rGrabRelR{:}); %not in the model currently

lLickR = cat(1,lLickR{:});
rLickR = cat(1,rLickR{:});
leverInR = cat(1,leverInR{:});
leverInR(:,sum(leverInR) == 0) = [];

handleSoundR = cat(1,handleSoundR{:});

lTacStimR = cat(1,lTacStimR{:});
rTacStimR = cat(1,rTacStimR{:});
lAudStimR = cat(1,lAudStimR{:});
rAudStimR = cat(1,rAudStimR{:});

lfirstTacStimR = cat(1,lfirstTacStimR{:});
rfirstTacStimR = cat(1,rfirstTacStimR{:});
lfirstAudStimR = cat(1,lfirstAudStimR{:});
rfirstAudStimR = cat(1,rfirstAudStimR{:});

spoutR = cat(1,spoutR{:});
spoutOutR = cat(1,spoutOutR{:});
spoutR(:,sum(spoutR) == 0) = [];
spoutOutR(:,sum(spoutOutR) == 0) = [];

rewardR = cat(1,rewardR{:});
prevRewardR = cat(1,prevRewardR{:});

ChoiceR = cat(1,ChoiceR{:});

prevChoiceR = cat(1,prevChoiceR{:});
prevStimR = cat(1,prevStimR{:});
nextChoiceR = cat(1,nextChoiceR{:});
repeatChoiceR = cat(1,repeatChoiceR{:});

waterR = cat(1,waterR{:});

slowPupilR = cat(1,slowPupilR{:});
slowPupilR(~isnan(slowPupilR(:,1)),:) = zscore(slowPupilR(~isnan(slowPupilR(:,1)),:));

%% create full design matrix

fullR = [timeR ChoiceR rewardR lGrabR rGrabR  ...
    lLickR rLickR handleSoundR lfirstTacStimR lTacStimR rfirstTacStimR rTacStimR ...
    lfirstAudStimR lAudStimR rfirstAudStimR rAudStimR prevRewardR prevChoiceR ...
    nextChoiceR waterR piezoAnalog piezoDigital piezoMoveAnalog piezoMoveDigital piezoMoveHiDigital whiskAnalog whiskDigital whiskHiDigital noseAnalog noseDigital noseHiDigital fastPupilAnalog fastPupilDigital fastPupilHiDigital ...
    slowPupilR faceAnalog faceDigital faceHiDigital bodyAnalog bodyDigital bodyHiDigital moveR vidR];



% labels for different regressor sets. It is REALLY important this agrees with the order of regressors in fullR.
regLabels = {
    'time' 'Choice' 'reward' 'lGrab' 'rGrab' 'lLick' 'rLick' 'handleSound' ...
    'lfirstTacStim' 'lTacStim' 'rfirstTacStim' 'rTacStim' 'lfirstAudStim' 'lAudStim' 'rfirstAudStim' 'rAudStim' ...
    'prevReward' 'prevChoice' 'nextChoice' 'water' 'piezoAnalog' 'piezoDigital' 'piezoMoveAnalog' 'piezoMoveDigital' 'piezoMoveHiDigital' 'whiskAnalog' 'whiskDigital' 'whiskHiDigital' 'noseAnalog' 'noseDigital' 'noseHiDigital' 'fastPupilAnalog' 'fastPupilDigital' 'fastPupilHiDigital' 'slowPupil' 'faceAnalog' 'faceDigital' 'faceHiDigital' 'bodyAnalog' 'bodyDigital' 'bodyHiDigital' 'Move' 'bhvVideo'};

%index to reconstruct different response kernels
regIdx = [
    ones(1,size(timeR,2))*find(ismember(regLabels,'time')) ...
    ones(1,size(ChoiceR,2))*find(ismember(regLabels,'Choice')) ...
    ones(1,size(rewardR,2))*find(ismember(regLabels,'reward')) ...
    ones(1,size(lGrabR,2))*find(ismember(regLabels,'lGrab')) ...
    ones(1,size(rGrabR,2))*find(ismember(regLabels,'rGrab')) ...
    ones(1,size(lLickR,2))*find(ismember(regLabels,'lLick')) ...
    ones(1,size(rLickR,2))*find(ismember(regLabels,'rLick')) ...
    ones(1,size(handleSoundR,2))*find(ismember(regLabels,'handleSound')) ...
    ones(1,size(lfirstTacStimR,2))*find(ismember(regLabels,'lfirstTacStim')) ...
    ones(1,size(lTacStimR,2))*find(ismember(regLabels,'lTacStim')) ...
    ones(1,size(rfirstTacStimR,2))*find(ismember(regLabels,'rfirstTacStim')) ...
    ones(1,size(rTacStimR,2))*find(ismember(regLabels,'rTacStim')) ...
    ones(1,size(lfirstAudStimR,2))*find(ismember(regLabels,'lfirstAudStim')) ...
    ones(1,size(lAudStimR,2))*find(ismember(regLabels,'lAudStim')) ...
    ones(1,size(rfirstAudStimR,2))*find(ismember(regLabels,'rfirstAudStim')) ...
    ones(1,size(rAudStimR,2))*find(ismember(regLabels,'rAudStim')) ...
    ones(1,size(prevRewardR,2))*find(ismember(regLabels,'prevReward')) ...
    ones(1,size(prevChoiceR,2))*find(ismember(regLabels,'prevChoice')) ...
    ones(1,size(nextChoiceR,2))*find(ismember(regLabels,'nextChoice')) ...
    ones(1,size(waterR,2))*find(ismember(regLabels,'water')) ...
    ones(1,size(piezoAnalog,2))*find(ismember(regLabels,'piezoAnalog')) ...
    ones(1,size(piezoDigital,2))*find(ismember(regLabels,'piezoDigital')) ...
    ones(1,size(piezoMoveAnalog,2))*find(ismember(regLabels,'piezoMoveAnalog')) ...
    ones(1,size(piezoMoveDigital,2))*find(ismember(regLabels,'piezoMoveDigital')) ...
    ones(1,size(piezoMoveHiDigital,2))*find(ismember(regLabels,'piezoMoveHiDigital')) ...
    ones(1,size(whiskAnalog,2))*find(ismember(regLabels,'whiskAnalog')) ...
    ones(1,size(whiskDigital,2))*find(ismember(regLabels,'whiskDigital')) ...
    ones(1,size(whiskHiDigital,2))*find(ismember(regLabels,'whiskHiDigital')) ...
    ones(1,size(noseAnalog,2))*find(ismember(regLabels,'noseAnalog')) ...
    ones(1,size(noseDigital,2))*find(ismember(regLabels,'noseDigital')) ...
    ones(1,size(noseHiDigital,2))*find(ismember(regLabels,'noseHiDigital')) ...
    ones(1,size(fastPupilAnalog,2))*find(ismember(regLabels,'fastPupilAnalog')) ...
    ones(1,size(fastPupilDigital,2))*find(ismember(regLabels,'fastPupilDigital')) ...
    ones(1,size(fastPupilHiDigital,2))*find(ismember(regLabels,'fastPupilHiDigital')) ...
    ones(1,size(slowPupilR,2))*find(ismember(regLabels,'slowPupil')) ...
    ones(1,size(faceAnalog,2))*find(ismember(regLabels,'faceAnalog')) ...
    ones(1,size(faceDigital,2))*find(ismember(regLabels,'faceDigital')) ...
    ones(1,size(faceHiDigital,2))*find(ismember(regLabels,'faceHiDigital')) ...
    ones(1,size(bodyAnalog,2))*find(ismember(regLabels,'bodyAnalog')) ...
    ones(1,size(bodyDigital,2))*find(ismember(regLabels,'bodyDigital')) ...
    ones(1,size(bodyHiDigital,2))*find(ismember(regLabels,'bodyHiDigital')) ...
    ones(1,size(moveR,2))*find(ismember(regLabels,'Move')) ...
    ones(1,size(vidR,2))*find(ismember(regLabels,'bhvVideo'))];
%% compute the indices of the event that we're aligning to - for easier recovery of kernels

preStimFrames = PRE_HANDLE_SECONDS*sRate;
postStimFrames = POST_HANDLE_SECONDS*sRate;

timeZero = zeros(1,size(timeR,2)); timeZero(preStimFrames) = 1; %handle grab time
choiceZero = zeros(1,size(ChoiceR,2)); choiceZero(preStimFrames + maxStimShift) = 1; %stim on time choice is realigned to stimulus
rewardZero = zeros(1,size(rewardR,2)); rewardZero(preStimFrames + maxStimShift) = 1; %stim on time, same alignment as choice
lGrabZero = zeros(1,size(lGrabR,2)); lGrabZero(mPreTime + 1) = 1; %frame of actual motor event
rGrabZero = zeros(1,size(rGrabR,2)); rGrabZero(mPreTime + 1) = 1;
lLickZero = zeros(1,size(lLickR,2)); lLickZero(mPreTime + 1) = 1;
rLickZero = zeros(1,size(rLickR,2)); rLickZero(mPreTime + 1) = 1;
handleSoundZero = zeros(1,size(handleSoundR,2)); handleSoundZero(1) = 1; %first frame marks event
lfirstTacStimZero = zeros(1,size(lfirstTacStimR,2)); lfirstTacStimZero(1) = 1; %first frame marks event
lTacStimZero = zeros(1,size(lTacStimR,2)); lTacStimZero(1) = 1;
rfirstTacStimZero = zeros(1,size(rfirstTacStimR,2)); rfirstTacStimZero(1) = 1;
rTacStimZero = zeros(1,size(rTacStimR,2)); rTacStimZero(1) = 1;
lfirstAudStimZero = zeros(1,size(lfirstAudStimR,2)); lfirstAudStimZero(1) = 1; %first frame marks event
lAudStimZero = zeros(1,size(lAudStimR,2)); lAudStimZero(1) = 1;
rfirstAudStimZero = zeros(1,size(rfirstAudStimR,2)); rfirstAudStimZero(1) = 1;
rAudStimZero = zeros(1,size(rAudStimR,2)); rAudStimZero(1) = 1;
prevRewardZero = zeros(1,size(prevRewardR,2)); prevRewardZero(preStimFrames) = 1; %previous trial regressors are aligned to handle grab on current trial
prevChoiceZero = zeros(1,size(prevChoiceR,2)); prevChoiceZero(preStimFrames) = 1;
nextChoiceZero = zeros(1,size(nextChoiceR,2)); nextChoiceZero(preStimFrames) = 1;
waterZero = zeros(1,size(waterR,2)); waterZero(1) = 1; %begins at water delivery, lots of trials are cut off by trial end
piezoAnalogZero = zeros(1,size(piezoAnalog,2)); %analog regressors have no alignment value
piezoDigitalZero = zeros(1,size(piezoDigital,2)); piezoDigitalZero(mPreTime + 1) = 1;
piezoMoveAnalogZero = zeros(1,size(piezoMoveAnalog,2));
piezoMoveDigitalZero = zeros(1,size(piezoMoveDigital,2)); piezoMoveDigitalZero(mPreTime + 1) = 1;
piezoMoveHiDigitalZero = zeros(1,size(piezoMoveHiDigital,2)); piezoMoveHiDigitalZero(mPreTime + 1) = 1;
whiskAnalogZero = zeros(1,size(whiskAnalog,2));
whiskDigitalZero = zeros(1,size(whiskDigital,2)); whiskDigitalZero(mPreTime + 1) = 1;
whiskHiDigitalZero = zeros(1,size(whiskHiDigital,2)); whiskHiDigitalZero(mPreTime + 1) = 1;
noseAnalogZero = zeros(1,size(noseAnalog,2));
noseDigitalZero = zeros(1,size(noseDigital,2)); noseDigitalZero(mPreTime + 1) = 1;
noseHiDigitalZero = zeros(1,size(noseHiDigital,2)); noseHiDigitalZero(mPreTime + 1) = 1;
fastPupilAnalogZero = zeros(1,size(fastPupilAnalog,2));
fastPupilDigitalZero = zeros(1,size(fastPupilDigital,2)); fastPupilDigitalZero(mPreTime + 1) = 1;
fastPupilHiDigitalZero = zeros(1,size(fastPupilHiDigital,2)); fastPupilHiDigitalZero(mPreTime + 1) = 1;
slowPupilZero = zeros(1,size(slowPupilR,2));
faceAnalogZero = zeros(1,size(faceAnalog,2));
faceDigitalZero = zeros(1,size(faceDigital,2)); faceDigitalZero(mPreTime + 1) = 1;
faceHiDigitalZero = zeros(1,size(faceHiDigital,2)); faceHiDigitalZero(mPreTime + 1) = 1;
bodyAnalogZero = zeros(1,size(bodyAnalog,2));
bodyDigitalZero = zeros(1,size(bodyDigital,2)); bodyDigitalZero(mPreTime + 1) = 1;
bodyHiDigitalZero = zeros(1,size(bodyHiDigital,2)); bodyHiDigitalZero(mPreTime + 1) = 1;
moveRZero = zeros(1,size(moveR,2));
vidRZero = zeros(1,size(vidR,2));

regZeroFrames = [timeZero choiceZero rewardZero lGrabZero rGrabZero lLickZero rLickZero handleSoundZero ...
    lfirstTacStimZero lTacStimZero rfirstTacStimZero rTacStimZero lfirstAudStimZero lAudStimZero ...
    rfirstAudStimZero rAudStimZero prevRewardZero prevChoiceZero nextChoiceZero waterZero piezoAnalogZero ...
    piezoDigitalZero piezoMoveAnalogZero  piezoMoveDigitalZero piezoMoveHiDigitalZero whiskAnalogZero ...
    whiskDigitalZero whiskHiDigitalZero noseAnalogZero noseDigitalZero noseHiDigitalZero fastPupilAnalogZero ...
    fastPupilDigitalZero fastPupilHiDigitalZero slowPupilZero faceAnalogZero faceDigitalZero faceHiDigitalZero ...
    bodyAnalogZero bodyDigitalZero bodyHiDigitalZero moveRZero vidRZero];

%%

% orthogonalize video against spout/handle movement
vidIdx = find(ismember(regIdx, find(ismember(regLabels,{'Move' 'bhvVideo'})))); %index for video regressors
trialIdx = ~isnan(mean(fullR(:,vidIdx),2)); %don't use trials that failed to contain behavioral video data
smallR = [leverInR spoutR spoutOutR];

for iRegs = 1 : length(vidIdx)
    Q = qr([smallR(trialIdx,:) fullR(trialIdx,vidIdx(iRegs))],0); %orthogonalize video against other regressors
    fullR(trialIdx,vidIdx(iRegs)) = Q(:,end); % transfer orthogonolized video regressors back to design matrix
end

% reject trials with broken regressors that contain NaNs
trialIdx = isnan(mean(fullR,2)); %don't use first trial or trials that failed to contain behavioral video data
fprintf(1, 'Rejected %d/%d trials for NaN entries in regressors\n', sum(trialIdx)/frames, trialCnt);
fullR(trialIdx,:) = []; %clear bad trials

Vc(:,trialIdx) = []; %clear bad trials
zeromeanVc = bsxfun(@minus, Vc, mean(Vc,2)); %should be zero-mean

regLabels;
fullR;


end


