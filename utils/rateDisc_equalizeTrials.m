function useIdx = rateDisc_equalizeTrials(useIdx, targIdx, sTargIdx, maxTrials, dualCase)
% ensure that trials in 'useIdx' are equally distributed based on 'targIdx'
% and 'sTargIdx'. If set, only select up to 'maxTrials'. set 'dualCase' to
% also adjust trialcounts for 'sTargIdx' but allow their total count to be
% different (e.g. when there are many more correct as incorrect trials).
% For example, this will return a similar amount of correct and incorrect
% left/right choices but the amount of correct and incorrect trials can
% differ.

if ~exist('sTargIdx','var') || length(sTargIdx) ~= length(sTargIdx)
    sTargIdx = [];
end
if ~exist('maxTrials','var') || isempty(maxTrials)
    maxTrials = inf;
end
if ~exist('dualCase','var') || isempty(dualCase)
    dualCase = false;
end

rng('default');
cIdx = [];
if isempty(sTargIdx) || dualCase
    
    case1 = find(targIdx & useIdx);
    case2 = find(~targIdx & useIdx);
    
    caseCnt = min([min([length(case1) length(case2)]) round(maxTrials/2)]); %get minimal case and compare against max. trialcount
    useIdx(case1) = false; %don't use any trials from case1
    useIdx(case1(randperm(length(case1), caseCnt))) = true; %re-activate subselection of case1
    
    useIdx(case2) = false; %don't use any trials from case2
    useIdx(case2(randperm(length(case2), caseCnt))) = true; %re-activate subselection of case2
    
    if dualCase
        % trials where sTargIdx is is true
        case1 = find(targIdx & sTargIdx & useIdx);
        case2 = find(~targIdx & sTargIdx & useIdx);
        
        caseCnt = min([min([length(case1) length(case2)]) round(maxTrials/2)]); %get minimal case and compare against max. trialcount
        useIdx(case1) = false; %don't use any trials from case1
        useIdx(case1(randperm(length(case1), caseCnt))) = true; %re-activate subselection of case1
        
        useIdx(case2) = false; %don't use any trials from case2
        useIdx(case2(randperm(length(case2), caseCnt))) = true; %re-activate subselection of case2
        
        % trials where sTargIdx is is false
        case1 = find(targIdx & ~sTargIdx & useIdx);
        case2 = find(~targIdx & ~sTargIdx & useIdx);
        
        caseCnt = min([min([length(case1) length(case2)]) round(maxTrials/2)]); %get minimal case and compare against max. trialcount
        useIdx(case1) = false; %don't use any trials from case1
        useIdx(case1(randperm(length(case1), caseCnt))) = true; %re-activate subselection of case1
        
        useIdx(case2) = false; %don't use any trials from case2
        useIdx(case2(randperm(length(case2), caseCnt))) = true; %re-activate subselection of case2
    end
    
elseif length(sTargIdx) == length(sTargIdx) || ~dualCase
    
    case1 = find(targIdx & sTargIdx & useIdx);
    case2 = find(targIdx & ~sTargIdx & useIdx);
    case3 = find(~targIdx & ~sTargIdx & useIdx);
    case4 = find(~targIdx & sTargIdx & useIdx);
    
    caseCnt = min([min([length(case1) length(case2) length(case3) length(case4)]) round(maxTrials/4)]); %get minimal case and compare against max. trialcount
    
    useIdx(case1) = false; %don't use any trials from case1
    useIdx(case1(randperm(length(case1), caseCnt))) = true; %re-activate subselection of case1 
    
    useIdx(case2) = false; %don't use any trials from case2
    useIdx(case2(randperm(length(case2), caseCnt))) = true; %re-activate subselection of case2
    
    useIdx(case3) = false; %don't use any trials from case3
    useIdx(case3(randperm(length(case3), caseCnt))) = true; %re-activate subselection of case3
    
    useIdx(case4) = false; %don't use any trials from case4
    useIdx(case4(randperm(length(case4), caseCnt))) = true; %re-activate subselection of case4 
   
    end
end