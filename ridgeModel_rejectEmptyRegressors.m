function [R, regLabels, regIdx, regZeroFrames, rejIdx] = ridgeModel_rejectEmptyRegressors(fullR,regLabels,regIdx,regZeroFrames)

MIN_ENTRIES = 10;

rejIdx = false(1,size(fullR,2));

%reject empty regressors
rejIdx = nansum(abs(fullR)) < MIN_ENTRIES; %for analog regressors with less than min entries
fprintf(1, 'Rejected %d of %d total regressors for emptiness (less than %d entries).\n', sum(rejIdx),length(rejIdx), MIN_ENTRIES);
regIdx = regIdx(~rejIdx);

fullR(:,rejIdx) = []; %clear empty and rank deficient regressors

regLabelsOld = regLabels;
regLabels = regLabels(unique(regIdx));

temp = []; count = 1;
for i = unique(regIdx) %remove the skiped indices for discared reginds
    temp(regIdx == i) = count;
    count = count+1;
end
regIdx = temp;

%print out the regressors that were fully discarded
discardLabels = regLabelsOld(~ismember(regLabelsOld,regLabels));
if length(discardLabels) > 0
    fprintf('Fully discarded regressor: %s due to emptiness\n', discardLabels{:});
else
    fprintf('\nNo regressors were FULLY discarded due to emptiness\n');
end
regZeroFrames = regZeroFrames(~rejIdx);
R = fullR;
end
