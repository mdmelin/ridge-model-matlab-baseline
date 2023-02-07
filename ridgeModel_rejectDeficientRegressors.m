function [R, regLabels, regIdx, regZeroFrames, rejIdx] = ridgeModel_rejectDeficientRegressors(fullR,regLabels,regIdx,regZeroFrames)

rejIdx = false(1,size(fullR,2));


%reject rank deficient regressors
[~, fullQRR] = qr(bsxfun(@rdivide,fullR(:,~rejIdx),sqrt(sum(fullR(:,~rejIdx).^2))),0); %orthogonalize design matrix
%figure; plot(abs(diag(fullQRR))); ylim([0 1.1]); title('Regressor orthogonality'); drawnow; %this shows how orthogonal individual regressors are to the rest of the matrix
if sum(abs(diag(fullQRR)) > max(size(fullR(:,~rejIdx))) * eps(fullQRR(1))) < size(fullR(:,~rejIdx),2) %check if design matrix is full rank
    temp = ~(abs(diag(fullQRR)) > max(size(fullR(:,~rejIdx))) * eps(fullQRR(1))); %reject regressors that cause rank-defficint matrix
    rejIdx(~rejIdx) = temp;
    deficientLabels = unique(regLabels(regIdx(temp)));
    fprintf('WARNING: %s is at least partially rank deficient. \n', deficientLabels{:});
    fprintf(1, 'Rejected %d of %d total regressors for rank deficiency.\n', sum(temp),length(rejIdx));
else
    fprintf('No regressors were rank deficient!\n');
    temp = zeros(1,length(regIdx));
end


fullR(:,rejIdx) = []; %clear empty and rank deficient regressors
regIdx = regIdx(~rejIdx); %clear empty and rank deficient regressors

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
    fprintf('Fully discarded regressor: %s due to rank deficiency \n', discardLabels{:});
else
    fprintf('\nNo regressors were FULLY discarded due to rank deficiency\n');
end
regZeroFrames = regZeroFrames(~rejIdx);
R = fullR;
% regMarkers = [1 diff(regIdx)]; %marks the indices where regressors begin
% figure; hold on;plot(regMarkers); plot(rejIdx); legend('regMarkers','rejIdx');

% the following code doesn't work because the analog regressors don't have
% an alignment frame by nature
% savedAlignmentFrameLabels = regLabels(regIdx(find(regZeroFrames)));%check if the alignment event frames got rejected
% rejectedAlignmentFrameLabels = regLabels(~ismember(regLabels, savedAlignmentFrameLabels));
% fprintf('WARNING: The alignment frame for %s was rejected. \n', rejectedAlignmentFrameLabels{:});

end
