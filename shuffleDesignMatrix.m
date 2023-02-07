function shuffledDesignMatrix = shuffleDesignMatrix(regLabels,regIdx,fullR,shuffleLabels)
shuffleLabelInds = find(ismember(regLabels,shuffleLabels)); %get all regressors except task vars
shuffleInds = find(ismember(regIdx,shuffleLabelInds));

for i = shuffleInds
    permutation = randperm(size(fullR,1));
    fullR(:,i) = fullR(permutation,i);
end
shuffledDesignMatrix = fullR;
end