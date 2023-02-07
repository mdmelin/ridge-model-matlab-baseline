function ridgeModel_saveResults(cPath,animal,rec, filename, Vm, zeromeanVc, U, R, betas, lambdas, cMap, cMovie, regLabels, regIdx, rejIdx, regZeroFrames)
TASK = 'SpatialDisc';

if ~strcmpi(filename(end-3:end),'.mat')
    filename = [filename '.mat'];
end
mkdir(cPath);
savepath = [cPath filesep filename];

save(savepath, 'Vm', 'zeromeanVc', 'U', 'R', 'betas', 'lambdas', 'cMap', 'cMovie', 'regIdx', 'regLabels', 'rejIdx', 'regZeroFrames');
end