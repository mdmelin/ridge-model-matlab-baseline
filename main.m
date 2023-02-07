clc;clear all;close all;
%% Get the animals and sessions

addpath([pwd filesep 'utils'])
DATAPATH = 'C:\Data\churchland\scratch\wfield_data\'; %path to the data
ANIMAL = 'mSM63';
SESSION = '09-Aug-2018';
NFOLDS = 10;
NFRAMES = 75; % number of frames per trial
MODEL_SAVEPATH = [pwd filesep 'saved_models'];
MODELNAME = 'demo_model'; % name to save the model with

%% Train the model

[regLabels,regIdx,R,regZeroFrames,zeromeanVc,U] = ridgeModel_returnDesignMatrix(DATAPATH, ANIMAL, SESSION);

% R is the design matrix
% regIdx specifies the regressor identiy of each column in the design
% regLabels gives the name of the predictors
% regZeroFrames specifies the colums that correspond to the 'zero time' of
% each regressor (ie when the event actually occurred)

% so in the 600th column of the design matrix belongs to 'handleSound':
sixhundredth_column = regLabels(regIdx(600))

% when recovering beta kernels, it is useful to know which colums
% correspond to the 'zero time' (ie when the regressor of interest actually
% occurs). This is only important for the "digital" regressors
handlesound_column_indices = find(regIdx == find(ismember(regLabels, 'handleSound')));
event_idx = find(regZeroFrames(handlesound_column_indices));
fprintf('\nThe handle grab occurs at column %i\n\n\n',handlesound_column_indices(event_idx));

% reject empty columns from the design matrix (and modify regLabels, regZeroFrames and
% regIdx appropriately)
[R, regLabels, regIdx, regZeroFrames, rejIdx] = ridgeModel_rejectEmptyRegressors(R,regLabels,regIdx,regZeroFrames);

% Now reject rank deficient columns from the design matrix
[R, regLabels, regIdx, regZeroFrames, rejIdx] = ridgeModel_rejectDeficientRegressors(R,regLabels,regIdx,regZeroFrames);

% Do the 10-fold cross validation
[Vm, betas, lambdas, cMap, cMovie] = ridgeModel_crossValidate(R,U,zeromeanVc,NFRAMES,NFOLDS);

% 10-fold crossvalidation, but this time sample trials instead of frames for cv-folds
[Vm, betas, lambdas, cMap, cMovie] =  crossValModel_trials(R,U, zeromeanVc, NFOLDS, NFRAMES);

% Save the results
%ridgeModel_saveResults(MODEL_SAVEPATH, ANIMAL, SESSION, MODELNAME, Vm, zeromeanVc, U, R, betas, lambdas, cMap, cMovie, regLabels, regIdx, rejIdx, regZeroFrames);

%% Bechmark a few methods of lambda estimation
% Benchmark lamba estimation via Karabastos algorithm,
% cross-validated grid search, and cross-validated fminbnd. 

% compute lambdas with bayesian MLE approach (Karabastos) and get R2
tic
cRidge = ridgeMML(zeromeanVc', R, true); %get beta weights and ridge penalty
disp('MLE: lambdas computed')
MLEruntime = toc;
MLEr2 = rateDisc_cvRidge(cRidge, zeromeanVc', R, NFOLDS, NFRAMES, U);


% lambda grid search
testRange = [1:500:2000 3000:2000:13000 15000:5000:30000];
tic
Cnt = 0;
crossvalR2 = NaN(1, length(testRange));
for x = testRange
    Cnt = Cnt + 1;
    crossvalR2(Cnt) = rateDisc_cvRidge(x, zeromeanVc', R, NFOLDS, NFRAMES, U);
end
crossvalRuntime = toc;

% use fminbnd
tic;
options = optimset('PlotFcns',@optimplotfval);
options.TolX = 10;
[cvRidge, cvR2] = fminbnd(@(u) -rateDisc_cvRidge(u, zeromeanVc', R, NFOLDS, NFRAMES, U), 1000, 30000, options);
newCrossvalRuntime = toc;

fprintf('MLE runtime = %f\n', MLEruntime)
fprintf('MLE R^2 = %f\n', MLEr2)
fprintf('grid-search crossVal runtime = %f\n', crossvalRuntime)
fprintf('grid-search crossVal R^2 = %f\n', max(crossvalR2))
fprintf('fminbnd crossVal runtime = %f\n', newCrossvalRuntime)
fprintf('fminbnd crossVal R^2 = %f\n', -cvR2)


%% Shuffle the design matrix columns - cvR^2 drops to zero

[regLabels,regIdx,R,regZeroFrames,zeromeanVc,U] = ridgeModel_returnDesignMatrix(DATAPATH, ANIMAL, SESSION);
shuffleLabels = regLabels; % shuffle every regressor
shuffledR = shuffleDesignMatrix(regLabels,regIdx,R,shuffleLabels);

[shuffledR, regLabels, regIdx, regZeroFrames, rejIdx] = ridgeModel_rejectEmptyRegressors(shuffledR,regLabels,regIdx,regZeroFrames);
[shuffledR, regLabels, regIdx, regZeroFrames, rejIdx] = ridgeModel_rejectDeficientRegressors(shuffledR,regLabels,regIdx,regZeroFrames);

[Vm, betas, lambdas, cMap2, cMovie] = ridgeModel_crossValidate(shuffledR,U,zeromeanVc,NFRAMES,NFOLDS); %need to adjust kernel zero points if they get discarded, or maybe make them nans

ridgeModel_saveResults(MODEL_SAVEPATH, ANIMAL, SESSION, 'shuffled_model', Vm, zeromeanVc, U, shuffledR, betas, lambdas, cMap, cMovie, regLabels, regIdx, rejIdx, regZeroFrames);



