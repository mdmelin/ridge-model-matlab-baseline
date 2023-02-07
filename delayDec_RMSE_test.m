% lambda effect example figure
cPath = 'Q:\BpodImager\MLE\mSM66\27-Jun-2018\';

load([cPath 'interpVc.mat'], 'Vc', 'frames');
load([cPath 'Vc.mat'], 'U');
load([cPath 'fullR.mat']);

ridgeFolds = 5;
cMask = squeeze(isnan(U(:,:,1)));
U = arrayShrink(U, cMask);

Vc = gpuArray(Vc);
fullR = gpuArray(fullR);
U = gpuArray(U);

% compute lambdas with bayesian MLE approach and get R2
tic
cRidge = ridgeMML(gather(Vc)', gather(fullR), true); %get beta weights and ridge penalty
disp('MLE: lambdas computed')
MLEruntime = toc;
MLEr2 = rateDisc_cvRidge(cRidge, Vc', fullR, ridgeFolds, frames, U);

Vc = gather(Vc);
fullR = gather(fullR);
U = gather(U);

% compute lambdas with cross-validation and get R2 for different values
testRange = [1:500:2000 3000:2000:13000 15000:5000:30000];
tic
Cnt = 0;
crossvalR2 = NaN(1, length(testRange));
for x = testRange
    Cnt = Cnt + 1;
    crossvalR2(Cnt) = rateDisc_cvRidge(x, Vc', fullR, ridgeFolds, frames, U);
end
crossvalRuntime = toc;

tic;
options = optimset('PlotFcns',@optimplotfval);
options.TolX = 10;
[cvRidge, cvR2] = fminbnd(@(u) -rateDisc_cvRidge(u, Vc', fullR, ridgeFolds, frames, U), 1000, 30000, options);
newCrossvalRuntime = toc;

%% make figure
figure('renderer','painters')
plot(testRange,crossvalR2, '-o')
nhline(MLEr2, 'r', 'linewidth', 2);
nhline(-cvR2, 'k', 'linewidth', 2);
nvline(cvRidge, 'k', 'linewidth', 2);
title(cPath);
niceFigure;

fprintf('MLE runtime = %f\n', MLEruntime)
fprintf('MLE R^2 = %f\n', MLEr2)

fprintf('grid-search crossVal runtime = %f\n', crossvalRuntime)
fprintf('grid-search crossVal R^2 = %f\n', max(crossvalR2))

fprintf('fminbnd crossVal runtime = %f\n', newCrossvalRuntime)
fprintf('new crossVal R^2 = %f\n', -cvR2)