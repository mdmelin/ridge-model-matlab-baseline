function [cvLoss, betas, cMap] = rateDisc_cvRidge(cRidge, Y, X, ridgeFolds, frames, U)
% function to run cross-validation for different ridge values.
% optimized for speed over ridgeMML
% cRidge is the lambda value to be tested, 
% Y is the data (temporal components for widefield) of size frames x components.
% X is the design matrix of size frames x regressors
% ridgeFolds is the number of folds for cross-validation
% frames is the number of frames per trial that is used for randomization
% U are spatial components of size pixels x components (optional input for widefield)

rng('default');

if strcmpi(class(Y), 'gpuArray')
    Vm = gpuArray(zeros(size(Y),'single')); %pre-allocate motor-reconstructed V
else
    Vm = zeros(size(Y),'single'); %pre-allocate motor-reconstructed V
end

% make index that contain trials instead of random data points
trialIdx = randperm(size(Y,1) / frames);
randIdx = 1 : size(Y,1);
randIdx = reshape(randIdx, frames, []);
randIdx = randIdx(:, trialIdx);
randIdx = randIdx(:)';

foldCnt = floor(size(Y,1) / ridgeFolds);
pY = size(Y, 2);
p = size(X, 2);  % Predictors

% Prep penalty matrix
ep = eye(p);

YMean = mean(Y, 1);
Y = bsxfun(@minus, Y, YMean);

orgX = X;

XStd = std(X, 0, 1);
X = bsxfun(@rdivide, X, XStd);
XMean = mean(X, 1);
X = bsxfun(@minus, X, XMean);
X(isnan(X)) = 0;

if length(cRidge) == 1 && strcmpi(class(Y), 'gpuArray')
    cRidge = repmat(cRidge, 1, pY);
end

for iFolds = 1:ridgeFolds
    
    dataIdx = true(1,size(Y,1));   
    dataIdx(randIdx(((iFolds - 1)*foldCnt) + (1:foldCnt))) = false; %index for training data
    if strcmpi(class(Y), 'gpuArray')
        betas = gpuArray(NaN(p, pY));
    else
        betas = NaN(p, pY);
    end
    
    cX = X(dataIdx,:);
    XTX = cX' * cX;
        
    % Compute X' * Y all at once, again for speed
    XTY = cX' * Y(dataIdx,:);

    % Compute betas for renormed X
    if length(cRidge) == 1
        betas = ((XTX + cRidge * ep) \ XTY);
    else
        for i = 1:pY
            betas(:, i) = ((XTX + cRidge(i) * ep) \ XTY(:, i));
        end
    end

    % Adjust betas to account for renorming.
    betas = bsxfun(@rdivide, betas, XStd');
    betas(isnan(betas)) = 0;
    Vm(~dataIdx,:) = (orgX(~dataIdx,:) * betas); %predict remaining data
end


%% compute RMSE
if exist('U', 'var')
    % compute R2
    Vc = Y';
    Vm = Vm';

    % shrink U if not shrunk
    if length(size(U)) == 3
        U = arrayShrink(U, squeeze(isnan(U(:,:,1))));
    end

    covVc = cov(Vc');  % S x S
    covVm = cov(Vm');  % S x S
    cCovV = bsxfun(@minus, Vm, mean(Vm,2)) * Vc' / (size(Vc, 2) - 1);  % S x S
    covP = sum((U * cCovV) .* U, 2)';  % 1 x P
    varP1 = sum((U * covVc) .* U, 2)';  % 1 x P
    varP2 = sum((U * covVm) .* U, 2)';  % 1 x P
    stdPxPy = varP1 .^ 0.5 .* varP2 .^ 0.5; % 1 x P
    cMap = gather((covP ./ stdPxPy)');
    cvLoss = nanmean(cMap.^2);
    
else
    %compute RMSE
    YT = bsxfun(@minus, Y, nanmean(Y,1));
    XT = bsxfun(@minus, Vm, nanmean(Vm,1));

    err = (YT - XT) .^ 2;
    cvLoss = sqrt(mean(err(:)));
end



