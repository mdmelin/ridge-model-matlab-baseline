function [Vm, cBeta, cR, subIdx, cRidge, keptLabels, cLabelInds, cMap, cMovie] =  crossValModel(fullR,U,Vc,cLabels,regLabels,regIdx,frames,ridgeFolds)
        
        regs2grab = ismember(regIdx,find(ismember(regLabels,cLabels))); %these are just the regressors chosen by labels, no rejection yet
        cR = fullR(:,regs2grab); %grab desired labels from design matrix
        %reject regressors
        rejIdx = nansum(abs(cR)) < 10;
        [~, fullQRR] = qr(bsxfun(@rdivide,cR(:,~rejIdx),sqrt(sum(cR(:,~rejIdx).^2))),0); %orthogonalize design matrix
        %figure; plot(abs(diag(fullQRR))); ylim([0 1.1]); title('Regressor orthogonality'); drawnow; %this shows how orthogonal individual regressors are to the rest of the matrix
        if sum(abs(diag(fullQRR)) > max(size(cR)) * eps(fullQRR(1))) < size(cR,2) %check if design matrix is full rank
            temp = ~(abs(diag(fullQRR)) > max(size(cR)) * eps(fullQRR(1)));
            rejIdx(~rejIdx) = temp; %reject regressors that cause rank-defficint matrix
        end
        
        cR = cR(:,~rejIdx); % reject regressors that are too sparse or rank-defficient
        
        regs2grab = regIdx(regs2grab); %get indices that have our desired labels
        
        temporary = unique(regs2grab);
        keptLabels = regLabels(temporary);
        for x = 1 : length(temporary)
            cLabelInds(regs2grab == temporary(x)) = x; %make it so that cLabelInds doesn't skip any integers when we move past a label we don't want
        end
        
        cLabelInds = cLabelInds(~rejIdx); %now reject the regressors (from our subselection of labels) that had NaN's or were rank deficient
        subIdx = cLabelInds; %get rid of this redundant variable later
        
        fprintf(1, 'Rejected %d/%d empty or rank deficient regressors\n', sum(rejIdx),length(rejIdx));
        
        discardLabels = cLabels(~ismember(cLabels,keptLabels));
        
        if length(discardLabels) > 0
            fprintf('\nFully discarded regressor: %s because of NaN''s or emptiness \n', discardLabels{:});
        else
            fprintf('\nNo regressors were FULLY discarded\n');
        end
        
        %now move on to the regression
        Vm = zeros(size(Vc),'single'); %pre-allocate motor-reconstructed V
        randIdx = randperm(size(Vc,2)); %generate randum number index
        foldCnt = floor(size(Vc,2) / ridgeFolds);
        cBeta = cell(1,ridgeFolds);
        
        for iFolds = 1:ridgeFolds
            dataIdx = true(1,size(Vc,2));
            
            if ridgeFolds > 1
                dataIdx(randIdx(((iFolds - 1)*foldCnt) + (1:foldCnt))) = false; %index for training data
                if iFolds == 1
                    [cRidge, cBeta{iFolds}] = ridgeMML(Vc(:,dataIdx)', cR(dataIdx,:), true); %get beta weights and ridge penalty for task only model
                else
                    [~, cBeta{iFolds}] = ridgeMML(Vc(:,dataIdx)', cR(dataIdx,:), true, cRidge); %get beta weights for task only model. ridge value should be the same as in the first run.
                end
                Vm(:,~dataIdx) = (cR(~dataIdx,:) * cBeta{iFolds})'; %predict remaining data
                
                if rem(iFolds,ridgeFolds/5) == 0
                    fprintf(1, 'Current fold is %d out of %d\n', iFolds, ridgeFolds);
                end
            else
                [cRidge, cBeta{iFolds}] = ridgeMML(Vc', cR, true); %get beta weights for task-only model.
                Vm = (cR * cBeta{iFolds})'; %predict remaining data
                disp('Ridgefold is <= 1, fit to complete dataset instead');
            end
        end
        
        % computed all predicted variance
        Vc = reshape(Vc,size(Vc,1),[]);
        Vm = reshape(Vm,size(Vm,1),[]);
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
        
        % movie for predicted variance
        cMovie = zeros(size(U,1),frames, 'single');
        for iFrames = 1:frames
            
            frameIdx = iFrames:frames:size(Vc,2); %index for the same frame in each trial
            cData = bsxfun(@minus, Vc(:,frameIdx), mean(Vc(:,frameIdx),2));
            cModel = bsxfun(@minus, Vm(:,frameIdx), mean(Vm(:,frameIdx),2));
            covVc = cov(cData');  % S x S
            covVm = cov(cModel');  % S x S
            cCovV = cModel * cData' / (length(frameIdx) - 1);  % S x S
            covP = sum((U * cCovV) .* U, 2)';  % 1 x P
            varP1 = sum((U * covVc) .* U, 2)';  % 1 x P
            varP2 = sum((U * covVm) .* U, 2)';  % 1 x P
            stdPxPy = varP1 .^ 0.5 .* varP2 .^ 0.5; % 1 x P
            cMovie(:,iFrames) = gather(covP ./ stdPxPy)';
            clear cData cModel
            
        end
        fprintf('Run finished. Mean Rsquared: %f... Median Rsquared: %f\n', mean(cMap(:)), median(cMap(:)));
    end