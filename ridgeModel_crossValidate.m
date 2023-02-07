function [Vm, betas, lambdas, cMap, cMovie] =  ridgeModel_crossValidate(fullR,U,Vc,frames,ridgeFolds)

Vm = zeros(size(Vc),'single'); %pre-allocate reconstructed V
randIdx = randperm(size(Vc,2)); %generate randum number index
foldCnt = floor(size(Vc,2) / ridgeFolds);
betas = cell(1,ridgeFolds);

for iFolds = 1:ridgeFolds
    dataIdx = true(1,size(Vc,2));

    if ridgeFolds > 1
        dataIdx(randIdx(((iFolds - 1)*foldCnt) + (1:foldCnt))) = false; %index for training data
        if iFolds == 1
            [lambdas, betas{iFolds}] = ridgeMML(Vc(:,dataIdx)', fullR(dataIdx,:), true); %get beta weights and ridge penalty for task only model
        else
            [~, betas{iFolds}] = ridgeMML(Vc(:,dataIdx)', fullR(dataIdx,:), true, lambdas); %get beta weights for task only model. ridge value should be the same as in the first run.
        end
        Vm(:,~dataIdx) = (fullR(~dataIdx,:) * betas{iFolds})'; %predict remaining data

        if rem(iFolds,ridgeFolds/5) == 0
            fprintf(1, 'Current fold is %d out of %d\n', iFolds, ridgeFolds);
        end
    else
        [lambdas, betas{iFolds}] = ridgeMML(Vc', fullR, true); %get beta weights for task-only model.
        Vm = (fullR * betas{iFolds})'; %predict remaining data
        disp('Ridgefold is <= 1, fit to complete dataset instead');
    end
end

[cMovie, cMap] = compute_predicted_variance(Vc, Vm, U, frames); % the dev set
fprintf('Run finished. RMSE: %f\n', median(cMovie(:).^2));

end