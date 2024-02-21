clc
clear all;
% 
dataName{1} = 'cifar10_tran';
path = './';
addpath(genpath(path));

for name = 1
    load(['.\dataset\', dataName{name}, '.mat'])
    numker = length(X);
    num = length(X{1});
    num_landmark = 15 * ceil(sqrt(num));
    rng(1);
    index = sort(datasample(1:num, num_landmark, 'replace', false));
    P = zeros(num, num_landmark,numker);
    W = zeros(num_landmark,num_landmark,numker);
    for ker = 1:numker
        data_temp = X{ker};
        data_temp = pre_process(data_temp);
        sample_row = data_temp(index,:);
        P(:,:,ker) = create_kernel(data_temp, sample_row);
    end
  
    %% initialization
    numclass = length(unique(Y));
    Y(Y<1) = numclass;
    numker = size(P,3);
    num = size(P,1);

    options.seuildiffsigma=1e-4;        % stopping criterion for weight variation
    %------------------------------------------------------
    % Setting some numerical parameters
    %------------------------------------------------------
    options.goldensearch_deltmax=1e-3; % initial precision of golden section search
    options.numericalprecision=1e-16;   % numerical precision weights below this value
    % are set to zero
    %------------------------------------------------------
    % some algorithms paramaters
    %------------------------------------------------------
    options.firstbasevariable='first'; % tie breaking method for choosing the base
    % variable in the reduced gradient method
    options.nbitermax=500;             % maximal number of iteration
    options.seuil=0;                   % forcing to zero weights lower than this
    options.seuilitermax=10;           % value, for iterations lower than this one
    options.miniter=0;                 % minimal number of iterations
    options.threshold = 1e-4;
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% --- SVD-basd SimpleMKKM---- %%

    %% --- The Proposed SimpleMKKM(SMKKM)---- %%
    W = P(index,:,:);

    index1 = datasample(1:length(index), 300, 'Replace', false);
    W2 = W( : ,index1,:);
    tic
    [~,Sigma1,~] = simpleMKKM_svd(W2,numclass,options);
    P_sum = sumKbeta(P, Sigma1.^2);
    W_sum2 = sumKbeta(W2, Sigma1.^2);
    [U,D,~] = svds(W_sum2, numclass);
    D = diag(1./diag(D));
    H2 = P_sum * (U * D);
    t2 = toc;
    res2 = myNMIACC(H2,Y,numclass);
    
    
   
    
end

