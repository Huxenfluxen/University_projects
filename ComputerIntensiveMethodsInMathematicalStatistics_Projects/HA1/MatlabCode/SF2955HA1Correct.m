%%% Project 1 SF2955 - Computer intensive methods in Statistics %%%
 
%% Problem 1
% load('RSSI-measurements-unknown-sigma.mat');
% load('RSSI-measurements.mat');
% load('stations.mat');
load('stations.mat');
P = (15*eye(5) + ones(5,5))/20; % Transition probability matrix
dt = 1/2;
alpha = 0.6;
sigma = 1/2;
phi_tilde = [1, dt, dt^2/2; 0, 1, dt; 0, 0, alpha];
psi_tilde_z = [dt^2/2; dt; 0];
psi_tilde_w = psi_tilde_z + [0; 0; 1];
phi = [phi_tilde, zeros(3); zeros(3), phi_tilde];
psi_z = [psi_tilde_z, zeros(3,1); zeros(3,1), psi_tilde_z];
psi_w = [psi_tilde_w, zeros(3,1); zeros(3,1), psi_tilde_w];
driving_set_list = [0, 0; 
               3.5, 0; 
               0, 3.5; 
               0, -3.5; 
               -3.5, 0];
m = 500;
X = zeros(6,m+1);
% Z = zeros(2,m+1);
% X0 state
muX0 = zeros(6,1);
sigmaX0= diag([500,5,5,200,5,5]);
X(:,1) = mvnrnd(muX0,sigmaX0);
w=mvnrnd([0;0],1/4*eye(2),m)';
[Z0, Zidx] = datasample(driving_set_list,1);
for i=1:m % create trajectory for X
    [Zi, Zidx] = datasample(driving_set_list,1,'Weights',P(Zidx,:));    
    X(:,i+1) = phi*X(:,i)+psi_z*Zi'+psi_w*w(:, i);
end
%% testing our SISR and known plotting trajectory
[~, y] = createY(X, pos_vec);
N = 2000;
sdV = 1.5;
X0 = mvnrnd(muX0,sigmaX0, N);
[tauSISy,wSISy,mostProbZy] = SISfunc(y, X0', pos_vec, P, driving_set_list,phi, psi_w, psi_z, sdV, true);
figure()
plot(X(1,:),X(4,:));
hold on;
plot(X(1,1), X(4,1), 'o');
plot(X(1,end), X(4,end), 'o');
plot(pos_vec(1,:), pos_vec(2,:), '*')
plot(tauSISy(1, :), tauSISy(4, :))
legend("true trajectory", "initial point", "end point", "SISR approximation")
xlabel("x1-position")
ylabel("x2-position")
hold off;
%% Problem 2
[muY, y] = createY(X, pos_vec);
sdV = 1.5^2*eye(6); % Covariance matrix of noise V
pYgivenX = mvnpdf(y', muY', sdV); % the conditional probability of out simulated RSSI
plot(pYgivenX)
%% Problem 3
load('RSSI-measurements.mat')
sdV = 1.5;
N = 10000;
resampling = false;
X0 = mvnrnd(muX0,sigmaX0, N);
tic;  % Start timing
[tauSIS,wSIS, mostProbZ] = SISfunc(Y, X0', pos_vec, P, driving_set_list,phi, psi_w, psi_z, sdV, resampling);
% mode of most prob Z is 2 
elapsedTime = toc;  % End timing and capture the elapsed time
disp(['Elapsed Time: ', num2str(elapsedTime), ' seconds']);
%% Plotting problem 3
% Plotting histograms of weights at different time instances
figure() % plot histograms of weights
nPlotVals = [1,5,10,15,20];
% nPlotVals = [10, 20, 50, 100, 300, 500];
nrsubplots = length(nPlotVals);
sgtitle("Histograms of weights from SIS")
upperBound = max(log10(wSIS(:,nPlotVals(1))));
lowerBound = min(log10(wSIS(:,nPlotVals(end))));
for i=1:nrsubplots
    subplot(nrsubplots,1,i);
    weightHist(nPlotVals(i), wSIS, lowerBound, upperBound);
end
% Plotting the expectancy using seq. imortance sampling w/o resampling
figure()
plot(tauSIS(1, :), tauSIS(4, :))
hold on
plot(tauSIS(1, 1), tauSIS(4, 1), 'o')
plot(tauSIS(1, end), tauSIS(4, end), 'o')
xlabel("x1-position")
ylabel("x2-position")
plot(pos_vec(1,:), pos_vec(2,:), '*')
hold off
title("Expectancy \tau using SIS, no resampling")
legend("trajectory", "start", "end", "stations")
% Plotting most probable Z at each state
%% 3. ESS
figure()
weightMean = mean(wSIS(:,1:50));
weightStd = std(wSIS(:,1:50));
cv = weightStd ./ weightMean;  
ess = N ./ (1 + cv.^2);
% plot(1:65, ess, "r*--")
scatter(1:50, ess);
title("Effective sample size")
% this suggests that some particles have a much higher leverage than others
% these will dominate the set, and lead to poor approximation
% weights become less uniform over time
%% Problem 4
load('RSSI-measurements.mat')
resampling = true;
sdV = 1.5;
N = 10000;
resampling = false;
X0 = mvnrnd(muX0,sigmaX0, N);
tic;  % Start timing
[tauSIS,wSIS, mostProbZ] = SISfunc(Y, X0', pos_vec, P, driving_set_list,phi, psi_w, psi_z, sdV, resampling);
% mode of most prob Z is 2 
elapsedTime = toc;  % End timing and capture the elapsed time
disp(['Elapsed Time: ', num2str(elapsedTime), ' seconds']);

%% Problem 5
load('RSSI-measurements-unknown-sigma.mat');
N = 10000;
m = 500;
X0 = mvnrnd(muX0,sigmaX0, N);
tic;  % Start timing
sdV_vec = 0.3:0.01:2.9;
log_like_vec = zeros(length(sdV_vec), 1);
tauSISR = zeros(6, m+1, length(sdV_vec));
for i = 1:length(sdV_vec)
    sdV = sdV_vec(i);
    [tauSISR(:,:,i),~, ~] = SISfunc(Y, X0', pos_vec, P, driving_set_list,phi, psi_w, psi_z, sdV, true);
    log_like = 0;
    for n = 1:m+1
        trans = transferPDF(Y(:,n), tauSISR(:, n, i), pos_vec, sdV);
        log_like = log_like + log(trans);
    end
    log_like_vec(i) = log_like;
end

[~, idx] = max(log_like_vec);
% [max_log_like, idx] = max(1/m*log(trans_vec));
best_sdV = sdV_vec(idx);
elapsedTime = toc;  % End timing and capture the elapsed time
disp(['Elapsed Time: ', num2str(elapsedTime), ' seconds']);
%% Plot the new estimated positions tau
% Plotting the expectancy using seq. importance sampling w/o resampling
figure()
plot(tauSISR(1, :, idx), tauSISR(4, :, idx))
hold on
plot(tauSISR(1, 1, idx), tauSISR(4, 1, idx), 'o')
plot(tauSISR(1, end, idx), tauSISR(4, end, idx), 'o')
xlabel("x1-position")
ylabel("x2-position")
plot(pos_vec(1,:), pos_vec(2,:), '*')
hold off
legend("trajectory", "start", "end", "stations")

%% Functions problem 2
% calculates transition density p(yn|xn)
function [mu_y, y] = createY(X, baseStations)
    % X is the 6xm matrix containing the position of the target at each time step
    % baseStations contains the positions of all 6 base stations
    % output is normal distributed RSSI measurements approximated at each time step m
    
    v = 90*ones(6,1); % Base station transmission power in dB
    eta = 3*ones(6,1); % Path-loss exponent
    xpos = X([1,4],:); % position of target at each time
    m = length(X(1,:)) - 1;
    euDis = zeros(6, m+1);
    for l = 1:6 % Iterate over each base station
        euDis(l,:) = vecnorm(xpos - baseStations(:,l)); % Using Matlab's nice built in matrix/vector addition
    end
    
    mu_y = v-10*eta.*log10(euDis); % expectation of the RSSI signall Y, 6xm+1 matrix
    
    Vrnd = mvnrnd(zeros(6,1),1.5*eye(6), m+1); % The random 6x1 vector V_n (6xm+1 matrix)
    y = v-10*eta.*log10(euDis) + Vrnd'; % RSSI signals (6xm+1 matrix)
end
%% Functions problem 3
function pYX = transferPDF(Yobs, xn, baseStations, sdV)
    % Yobs 6x1 vector
    % xn 6xN matrix of N particles
    % output pYX Nx1 vector of probability densities (?) for each particle
    v = 90*ones(6,1); % Base station transmission power in dB
    eta = 3*ones(6,1); % Path-loss exponent
    xpos = xn([1,4],:); % position of target at each time
    N = size(xn,2);
    sdV = sdV^2*eye(6); % Covariance matrix of noise V
    euDis = zeros(6, N);
    for l = 1:6 % Iterate over each base station
        euDis(l,:) = vecnorm(xpos - baseStations(:,l)); % Using Matlab's nice built in matrix/vector addition
    end
    muY = v-10*eta.*log10(euDis); % expectation of the RSSI signal Y
    pYX = prod(mvnpdf(repmat(Yobs', N, 1),muY', sdV),2);
end
%%% Sequential Importance Sampling for Gaussian Hidden Markov Model %%%
function [tau,w, mostProbZ] = SISfunc(Y, X0, baseStations, P, driving_set_list,phi, psi_w, psi_z, sdV, SISR)
    % X0 is a 6xN matrix (6 X:s for each of the N simulated particles at the current time n)
    % Y is a 6xn matrix (6 signals at current time n)
    % Z0 is an 2xN vector of N random directions (two values for each particle)
    % N is the number of particles simulated, possible up to 10000
    
    m = size(Y,2);
    N = size(X0,2);
    tau = zeros(6, m); % vector of estimates
    X = zeros(6, N, m);
    X(:,:,1) = X0;
    w = zeros(N,m); % N observations for each n time instance. w saves all weights
    w(:,1) = transferPDF(Y(:,1),X0, baseStations, sdV); % finding N weights for the first N particles
    omega = sum(w(:,1),1); % sum over all N weights from observation 0
    tau(:, 1) = 1/omega*X(:,:,1)*w(:,1); % 6x1 vector
    
    Zidx = zeros(N,m);
    [Z0, Zidx(:,1)] = datasample(driving_set_list,N); % sets first column of Zidx to N randoms betw 1 and 5
    Z = Z0;
    W = normrnd(0, 0.5, N, 2, m-1);
    % mostProbZ = zeros(1,m);
    % mostProbZ(1) = mode(Zidx(:,1));
    if SISR
        for n = 2:m
            % Resampling
            ind = randsample(N,N,true,w(:,n-1)); % selection step with resampling
            X(:,:,n-1) = X(:,ind,n-1); % resample the particles
            X(:,:,n) = phi*X(:,:,n-1) + psi_w*W(:,:,n-1)'; % Mutation
            
            % Update Z
            ZprevInd = Zidx(:,n-1); % Nx1 vector of z-indices from previous state
            ZnewInd = zeros(N,1);
            for i = 1:5
                indReplace = find(ZprevInd == i); % indices in previous Z that are i
                nri = length(indReplace); % number of elements in previous Z that are i
                ZnewInd(indReplace) = randsample(5,nri,true,P(i,:)); % replace the nri new Z from i indeces using P
            end
            Zidx(:,n)=ZnewInd;
            Z= driving_set_list(ZnewInd,:);
            
            % for i = 1:N
            %     [Z(i,:), Zidx(i,n)] = datasample(driving_set_list,1,'Weights',P(:, Zidx(i,n-1)));
            % end
            % mostProbZ(n) = mode(Zidx(:,n));
            X(:,:,n) = X(:,:,n) + psi_z*Z'; % update step    
            
            transPDF = transferPDF(Y(:,n),X(:,:,n), baseStations, sdV); % Weight Update
            w(:,n) = transPDF;
            
            w(:,n) = w(:,n) / sum(w(:,n),1); % Normalization of weights
            
            tau(:, n) = X(:,:,n)*w(:,n); % Estimation
            %tau(:, n) = 1/N*sum(X(:,:,n),2); % Estimation as in
            %instructions??
        end
     else
        for n = 2:m % main loop in time up to m
            X(:,:,n) = phi*X(:,:,n-1)+psi_z*Z'+psi_w*W(:,:,n-1)'; % mutation
            
            % Update Z
            ZprevInd = Zidx(:,n-1); % Nx1 vector of z-indices from previous state
            ZnewInd = zeros(N,1);
            for i = 1:5
                indReplace = find(ZprevInd == i); % indeces in previous Z that are i
                nri = length(indReplace); % number of elements in previous Z that are i
                ZnewInd(indReplace) = randsample(5,nri,true,P(i,:)); % replace the nri new Z from i indeces using P
            end
            Zidx(:,n)=ZnewInd;
            Z = driving_set_list(ZnewInd,:);
            
            transPDF = transferPDF(Y(:,n),X(:,:,n), baseStations, sdV);
            w(:,n) = w(:,n-1).*transPDF;
            omega = sum(w(:,n),1);
            tau(:, n) = 1/omega*X(:,:,n)*w(:,n); % estimation
        end
    end
    mostProbZ = mode(Zidx,1);
end
function weightHist(n, weightMat, lowerBound, upperBound)
    data = weightMat(:, n);
    filtered_data = data(data~= 0); % removes 0 from weights, so we can log all values
    histogram(log10(filtered_data));
    xlim([lowerBound - 1, upperBound + 1]);
    title("n = "+ n);
end