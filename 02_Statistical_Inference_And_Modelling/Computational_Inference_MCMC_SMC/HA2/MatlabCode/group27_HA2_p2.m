%% SF2955 Home Assignment 2, part 1, group 27

% run first this section of the code, then run each section below to plot the results

y = load("hmc-observations.csv");
sig = 2;
sigma = [5,0;0,1/2];
sigmaInv = inv(sigma); %[0.2, 0; 0, 0.2];
theta0 = mvnrnd([0, 0], sigma)';



% running the HMC
N = 10000; % set to 10 000 here for faster computation
L = 55;
epsilon = 0.04;

[thetaHMC, accepted_rateHMC] = HMC(theta0, sig, sigmaInv, N,L,y, epsilon);


% running the MH
N2 = 10000; 
burnin = 0.2*N2;
sdv = 0.5; % to be calibrated

[thetaMH, accepted_rateMH] = MH(theta0, sdv, sig, sigmaInv, N2, y);


%% Grid search to find suitable values of L and epsilon for HMC

% N3 = 1000;
% theta0 = mvnrnd([0, 0], sigma)';
% L_mat = [15, 20, 25, 30, 40, 50, 75, 100];
% eps = 0.3:0.1:1.7;
% L_len = length(L_mat);
% eps_len = length(eps);
% acc_mat = zeros(L_len,eps_len);
% for l = 1:L_len
%     for k = 1:eps_len
%         epsilon = eps(k)/L_mat(l);
%         [~, accptd] = HMC(theta0, sig, sigmaInv, N3, L_mat(l), y, epsilon);
%         acc_mat(l, k) = accptd;
%     end
% end



%% POSTERIOR DENSITY PLOTS

figure()
sgtitle('Posterior Density f(\theta|y)')

subplot(1,5,1)
colorbar

subplot(1,5,2)
histogram2(thetaHMC(1,:), thetaHMC(2,:),'FaceColor','flat')
xlabel("\theta_2");
ylabel("\theta_1");
title("HMC with L="+L+", \epsilon="+epsilon)

subplot(1,5,3)
histogram2(thetaHMC(1,:), thetaHMC(2,:),'FaceColor','flat')
xlabel("\theta_2");
ylabel("\theta_1");
title("HMC with L="+L+", \epsilon="+epsilon)

subplot(1,5,4)
histogram2(thetaMH(1,:), thetaMH(2,:),'FaceColor','flat')
xlabel("\theta_2");
ylabel("\theta_1");
title("MH with \zeta="+sdv)

subplot(1,5,5)
histogram2(thetaMH(1,:), thetaMH(2,:),'FaceColor','flat')
xlabel("\theta_2");
ylabel("\theta_1");
title("MH with \zeta="+sdv)



%% STATE SPACE EXPLORATION
figure()
sgtitle("State Space Exploration")

subplot(1,4,1)
plot(1:N, thetaHMC(1,:));
title("HMC, \theta_1")
xlabel('N');
ylabel('\theta_1');

subplot(1,4,2)
plot(1:N, thetaHMC(2,:));
title("HMC, \theta_2")
xlabel('N');
ylabel('\theta_2');

subplot(1,4,3)
plot(burnin:N, thetaMH(1,burnin:end));
title("MH, \theta_1")
xlabel('N');
ylabel('\theta_1');

subplot(1,4,4)
plot(burnin:N, thetaMH(2,burnin:end));
title("MH, \theta_2")
xlabel('N');
ylabel('\theta_2');



%% AUTOCORRELATION PLOTS

figure()
subplot(1,4,1)
[acf,lags] = autocorr(thetaHMC(1, :));
stem(lags, acf, 'filled');
title("ACF for \theta_1 with HMC")
xlabel("LAG")
ylabel("ACF")

subplot(1,4,2)
[acf,lags] = autocorr(thetaHMC(2, :));
stem(lags, acf, 'filled');
title("ACF for \theta_2 with HMC")
xlabel("LAG")
ylabel("ACF")

subplot(1,4,3)
[acf,lags] = autocorr(thetaMH(1, burnin:end));
stem(lags, acf, 'filled');
title("ACF for \theta_1 with MH")
xlabel("LAG")
ylabel("ACF")

subplot(1,4,4)
[acf,lags] = autocorr(thetaMH(2, burnin:end));
stem(lags, acf, 'filled');
title("ACF for \theta_2 with MH")
xlabel("LAG")
ylabel("ACF")



%% FUNCTIONS
% METROPOLIS-HASTINGS 
function [thetaMat, accepted_rate] = MH(theta0, sdv, sig, sigmaInv, N,y)    
    theta = mvnrnd(theta0, sdv^2*eye(2))';

    f = @(theta) exp(-1/(2*sig^2)* sum((y-(theta(1)^2+theta(2)^2)).^2) - 1/2* theta' * sigmaInv * theta);
    thetaMat = zeros(2,N);
    
    thetaMat(:,1) = theta;
    accepted = 0;
    for i = 2:N       
        thetaStar = mvnrnd(thetaMat(:,i-1), sdv^2*eye(2))';
        
        alpha = min(1, f(thetaStar)/(f(thetaMat(:,i-1))));
        
        if rand <= alpha
            thetaMat(:,i)=thetaStar;
            accepted = accepted + 1;
        else
            thetaMat(:,i)=thetaMat(:,i-1);
        end
    end
    accepted_rate = accepted/(N-1);
end



% LEAPFROG
function [theta, v] = leapFrog(theta_0, y, v_0, sig, sigmaInv, epsilon, L)
    grad_U = @(theta, y) -1/(2*sig^2)*sum(y - theta'*theta)*theta + sigmaInv*theta;
    theta = theta_0;
    v = v_0 - 0.5*epsilon*grad_U(theta, y);
    for m = 1:L-1
        theta = theta + epsilon*v;
        v = v - epsilon*grad_U(theta, y);
    end
    theta = theta + epsilon*v;
    v = v - 0.5*epsilon*grad_U(theta, y);
    v = -v;
end

% HAMILTONIAN MONTE CARLO
function [theta, accepted_rate] = HMC(theta0, sig, sigmaInv, N, L, y, epsilon)
    H = @(theta, y, v) (1/sig^2*sum((y-theta'*theta).^2) + 0.5*theta'*(sigmaInv*theta) + 0.5*sum(v.^2));
    theta = zeros(2, N);
    theta(:, 1) = theta0;
    accepted = 0;
    for n = 2:N
        v0 = mvnrnd([0,0], eye(2))';
        [theta_n, v] = leapFrog(theta(:, n-1), y, v0, sig, sigmaInv, epsilon, L);
        alpha = min(1, exp(H(theta(:, n-1), y, v0) - H(theta_n, y, v)));
        u = rand;
        if u <= alpha
            theta(:, n) = theta_n;
            accepted = accepted + 1;
        else
            theta(:, n) = theta(:, n-1);
        end
    end
    accepted_rate = accepted / (N-1);
end

