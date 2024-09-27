
%% SF2955 Home Assignment 2, part 1, group 27

% run first this section of the code, then run each section below to plot the results


clear;
load("coal-mine.csv"); % saves as coal_mine
N = 10000;
burnin = 2000;
breakpoints = 4;
rho = 1;
v=1;
[tMat, thetaVec, lambdaMat, meanVec] = MCMC(N,coal_mine, breakpoints, rho, v, burnin);


%% HISTOGRAM OF T FOR DIFFERENT BREAKPOINTS 

figure(); 
for breakpoints = 1:4
    subplot(1, 4, breakpoints); 
    [tMat, thetaVec, lambdaMat, meanVec] = MCMC(N, coal_mine, breakpoints, rho, v, burnin);

    for i = 2:breakpoints + 1
        histogram(tMat(i,burnin:end), 20);
        hold on;
    end
    hold off;
    title(['{' num2str(breakpoints) '} breakpoints']);
    xlabel("year")
    ylabel("weight")
end
sgtitle('Distribution of breakpoints');



%% HISTOGRAM OF INTENSITIES FOR DIFFERENT BREAKPOINTS

figure(); 
for breakpoints = 1:4
    subplot(1, 4, breakpoints); 
    [tMat, thetaVec, lambdaMat, meanVec] = MCMC(N, coal_mine, breakpoints, rho, v, burnin);

    for i = 1:breakpoints + 1
        histogram(lambdaMat(i,burnin:end),20);

        hold on;

    end
    legend("interval 1", "interval 2", "interval 3", "interval 4", "interval 5");
    hold off;
    title(['{' num2str(breakpoints) '} breakpoints']);
    xlabel("intensity")
    ylabel("weight")
end
sgtitle('Distribution of intensity \lambda in each interval, for 1,2,3 and 4 breakpoints');


%% HISTOGRAM OF THETA FOR DIFFERENT V, DIFFERENT BREAKPOINTS

i = 0;
for v=[1/20, 1/2, 5, 50, 500]
    i = i+1;
    subplot(1, 5,i);

    for breakpoints=1:4      
        [tMat, thetaVec, lambdaMat, meanVec] = MCMC(N, coal_mine, breakpoints, rho, v, burnin);
        histogram(thetaVec(burnin:end),20);
        hold on;
    end  
    title( "v = " + v);

    hold off;
    legend("1 breakpoint", "2 breakpoints", "3 breakpoints", "4 breakpoints")  
    xlabel("\theta")
    ylabel("weight")

    sgtitle('Distribution of \theta, for 1,2,3 and 4 breakpoints and different values of \nu');
end

%% HISTOGRAM OF T FOR DIFFERENT V

figure();
i = 0;
for v=[1/20, 1/2, 5, 50, 500]
    i = i+1;
    subplot(1, 5,i);
    breakpoints=1;      
    [tMat, thetaVec, lambdaMat, meanVec] = MCMC(20000, coal_mine, breakpoints, rho, v, burnin);
    histogram(tMat(2,burnin:end),20);
    title( "v = " + v);
    xlabel("years")
    ylabel("weight")
    sgtitle('Distribution of 1 breakpoint for different values of \nu, N = 20000');
end



%% MIXING FOR DIFFERENT VALUES OF RHO

roVals = [1/80, 1/6,1/2,1,5, 20];
lr = length(roVals);

% BREAKPOINT MEAN CONVERGENCE PLOTS
figure;
i = 1;
for rho = roVals
    subplot(1,lr,i)
    [tMat, thetaVec, lambdaMat, meanVec] =MCMC(N,coal_mine, breakpoints, rho, v, burnin);
    plot(burnin+1:N, meanVec(2:end-1,:), "LineWidth", 1.5)
    title( ['\rho = {' num2str(rho) '}' ]);
    xlabel("N")
    ylabel("Expected breakpoint")

    i = i+1;
end
sgtitle([' Expected value of 1 breakpoints, with burnin = {' num2str(burnin) '}, N = {' num2str(N) '} for different values of \rho'])

% STATE SPACE EXPLORATION PLOTS
figure;
i = 1;
for rho = roVals
    subplot(1,lr,i)
    [tMat, thetaVec, lambdaMat, meanVec] =MCMC(N,coal_mine, breakpoints, rho, v, burnin);
    plot(burnin:N,tMat(2,burnin:end))
    title( ['\rho = {' num2str(rho) '}' ]);
    xlabel("N")
    ylabel("Expected breakpoint")
    i = i+1;
end
sgtitle([' State Space Exploration of 1 breakpoints, with burnin = {' num2str(burnin) '}, N = {' num2str(N) '} for different values of \rho'])

% ACF PLOTS
figure();
i=1;
for rho = roVals

    [tMat, thetaVec, lambdaMat, meanVec] = MCMC(N,coal_mine, breakpoints, rho, v, burnin);
    [acf,lags] = autocorr(tMat(2, burnin:end));

    subplot(1,lr,i)
    stem(lags, acf, 'filled');
    title(['\rho = {' num2str(rho) '}']);
    xlabel("LAG")
    ylabel("ACF")
    i=i+1;
end
sgtitle("Mixing for MH: ACF plots with different values of \rho, for 1 breakpoint")



%% MCMC FUNCTION

function [tMat, thetaVec, lambdaMat, meanVec] = MCMC(N, data, breakpoints, ro, v, burnin)
    % MCMC algorithm which returns a (breakpoints+2)xN matrix tMat of breakpoints, including first and last years,
    % a 1xN thetaVec of parameter theta, a (breakpoints+1)xN matrix
    % lambdaMat containing intensities in each interval, and a
    % (breakpoints+2)xN matrix meanVec containing the mean of the fist n
    % t-values for each breakpoint, for burnin <n <= N.
    
    d = breakpoints +1; % d is number of intervals
    

    theta0 = gamrnd(2,v);
    thetaVec = zeros(1,N);
    thetaVec(1) = theta0;
    
    lambda0 =  gamrnd(2, theta0,d,1);
    lambdaMat = zeros(d,N);
    lambdaMat(:,1) = lambda0;
    
    tMat = zeros(d+1,N);
    tMat(1,:) = 1851;
    tMat(d+1,:) = 1963;
    tMat(2:d,1) = transpose(sort(1851+rand(1,d-1)*(1963-1851)));
    
    meanVec = zeros(d+1,N-burnin); % saves expected value of t at each N

    for i = 2:N
         % Gibbs sampling to update theta
        thetaVec(i) = gamrnd(2+2*d, 1/(v + sum(lambdaMat(:,i-1))));
    

        % Gibbs sampling to update lamdba, for each interval
        for j = 1:d 
            nj = sum((data >= tMat(j,i-1)) & (data < tMat(j+1,i-1)));
            lambdaMat(j, i) = gamrnd(2+nj, 1/(thetaVec(i) + tMat(j+1,i-1) - tMat(j,i-1)));
        end
        

        % Metropolis-Hastings algorithm to update t using random walk proposal
        for j=2:d 
            R = ro * (tMat(j+1,i-1) - tMat(j-1,i-1));
            epsilon = -R + 2*R*rand(1);
            
            tj = tMat(j,i-1); % current t
            tStar = tj + epsilon; % next t suggestion
    
            ft = (tMat(j+1,i-1)-tj)*(tj-tMat(j-1,i-1))*exp(-lambdaMat(j,i)*(tMat(j+1,i-1)-tj))*exp(-lambdaMat(j,i)*(tj-tMat(j-1,i-1)));
            ftStar = (tMat(j+1,i-1)-tStar)*(tStar-tMat(j-1,i-1))*exp(-lambdaMat(j,i)*(tMat(j+1,i-1)-tStar))*exp(-lambdaMat(j,i)*(tStar-tMat(j-1,i-1)));
    
            q = ftStar/ft; % acceptance criteria for alpha
            
            % make sure tStar is within range, otherwise set q=0 so tStar is not used
            if tStar <= tMat(j-1,i)
                q = 0;
            elseif tStar >= tMat(j+1,i-1)
                q = 0; 
            end
                  
            alpha = min(1, q);
            u = rand(1);
    
            % update t
            if u <= alpha
                tMat(j,i)=tStar;
            else
                tMat(j,i)=tMat(j,i-1);
            end
        end
        
        % calculate mean of first N values, efter burnin
        if i>burnin
            t = tMat(:, burnin:i); 
            m = mean(t,2);
            meanVec(:,i-burnin) = m;
        end
    end
end
