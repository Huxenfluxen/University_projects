using LinearAlgebra
using Plots
using Arpack

function powerMethod(startingVec, A, N) 
    "finds approximation of the eigenvector for the largest 
    eigenvalue, and vector of absolute error at each iteration.
    - this version returns the final approximation, and error in each iteration"
    
    errorVec = zeros(N);
    v = startingVec/norm(startingVec);
    eigVals = eigen(A).values;
    max_eigVal = maximum(abs.(eigVals));
    approxEigVal=0
    for k = 1:N
        w = A * v;
        v = w / norm(w);  # normalize v
        approxEigVal = (v' * A * v);
        errorVec[k] = abs(max_eigVal - approxEigVal);  # absolute error

    end
    
    return approxEigVal, errorVec
end


function RQI(startingVec, A, N)
    "finds approximation of the eigenvalue that corresponds to 
    the initial eigenvector.
    - returns final approximation and error in each iteration "

    errorVec = zeros(N);
    v = startingVec / norm(startingVec);
    mu = (v' * A * v); # intitial eigenvalue, found through Rayleigh Quotient
    eigVals = eigen(A).values;
    closest_eigVal = eigVals[argmin(abs.(eigVals.-mu))]; # find the eigVal closest to initial guess
    for k = 1:N
        Ashifted = A - (mu * I(size(A,1))) + 1e-10 * I(size(A,1)); # add perturbation
        LU = lu(Ashifted);
        w = LU.U \ (LU.L \ (LU.P*v));
        v = w /norm(w);
        mu = (v' * A * v);
        closest_eigVal = eigVals[argmin(abs.(eigVals.-mu))]; # update closest eigVal
        errorVec[k] = abs(closest_eigVal - mu); 
    end
    return mu, errorVec
end


A1 = [1 2 3; 2 0 2; 3 2 9];
A2 = [1 2 4; 2 0 2; 3 2 9];
x0 = [1; 0; -1];
x0 = x0/norm(x0);



N = 10; # number of iterations
x = 1:N;


finalPM, errorVecPM = powerMethod(x0, A1, N)
yPM = errorVecPM.+ 1e-10

finalRQI1, errorVecRQI1 = RQI(x0, A1, N)
yRQI1 = errorVecRQI1 .+ 1e-10

finalRQI2, errorVecRQI2 = RQI(x0, A2, N)
yRQI2 = errorVecRQI2 .+ 1e-10

plot(x, yPM, yscale=:log10, label = "Power Method for A_1", linewidth=3, title = "Eigenvalue convergence of PM and RQI", xaxis="Iterations", yaxis="Absolute Error of Eigenvalue")
plot!(x, yRQI1, yscale=:log10, label="Rayleigh otient Iteration for A_1", linewidth=3)
plot!(x, yRQI2, yscale=:log10, label="Rayleigh Quotient Iteration for A_2", linewidth=3)
#savefig("EigvalErrorPlot.pdf")

