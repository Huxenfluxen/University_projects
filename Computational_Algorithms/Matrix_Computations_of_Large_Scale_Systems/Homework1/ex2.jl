using Pkg
using MatrixDepot, Random
using TickTock
using LinearAlgebra
using Plots
include("arnoldi.jl")

nn=780;
Random.seed!(0)
A=matrixdepot("wathen",nn,nn)
b = rand(size(A,1),1);
M = [5, 10, 20, 50, 100];

function runTimeArnoldi(method, iters, A, b, M)
    mLen = length(M);
    orthVec = zeros(1,mLen);
    timeVec = zeros(1,mLen);
    for i = 1:mLen
        tick()
        Q,H = arnoldi(A,b,M[i], method, iters);
        time = tok()
        timeVec[i] = time;
        orthVec[i] = norm(Q'*Q-I) ;
    end

    return orthVec, timeVec
end

"We run each method twice, saving the second run. 
This way the the order of the methods does not matter when timing them"

# runTimeArnoldi(classical_GS, 2, A, b, M );
# cgs2Orths, cgs2Times = runTimeArnoldi(classical_GS, 2, A, b, M );
# runTimeArnoldi(classical_GS, 3, A, b, M );
# cgs3Orths, cgs3Times = runTimeArnoldi(classical_GS, 3, A, b, M );
# runTimeArnoldi(classical_GS, 1, A, b, M );
# cgs1Orths, cgs1Times = runTimeArnoldi(classical_GS, 1, A, b, M );
# runTimeArnoldi(modified_GS, 1, A, b, M );
# mgsOrths, mgsTimes = runTimeArnoldi(modified_GS, 1, A, b, M );

"plotting the time vectors for the different methods:"
plot(M,cgs1Times', label = "single GS", seriestype=:path, marker=:circle, markersize=3, linewidth=3)
plot!(M,mgsTimes',seriestype=:path, marker=:circle, markersize=3, linewidth=3, title = "CPU Time of Arnoldi Method" , xaxis ="m = number of iterations", yaxis="time (s)" , label = "modified GS")
plot!(M,cgs2Times', seriestype=:path, marker=:circle, markersize=3, linewidth=3,label = "double GS")
plot!(M,cgs3Times', seriestype=:path, marker=:circle, markersize=3, linewidth=3, label = "triple GS")

#savefig("arnoldiTime.pdf")



