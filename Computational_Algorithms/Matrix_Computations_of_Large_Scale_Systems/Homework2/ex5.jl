using MAT, Plots, LinearAlgebra
include("GMRES.jl")
include("CG.jl")

A5=matread("cgn_illustration.mat")["A"]; # size 100000x100000
b5=matread("cgn_illustration.mat")["b"]; # size 100000x1

mGMRES = 80;
mCGN = 400;


_, _, timeVecGMRES = GMRES(A5, b5, mGMRES, computeResidual=true, timing=true)
_, relResNorm5 =  GMRES(A5, b5, mGMRES, computeResidual=true, timing=false)
relResNorm5 = relResNorm5 .* norm(b5); # dont want the normalized res here


_, resCGN, _ = cgn(A5, b5, mCGN, true);
_, _, timeCGN = cgn(A5, b5, mCGN, false);



plotGMRESiters = plot(1:mGMRES, relResNorm5, xlim = [0,80], yscale = :log10, xaxis = "Iteration", yaxis = "||Ax-b||_2", title = "Residual Norm as a function of iterations", linewidth=3, label = "GMRES", color=:black, linestyle=:dash);
plot!(1:mCGN, resCGN, label = "CGN", linewidth=3, color = :red)
display(plotGMRESiters)
savefig("ex5iters.pdf")

plotGMREStime = plot(timeVecGMRES.*10^(-9), relResNorm5,xlim = [0,2.5], yscale = :log10, xaxis = "CPU-time (seconds)", yaxis = "||Ax-b||_2", title = "Residual Norm as a function of time", linewidth=3, label = "GMRES", color=:black, linestyle=:dash);
plot!(timeCGN.*10^(-9), resCGN, label = "CGN", linewidth=3, color = :red)
display(plotGMREStime)
savefig("ex5times.pdf")


