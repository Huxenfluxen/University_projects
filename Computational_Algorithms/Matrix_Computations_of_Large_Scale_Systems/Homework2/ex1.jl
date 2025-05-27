using Random, SparseArrays, LinearAlgebra, Plots
include("GMRES.jl")


α = [1,5,10,100];
m1 = 100
n = 100
b = rand(n, 1)  

"Create the different A matrices with α"
A1 = sprand(n, n, 0.5) + α[1] * I(n) 
A1 = A1 / norm(A1, 1)

A2 = sprand(n, n, 0.5) + α[2] * I(n) 
A2 = A2 / norm(A2, 1)

A3 = sprand(n, n, 0.5) + α[3] * I(n) 
A3 = A3 / norm(A3, 1)

A4 = sprand(n, n, 0.5) + α[4] * I(n) 
A4 = A4 / norm(A4, 1)


"results of GMRES convergence using different α: "
relError1, relResNorm1 = GMRES(A1, b, m1, computeResidual=true)
relError2, relResNorm2 = GMRES(A2, b, m1, computeResidual=true)
relError3, relResNorm3 = GMRES(A3, b, m1, computeResidual=true)
relError4, relResNorm4 = GMRES(A4, b, m1, computeResidual=true)


"create the disks"
c1 = (0.00035, 0.0);  c2 = (0.0017, 0.0);  c3 = (0.00285, 0.0);  c4 = (0.008, 0.0);  
r1 = 0.0014;          r2 = 0.0012;         r3 = 0.00105;         r4 = 0.0003;         

θ = LinRange(0, 2π, 100)  

x1 = c1[1] .+ r1 .* cos.(θ);  y1 = c1[2] .+ r1 .* sin.(θ);  
x2 = c2[1] .+ r2 .* cos.(θ);  y2 = c2[2] .+ r2 .* sin.(θ);  
x3 = c3[1] .+ r3 .* cos.(θ);  y3 = c3[2] .+ r3 .* sin.(θ);  
x4 = c4[1] .+ r4 .* cos.(θ);  y4 = c4[2] .+ r4 .* sin.(θ);  

f1, con1 = convergence_rate_GMRES(0.01, c1[1], r1);
f2, con2 = convergence_rate_GMRES(0.01, c2[1], r2);
f3, con3 = convergence_rate_GMRES(0.01, c3[1], r3);
f4, con4 = convergence_rate_GMRES(0.01, c4[1], r4);


"convergence bounds calculated from disks"
convBound1 = 1.05 .* 2 .^(0:m1-1);
convBound2 = con2 .* (f2) .^ (0:m1-1);
convBound3 = con3 .* (f3) .^ (0:m1-1);
convBound4 = con4 .* (f4) .^ (0:m1-1);




#### 1A plots


p1 = plot(1:m1, relResNorm1, yscale = :log10, xlabel="Iteration", ylabel="Error", title="α = $(α[1])", linewidth=3, label = "||Ax-b||/||b||", legendfontsize=7, legend = :bottomleft, color=:red)
    plot!(1:m1, relError1, yscale = :log10, linewidth=3, label = "||x-x*||/||x*||", color = :black, linestyle=:dash)
    #plot!(1:m, convBound1, linewidth=3, label = "Convergence bound", color=:blue, linestyle=:dash)

p2 = plot(1:m1, relResNorm2, yscale = :log10, ylim = [10^(-17), 10], xlabel="Iterations", ylabel="Error", title="α = $(α[2])", linewidth=3, label = "||Ax-b||/||b||", legend=:topright, legendfontsize=7, color=:red)
    plot!(1:m1, relError2, yscale = :log10, linewidth=3, label = "||x-x*||/||x*||", color = :black, linestyle=:dash)
    plot!(1:m1, convBound2, linewidth=3, label = "theory prediction", color=:blue, linestyle=:dash)

p3 = plot(1:m1, relResNorm3, yscale = :log10, ylim = [10^(-17), 10], xlabel="Iterations", ylabel="Error", title="α = $(α[3])", linewidth=3, label = "||Ax-b||/||b||", legendfontsize=7, color=:red)
    plot!(1:m1, relError3, yscale = :log10, linewidth=3, label = "||x-x*||/||x*||", color = :black, linestyle=:dash)
    plot!(1:m1, convBound3, linewidth=3, label = "theory prediction", color=:blue, linestyle=:dash)

p4 = plot(1:m1, relResNorm4, yscale = :log10, ylim = [10^(-17), 10], xlabel="Iterations", ylabel="Error", title="α = $(α[4])", linewidth=3, label = "||Ax-b||/||b||", legendfontsize=7, color=:red)
    plot!(1:m1, relError4, yscale = :log10, linewidth=3, label = "||x-x*||/||x*||", color = :black, linestyle=:dash)
    plot!(1:m1, convBound4, linewidth=3, label = "theory prediction", color=:blue, linestyle=:dash)

p = plot(p1,p2,p3,p4, layout=(2, 2));
plot(p)
display(p)
savefig("convergenceGMRES.pdf")






### 1B plots

"plot eigenvalues for all α: ";
s1 = scatter(eigvals(Matrix(A1)), marker=:o, label="Eigvals", title="α = 1");
plot!(x1, y1, aspect_ratio=:equal, label="Localization disk")
s2 = scatter(eigvals(Matrix(A2)), marker=:o, title="α = 5", label="Eigvals");
plot!(x2, y2, aspect_ratio=:equal, label="Localization disk")
s3 = scatter(eigvals(Matrix(A3)), marker=:o, title="α = 10", label="Eigvals");
plot!(x3, y3, aspect_ratio=:equal, label="Localization disk")
s4 = scatter(eigvals(Matrix(A4)), marker=:o, title="α = 100", label="Eigvals");
plot!(x4, y4, aspect_ratio=:equal, label="Localization disk")
s = plot(s1,s2,s3,s4, layout=(2,2))
savefig("eigvalsAlpha.pdf")
display(s)



#### 1c) Benchmarking the GMRES method - takes quite some time!
        # uncomment last line to do this

N = [200 500 1000]
M = [5 10 20 50 100]
α_vals2 = [1 100]

# times, resNorms, times_backslash, resNorms_backslash = time_GMRES(N, M, α_vals2)
