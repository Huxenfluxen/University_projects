using LinearAlgebra, Plots, LaTeXStrings, BenchmarkTools
include("schur_parlett.jl")
include("HW3_functions.jl")


### 3a)

A_3a = [1 4 4; 3 -1 3; -1 4 4];
f_3a=z->sin(z);
F_3a=schur_parlett(A_3a,f_3a);


### 3b)

A_3b=rand(100,100)+1im*rand(100,100);
A_3b=A_3b/norm(A_3b);
Nvec = range(0,200,20);

#### this function times using belapsed, so it takes a few minutes to run
schurTimes, naiveTimes = timeSchurVsNaive(A_3b,Nvec)

plot(Nvec, schurTimes, label="Schur-Parlett", xlabel = "N", ylabel = "CPU-time", linestyle=:dot, 
    color=:red, linewidth=3, title = "CPU-timing of Schur-Parlett VS Naive Version");
plot!(Nvec, naiveTimes, label="Naive", linestyle=:dash, color=:black, linewidth=3)
#savefig("schurTimes.png");