using LinearAlgebra, Plots, LaTeXStrings, BenchmarkTools

include("alpha_example.jl")
include("HW3_functions.jl")

### plotting number of iterations and predicted iterations vs α
α_vec = range(1, 1e5, 100)
m=20;
k_vec, predicted_k_vec = QR_ItersAndPredictions(α_vec,m);


plot(α_vec, k_vec, xaxis=:log10, label="Number of iterations", xlabel="α", ylabel="Number of terations, k", 
     title="Number of iterations vs α", color=:red, legend=:bottomright)
plot!(α_vec, predicted_k_vec, label="Predicted number of iterations", linestyle=:dashdot, color=:gray)
xticks = [10, 100, 1000, 10000, 100000]  
xtick_labels = [L"10^1", L"10^2", L"10^3", L"10^4", L"10^5"]  
plot!(xticks=(xticks, xtick_labels))
#savefig("ex1itersAlpha")