using MAT
using Distances
using Random
using LinearAlgebra
using Statistics
using Plots

Bwedge_file = "C:/Users/ville/MinaJuliaProgram/SF2524MatrixComputations/Bwedge.mat"
Bwedge = matread(Bwedge_file)["B"]

### Loading the Bwedge matrix
# Bwedge=matread("Bwedge.mat")["B"];

### Computing eigenvalues, using mean as a "good enough" centre of circle, finding outliers
eig_vals_Bwedge = eigen(Bwedge).values
eig_mean = mean(eig_vals_Bwedge)
# outlier_indicators = abs.(eig_vals_Bwedge .- eig_mean) .> 10 # or use: partialsortperm(abs.(...), 1:3, rev=true)
outlier_indicators = partialsortperm(abs.(eig_vals_Bwedge .- eig_mean), 1:3, rev=true)
eig_vals_outliers = eig_vals_Bwedge[outlier_indicators]




### Create the circles
c_1 = -4.5
ρ_1 = 14
c_2 = -24.5 .- 13im
ρ_2 = 26
c_3 = -24.5 .+ 13im
ρ_3 = 26
θ = range(0, 2π, length=100)
circ_1 = ρ_1*exp.(1im*θ) .+ c_1
circ_2 = ρ_2*exp.(1im*θ) .+ c_2
circ_3 = ρ_3*exp.(1im*θ) .+ c_3

### Plot all eigenvalues, outliers and circles!
plot(title="Outliers of Bwedge", legend=:outertopright)
scatter(eig_vals_Bwedge, label="Eigenvalues of Bwedge", marker=:circle, markersize=3,title="Eigenvalues of Bwedge")
scatter!(eig_vals_outliers, label="Eigenvalues of fastests convergence with Arnoldi method", marker=:circle, markersize=10, color=:red)
scatter!([-4.5, - 24.5 + 13im, - 24.5 - 13im], label="Centers of discs", marker=:star, markersize=10, color=:red)
annotate!(-4.5, -3, text("c1"))
annotate!(-24.5, 16, text("c3"))
annotate!(-24.5, -16, text("c2"))
annotate!(real(eig_vals_outliers[2])+2, imag(eig_vals_outliers[2])+1, text("λ2", :left))
annotate!(real(eig_vals_outliers[3])+2, imag(eig_vals_outliers[3])-1, text("λ3", :left))
annotate!(real(eig_vals_outliers[1])+2, imag(eig_vals_outliers[1])-1, text("λ1", :left))
plot!(circ_1, label="Disc excluding λ1", markersize=3)
plot!(circ_2, label="Disc excluding λ2", markersize=3)
plot!(circ_3, label="Disc excluding λ3", markersize=3)
plot!(size=(800, 600))
#savefig("eig_vals_Bwedge.pdf")

### Convergence rates (error is to the power of m-1)
α_1 = ρ_1/abs(eig_vals_outliers[1] - c_1)
α_2 = ρ_2/abs(eig_vals_outliers[2] - c_2)
α_3 = ρ_3/abs(eig_vals_outliers[3] - c_3)

### Want the convergence value to show in the plot somehow!
annotate!([(real(eig_vals_outliers[1]), imag(eig_vals_outliers[1]), text("α1 = $α_1", :left, 10)),
           (real(eig_vals_outliers[2]), imag(eig_vals_outliers[2]), text("α2 = $α_2", :left, 10)),
           (real(eig_vals_outliers[3]), imag(eig_vals_outliers[3]), text("α3 = $α_3", :left, 10))])

### Generate figure with Ritz-values for Arnoldi method
### Based on a) we expect fast convergence to eigenvalue λ_1 = -47
### We expect about 20 or 21 iterations to reach an error of 10^-10
M = [2 3 8 10 20 30 40]
include("arnoldi.jl")

Random.seed!(0)
b = rand(ComplexF64, size(Bwedge,1),1);
_, h = arnoldi(Bwedge, b, maximum(M), classical_GS, 2)

p = plot(layout=(4, 2), size=(1000, 600), title="Approximated Eigenvalues of Bwedge")

for (i, m) in enumerate(M)
    arnoldi_eigvals = eigen(h[1:m, 1:m]).values
    scatter!(p[i], arnoldi_eigvals, label="m = $m", marker=:circle, markersize=5)
    scatter!(p[i], eig_vals_outliers, label="True eigvals", marker=:x, markersize=3)
end

savefig(p, "EigvalPlot_Arnoldi.pdf")

iterations = 40
### Error vector of the outlier
errorMat = zeros(iterations, 1)
for m = 1:iterations
    arnoldi_eigvals = eigen(h[1:m, 1:m]).values
    arnoldi_eigval_outliers = arnoldi_eigvals[1]

    # closest_eigVal = eig_vals_outliers[argmin.(abs.(eig_vals_outliers.-arnoldi_eigval_outliers))]
    errorMat[m] = abs.(eig_vals_outliers[1] - arnoldi_eigval_outliers)
end
# α_1_values = [α_1^m for m in 1:iterations]
plot(1:iterations, errorMat, yscale=:log10, title="Convergence of furthest eigenvalue outlier", xaxis="iterations", yaxis="absolute error")
# plot!(1:iterations, α_1_values, label="α_1^m", linestyle=:dash)
savefig("ErrorFurthestOutlier.png")


### Arnoldi shift

m = 20
eigen_val = -9.8 + 2im
eigval_true = eig_vals_Bwedge[argmin(abs.(eig_vals_Bwedge .- eigen_val))]
error_rates = zeros(3,1)
σ = [-10 -7+2im -9.8+1.5im]
for i = 1:3
    _, H = arnoldi_shift(Bwedge, b, m, σ[i])
    eigens_shift = 1 ./ (eigen(H[1:m, 1:m]).values) .+ σ[i]
    # error_rates[i] = minimum(abs.(eigens_shift .- eigen_val))
    eig_closest = eigens_shift[argmin(abs.(eigens_shift .- eigval_true))]
    error_rates[i] = abs(eig_closest - eigval_true)
end