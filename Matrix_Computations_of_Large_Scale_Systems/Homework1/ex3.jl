using LinearAlgebra, MatrixDepot# for: eig, norm, etc
using Arpack # Built in Arnoldi and Lancsos iterations, to access "true" eigenvalues for sparse matrices
using Plots
using LaTeXStrings
include("arnoldi.jl") # this is also where the GS functions are

function powerMethod_findK(startingVec, A, N) 
    "finds approximation of the eigenvector for the largest 
    eigenvalue, and vector of absolute error at each iteration.
    - This version returns normalized iterates K = [b, Ab, A^2b, ...] of length N"
    
    K = zeros(size(A,1), N)
    v = startingVec/norm(startingVec)
    #l=0
    for k = 1:N
        K[:,k] = v';
        w = A * v;
        v = w / norm(w);  # normalize v
        #l = (v' * A * v) 
    end
    return K
end

# Making this a function since julia likes functions when there are big computations happening
function plot_eigvals_arnoldi_and_PM_generated_krylov(A, b, M) 

    K_M = powerMethod_findK(b, A, M)
    _, h = arnoldi(A3, b, M, classical_GS, 2)
    p = plot(title= L"Approximated from \Left(K_m^T K_m\Right )^{-1}K_m^T A K_m and Arnoldi method", xlabel="m", ylabel="Real part of eigenvals approx", ylims=(0, 500))

    for m in 1:M
        K_m =K_M[:, 1:m]
        matrix_PM = (K_m'*K_m) \ (K_m'*A*K_m)
    
        PM_eigvals_real = real(eigen(matrix_PM).values) # We plot the real part of the eigvals
        arnoldi_eigvals_real = real(eigen(h[1:m, 1:m]).values)
        
        scatter!(fill(m, length(PM_eigvals_real)), PM_eigvals_real, label=(m == 1 ? "Primitive variant of  Arnoldi" : ""), marker=:x, markersize=3)
        scatter!(fill(m, length(arnoldi_eigvals_real)), arnoldi_eigvals_real, label=(m == 1 ? "Arnoldi Method" : ""), marker=:circle, markersize=3)
        println("m = ", m, " done")
    end
    return p
end

Random.seed!(0)
nn3=780;
# nn3 = 10;
A3=matrixdepot("wathen",nn3,nn3)
b = rand(size(A3,1),1);
M = 80; # number of iterations, m= 1,2,...,M

#p = plot_eigvals_arnoldi_and_PM_generated_krylov(A3, b, M)
plot!(p, title="Eigenvalue approximations")  # Update the title
savefig(p, "EigvalPlot_PM_vs_Arnoldi.pdf")


# display(plot)

