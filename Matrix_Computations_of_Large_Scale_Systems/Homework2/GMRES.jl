using LinearAlgebra, SparseArrays, Plots
using BenchmarkTools


### a) Plotting the norm of the relative error and the relative residual
### Function to compute the relative errors and relative rel_residuals
### compared to the true value found through backslash operation
function GMRES(A, b, M; computeResidual=true, timing=false)
    
    timeVec = timing ? zeros(M) : nothing; # times each iteration
    

    n = length(b)
    Q = zeros(ComplexF64, n, M + 1)
    H = zeros(ComplexF64, M + 1, M)
    nb = norm(b)
    Q[:, 1] = b / nb
    X = zeros(ComplexF64, size(A, 1), M)

    e1 = zeros(ComplexF64, M + 1)
    e1[1] = nb  # the e1 vector with norm of b as first element (||b||e1)
    t0 = Int(time_ns()); # Time in nano seconds
    for m in 1:M
        w = A * Q[:, m]
        h, β, z = double_GS(Q, w, m)
        H[1:(m + 1), m] = [h; β]
        Q[:, m + 1] = z / β
        # up to here the method is the same as Arnoldi

        zStar = H[1:(m + 1), 1:m] \ e1[1:m+1]  # solve least-squares problem, fast with backslash for small matrix
        X[:, m] = Q[:, 1:m] * zStar; # approximation of x_m
        timing ? timeVec[m] = Int(time_ns())-t0 : nothing;
    end
    if computeResidual
        rel_residuals = mapslices(norm, A*X .- b, dims=1) ./ norm(b)
        x_true = A\b
        rel_errors = mapslices(norm, x_true .- X, dims=1) ./ norm(x_true)
        return rel_errors', rel_residuals', timeVec
    else
        x = X[:, M]
        return x
    end
end




function time_GMRES(N, M, α_vals)
    alphas = length(α_vals)
    times = zeros(alphas*length(M), length(N))
    resNorms = zeros(alphas*length(M), length(N))
    times_backslash = zeros(alphas, length(N))
    resNorms_backslash = zeros(alphas, length(N))
    computeResidual = false
    timing = false # this is done with benchmark instead
    
    for (i, n) in enumerate(N)
        println(i)
        b = rand(n,1)
        j = 0
        k = 0
        for α in α_vals
            println("alpha")
            k += 1
            A = sprand(n, n, 0.5)
            A+= α*sparse(I, n, n)
            A /= norm(A, 1)
            times_backslash[k, i] = @belapsed ($A\$b)
            x_true = A\b
            resNorms_backslash[k, i] = norm(A*x_true - b)
            for m in M
                j += 1
                times[j, i] = @belapsed (GMRES($A, $b, $m, computeResidual=$computeResidual, timing=$timing))
                x_m = GMRES(A, b, m; computeResidual=computeResidual, timing=false)
                resNorms[j, i] = norm(A*x_m - b)
            end
        end
    end
    return times, resNorms, times_backslash, resNorms_backslash
end



function double_GS(Q::Matrix{ComplexF64}, w::Vector{ComplexF64}, k::Int)
    "double GS from hw1"
    Q = Q[:, 1:k]
    h = Q' * w
    y = w - Q * h
    
    g = Q' * y
    y = y - Q * g
    h = h + g
    β = norm(y)
    return h, β, y
end


function convergence_rate_GMRES(λ, center, radius)
    constant = (norm(λ-center)+radius)/(λ)
    convFactor = radius/center
    return convFactor, constant
end



