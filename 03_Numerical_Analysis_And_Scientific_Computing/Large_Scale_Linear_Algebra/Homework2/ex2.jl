using SparseArrays
include("GMRES.jl")
include("CG.jl")


n = 100
e = ones(n-1)
A = -spdiagm(-1 => e, 0 => -6*ones(n), 1 => e)
b = kron(ones(Int(n/2)), [0; 1]) + ones(n)

m = 30

_ , gmresNorm = GMRES(A, b, m, computeResidual=true);
gmresNorm = gmresNorm .* norm(b); # dont want the normalized res here

X, cgNorm, _ = cg(A, b, m); # X contains all p iters from cg


# cglsNorm = zeros(m)
# for p = 1:m
#     try
#         Xp = X[:, 1:p]
#         AX = A * Xp
#         AXA = AX' * AX
#         AXb = AX' * b
#         z = AXA \ AXb
#         x_m = Xp * z
#         cglsNorm[p] = norm(A * x_m - b)
#     catch err
#         # Skip iteration if there is a SingularException error...
#         println("skipped iteration", p)
#         cglsNorm[p] = NaN  
#     end
# end

cglsNorm = zeros(m)
X_ls = zeros(n, m)
for p = 1:m
    # try
    Xp = X[:, 1:p]
    AX = A * Xp
    AXt = AX'
    # AXA = (Xp'*A')*(A*Xp) # Maybe faster?
    # AXb = XP'*(A*b) # Maybe faster?
    AXA = AXt * AX
    AXb = AXt * b
    
    # Solve the normal equations
    try
        z = AXA \ AXb
        x_ls = Xp * z
        cglsNorm[p] = norm(A * x_ls - b)
    catch
        z = (AXA + 1e-13 * I(p)) \ AXb
        x_ls = Xp * z
        cglsNorm[p] = norm(A * x_ls - b)
    end
    println(p)
end


p = scatter(1:m, cgNorm, yscale =:log10, label = "Method 1: CG", marker =:star, legend = :bottomleft, title = "Convergence", yaxis = "||Ax-b||_2", xaxis = "Iteration p")
    scatter!(1:m, gmresNorm, yscale =:log10, label = "Method 2: GMRES", marker =:o)
    scatter!(1:m, cglsNorm, yscale =:log10, label = "Method 3: CG with LSQ", marker =:x)
display(p)
#savefig("cgls.pdf")


# pp1 = scatter(1:m, cgNorm, xlim = [0,2], ylim = [10^(0.6),10^(0.64)] , yscale =:log10, label = "Method 1: CG", marker =:star, legend = :bottomleft, title = "Iteration 1", yaxis = "||Ax-b||_2",  legendfontsize=5)
#     scatter!(1:m, gmresNorm, yscale =:log10, label = "Method 2: GMRES", marker =:o)
#     scatter!(1:m, cglsNorm, yscale =:log10, label = "Method 3: CG with LSQ", marker =:x)

# pp2 = scatter(1:m, cgNorm, xlim = [1,3], ylim = [10^(-0.818),10^(-0.814)] , yscale =:log10, label = "Method 1: CG", marker =:star, legend = :bottomleft, title = "Iteration 2", yaxis = "||Ax-b||_2",  legendfontsize=6)
#     scatter!(1:m, gmresNorm, yscale =:log10, label = "Method 2: GMRES", marker =:o)
#     scatter!(1:m, cglsNorm, yscale =:log10, label = "Method 3: CG with LSQ", marker =:x)


# pp3 = scatter(1:m, cgNorm, xlim = [2,4], ylim = [10^(-1.62),10^(-1.58)] , yscale =:log10, label = "Method 1: CG", marker =:star, legend = :bottomleft, title = "Iteration 3", yaxis = "||Ax-b||_2",  legendfontsize=6)
#     scatter!(1:m, gmresNorm, yscale =:log10, label = "Method 2: GMRES", marker =:o)
#     scatter!(1:m, cglsNorm, yscale =:log10, label = "Method 3: CG with LSQ", marker =:x)


# pp4 = scatter(1:m, cgNorm, xlim = [3.1,4.9], ylim = [10^(-2.38),10^(-2.34)] , yscale =:log10, label = "Method 1: CG", marker =:star, legend = :bottomleft, title = "Iteration 4", yaxis = "||Ax-b||_2",  legendfontsize=6)
#     scatter!(1:m, gmresNorm, yscale =:log10, label = "Method 2: GMRES", marker =:o)
#     scatter!(1:m, cglsNorm, yscale =:log10, label = "Method 3: CG with LSQ", marker =:x)


# plot(pp1,pp2,pp3,pp4, layout=(2,2))
# savefig("cglsZoomed.pdf")

ss = scatter(eigvals(Matrix(A)), marker=:o, label="Eigvals", title="α = 1");
display(ss)