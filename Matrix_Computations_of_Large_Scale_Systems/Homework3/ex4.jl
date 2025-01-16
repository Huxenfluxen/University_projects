using LinearAlgebra, Plots

p = ϵ -> [exp(pi) (exp(pi + ϵ)-exp(pi))/ϵ; 0 exp(pi + ϵ)]
A = ϵ -> [pi 1; 0 pi + ϵ]

ϵ_vec = 10 .^range(-10, -1, 90)
res_vec = zeros(length(ϵ_vec))
for (i, ϵ) in enumerate(ϵ_vec)
    local V, D
    eig = eigen(A(ϵ))  
    V = eig.vectors  # Matrix of eigenvectors
    D = eig.values 
    F = Diagonal(exp.(D))
    println("V: ", V)
    println("X: ", [1 1; 0 ϵ])
    
    FJ=V*F*inv(V)

    res_vec[i] = norm(FJ - p(ϵ))
    println(res_vec[i])
end

tick_positions = 10.0 .^ (-10:-1)
tick_labels = ["10^{-$(i)}" for i in 10:-1:1]

plot(ϵ_vec, res_vec,
 xlabel="ε", xscale=:log10, xticks=(tick_positions, tick_labels),
 ylabel="||f(A) - F||", yscale=:log10, yaxis=[10^(-15), 0] ,
 grid=true, lw=1, marker=:circle,
 legend=false, title="Residual vs ε")

 plot!(grid=:on, gridalpha=1, minorgrid=true, minorgridalpha=0.15)
#savefig("Residual_vs_epsilon_exp(A)")
