include("alpha_example.jl")



#################### functions EX 1 ####################

function errfun(A)
    return maximum(abs.(tril(A, -1)))
end


# Basic QR-method
function basic_QR(A)
    tolerance = 1e-10
    U = I(size(A, 1))
    tol = 1
    k = 0
    while tol > tolerance
        Q, R = qr(A)    # Basic QR factorization
        A = R*Q
        U = U*Q
        tol = errfun(A)
        k += 1 # number of iterations
    end
    return A, U, k
end

function find_convergence_rate(A)
    largest = 0
    eigvalsA = sort(eigvals(A), rev = true) # sort largest to smallest
    for i = 1:length(eigvalsA)-1
        l = eigvalsA[i+1]/eigvalsA[i]
        if l > largest
            largest = l
            #println("1: ", eigvalsA[i], " 2: ", eigvalsA[i+1])
        end
    end
    return largest
end

function QR_ItersAndPredictions(α_vec,m)
    tolerance = 1e-10
    predicted_k_vec = zeros(length(α_vec))
    k_vec = zeros(length(α_vec))
    for (i, α) in enumerate(α_vec)
        local A, β
        A = alpha_example(α, m)
        _, _, k_vec[i] = basic_QR(A)
        β = abs(find_convergence_rate(A))
        predicted_k_vec[i] = log(tolerance) / log(β)
    end
    return k_vec, predicted_k_vec
end




#################### functions EX 2 ####################

function efficient_hessenberg_red(A)
    "function corresponds to Algorithm 2 in Block 3.
    returns Hessenberg reduced A"
    n = size(A,1)
    for k=1:n-2
        x=A[k+1:n,k]
        ρ = sign(x[1]);
        α = ρ*norm(x);
        e1 = Matrix{Float64}(I, n-k, 1);
        z=x-α*e1;
        u=z/norm(z);
        A[k+1:n, k:n] -= 2 * u * (u' * A[k+1:n, k:n]) ;
        A[1:n, k+1:n] -= 2 * (A[1:n, k+1:n] * u) * u';
    end
    return A
end

function timeHessenberg(mVec)
    "computes CPU time of the naive Hessenberg and efficient Hessenberg"
    
    naiveTimes = zeros(length(mVec));
    efficientTimes = zeros(length(mVec));

    for (i,m) = enumerate(mVec)
        local A, H1, H2
        A=alpha_example(1,m);
        
        t0n = Int(time_ns())
        H1 = naive_hessenberg_red(A);
        naiveTimes[i] = (Int(time_ns())-t0n)*10^(-9)

        t0e = Int(time_ns())
        H2 = efficient_hessenberg_red(A);
        efficientTimes[i] = (Int(time_ns())-t0e)*10^(-9)
    end
    return naiveTimes, efficientTimes
end

function oneStep_ShiftedQR_errors(ϵVec, shifts)
    "computes error after one step of the shifted QR method
    given vectors of epsilon-values and shifts.
    Returns matrix hVals corresponding to table in 2d, ie:
        fist column = ϵ, 
        second column = error using first shift, 
        third column = error using second shift."

    hVals = zeros(length(ϵVec), 3);
    for (i,ϵ) = enumerate(ϵVec)
        local A
        hVals[i,1] = ϵ;
        A = [3 2; ϵ 1];
        for (j,shift) = enumerate(shifts)
            local H
            σ = shift*I(size(A,1))
            Q, R = qr(A-σ); 
            H = R*Q + σ;
            hVals[i,j+1] = abs(H[2,1]);
        end

    end
    return hVals
end


#################### functions EX 3 ####################


function naive_exp(A, N)
    B=A; 
    for i=1:N-1
        B=B*A
    end
    return B
end

function timeSchurVsNaive(A,Nvec)
    
    n=length(Nvec);
    schurTimes = zeros(n);
    naiveTimes = zeros(n);

    for (i,N) = enumerate(Nvec)
        local f
        f = z -> z^N;
        schurTimes[i] = @belapsed (schur_parlett($A,$f))
        naiveTimes[i] = @belapsed (naive_exp($A,$N))
    end
    return schurTimes, naiveTimes
end




#################### functions EX 5 ####################

function scalingSquaringTaylor(A, m, j)
    A = A/(2^j);
    n = size(A,1)
    T = I(n);
    Ak = I(n); 
    
    for k = 1:m-1
        Ak = (A * Ak) / (k) 
        T += Ak             
    end
    for _=1:j
        T = T*T
    end
    return T
end

function errAndTimeScSq(A,M,J, Ttrue, timing = false)
    logErrs = zeros(M,J+1);
    ScSqTimes = zeros(M,J+1);
    for m=1:M
        for j=0:J
            local Tapprox
            Tapprox = scalingSquaringTaylor(A,m,j);
            logErrs[m,j+1] = log10(norm(Tapprox-Ttrue));
            if timing
                ScSqTimes[m,j+1] = @belapsed (scalingSquaringTaylor($A,$m,$j));
            end
        end
    end
    return logErrs, ScSqTimes
end