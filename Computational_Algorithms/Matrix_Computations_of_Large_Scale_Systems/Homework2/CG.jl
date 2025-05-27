function cg(A, b, N)
    T = eltype(b)  
    x = zeros(T, size(b)) 
    r = vec(b)  
    p = r
    n = length(b)
    rv = zeros(T, N)
    tv = zeros(N)    
    t0 = Int(time_ns())  
    X = zeros(n,N);
    for k = 1:N
        rr = r' * r
        Ap = A*p; 
        alpha = rr / (p' * (Ap))  
        x += alpha * p 
        X[:,k] = x;            
        r -= alpha * (Ap)        
        beta = (r' * r) / rr       
        p = r + beta * p            
        rv[k] = norm(A * x - b)     
        tv[k] = Int(time_ns()) - t0 
    end

    return X, rv, tv
end



function cgn(A, b, N, compute_residual::Bool)
    T = eltype(b)  
    x = zeros(T, size(b)) 
    r = A'* vec(b)  # gcn modification: multiply A' by b 
    p = r
    n = length(b)
    rv = compute_residual ? zeros(T, N) : nothing
    tv = zeros(N)    
    t0 = Int(time_ns())  # Time in nanoseconds

    for k = 1:N
        rr = r' * r
        Ap = A'*(A*p); # first matrix-vector product
        alpha = rr / (p' * (Ap))  #second matrix-vector product
        x += alpha * p             
        r -= alpha * (Ap)        
        beta = (r' * r) / rr       
        p = r + beta * p            
        if compute_residual
            rv[k] = norm(A * x - b)     # Residual norm 
        end
        tv[k] = Int(time_ns()) - t0 # Timestamp
    end

    return x, rv, tv
end



