using SparseArrays


function modified_GS(Q,w,k, iters)
    y=w
    q=Q[:,1];
    h = zeros(k,1)
    for i=1:k
        q=Q[:,i]; 
        h[i] = q' * y;
        y = y-q*h[i];
    end
    β = norm(y);
    return h, β, y
end


function classical_GS(Q::Matrix{ComplexF64}, w::Vector{ComplexF64}, k::Int, iters::Int)#(Q,w,k,iters)
    Q = Q[:, 1:k]
    h = Q'*w;
    y = w-Q*h;
    β = norm(y)
    
    for i = 1:iters-1 # iters refers to double and triple CGS
        #y = y/β;
        g = Q'*y;
        y = y-Q*g;
        h = h+g;
        β = norm(y)
    end
    
    return h, β, y
end


#function arnoldi(A::Matrix{ComplexF64}, b::Matrix{ComplexF64}, m::Int64, my_hw1_gs::Function, iters::Int64)#(A,b,m,my_hw1_gs,iters)
function arnoldi(A,b,m,my_hw1_gs,iters)
    n=length(b);
    Q=zeros(ComplexF64, n,m+1);
    H=zeros(ComplexF64, m+1,m);
    Q[:,1]=b/norm(b);

    for k=1:m
        w=A*Q[:,k]; # Matrix-vector product with last element
        # Orthogonalize w against columns of Q.
        # Implement this function or replace call with code for orthogonalizatio
        h,β,z=my_hw1_gs(Q,w,k, iters);
        #Put Gram-Schmidt coefficients into H
        H[1:(k+1),k]=[h;β];
        # normalize
        Q[:,k+1]=z/β;
    end
    return Q,H
end


function arnoldi_shift(A::Matrix{ComplexF64}, b::Matrix{ComplexF64}, m::Int64, μ)
    n=length(b);
    Q=zeros(ComplexF64, n,m+1);
    H=zeros(ComplexF64, m+1,m);
    Q[:,1]=b/norm(b);
    B = A - μ*sparse(I,n,n)
    LU = lu(B)

    for k=1:m
        w=LU.U\(LU.L\Q[:,k]); # Matrix-vector product with last element
        # Orthogonalize w against columns of Q.
        # Implement this function or replace call with code for orthogonalizatio
        h,β,z=classical_GS(Q,w,k,2);
        #Put Gram-Schmidt coefficients into H
        H[1:(k+1),k]=[h;β];
        # normalize
        Q[:,k+1]=z/β;
    end
    return Q,H
end

