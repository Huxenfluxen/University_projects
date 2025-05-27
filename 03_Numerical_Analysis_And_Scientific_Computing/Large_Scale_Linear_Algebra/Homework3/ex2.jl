using BenchmarkTools
include("naive_hessenberg_red.jl")
include("alpha_example.jl")
include("HW3_functions.jl")

### 2a)

# see report


### 2b)

# see HW3_functions.jl



### 2c)
mVec = [10 100 200 300 400];
# computes CPU time of the naive hessenberg and efficient hessenberg algos
naiveTimes, efficientTimes = timeHessenberg(mVec);
 


#### 2d)
ϵVec = [0.1 ./ (10 .^ (0:9)); 0];
shifts = [0,1];
# computes the error of the one step shifted QR
hVals = oneStep_ShiftedQR_errors(ϵVec, shifts);






