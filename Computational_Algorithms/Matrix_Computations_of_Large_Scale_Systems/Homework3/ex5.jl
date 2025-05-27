using LinearAlgebra, Random, BenchmarkTools
#include("ex5_functions.jl")
include("HW3_functions.jl")

Random.seed!(0)
n_5 = 256;
A_5 = randn(n_5, n_5)
A_5 = A_5 / norm(A_5);
A_5 = A_5 - 3I;
Ttrue = exp(A_5);


M=7;
J=7;

####    calculate errors
ScSqErrs, _ = errAndTimeScSq(A_5,M,J, Ttrue, false);

####     uncomment to time every iterations:
# _, ScSqTimes = errAndTimeScSq(A_5,M,J, Ttrue, true)  



###     uncomment to calculate cpu time for j=0, m=25:
# timeForTaylor = @belapsed (scalingSquaringTaylor($A_5,$25,$0));

