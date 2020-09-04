using Random, Statistics, FreqTables
using LinearAlgebra, Optim, GLM
using DataFrames, CSV, HTTP


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println("β_lm estimates:", bols_lm)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit(β, X, y)

    # your turn
    diff_u = X*β
    loglike = sum(y.*diff_u - log.(1 .+ exp.(diff_u)))
    loglike = -loglike

    return loglike
end

β_llk = optimize(β -> logit(β, X,y),
                 rand(size(X,2)),
                 # LBFGS(),
                 Newton(),
                 Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(β_llk.minimizer)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
b_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("β_glm estimates:", b_glm)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==10,:occupation] .= 9
df[df.occupation.==11,:occupation] .= 9
df[df.occupation.==12,:occupation] .= 9
df[df.occupation.==13,:occupation] .= 9
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(β, X, y)

    # your turn
    diff_u = X*β
    mloglike = [(y.==i).*diff_u - log.(1 .+ exp.(diff_u)) for i in unique(y)]
    mloglike = sum(mloglike)
    mloglike = -mloglike

    return mloglike
end


β_mllk = optimize(β -> mlogit(β, X,y),
                 rand(size(X,2)), #,2?
                 LBFGS(),
                 Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(β_mllk.minimizer)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
