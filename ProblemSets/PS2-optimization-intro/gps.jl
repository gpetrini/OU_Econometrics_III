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
println("β_lm estimates:", b_lm)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
b_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("β_glm estimates:", b_glm)



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
