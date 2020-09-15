using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables
using ForwardDiff # for bonus at the very end

function wrapall()

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

function mlogit_with_Z(theta, X, Z, y)
        
        alpha = theta[1:end-1]
        gamma = theta[end] # Indexing Z variables at the end
        K = size(X,2) # N Variables
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J) # Same as PS2.jl ...
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

startvals = [2*rand(7*size(X,2)).-1; .1] # 7 == J - 1
td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
# run the optimizer
theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
theta_hat_mle_ad = theta_hat_optim_ad.minimizer
# evaluate the Hessian at the estimates
H  = Optim.hessian!(td, theta_hat_mle_ad)
theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println([theta_hat_mle_ad theta_hat_mle_ad_se]) # these standard errors match Stata

function nested_logit_with_Z(theta, X, Z, y, nesting_structure)
        
        alpha = theta[1:end-3]
        lambda = theta[end-2:end-1]
        gamma = theta[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [repeat(alpha[1:K],1,length(nesting_structure[1])) repeat(alpha[K+1:2K],1,length(nesting_structure[2])) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        lidx  = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            if j in nesting_structure[1]
                lidx[:,j] = exp.( (X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[1] )
            elseif j in nesting_structure[2]
                lidx[:,j] = exp.( (X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[2] )
            else
                lidx[:,j] = exp.(zeros(N))
            end
        end
        for j=1:J
            if j in nesting_structure[1]
                num[:,j] = lidx[:,j].*sum(lidx[:,nesting_structure[1][:]];dims=2).^(lambda[1]-1)
            elseif j in nesting_structure[2]
                num[:,j] = lidx[:,j].*sum(lidx[:,nesting_structure[2][:]];dims=2).^(lambda[2]-1)
            else
                num[:,j] = lidx[:,j]
            end
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end
nesting_structure = [[1 2 3], [4 5 6 7]] # This is a list with the mentioned nested structure

startvals = [2*rand(2*size(X,2)).-1; 
		1; 1; .1 # Unclear
		]    
# Same as before
td = TwiceDifferentiable(theta -> nested_logit_with_Z(theta, X, Z, y, nesting_structure), startvals; autodiff = :forward)
# run the optimizer
nlogit_theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
nlogit_theta_hat_mle_ad = nlogit_theta_hat_optim_ad.minimizer
# evaluate the Hessian at the estimates
H  = Optim.hessian!(td, nlogit_theta_hat_mle_ad)
nlogit_theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println([nlogit_theta_hat_mle_ad nlogit_theta_hat_mle_ad_se]) # these standard errors match Stata

    return nothing
end
wrapall()
