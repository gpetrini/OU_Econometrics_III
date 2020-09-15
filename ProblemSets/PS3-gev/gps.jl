### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 7085c940-f389-11ea-1dec-59018061e9b5
using HTTP

# ╔═╡ fb93f836-f389-11ea-26e6-678b961ba787
using LinearAlgebra, GLM, Optim, ForwardDiff

# ╔═╡ fbcefcc6-f389-11ea-2258-6fa3ead65683
using Random, Statistics, FreqTables

# ╔═╡ fbd5e410-f389-11ea-03f8-b90572b6e609
using DataFrames, CSV

# ╔═╡ 57fae8cc-f38b-11ea-3321-21e2b433ebc3
begin
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS3-gev/nlsw88w.csv"
	df = CSV.read(HTTP.get(url).body)
	X = [df.age df.white df.collgrad]
	Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
	df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
	y = df.occupation;
	
end

# ╔═╡ 59dc3cb8-f390-11ea-02c0-1f33de75a368
function mlogit(β, γ, X, Z, y)
	K = size(X,2) # nvars
    J = length(unique(y)) # n options
    N = length(y) # N
    bigY = zeros(N,J)
    bigY[:,J] = y.==J # di
	params = [β.*ones(K,J-1) zeros(K,1) γ.*ones(K,1)] # Setting βJ to zero
    #params = [reshape(β,K,J-1) zeros(K,1) reshape(γ,K,1)] # Setting βJ to zero
	T = promote_type(eltype(X),eltype(β)) # this line is new. I is relevant for solution algorithm
    num   = zeros(T,N,J)                      # this line is new
    dem   = zeros(T,N)                        # this line is new
	    
	for j=1:J
		Xs = (X*params[:,j])
		Zs = (Z[:,j] - Z[:,J])*params[end,end][1]
		num[:,j] = exp.(Xs + Zs)
        dem .+= num[:,j] # Does not change
     end
        
     P = num./repeat(dem,1,J)
        
     loglike = -sum( bigY.*log.(P) ) # Negative to maximize (minimize negative)
        
     return loglike
end

# ╔═╡ dcd7003e-f3ac-11ea-0a64-79dd7a7ce64d
begin
	nvars = size(X,2);
	func = TwiceDifferentiable(
		vars -> mlogit(
			vars[1:nvars], 
			vars[end]', X, Z, y),
		zeros(nvars+1);  # or ones
		autodiff=:forward)
end

# ╔═╡ 916db0da-f395-11ea-3b20-85a3a8f4dc4e
opt = optimize(func, ones(nvars+1),
	LBFGS(), 
	Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50)
)

# ╔═╡ 5d21c9b2-f3bd-11ea-07b2-3537c0bbab7b
# evaluate the Hessian at the estimates
H  = Optim.hessian!(func, opt.minimizer)

# ╔═╡ 5d23281e-f3bd-11ea-2910-d7d85c6dc661
σ = sqrt.(diag(inv(H))) # standard errors = sqrt(diag(inv(H))) [usually it's -H but we've already multiplied the obj fun by -1]

# ╔═╡ aebc9c80-f392-11ea-39f2-2d6856bb2beb
opt.minimizer[end]

# ╔═╡ e30c1074-f3c1-11ea-1423-ed75c21954f5


# ╔═╡ de67438e-f3c1-11ea-39be-b5411a7aaf13


# ╔═╡ c28c774e-f3c1-11ea-169f-3da012e05660


# ╔═╡ a7d8eb24-f3c1-11ea-1fbd-5b725349f0c3


# ╔═╡ a20731ba-f3c1-11ea-0348-61490921bafe


# ╔═╡ 9f6ebe64-f3c1-11ea-290e-918cfe19f6d5


# ╔═╡ 98fc362e-f38f-11ea-17ae-d3f68d90ce79


# ╔═╡ 8a224998-f38f-11ea-21b4-4b8a5b38f0ae


# ╔═╡ 6977c62a-f38f-11ea-3813-e76984698d90


# ╔═╡ Cell order:
# ╠═7085c940-f389-11ea-1dec-59018061e9b5
# ╠═fb93f836-f389-11ea-26e6-678b961ba787
# ╠═fbcefcc6-f389-11ea-2258-6fa3ead65683
# ╠═fbd5e410-f389-11ea-03f8-b90572b6e609
# ╠═57fae8cc-f38b-11ea-3321-21e2b433ebc3
# ╠═59dc3cb8-f390-11ea-02c0-1f33de75a368
# ╠═dcd7003e-f3ac-11ea-0a64-79dd7a7ce64d
# ╠═916db0da-f395-11ea-3b20-85a3a8f4dc4e
# ╠═5d21c9b2-f3bd-11ea-07b2-3537c0bbab7b
# ╠═5d23281e-f3bd-11ea-2910-d7d85c6dc661
# ╠═aebc9c80-f392-11ea-39f2-2d6856bb2beb
# ╠═e30c1074-f3c1-11ea-1423-ed75c21954f5
# ╠═de67438e-f3c1-11ea-39be-b5411a7aaf13
# ╠═c28c774e-f3c1-11ea-169f-3da012e05660
# ╠═a7d8eb24-f3c1-11ea-1fbd-5b725349f0c3
# ╠═a20731ba-f3c1-11ea-0348-61490921bafe
# ╠═9f6ebe64-f3c1-11ea-290e-918cfe19f6d5
# ╠═98fc362e-f38f-11ea-17ae-d3f68d90ce79
# ╠═8a224998-f38f-11ea-21b4-4b8a5b38f0ae
# ╠═6977c62a-f38f-11ea-3813-e76984698d90
