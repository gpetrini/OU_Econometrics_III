### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 4afee820-f909-11ea-06bc-d51a333b2baa
using Distributions, Random, Statistics, FreqTables

# ╔═╡ 3c23623a-f90a-11ea-2b40-05b297cb37dd
using Optim, GLM, LinearAlgebra

# ╔═╡ d989e292-f90a-11ea-3e73-516abb783304
using HTTP, CSV, DataFrames

# ╔═╡ 08f4baf6-f90c-11ea-0ead-4330f04896db
include("lgwt.jl")

# ╔═╡ 502cd58e-f911-11ea-134b-b720cf9f5592
md"""
# Loading packages and Data
"""

# ╔═╡ 78789bb2-f921-11ea-3f1d-2d6d33d9fb8b
md"""
# Question $1$
Estimate a multinomial logit (with alternative-specific covariates $Z$) on the following data set, which is a panel form of the same data as Problem Set 3. You should be able to simply re-use your code from Problem Set 3. However, I would ask that you use automatic differentiation to speed up your estimation, and to obtain the standard errors of your estimates.
"""

# ╔═╡ c40137b0-f921-11ea-0655-45cdc040cd54
begin
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS4-mixture/nlsw88t.csv"
	df = CSV.read(HTTP.get(url).body)
	X = [df.age df.white df.collgrad]
	Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
	y = df.occ_code
end

# ╔═╡ d2596706-f921-11ea-23fd-ef3d19f4e322
md"""
**Note:** this took my machine about 30 minutes to estimate using random starting values. You might consider using the estimated values from Question 1 of PS3.

The choice set is identical to that of Problem Set 3 and represents possible occupations and is structured  as follows.

1. Professional/Technical 
2. Managers/Administrators
3. Sales                  
4. Clerical/Unskilled     
5. Craftsmen              
6. Operatives             
7. Transport              
8. Other                  
"""

# ╔═╡ c58ae742-f922-11ea-0a1a-63f0c6a48759
begin
	function mlogit_with_Z(theta, X, Z, y)
	        
	        alpha = theta[1:end-1]
	        gamma = theta[end]
	        K = size(X,2)
	        J = length(unique(y))
	        N = length(y)
	        bigY = zeros(N,J)
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
	    startvals = [2*rand(7*size(X,2)).-1; .1]
	    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
	    # run the optimizer
	    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
	    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
	    # evaluate the Hessian at the estimates
	    H  = Optim.hessian!(td, theta_hat_mle_ad)
	    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
	
end

# ╔═╡ dd2547f4-f921-11ea-1ba1-fd07f1bed50a
md"""
# Question $2$

Does the estimated coefficient $\hat{\gamma}$ make more sense now than in Problem Set 3?
"""

# ╔═╡ b3340562-f925-11ea-1983-a1eca1e95821
γ = (theta_hat_mle_ad[end]/100)

# ╔═╡ 1b8742de-f90b-11ea-3f90-991ccc07fcae
md"""
# Question 3

Now we will estimate the mixed logit version of the model in Question 1, where $\gamma$ is distributed $N\left(\tilde{\gamma},\sigma^2_{\gamma}\right)$. Following the notes, the  formula for the choice probabilities will be


While this looks daunting, we can slightly modify the objective function from Question 1. 

The first step is to recognize that we will need to \textit{approximate} the integral in \eqref{eq:intlike}. There are many ways of doing this, but we will use something called Gauss-Legendre quadrature.\footnote{Another popular method of approximating the integral is by simulation.} We can rewrite the integral in \eqref{eq:intlike} as a (weighted) discrete summation:

$
\begin{align}
\label{eq:quadlike}
\ell\left(X,Z;\beta,\mu_{\gamma},\sigma_{\gamma}\right)&=\sum_{i=1}^N\sum_{t=1}^T \log\left\{\sum_{k}\omega_{r}\prod_{j}\left[\frac{\exp\left(X_{it}\left(\beta_{j}-\beta_{J}\right)+\xi_r\left(Z_{itj}-Z_{itJ}\right)\right)}{\sum_k \exp\left(X_{it}\left(\beta_{k}-\beta_{J}\right)+\xi_r\left(Z_{itk}-Z_{itJ}\right)\right)}\right]^{d_{itj}}f\left(\xi_r\right)\right\}
\end{align}
$

where $\omega_r$ are the quadrature weights and $\xi_r$ are the quadrature points. $\tilde{\gamma}$ and $\sigma_{\gamma}$ are parameters of the distribution function $f\left(\cdot\right)$.


1. Before we dive in, let's learn how to use quadrature. In the folder where this problem set is posted on GitHub, there is a file called \texttt{lgwt.jl}. This is a function that returns the $\omega$'s and $\xi$'s for a given choice of $K$ (number of quadrature points) and bounds of integration $[a,b]$.
    

Let's practice doing quadrature using the density of the Normal distribution.
"""

# ╔═╡ d0ee42d2-f90e-11ea-2409-dd20c2aba3fe
d = Normal(0,1); # μ = 0, σ = 1

# ╔═╡ dea66a7e-f90e-11ea-3e27-f9f08da3cdf1
md"""
Once we have the distribution defined, we can do things like evaluate its density or probability, take draws from it, etc.
    
We want to verify that $\int \phi(x)dx$ is equal to 1 (i.e. integrating over the density of the support equals 1). We also want to verify that $\int x\phi(x)dx$ is equal to $\mu$ (which for the distribution above is 0).
    
When using quadrature, we should try to pick a number of points and bounds that will minimize computational resources, but still give us a good approximation to the integral. For Normal distributions, $\pm 4\sigma$ will get us there.

get quadrature node and weights for 7 grid points
"""

# ╔═╡ 2909be18-f90c-11ea-3974-d347c718d61f
nodes, weights = lgwt(10, -5,5); # ±5σ

# ╔═╡ 0edfe3f2-f90f-11ea-0388-550a5584c468
md"""
now compute the integral over the density and verify it's 1
"""

# ╔═╡ e73ce9e6-f90c-11ea-3924-1b474a3182b3
sum(weights .* pdf.(d, nodes))

# ╔═╡ 22f79f3a-f90f-11ea-2447-c77328685e30
md"""
now compute the expectation and verify it's $0$
"""

# ╔═╡ e74a2fde-f90c-11ea-2d85-736fb0aa9d1d
sum(weights.* nodes .* pdf.(d, nodes))

# ╔═╡ f5d6438a-f90c-11ea-01a9-8f7d4f1434c8
md"""
To get some more practice, I'd like you to use quadrature to compute the following integrals:

$\int_{-5\sigma}^{5\sigma}x^{2}f\left(x\right)dx$ where $f\left(\cdot\right)$ is the pdf of a $N\left(0,2\right)$ distribution using 10 quadrature points
"""

# ╔═╡ 79061b6a-f910-11ea-2340-478981473b5f
sum(weights .* nodes.^2 .* pdf.(Normal(0,2), nodes))

# ╔═╡ d568097e-f90c-11ea-2d78-b7d7ab4ba704
md"""
## Monte Carlo integration

An alternative to quadrature is Monte Carlo integration. Under this approach, we approximate the integral of $f$ by averaging over a function of many random numbers. Formally, we have that
    
$\int_a^b f\left(x\right)dx \approx \left(b-a\right)\frac{1}{D}\sum_{i=1}^D f\left(X_{i}\right)$

where $D$ is the number of random draws and where each $X_i$ is drawn from a $U[a,b]$ interval
    
- With $D=1,000,000$, use the formula abobe to approximate $\int_{-5\sigma}^{5\sigma}x^{2}f\left(x\right)dx$ where $f\left(\cdot\right)$ is the pdf of a $N\left(0,2\right)$ and verify that it is (very) close to 4
-  Do the same for $\int_{-5\sigma}^{5\sigma}xf\left(x\right)dx$ where $f\left(\cdot\right)$ and verify that it is very close to 0
- Do the same for $\int_{-5\sigma}^{5\sigma}f\left(x\right)dx$ and verify that it is very close to 1
-  Comment on how well the simulated integral approximates the true value when $D=1,000$ compared to when $D=1,000,000$.
"""

# ╔═╡ 0463c252-f911-11ea-2ca2-09693e84fd6f
σ = 2	

# ╔═╡ d1259856-f90c-11ea-1a7b-491d175ceba9
function montecarlo(a, b, D)
	xrand = rand(Uniform(a,b), D)
	σ = ((b-a)/D) * sum(xrand.^2 .* pdf(Normal(0,2), xrand))
	μ = ((b-a)/D) * sum(xrand .* pdf(Normal(0,2), xrand))
	FDP = ((b-a)/D) * sum(pdf(Normal(0,2), xrand))
	
	return σ, μ, FDP
end

# ╔═╡ b4e8ce6c-f914-11ea-31ba-cf912ec061c4
montecarlo(-5σ, 5σ, 10^6)

# ╔═╡ Cell order:
# ╟─502cd58e-f911-11ea-134b-b720cf9f5592
# ╠═4afee820-f909-11ea-06bc-d51a333b2baa
# ╠═3c23623a-f90a-11ea-2b40-05b297cb37dd
# ╠═d989e292-f90a-11ea-3e73-516abb783304
# ╠═08f4baf6-f90c-11ea-0ead-4330f04896db
# ╟─78789bb2-f921-11ea-3f1d-2d6d33d9fb8b
# ╠═c40137b0-f921-11ea-0655-45cdc040cd54
# ╟─d2596706-f921-11ea-23fd-ef3d19f4e322
# ╠═c58ae742-f922-11ea-0a1a-63f0c6a48759
# ╟─dd2547f4-f921-11ea-1ba1-fd07f1bed50a
# ╠═b3340562-f925-11ea-1983-a1eca1e95821
# ╟─1b8742de-f90b-11ea-3f90-991ccc07fcae
# ╠═d0ee42d2-f90e-11ea-2409-dd20c2aba3fe
# ╟─dea66a7e-f90e-11ea-3e27-f9f08da3cdf1
# ╠═2909be18-f90c-11ea-3974-d347c718d61f
# ╟─0edfe3f2-f90f-11ea-0388-550a5584c468
# ╠═e73ce9e6-f90c-11ea-3924-1b474a3182b3
# ╟─22f79f3a-f90f-11ea-2447-c77328685e30
# ╠═e74a2fde-f90c-11ea-2d85-736fb0aa9d1d
# ╟─f5d6438a-f90c-11ea-01a9-8f7d4f1434c8
# ╠═79061b6a-f910-11ea-2340-478981473b5f
# ╟─d568097e-f90c-11ea-2d78-b7d7ab4ba704
# ╠═0463c252-f911-11ea-2ca2-09693e84fd6f
# ╠═d1259856-f90c-11ea-1a7b-491d175ceba9
# ╠═b4e8ce6c-f914-11ea-31ba-cf912ec061c4
