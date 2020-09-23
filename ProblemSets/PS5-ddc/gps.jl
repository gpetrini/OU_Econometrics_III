### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ b1226c30-fdc8-11ea-1020-c1a5f67bc6aa
begin
	using Pkg
	Pkg.add("DataFramesMeta")
end

# ╔═╡ 3c45bc76-fdc5-11ea-21e4-d7973210099b
begin
	using DataFrames
	using CSV
	using HTTP
	using DataFramesMeta
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
	df = CSV.read(HTTP.get(url).body)
	df = @transform(df, bus_id = 1:size(df,1)) # Create bus id variable
end

# ╔═╡ 1ec897ae-fdc5-11ea-27b5-554b053bc087
md"""
# Introduction

In this problem set, we will explore a simplified version of the Rust (1987, \textit{Econometrica}) bus engine replacement model. Let's start by reading in the data.
"""

# ╔═╡ b2919210-fdc5-11ea-24d2-bba0c930d349
md"""
# Static estimation

Reshape the data into "long" panel format, calling your long dataset *df\_long*. I have included code on how to do this in the *PS5starter.jl*

1. Create bus id variable
"""

# ╔═╡ Cell order:
# ╟─1ec897ae-fdc5-11ea-27b5-554b053bc087
# ╠═b1226c30-fdc8-11ea-1020-c1a5f67bc6aa
# ╠═3c45bc76-fdc5-11ea-21e4-d7973210099b
# ╟─b2919210-fdc5-11ea-24d2-bba0c930d349
