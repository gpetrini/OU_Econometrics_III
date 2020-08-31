using Random, Distributions, FreqTables, Statistics
using LinearAlgebra
using JLD2, CSV, DataFrames

Random.seed!(1234)

A = rand(Uniform(-5,10), 10,7)
