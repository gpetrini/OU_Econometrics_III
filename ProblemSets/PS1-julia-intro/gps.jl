using Random, Distributions, FreqTables, Statistics
using LinearAlgebra
using JLD2, JLD, CSV, DataFrames

Random.seed!(1234);

function q1()
    #Initializing variables and practice with basic matrix operations
    # 1a
    A = rand(Uniform(-5,10), 10,7)

    # i
    B = rand(Normal(-2, 15), 10, 7)

    # ii
    C = [A[1:5, 1:5] B[1:5, end-1:end:-1]] #TODO Check latter

    # iii
    D = copy(A)
    D[A .> 0] .= 0
    D

    length(A)

    # 1c
    unique(D) |> length

    # 1d
    E = reshape(B, (length(B),1))

    # 1e
    F = cat(A, B, dims=3)

    # 1f
    #F = permutedims(F, (3,1,2))

    # 1g
    G = kron(B, C)
    #kron(C,F)

    # 1h
    save("./PS1-julia-intro/matrixpractice.jld", 
        "A", A,
        "B", B, 
        "C", C, 
        "D", D, 
        "E", E, 
        "F", F,
        "G", G)

    # 1i
    save("./PS1-julia-intro/firstmatrix.jld", "A", A,"B", B, "C", C, "D", D)

    # 1j
    convert(DataFrame, C) |> CSV.write("./PS1-julia-intro/Cmatrix.csv", )

    # 1k
    convert(DataFrame, D) |> CSV.write("./PS1-julia-intro/Dmatrix.csv", delim="\t")

    return A, B, C, D

end


function q2(A,B,C)
#  Practice with loops and comprehensions
# 2a
    AB = [x*y for (x,y) in zip(A, B)]

    AB2 = A .* B

    # 2b

    Cprime = Vector{Float64}()
    for i in C
        if  -5 <= i  <= 5.0
            append!(Cprime, i)
        end
    end

    Cprime2 = C[-5 .<= C .<= 5]

    # 2c
    N = 15169
    K = 6
    T = 5
    X = zeros(N,K,T)
    for t in 1:T
        X[:,1,t] .= ones(N) # stationary over time.
        X[:,2,t] .= rand(Bernoulli(.75*(6-t)/5), N)
        X[:,3,t] .= rand(Normal(15+t-1, 5*(t-1)), N)
        X[:,4,t] .= rand(Normal(π*(6-t)/3, 1/ℯ), N)
        X[:,5,t] .= rand(Binomial(20, 0.6), N)
        X[:,6,t] .= rand(Binomial(20,0.5),N) # Discrete normal stationary over time.
    end
    # 2d

    β = zeros(K,T)
    β[1,:] .= [1 + t*(.25) for t in 1:T] # d1
    β[2,:] .= [log(t) for t in 1:T] # d2
    β[3,:] .= [-sqrt(t) for t in 1:T] # d3
    β[4,:] .= [(ℯ^(t)  - ℯ^(t+1)) for t in 1:T] # d4
    β[5,:] .= [t for t in 1:T] # d5
    β[6,:] .= [t/3 for t in 1:T] # d6

    # 2e
    Y = zeros(N,T)
    for t in 1:T # How to use comprehensions in this case?
       Y[:,t] .= X[:,:,t]*β[:,t] + rand(Normal(0, .36), N)
    end
end

# 3
function q3()
    df = CSV.read("./PS1-julia-intro/nlsw88.csv")# |> dropmissing
    save("./PS1-julia-intro/nlsw88.jld.jld", "df", df)
    # 3b
    println((df.never_married |> sum)*100/nrow(df) |> round,"% sample has never been married")
    println((df.collgrad |> sum)*100/nrow(df) |> round,"% are college graduates")

    # 3c
    freqtable(df.race) |> prop
    # 3d 
    summarystats = describe(df)
    println(df.grade .|> ismissing |> sum, " grades observations are missing")

    # 3e
    freqtable(df.occupation, df.industry) |> prop

    # 3f
    gdf = groupby(df, ["occupation","industry"], sort=true) 
    combine(gdf, :wage => mean => :avg_ind_occ_wage) |> dropmissing 
    return nothing
end


# Practice with functions
# 4a Load firstmatrix.jld.
function q4()
    load("./PS1-julia-intro/firstmatrix.jld")

    function matrixops(m1,m2)
        if size(m1) != size(m2)
            error("inputs must have the same size.")
        end
        #=
        Takes as inputs the matrices A and B from question (a) of problem 1 and has three outputs: 
            (i) the element-by-element product of the inputs, 
            (ii) the product A'B, and 
            (iii) the sum of all the elements of A + B.
        =#
        return m1 .* m2, m1' * m2, m1 + m2
    end

    a,b, c = matrixops(A, B);

    df = CSV.read("./PS1-julia-intro/nlsw88.csv") |> dropmissing
    convert(Array,df.ttl_exp); convert(Array,df.wage);
    matrixops(
        df.ttl_exp, 
        df.wage);
    return nothing
end

A, B, C, D = q1();
q2(A,B,C)
q3()
q4()