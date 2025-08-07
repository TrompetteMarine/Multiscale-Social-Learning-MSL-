# networks.jl - Network generation utilities
module Networks

using Graphs, Random, LinearAlgebra

export create_network, network_stats

"""
    create_network(N, type, params)

Create adjacency matrix for different network topologies.
"""
function create_network(N::Int, type::Symbol, params::Dict=Dict())
    if type == :fully_connected
        return fully_connected(N)
    elseif type == :small_world
        k = get(params, :k, 6)
        p = get(params, :p, 0.3)
        return watts_strogatz(N, k, p)
    elseif type == :scale_free
        m = get(params, :m, 3)
        return barabasi_albert(N, m)
    elseif type == :random
        p = get(params, :p, 0.1)
        return erdos_renyi(N, p)
    elseif type == :ring
        k = get(params, :k, 4)
        return ring_lattice(N, k)
    else
        @warn "Unknown network type $type, using fully connected"
        return fully_connected(N)
    end
end

"""
    fully_connected(N)

Create fully connected network (everyone connects to everyone).
"""
function fully_connected(N::Int)
    W = ones(N, N) / N
    for i in 1:N
        W[i,i] = 0.0
    end
    # Normalize rows
    return W ./ sum(W, dims=2)
end

"""
    watts_strogatz(N, k, p)

Create Watts-Strogatz small-world network.
"""
function watts_strogatz(N::Int, k::Int, p::Float64)
    g = Graphs.watts_strogatz(N, k, p)
    W = Matrix(adjacency_matrix(g))
    W = Float64.(W)
    
    # Make symmetric and normalize
    W = (W + W') / 2
    row_sums = sum(W, dims=2)
    row_sums[row_sums .== 0] .= 1  # Avoid division by zero
    
    return W ./ row_sums
end

"""
    barabasi_albert(N, m)

Create Barabási-Albert scale-free network.
"""
function barabasi_albert(N::Int, m::Int)
    g = Graphs.barabasi_albert(N, m)
    W = Matrix(adjacency_matrix(g))
    W = Float64.(W)
    
    # Make symmetric
    W = (W + W') / 2
    row_sums = sum(W, dims=2)
    row_sums[row_sums .== 0] .= 1
    
    return W ./ row_sums
end

"""
    erdos_renyi(N, p)

Create Erdős-Rényi random network.
"""
function erdos_renyi(N::Int, p::Float64)
    g = Graphs.erdos_renyi(N, p)
    W = Matrix(adjacency_matrix(g))
    W = Float64.(W)
    
    # Ensure connectivity (add small weight to all edges)
    W = W + 0.01 * ones(N, N)
    for i in 1:N
        W[i,i] = 0.0
    end
    
    row_sums = sum(W, dims=2)
    return W ./ row_sums
end

"""
    ring_lattice(N, k)

Create ring lattice with k nearest neighbors.
"""
function ring_lattice(N::Int, k::Int)
    W = zeros(N, N)
    
    for i in 1:N
        for j in 1:k÷2
            # Connect to neighbors on each side
            right = mod1(i + j, N)
            left = mod1(i - j, N)
            W[i, right] = 1.0
            W[i, left] = 1.0
        end
    end
    
    row_sums = sum(W, dims=2)
    row_sums[row_sums .== 0] .= 1
    
    return W ./ row_sums
end

"""
    network_stats(W)

Compute basic network statistics.
"""
function network_stats(W::Matrix{Float64})
    N = size(W, 1)
    A = W .> 0  # Binary adjacency
    
    # Degree distribution
    degrees = vec(sum(A, dims=2))
    
    # Clustering coefficient
    clustering = 0.0
    for i in 1:N
        neighbors = findall(A[i, :])
        k = length(neighbors)
        if k >= 2
            # Count triangles
            triangles = 0
            for j in neighbors, l in neighbors
                if j < l && A[j, l]
                    triangles += 1
                end
            end
            clustering += 2 * triangles / (k * (k - 1))
        end
    end
    clustering /= N
    
    # Spectral properties
    eigenvals = eigvals(W)
    spectral_gap = 1.0 - maximum(abs.(eigenvals[2:end]))
    
    return Dict(
        :N => N,
        :density => sum(A) / (N * (N - 1)),
        :avg_degree => mean(degrees),
        :clustering => clustering,
        :spectral_gap => spectral_gap,
        :connected => all(degrees .> 0)
    )
end

end # module
