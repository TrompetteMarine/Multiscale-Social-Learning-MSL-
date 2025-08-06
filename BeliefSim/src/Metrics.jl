module Metrics
using Statistics, StatsBase

export ShiftPars, shift_estimator, layer_feature

# ----------------------------------------------------------------------
struct ShiftPars
    kernel::Function      # K(u)
    hN::Float64           # bandwidth h_N
end

# default: Epanechnikov kernel, h_N = c / √N  (c chosen later)
ShiftPars(N; c = 1.06) = ShiftPars(u -> max(0, 0.75 * (1 - u^2)), c / sqrt(N))

# ----------------------------------------------------------------------
# layer_feature picks g_ℓ(Z)   (ℓ ∈ {:micro,:meso,:macro})
function layer_feature(x::Vector{Vector{Float64}}, ℓ::Symbol, W::AbstractMatrix)
    #x = first.(Zs)                     # beliefs vector at time T
    if ℓ == :micro
        return x
    elseif ℓ == :meso                 # mean belief of each agent’s neighbourhood
        return (W * x)                # W is row-stochastic, so W*x is local mean
    elseif ℓ == :macro
        return fill(mean(x), length(x))
    else
        error("layer ℓ must be :micro, :meso or :macro")
    end
end

raw"""
    shift_estimator(t_vec, g_vec, T, pars)

Implements Eq.(29):
    \hat{δ}_{ℓ}(T) = (1 / (N h_N))  Σ_i  K((t_i - T) / h_N)  g_i
"""

function shift_estimator(t_vec::Vector{Float64},
                         g_vec::Vector{Float64},
                         T::Float64,
                         pars::ShiftPars)
    N = length(g_vec)
    K, h = pars.kernel, pars.hN
    return sum(i -> K((t_vec[i] - T)/h) * g_vec[i], 1:N) / (N * h)
end

end # module
