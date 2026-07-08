
# |Ψ(x₁, x₂)⟩ = |ϕ₁(x₁)⟩ |ϕ₂(x₂)⟩
#
# ϕᵢ(x) = ∑ₙ(Cₙᵢ χₙ(x))
# χₙ(x) = √(2 / L) sin(n π x / L)

# E = ⟨Ψ|H|Ψ⟩
#
# H = h + g
#
#   h = T + V = ∑ᵢ(hᵢ)
#
#       T = ∑ᵢ(-1/2mᵢ ∂²/∂xᵢ²)
#
#       V = ∑ᵢ(Vᵢ(xᵢ))
#       V₁ = -V₂ (Opposite charges)
#
# g = ∑ᵢ˲ⱼ(gᵢⱼ(xᵢ, xⱼ))
#   = g(x₁, x₂)

# E = ⟨Ψ|H|Ψ⟩
#   = ∑ᵢ⟨ϕᵢ|hᵢ|ϕᵢ⟩
#   + ∑ᵢ˲ⱼ⟨ϕᵢϕⱼ|gᵢⱼ|ϕᵢϕⱼ⟩

# ⟨ϕᵢ|hᵢ|ϕᵢ⟩ = ∑ₘₙ(CₘᵢCₙᵢ ⟨χₘ|hᵢ|χₙ⟩) = ∑ₘₙ(CₘᵢCₙᵢ hⁱₘₙ)
#
# Tⁱₘₙ = -1/(mᵢL) ∫(sin(m π x / L) ∂²/∂x² sin(n π x / L) dx)
#      = δₘₙ m² π² / (2 mᵢ L²)
#
# Vₘₙ = 2/L ∫(sin(m π x / L) V(x) sin(n π x / L) dx)
#
# sin(m π x / L) sin(n π x / L) =
# 1/2 (cos((m - n) π x / L) - cos((m + n) π x / L))
#
# Vₘₙ = 1/L ∫(cos((m - n) π x / L) V(x) dx)
#     - 1/L ∫(cos((m + n) π x / L) V(x) dx)
#     = Vₘ₋ₙ - Vₘ₊ₙ

# ⟨ϕ₁ϕ₂|g|ϕ₁ϕ₂⟩ = ∑ₘₙᵣₛ(Cₘ₁Cₙ₂Cᵣ₁Cₛ₂ ⟨χₘχₙ|g|χᵣχₛ⟩)
#
# gₘᵣₙₛ = ⟨χₘχₙ|g|χᵣχₛ⟩ =
# 4 / L² ∫(sin(m π x₁ / L) sin(n π x₂ / L) g(x₁, x₂)
#          sin(r π x₁ / L) sin(n π x₂ / L) dx₁ dx₂)
# = 1 / L² (∫(cos((m - r) π x₁ / L) cos((n - s) π x₂ / L) g(x₁, x₂) dx₁ dx₂)
#         - ∫(cos((m + r) π x₁ / L) cos((n - s) π x₂ / L) g(x₁, x₂) dx₁ dx₂)
#         - ∫(cos((m - r) π x₁ / L) cos((n + s) π x₂ / L) g(x₁, x₂) dx₁ dx₂)
#         + ∫(cos((m + r) π x₁ / L) cos((n + s) π x₂ / L) g(x₁, x₂) dx₁ dx₂))
# = gₘ₋ᵣ,ₙ₋ₛ - gₘ₊ᵣ,ₙ₋ₛ - gₘ₋ᵣ,ₙ₊ₛ + gₘ₊ᵣ,ₙ₊ₛ

# Fₘₙ = hₘₙ + ∑ᵣₛ(Dᵣₛ (gₘₙᵣₛ - ½ gₘₛᵣₙ))

using LinearAlgebra
using FFTW
using DSP
using Printf

function construct_1e_potential_matrix_dct(N, M, L, V)
    xs = range(0, L, length=(M + 1))[1:(end-1)]
    xs = xs .+ step(xs) / 2

    println("Sampling potential at $M points:")
    sample = @time V.(xs)

    println("Computing DCT:")
    @time begin
        dct_plan = FFTW.plan_r2r!(sample, FFTW.REDFT10;
            num_threads=Threads.nthreads())
        FFTW.mul!(sample, dct_plan, sample)
    end

    V_cos = @view sample[1:(2N+1)]

    V_cos .*= 1 / 2M

    V_mat = zeros(N, N)

    for n in 1:N, m in n:N
        V_mat[m, n] = V_cos[begin+m-n] - V_cos[begin+m+n]
    end

    Symmetric(V_mat, :L)
end

# V(x) = -q / √((x - x0)^2 + d^2)
function make_atomic_potential(q, x0, d)
    x -> -q / √((x - x0)^2 + d^2)
end

function make_molecular_potential(qs, xs, d)
    x -> sum(-q / √((x - x0)^2 + d^2) for (q, x0) in zip(qs, xs))
end

function sample_function(xs, ys, f)
    sample = zeros(Float64, length(xs), length(ys))

    @inbounds begin
        Threads.@threads for j in eachindex(ys)
            y = ys[j]
            for (i, x) in enumerate(xs)
                sample[i, j] = f(x, y)
            end
        end
    end

    sample
end

function construct_g_cos_dct(N, M, L, d)
    xs = range(0, L, length=(M + 1))[1:(end-1)]

    xs = xs .+ step(xs) / 2

    pot_func = (x1, x2) -> 1 / √((x1 - x2)^2 + d^2)

    println("Sampling potential at $(M^2) points:")

    sample = @time sample_function(xs, xs, pot_func)

    println("Computing DCT:")
    @time begin
        dct_plan = FFTW.plan_r2r!(sample, FFTW.REDFT10;
            num_threads=Threads.nthreads())
        FFTW.mul!(sample, dct_plan, sample)
    end

    result = sample[1:(2N+1), 1:(2N+1)]

    result .*= 1 / (4 * M^2)

    result
end

# gₘₙᵣₛ = gₘ₋ₙ,ᵣ₋ₛ - gₘ₊ₙ,ᵣ₋ₛ - gₘ₋ₙ,ᵣ₊ₛ + gₘ₊ₙ,ᵣ₊ₛ
function construct_2e_potential_tensor_dct(N, M, L, d)
    g_cos = construct_g_cos_dct(N, M, L, d)

    g_mat = zeros(N, N, N, N)

    for s in 1:N, r in s:N
        slice = @view g_mat[:, :, r, s]

        for n in 1:N, m in n:N
            integral = g_cos[begin+m-n, begin+r-s] -
                       g_cos[begin+m+n, begin+r-s] -
                       g_cos[begin+m-n, begin+r+s] +
                       g_cos[begin+m+n, begin+r+s]

            slice[m, n] = integral
            if m != n
                slice[n, m] = integral
            end
        end

        if r != s
            g_mat[:, :, s, r] .= slice
        end
    end

    g_mat
end

function construct_h(N, L, V, M_1e)
    h = if isnothing(V)
        zeros(N, N)
    else
        construct_1e_potential_matrix_dct(N, M_1e, L, V)
    end

    # Kinetic contribution
    for n in 1:N
        h[n, n] += n^2 * π^2 / (2 * L^2)
    end

    h
end

function construct_fock_naive(N, h, g, D)
    G = zeros(N, N)

    for n in 1:N, m in 1:N
        element = 0.0

        for s in 1:N, r in 1:N
            element += D[r, s] * (g[m, n, r, s] - 0.5 * g[m, s, r, n])
        end

        G[m, n] = element
    end

    E = (h[:] + 0.5 * G[:]) ⋅ D[:]

    @show E

    h + G
end

function construct_coulomb_matrix(g_cos, D)
    N = size(D, 1)

    x = zeros(2N + 1)

    for i in 0:2N
        element = 0.0
        for s in 1:N, r in 1:N
            element += (
                g_cos[begin+i, begin+abs(r-s)] - g_cos[begin+i, begin+r+s]
            ) * D[r, s]
        end
        x[begin+i] = element
    end

    G = zeros(N, N)

    for n in 1:N, m in 1:N
        G[m, n] = x[begin+abs(m-n)] - x[begin+m+n]
    end

    G
end

function construct_fock_direct(N, h, g_cos, D)
    g = construct_2e_potential_tensor_dct(N, 10_000, 1, 0.05)

    G_exc_naive = zeros(N, N)

    for n in 1:N, m in 1:N
        element = 0.0

        for s in 1:N, r in 1:N
            element += D[r, s] * g[m, s, r, n]
        end

        G_exc_naive[m, n] = -0.5 * element
    end

    G_coul = construct_coulomb_matrix(g_cos, D)

    @show maximum(abs, G_coul_naive - G_coul)

    G = G_coul + G_exc_naive

    E = (h[:] + 0.5 * G[:]) ⋅ D[:]

    @show E

    h + G
end

function do_hf_naive(N, L, V, d, nocc; tol=1e-5, M_1e=10_000_000, M_2e=10_000)
    C = zeros(N, nocc)
    for i in 1:nocc
        C[i, i] = 1
    end

    println("Constructing h and g:")
    @time begin
        h = construct_h(N, L, V, M_1e)
        g = construct_2e_potential_tensor_dct(N, M_2e, L, d)
    end

    D = 2.0 * C * C'

    println("Constructing Fock:")
    F = @time construct_fock_naive(N, h, g, D)

    FC = F * C
    grad = FC - C * (C'FC)

    @printf "Init grad: %.2e\n" maximum(abs, grad)

    i_iter = 1

    while maximum(abs, grad) > tol
        println("\nIteration $i_iter:\n")
        i_iter += 1

        e, C = eigen(Symmetric(F), 1:nocc)

        D = 2.0 * C * C'

        println("Constructing Fock:")
        F = @time construct_fock_naive(N, h, g, D)

        FC = F * C
        grad = FC - C * (C'FC)

        @printf "Grad: %.2e\n" maximum(abs, grad)
    end

    C
end

function do_hf_diis(N, L, V, d, nocc;
    tol=1e-5, M_1e=10_000_000, M_2e=10_000, diis_max_hist=8)
    C = zeros(N, nocc)
    for i in 1:nocc
        C[i, i] = 1
    end

    println("Constructing h and g:")
    @time begin
        h = construct_h(N, L, V, M_1e)
        g = construct_2e_potential_tensor_dct(N, M_2e, L, d)
    end

    D = 2.0 * C * C'

    fock_history = Float64[]
    grad_history = Float64[]

    println("Constructing Fock:")
    F = @time construct_fock_naive(N, h, g, D)

    append!(fock_history, F)

    FC = F * C
    grad = FC - C * (C'FC)

    maxabsgrad = maximum(abs, grad)
    @printf "Init grad: %.2e\n" maxabsgrad

    append!(grad_history, grad)

    n_in_history = 1

    i_iter = 1

    while true
        println("\nIteration $i_iter:\n")
        i_iter += 1

        e, C = eigen(Symmetric(F), 1:nocc)

        D = 2.0 * C * C'

        println("Constructing Fock:")
        F = @time construct_fock_naive(N, h, g, D)

        FC = F * C
        grad = FC - C * (C'FC)

        maxabsgrad = maximum(abs, grad)
        @printf "Grad: %.2e\n" maxabsgrad

        if maxabsgrad < tol
            break
        end

        if n_in_history >= diis_max_hist
            # fock_history = [fock_history[(length(F)+1):end]; F[:]]
            # grad_history = [grad_history[(length(grad)+1):end]; grad[:]]

            empty!(fock_history)
            empty!(grad_history)

            append!(fock_history, F)
            append!(grad_history, grad)

            n_in_history = 1

            continue
        else
            append!(fock_history, F)
            append!(grad_history, grad)

            n_in_history += 1
        end

        println("Solving diis problem with history length: $n_in_history")
        error_mat = reshape(grad_history, N * nocc, n_in_history)
        diis_coeff = solve_diis(error_mat)

        fock_hist_mat = reshape(fock_history, N^2, n_in_history)

        new_fock = fock_hist_mat * diis_coeff
        fock_hist_mat[:, end] .= new_fock
        F = reshape(new_fock, N, N)

        FC = F * C
        new_grad = FC - C * (C'FC)

        error_mat[:, end] .= new_grad[:]

        @printf "Grad after diis: %.2e\n" maximum(abs, new_grad)
    end

    C
end

# diis:
# Fmo = C' * F * C
#
# C = [ Co Cv ]
#
# [ Foo Fov ] = [ Co' ] F [ Co Cv ]
# [ Fvo Fvv ]   [ Cv' ]
#
# Fvo = Cv' F Co
#
# Project out occ space
# Pv = I - Co * Co'
#
# err = Pv F Co = F Co - Co Co' F Co = F Co - Foo
#
# new_err = ∑_i(err_i c_i)
# ∑_i(c_i) = 1
# minimize Z = |new_err|^2
# g(c) = ∑_i(c_i) - 1
#
# Z = new_err'new_err = ∑_ij(err_i'err_j c_i c_j)
#
# L = Z + λ g
#
# ∂L/∂c_i = ∂Z/∂c_i + λ
#
# ∂Z/∂c_i = ∑_jk(err_j'err_k (δ_ij c_k + c_j δ_ik))
#         = 2 ∑_j(err_i'err_j c_j)
#
# ∂L/∂c_i = 2 ∑_j(err_i'err_j c_j) + λ = 0
#
# ∂L/∂λ = ∑_i(c_i) - 1 = 0
#
# A x = b
#
# [ 2 err'err 1 ] [ c ] = [ 0 ]
# [ 1         0 ] [ λ ]   [ 1 ]
function solve_diis(error_mat)
    n = size(error_mat, 2)

    S = error_mat'error_mat

    onevec = ones(n)

    A = [
        S onevec
        onevec' 0
    ]

    b = zeros(n + 1)
    b[end] = 1

    x = A\b

    x[1:(end-1)]
end
