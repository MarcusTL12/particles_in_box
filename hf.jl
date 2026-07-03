
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

using QuadGK
using HCubature
using LinearAlgebra
using FFTW

# Vₘₙ = ⟨χₘ|V|χₙ⟩ = 2/L ∫sin(m π x / L) V(x) sin(n π x / L) dx
function potential_1e_integral(m, n, L, V)
    function integrand(x)
        a = π * x / L
        2 / L * sin(m * a) * V(x) * sin(n * a)
    end

    quadgk(integrand, 0, L; atol=1e-10)[1]
end

function construct_1e_potential_matrix_naive(N, L, V)
    V_mat = zeros(N, N)

    for n in 1:N, m in n:N
        V_mat[m, n] = potential_1e_integral(m, n, L, V)
    end

    Symmetric(V_mat, :L)
end

# Vₙ = 1/L ∫(cos(n π x / L) V(x) dx)
function cosine_potential_1e_integral(n, L, V)
    function integrand(x)
        1/L * cos(n * π * x / L) * V(x)
    end

    quadgk(integrand, 0, L; atol=1e-10)[1]
end

# Vₘₙ = Vₘ₋ₙ - Vₘ₊ₙ
function construct_1e_potential_matrix(N, L, V)
    V_cos = zeros(2N + 1)

    for n in 0:2N
        V_cos[begin+n] = cosine_potential_1e_integral(n, L, V)
    end

    V_mat = zeros(N, N)

    for n in 1:N, m in n:N
        V_mat[m, n] = V_cos[begin+m-n] - V_cos[begin+m+n]
    end

    Symmetric(V_mat, :L)
end

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

# gₘₙᵣₛ = ⟨χₘχᵣ|g|χₙχₛ⟩
function potential_2e_integral(m, n, r, s, L, d)
    πonL = π / L

    prefac = 4 / L^2

    function integrand((x1, x2))
        prefac *
        sin(m * πonL * x1) * sin(r * πonL * x2) *
        sin(n * πonL * x1) * sin(s * πonL * x2) *
        1 / √((x1 - x2)^2 + d^2)
    end

    hcubature(integrand, (0, 0), (L, L); atol=1e-10)[1]
end

function construct_2e_potential_tensor_naive(N, L, d)
    g_mat = zeros(N, N, N, N)

    i_eval = 0

    for s in 1:N, r in s:N
        slice = @view g_mat[:, :, r, s]

        for n in 1:N, m in n:N
            i_eval += 1
            integral = potential_2e_integral(m, n, r, s, L, d)

            slice[m, n] = integral
            if m != n
                slice[n, m] = integral
            end
        end

        if r != s
            g_mat[:, :, s, r] .= slice
        end
    end

    println("Evaluated $i_eval integrals")

    g_mat
end

function cosine_potential_2e_integral(m, n, L, d)
    πonL = π / L

    prefac = 1 / L^2

    function integrand((x1, x2))
        prefac *
        cos(m * πonL * x1) * cos(n * πonL * x2) *
        1 / √((x1 - x2)^2 + d^2)
    end

    hcubature(integrand, (0, 0), (L, L); atol=1e-10)[1]
end

function construct_g_cos(N, L, d)
    g_cos = zeros(2N + 1, 2N + 1)

    i_eval = 0

    for n in 0:2N, m in n:2:2N
        i_eval += 1
        integral = cosine_potential_2e_integral(m, n, L, d)
        g_cos[begin+m, begin+n] = integral
    end

    println("Evaluated $i_eval integrals")

    Symmetric(g_cos, :L)
end

# gₘₙᵣₛ = gₘ₋ₙ,ᵣ₋ₛ - gₘ₊ₙ,ᵣ₋ₛ - gₘ₋ₙ,ᵣ₊ₛ + gₘ₊ₙ,ᵣ₊ₛ
function construct_2e_potential_tensor(N, L, d)
    g_cos = construct_g_cos(N, L, d)

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
