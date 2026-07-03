using QuadGK
using HCubature

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
