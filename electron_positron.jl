using HCubature
using LinearAlgebra

# Just to ensure orthonormal basis
# ⟨mn|rs⟩ = δ_mn,rs
function ovlp_integral(m, n, r, s, L)
    πonL = π / L

    function combined_function((x1, x2))
        4 / L^2 *
        sin(m * πonL * x1) * sin(n * πonL * x2) *
        sin(r * πonL * x1) * sin(s * πonL * x2)
    end

    hcubature(combined_function, (0, 0), (L, L); atol=1e-10)
end

# ⟨mn|T|mn⟩
function kin_mat_diag((m, n), m1, m2, L)
    π^2 / (2 * L^2) * (m^2 / m1 + n^2 / m2)
end

# ⟨mn|V|rs⟩
function potential_integral((m, n), (r, s), q1, q2, d, L)
    πonL = π / L

    prefac = 4 / L^2 * q1 * q2

    function combined_function((x1, x2))
        prefac *
        sin(m * πonL * x1) * sin(n * πonL * x2) *
        sin(r * πonL * x1) * sin(s * πonL * x2) /
        √((x1 - x2)^2 + d^2)
    end

    hcubature(combined_function, (0, 0), (L, L); atol=1e-10)[1]
end

function cosine_potential_integral(m, n, q1, q2, d, L)
    πonL = π / L

    prefac = q1 * q2 / L^2

    function combined_function((x1, x2))
        prefac *
        cos(m * πonL * x1) *
        cos(n * πonL * x2) /
        √((x1 - x2)^2 + d^2)
    end

    hcubature(combined_function, (0, 0), (L, L); atol=1e-8)[1]
end

function construct_hamiltonian(N, q1, q2, m1, m2, d, L)
    V = zeros(2N + 1, 2N + 1)

    sig_ind_pairs = [(m, n) for n in 0:2N for m in n:2:2N]

    println("N integrals = ", length(sig_ind_pairs))

    Threads.@threads :greedy for (m, n) in sig_ind_pairs
        V[begin+m, begin+n] = cosine_potential_integral(m, n, q1, q2, d, L)
    end

    V = Symmetric(V, :L)

    H = zeros(N, N, N, N)

    for s in 1:N, r in 1:N, n in 1:N, m in 1:N
        H[m, n, r, s] = V[begin+abs(m - r), begin+abs(n - s)] -
                        V[begin+abs(m - r), begin+(n+s)] -
                        V[begin+(m+r), begin+abs(n - s)] +
                        V[begin+(m+r), begin+(n+s)]
    end

    for n in 1:N, m in 1:N
        H[m, n, m, n] += kin_mat_diag((m, n), m1, m2, L)
    end

    reshape(H, N^2, N^2)
end
