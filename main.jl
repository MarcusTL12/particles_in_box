using HCubature
using LinearAlgebra

function potential(q1, q2, x1, x2, d)
    q1 * q2 / √((x1 - x2)^2 + d^2)
end

function make_potfunc(q1, q2, d)
    function pot((x1, x2))
        q1 * q2 / √((x1 - x2)^2 + d^2)
    end
end

function make_basis_func(n, L)
    nrm = √(2 / L)

    function ϕ(x)
        nrm * sin(n * π / L * x)
    end
end

function h_nn(n, L)
    n^2 * π^2 / (2 * L^2)
end

function g_mnrs(m, n, r, s, L, d)
    nrm = 4 / L^2
    πonL = π / L
    d2 = d^2

    function integrand((x1, x2))
        nrm *
        sin(m * πonL * x1) *
        sin(n * πonL * x1) *
        sin(r * πonL * x2) *
        sin(s * πonL * x2) /
        √((x1 - x2)^2 + d2)
    end

    hcubature(integrand, (0, 0), (L, L); atol=1e-10)[1]
end

function diag_D_guess(n_occ, n_vir)
    d = zeros(n_occ + n_vir)
    for i in 1:n_occ
        d[i] = 2.0
    end
    Diagonal(d)
end

function construct_D(C, n_occ)
    C_occ = @view C[:, 1:n_occ]

    2 * C_occ * C_occ'
end

function construct_fock(D, L, d)
    N = size(D, 1)
    @assert size(D, 2) == N

    F = zeros(N, N)

    for n in 1:N, m in 1:N
        elem = 0.0
        for s in 1:N, r in 1:N
            elem += D[r, s] * g_mnrs(m, n, r, s, L, d)
        end
        F[m, n] = elem
    end

    for n in 1:N
        F[n, n] += h_nn(n, L)
    end

    F
end
