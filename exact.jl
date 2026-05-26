using HCubature
using LinearAlgebra

function make_chi_func(m, n, L)
    πonL = π / L

    function f(x1, x2)
        (x1 - x2) * (sin(πonL * m * x1) * sin(πonL * n * x2) +
                     sin(πonL * n * x1) * sin(πonL * m * x2))
    end
end

function make_phi_func(m, n, L)
    πonL = π / L
    r2 = √2

    function f(x1, x2)
        r2 * (sin(πonL * m * x1) * sin(πonL * n * x2) -
              sin(πonL * n * x1) * sin(πonL * m * x2))
    end
end

function make_chi_basis(N, L)
    basis = [make_chi_func(1, 1, L)]

    for k in 3:N
        n = k - 1
        m = 1
        while n >= m
            push!(basis, make_chi_func(m, n, L))
            m += 1
            n -= 1
        end
    end

    basis
end

function make_phi_basis(N, L)
    basis = [make_phi_func(1, 2, L)]

    for k in 4:N
        n = k - 1
        m = 1
        while n > m
            push!(basis, make_phi_func(m, n, L))
            m += 1
            n -= 1
        end
    end

    basis
end

function make_lincomb(basis, coeffs)
    function func(x1, x2)
        sum(c * f(x1, x2) for (f, c) in zip(basis, coeffs))
    end
end

function transform(basis, coeff_mat)
    [make_lincomb(basis, col) for col in eachcol(coeff_mat)]
end

function ovlp_integral(f, g, L)
    function func((x1, x2),)
        f(x1, x2) * g(x1, x2)
    end

    hcubature(func, (0, 0), (L, L); atol=1e-10)
end

function ovlp_matrix(basis, L, rbasis=basis)
    mat = zeros(length(basis), length(rbasis))

    for j in axes(mat, 2), i in axes(mat, 1)
        mat[i, j] = ovlp_integral(basis[i], rbasis[j], L)[1]
    end

    mat
end
