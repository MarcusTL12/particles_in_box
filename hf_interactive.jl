using GLMakie
using IntelVectorMath

include("hf.jl")

function evaluate_sin_basis_naive(N, nx)
    xs = range(0, π, length=nx)

    ys = zeros(length(xs), N)

    all_xs = zeros(length(xs), N)

    for n in 1:N, (i, x) in enumerate(xs)
        all_xs[i, n] = n * x
    end

    IVM.sin!(ys, all_xs)

    ys
end

function evaluate_sin_basis(N, nx)
    xs = range(0, π, length=nx)[1:(end-1)]

    ys = zeros(nx, N)

    IVM.sin!((@view ys[1:(end-1), 1]), collect(xs))
    ys[end, 1] = ys[1, 1]

    @inbounds for n in 2:N
        j = 0
        s = false
        for i in 0:(nx-1)
            ys[begin+i, n] = s ? ys[begin+j, 1] : -ys[begin+j, 1]

            j += n
            if j >= nx - 1
                j -= nx - 1
                s = !s
            end
        end
    end

    ys
end

function interactive()
    f = Figure(size=(1920, 1080))

    axis_orb = Axis(f[1, 1][1, 1])
    axis_potential = Axis(f[1, 1][2, 1])

    N = Observable(5)
    L = Observable(1.0)
    d = Observable(0.05)
    nocc = Observable(1)

    basis_eval = lift(N) do N
        evaluate_sin_basis(N, 1000)
    end

    e, C = do_hf_diis(N[], L[], nothing, d[], nocc[])

    f
end
