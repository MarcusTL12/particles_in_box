using FFTW
using HCubature

function potential1d(q1, q2, x1, x2, d)
    r = √((x1 - x2)^2 + d^2)

    -2π * r * q1 * q2
end

function potential3d(q1, q2, x1, x2, d)
    q1 * q2 / √((x1 - x2)^2 + d^2)
end

function potential_square(q1, q2, x1, x2, d)
    if abs(x1 - x2) < d
        q1 * q2
    else
        0.0
    end
end

potential(params...) = potential3d(params...)

function potential_atom(q_atom, q1, q2, x1, x2, d, L)
    potential(q1, q2, x1, x2, d) +
    potential(q1, q_atom, x1, L / π, d) +
    potential(q1, q_atom, x1, 2L / π, d) +
    potential(q2, q_atom, x2, L / π^2, d) +
    potential(q2, q_atom, x2, 3L / π^2, d)
end

function cosine_potential_integral(m, n, L, potential_func)
    πonL = π / L

    prefac = 1 / L^2

    function combined_function((x1, x2))
        prefac *
        cos(m * πonL * x1) *
        cos(n * πonL * x2) *
        potential_func(x1, x2)
    end

    hcubature(combined_function, (0, 0), (L, L); atol=1e-10)
end

function construct_cosine_integral_matrix(N, L, pot_func)
    V = zeros(2N + 1, 2N + 1)

    sig_ind_pairs = [(m, n) for n in 0:2N for m in 0:2N]

    println("N integrals = ", length(sig_ind_pairs))

    Threads.@threads :greedy for (m, n) in sig_ind_pairs
        V[begin+m, begin+n] = cosine_potential_integral(m, n, L, pot_func)[1]
    end

    V
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

function do_dct(N, M, L, pot_func)
    xs = range(0, L, length=(M + 1))[1:end-1]

    xs = xs .+ step(xs) / 2

    println("Sampling potential at $(M^2) points:")

    sample = @time sample_function(xs, xs, pot_func)

    println("Computing DCT:")
    @time begin
        dct_plan = FFTW.plan_r2r!(sample, FFTW.REDFT10; num_threads=12)
        FFTW.mul!(sample, dct_plan, sample)
    end

    result = sample[1:(2N+1), 1:(2N+1)]

    result .*= 1 / (4 * M^2)

    result
end
