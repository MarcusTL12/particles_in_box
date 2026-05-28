using HCubature
using LinearAlgebra
using GLMakie
using Printf

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

function potential_atom(q1, q2, x1, x2, d, L)
    q_atom = 1.0

    potential(q1, q2, x1, x2, d) +
    potential(q1, q_atom, x1, L / 2, d) +
    potential(q2, q_atom, x2, L / 2, d)
end

# ⟨mn|V|rs⟩
function potential_integral((m, n), (r, s), q1, q2, d, L)
    πonL = π / L

    prefac = 4 / L^2

    function combined_function((x1, x2))
        prefac *
        sin(m * πonL * x1) * sin(n * πonL * x2) *
        sin(r * πonL * x1) * sin(s * πonL * x2) *
        potential_atom(q1, q2, x1, x2, d, L)
    end

    hcubature(combined_function, (0, 0), (L, L); atol=1e-10)[1]
end

function cosine_potential_integral(m, n, q1, q2, d, L)
    πonL = π / L

    prefac = 1 / L^2

    function combined_function((x1, x2))
        prefac *
        cos(m * πonL * x1) *
        cos(n * πonL * x2) *
        potential_atom(q1, q2, x1, x2, d, L)
    end

    hcubature(combined_function, (0, 0), (L, L); atol=1e-5, maxevals=1000000)[1]
end

function construct_hamiltonian(N, q1, q2, m1, m2, d, L)
    V = zeros(2N + 1, 2N + 1)

    # sig_ind_pairs = [(m, n) for n in 0:2N for m in n:2:2N]
    # sig_ind_pairs = [(m, n) for n in 0:2N for m in (n % 2):2:2N]
    sig_ind_pairs = [(m, n) for n in 0:2N for m in (n % 2):2:2N]

    println("N integrals = ", length(sig_ind_pairs))

    Threads.@threads :greedy for (m, n) in sig_ind_pairs
        V[begin+m, begin+n] = cosine_potential_integral(m, n, q1, q2, d, L)
    end

    # display(V)

    # V = Symmetric(V, :L)

    H = zeros(N, N, N, N)

    for s in 1:N, r in 1:N, n in 1:N, m in 1:N
        # H[m, n, r, s] = potential_integral((m, n), (r, s), q1, q2, d, L)
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

function interactive()
    f = Figure(size=(1920, 1080))

    aspect = (1.0, 1.0, 0.5)

    ax2d = Axis(f[1, 1][1, 1])
    ax3d = Axis3(f[1, 1][1, 2];
        viewmode=:fitzoom, aspect=aspect, perspectiveness=0.5)

    axpot = Axis(f[1, 1][2, 1]; height=200)

    config_panel = f[1, 2]

    L = Observable(1.0)

    xs = @lift range(0, $L, length=200)
    xs_wireframe = @lift range(0, $L, length=50)

    m = 1
    n = 1

    zs = Observable([sin(m * π / L[] * x) * sin(n * π / L[] * y)
                     for x in xs[], y in xs[]])
    zs_wireframe = [sin(m * π / L[] * x) * sin(n * π / L[] * y)
                    for x in xs_wireframe[], y in xs_wireframe[]]

    colormap = :redsblues

    colorrange = lift(zs) do vals
        edge = maximum(abs, vals)

        (-edge, edge)
    end

    colorlevels = lift(zs) do vals
        edge = maximum(abs, vals) * 1.1

        range(-edge, edge, length=20)
    end

    contourf!(ax2d, xs, xs, zs; colormap=colormap, levels=colorlevels)
    surface!(ax3d, xs, xs, zs;
        transparency=true, colormap=colormap, colorrange=colorrange)
    my_wf = wireframe!(ax3d, xs_wireframe, xs_wireframe, zs_wireframe;
        overdraw=true, transparency=true,
        color=(:black, 0.1))

    d_slider = Slider(config_panel[1, 1][2, 1];
        range=0.01:0.01:1.0, startvalue=0.2, horizontal=false).value

    Label(config_panel[1, 1][1, 1], @lift @sprintf "d = %.2f" $d_slider)

    n_max_default = "1"
    q1_default = "-1"
    q2_default = "-1"
    m1_default = "1"
    m2_default = "1"
    state_default = "1"
    L_default = "1.0"

    energy = Observable(0.0)

    Label(config_panel[2, 1][1, 1], "n_max =")
    max_n_box = Textbox(config_panel[2, 1][1, 2];
        validator=Int, placeholder=n_max_default)
    max_n_box.stored_string[] = n_max_default

    Label(config_panel[2, 1][2, 1], "q1 =")
    q1_box = Textbox(config_panel[2, 1][2, 2];
        validator=Float64, placeholder=q1_default)
    q1_box.stored_string[] = q1_default

    Label(config_panel[2, 1][3, 1], "q2 =")
    q2_box = Textbox(config_panel[2, 1][3, 2];
        validator=Float64, placeholder=q2_default)
    q2_box.stored_string[] = q2_default

    Label(config_panel[2, 1][4, 1], "m1 =")
    m1_box = Textbox(config_panel[2, 1][4, 2];
        validator=Float64, placeholder=m1_default)
    m1_box.stored_string[] = m1_default

    Label(config_panel[2, 1][5, 1], "m2 =")
    m2_box = Textbox(config_panel[2, 1][5, 2];
        validator=Float64, placeholder=m2_default)
    m2_box.stored_string[] = m2_default

    Label(config_panel[2, 1][6, 1], "L =")
    L_box = Textbox(config_panel[2, 1][6, 2];
        validator=Float64, placeholder=L_default)
    L_box.stored_string[] = L_default

    Label(config_panel[2, 1][7, 1], "state =")
    state_box = Textbox(config_panel[2, 1][7, 2];
        validator=Float64, placeholder=state_default)
    state_box.stored_string[] = state_default

    Label(config_panel[2, 1][8, 1], (@lift @sprintf "E = %.3f" $energy))

    is_solving = Observable(false)

    solve_btn = Button(config_panel[3, 1];
        label=(@lift $is_solving ? "Solving..." : "  Solve!   "))

    max_n = lift(max_n_box.stored_string) do text
        parse(Int, text)
    end

    q1 = lift(q1_box.stored_string) do text
        parse(Float64, text)
    end

    q2 = lift(q2_box.stored_string) do text
        parse(Float64, text)
    end

    m1 = lift(m1_box.stored_string) do text
        parse(Float64, text)
    end

    m2 = lift(m2_box.stored_string) do text
        parse(Float64, text)
    end

    on(L_box.stored_string) do text
        L[] = parse(Float64, text)
    end

    pot_xs = @lift range(-$L, $L, length=1000)
    pot_ys = @lift [potential($q1, $q2, 0, x, $d_slider) for x in $pot_xs]

    on(L) do L
        xlims!(axpot, (-L, L))
        xlims!(ax2d, (0, L))
        ylims!(ax2d, (0, L))
    end

    pot_ylims = @lift ($q1, $q2, $d_slider)

    on(pot_ylims) do _
        ylims!(axpot, (nothing, nothing))
    end
    lines!(axpot, pot_xs, pot_ys)

    energies = Observable(zeros(0))
    coeffs = Observable(zeros(0, 0))

    is_solving = Observable(false)

    state = Observable(1)

    on(state_box.stored_string) do text
        state[] = parse(Int, text)
    end

    onany(state, energies) do i, _
        energy[] = energies[][i]
    end

    on(is_solving) do val
        if !val
            notify(state)
        end
    end

    basis_eval_wireframe = Observable(zeros(0, 0))

    basis_eval = lift(max_n) do N
        πonL = π / L[]
        basis_eval_wireframe[] = [sin(n * πonL * x)
                                  for n in 1:N, x in xs_wireframe[]]
        [sin(n * πonL * x) for n in 1:N, x in xs[]]
    end

    on(solve_btn.clicks) do _
        if !is_solving[]
            is_solving[] = true

            @async begin
                println("Constructing Hamiltonian:")
                @show q1 q2
                H = @time construct_hamiltonian(
                    max_n[], q1[], q2[], m1[], m2[], d_slider[], L[])

                println("Diagonalizing $(size(H, 1))x$(size(H, 2)) matrix")

                e, C = @time eigen(H)

                energies[] = e
                coeffs[] = C

                is_solving[] = false
            end
        end
    end

    onany(state, L) do state_ind, L
        print("Rendering...")

        be = basis_eval[]
        bew = basis_eval_wireframe[]

        C = @view coeffs[][:, state_ind]
        C = reshape(C, size(be, 1), size(be, 1))

        for j in 1:length(xs[]), i in 1:length(xs[])
            z = 0.0

            for n in axes(be, 1), m in axes(be, 1)
                z += be[m, i] * be[n, j] * C[m, n]
            end

            zs[][i, j] = z
        end

        s = 1.0

        if maximum(abs, zs[]) > maximum(zs[])
            s = -1.0
            zs[] .*= -1.0
        end

        for j in 1:length(xs_wireframe[]), i in 1:length(xs_wireframe[])
            z = 0.0

            for n in axes(bew, 1), m in axes(bew, 1)
                z += bew[m, i] * bew[n, j] * C[m, n] * s
            end

            zs_wireframe[i, j] = z
        end

        notify(zs)

        delete!(ax3d, my_wf)
        my_wf = wireframe!(ax3d, xs_wireframe, xs_wireframe, zs_wireframe;
            overdraw=true, transparency=true,
            color=(:black, 0.1))

        zlims!(ax3d, (nothing, nothing))

        println("Done!")
    end

    f
end
