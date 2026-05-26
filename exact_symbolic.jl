using SymPy

x1, x2 = symbols("x1 x2")

L = symbols("L")

m, n = symbols("m n", integer=true, positive=true)

phi = (x1 - x2) * (sin(m * Sym(π) / L * x1) * sin(n * Sym(π) / L * x2) +
                   sin(n * Sym(π) / L * x1) * sin(m * Sym(π) / L * x2))
