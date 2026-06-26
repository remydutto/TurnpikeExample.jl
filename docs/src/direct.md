# Direct methods

Consider the following optimal control problem:

```math
\left\{
\begin{array}{ll}
\displaystyle \min_{x,y,u}
J(x,y,u)
\coloneqq
\frac{1}{2}\int_0^T \left(x^2(t) + y^2(t)\right)\,\mathrm dt,
\\[1em]

\dot x(t) = u(t),
\\[0.5em]

\displaystyle \dot y(t) =- u(t) - \frac{y(t)}{2},
\\[0.5em]

-M \leq u(t) \leq M,
\\[0.5em]

(x(0), y(0)) = (1, -0.5),
\qquad
(x(T), y(T)) = (0.5, 0.5).
\end{array}
\right.
```

The associated stationary problem is

```math
\min_{(x,y,u)}
\left\{
x^2 + y^2
\;\middle|\;
(x,y,u) \in \mathbb R^2 \times [-M,M],
\quad
u = 0,
\quad
u - \frac{y}{2} = 0
\right\}.
```

## Packages

```@example main
using OptimalControl
using NLPModelsIpopt
using Plots
```

## Auxiliary formulation

To reduce the optimal control problem to a Mayer form, we introduce an auxiliary state `c` such that

```math
\dot c(t)
=
\frac{1}{2}\left(x^2(t) + y^2(t)\right),
\qquad
c(0)=0.
```

Then minimizing

```math
\frac{1}{2}\int_0^T \left(x^2(t)+y^2(t)\right)\,\mathrm dt
```

is equivalent to minimizing `c(T)`.

We define

```math
s(t) = (c(t),x(t),y(t)).
```

The dynamics can be written as

```math
\dot s(t)
=
F_0(s(t)) + u(t)F_1(s(t)),
```

where

```math
F_0(s)
=
\begin{pmatrix}
\frac{1}{2}(x^2+y^2)
\\
0
\\
-\frac{y}{2}
\end{pmatrix},
\qquad
F_1(s)
=
\begin{pmatrix}
0
\\
1
\\
-1
\end{pmatrix}.
```


## Problem data

```@example main
T = 50.0
M = 1.0

s0 = [0.0, 1.0, -0.5]

xT = 0.5
yT = 0.5

nothing
```

```@example main
F0(s) = [
    0.5 * (s[2]^2 + s[3]^2),
    0.0,
    -s[3] / 2,
]

F1(s) = [
    0.0,
    1.0,
    -1.0,
]

nothing
```

## OCP definition

We define the optimal control problem as follows:

```@example main
ocp = @def begin
    t ∈ [0, T], time

    s = (c, x, y) ∈ R³, state
    u ∈ R, control

    s(0) == s0

    x(T) == xT
    y(T) == yT

    -M ≤ u(t) ≤ M

    ṡ(t) == F0(s(t)) + u(t) * F1(s(t))

    c(T) → min
end

nothing
```

## Direct solve

We first solve the problem with the default direct transcription method.

```@example main
direct_sol = solve(ocp)

nothing
```

```@example main
plt_direct = plot(
    direct_sol;
    label = "direct",
    size = (800, 800),
)

plt_direct
```

!!! warning "Control chattering"
    The control exhibits **chattering**: it switches infinitely fast between its bounds
    on a set of positive measure. This is consistent with a singular arc structure,
    where the Pontryagin maximum principle fails to determine the optimal control
    uniquely from the sign of the switching function.



## Integration schemes and continuation

Direct methods rely on a discretization of the dynamics on a time grid

```math
t_0 < t_1 < \cdots < t_N.
```

Several integration schemes are available.

| Symbol              | Method                    | Order |
| :------------------ | :------------------------ | :---: |
| `:euler`            | Explicit Euler            |   1   |
| `:trapeze`          | Trapezoidal rule          |   2   |
| `:midpoint`         | Implicit midpoint         |   2   |
| `:gauss_legendre_2` | Gauss--Legendre, 2 stages |   4   |
| `:gauss_legendre_3` | Gauss--Legendre, 3 stages |   6   |

A useful numerical strategy is continuation. One solves a sequence of increasingly accurate problems, using each solution as a warm start for the next one:

```math
(N_1,\varepsilon_1)
\longrightarrow
(N_2,\varepsilon_2)
\longrightarrow
\cdots
\longrightarrow
(N_k,\varepsilon_k),
\qquad
N_1 < \cdots < N_k,
\qquad
\varepsilon_1 > \cdots > \varepsilon_k.
```

We now compare three discretization schemes for a fixed grid size and tolerance.

```@example main
sol_trapeze = solve(
    ocp;
    grid_size = 150,
    scheme = :trapeze,
    tol = 1e-8,
    init = direct_sol,
)

sol_midpoint = solve(
    ocp;
    grid_size = 150,
    scheme = :midpoint,
    tol = 1e-8,
    init = direct_sol,
)

sol_gauss2 = solve(
    ocp;
    grid_size = 150,
    scheme = :gauss_legendre_2,
    tol = 1e-8,
    init = direct_sol,
)

nothing
```

```@example main
plt_gauss2 = plot(
    sol_gauss2;
    label = "Gauss--Legendre 2",
    size = (800, 800),
)

plt_gauss2
```

## Comparison of the controls

We compare the controls obtained with the different schemes on the initial time interval.

```@example main
u_trapeze(t) = control(sol_trapeze)(t)
u_midpoint(t) = control(sol_midpoint)(t)
u_gauss2(t) = control(sol_gauss2)(t)

plt_controls = plot(
    u_trapeze,
    0,
    10;
    label = "Trapeze",
    linewidth = 2,
    xlabel = "t",
    ylabel = "u(t)",
    frame = :box,
    grid = true,
    size = (820, 480),
)

plot!(
    plt_controls,
    u_midpoint,
    0,
    10;
    label = "Midpoint",
    linewidth = 2,
)

plot!(
    plt_controls,
    u_gauss2,
    0,
    10;
    label = "Gauss--Legendre 2",
    linewidth = 2,
)

plt_controls
```

## Continuation example

We can also explicitly perform a continuation procedure by first solving a coarse problem and then refining the discretization.

```@example main
sol_continuation = solve(
    ocp;
    grid_size = 50,
    scheme = :midpoint,
    tol = 1e-4,
)

sol_continuation = solve(
    ocp;
    grid_size = 150,
    scheme = :gauss_legendre_2,
    tol = 1e-8,
    init = sol_continuation,
)

nothing
```

```@example main
plt_continuation = plot(
    sol_continuation;
    label = "continuation",
    size = (800, 800),
)

plt_continuation
```

## Reducing control chattering

When the optimal control is bang-bang or involves a singular arc, direct methods
inevitably produce rapid oscillations at the grid scale — the discretized control
switches between its bounds to approximate an infinite switching frequency it cannot resolve.

A standard remedy is to add a small quadratic regularization term to the cost:

$$
J_\varepsilon(x,y,u)
= \frac{1}{2}\int_0^T \left(x^2(t)+y^2(t)\right)\mathrm{d}t
+ \varepsilon \int_0^T u^2(t)\,\mathrm{d}t,
$$

where $0 < \varepsilon \ll 1$. The penalty on $u^2$ makes the Hamiltonian strictly
concave in $u$, which regularizes the bang-bang structure and yields a smooth,
numerically well-behaved control. The solution then converges to the original
bang-bang optimal control as $\varepsilon \to 0$.

In the auxiliary formulation, this corresponds to replacing the cost-state dynamics by

```math
\dot c(t)
=
\frac{1}{2}\left(x^2(t)+y^2(t)\right)
+
\varepsilon u^2(t).
```

```@example main
ε = 1e-3

ocp_regularized = @def begin
    t ∈ [0, T], time

    s = (c, x, y) ∈ R³, state
    u ∈ R, control

    s(0) == s0

    x(T) == xT
    y(T) == yT

    -M ≤ u(t) ≤ M

    ṡ(t) == [
        0.5 * (x(t)^2 + y(t)^2) + ε * u(t)^2,
        u(t),
        -u(t) - y(t) / 2,
    ]

    c(T) → min
end

nothing
```

```@example main
sol_regularized = solve(
    ocp_regularized;
    grid_size = 150,
    scheme = :trapeze,
    tol = 1e-8,
    init = sol_trapeze,
)

nothing
```

```@example main
plt_regularized = plot(
    sol_regularized;
    label = "regularized",
    size = (800, 800),
)

plt_regularized
```

We now compare the regularized control with the controls obtained before.

```@example main
u_regularized(t) = control(sol_regularized)(t)

plt_regularized_control = plot(
    u_trapeze,
    0,
    10;
    label = "Trapeze",
    linewidth = 1.5,
    xlabel = "t",
    ylabel = "u(t)",
    frame = :box,
    grid = true,
    size = (820, 480),
)

plot!(
    plt_regularized_control,
    u_gauss2,
    0,
    10;
    label = "Gauss--Legendre 2",
    linewidth = 1.5,
)

plot!(
    plt_regularized_control,
    u_regularized,
    0,
    10;
    label = "Trapeze + L² regularization",
    linewidth = 2.5,
)

plt_regularized_control
```

!!! note
    The regularized problem is a perturbation of the original — its solution differs
    from the true bang-bang control, but converges to it as ε → 0. In practice, it
    is used either as a smoother when the bang-bang structure makes the direct solve
    unstable, or as a warm start for a subsequent solve at ε = 0.
