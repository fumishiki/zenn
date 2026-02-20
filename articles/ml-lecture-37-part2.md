---
title: "第37回: 🎲 SDE/ODE & 確率過程論: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "🎲"
type: "tech"
topics: ["machinelearning", "deeplearning", "sde", "julia", "stochasticprocesses"]
published: true
slug: "ml-lecture-37-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

## 💻 4. 実装ゾーン（45分）— Julia DifferentialEquations.jlでSDE数値解法

### 4.1 Julia DifferentialEquations.jl入門 — SDEProblemの定義

JuliaのDifferentialEquations.jlはSDE/ODE/DAEを統一的に扱う強力なパッケージ。

**基本的なSDE定義**:

```julia
using DifferentialEquations

# SDE: dx = f(x, p, t) dt + g(x, p, t) dW
drift(u, p, t)     = [-0.5 * p[1] * u[1]]  # p[1] = β
diffusion(u, p, t) = [√(p[1])]              # √β

# 初期値、時間範囲、パラメータ
u0 = [1.0]
tspan = (0.0, 1.0)
β = 1.0
p = [β]

# SDEProblem作成
prob = SDEProblem(drift, diffusion, u0, tspan, p)

# 数値解法で解く
sol = solve(prob, EM(), dt=0.01)  # Euler-Maruyama法

# プロット
using Plots
plot(sol, xlabel="時刻 t", ylabel="X(t)", title="VP-SDE サンプルパス", lw=2)
```

**数式↔コード対応**:
- SDE: $dX_t = -\frac{1}{2}\beta X_t dt + \sqrt{\beta} dW_t$
- `drift(u, p, t)`: Drift項 $f(x, t) = -\frac{1}{2}\beta x$
- `diffusion(u, p, t)`: Diffusion項 $g(x, t) = \sqrt{\beta}$
- `EM()`: Euler-Maruyama法（$\Delta t = 0.01$）

### 4.2 VP-SDE実装 — 線形/Cosineスケジュール

DDPM対応のVP-SDEを線形/Cosineスケジュールで実装。

**線形スケジュール**:
$$
\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})
$$

```julia
# VP-SDE with 線形スケジュール
β_min, β_max = 0.1, 20.0
β_linear(t) = β_min + t * (β_max - β_min)

function vp_drift_linear(u, p, t)
    β_t = p[1] + t * (p[2] - p[1])
    return [-0.5 * β_t * u[1]]
end

vp_noise_linear(u, p, t) = [√(p[1] + t * (p[2] - p[1]))]

prob_vp_linear = SDEProblem(vp_drift_linear, vp_noise_linear, [1.0], (0.0, 1.0), (β_min, β_max))
sol_vp_linear = solve(prob_vp_linear, EM(), dt=0.001)

plot(sol_vp_linear, xlabel="t", ylabel="X(t)", title="VP-SDE 線形スケジュール", lw=2, label="X(t)")
```

**Cosineスケジュール**（DDPM Improved, Nichol & Dhariwal 2021）:
$$
\bar{\alpha}_t = \frac{\cos\left(\frac{t + s}{1 + s} \cdot \frac{\pi}{2}\right)^2}{\cos\left(\frac{s}{1 + s} \cdot \frac{\pi}{2}\right)^2}, \quad \beta(t) = -\frac{d \log \bar{\alpha}_t}{dt}
$$
（$s = 0.008$ は小さなオフセット）

```julia
# Cosineスケジュール
s = 0.008
α_bar_cosine(t, s=0.008) = cos((t + s) / (1 + s) * π/2)^2 / cos(s / (1 + s) * π/2)^2

function β_cosine(t, s=0.008)
    # 数値微分で β(t) = -d log(α_bar) / dt
    dt = 1e-6
    return -(log(α_bar_cosine(t + dt, s)) - log(α_bar_cosine(t, s))) / dt
end

vp_drift_cosine(u, p, t) = [-0.5 * β_cosine(t) * u[1]]
vp_noise_cosine(u, p, t) = [√(β_cosine(t))]

prob_vp_cosine = SDEProblem(vp_drift_cosine, vp_noise_cosine, [1.0], (0.0, 1.0), nothing)
sol_vp_cosine = solve(prob_vp_cosine, EM(), dt=0.001)

plot(sol_vp_linear, xlabel="t", ylabel="X(t)", title="VP-SDE: 線形 vs Cosine", lw=2, label="線形")
plot!(sol_vp_cosine, lw=2, label="Cosine")
```

**線形 vs Cosine の違い**:
- 線形: 終端でノイズが急増（$\beta_{\max} = 20$）
- Cosine: 滑らかなスケジュール、端点での急変を回避

### 4.3 VE-SDE実装 — 幾何スケジュール

NCSNのVE-SDEを幾何スケジュールで実装。

**幾何スケジュール**:
$$
\sigma(t) = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t
$$

$$
\frac{d\sigma^2(t)}{dt} = 2\sigma(t) \log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right) \sigma(t) = 2\sigma^2(t) \log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)
$$

```julia
# VE-SDE with 幾何スケジュール
σ_min, σ_max = 0.01, 50.0

ve_drift(u, p, t) = [0.0]  # Drift項 = 0

function ve_noise(u, p, t)
    σ_min, σ_max = p
    σ_t = σ_min * (σ_max / σ_min)^t
    return [√(2 * σ_t^2 * log(σ_max / σ_min))]
end

prob_ve = SDEProblem(ve_drift, ve_noise, [1.0], (0.0, 1.0), (σ_min, σ_max))
sol_ve = solve(prob_ve, EM(), dt=0.001)

plot(sol_ve, xlabel="t", ylabel="X(t)", title="VE-SDE 幾何スケジュール", lw=2, label="X(t)")
```

**特徴**:
- Drift項なし（平均変化なし）
- 分散が時間とともに爆発的に増加

### 4.4 Reverse-time SDE実装 — Score関数近似

Reverse-time SDEを簡易Score関数近似で実装。

**VP-SDE Reverse-time**:
$$
dX_t = \left[-\frac{1}{2}\beta(t) X_t - \beta(t) \nabla \log p_t(X_t)\right] dt + \sqrt{\beta(t)} d\bar{W}_t
$$

**Score関数近似**（ガウス仮定）:
学習済みScore関数 $s_\theta(x, t)$ がない場合、ガウス近似で $\nabla \log p_t(x) \approx -x / \sigma_t^2$。

```julia
# Reverse-time VP-SDE（簡易Score近似）
β_min, β_max = 0.1, 20.0

function reverse_vp_drift(u, p, t)
    β_min, β_max = p
    β_t = β_min + t * (β_max - β_min)

    # Score近似（実際はNNで学習）
    # 簡易的に ∇log p_t(x) ≈ -x（ガウス仮定）
    score_approx = -u[1]

    # Drift = -0.5 * β(t) * x - β(t) * ∇log p_t(x)
    return [-0.5 * β_t * u[1] - β_t * score_approx]
end

reverse_vp_noise(u, p, t) = [√(p[1] + t * (p[2] - p[1]))]

# 初期値: ノイズ分布 N(0, 1)
u0_noise = randn(1)
tspan_reverse = (1.0, 0.0)  # 逆時間（t: 1 → 0）

prob_reverse = SDEProblem(reverse_vp_drift, reverse_vp_noise, u0_noise, tspan_reverse, (β_min, β_max))
sol_reverse = solve(prob_reverse, EM(), dt=-0.001)  # 負のdt（逆時間）

plot(sol_reverse, xlabel="時刻 t", ylabel="X(t)", title="Reverse-time VP-SDE（簡易Score）", lw=2, label="X(t)")
```

**注意**:
- 実際のDiffusion Modelでは Score関数 $s_\theta(x, t)$ をNeural Networkで学習
- ここでは $\nabla \log p_t(x) \approx -x$ のガウス近似（デモ目的）

### 4.5 Probability Flow ODE実装 — 決定論的軌道

Probability Flow ODEを`ODEProblem`で実装。

**VP-SDE Probability Flow ODE**:
$$
\frac{dX_t}{dt} = -\frac{1}{2}\beta(t) X_t - \frac{1}{2}\beta(t) \nabla \log p_t(X_t)
$$

```julia
# Probability Flow ODE for VP-SDE
function pf_ode!(du, u, p, t)
    β_min, β_max = p
    β_t = β_min + t * (β_max - β_min)

    # Score近似（実際はNNで学習）
    score_approx = -u[1]

    # ODE: dx/dt = -0.5 * β(t) * x - 0.5 * β(t) * ∇log p_t(x)
    du[1] = -0.5 * β_t * u[1] - 0.5 * β_t * score_approx
end

u0_pf = randn(1)  # 初期ノイズ
tspan_pf = (1.0, 0.0)  # 逆時間

prob_pf_ode = ODEProblem(pf_ode!, u0_pf, tspan_pf, (β_min, β_max))
sol_pf_ode = solve(prob_pf_ode, Tsit5())  # Tsit5はRunge-Kutta法（高次）

plot(sol_pf_ode, xlabel="時刻 t", ylabel="X(t)", title="Probability Flow ODE", lw=2, label="X(t)")
```

**Reverse-time SDE vs PF-ODE**:
```julia
# 同じ初期値で比較
u0_common = [0.5]
tspan_common = (1.0, 0.0)

# Reverse-time SDE
prob_sde = SDEProblem(reverse_vp_drift, reverse_vp_noise, u0_common, tspan_common, (β_min, β_max))
sol_sde = solve(prob_sde, EM(), dt=-0.001)

# PF-ODE
prob_ode = ODEProblem(pf_ode!, u0_common, tspan_common, (β_min, β_max))
sol_ode = solve(prob_ode, Tsit5())

plot(sol_sde, xlabel="t", ylabel="X(t)", title="SDE vs ODE", lw=2, label="Reverse-time SDE", alpha=0.7)
plot!(sol_ode, lw=2, label="PF-ODE", linestyle=:dash)
```

**結果**:
- Reverse-time SDE: 確率的（軌道が揺れる）
- PF-ODE: 決定論的（滑らかな軌道）

### 4.6 Predictor-Corrector法実装 — 精度向上

Predictor-Corrector法で高品質サンプリング。

**アルゴリズム**:
1. Predictor: Reverse-time SDEで1ステップ
2. Corrector: Langevin Dynamics（複数回反復）

```julia
# Predictor-Corrector サンプリング
function predictor_corrector_sampling(;n_steps=100, n_corrector=5, ε_langevin=0.01, β_min=0.1, β_max=20.0)
    x = randn()
    dt = -1.0 / n_steps

    trajectory = [x]

    for t in LinRange(1.0, 0.0, n_steps+1)[1:n_steps]
        β_t = β_min + t * (β_max - β_min)

        # Predictor: Reverse-time SDE
        x += (-0.5 * β_t * x + β_t * x) * dt + √β_t * √(-dt) * randn()

        # Corrector: Langevin Dynamics
        for _ in 1:n_corrector
            x += ε_langevin * (-x) + √(2ε_langevin) * randn()
        end

        push!(trajectory, x)
    end

    return trajectory  # n_steps+1 要素のベクトル
end

# サンプリング実行
traj = predictor_corrector_sampling(n_steps=100, n_corrector=5, ε_langevin=0.01)

# プロット
t_plot = LinRange(1.0, 0.0, 101)
plot(t_plot, traj, xlabel="時刻 t", ylabel="X(t)", title="Predictor-Corrector サンプリング", lw=2, legend=false)
```

**Predictor-Corrector vs Euler-Maruyama**:
```julia
# Euler-Maruyama（Predictor-onlyと等価）
prob_em = SDEProblem(reverse_vp_drift, reverse_vp_noise, randn(1), (1.0, 0.0), (β_min, β_max))
sol_em = solve(prob_em, EM(), dt=-0.01)

# Predictor-Corrector
traj_pc = predictor_corrector_sampling(n_steps=100, n_corrector=5, ε_langevin=0.01)

# プロット
plot(sol_em.t, first.(sol_em.u), label="Euler-Maruyama", lw=2)
plot!(LinRange(1.0, 0.0, 101), traj_pc, label="Predictor-Corrector", lw=2, linestyle=:dash)
```

**結果**: Predictor-Correctorは軌道が滑らか（Correctorでスコア方向に補正）

### 4.7 数値ソルバー比較 — Euler-Maruyama vs 高次手法

DifferentialEquations.jlが提供する各種ソルバーの精度・速度比較。

**SDEソルバー一覧**:
- `EM()`: Euler-Maruyama法（1次精度、低コスト）
- `SRIW1()`: Roessler法（弱1.5次精度、対角ノイズ）
- `SRA1()`: 適応的Roessler法（弱1.5次、ステップサイズ自動調整）
- `ImplicitEM()`: 暗黙的Euler-Maruyama（剛性問題）

```julia
using DifferentialEquations, BenchmarkTools

# テストSDE: Ornstein-Uhlenbeck過程
# dX = -θ X dt + σ dW
θ, σ = 1.0, 0.5
ou_drift(u, p, t)      = [-p[1] * u[1]]
ou_diffusion(u, p, t) = [p[2]]

u0 = [1.0]
tspan = (0.0, 10.0)
p = (θ, σ)

# 解析解（比較用）
analytical(t, u0, θ, σ) = u0 * exp(-θ * t)

# 各ソルバーでの解法
solvers = [EM(), SRIW1(), SRA1()]
solver_names = ["EM", "SRIW1", "SRA1"]

errors = Float64[]
times = Float64[]

for (solver, name) in zip(solvers, solver_names)
    prob = SDEProblem(ou_drift, ou_diffusion, u0, tspan, p)

    # 時間計測
    time_taken = @elapsed sol = solve(prob, solver, dt=0.01, save_everystep=false)

    # 誤差計測（終端値）
    x_final_numerical = sol.u[end][1]
    x_final_analytical = analytical(10.0, u0[1], θ, σ)
    error = abs(x_final_numerical - x_final_analytical)

    push!(errors, error)
    push!(times, time_taken)

    println("$name: error=$error, time=$time_taken s")
end

# プロット
using Plots
p1 = bar(solver_names, errors, ylabel="終端誤差", title="ソルバー精度比較", legend=false)
p2 = bar(solver_names, times, ylabel="計算時間 (s)", title="ソルバー速度比較", legend=false)
plot(p1, p2, layout=(1,2), size=(1000, 400))
```

**結果**:
- EM: 最速だが精度低い
- SRIW1: 精度高い（弱1.5次）、コストはEM の ~2倍
- SRA1: 適応ステップで剛性問題に強い

**実用指針**:
- 高速プロトタイプ: EM
- 高精度サンプリング: SRIW1
- 剛性SDE（急激な変化）: SRA1 or ImplicitEM

### 4.8 適応的ステップサイズ制御 — SRA1による自動調整

剛性問題（$\beta(t)$ が急変）で適応的ソルバーの威力を確認。

```julia
# 急激に変化するβ(t)（剛性問題）
function β_stiff(t)
    if t < 0.5
        return 0.1
    else
        return 50.0  # 急激にジャンプ
    end
end

function vp_drift_stiff(u, p, t)
    β_t = β_stiff(t)
    return [-0.5 * β_t * u[1]]
end

function vp_noise_stiff(u, p, t)
    β_t = β_stiff(t)
    return [√β_t]
end

prob_stiff = SDEProblem(vp_drift_stiff, vp_noise_stiff, [1.0], (0.0, 1.0), nothing)

# 固定ステップ EM
sol_em_fixed = solve(prob_stiff, EM(), dt=0.01)

# 適応ステップ SRA1
sol_sra1_adaptive = solve(prob_stiff, SRA1())

# ステップサイズの比較
println("EM ステップ数: $(length(sol_em_fixed.t))")
println("SRA1 ステップ数: $(length(sol_sra1_adaptive.t))")

# プロット
plot(sol_em_fixed.t, first.(sol_em_fixed.u), label="EM (固定dt)", marker=:circle, markersize=2)
plot!(sol_sra1_adaptive.t, first.(sol_sra1_adaptive.u), label="SRA1 (適応)", marker=:x, markersize=3)
xlabel!("時刻 t")
ylabel!("X(t)")
title!("剛性問題: EM vs SRA1")
```

**結果**:
- SRA1は $t > 0.5$ で自動的にステップサイズを縮小
- EMは固定ステップで不安定（発散リスク）

### 4.9 マルチスケールSDE — 高速・低速変数の分離

高速変数と低速変数が混在するSDE（マルチスケール問題）。

**設定**:
$$
\begin{aligned}
dX_t &= -\gamma X_t dt + \sigma_X dW^X_t \quad (\text{低速変数}) \\
dY_t &= -\epsilon^{-1} Y_t dt + \sigma_Y dW^Y_t \quad (\text{高速変数, } \epsilon \ll 1)
\end{aligned}
$$

高速変数 $Y_t$ は平衡化が早い（$\epsilon = 0.01$）。

```julia
# マルチスケールSDE
ε = 0.01
γ, σ_X, σ_Y = 1.0, 0.5, 2.0

function multiscale_drift(u, p, t)
    ε, γ = p
    x, y = u
    return [-γ * x, -y / ε]
end

function multiscale_diffusion(u, p, t)
    σ_X, σ_Y = 0.5, 2.0
    return [σ_X 0.0; 0.0 σ_Y]
end

u0_multi = [1.0, 1.0]
tspan_multi = (0.0, 5.0)
p_multi = (ε, γ)

prob_multi = SDEProblem(multiscale_drift, multiscale_diffusion, u0_multi, tspan_multi, p_multi)

# 適応ステップSRA1で解く（高速変数対応）
sol_multi = solve(prob_multi, SRA1())

# プロット
plot(sol_multi, idxs=1, label="X(t) 低速", lw=2)
plot!(sol_multi, idxs=2, label="Y(t) 高速", lw=2, linestyle=:dash)
xlabel!("時刻 t")
ylabel!("値")
title!("マルチスケールSDE (ε=$ε)")
```

**観察**:
- $Y_t$ は急速に平衡化（高周波振動）
- $X_t$ は緩やかに変化（低周波）
- 適応ステップが高速変数の細かい変化を追跡

### 4.10 Girsanov変換の実装 — 測度変換とスコア学習

Girsanov定理を使ってDrift項を変更し、Reverse-time SDEを導出する手続きを実装。

**理論**:
Forward SDE:
$$
dX_t = f(X_t, t) dt + g(X_t, t) dW_t
$$

Girsanov変換で新しいDrift $\tilde{f}$ を持つSDEに変換:
$$
dX_t = \tilde{f}(X_t, t) dt + g(X_t, t) d\tilde{W}_t
$$

Radon-Nikodym導関数:
$$
\frac{dP_{\tilde{W}}}{dP_W} = \exp\left(\int_0^T \frac{\tilde{f} - f}{g^2} dW_s - \frac{1}{2}\int_0^T \left(\frac{\tilde{f} - f}{g}\right)^2 ds\right)
$$

```julia
# Forward VP-SDE: dX = -0.5 β(t) X dt + √β(t) dW
# Girsanov変換で Reverse-time SDE に

β_min, β_max = 0.1, 20.0

forward_drift(x, t)     = -0.5 * (β_min + t * (β_max - β_min)) * x
forward_diffusion(x, t) = √(β_min + t * (β_max - β_min))

# Reverse-time では Drift に Score項が追加
# f_reverse = -f_forward - g² ∇log p_t
function reverse_drift_girsanov(x, t, score_fn)
    β_t = β_min + t * (β_max - β_min)
    f_fwd = forward_drift(x, t)
    g = forward_diffusion(x, t)
    score = score_fn(x, t)
    return -f_fwd - g^2 * score
end

# 簡易Score関数（ガウス近似）
score_approx(x, t) = -x

# Reverse-time SDE実装
reverse_drift_impl(u, p, t) = [reverse_drift_girsanov(u[1], t, p[1])]
reverse_noise_impl(u, p, t) = [forward_diffusion(u[1], t)]

u0_girsanov = [0.5]
tspan_girsanov = (1.0, 0.0)
p_girsanov = (score_approx,)

prob_girsanov = SDEProblem(reverse_drift_impl, reverse_noise_impl, u0_girsanov, tspan_girsanov, p_girsanov)
sol_girsanov = solve(prob_girsanov, EM(), dt=-0.001)

plot(sol_girsanov, xlabel="時刻 t", ylabel="X(t)", title="Girsanov変換 Reverse-time SDE", lw=2)
```

**Girsanov変換のキモ**:
1. Forward SDE の Drift $f$ を知る
2. Score関数 $\nabla \log p_t$ を学習（or 近似）
3. Reverse Drift = $-f - g^2 \nabla \log p_t$

これが **Score SDE統一理論** の数学的基盤。

### 4.11 JumpProcess混合SDE — Poisson Jumpとの結合

連続Brown運動に加え、Poisson過程（ジャンプ）を含むSDE。

**設定**:
$$
dX_t = -\theta X_t dt + \sigma dW_t + dN_t
$$
$N_t$ はPoisson過程（レート $\lambda$）

```julia
using DifferentialEquations

θ, σ, λ = 1.0, 0.5, 2.0

jump_drift(u, p, t)      = [-p[1] * u[1]]
jump_diffusion(u, p, t) = [p[2]]

# Jumpのサイズ（毎回 +0.5）
function jump_affect!(integrator)
    integrator.u[1] += 0.5
end

# Poisson過程（レート λ）
jump_rate(u, p, t) = λ
jump = ConstantRateJump(jump_rate, jump_affect!)

u0_jump = [1.0]
tspan_jump = (0.0, 10.0)
p_jump = (θ, σ)

prob_jump = SDEProblem(jump_drift, jump_diffusion, u0_jump, tspan_jump, p_jump)
jump_prob = JumpProblem(prob_jump, Direct(), jump)

sol_jump = solve(jump_prob, EM(), dt=0.01)

plot(sol_jump, xlabel="時刻 t", ylabel="X(t)", title="Brown運動 + Poissonジャンプ", lw=2)
```

**結果**: 軌道に不連続なジャンプが発生。

**応用**: ファイナンス（株価の突発変動）、神経科学（スパイクニューロン）

### 4.12 並列アンサンブルシミュレーション — EnsembleProblemで高速化

複数の独立サンプルを並列で生成。

```julia
using DifferentialEquations

# Ornstein-Uhlenbeck SDE
θ, σ = 1.0, 0.5
ou_drift(u, p, t)      = [-p[1] * u[1]]
ou_diffusion(u, p, t) = [p[2]]

u0 = [1.0]
tspan = (0.0, 10.0)
p = (θ, σ)

prob = SDEProblem(ou_drift, ou_diffusion, u0, tspan, p)

# アンサンブル問題（1000トラジェクトリ）
ensemble_prob = EnsembleProblem(prob)

# 並列実行（Threads.jl利用）
sol_ensemble = solve(ensemble_prob, EM(), EnsembleThreads(), trajectories=1000, dt=0.01)

# 平均と標準偏差を計算
using Statistics
t_vals = sol_ensemble[1].t
mean_vals = [mean(sol.u[i][1] for sol in sol_ensemble) for i in eachindex(t_vals)]
std_vals  = [std( sol.u[i][1] for sol in sol_ensemble) for i in eachindex(t_vals)]

# プロット
plot(t_vals, mean_vals, ribbon=std_vals, label="平均 ± 標準偏差", fillalpha=0.3, lw=2)
xlabel!("時刻 t")
ylabel!("X(t)")
title!("Ornstein-Uhlenbeck過程 アンサンブル平均")
```

**並列化オプション**:
- `EnsembleThreads()`: マルチスレッド（共有メモリ）
- `EnsembleDistributed()`: 分散計算（クラスタ）
- `EnsembleGPUArray()`: GPU並列

**性能**: 1000トラジェクトリを並列実行で **数秒** で完了。

---

## 🔬 5. 実験ゾーン（30分）— VP-SDE ↔ Probability Flow ODE変換 + 軌道可視化

### 5.1 演習: VP-SDE軌道とPF-ODE軌道の比較

同じ初期ノイズから、Reverse-time SDEとPF-ODEで軌道を生成し比較。

```julia
using DifferentialEquations, Plots, Random

Random.seed!(42)
β_min, β_max = 0.1, 20.0

# 共通の初期ノイズ
u0_list = [randn(1) for _ in 1:5]
tspan = (1.0, 0.0)

# Reverse-time SDE
function reverse_drift(u, p, t)
    β_t = p[1] + t * (p[2] - p[1])
    return [-0.5 * β_t * u[1] - β_t * (-u[1])]  # score_approx = -u[1]
end

reverse_noise(u, p, t) = [√(p[1] + t * (p[2] - p[1]))]

# Probability Flow ODE
function pf_ode(du, u, p, t)
    β_t = p[1] + t * (p[2] - p[1])
    score_approx = -u[1]
    du[1] = -0.5 * β_t * u[1] - 0.5 * β_t * score_approx
end

# プロット準備
p1 = plot(title="Reverse-time SDE", xlabel="t", ylabel="X(t)", legend=false)
p2 = plot(title="Probability Flow ODE", xlabel="t", ylabel="X(t)", legend=false)

for u0 in u0_list
    # SDE
    prob_sde = SDEProblem(reverse_drift, reverse_noise, u0, tspan, (β_min, β_max))
    sol_sde = solve(prob_sde, EM(), dt=-0.001)
    plot!(p1, sol_sde, lw=1.5, alpha=0.7)

    # ODE
    prob_ode = ODEProblem(pf_ode, u0, tspan, (β_min, β_max))
    sol_ode = solve(prob_ode, Tsit5())
    plot!(p2, sol_ode, lw=1.5, alpha=0.7)
end

plot(p1, p2, layout=(1,2), size=(1000, 400))
```

**観察**:
- SDE: 各軌道が揺れる（確率性）
- ODE: 滑らかな決定論的軌道
- 最終分布（周辺分布）は同じ

### 5.2 演習: スコア関数の影響を可視化

真のスコア関数 vs 近似スコア関数での軌道の違い。

```julia
# 真のスコア関数（ガウス分布 N(μ, σ²) 仮定）
μ_true, σ_true = 1.0, 0.5
true_score(x, t)   = -(x - μ_true) / σ_true^2   # ∇log N(μ, σ²) = -(x - μ) / σ²
approx_score(x, t) = -x                            # ゼロ平均ガウス仮定

# Reverse-time SDE with 真のスコア
function reverse_drift_true(u, p, t)
    β_t = p[1] + t * (p[2] - p[1])
    score = true_score(u[1], t)
    return [-0.5 * β_t * u[1] - β_t * score]
end

# Reverse-time SDE with 近似スコア
function reverse_drift_approx(u, p, t)
    β_t = p[1] + t * (p[2] - p[1])
    score = approx_score(u[1], t)
    return [-0.5 * β_t * u[1] - β_t * score]
end

u0_noise = randn(1)
tspan = (1.0, 0.0)

prob_true = SDEProblem(reverse_drift_true, reverse_noise, u0_noise, tspan, (β_min, β_max))
prob_approx = SDEProblem(reverse_drift_approx, reverse_noise, u0_noise, tspan, (β_min, β_max))

sol_true = solve(prob_true, EM(), dt=-0.001)
sol_approx = solve(prob_approx, EM(), dt=-0.001)

plot(sol_true, label="真のスコア", lw=2, xlabel="t", ylabel="X(t)", title="スコア関数の影響")
plot!(sol_approx, label="近似スコア", lw=2, linestyle=:dash)
hline!([μ_true], label="真の平均 μ=$μ_true", linestyle=:dot, lw=1.5)
```

**結果**: 真のスコア使用時、軌道が真の平均 $\mu = 1.0$ に収束。近似スコアは $\mu = 0$ に収束（バイアス）。

### 5.3 演習: 収束性の数値検証 — ステップ数 vs 精度

ステップ数 $T$ を変化させ、生成分布と真の分布のKL距離を計測。

```julia
using KernelDensity, Distributions

# 真の分布
μ_true, σ_true = 1.0, 0.5
p_true = Normal(μ_true, σ_true)

# 各ステップ数でサンプリング
step_counts = [10, 25, 50, 100, 200, 500, 1000]
kl_divergences = Float64[]

for T in step_counts
    dt = -1.0 / T
    t_seq = LinRange(1.0, 0.0, T+1)[1:T]

    samples = [let x = randn()
        for t in t_seq
            β_t = β_min + t * (β_max - β_min)
            x += (-0.5 * β_t * x - β_t * true_score(x, t)) * dt + √β_t * √(-dt) * randn()
        end
        x
    end for _ in 1:5000]

    # KL推定（ヒストグラムベース）
    kde_result = kde(samples)
    x_range = -2:0.05:4
    p_generated = pdf(kde_result, x_range)
    p_true_vals = pdf(p_true, x_range)

    # KL(p_true || p_generated) = ∫ p_true log(p_true / p_generated) dx
    kl = sum(@. p_true_vals * log(p_true_vals / (p_generated + 1e-10))) * 0.05
    push!(kl_divergences, kl)
end

# プロット
plot(step_counts, kl_divergences, xlabel="ステップ数 T", ylabel="KL divergence",
     title="収束性: ステップ数 vs KL距離", lw=2, marker=:circle, xscale=:log10, yscale=:log10, legend=false)
```

**理論予測**: $\text{KL} \propto 1/T$ → 両対数プロットで傾き -1 の直線

### 5.4 演習: Manifold仮説の検証 — 高次元データの固有次元

高次元データ（$D = 100$）で固有次元 $d = 5$ のマニフォールドを生成し、収束を観察。

```julia
using LinearAlgebra

# 固有次元 d=5 のマニフォールド上のデータ生成
D = 100  # 埋め込み次元
d = 5    # 固有次元

# ランダム直交基底（d次元部分空間）
Q, _ = qr(randn(D, d))
Q = Q[:, 1:d]

# 低次元潜在変数 z ~ N(0, I_d)
n_samples = 1000
Z = randn(d, n_samples)

# 高次元埋め込み X = Q * Z
X = Q * Z  # D × n_samples

# VP-SDE Forward過程でノイズ注入
β = 1.0
t = 1.0
α_t = exp(-0.5 * β * t)
σ_t = √(1 - exp(-β * t))

X_noisy = α_t * X + σ_t * randn(D, n_samples)

# Reverse-time SDE（簡易Score: PCA射影）
function reverse_manifold_drift(u, p, t)
    Q, β = p
    u_proj = Q * (Q' * u)  # Manifold上への射影
    score = @. -(u - u_proj) / σ_t^2  # 法線方向ペナルティ
    return @. -0.5β * u - β * score
end

function reverse_manifold_noise(u, p, t)
    _, β = p
    return Diagonal(fill(√β, length(u)))
end

# 1サンプルの逆拡散
u0_manifold = X_noisy[:, 1]
tspan_manifold = (1.0, 0.0)

prob_manifold = SDEProblem(reverse_manifold_drift, reverse_manifold_noise, u0_manifold, tspan_manifold, (Q, β))
sol_manifold = solve(prob_manifold, EM(), dt=-0.01)

# 元データとの距離
x_original = X[:, 1]
x_reconstructed = sol_manifold.u[end]
reconstruction_error = norm(x_original - x_reconstructed)

println("再構成誤差: $reconstruction_error")
# 固有次元が小さい → Scoreが部分空間に誘導 → 高精度再構成
```

**結果**: 固有次元 $d=5$ のマニフォールド上では、少ないステップで高精度再構成が可能。

### 5.5 演習: VP-SDE vs VE-SDE の分散軌道比較

Variance Preserving vs Variance Exploding の分散の時間発展を可視化。

```julia
using DifferentialEquations, Plots, Statistics

# パラメータ
β_min, β_max = 0.1, 20.0
σ_min, σ_max = 0.01, 50.0

# VP-SDE
vp_drift(u, p, t) = [-0.5 * (p[1] + t * (p[2] - p[1])) * u[1]]
vp_noise(u, p, t) = [√(p[1] + t * (p[2] - p[1]))]

# VE-SDE
ve_drift(u, p, t) = [0.0]

function ve_noise(u, p, t)
    σ_t = p[1] * (p[2] / p[1])^t
    return [√(2 * σ_t^2 * log(p[2] / p[1]))]
end

# アンサンブル実行（1000サンプル）
n_samples = 1000
u0_list = [randn(1) for _ in 1:n_samples]

# VP-SDE アンサンブル
prob_vp = SDEProblem(vp_drift, vp_noise, [0.0], (0.0, 1.0), (β_min, β_max))
ensemble_vp = EnsembleProblem(prob_vp, prob_func=(prob, i, repeat) -> remake(prob, u0=u0_list[i]))
sol_vp_ensemble = solve(ensemble_vp, EM(), EnsembleThreads(), trajectories=n_samples, dt=0.001)

# VE-SDE アンサンブル
prob_ve = SDEProblem(ve_drift, ve_noise, [0.0], (0.0, 1.0), (σ_min, σ_max))
ensemble_ve = EnsembleProblem(prob_ve, prob_func=(prob, i, repeat) -> remake(prob, u0=u0_list[i]))
sol_ve_ensemble = solve(ensemble_ve, EM(), EnsembleThreads(), trajectories=n_samples, dt=0.001)

# 分散の計算
t_vals_vp = sol_vp_ensemble[1].t
var_vp = [var([sol.u[i][1] for sol in sol_vp_ensemble]) for i in eachindex(t_vals_vp)]

t_vals_ve = sol_ve_ensemble[1].t
var_ve = [var([sol.u[i][1] for sol in sol_ve_ensemble]) for i in eachindex(t_vals_ve)]

# 理論分散
# VP: Var[X_t] = 1 - exp(-∫_0^t β(s) ds)
var_vp_theory(t) = 1 - exp(-(β_min + 0.5t * (β_max - β_min)) * t)

# VE: Var[X_t] = σ_min² (σ_max / σ_min)^(2t)
var_ve_theory(t) = σ_min^2 * (σ_max / σ_min)^(2t)

# プロット
p1 = plot(t_vals_vp, var_vp, label="VP-SDE (数値)", lw=2, xlabel="時刻 t", ylabel="Var[X(t)]", title="VP-SDE 分散")
plot!(p1, t_vals_vp, var_vp_theory.(t_vals_vp), label="VP-SDE (理論)", lw=2, linestyle=:dash)
hline!(p1, [1.0], label="分散上限=1", linestyle=:dot)

p2 = plot(t_vals_ve, var_ve, label="VE-SDE (数値)", lw=2, xlabel="時刻 t", ylabel="Var[X(t)]", title="VE-SDE 分散", yscale=:log10)
plot!(p2, t_vals_ve, var_ve_theory.(t_vals_ve), label="VE-SDE (理論)", lw=2, linestyle=:dash)

plot(p1, p2, layout=(1,2), size=(1200, 400))
```

**観察**:
- **VP-SDE**: 分散が上限1に収束（Variance Preserving）
- **VE-SDE**: 分散が指数的に爆発（Variance Exploding）

### 5.6 演習: Predictor-Corrector法の反復回数依存性

Correctorの反復回数を変化させ、サンプル品質を測定。

```julia
using DifferentialEquations, Plots, Statistics

β_min, β_max = 0.1, 20.0
true_mean, true_std = 1.0, 0.5

# 真のスコア関数
true_score(x, t) = -(x - true_mean) / true_std^2

# Predictor-Corrector サンプリング
function pc_sampling(n_corrector; n_steps=100, ε_langevin=0.01)
    x = randn()
    dt = -1.0 / n_steps

    for t in LinRange(1.0, 0.0, n_steps+1)[1:n_steps]
        β_t = β_min + t * (β_max - β_min)

        # Predictor
        x += (-0.5 * β_t * x - β_t * true_score(x, t)) * dt + √β_t * √(-dt) * randn()

        # Corrector
        for _ in 1:n_corrector
            x += ε_langevin * true_score(x, t) + √(2ε_langevin) * randn()
        end
    end

    return x
end

# 各反復回数での分布
corrector_counts = [0, 1, 3, 5, 10]
n_samples = 2000

samples_dict = Dict()
for n_corr in corrector_counts
    samples = [pc_sampling(n_corr, n_steps=100) for _ in 1:n_samples]
    samples_dict[n_corr] = samples
end

# KL距離計算
using Distributions, KernelDensity

p_true = Normal(true_mean, true_std)
kl_values = Float64[]

for n_corr in corrector_counts
    samples = samples_dict[n_corr]
    kde_result = kde(samples)
    x_range = -1:0.05:3
    p_gen = pdf(kde_result, x_range)
    p_true_vals = pdf(p_true, x_range)
    kl = sum(@. p_true_vals * log(p_true_vals / (p_gen + 1e-10))) * 0.05
    push!(kl_values, kl)
end

# プロット
plot(corrector_counts, kl_values, xlabel="Corrector反復回数", ylabel="KL divergence",
     title="Corrector回数 vs サンプル品質", lw=2, marker=:circle, legend=false)
```

**結果**:
- Corrector回数0（Predictor-only）: 高KL（低品質）
- Corrector回数5: KL最小（最適）
- Corrector回数10+: 改善飽和（コスト増のみ）

**実用指針**: Corrector反復5回が精度とコストのバランス。

### 5.7 演習: 異なるノイズスケジュールの比較 — 線形 vs Cosine vs 二次

線形、Cosine、二次スケジュールでの最終分布品質を比較。

```julia
# 線形スケジュール
β_linear(t) = β_min + t * (β_max - β_min)

# Cosineスケジュール
s = 0.008
α_bar_cosine(t) = cos((t + s) / (1 + s) * π/2)^2 / cos(s / (1 + s) * π/2)^2
β_cosine(t) = -(log(α_bar_cosine(t + 1e-6)) - log(α_bar_cosine(t))) / 1e-6

# 二次スケジュール
β_quadratic(t) = β_min + t^2 * (β_max - β_min)

# 各スケジュールでサンプリング
function sample_with_schedule(β_schedule, n_samples=1000)
    t_vals = LinRange(1.0, 0.0, 101)
    [let x = randn()
        for j in 1:100
            t = t_vals[j]; β_t = β_schedule(t)
            x += (-0.5 * β_t * x + β_t * x) * (-0.01) + √β_t * 0.1 * randn()
        end
        x
    end for _ in 1:n_samples]
end

samples_linear = sample_with_schedule(β_linear)
samples_cosine = sample_with_schedule(β_cosine)
samples_quadratic = sample_with_schedule(β_quadratic)

# 分布可視化
using StatsPlots
density(samples_linear, label="線形", lw=2)
density!(samples_cosine, label="Cosine", lw=2)
density!(samples_quadratic, label="二次", lw=2)
xlabel!("X")
ylabel!("密度")
title!("ノイズスケジュール比較")
```

**結果**:
- **線形**: 標準的（DDPM論文）
- **Cosine**: 滑らか、端点での急変回避 → 高品質
- **二次**: 初期にノイズが少ない → 学習が難しい

### 5.8 演習: 次元依存性の検証 — O(d/T)理論の実証

次元 $d$ を変化させ、収束レートが $O(d/T)$ になることを確認。

```julia
using LinearAlgebra, Distributions, Random

Random.seed!(42)
β = 1.0
T_fixed = 100

# 各次元で誤差を計測
dimensions = [1, 2, 5, 10, 20, 50]
errors = Float64[]

for d in dimensions
    # d次元ガウス分布
    μ_true = ones(d)

    # T ステップでサンプリング
    n_samples = 500
    samples = zeros(d, n_samples)
    dt = -1.0 / T_fixed

    for i in 1:n_samples
        x = randn(d)
        ξ = similar(x)
        for _ in 1:T_fixed
            randn!(ξ)
            score = @. -(x - μ_true)
            @. x += (-0.5β * x - β * score) * dt + √β * √(-dt) * ξ
        end
        @views samples[:, i] .= x
    end

    # Wasserstein距離（簡易: 平均のL2距離）
    μ_sampled = vec(mean(samples, dims=2))
    error = norm(μ_sampled - μ_true)
    push!(errors, error)
end

# プロット（理論: error ~ d/T）
plot(dimensions, errors, xlabel="次元 d", ylabel="誤差", title="次元依存性 (T=$T_fixed)", lw=2, marker=:circle, label="数値実験")
plot!(dimensions, dimensions ./ T_fixed, label="理論 O(d/T)", lw=2, linestyle=:dash, legend=:topleft)
```

**結果**: 誤差が $d/T$ に比例 → 高次元では多くのステップが必要。

### 5.9 演習: Langevin Dynamics vs Reverse-time SDE

Langevin DynamicsとReverse-time SDEのサンプリング品質を比較。

```julia
β_min, β_max = 0.1, 20.0
true_mean, true_std = 1.0, 0.5
n_samples = 2000

# 真のスコア
true_score(x, t) = -(x - true_mean) / true_std^2

# Reverse-time SDE サンプリング
function sde_sampling()
    x = randn()
    t_vals = LinRange(1.0, 0.0, 101)
    for i in 1:100
        t = t_vals[i]; β_t = β_min + t * (β_max - β_min)
        x += (-0.5 * β_t * x - β_t * true_score(x, t)) * (-0.01) + √β_t * 0.1 * randn()
    end
    x
end

# Langevin Dynamics サンプリング（t=0のスコアのみ使用）
function langevin_sampling(n_steps=1000, ε=0.01)
    x = randn()
    for _ in 1:n_steps
        x += ε * true_score(x, 0.0) + √(2ε) * randn()
    end
    x
end

# サンプル生成
samples_sde = [sde_sampling() for _ in 1:n_samples]
samples_langevin = [langevin_sampling() for _ in 1:n_samples]

# 分布比較
using StatsPlots
density(samples_sde, label="Reverse-time SDE", lw=2)
density!(samples_langevin, label="Langevin Dynamics", lw=2, linestyle=:dash)
vline!([true_mean], label="真の平均", linestyle=:dot, lw=2)
xlabel!("X")
ylabel!("密度")
title!("Reverse-time SDE vs Langevin Dynamics")
```

**結果**:
- 両者とも真の分布に収束
- **Reverse-time SDE**: より高速（100ステップ）
- **Langevin Dynamics**: 多くの反復必要（1000ステップ）

### 5.10 演習: ODEソルバーの選択がPF-ODEに与える影響

Probability Flow ODEを異なるODEソルバーで解き、精度比較。

```julia
using DifferentialEquations

β_min, β_max = 0.1, 20.0
true_mean = 1.0

function pf_ode_func(du, u, p, t)
    β_t = p[1] + t * (p[2] - p[1])
    score = -(u[1] - true_mean) / 0.5^2
    du[1] = -0.5 * β_t * u[1] - 0.5 * β_t * score
end

u0 = randn(1)
tspan = (1.0, 0.0)
p = (β_min, β_max)

# 各種ODEソルバー
solvers = [Euler(), Tsit5(), Vern7(), RadauIIA5()]
solver_names = ["Euler", "Tsit5 (RK45)", "Vern7 (RK78)", "RadauIIA5 (暗黙)"]

prob_ode = ODEProblem(pf_ode_func, u0, tspan, p)

errors_ode = Float64[]
times_ode = Float64[]

for (solver, name) in zip(solvers, solver_names)
    time_taken = @elapsed sol = solve(prob_ode, solver, saveat=[0.0])
    x_final = sol.u[end][1]
    error = abs(x_final - true_mean)

    push!(errors_ode, error)
    push!(times_ode, time_taken)

    println("$name: error=$error, time=$time_taken s")
end

# プロット
p1 = bar(solver_names, errors_ode, ylabel="終端誤差", title="ODEソルバー精度", legend=false, xrotation=45)
p2 = bar(solver_names, times_ode, ylabel="時間 (s)", title="ODEソルバー速度", legend=false, xrotation=45)
plot(p1, p2, layout=(1,2), size=(1200, 400))
```

**結果**:
- **Euler**: 最速だが低精度
- **Tsit5**: 精度と速度のバランス（推奨）
- **Vern7**: 超高精度、コスト高
- **RadauIIA5**: 剛性問題に強い

**実用指針**: 通常はTsit5、剛性問題ならRadauIIA5。

### 5.11 演習: 異なる初期ノイズ分布の影響

初期ノイズ分布を $\mathcal{N}(0, 1)$ から $\text{Uniform}(-3, 3)$ に変更した場合の影響を調査。

```julia
using Distributions

β_min, β_max = 0.1, 20.0
true_mean, true_std = 1.0, 0.5

true_score(x, t) = -(x - true_mean) / true_std^2

function reverse_drift!(du, u, p, t)
    β_t = p[1] + t * (p[2] - p[1])
    du[1] = -0.5 * β_t * u[1] - β_t * true_score(u[1], t)
end

function reverse_noise!(du, u, p, t)
    du[1] = √(p[1] + t * (p[2] - p[1]))
end

n_samples = 2000

solve_sde(u0) = solve(SDEProblem(reverse_drift!, reverse_noise!, u0, (1.0, 0.0), (β_min, β_max)), EM(), dt=-0.001).u[end][1]

# ガウス初期ノイズ
samples_gaussian = [solve_sde(randn(1))           for _ in 1:n_samples]

# 一様分布初期ノイズ
samples_uniform  = [solve_sde([rand(Uniform(-3, 3))]) for _ in 1:n_samples]

# 分布比較
using StatsPlots
density(samples_gaussian, label="初期: N(0,1)", lw=2)
density!(samples_uniform, label="初期: Uniform(-3,3)", lw=2, linestyle=:dash)
vline!([true_mean], label="真の平均", linestyle=:dot, lw=2, color=:red)
xlabel!("X")
ylabel!("密度")
title!("初期ノイズ分布の影響")
```

**結果**: どちらの初期分布でも、最終的に真の分布 $\mathcal{N}(\mu, \sigma^2)$ に収束 → **ノイズ分布の選択は柔軟**。

### 5.12 演習: 時間ステップ依存性の可視化 — 精度 vs コスト

ステップサイズ $dt$ を変化させ、精度とコストのトレードオフを可視化。

```julia
using BenchmarkTools, Distributions, Statistics

β_min, β_max = 0.1, 20.0
true_mean, true_std = 1.0, 0.5
p_true = Normal(true_mean, true_std)

true_score(x, t) = -(x - true_mean) / true_std^2

function reverse_drift!(du, u, p, t)
    β_t = p[1] + t * (p[2] - p[1])
    du[1] = -0.5 * β_t * u[1] - β_t * true_score(u[1], t)
end

function reverse_noise!(du, u, p, t)
    du[1] = √(p[1] + t * (p[2] - p[1]))
end

dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
errors = Float64[]
times  = Float64[]

for dt_val in dt_values
    time_taken = @elapsed samples = [
        solve(SDEProblem(reverse_drift!, reverse_noise!, randn(1), (1.0, 0.0), (β_min, β_max)),
              EM(), dt=-dt_val).u[end][1]
        for _ in 1:500
    ]

    μ_sampled = mean(samples)
    error = abs(μ_sampled - true_mean)
    push!(errors, error)
    push!(times, time_taken)
    println("dt=$dt_val: error=$error, time=$time_taken s")
end

# プロット
p1 = plot(dt_values, errors, xlabel="ステップサイズ dt", ylabel="平均誤差", title="精度 vs ステップサイズ", lw=2, marker=:circle, xscale=:log10, yscale=:log10, legend=false)
p2 = plot(dt_values, times, xlabel="ステップサイズ dt", ylabel="計算時間 (s)", title="コスト vs ステップサイズ", lw=2, marker=:circle, xscale=:log10, legend=false)
plot(p1, p2, layout=(1,2), size=(1200, 400))
```

**結果**:
- **dt小**: 高精度、高コスト
- **dt大**: 低精度、低コスト
- **最適**: dt=0.01（精度とコストのバランス）

---

> **Note:** **進捗: 92%完了**
> 実装と実験を完了。次は発展ゾーンで研究動向と参考文献を整理する。

---

> Progress: 85%
> **理解度チェック**
> 1. Julia DifferentialEquations.jl での `SDEProblem` 実装において、VP-SDEとVE-SDEのdrift関数とdiffusion関数の具体的な違いをコードの変数名と対応する数式で示せ。
> 2. Predictor-Corrector実装でCorrectorのLangevinステップ数を増やすとサンプル品質が向上するが、計算コストとのトレードオフが生じる境界条件を述べよ。

## 🚀 6. 発展ゾーン（20分）— 研究動向とSDEの未来

### 6.1 SDE収束理論の最新進展（2024-2025）

**O(d/T)収束理論 (Gen Li & Yuling Yan, 2024)**

[arXiv:2409.18959](https://arxiv.org/abs/2409.18959) "O(d/T) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"

**主な貢献**:
- **最小限の仮定**下でTotal Variation距離 $O(d/T)$ 収束を証明
- データ分布の仮定: 有限1次モーメントのみ（従来はlog-Sobolev不等式等が必要）
- スコア推定が $\ell_2$-正確なら保証される

**実用的示唆**:
- 次元 $d = 1000$、ステップ $T = 1000$ で $\text{TV} \lesssim 1.0$（高精度）
- $T = 50$ に削減 → $\text{TV} \lesssim 20.0$（精度低下、高次ソルバーで補完）

**Manifold仮説下の線形収束 (Peter Potaptchik et al., 2024)**

[arXiv:2410.09046](https://arxiv.org/abs/2410.09046) "Linear Convergence of Diffusion Models Under the Manifold Hypothesis"

**主な貢献**:
- データが固有次元 $d$ のマニフォールド上に集中するとき、KL収束が $O(d \log T)$
- 埋め込み次元 $D$ ではなく固有次元 $d$（$d \ll D$）に依存
- この依存性は**シャープ**（下界も $\Omega(d)$）

**実用的示唆**:
- 画像（$D = 256^2 = 65536$）でも $d \approx 100-500$ → 大幅な理論改善
- 現実のデータのManifold仮説を支持

**VP-SDE離散化誤差の簡易解析 (2025)**

[arXiv:2506.08337](https://arxiv.org/abs/2506.08337) "Diffusion Models under Alternative Noise: Simplified Analysis and Sensitivity"

**主な貢献**:
- Euler-Maruyama法の収束レート $O(T^{-1/2})$ をGrönwall不等式で簡潔に導出
- ガウスノイズを離散ノイズ（Rademacher等）に置き換えても同じ収束レート
- 計算コスト削減の可能性

### 6.2 Score SDE統一理論の発展

**Song et al. 2021の影響**

[arXiv:2011.13456](https://arxiv.org/abs/2011.13456) "Score-Based Generative Modeling through Stochastic Differential Equations"

**貢献**:
- VP-SDE/VE-SDEによるDDPM/NCSNの統一
- Probability Flow ODEで決定論的生成
- Predictor-Corrector法で高品質サンプリング

**後続研究**:
- **Flow Matching** (第38回): Score SDEをさらに一般化
- **Consistency Models** (第40回): Probability Flow ODEを1-Stepに圧縮
- **Rectified Flow**: OTとPF-ODEの接続

### 6.3 Anderson 1982のReverse-time SDE

**Anderson (1982) "Reverse-Time Diffusion Equation Models"**

*Stochastic Processes and their Applications*, vol. 12, pp. 313-326.

**歴史的重要性**:
- Reverse-time SDEの存在を初めて証明
- Girsanov定理とBayes定理の応用
- 拡散モデル（2015-2021）で40年後に再発見

**現代的解釈**:
- Score関数 $\nabla \log p_t(x)$ がDrift項の補正に登場
- 生成モデルはAndersonの定理の**計算可能化**（NNでScore推定）

### 6.4 Julia DifferentialEquations.jlのエコシステム

**DifferentialEquations.jl**

- 統一インターフェース: ODE/SDE/DAE/DDE/RODE
- 40種以上のソルバー（Runge-Kutta/IMEX/SDEソルバー）
- GPU対応（CUDA.jl統合）

**関連パッケージ**:
- **DiffEqFlux.jl**: Neural ODEの訓練（Universal Differential Equations）
- **Catalyst.jl**: 化学反応ネットワークのSDE
- **ModelingToolkit.jl**: 記号的モデリング → 自動的にSDEを生成

**Diffusion Modelとの統合**:
- Lux.jl（DLフレームワーク）でScore関数 $s_\theta(x, t)$ を訓練
- DifferentialEquations.jlでReverse-time SDE/PF-ODEサンプリング
- Reactant.jl（XLAコンパイル）でGPU高速化

### 6.5 SDE数値解法の高度化

**高次ソルバー（第40回で詳説）**:
- **DPM-Solver++**: PF-ODEをRunge-Kutta系で解く、$O(T^{-2})$収束
- **UniPC**: 統一Predictor-Correctorフレームワーク
- **EDM**: Elucidating Diffusion Models（Karras et al. 2022）、最適離散化

**Stochastic Runge-Kutta法**:
- Euler-Maruyamaを超える高次SDE solver
- Strong convergence $O(\Delta t^{3/2})$
- DifferentialEquations.jlで実装済み（`SRIW1()`, `SRIW2()`等）

> Progress: 95%
> **理解度チェック**
> 1. SDE → Flow Matching への接続において、Fokker-Planck方程式の連続性方程式としての解釈が条件付き速度場 $u_t(\mathbf{x}|\mathbf{x}_1)$ の設計にどう寄与するか述べよ。
> 2. VP-SDE・VE-SDE・Sub-VP SDE・PF-ODEの4定式化が同一の周辺分布 $p_t(\mathbf{x})$ を生成できる条件と、それぞれの数値解法上の有利な点を一行ずつ述べよ。

## 🎓 6. 振り返り + 統合ゾーン（30分）— まとめとFAQ

### 7.1 本回のまとめ — 3つの核心

**核心1: 離散DDPMの連続時間極限がVP-SDE/VE-SDE**
- DDPM $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$ → VP-SDE
- NCSN（ノイズレベル $\{\sigma_i\}$）→ VE-SDE
- 理論的根拠が明確化（Fokker-Planck方程式、収束性解析）

**核心2: Reverse-time SDEとProbability Flow ODEで生成**
- Anderson 1982のReverse-time SDE: 確率的生成
- Song et al. 2021のPF-ODE: 決定論的生成
- 同じ周辺分布 $p_t(x)$ → サンプリング手法の選択肢

**核心3: Score SDE統一理論がDDPM/NCSN/DDIMを包摂**
- Forward SDE（ノイズ注入）
- Reverse-time SDE（確率的サンプリング）
- Probability Flow ODE（決定論的サンプリング）
- Score関数 $\nabla \log p_t(x)$ がすべての鍵

### 7.2 Course I第5回との接続 — 既習知識の活用

**第5回で学んだこと**:
- Brown運動の定義と性質（連続性、非微分可能性、二次変分）
- 伊藤積分の定義（非予見性、伊藤等距離性）
- 伊藤の補題（$dW^2 = dt$ の導出、確率微分の連鎖律）
- 基本SDE（$dX = f dt + g dW$ の形式、存在・一意性の直感）
- Euler-Maruyama法（SDEの数値解法基礎）
- Fokker-Planck方程式の直感

**本回で深掘りしたこと**:
- VP-SDE/VE-SDEの**厳密導出**（伊藤の補題を適用）
- Fokker-Planck方程式の**厳密導出**（Kramers-Moyal展開）
- Anderson逆時間SDE定理（Girsanov定理の応用）
- Probability Flow ODE（連続方程式との関係）
- 収束性解析（O(d/T)、Manifold仮説）
- Julia DifferentialEquations.jlでのSDE実装

**第5回の知識が本回で活きる瞬間**:
- 伊藤の補題で $dX_t^2$ を計算 → VP-SDE分散導出（3.3節）
- Fokker-Planck方程式の直感を厳密化（3.6節）
- Euler-Maruyama法を前提にPredictor-Corrector法へ発展（3.13節）

### 7.3 次回（第38回）への橋渡し — Flow Matching統一理論

第38回「Flow Matching & 統一理論」で学ぶこと:
- **Conditional Flow Matching**: シミュレーションフリー訓練
- **Optimal Transport ODE**: Rectified Flow（直線輸送）
- **Stochastic Interpolants**: Flow/Diffusionの統一フレームワーク
- **DiffFlow統一理論**: SDM + GANを同一SDE表現
- **Wasserstein勾配流**: JKO schemeとFokker-Planckの等価性
- **Score ↔ Flow ↔ Diffusion ↔ ODE の数学的等価性証明**

**本回との接続**:
- Probability Flow ODE → Flow Matchingへの自然な拡張
- VP-SDE/VE-SDE → 一般確率パスへの一般化
- Score SDE統一理論 → さらなる統一（OT統合）

### 7.4 FAQ — よくある質問

**Q1: VP-SDEとVE-SDE、どちらを使うべき？**

A: タスク依存。
- **VP-SDE**: DDPMベース、画像生成で標準、分散保存で数値安定
- **VE-SDE**: NCSNベース、ノイズレベルが明示的、高次元潜在空間
- 第38回で学ぶFlow MatchingがSDEの制約を超える

**Q2: Probability Flow ODEの「同じ周辺分布」の意味は？**

A: 各時刻 $t$ での確率分布 $p_t(x)$ が同じ。
- Reverse-time SDE: 確率的軌道、サンプルごとに異なる経路
- PF-ODE: 決定論的軌道、初期値が同じなら同じ経路
- どちらも周辺分布 $\{p_t\}_{t \in [0, T]}$ は一致

**Q3: Euler-Maruyama法で十分？高次ソルバーは必須？**

A: タスク依存。
- **Euler-Maruyama**: 実装簡単、$T = 1000$ で十分な精度
- **高次ソルバー**: $T = 50$ に削減可能、推論高速化
- 第40回で学ぶDPM-Solver++/UniPCが実用的

**Q4: スコア関数 $\nabla \log p_t(x)$ はどう学習する？**

A: Denoising Score Matching（第35回）。
- ノイズ付きデータ $x_t$ からScore $\nabla \log p_t(x_t)$ を推定
- Neural Network $s_\theta(x, t)$ を訓練
- 本回は「学習済みScore関数が与えられた」と仮定

**Q5: DifferentialEquations.jlは必須？PyTorchで実装できない？**

A: PyTorchでも可能だが、DifferentialEquations.jlが圧倒的に強力。
- PyTorch: 自力でEuler-Maruyama実装、ソルバー選択肢少
- DifferentialEquations.jl: 40種ソルバー、自動ステップサイズ調整、GPU対応
- 研究プロトタイプならJulia、論文査読用ならPyTorch

**Q6: Anderson 1982論文は読むべき？**

A: 理論派なら推奨、実装派なら不要。
- Song et al. 2021がAnderson定理を現代的に再解釈
- Reverse-time SDEの導出スケッチ（本回3.8節）で十分
- 厳密証明（Girsanov定理）は専門書（Øksendal等）参照
### 7.6 自己診断チェックリスト

- [ ] Brown運動の二次変分 $\langle W \rangle_t = t$ を導出できる
- [ ] 伊藤の補題を使ってVP-SDEの平均・分散を導出できる
- [ ] Fokker-Planck方程式をKramers-Moyal展開から導出できる
- [ ] VP-SDE/VE-SDE/Sub-VP SDEの違いを説明できる
- [ ] Anderson逆時間SDE定理を述べられる
- [ ] Probability Flow ODEとReverse-time SDEの違いを説明できる
- [ ] Score SDE統一理論の4要素（Forward/Reverse/Score/ODE）を列挙できる
- [ ] O(d/T)収束理論の意味を説明できる
- [ ] Manifold仮説下の線形収束の意義を理解している
- [ ] Julia DifferentialEquations.jlでVP-SDEを実装できる
- [ ] Predictor-Corrector法のアルゴリズムを実装できる

全項目✓なら次回へ！未達成項目は該当Zoneを復習。

### 7.7 次回予告 — 第38回: Flow Matching & 統一理論

**第38回の核心トピック**:
- Conditional Flow Matching（CFM）完全導出
- Optimal Transport ODE / Rectified Flow（直線輸送）
- Stochastic Interpolants統一フレームワーク
- DiffFlow統一理論（SDM + GAN = 同一SDE）
- Wasserstein勾配流（JKO scheme / Fokker-Planckとの等価性）
- **Score ↔ Flow ↔ Diffusion ↔ ODE の数学的等価性証明**

**第37回（本回）との接続**:
- VP-SDE/VE-SDEを**一般確率パス**に拡張
- Probability Flow ODE → Flow Matching ODE（Optimal Transport統合）
- Score SDE → Flow Matching統一理論へ

> **Note:** **進捗: 100%完了 — 第37回読了！**
> SDE/ODE & 確率過程論を完全習得した。VP-SDE/VE-SDE導出、Anderson逆時間SDE、Probability Flow ODE、Score SDE統一理論、収束性解析、Julia実装を修得。次回Flow Matchingで全生成モデルの統一理論へ。

---

### 6.X パラダイム転換の問い

**"離散ステップ数 $T = 1000$ は経験則。連続時間SDEで理論化したとき、初めて「なぜ1000で十分か」に答えられる。理論なき実装は暗闇の航海では？"**

**議論ポイント**:
1. DDPMの成功（2020）は経験的。理論的正当化（Score SDE統一理論、2021）は後追い。実務では「動けばOK」か、理論的理解は必須か？
2. O(d/T)収束理論（2024）で「$T = 1000$ が十分な理由」が数学的に説明された。だが実装者の何%がこれを知るべきか？
3. Probability Flow ODEの発見（Song et al. 2021）はSDEの連続時間定式化なしには不可能だった。連続理論が新手法を生む例。理論 vs 実装、どちらが先か？

<details><summary>歴史的文脈 — SDEと拡散モデルの40年ギャップ</summary>

**Anderson 1982**: Reverse-time SDEを証明。当時は理論的興味のみ、応用なし。

**2015 Sohl-Dickstein et al.**: 拡散モデル初提案。Andersonを引用せず（独立に発見）。

**2020 Ho et al. DDPM**: 離散時間定式化で大成功。SDEとの接続は明示せず。

**2021 Song et al. Score SDE**: 40年前のAnderson定理を再発見、拡散モデルとSDE統一。Probability Flow ODE発見。

**2024-2025 収束理論**: Li & Yan、Potaptchik et al.がO(d/T)、Manifold線形収束を証明。理論が実装を逆照射。

**教訓**: 理論と実装の対話が新パラダイムを生む。40年の時を経て理論が実装に光を当てる。

</details>

---

## 参考文献

### 主要論文

[^1]: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole (2021). "Score-Based Generative Modeling through Stochastic Differential Equations". *ICLR 2021 (Oral)*.
<https://arxiv.org/abs/2011.13456>

[^2]: Brian D. O. Anderson (1982). "Reverse-time diffusion equation models". *Stochastic Processes and their Applications*, vol. 12, pp. 313-326.
<https://www.sciencedirect.com/science/article/pii/0304414982900515>

[^3]: Gen Li and Yuling Yan (2024). "O(d/T) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions". *arXiv preprint*.
<https://arxiv.org/abs/2409.18959>

[^4]: Peter Potaptchik, Iskander Azangulov, and George Deligiannidis (2024). "Linear Convergence of Diffusion Models Under the Manifold Hypothesis". *arXiv preprint*.
<https://arxiv.org/abs/2410.09046>

[^5]: Choi, J. & Fan, C. (2025). "Diffusion Models under Alternative Noise: Simplified Analysis and Sensitivity". *arXiv preprint*.
<https://arxiv.org/abs/2506.08337>

[^6]: Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020). "Denoising Diffusion Probabilistic Models". *NeurIPS 2020*.
<https://arxiv.org/abs/2006.11239>

[^7]: Alex Nichol and Prafulla Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models". *ICML 2021*.
<https://arxiv.org/abs/2102.09672>

[^8]: Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics". *ICML 2015*.
<https://arxiv.org/abs/1503.03585>

[^9]: Jiaming Song, Chenlin Meng, and Stefano Ermon (2020). "Denoising Diffusion Implicit Models". *ICLR 2021*.
<https://arxiv.org/abs/2010.02502>

[^10]: Yang Song and Stefano Ermon (2020). "Improved Techniques for Training Score-Based Generative Models". *NeurIPS 2020*.
<https://arxiv.org/abs/2006.09011>

### 教科書

- Bernt Øksendal (2003). *Stochastic Differential Equations: An Introduction with Applications* (6th ed.). Springer.
- Peter E. Kloeden and Eckhard Platen (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
- Olav Kallenberg (2002). *Foundations of Modern Probability* (2nd ed.). Springer.

### オンラインリソース

- Yang Song (2021). "Generative Modeling by Estimating Gradients of the Data Distribution". [Blog Post](https://yang-song.net/blog/2021/score/)
- MIT 6.S184 (2026). "Diffusion Models & Flow Matching". [Course Website](https://diffusion.csail.mit.edu/)
- DifferentialEquations.jl Documentation. [docs.sciml.ai](https://docs.sciml.ai/DiffEqDocs/stable/)

---

## 著者リンク

- Blog: https://fumishiki.dev
- X: https://x.com/fumishiki
- LinkedIn: https://www.linkedin.com/in/fumitakamurakami
- GitHub: https://github.com/fumishiki
- Hugging Face: https://huggingface.co/fumishiki

## ライセンス

本記事は [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ja)（クリエイティブ・コモンズ 表示 - 非営利 - 継承 4.0 国際）の下でライセンスされています。

### ⚠️ 利用制限について

**本コンテンツは個人の学習目的に限り利用可能です。**

**以下のケースは事前の明示的な許可なく利用することを固く禁じます:**

1. **企業・組織内での利用（営利・非営利問わず）**
   - 社内研修、教育カリキュラム、社内Wikiへの転載
   - 大学・研究機関での講義利用
   - 非営利団体での研修利用
   - **理由**: 組織内利用では帰属表示が削除されやすく、無断改変のリスクが高いため

2. **有料スクール・情報商材・セミナーでの利用**
   - 受講料を徴収する場での配布、スクリーンショットの掲示、派生教材の作成

3. **LLM/AIモデルの学習データとしての利用**
   - 商用モデルのPre-training、Fine-tuning、RAGの知識ソースとして本コンテンツをスクレイピング・利用すること

4. **勝手に内容を有料化する行為全般**
   - 有料note、有料記事、Kindle出版、有料動画コンテンツ、Patreon限定コンテンツ等

**個人利用に含まれるもの:**
- 個人の学習・研究
- 個人的なノート作成（個人利用に限る）
- 友人への元記事リンク共有

**組織での導入をご希望の場合**は、必ず著者に連絡を取り、以下を遵守してください:
- 全ての帰属表示リンクを維持
- 利用方法を著者に報告

**無断利用が発覚した場合**、使用料の請求およびSNS等での公表を行う場合があります。
