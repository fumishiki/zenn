---
title: "第38回: Flow Matching & 生成モデル統一理論: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "🌀"
type: "tech"
topics: ["machinelearning", "deeplearning", "flowmatching", "julia", "diffusion"]
published: true
---

## Zone 4: 実装ゾーン — Julia Flow Matching実装

理論を手を動かして確かめよう。ここでは、**Conditional Flow Matching (CFM)**の完全な実装を通じて、理論の各要素が実コードにどう対応するかを学ぶ。

---

### 4.1 実装の全体像

実装する内容：

1. **Gaussian Probability Paths**（OT Path / VP Path）
2. **Conditional Vector Field** $\mathbf{u}_t(\mathbf{x}|\mathbf{x}_1)$
3. **CFM Loss**の訓練ループ
4. **ODE Sampling**（Euler法 / RK4法）
5. **2次元玩具データセット**での可視化

実装言語：**Julia 1.11**（Lux.jl + Optimisers.jl + DifferentialEquations.jl）

---

### 4.2 依存パッケージ

```julia:setup.jl
using Lux, Random, Optimisers, Zygote
using DifferentialEquations, Distributions
using Plots, StatsBase

# Set random seed
rng = Random.default_rng()
Random.seed!(rng, 42)
```

---

### 4.3 データセット生成

2次元の**2峰ガウス混合**をターゲット分布とする：

```julia:dataset.jl
"""
Target distribution: mixture of 2 Gaussians
    p_data(x) = 0.5*N([-2, 0], I) + 0.5*N([2, 0], I)
"""
function sample_target(n::Int; rng=Random.default_rng())
    d = 2  # dimension
    x = zeros(Float32, d, n)

    for i in 1:n
        # Randomly choose component
        if rand(rng) < 0.5
            # First mode: μ = [-2, 0]
            x[:, i] = randn(rng, Float32, d) .+ Float32[-2, 0]
        else
            # Second mode: μ = [2, 0]
            x[:, i] = randn(rng, Float32, d) .+ Float32[2, 0]
        end
    end

    return x
end

"""
Source distribution: standard Gaussian N(0, I)
"""
function sample_source(n::Int, d::Int=2; rng=Random.default_rng())
    return randn(rng, Float32, d, n)
end
```

---

### 4.4 Probability Path定義

前述の理論に基づき、**Optimal Transport Path**と**VP Path**を実装する。

```julia:paths.jl
"""
Gaussian Probability Path: μ_t(x₁|x₀) と Σ_t

Parameters:
  - path_type: :ot (Optimal Transport) or :vp (Variance Preserving)
"""
struct GaussianPath{T}
    path_type::Symbol  # :ot or :vp
    σ_min::T
end

# Default: OT path with minimal noise
GaussianPath() = GaussianPath{Float32}(:ot, 1f-5)

"""
Compute μ_t(x₁, x₀) and σ_t at time t
"""
function path_params(gp::GaussianPath{T}, t::T, x_1, x_0) where T
    if gp.path_type == :ot
        # Optimal Transport: μ_t = t*x₁ + (1-t)*x₀, σ_t = σ_min
        μ_t = t .* x_1 .+ (1 - t) .* x_0
        σ_t = gp.σ_min
    elseif gp.path_type == :vp
        # Variance Preserving: μ_t = t*x₁, σ_t = √(1 - t²)
        μ_t = t .* x_1
        σ_t = sqrt(1 - t^2)
    else
        error("Unknown path type: $(gp.path_type)")
    end

    return μ_t, σ_t
end

"""
Sample from conditional distribution q_t(x|x₁, x₀)
    x_t ~ N(μ_t, σ_t²I)
"""
function sample_conditional(gp::GaussianPath, t, x_1, x_0; rng=Random.default_rng())
    μ_t, σ_t = path_params(gp, t, x_1, x_0)
    d = size(x_1, 1)
    ε = randn(rng, Float32, d, size(x_1, 2))
    return μ_t .+ σ_t .* ε
end

"""
Compute conditional vector field u_t(x|x₁, x₀)
    u_t = ∂μ_t/∂t + (σ_t σ'_t / σ_t²)(x - μ_t)
"""
function conditional_vector_field(gp::GaussianPath{T}, t::T, x_t, x_1, x_0) where T
    μ_t, σ_t = path_params(gp, t, x_1, x_0)

    if gp.path_type == :ot
        # ∂μ_t/∂t = x₁ - x₀, σ'_t = 0
        u_t = x_1 .- x_0
    elseif gp.path_type == :vp
        # ∂μ_t/∂t = x₁, σ'_t = -t/√(1-t²)
        dμ_dt = x_1
        dσ_dt = -t / sqrt(1 - t^2 + 1f-8)
        u_t = dμ_dt .+ (dσ_dt / (σ_t + 1f-8)) .* (x_t .- μ_t)
    end

    return u_t
end
```

**重要なポイント**：
- OT Pathでは$\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$（定数！）
- VP Pathでは$\mathbf{u}_t$が$\mathbf{x}_t$に依存する

---

### 4.5 Vector Field Network

時刻$t$と位置$\mathbf{x}_t$から速度$\mathbf{v}_\theta(\mathbf{x}_t, t)$を予測するネットワーク。

```julia:network.jl
"""
Time-conditional MLP for vector field prediction
    v_θ(x_t, t): (d+1) → 128 → 128 → d
"""
function build_vector_field_net(d::Int=2)
    return Chain(
        Dense(d + 1 => 128, gelu),
        Dense(128 => 128, gelu),
        Dense(128 => d)
    )
end

"""
Forward pass with time conditioning
    Input: x_t (d × batch), t (batch,)
    Output: v_θ(x_t, t) (d × batch)
"""
function (model::Chain)(x_t::AbstractMatrix, t::AbstractVector, ps, st)
    # Concatenate x_t and t
    batch_size = size(x_t, 2)
    t_expand = reshape(t, 1, batch_size)  # (1 × batch)
    input = vcat(x_t, t_expand)           # (d+1 × batch)

    return model(input, ps, st)
end
```

---

### 4.6 CFM Loss実装

理論式（Zone 3.1）のLossを実装する：

```math
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\left\|\mathbf{v}_\theta(t, \mathbf{x}_t) - \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0)\right\|^2\right]
```

```julia:loss.jl
"""
Conditional Flow Matching Loss
"""
function cfm_loss(model, ps, st, path::GaussianPath, batch_size::Int; rng=Random.default_rng())
    # Sample time uniformly
    t = rand(rng, Float32, batch_size)

    # Sample x₀ ~ N(0, I) and x₁ ~ p_data
    x_0 = sample_source(batch_size; rng=rng)
    x_1 = sample_target(batch_size; rng=rng)

    # Sample x_t ~ q_t(x|x₁, x₀)
    x_t = sample_conditional(path, t, x_1, x_0; rng=rng)

    # Compute target vector field
    u_t = conditional_vector_field(path, t, x_t, x_1, x_0)

    # Model prediction
    v_pred, st = model(x_t, t, ps, st)

    # MSE loss
    loss = mean((v_pred .- u_t).^2)

    return loss, st
end
```

---

### 4.7 訓練ループ

```julia:train.jl
"""
Train Flow Matching model
"""
function train_flow_matching(;
    n_epochs=1000,
    batch_size=256,
    learning_rate=1f-3,
    path_type=:ot,
    rng=Random.default_rng()
)
    # Initialize model
    d = 2
    model = build_vector_field_net(d)
    ps, st = Lux.setup(rng, model)

    # Optimizer
    opt_state = Optimisers.setup(Adam(learning_rate), ps)

    # Path
    path = GaussianPath{Float32}(path_type, 1f-5)

    # Training loop
    losses = Float32[]

    for epoch in 1:n_epochs
        # Compute loss and gradient
        (loss, st), back = Zygote.pullback(ps) do p
            cfm_loss(model, p, st, path, batch_size; rng=rng)
        end

        # Update parameters
        grads = back((one(loss), nothing))[1]
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        push!(losses, loss)

        if epoch % 100 == 0
            @info "Epoch $epoch: Loss = $(loss)"
        end
    end

    return model, ps, st, losses
end
```

---

### 4.8 ODE Sampling

訓練後、ODEを解いてサンプル生成：

```math
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \mathbf{v}_\theta(\mathbf{x}_t, t), \quad \mathbf{x}_0 \sim \mathcal{N}(0, I)
```

```julia:sampling.jl
"""
Sample from learned flow via ODE solving
"""
function sample_flow(model, ps, st, n_samples::Int;
                     solver=Euler(), dt=0.01, rng=Random.default_rng())
    d = 2

    # Initial noise x₀ ~ N(0, I)
    x_0 = sample_source(n_samples; rng=rng)

    # Define ODE: dx/dt = v_θ(x, t)
    function ode_func!(dx, x, p, t)
        t_batch = fill(Float32(t), n_samples)
        v, _ = model(x, t_batch, ps, st)
        dx .= v
    end

    # Solve ODE from t=0 to t=1
    tspan = (0.0f0, 1.0f0)
    prob = ODEProblem(ode_func!, x_0, tspan)
    sol = solve(prob, solver; dt=dt, saveat=[1.0f0])

    # Return x₁ (final state)
    return sol.u[end]
end
```

**注**：
- `Euler()`: 1次精度（速い）
- `RK4()`: 4次精度（高精度）
- Rectified Flowでは1-stepで十分（$\Delta t = 1$）

---

### 4.9 可視化

```julia:visualize.jl
"""
Visualize training progress and generated samples
"""
function visualize_results(model, ps, st, losses; n_samples=1000)
    # Plot 1: Training loss curve
    p1 = plot(losses, xlabel="Epoch", ylabel="CFM Loss",
              label="", title="Training Loss", lw=2)

    # Plot 2: Generated samples vs真のデータ
    x_real = sample_target(n_samples)
    x_gen = sample_flow(model, ps, st, n_samples)

    p2 = scatter(x_real[1, :], x_real[2, :], label="Real Data",
                 alpha=0.5, ms=2, color=:blue)
    scatter!(p2, x_gen[1, :], x_gen[2, :], label="Generated",
             alpha=0.5, ms=2, color=:red)
    title!(p2, "Real vs Generated Samples")

    # Plot 3: Trajectory visualization (single sample)
    x_0_single = randn(Float32, 2, 1)

    function ode_trajectory!(dx, x, p, t)
        t_batch = [Float32(t)]
        v, _ = model(x, t_batch, ps, st)
        dx .= v
    end

    tspan = (0.0f0, 1.0f0)
    prob = ODEProblem(ode_trajectory!, x_0_single, tspan)
    sol = solve(prob, RK4(); dt=0.05, saveat=0.05)

    trajectory = hcat(sol.u...)
    p3 = plot(trajectory[1, :], trajectory[2, :],
              marker=:circle, label="Flow Trajectory", lw=2)
    scatter!(p3, [x_0_single[1]], [x_0_single[2]],
             label="x₀", ms=8, color=:green)
    scatter!(p3, [trajectory[1, end]], [trajectory[2, end]],
             label="x₁", ms=8, color=:red)
    title!(p3, "Single Sample Trajectory")

    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
end
```

---

### 4.10 実行例

```julia:main.jl
# Train OT-based CFM
model_ot, ps_ot, st_ot, losses_ot = train_flow_matching(
    n_epochs=1000,
    batch_size=256,
    learning_rate=1f-3,
    path_type=:ot
)

# Visualize
visualize_results(model_ot, ps_ot, st_ot, losses_ot)

# Train VP-based CFM for comparison
model_vp, ps_vp, st_vp, losses_vp = train_flow_matching(
    path_type=:vp
)
```

**期待される結果**：
- OT Pathの方が収束が速い（直線経路）
- VP Pathは若干迂回するが、安定性が高い
- どちらも真の分布を正確に再現

---

### 4.11 実装のポイント整理

| 理論要素 | 実装上の対応 |
|----------|--------------|
| $\mathbf{u}_t(\mathbf{x}\|\mathbf{x}_1, \mathbf{x}_0)$ | `conditional_vector_field()` |
| $\mu_t(\mathbf{x}_1, \mathbf{x}_0)$ | `path_params()` の `μ_t` |
| $q_t(\mathbf{x}\|\mathbf{x}_1, \mathbf{x}_0)$ | `sample_conditional()` |
| $\mathcal{L}_{\text{CFM}}$ | `cfm_loss()` のMSE |
| ODE Sampling | `sample_flow()` の `solve(ODEProblem)` |

:::message
**実装の核心**
CFMの実装は驚くほどシンプル。Diffusion Modelのような複雑なノイズスケジュール、多段階逆過程、score networkの工夫は一切不要。**直線経路（OT Path）+ MSE Loss + ODE Solver**だけで十分だ。
:::

---

## Zone 5: 実験ゾーン — 演習と検証

理論と実装を踏まえ、以下の演習を通じて理解を深めよう。

---

### 演習1: OT Path vs VP Pathの比較

**問題**：
Zone 4の実装で、`:ot`と`:vp`の両方を訓練し、以下を比較せよ：

1. **訓練速度**（同じlossに到達するepoch数）
2. **生成品質**（2-Wasserstein距離で定量評価）
3. **軌道の直線性**（始点→終点の直線からの平均偏差）

**ヒント**：
- Wasserstein距離：`using OptimalTransport; w2 = wasserstein(x_real, x_gen, 2)`
- 直線性：各時刻$t$での位置と直線$(1-t)\mathbf{x}_0 + t\mathbf{x}_1$の距離

**期待される観察**：
- OT Pathの方が訓練が速く、軌道も直線に近い
- VP Pathは初期段階で大きく迂回する

---

### 演習2: Rectified Flowの1-step生成

**問題**：
Rectified Flow（arXiv:2209.03003）は、OT Pathを**再学習**することで1-stepサンプリングを可能にする。次の手順で実装せよ：

**Step 1: 初期CFMの訓練**

```julia
model_1, ps_1, st_1, _ = train_flow_matching(path_type=:ot, n_epochs=1000)
```

**Step 2: 軌道の再サンプリング**

訓練済みモデルで$\mathbf{x}_0 \to \mathbf{x}_1$の軌道を生成し、新しいペア$(\mathbf{x}_0', \mathbf{x}_1')$を作る：

```julia
function resample_trajectories(model, ps, st, n_samples)
    x_0 = sample_source(n_samples)
    x_1 = sample_flow(model, ps, st, n_samples)  # ODE solve
    return x_0, x_1
end
```

**Step 3: 直線経路での再訓練**

新しいペア$(\mathbf{x}_0', \mathbf{x}_1')$に対し、**完全な直線**を目標とする：

```julia
function rectified_loss(model, ps, st, x_0, x_1, batch_size)
    idx = rand(1:size(x_0, 2), batch_size)
    t = rand(Float32, batch_size)

    x_t = t .* x_1[:, idx] .+ (1 .- t) .* x_0[:, idx]
    u_t = x_1[:, idx] .- x_0[:, idx]  # 常に直線方向

    v_pred, st = model(x_t, t, ps, st)
    loss = mean((v_pred .- u_t).^2)

    return loss, st
end
```

**Step 4: 1-step生成のテスト**

```julia
# Resample
x_0_new, x_1_new = resample_trajectories(model_1, ps_1, st_1, 10000)

# Re-train
model_2, ps_2, st_2, _ = train_with_rectified_loss(x_0_new, x_1_new)

# 1-step sampling (Euler with Δt=1)
x_0_test = sample_source(1000)
v_final, _ = model_2(x_0_test, ones(Float32, 1000), ps_2, st_2)
x_1_gen = x_0_test .+ v_final  # Single step!
```

**検証**：
- 1-step生成の品質が、初期モデルの50-step ODEに匹敵することを確認せよ

---

### 演習3: Score ↔ Flow等価性の数値検証

**問題**：
Zone 3.5の理論的等価性を数値的に検証せよ。

**Step 1: Diffusion Modelの訓練**

標準的なDDPMを訓練し、score function $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$を学習：

```julia
# Score network: ε_θ(x_t, t) ≈ -√(β_t) ∇log p_t(x_t)
function train_score_model(...)
    # DDPM training (Zone 3.5の式を使用)
end
```

**Step 2: Score → Flowの変換**

Probability Flow ODE (3.5.3の式) を使って、scoreから速度場を計算：

```julia
function score_to_flow(ε_θ, x_t, t, β_t)
    # v_t(x) = -1/2 β_t [x + ε_θ(x_t, t)]
    return -0.5 * β_t * (x_t .+ ε_θ(x_t, t))
end
```

**Step 3: 直接Flow Matchingとの比較**

CFMで訓練した速度場$\mathbf{v}_\theta$と、scoreから計算した速度場を比較：

```julia
# Sample test points
x_test = sample_target(100)
t_test = rand(Float32, 100) .* 0.9 .+ 0.05  # t ∈ [0.05, 0.95]

# CFM prediction
v_cfm, _ = model_cfm(x_test, t_test, ps_cfm, st_cfm)

# Score-based prediction
ε_pred, _ = model_score(x_test, t_test, ps_score, st_score)
v_score = score_to_flow(ε_pred, x_test, t_test, β(t_test))

# Compute correlation
correlation = cor(vec(v_cfm), vec(v_score))
println("Score ↔ Flow correlation: $correlation")
```

**期待される結果**：
- 相関係数が0.95以上（ほぼ一致）
- 生成サンプルの品質も同等

---

### 演習4: DiffFlowのハイブリッド訓練

**問題**：
Zone 3.6のDiffFlowを簡易実装し、$\lambda$の効果を調べよ。

**Discriminator追加**：

```julia
function build_discriminator(d::Int=2)
    return Chain(
        Dense(d + 1 => 64, gelu),
        Dense(64 => 64, gelu),
        Dense(64 => 1, sigmoid)
    )
end
```

**DiffFlow Loss**：

```julia
function diffflow_loss(model, disc, ps_m, ps_d, st_m, st_d, λ, batch_size)
    # CFM term
    loss_cfm, st_m = cfm_loss(model, ps_m, st_m, path, batch_size)

    # GAN term
    x_real = sample_target(batch_size)
    x_fake = sample_flow(model, ps_m, st_m, batch_size)

    d_real, st_d = disc(vcat(x_real, zeros(Float32, 1, batch_size)), ps_d, st_d)
    d_fake, st_d = disc(vcat(x_fake, ones(Float32, 1, batch_size)), ps_d, st_d)

    loss_d = -mean(log.(d_real .+ 1f-8) .+ log.(1 .- d_fake .+ 1f-8))
    loss_g = -mean(log.(d_fake .+ 1f-8))

    # Combined
    total_loss = loss_cfm + λ * loss_g

    return total_loss, loss_d, st_m, st_d
end
```

**実験**：
- $\lambda \in \{0, 0.01, 0.1, 1.0\}$で訓練
- 各設定でFID（または2-Wasserstein距離）を計算
- 訓練安定性（lossの分散）を比較

**仮説**：
- $\lambda=0$：最も安定だが、サンプリングが遅い
- $\lambda=0.1$：品質と速度のバランスが最良
- $\lambda=1.0$：不安定化（mode collapse発生の可能性）

---

### 演習5: Wasserstein勾配流の可視化

**問題**：
JKOスキーム（Zone 3.7.5）を用いて、2次元分布の勾配流を可視化せよ。

**設定**：
- 初期分布$p_0 = \mathcal{N}([3, 3], I)$
- 目標分布$p_{\text{data}} = 0.5\mathcal{N}([-2, 0], I) + 0.5\mathcal{N}([2, 0], I)$
- 目的関数$\mathcal{F}[p] = \mathrm{KL}(p \| p_{\text{data}})$

**実装**：

```julia
using OptimalTransport

function jko_step(p_current, p_target, τ)
    # Solve: min_p [KL(p||p_target) + W_2(p, p_current)²/(2τ)]
    # Use Sinkhorn algorithm for OT plan
    M = pairwise_distance(p_current, p_target)
    γ = sinkhorn(M, τ)

    # Update via transport plan
    p_next = apply_transport(p_current, γ)

    return p_next
end

# Iteration
p = sample_source(1000)
for k in 1:50
    p = jko_step(p, sample_target(1000), τ=0.1)
    # Visualize every 10 steps
end
```

**可視化**：
- 各ステップで分布のscatter plotをアニメーション化
- 軌道が「滑らかに」2峰ガウスに収束することを確認

---

### 実験のまとめ

| 演習 | 確認する理論 | 重要な観察 |
|------|--------------|------------|
| 演習1 | OT vs VP Path | OT = 直線 → 高効率 |
| 演習2 | Rectified Flow | 再訓練で1-step化可能 |
| 演習3 | Score ↔ Flow等価性 | 数値的にほぼ一致 |
| 演習4 | DiffFlow統一 | $\lambda$でDiffusion↔GAN連続変化 |
| 演習5 | Wasserstein勾配流 | JKO = 離散勾配降下 |

:::message
**実験の本質**
理論は美しいが、手を動かして初めて「なぜこれが革命的か」が腹落ちする。特に演習2のRectified Flowでは、**1-stepで高品質な画像が生成される瞬間**に立ち会える。これは、理論が実用に直結する稀有な例だ。
:::

---

## Zone 6: 振り返り + 統合ゾーン（30min）

Flow Matchingは急速に進化している。ここでは、2024-2025年の最新研究と、未解決の課題を紹介する。

---

### 6.1 Flow Map Matching (Boffi+ NeurIPS 2025)

**問題意識**：
従来のCFMは、各サンプル$(\mathbf{x}_0, \mathbf{x}_1)$ごとに**独立に**条件付き速度場$\mathbf{u}_t(\mathbf{x}|\mathbf{x}_1)$を計算する。しかし、これは次の非効率を生む：

- サンプル間の**共通構造**（例：顔画像の目の位置）を活用できない
- 高次元データで計算コストが増大

**Flow Map Matchingの提案**：

「条件付き速度場」ではなく、**輸送写像**（transport map）$\mathbf{T}_t: \mathbb{R}^d \to \mathbb{R}^d$を直接学習する。

```math
\mathbf{x}_t = \mathbf{T}_t(\mathbf{x}_0), \quad \mathbf{v}_t(\mathbf{x}_t) = \frac{\partial \mathbf{T}_t}{\partial t}(\mathbf{T}_t^{-1}(\mathbf{x}_t))
```

**利点**：
1. **Amortization**：一度$\mathbf{T}_t$を学習すれば、任意の$\mathbf{x}_0$に適用可能
2. **幾何学的制約**の統合（例：体積保存、曲率制約）
3. **逆写像**$\mathbf{T}_t^{-1}$も学習可能（双方向生成）

**実験結果**（ImageNet 64×64）：

| 手法 | FID ↓ | Sampling Steps | 訓練時間 |
|------|-------|----------------|----------|
| CFM | 2.31 | 50 | 100% |
| **Flow Map Matching** | **2.18** | **50** | **75%** |

---

### 6.2 Variational Rectified Flow (Guo+ 2025)

**問題**：
Rectified Flowの再訓練（reflow）は、軌道を直線に近づけるが、**理論的保証**がない。どの程度の再訓練で最適になるか？

**変分定式化**：

最適輸送写像を**変分問題**として定式化：

```math
\min_{\mathbf{T}} \mathbb{E}\left[\|\mathbf{T}(\mathbf{x}_0) - \mathbf{x}_1\|^2\right] + \lambda\,\mathrm{KL}(q_{\mathbf{T}} \| p_{\text{data}})
```

ここで：
- 第1項：輸送コスト（直線性）
- 第2項：分布一致性
- $\lambda$：正則化パラメータ

**理論的成果**：
- 再訓練の**収束レート**を導出：$O(1/\sqrt{K})$（$K$=再訓練回数）
- 最適$\lambda$の選択基準を提供

**実用的インパクト**：
- 再訓練を2-3回で打ち切る理論的根拠
- 計算コスト削減

---

### 6.3 Multitask Stochastic Interpolants (Negrel+ 2025)

**動機**：
画像生成では、複数の条件（テキスト、スタイル、解像度）を同時に扱いたい。

**提案**：
Stochastic Interpolants（Zone 3.4）を**マルチタスク学習**に拡張：

```math
\mathcal{L}_{\text{multi}} = \sum_{k=1}^K w_k\,\mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1^{(k)}}\left[\left\|\mathbf{v}_\theta^{(k)}(t, \mathbf{x}_t) - \mathbf{u}_t^{(k)}\right\|^2\right]
```

ここで：
- $k$：タスクインデックス（例：$k=1$はテキスト条件、$k=2$はスタイル条件）
- $w_k$：タスク重み
- $\mathbf{v}_\theta^{(k)}$：タスク固有の速度場

**技術的工夫**：
- **Adapter Modules**：共通バックボーン + タスク固有層
- **Task Balancing**：各タスクのlossを動的に調整（GradNormアルゴリズム）

**実験**：
- Text-to-ImageとStyle Transferを同時訓練
- 単一タスク訓練より**30%の計算削減**、品質は同等

---

### 6.4 Flow Matching for Discrete Domains

**課題**：
これまでのFlow Matchingは**連続空間**$\mathbb{R}^d$を仮定。しかし、テキスト、グラフ、分子などは**離散構造**を持つ。

**現在のアプローチ**：

1. **Embedding Space Flow**（Campbell+ 2024）
   - 離散トークンを連続embeddingに写像
   - Embedding空間でFlow Matching
   - デコード時に最近傍トークンに丸める

   **問題**：丸め誤差、embedding空間の非自然性

2. **Continuous-Time Markov Chain Flow**（Sun+ 2024）
   - 離散状態間の遷移確率をFlowとして定式化
   - Rate matrix $\mathbf{Q}_t$を学習

   ```math
   \frac{\partial p_t}{\partial t} = p_t \mathbf{Q}_t
   ```

   **問題**：状態空間が大きいと$\mathbf{Q}_t$の次元爆発

**未解決問題**：
- 離散Flowの**最適輸送理論**の確立
- 効率的なサンプリングアルゴリズム

---

### 6.5 High-Resolution Image GenerationへのScale

**現状**：
- CIFAR-10 (32×32)：FID ~2
- ImageNet 64×64：FID ~2.5
- **ImageNet 256×256**：FID ~5-7（Diffusionに劣る）

**ボトルネック**：

1. **Memory**：高解像度では速度場ネットワークが巨大化
2. **ODE Stiffness**：複雑なデータでODEが「硬い」（stiff）になり、数値誤差が蓄積

**研究方向**：

**a) Latent Flow Matching**（Dao+ 2024）
- VAEの潜在空間でFlow Matching
- Stable Diffusionと同様のアプローチ
- ImageNet 256×256でFID **3.2**達成

**b) Multi-Scale Flow**（Kim+ 2024）
- 低解像度→高解像度の段階的生成
- 各スケールで独立なFlow
- メモリ効率が大幅向上

**c) Adaptive Step Size ODE Solver**
- DiffEq.jlの`Tsit5()`など、適応的ソルバーを活用
- Stiffnessを自動検出してステップサイズ調整

---

### 6.6 未解決の理論的問題

**Problem 1: 非凸最適化の保証**

CFM Lossは非凸だが、実際には局所最適に陥らない。なぜか？

**予想**：
- Over-parameterization（ニューラルネットが過剰に大きい）
- Loss landscapeが「フラット」（implicit regularization）

**必要な理論**：Neural Tangent Kernel (NTK)解析、Mean Field理論

---

**Problem 2: 最適なProbability Pathの選択**

OT Path、VP Path、General Pathのうち、**データ依存で最適な経路**を自動選択できるか？

**アイデア**：
- Meta-learning：複数のpathで訓練し、validation lossで選択
- Adaptive Path：データ分布の幾何学的特性（曲率、位相）から経路を構築

---

**Problem 3: サンプリング複雑度の下界**

Rectified Flowは1-stepを主張するが、**理論的に必要な最小ステップ数**は？

**既知の結果**：
- Lipschitz連続な速度場では、$O(\epsilon^{-1})$ステップで$\epsilon$-近似（標準的ODE理論）

**Open Question**：
- データの「複雑さ」（例：モード数、次元）と必要ステップ数の関係
- 1-stepが可能な条件の特徴づけ

---

### 6.7 応用領域の拡大

Flow Matchingは画像生成を超えて広がっている：

**a) 分子設計**（Drug Discovery）
- タンパク質の3D構造生成（AlphaFold的応用）
- 化学的制約（結合長、角度）をFlowに組み込む

**b) 音声合成**
- WaveNetの代替としてのFlow-based TTS
- リアルタイム生成（低レイテンシ）

**c) 強化学習**
- 行動ポリシーの生成モデル化
- Flow Matching + Actor-Critic

**d) 気象予測**
- 時空間データの確率的予測
- Ensemble生成（複数の未来軌道）

---

### 6.8 最新論文リスト（2024-2025）

訓練効率とスケーラビリティに関する最新研究：

1. **Flow Map Matching**（Boffi+ 2024, arXiv:2406.07507）
   - 輸送写像の直接学習

2. **Variational Rectified Flow**（Guo+ 2025, arXiv:2502.09616）
   - 変分定式化と収束保証

3. **Multitask Stochastic Interpolants**（Negrel+ 2025, arXiv:2508.04605）
   - マルチタスク学習への拡張

4. **Meta AI Flow Matching Guide**（2024, arXiv:2412.06264）
   - 実装ベストプラクティス集

5. **Discrete Flow Matching**（Campbell+ 2024）
   - テキスト生成への応用

:::details さらに深掘りしたい読者へ
Flow Matchingのコミュニティは活発で、毎月新しい論文が登場する。以下のリソースが有用：

- **GitHub**: `atong01/conditional-flow-matching`（公式実装）
- **Papers with Code**: "Flow Matching"タグでフィルタ
- **Twitter**: #FlowMatching ハッシュタグ（研究者の議論）

特に、**ICLR 2025 Workshop on Flow-Based Models**では、未公開の最新研究が議論される。
:::

---

## Zone 7: 振り返りゾーン — 全体の統合とFAQ

ここまでの長い旅を振り返り、重要なポイントを整理しよう。

---

### 7.1 この講義で学んだこと

**核心的洞察**：

1. **生成モデルの統一理論**
   - Score Matching、Diffusion Models、Flow Matching、GANsは、すべて**最適輸送理論のWasserstein勾配流**として理解できる
   - 違いは「目的関数$\mathcal{F}$」と「離散化手法」だけ

2. **Conditional Flow Matching (CFM)の革新性**
   - **周辺化トリック**により、周辺速度場$\mathbf{v}_t$を学習せずに、条件付き速度場$\mathbf{u}_t(\mathbf{x}|\mathbf{x}_1)$だけで訓練可能
   - Simulation-free（SDEを解かずに訓練できる）

3. **Optimal Transport (OT) Pathの優位性**
   - 直線経路 → 最短距離 → 少ないステップで高品質生成
   - Rectified Flowで1-step生成も可能

4. **Stochastic Interpolantsの一般性**
   - FlowとDiffusionを統一する枠組み
   - 確率的揺らぎ$\sigma_t$の選択で連続的に移行

5. **DiffFlowの統一視点**
   - SDMとGANが**同一SDE**から導出される
   - $g(t)$（拡散係数）と$\lambda$（GAN項の重み）で連続的に制御

---

### 7.2 重要な数式の総まとめ

**CFM Loss**：
```math
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\left\|\mathbf{v}_\theta(t, \mathbf{x}_t) - \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0)\right\|^2\right]
```

**Gaussian Probability Path**（OT）：
```math
\mu_t(\mathbf{x}_1, \mathbf{x}_0) = t\mathbf{x}_1 + (1-t)\mathbf{x}_0, \quad \sigma_t = \sigma_{\min}
```

**条件付き速度場**（OT Path）：
```math
\mathbf{u}_t(\mathbf{x} | \mathbf{x}_1, \mathbf{x}_0) = \mathbf{x}_1 - \mathbf{x}_0
```

**Score ↔ Flow等価性**：
```math
\mathbf{v}_t(\mathbf{x}) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2\nabla_{\mathbf{x}}\log p_t(\mathbf{x})
```

**Wasserstein勾配流**：
```math
\mathbf{v}_t = -\nabla \frac{\delta \mathcal{F}}{\delta p}\bigg|_{p=p_t}
```

---

### 7.3 実装のチェックリスト

Flow Matchingを実装する際の必須要素：

- [ ] **Probability Path**の定義（`path_params()`）
- [ ] **条件付き速度場**の計算（`conditional_vector_field()`）
- [ ] **CFM Loss**の実装（MSE between $\mathbf{v}_\theta$ and $\mathbf{u}_t$）
- [ ] **時刻条件付きネットワーク**（入力に$t$を結合）
- [ ] **ODE Solver**（DifferentialEquations.jlなど）
- [ ] **可視化**（軌道、サンプル、loss curve）

---

### 7.4 よくある質問（FAQ）

**Q1: Flow MatchingとDiffusion Models、どちらを使うべき？**

**A**：
- **Flow Matching**：サンプリング速度が重要な場合（リアルタイム生成、1-step化）
- **Diffusion Models**：既存の大規模実装（Stable Diffusion）を活用したい場合
- **両者のハイブリッド**（DiffFlow）：最高品質を追求する場合

**現時点の推奨**：新規プロジェクトなら**Flow Matching**。理由：
- シンプルな実装
- 高速サンプリング
- 理論的に洗練されている

---

**Q2: なぜOT Pathが最適なのか？**

**A**：
最適輸送理論により、$p_0$から$p_1$への「最短経路」がOT Pathであることが保証される。数学的には：

```math
W_2(p_0, p_1)^2 = \inf_{\pi} \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim \pi}\left[\|\mathbf{x}_1 - \mathbf{x}_0\|^2\right]
```

この最適解が直線経路$\mu_t = t\mathbf{x}_1 + (1-t)\mathbf{x}_0$を与える（Gaussianの場合）。

---

**Q3: Rectified Flowの再訓練は本当に必要？**

**A**：
**データ依存**。簡単な分布（MNIST、2D toy data）では初回訓練でほぼ直線。複雑な分布（ImageNet）では1-2回の再訓練で大幅改善。

**判断基準**：
- 軌道の直線性を測定（平均偏差）
- 1-step生成の品質をチェック
- 改善が見られなくなったら終了

---

**Q4: 高次元データ（例：1024×1024画像）でもFlow Matchingは有効？**

**A**：
**Latent Space Flow Matching**が有効。手順：

1. VAEで画像を低次元潜在空間に圧縮（例：1024×1024 → 64×64×4）
2. 潜在空間でFlow Matching訓練
3. デコーダで画像に戻す

Stable Diffusionと同じアプローチ。Meta AIのFlow Matching Guide（arXiv:2412.06264）に詳細あり。

---

**Q5: 実装で最もハマりやすいバグは？**

**A**：
**Top 3**：

1. **時刻$t$の範囲ミス**
   - 訓練では$t \in (0, 1)$だが、サンプリングでは$t=0$と$t=1$の境界も必要
   - 解決：`t = rand() * 0.98 + 0.01`で訓練、サンプリングは`t ∈ [0, 1]`

2. **ベクトル場の符号ミス**
   - $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$を$\mathbf{x}_0 - \mathbf{x}_1$と書いてしまう
   - 解決：Zone 1のインタラクティブ例で可視化して確認

3. **ODEの数値誤差**
   - Euler法でステップサイズが大きすぎる
   - 解決：RK4法を使う、またはステップサイズを半分に

---

**Q6: Wasserstein勾配流の理解は必須？**

**A**：
**実装には不要、理論の深い理解には必須**。

- 実装者：Zone 4のコードだけ読めばOK
- 研究者：Zone 3.7を熟読し、Jordan+ (1998) の原論文へ
- 数学的背景：測度論、変分法、PDE

---

### 7.5 次のステップ

**Level 1（初学者）**：
- [ ] Zone 4の実装を完全に再現
- [ ] 演習1-3を解く
- [ ] 2D toy datasetで可視化

**Level 2（中級者）**：
- [ ] MNIST/CIFAR-10でFlow Matching訓練
- [ ] Rectified Flow実装
- [ ] 演習4-5に挑戦

**Level 3（上級者）**：
- [ ] Latent Flow Matching実装（VAE統合）
- [ ] 最新論文（Zone 6.8）を実装
- [ ] 独自の応用領域で実験（音声、分子など）

**Level 4（研究者）**：
- [ ] 未解決問題（Zone 6.6）に取り組む
- [ ] 新しいProbability Pathを提案
- [ ] ICLR/NeurIPSに投稿

---

### 7.6 リソース集

**公式実装**：
- `atong01/conditional-flow-matching`（PyTorch、reference実装）
- `FluxML/Flux.jl`（Julia、本講義のベース）

**論文**：
- Flow Matching原論文（Lipman+ ICLR 2023, arXiv:2210.02747）
- Stochastic Interpolants（Albergo+ 2023, arXiv:2303.08797）
- DiffFlow（Zhang+ 2023, arXiv:2307.02159）

**チュートリアル**：
- Meta AI Flow Matching Guide（arXiv:2412.06264）
- Hugging Face Diffusers（Flow Matching実装例）

**数学的背景**：
- Optimal Transport（Villani, "Topics in Optimal Transportation"）
- Wasserstein Gradient Flow（Jordan+ "The Variational Formulation of the Fokker-Planck Equation", 1998）

---

## Paradigm-Breaking Question: 生成モデルの「次」は何か？

ここまでの講義で、我々は生成モデルの統一理論に到達した。Score Matching、Diffusion、Flow、GANは、すべて**Wasserstein勾配流**という同じ山の異なる登山ルートだ。

しかし、問いは残る：

> **「この統一理論の先に、さらなるパラダイムシフトはあるのか？」**

---

### 現在の限界

どれほど洗練されても、現在の生成モデルは本質的に**データの模倣**だ：

- 訓練データ$p_{\text{data}}$を近似する分布$p_\theta$を学習
- 新しい「創造」ではなく、「既存データの補間」

**具体例**：
- Stable Diffusionは、訓練データにない完全に新しい概念（例：「量子もつれを可視化した抽象画」）を生成できない
- Flow Matchingも、$p_0$から$p_{\text{data}}$への最適経路を学ぶだけ

---

### 次のパラダイムへの示唆

**方向1: 因果生成モデル**

現在のモデルは**相関**を学ぶが、**因果関係**は学ばない。

**必要な要素**：
- 構造因果モデル（SCM）とFlowの統合
- 介入（intervention）と反事実（counterfactual）の生成

**想像される応用**：
- 「この薬を投与しなかったら、どうなっていたか？」の画像生成
- 因果的に整合した未来予測

---

**方向2: アクティブ生成（Active Generation）**

現在のモデルは**受動的**（プロンプトに反応するだけ）。

**次世代**：
- 生成モデル自身が「次に何を生成すべきか」を能動的に決定
- 強化学習との深い統合（reward-conditioned flow）

**例**：
- ユーザーの意図を予測して、先回りで画像を提案
- 対話的な創造（AI: 「この色をもっと鮮やかにしますか？」）

---

**方向3: 物理法則埋め込み生成**

画像生成は自由すぎる（物理的にあり得ない画像も生成）。

**制約付き生成**：
- Navier-Stokes方程式を満たす流体シミュレーション画像
- 熱力学第二法則を満たすプロセス動画
- Flow MatchingのPathに**微分方程式制約**を埋め込む

**技術**：
- Physics-Informed Neural Networks (PINN) + Flow Matching
- Symplectic Flow（ハミルトン力学保存）

---

**方向4: 意味的連続性の探求**

OT Pathは「座標空間」で直線だが、「意味空間」では？

**問い**：
- 「猫」から「犬」への最適な変形経路は、座標の線形補間か？
- むしろ「猫 → ネコ科 → 動物 → イヌ科 → 犬」のような**概念階層**を辿るべきでは？

**研究**：
- 意味的距離（semantic distance）の定義
- 概念グラフ上のFlow

---

### あなたへの問い

このコースを修了したあなたに、最後の問いを投げかけたい：

**「Flow Matchingの次に来る、あなた自身の生成モデルは何か？」**

- それは、因果を扱うか？
- 物理法則を尊重するか？
- 意味的な構造を持つか？
- それとも、まったく別の原理に基づくか？

理論は道具だ。**真の創造は、道具を超えたところにある**。

---

**Congratulations!** 🎉

あなたは、生成モデルの最前線に到達した。ここから先は、あなた自身が道を切り拓く番だ。
---

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
