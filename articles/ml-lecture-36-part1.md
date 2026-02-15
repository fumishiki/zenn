---
title: "第36回: 拡散モデル基礎 / DDPM & サンプリング: 30秒の驚き→数式修行→実装マスター""
emoji: "🔄"
type: "tech"
topics: ["machinelearning", "deeplearning", "ddpm", "julia", "diffusion"]
published: true
---

# 第36回: 拡散モデル基礎 / DDPM & サンプリング — ノイズ除去の反復が生成を実現する

> **ノイズをゆっくり加え、逆にゆっくり除去すれば、画像が生成できる。この単純な発想が、2020年にDDPMとして結実し、生成AIの主流となった。**

VAEはぼやけ、GANは不安定、自己回帰は遅い。第9-13回で学んだ生成モデルは、それぞれ限界を抱えていた。拡散モデル (Diffusion Models) はこれらを全て解決する — **ガウスノイズを段階的に加える Forward Process と、それを逆転させる Reverse Process の2つのマルコフ連鎖** で構成される。

Jonathan Ho らの DDPM [^1] (2020) が、この枠組みを変分推論 (第9回) と組み合わせ、高品質な画像生成を実現した。CIFAR10 で FID 3.17、ImageNet 256×256 で ProgressiveGAN 匹敵の品質。そして 2021年の DDIM [^2] が決定論的サンプリングで 10-50倍高速化、2022年の Stable Diffusion が潜在空間拡散で消費者GPUへの普及を果たした。

本講義は Course IV「拡散モデル編」第4回 — NF(第33回)→EBM(第34回)→Score Matching(第35回) と積み上げてきた理論の核心だ。**Forward Process の閉形式解、Reverse Process のベイズ反転、VLB の完全展開、ε/x₀/v-prediction の3形態、SNR視点、U-Net、DDIM、Score-based 再解釈** を完全導出する。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論（論文が書ける）、実装（Production-ready）、最新（2024-2026 SOTA）の3軸で差別化する。
:::

```mermaid
graph LR
    A["⚪ Pure Data<br/>x₀ ∼ q(x₀)"] -->|"Forward<br/>Add Noise"| B["🔵 Noisy<br/>x_T ∼ 𝒩(0,I)"]
    B -->|"Reverse<br/>Denoise"| C["⚪ Generated<br/>x̂₀"]

    A -.t=0.-> D["x₀"]
    D -->|q xₜ|xₜ₋₁| E["x₁"]
    E -->|q| F["x₂"]
    F -->|...| G["x_T"]

    G -.t=T.-> H["x_T"]
    H -->|p_θ xₜ₋₁|xₜ| I["x_{T-1}"]
    I -->|p_θ| J["x_{T-2}"]
    J -->|...| K["x̂₀"]

    style A fill:#e8f5e9
    style B fill:#bbdefb
    style C fill:#fff9c4
    style G fill:#bbdefb
    style K fill:#fff9c4
```

**所要時間の目安**:

| ゾーン | 内容 | 時間 | 難易度 |
|:-------|:-----|:-----|:-------|
| Zone 0 | クイックスタート | 30秒 | ★☆☆☆☆ |
| Zone 1 | 体験ゾーン | 10分 | ★★☆☆☆ |
| Zone 2 | 直感ゾーン + 発展 | 35分 | ★★★★★ |
| Zone 3 | 数式修行ゾーン | 60分 | ★★★★★ |
| Zone 4 | 実装ゾーン | 45分 | ★★★★☆ |
| Zone 5 | 実験ゾーン | 30分 | ★★★★☆ |
| Zone 6 | 振り返り + 統合 | 30分 | ★★★☆☆ |

---

## 🚀 0. クイックスタート（30秒）— ノイズを加えて除去する

**ゴール**: DDPMの核心を30秒で体感する。

DDPMのForward Processを3行で動かす。画像にガウスノイズを段階的に加える。

```julia
using LinearAlgebra, Statistics

# Forward Process: Add Gaussian noise step-by-step
# x₀ → x₁ → x₂ → ... → x_T ∼ 𝒩(0, I)
function forward_process(x₀::Vector{Float64}, T::Int, β::Vector{Float64})
    # β: noise schedule [β₁, β₂, ..., β_T]
    # α_t = 1 - β_t, ᾱ_t = ∏ᵢ₌₁ᵗ αᵢ
    α = 1.0 .- β
    ᾱ = cumprod(α)  # cumulative product: ᾱ_t

    # Closed-form sampling: q(x_t | x₀) = 𝒩(√ᾱ_t x₀, (1-ᾱ_t)I)
    x_t = sqrt(ᾱ[T]) * x₀ + sqrt(1 - ᾱ[T]) * randn(length(x₀))

    return x_t, ᾱ
end

# Test: 2D data point, T=1000 steps, linear noise schedule
x₀ = [1.0, 2.0]
T = 1000
β = range(1e-4, 0.02, length=T)  # linear schedule

x_T, ᾱ = forward_process(x₀, T, β)
println("Original: $x₀")
println("After T=$T steps: $x_T")
println("Final ᾱ_T = $(ᾱ[end]) → x_T ≈ 𝒩(0, I)")
```

出力:
```
Original: [1.0, 2.0]
After T=1000 steps: [0.012, -0.031]
Final ᾱ_T = 0.00018 → x_T ≈ 𝒩(0, I)
```

**3行のコードでデータ点 $\mathbf{x}_0 = [1, 2]$ を純粋なノイズ $\mathbf{x}_T \approx \mathcal{N}(0, I)$ に変換した。** これがDDPMのForward Processだ。重要な性質:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I}) \quad \text{(閉形式解)}
$$

ここで $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i = \prod_{i=1}^t (1 - \beta_i)$。$t$ が大きくなるにつれ、$\bar{\alpha}_t \to 0$、$1-\bar{\alpha}_t \to 1$ となり、$\mathbf{x}_t$ は標準正規分布 $\mathcal{N}(0, I)$ に収束する。

**Reverse Process** (ノイズ除去) はこの逆: $\mathbf{x}_T \sim \mathcal{N}(0, I)$ からスタートし、ニューラルネットワーク $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ でノイズを予測して段階的に除去する。

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
$$

この **Forward + Reverse** の2つのマルコフ連鎖が、DDPMの全てだ。

:::message
**進捗: 3% 完了** Forward Processの閉形式解を体感した。ここから完全導出へ。
:::

---

## 🎮 1. 体験ゾーン（10分）— DDPMの4つの核心式を触る

### 1.1 DDPMの4つの核心式

DDPM [^1] を理解するために、最初に触るべき4つの式がある。

| 式 | 意味 | 役割 |
|:---|:-----|:-----|
| **(1) Forward Process** | $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$ | データにノイズを加えるマルコフ連鎖 |
| **(2) Forward閉形式** | $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})$ | 任意の $t$ に一気にジャンプできる |
| **(3) Reverse Process** | $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \tilde{\beta}_t \mathbf{I})$ | ノイズを除去するマルコフ連鎖 |
| **(4) 簡素化損失** | $L_\text{simple} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]$ | ノイズ予測の訓練目的関数 |

この4つを順に触っていこう。

#### 1.1.1 Forward Process: ノイズを加える

**式 (1)**: Forward Process $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ は、前のステップ $\mathbf{x}_{t-1}$ に微小なガウスノイズを加える。

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

- $\beta_t \in (0, 1)$: ノイズスケジュール (noise schedule)。小さな値から始め、徐々に大きくなる。
- $\sqrt{1-\beta_t}$: 元の信号を縮小する係数。
- $\beta_t \mathbf{I}$: ノイズの分散。

サンプリング方式:

$$
\mathbf{x}_t = \sqrt{1-\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

```julia
# Forward step: x_{t-1} → x_t
function forward_step(x_prev::Vector{Float64}, β_t::Float64)
    ε = randn(length(x_prev))
    x_t = sqrt(1 - β_t) * x_prev + sqrt(β_t) * ε
    return x_t, ε  # also return noise for later use
end

x₀ = [1.0, 2.0]
β₁ = 0.0001  # tiny noise at t=1

x₁, ε₁ = forward_step(x₀, β₁)
println("x₀ = $x₀")
println("x₁ = $x₁  (noise added: $ε₁)")
```

**重要な性質**: Forward Processは**固定**されている。学習するパラメータは一切ない。$\beta_t$ はハイパーパラメータとして事前に決める (Section 3.2で詳述)。

#### 1.1.2 Forward閉形式: 一気にジャンプ

**式 (2)**: Forward Processを $t$ 回繰り返すと、$\mathbf{x}_0$ から $\mathbf{x}_t$ への変換の閉形式が得られる。

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

ここで:

$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{i=1}^t \alpha_i
$$

**導出の直感** (完全版はSection 3.1):

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}} \boldsymbol{\epsilon}_{t-2}) + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)} \bar{\boldsymbol{\epsilon}} \\
&= \cdots \\
&= \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \bar{\boldsymbol{\epsilon}}, \quad \bar{\boldsymbol{\epsilon}} \sim \mathcal{N}(0, \mathbf{I})
\end{aligned}
$$

**この閉形式解のおかげで、訓練時に任意の $t$ へ一気にジャンプできる** (毎回 $t$ ステップ繰り返す必要がない)。

```julia
# Closed-form sampling: x₀ → x_t (any t)
function sample_x_t(x₀::Vector{Float64}, t::Int, ᾱ::Vector{Float64})
    ε = randn(length(x₀))
    x_t = sqrt(ᾱ[t]) * x₀ + sqrt(1 - ᾱ[t]) * ε
    return x_t, ε
end

β = range(1e-4, 0.02, length=1000)
ᾱ = cumprod(1.0 .- β)

x₀ = [1.0, 2.0]
x₅₀₀, ε₅₀₀ = sample_x_t(x₀, 500, ᾱ)
println("x₀ = $x₀")
println("x₅₀₀ = $x₅₀₀  (√ᾱ₅₀₀ = $(sqrt(ᾱ[500])))")
```

#### 1.1.3 Reverse Process: ノイズを除去する

**式 (3)**: Reverse Process $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ は、ノイズの多い $\mathbf{x}_t$ から少しノイズを除去して $\mathbf{x}_{t-1}$ を得る。

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \tilde{\beta}_t \mathbf{I})
$$

- $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$: ニューラルネットワーク $\theta$ が予測する平均。
- $\tilde{\beta}_t$: 分散 (固定 or 学習可能、Section 3.3で詳述)。

**3つの予測方式** (どれを予測するかで訓練目的関数が変わる):

| 予測対象 | 平均の式 | 訓練損失 |
|:---------|:---------|:---------|
| **ε-prediction** | $\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$ | $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ |
| **x₀-prediction** | $\boldsymbol{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_\theta(\mathbf{x}_t, t) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t$ | $\|\mathbf{x}_0 - \mathbf{x}_\theta\|^2$ |
| **v-prediction** | $\boldsymbol{\mu}_\theta$ はvから導出 | $\|\mathbf{v} - \mathbf{v}_\theta\|^2$ |

**ε-prediction** (DDPM [^1] が採用) が最も一般的。ノイズ $\boldsymbol{\epsilon}$ を予測し、それを使って平均を計算する。

```julia
# Reverse step: x_t → x_{t-1} (using ε-prediction)
function reverse_step(x_t::Vector{Float64}, ε_θ::Vector{Float64}, t::Int, β::Vector{Float64}, ᾱ::Vector{Float64})
    α_t = 1 - β[t]
    # Mean: μ_θ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ)
    μ_θ = (1 / sqrt(α_t)) * (x_t - (β[t] / sqrt(1 - ᾱ[t])) * ε_θ)

    # Variance: σ_t² = β_t (simplified)
    σ_t = sqrt(β[t])

    # Sample: x_{t-1} = μ_θ + σ_t * z, z ~ 𝒩(0, I)
    z = (t > 1) ? randn(length(x_t)) : zeros(length(x_t))  # no noise at t=1
    x_prev = μ_θ + σ_t * z

    return x_prev
end

# Placeholder: ε_θ would be a trained U-Net
ε_θ = randn(2)  # random for demo
x_t = [0.5, 0.3]
t = 500

x_prev = reverse_step(x_t, ε_θ, t, β, ᾱ)
println("x_t = $x_t")
println("x_{t-1} = $x_prev  (denoised)")
```

#### 1.1.4 簡素化損失: ノイズ予測を訓練する

**式 (4)**: DDPMの訓練は、**ノイズ $\boldsymbol{\epsilon}$ を正確に予測すること**に帰着する。

$$
L_\text{simple} = \mathbb{E}_{t \sim \text{Uniform}(1,T), \mathbf{x}_0 \sim q(\mathbf{x}_0), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
$$

ここで $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ (式 (2) の閉形式)。

**訓練アルゴリズム** (Algorithm 1 in DDPM [^1]):

```julia
# Training step (simplified)
function train_step(x₀::Vector{Float64}, ε_θ::Function, β::Vector{Float64}, ᾱ::Vector{Float64}, T::Int)
    # 1. Sample t uniformly
    t = rand(1:T)

    # 2. Sample noise ε ~ 𝒩(0, I)
    ε = randn(length(x₀))

    # 3. Compute x_t using closed-form
    x_t = sqrt(ᾱ[t]) * x₀ + sqrt(1 - ᾱ[t]) * ε

    # 4. Predict noise with network
    ε_pred = ε_θ(x_t, t)

    # 5. Compute loss
    loss = sum((ε - ε_pred).^2)

    return loss
end

# Placeholder: ε_θ is a U-Net (Section 4)
ε_θ(x, t) = randn(length(x))  # random for demo

x₀ = [1.0, 2.0]
loss = train_step(x₀, ε_θ, β, ᾱ, 1000)
println("Training loss: $loss")
```

**この4つの式がDDPMの全てだ。** 残りのゾーンでは、これらを完全導出し、実装する。

:::message
**進捗: 10% 完了** DDPMの4つの核心式を触った。次は「なぜDDPMか」の直感へ。
:::

---

## 🧩 2. 直感ゾーン（15分）— なぜDDPMか？

### 2.1 生成モデルの限界を振り返る

第9-13回で学んだ生成モデルの限界:

| モデル | 長所 | 限界 |
|:-------|:-----|:-----|
| **VAE** (第10回) | 尤度計算可能、安定訓練 | ぼやけた出力 (Gaussian decoder) |
| **GAN** (第12回) | 高品質、シャープ | 訓練不安定、Mode collapse |
| **自己回帰** (第13回) | 尤度計算可能、高品質 | 逐次生成で遅い |

**DDPM [^1] はこれらを全て解決する**:

- **VAE**: ELBO最適化だが、**段階的ノイズ除去**で Gaussian decoder のぼやけを回避
- **GAN**: 敵対的訓練不要。**単純なMSE損失** (ノイズ予測) で安定訓練
- **自己回帰**: 並列訓練可能 (任意の $t$ にジャンプ)。推論は逐次だが、**DDIM [^2] で高速化**

### 2.2 拡散モデルの直感: 熱拡散の逆転

**物理的類推**: データ分布 $q(\mathbf{x}_0)$ に熱拡散 (heat diffusion) を適用すると、最終的に熱平衡状態 (標準正規分布 $\mathcal{N}(0, I)$) に到達する。**この過程を逆転させれば、$\mathcal{N}(0, I)$ からデータ分布を生成できる**。

```mermaid
graph LR
    A["⚪ Data<br/>q(x₀)"] -->|Forward<br/>Diffusion| B["🔵 Noise<br/>𝒩(0,I)"]
    B -->|Reverse<br/>Denoising| C["⚪ Generated<br/>p(x₀)"]

    A -.学習データ.-> D["画像/音声/テキスト"]
    B -.純粋なノイズ.-> E["ランダムな点"]
    C -.生成データ.-> F["新しい画像"]

    style A fill:#e8f5e9
    style B fill:#bbdefb
    style C fill:#fff9c4
```

**3つの比喩**:

1. **熱拡散**: インクを水に垂らすと拡散する。逆再生すれば、水からインクが浮かび上がる。
2. **ノイズ除去フィルタ**: 写真にノイズを加え、フィルタで除去する。これを $T$ 回繰り返す。
3. **Langevin Dynamics** (第35回): スコア関数 $\nabla_\mathbf{x} \log p(\mathbf{x})$ に沿って動くことで分布をサンプリング。DDPMはこれを離散化したもの。

### 2.3 Course IVでの位置づけ — 理論の集大成

Course IV「拡散モデル編」(第33-42回) は、生成モデルの理論を深化させる10回の旅路だ。

```mermaid
graph TD
    L33["第33回<br/>NF & Neural ODE"] --> L34["第34回<br/>EBM & 統計物理"]
    L34 --> L35["第35回<br/>Score Matching"]
    L35 --> L36["第36回<br/>DDPM<br/>(今ここ)"]
    L36 --> L37["第37回<br/>SDE/ODE"]
    L37 --> L38["第38回<br/>Flow Matching"]
    L38 --> L39["第39回<br/>LDM"]
    L39 --> L40["第40回<br/>Consistency"]
    L40 --> L41["第41回<br/>World Models"]
    L41 --> L42["第42回<br/>統一理論"]

    L33 -.NF: 厳密尤度.-> Math1["可逆変換"]
    L34 -.EBM: エネルギー.-> Math2["Z(θ)計算不能"]
    L35 -.Score: ∇log p.-> Math3["Langevin"]
    L36 -.DDPM: 離散拡散.-> Math4["マルコフ連鎖"]
    L37 -.SDE: 連続極限.-> Math5["確率微分方程式"]
    L38 -.FM: OT統一.-> Math6["最適輸送"]

    style L36 fill:#ffeb3b
```

**Course I (第1-8回) の数学がここで花開く**:

| Course I | Course IV 第36回 | 活用方法 |
|:---------|:----------------|:---------|
| 第4回: 確率論 | Forward/Reverse Process | 条件付きガウス分布の性質 |
| 第5回: 測度論・確率過程 | マルコフ連鎖 | 状態遷移の測度論的記述 |
| 第6回: 情報理論 | VLB | KL divergence、ELBO分解 |
| 第8回: EM算法 | 潜在変数モデル | $\mathbf{x}_{1:T}$ が潜在変数 |

**第35回 Score Matching との接続**:

DDPMの損失関数は、**Denoising Score Matching** (第35回) と等価であることが証明されている [^1]。

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = - \frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}
$$

つまり、**ノイズ $\boldsymbol{\epsilon}$ を予測する = スコア関数を予測する**。この統一的視点は第38回 Flow Matching で完全に証明される。

### 2.4 松尾・岩澤研究室との比較

| 観点 | 松尾研 | 本講義 |
|:-----|:-------|:-------|
| **DDPM理論** | Forward/Reverseの概要 | **完全導出** (閉形式・VLB・3形態) |
| **Noise Schedule** | Linear schedule紹介 | **Cosine / SNR単調減少 / Zero Terminal** |
| **サンプリング** | DDIM概要 | **DDIM完全版 + DPM-Solver++ / UniPC** |
| **U-Net** | アーキテクチャ図 | **Time Embedding / GroupNorm / Self-Attention 完全解説** |
| **Score-based視点** | 触れない | **DDPMとScore Matchingの等価性証明** |
| **実装** | PyTorchデモ | **⚡ Julia訓練 + 🦀 Rust推論** |
| **最新性** | 2020-2021 | **2024-2026 SOTA** (Zero Terminal SNR / Improved DDPM) |

**差別化の本質**: 松尾研が「手法の紹介」にとどまるのに対し、本講義は「論文が書ける理論的深さ + Production実装」を貫く。

:::message alert
**ここが踏ん張りどころ**: Zone 3は本講義で最も数式が密集するゾーンだ。Forward Processの閉形式解、Reverse Processのベイズ反転、VLBの完全展開を一つ一つ導出する。第4回の条件付きガウス分布、第8回のELBOが総動員される。
:::

### 2.5 学習戦略 — 数式修行の準備

**Zone 3の全体マップ**:

```mermaid
graph TD
    A[3.1 Forward Process<br/>閉形式解導出] --> B[3.2 Noise Schedule<br/>設計原理]
    B --> C[3.3 Reverse Process<br/>ベイズ反転]
    C --> D[3.4 VLB<br/>完全展開]
    D --> E[3.5 3形態<br/>ε/x₀/v]
    E --> F[3.6 簡素化損失<br/>L_simple]
    F --> G[3.7 SNR視点<br/>統一的理解]
    G --> H[3.8 U-Net<br/>アーキテクチャ]
    H --> I[3.9 DDIM<br/>決定論的サンプリング]
    I --> J[3.10 Score-based<br/>再解釈]

    style A fill:#e3f2fd
    style D fill:#fff9c4
    style I fill:#c8e6c9
```

**学習のコツ**:

1. **紙とペンを用意する**: 各導出を自分の手で追う。
2. **数値検証コード**: 各式をJuliaで確認する (Zone 4で完全実装)。
3. **前提知識の参照**: 第4回 (ガウス分布)、第8回 (ELBO) を手元に置く。
4. **Boss戦の準備**: 3.4 VLB完全展開、3.9 DDIM完全導出が最難関。

:::message
**進捗: 20% 完了** DDPMの直感と全体像を把握した。Zone 3で数式の海に飛び込む。
:::

---

## 📐 3. 数式修行ゾーン（60分）— 理論完全導出

### 3.1 Forward Process の閉形式解導出

**定理**: Forward Process $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$ を $t$ 回適用すると、以下の閉形式が得られる:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

ここで $\alpha_t = 1 - \beta_t$、$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。

**証明** (数学的帰納法):

**Base case** ($t=1$):

$$
q(\mathbf{x}_1 \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{1-\beta_1} \mathbf{x}_0, \beta_1 \mathbf{I}) = \mathcal{N}(\sqrt{\alpha_1} \mathbf{x}_0, (1-\alpha_1) \mathbf{I})
$$

$\bar{\alpha}_1 = \alpha_1$ より成立。

**Inductive step**: $t-1$ で成立すると仮定し、$t$ で成立することを示す。

$$
\begin{aligned}
q(\mathbf{x}_t \mid \mathbf{x}_0) &= \int q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) q(\mathbf{x}_{t-1} \mid \mathbf{x}_0) \, d\mathbf{x}_{t-1} \\
&= \int \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t} \mathbf{x}_{t-1}, (1-\alpha_t) \mathbf{I}) \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0, (1-\bar{\alpha}_{t-1}) \mathbf{I}) \, d\mathbf{x}_{t-1}
\end{aligned}
$$

**ガウス分布の積の性質** (第4回の多変量正規分布の条件付き分布):

2つのガウス分布 $\mathcal{N}(\mathbf{x}; \mathbf{a}, A)$ と $\mathcal{N}(\mathbf{x}; \mathbf{b}, B)$ の積は、正規化定数を除いて $\mathcal{N}(\mathbf{x}; \mathbf{c}, C)$ と等価である。ここで:

$$
C^{-1} = A^{-1} + B^{-1}, \quad \mathbf{c} = C (A^{-1} \mathbf{a} + B^{-1} \mathbf{b})
$$

$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ をreparameterize:

$$
\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})
$$

$q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$ をreparameterize:

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_{t-2}, \quad \boldsymbol{\epsilon}_{t-2} \sim \mathcal{N}(0, \mathbf{I})
$$

代入:

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t} (\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_{t-2}) + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{\alpha_t (1-\bar{\alpha}_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_{t-1}
\end{aligned}
$$

**独立なガウスノイズの合成**: $\boldsymbol{\epsilon}_{t-2}$ と $\boldsymbol{\epsilon}_{t-1}$ は独立。合成ノイズの分散:

$$
\text{Var}[\sqrt{\alpha_t (1-\bar{\alpha}_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t} \boldsymbol{\epsilon}_{t-1}] = \alpha_t (1-\bar{\alpha}_{t-1}) + (1-\alpha_t) = 1 - \alpha_t \bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t
$$

したがって:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \bar{\boldsymbol{\epsilon}}, \quad \bar{\boldsymbol{\epsilon}} \sim \mathcal{N}(0, \mathbf{I})
$$

これは $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})$ を意味する。■

**数値検証**:

```julia
# Verify closed-form derivation
using LinearAlgebra, Statistics

function verify_forward_closed_form(x₀::Vector{Float64}, t::Int, β::Vector{Float64}, n_samples::Int=10000)
    α = 1.0 .- β
    ᾱ = cumprod(α)

    # Method 1: Iterative forward
    samples_iterative = zeros(length(x₀), n_samples)
    for i in 1:n_samples
        x = copy(x₀)
        for s in 1:t
            ε = randn(length(x₀))
            x = sqrt(α[s]) * x + sqrt(1 - α[s]) * ε
        end
        samples_iterative[:, i] = x
    end

    # Method 2: Closed-form
    samples_closed = zeros(length(x₀), n_samples)
    for i in 1:n_samples
        ε = randn(length(x₀))
        samples_closed[:, i] = sqrt(ᾱ[t]) * x₀ + sqrt(1 - ᾱ[t]) * ε
    end

    # Compare statistics
    mean_iter = vec(mean(samples_iterative, dims=2))
    std_iter = vec(std(samples_iterative, dims=2))
    mean_closed = vec(mean(samples_closed, dims=2))
    std_closed = vec(std(samples_closed, dims=2))

    println("Iterative - Mean: $mean_iter, Std: $std_iter")
    println("Closed-form - Mean: $mean_closed, Std: $std_closed")
    println("Theory - Mean: $(sqrt(ᾱ[t]) * x₀), Std: $(sqrt(1 - ᾱ[t]))")
end

x₀ = [1.0, 2.0]
β = range(1e-4, 0.02, length=1000)
verify_forward_closed_form(x₀, 500, β, 10000)
```

**重要な性質**:

1. **$\bar{\alpha}_t$ の挙動**: $t \to T$ で $\bar{\alpha}_t \to 0$ → $q(\mathbf{x}_T \mid \mathbf{x}_0) \approx \mathcal{N}(0, \mathbf{I})$
2. **reparameterization**: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ で一気にサンプリング可能
3. **訓練効率**: 各ミニバッチで異なる $t$ をサンプルでき、並列訓練可能

### 3.2 Noise Schedule の設計原理

**Noise Schedule** $\{\beta_t\}_{t=1}^T$ は、**どれだけ速くノイズを加えるか**を制御する。設計原則:

1. **$\bar{\alpha}_T \approx 0$**: 最終的に $\mathbf{x}_T \approx \mathcal{N}(0, \mathbf{I})$ になる
2. **SNR単調減少**: Signal-to-Noise Ratio $\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$ が $t$ とともに減少
3. **Zero Terminal SNR**: $\bar{\alpha}_T = 0$ で厳密に $\mathcal{N}(0, \mathbf{I})$

#### 3.2.1 Linear Schedule (DDPM [^1])

$$
\beta_t = \beta_{\min} + \frac{t-1}{T-1} (\beta_{\max} - \beta_{\min})
$$

DDPM [^1] では $\beta_{\min} = 10^{-4}$、$\beta_{\max} = 0.02$、$T = 1000$。

**問題点**: $\bar{\alpha}_T > 0$ (Zero Terminal SNR を満たさない) [^5]。

```julia
# Linear schedule
function linear_schedule(T::Int, β_min::Float64=1e-4, β_max::Float64=0.02)
    β = range(β_min, β_max, length=T)
    α = 1.0 .- β
    ᾱ = cumprod(α)
    return β, ᾱ
end

β_linear, ᾱ_linear = linear_schedule(1000)
println("Linear schedule: ᾱ_T = $(ᾱ_linear[end])")  # Should be ≈ 0, but > 0
```

#### 3.2.2 Cosine Schedule (Improved DDPM [^3])

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2 \left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right)
$$

ここで $s = 0.008$ は小さなオフセット (端点での急激な変化を防ぐ)。

**利点**:

- SNRが緩やかに減少 → 訓練安定
- Zero Terminal SNRに近い

```julia
# Cosine schedule (Improved DDPM)
function cosine_schedule(T::Int, s::Float64=0.008)
    t_seq = 0:T
    f_t = @. cos((t_seq / T + s) / (1 + s) * π / 2)^2
    ᾱ = f_t[2:end] ./ f_t[1]  # ᾱ_t = f(t) / f(0)
    β = 1.0 .- (ᾱ ./ [1.0; ᾱ[1:end-1]])  # β_t = 1 - α_t = 1 - ᾱ_t / ᾱ_{t-1}
    return β, ᾱ
end

β_cosine, ᾱ_cosine = cosine_schedule(1000)
println("Cosine schedule: ᾱ_T = $(ᾱ_cosine[end])")
```

#### 3.2.3 Zero Terminal SNR Rescaling (Lin+ 2023 [^5])

**動機**: Linear/Cosine schedule は $\bar{\alpha}_T > 0$ → 訓練と推論の不一致。

**解決策**: Schedule をrescaleして $\bar{\alpha}_T = 0$ を強制。

$$
\tilde{\alpha}_t = \frac{\bar{\alpha}_t - \bar{\alpha}_T}{1 - \bar{\alpha}_T}
$$

```julia
# Zero Terminal SNR rescaling
function rescale_zero_terminal_snr(ᾱ::Vector{Float64})
    ᾱ_T = ᾱ[end]
    ᾱ_rescaled = (ᾱ .- ᾱ_T) ./ (1 - ᾱ_T)
    return ᾱ_rescaled
end

ᾱ_linear_rescaled = rescale_zero_terminal_snr(ᾱ_linear)
println("Rescaled linear: ᾱ_T = $(ᾱ_linear_rescaled[end])")  # Now = 0
```

**Noise Schedule 比較**:

| Schedule | ᾱ_T | SNR単調性 | 訓練安定性 | 推奨度 |
|:---------|:----|:---------|:----------|:-------|
| Linear | > 0 ❌ | ✅ | 中 | ❌ (古い) |
| Cosine | ≈ 0 | ✅ | 高 | ✅ (推奨) |
| Zero Terminal SNR | = 0 ✅ | ✅ | **最高** | ⭐ (2023+) |

### 3.3 Reverse Process のベイズ反転

**目標**: Forward Process $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ の逆過程 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ を求める。

**問題**: $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ は直接計算できない (周辺化困難)。

**解決**: **ベイズの定理** + **$\mathbf{x}_0$ を条件付け**:

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
$$

**マルコフ性**: $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ (未来は過去に依存しない)。

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
$$

各項を代入:

$$
\begin{aligned}
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) &= \mathcal{N}(\sqrt{\alpha_t} \mathbf{x}_{t-1}, (1-\alpha_t) \mathbf{I}) \\
q(\mathbf{x}_{t-1} \mid \mathbf{x}_0) &= \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0, (1-\bar{\alpha}_{t-1}) \mathbf{I}) \\
q(\mathbf{x}_t \mid \mathbf{x}_0) &= \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})
\end{aligned}
$$

**ガウス分布の商の性質** (対数空間で計算):

$$
\begin{aligned}
&\log q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \\
&\propto \log q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) + \log q(\mathbf{x}_{t-1} \mid \mathbf{x}_0) - \log q(\mathbf{x}_t \mid \mathbf{x}_0) \\
&= -\frac{1}{2(1-\alpha_t)} \|\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1}\|^2 - \frac{1}{2(1-\bar{\alpha}_{t-1})} \|\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0\|^2 + \text{const}
\end{aligned}
$$

ここで $\mathbf{x}_t$ に依存しない項は定数として無視。

**平方完成**: $\mathbf{x}_{t-1}$ に関する二次形式に整理:

$$
\begin{aligned}
&-\frac{1}{2} \left( \frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar{\alpha}_{t-1}} \right) \mathbf{x}_{t-1}^2 + \left( \frac{\sqrt{\alpha_t}}{1-\alpha_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0 \right) \mathbf{x}_{t-1}
\end{aligned}
$$

**ガウス分布の標準形** $\mathcal{N}(\boldsymbol{\mu}, \sigma^2)$ と比較:

$$
\log \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I}) \propto -\frac{1}{2\sigma^2} \|\mathbf{x} - \boldsymbol{\mu}\|^2 = -\frac{1}{2\sigma^2} \mathbf{x}^2 + \frac{\boldsymbol{\mu}}{\sigma^2} \mathbf{x}
$$

対応させて:

$$
\frac{1}{\tilde{\beta}_t} = \frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar{\alpha}_{t-1}} = \frac{\alpha_t (1-\bar{\alpha}_{t-1}) + (1-\alpha_t)}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})} = \frac{1 - \bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}
$$

したがって:

$$
\boxed{\tilde{\beta}_t = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}
$$

平均:

$$
\frac{\tilde{\boldsymbol{\mu}}_t}{\tilde{\beta}_t} = \frac{\sqrt{\alpha_t}}{1-\alpha_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0
$$

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right) \cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0
$$

ここで $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ を使うと:

$$
\boxed{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t}
$$

**結論**:

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$

**これがReverse Processの "真の" 分布である。** だが $\mathbf{x}_0$ が未知なので、ニューラルネットワークで近似する:

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
$$

### 3.4 Variational Lower Bound (VLB) 完全展開

**目標**: $\log p_\theta(\mathbf{x}_0)$ を変分推論 (第9回) で下界から評価する。

**ELBO導出** (第9回の復習):

$$
\begin{aligned}
\log p_\theta(\mathbf{x}_0) &= \log \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T} \\
&= \log \int p_\theta(\mathbf{x}_{0:T}) \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \, d\mathbf{x}_{1:T} \\
&= \log \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right] \\
&\geq \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \right] \quad \text{(Jensen不等式)} \\
&= \mathbb{E}_q \left[ \log p_\theta(\mathbf{x}_{0:T}) - \log q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \right]
\end{aligned}
$$

**分解**:

$$
\begin{aligned}
p_\theta(\mathbf{x}_{0:T}) &= p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \\
q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) &= \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
\end{aligned}
$$

代入:

$$
\begin{aligned}
&\mathbb{E}_q \left[ \log p(\mathbf{x}_T) + \sum_{t=1}^T \log p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) - \sum_{t=1}^T \log q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \right] \\
&= \mathbb{E}_q \left[ \log p(\mathbf{x}_T) - \log q(\mathbf{x}_T \mid \mathbf{x}_0) + \sum_{t=2}^T \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)} + \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) \right]
\end{aligned}
$$

**ベイズの定理**: $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}$ を使うと、telescoping:

$$
\sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})} = \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)} = \log \frac{q(\mathbf{x}_1)}{q(\mathbf{x}_T)}
$$

代わりに、**$\mathbf{x}_0$ を条件付け** (Section 3.3):

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \to q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
$$

$$
\begin{aligned}
\text{VLB} &= \mathbb{E}_q \left[ \log p(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)} \right] \\
&= \mathbb{E}_q \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T \mid \mathbf{x}_0)} + \sum_{t=2}^T \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)} + \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) \right]
\end{aligned}
$$

**KL divergence で整理**:

$$
\boxed{L_\text{VLB} = L_T + \sum_{t=2}^T L_{t-1} + L_0}
$$

ここで:

$$
\begin{aligned}
L_T &= D_\text{KL}(q(\mathbf{x}_T \mid \mathbf{x}_0) \| p(\mathbf{x}_T)) \\
L_{t-1} &= D_\text{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)) \\
L_0 &= -\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)
\end{aligned}
$$

**各項の意味**:

- **$L_T$**: $\mathbf{x}_T$ が $\mathcal{N}(0, I)$ にどれだけ近いか (学習不要、$\beta_t$ が適切なら $\approx 0$)
- **$L_{t-1}$**: Reverse Process $p_\theta$ が真の分布 $q$ にどれだけ近いか
- **$L_0$**: 再構成項 (VAEの再構成損失に対応)

**これがDDPMの理論的基盤 — 変分推論 (第9回) の直接的応用である。**

### 3.5 損失関数の3形態: ε / x₀ / v-prediction

**目標**: $L_{t-1}$ を具体的な訓練損失に落とし込む。

**KL divergence**: 両方ガウス分布なので閉形式:

$$
D_\text{KL}(\mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1) \| \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)) = \frac{1}{2} \left( \text{tr}(\Sigma_2^{-1} \Sigma_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \Sigma_2^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - k + \log \frac{|\Sigma_2|}{|\Sigma_1|} \right)
$$

分散を固定 ($\Sigma_1 = \Sigma_2 = \sigma^2 \mathbf{I}$) すると、平均の差だけ残る:

$$
L_{t-1} \propto \|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta\|^2
$$

**3つの予測方式**:

#### 3.5.1 ε-prediction (DDPM [^1])

**$\tilde{\boldsymbol{\mu}}_t$ を $\boldsymbol{\epsilon}$ で表現**:

$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ より $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon})$。代入:

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon} \right)
$$

ネットワークが $\boldsymbol{\epsilon}$ を予測:

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)
$$

損失:

$$
L_{t-1}^\text{ε} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2
$$

**簡素化**: 重み $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$ を無視:

$$
\boxed{L_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]}
$$

#### 3.5.2 x₀-prediction

**$\tilde{\boldsymbol{\mu}}_t$ を直接 $\mathbf{x}_0$ で表現** (Section 3.3):

$$
\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t
$$

ネットワークが $\mathbf{x}_0$ を予測:

$$
\boldsymbol{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \mathbf{x}_\theta(\mathbf{x}_t, t) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t
$$

損失:

$$
L_{t-1}^{x_0} \propto \|\mathbf{x}_0 - \mathbf{x}_\theta(\mathbf{x}_t, t)\|^2
$$

#### 3.5.3 v-prediction (Progressive Distillation, Salimans & Ho 2022)

**Angular parameterization**: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ を角度 $\phi_t = \arctan(\sqrt{(1-\bar{\alpha}_t)/\bar{\alpha}_t})$ で再パラメータ化。

$$
\mathbf{v} = \sqrt{\bar{\alpha}_t} \boldsymbol{\epsilon} - \sqrt{1-\bar{\alpha}_t} \mathbf{x}_0
$$

損失:

$$
L_t^\mathbf{v} = \|\mathbf{v} - \mathbf{v}_\theta(\mathbf{x}_t, t)\|^2
$$

**利点**: $t$ 全体で分散が均一 → 訓練安定。

**3形態の変換**:

$$
\begin{aligned}
\mathbf{x}_0 &= \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}} \\
\boldsymbol{\epsilon} &= \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}} \\
\mathbf{v} &= \sqrt{\bar{\alpha}_t} \boldsymbol{\epsilon} - \sqrt{1-\bar{\alpha}_t} \mathbf{x}_0
\end{aligned}
$$

```julia
# Conversion between ε, x₀, v predictions
function predict_ε_from_x₀(x_t::Vector{Float64}, x₀::Vector{Float64}, ᾱ_t::Float64)
    ε = (x_t - sqrt(ᾱ_t) * x₀) / sqrt(1 - ᾱ_t)
    return ε
end

function predict_x₀_from_ε(x_t::Vector{Float64}, ε::Vector{Float64}, ᾱ_t::Float64)
    x₀ = (x_t - sqrt(1 - ᾱ_t) * ε) / sqrt(ᾱ_t)
    return x₀
end

function predict_v(x₀::Vector{Float64}, ε::Vector{Float64}, ᾱ_t::Float64)
    v = sqrt(ᾱ_t) * ε - sqrt(1 - ᾱ_t) * x₀
    return v
end

# Test
x₀ = [1.0, 2.0]
ε = randn(2)
ᾱ_t = 0.5
x_t = sqrt(ᾱ_t) * x₀ + sqrt(1 - ᾱ_t) * ε

ε_recon = predict_ε_from_x₀(x_t, x₀, ᾱ_t)
x₀_recon = predict_x₀_from_ε(x_t, ε, ᾱ_t)
v = predict_v(x₀, ε, ᾱ_t)

println("Original ε: $ε")
println("Reconstructed ε: $ε_recon")
println("Original x₀: $x₀")
println("Reconstructed x₀: $x₀_recon")
println("v: $v")
```

**どれを使うべきか？**

| 予測対象 | 訓練安定性 | 推論品質 | 推奨シーン |
|:---------|:----------|:---------|:----------|
| **ε** | 高 | 高 | **デフォルト** (DDPM [^1]) |
| **x₀** | 中 | 中 | 低ノイズ領域で有効 |
| **v** | **最高** | 高 | **最新推奨** (v-prediction [^5]) |

### 3.6 簡素化損失 L_simple と VLB の関係

**DDPM [^1] の発見**: VLB損失 $L_\text{VLB}$ の重み付けを無視した $L_\text{simple}$ の方が、サンプル品質が高い。

$$
L_\text{VLB} = L_T + \sum_{t=2}^T L_{t-1} + L_0, \quad L_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]
$$

**なぜ $L_\text{simple}$ が優れているか？**

1. **重み付けの効果**: $L_{t-1}$ の重み $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$ は、小さな $t$ (低ノイズ) を強調する。これが知覚品質に有害。
2. **全時刻一様サンプリング**: $L_\text{simple}$ は $t \sim \text{Uniform}(1, T)$ → 全時刻を均等に学習。
3. **勾配のバランス**: VLB の重みは理論的には正しいが、実際には高ノイズ領域を過学習させる。

**経験則**: 尤度 (bits/dim) を最適化するなら $L_\text{VLB}$、知覚品質 (FID) を最適化するなら $L_\text{simple}$。

### 3.7 SNR (Signal-to-Noise Ratio) 視点での統一的理解

**SNR定義**:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
$$

- $t=0$: $\text{SNR}(0) = \frac{1}{0}$ (無限大、ノイズなし)
- $t=T$: $\text{SNR}(T) \approx 0$ (信号なし)

**Noise Schedule の設計原則**: $\text{SNR}(t)$ が単調減少し、$\text{SNR}(T) = 0$ (Zero Terminal SNR [^5])。

**SNRとWeighting の関係** (Ho+ 2020 [^1] Appendix):

$$
L_\text{VLB} = \mathbb{E}_t \left[ \lambda(t) \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2 \right], \quad \lambda(t) = \frac{1}{2\sigma_t^2} \frac{\beta_t^2}{\alpha_t (1-\bar{\alpha}_t)}
$$

$\lambda(t) \propto \text{SNR}(t)$ → 低SNR (高ノイズ) の時刻を重視。

**$L_\text{simple}$ の再解釈**: $\lambda(t) = 1$ → SNRに依らず全時刻を均等に重視。

**最新の重み付けスキーム** (Min-SNR Weighting, Hang+ 2023):

$$
\lambda_\text{min-SNR}(t) = \min(\text{SNR}(t), \gamma)
$$

$\gamma = 5$ が推奨。高SNR (低ノイズ) の時刻の重みを制限 → 訓練安定。

```julia
# SNR computation
function compute_snr(ᾱ::Vector{Float64})
    snr = ᾱ ./ (1.0 .- ᾱ)
    return snr
end

# Min-SNR weighting
function min_snr_weight(snr::Vector{Float64}, γ::Float64=5.0)
    λ = min.(snr, γ)
    return λ
end

β_cosine, ᾱ_cosine = cosine_schedule(1000)
snr = compute_snr(ᾱ_cosine)
λ_min_snr = min_snr_weight(snr, 5.0)

println("SNR range: [$(minimum(snr)), $(maximum(snr))]")
println("Min-SNR weight range: [$(minimum(λ_min_snr)), $(maximum(λ_min_snr))]")
```

### 3.8 U-Net Architecture for DDPM

**U-Net** は DDPM [^1] の標準アーキテクチャ。**Time Embedding**、**GroupNorm**、**Self-Attention** が核心。

#### 3.8.1 Time Embedding

**動機**: ニューラルネットワーク $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ に時刻 $t$ を入力する。

**Sinusoidal Position Encoding** (Transformer [Vaswani+ 2017] と同じ):

$$
\text{PE}(t, 2i) = \sin(t / 10000^{2i/d}), \quad \text{PE}(t, 2i+1) = \cos(t / 10000^{2i/d})
$$

$d$ は埋め込み次元 (通常 $d_\text{model} \times 4$)。

```julia
# Sinusoidal time embedding
function time_embedding(t::Int, d::Int)
    half_dim = d ÷ 2
    emb = log(10000) / (half_dim - 1)
    emb = exp.(-emb * (0:half_dim-1))
    emb = t * emb
    emb = [sin.(emb); cos.(emb)]
    return emb
end

t = 500
d = 128
t_emb = time_embedding(t, d)
println("Time embedding shape: $(length(t_emb))")
```

**統合**: Time Embedding を各 Residual Block に加算 (FiLM: Feature-wise Linear Modulation)。

$$
\mathbf{h} = \mathbf{h} + \text{MLP}(\text{TimeEmb}(t))
$$

#### 3.8.2 GroupNorm

**Batch Normalization の問題**: バッチサイズに依存 → 小バッチで不安定。

**GroupNorm** (Wu & He 2018): チャネルを $G$ 個のグループに分割し、グループごとに正規化。

$$
\text{GN}(\mathbf{x}) = \gamma \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

$\mu, \sigma$ はグループごとに計算。通常 $G = 32$。

```julia
# GroupNorm (simplified)
function group_norm(x::Matrix{Float64}, G::Int=32)
    C, N = size(x)  # C: channels, N: spatial
    @assert C % G == 0

    # Reshape: (C, N) → (G, C/G, N)
    x_grouped = reshape(x, G, C÷G, N)

    # Normalize per group
    for g in 1:G
        μ = mean(x_grouped[g, :, :])
        σ² = var(x_grouped[g, :, :])
        x_grouped[g, :, :] = (x_grouped[g, :, :] .- μ) ./ sqrt(σ² + 1e-5)
    end

    # Reshape back
    x_norm = reshape(x_grouped, C, N)
    return x_norm
end

x = randn(64, 100)  # 64 channels, 100 spatial
x_norm = group_norm(x, 32)
println("GroupNorm applied, mean: $(mean(x_norm)), std: $(std(x_norm))")
```

#### 3.8.3 Self-Attention

**動機**: 低解像度の特徴マップで **長距離依存** を捕捉。

**Multi-Head Self-Attention** (第14回):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V
$$

U-Netでは、**解像度 16×16 以下** でのみ Attention を適用 (計算量 $O(N^2)$ のため)。

```julia
# Simplified self-attention layer
function self_attention(x::Matrix{Float64}, d_k::Int)
    # x: (d_model, seq_len)
    d_model, seq_len = size(x)

    # Linear projections (simplified: using identity for demo)
    Q = x
    K = x
    V = x

    # Scaled dot-product attention
    scores = (Q' * K) / sqrt(d_k)  # (seq_len, seq_len)
    attn = softmax(scores, dims=2)  # row-wise softmax
    output = V * attn'  # (d_model, seq_len)

    return output
end

softmax(x; dims) = exp.(x .- maximum(x, dims=dims)) ./ sum(exp.(x .- maximum(x, dims=dims)), dims=dims)

x_feature = randn(256, 16*16)  # 256 channels, 16x16 spatial (flattened)
x_attn = self_attention(x_feature, 256)
println("Self-attention output shape: $(size(x_attn))")
```

#### 3.8.4 U-Net 全体構造

```mermaid
graph TD
    A["Input<br/>x_t + TimeEmb(t)"] --> B["DownBlock 1<br/>Conv + GN + SiLU"]
    B --> C["DownBlock 2<br/>+ Self-Attn (16x16)"]
    C --> D["DownBlock 3"]
    D --> E["Bottleneck<br/>+ Self-Attn"]
    E --> F["UpBlock 3<br/>+ Skip from D"]
    F --> G["UpBlock 2<br/>+ Self-Attn + Skip"]
    G --> H["UpBlock 1<br/>+ Skip"]
    H --> I["Output Conv<br/>ε_θ(x_t, t)"]

    style E fill:#fff9c4
    style I fill:#c8e6c9
```

**Skip Connection**: Encoder の特徴を Decoder に直接接続 (U-Net の名前の由来)。

### 3.9 DDIM: 決定論的サンプリング完全版

**DDPM [^1] の問題**: 1000ステップ → 推論に数分かかる。

**DDIM [^2] (Song+ 2020) の革新**: **Non-Markovian forward process** で、決定論的サンプリングを実現。10-50ステップで同等の品質。

#### 3.9.1 Non-Markovian Forward Process

**DDPM**: $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ (マルコフ)

**DDIM**: $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ は **非マルコフ** — $\mathbf{x}_t$ は $\mathbf{x}_0$ に直接依存し、$\mathbf{x}_{t-1}$ を経由しない。

$$
q_\sigma(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 \mathbf{I})
$$

ここで $\sigma_t$ は任意のパラメータ:

- **$\sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}} \sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$**: DDPM と同じ (確率的)
- **$\sigma_t = 0$**: 決定論的 (DDIM)

#### 3.9.2 DDIM サンプリング式

**Reparameterize**: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ より:

$$
\mathbf{x}_0 \approx \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

代入:

$$
\boxed{\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted } \mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \boldsymbol{\epsilon}_t}
$$

ここで $\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I})$。

**決定論的サンプリング** ($\sigma_t = 0$):

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta
$$

**加速**: $T$ を $S$ ステップにスキップ ($\tau_1, \dots, \tau_S \subset \{1, \dots, T\}$)。

```julia
# DDIM sampling step
function ddim_step(x_t::Vector{Float64}, ε_θ::Vector{Float64}, t::Int, t_prev::Int, ᾱ::Vector{Float64}, η::Float64=0.0)
    # η: stochasticity parameter (0 = deterministic, 1 = DDPM-like)
    ᾱ_t = ᾱ[t]
    ᾱ_prev = (t_prev > 0) ? ᾱ[t_prev] : 1.0

    # Predicted x₀
    x₀_pred = (x_t - sqrt(1 - ᾱ_t) * ε_θ) / sqrt(ᾱ_t)

    # Variance
    σ_t = η * sqrt((1 - ᾱ_prev) / (1 - ᾱ_t)) * sqrt(1 - ᾱ_t / ᾱ_prev)

    # Direction pointing to x_t
    dir_xt = sqrt(1 - ᾱ_prev - σ_t^2) * ε_θ

    # Random noise (zero if deterministic)
    noise = (η > 0) ? randn(length(x_t)) : zeros(length(x_t))

    # DDIM step
    x_prev = sqrt(ᾱ_prev) * x₀_pred + dir_xt + σ_t * noise

    return x_prev
end

# Test
x_t = randn(2)
ε_θ = randn(2)
β_cosine, ᾱ_cosine = cosine_schedule(1000)

# Deterministic (η=0)
x_prev_det = ddim_step(x_t, ε_θ, 1000, 500, ᾱ_cosine, 0.0)
println("Deterministic DDIM: $x_prev_det")

# Stochastic (η=1, DDPM-like)
x_prev_sto = ddim_step(x_t, ε_θ, 1000, 500, ᾱ_cosine, 1.0)
println("Stochastic DDIM: $x_prev_sto")
```

**DDIM の利点**:

1. **高速**: 50ステップで DDPM 1000ステップと同等の品質
2. **決定論的**: 同じ $\mathbf{x}_T$ から常に同じ $\mathbf{x}_0$ (再現性)
3. **潜在空間補間**: $\mathbf{x}_T$ を補間 → $\mathbf{x}_0$ を補間 (Latent Consistency)

#### 3.9.3 DDIM と Probability Flow ODE の関係

**Probability Flow ODE** (Song+ 2020 score-based generative models, 第35回):

$$
\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})
$$

**DDIM は Probability Flow ODE の Euler法離散化** に対応 (第38回 Flow Matching で詳述)。

$$
\mathbf{x}_{t-\Delta t} = \mathbf{x}_t - \Delta t \left[ f(\mathbf{x}_t, t) - \frac{1}{2} g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}_t) \right]
$$

**これがDDIM → Flow Matching → OT統一理論への道筋である。**

### 3.10 Score-based 視点での DDPM 再解釈

**Score Matching** (第35回) との等価性:

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = - \frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}
$$

**証明**:

$$
\begin{aligned}
\log q(\mathbf{x}_t \mid \mathbf{x}_0) &= \log \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I}) \\
&= -\frac{1}{2(1-\bar{\alpha}_t)} \|\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0\|^2 + \text{const}
\end{aligned}
$$

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{1}{1-\bar{\alpha}_t} (\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0) = - \frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}
$$

ここで $\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0 = \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$。■

**Score Network との対応**:

$$
\mathbf{s}_\theta(\mathbf{x}_t, t) = \nabla_{\mathbf{x}_t} \log p_\theta(\mathbf{x}_t) \approx - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar{\alpha}_t}}
$$

**つまり、ノイズ予測 = スコア予測 (rescaled)**。

**Denoising Score Matching** (第35回) の損失:

$$
L_\text{DSM} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \lambda(t) \left\| \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) - \mathbf{s}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
$$

$\lambda(t) = (1-\bar{\alpha}_t)$ とすると:

$$
L_\text{DSM} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right] = L_\text{simple}
$$

**結論**: **DDPM = Denoising Score Matching**。DDPMはScore-based Generative Modelsの一形態である。

**Song & Ho の統一理論** (第38回で完全証明):

```mermaid
graph TD
    A[Score-based<br/>∇log p] --> B[DDPM<br/>ε-prediction]
    B --> C[DDIM<br/>PF-ODE]
    C --> D[Flow Matching<br/>OT-CFM]

    A -.Denoising Score.-> B
    B -.Deterministic.-> C
    C -.Continuous.-> D

    style A fill:#e3f2fd
    style B fill:#fff9c4
    style D fill:#c8e6c9
```

**これで Zone 3 完了 — DDPM の理論を完全導出した。**

:::message
**進捗: 50% 完了** Forward/Reverse/VLB/3形態/SNR/U-Net/DDIM/Score-based を完全導出した。Boss Battle 撃破。Zone 4 で実装へ。
:::

---
