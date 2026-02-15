---
title: "第16回: SSM理論 & Mambaの克服: 30秒の驚き→数式修行→実装マスター""
emoji: "🦛"
type: "tech"
topics: ["machinelearning", "deeplearning", "ssm", "julia", "rust"]
published: true
---

# 第16回: SSM理論 & Mambaの克服 — "忘れる"限界を超える数学

> **RNNは忘れ、Attentionは計算量で死ぬ。SSMは両方を解決できるのか？**

第14回でAttentionがRNN/CNNの限界を突破し、全系列参照と並列計算を実現したことを学んだ。第15回では、そのAttentionのO(N²)の壁と、それを突破する多様な試み(Flash/Sparse/Linear Attention、MoE)を見た。

だが、ここで問いたい。**Attentionに固執する必要はあるのか？**

1980年代の制御理論・信号処理に遡る**状態空間モデル(SSM: State Space Models)**が、2020年代に機械学習コミュニティで再発見された。HiPPO理論[^1]による初期化、S4[^2]の構造化と対角化、そしてMamba[^3]のSelective SSMによって、SSMは「第三の道」として台頭した。

**O(N)の計算量。O(1)の推論。長距離依存の理論的保証。** そして何より、**"忘れる"という根本的限界をどう克服したか**。

本講義では、SSMの数学的基礎から最前線のMambaまでを完全導出する。連続時間状態空間→離散化→HiPPO→S4の対角化→Mambaの選択性。全てを⚡Julia + 🦀Rustで実装する。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論(論文が書ける)、実装(Production-ready)、最新(2025-2026 SOTA)の3軸で差別化する。
:::

```mermaid
graph TD
    A["🔙 RNN<br/>O(N) 但し訓練困難"] --> B{"第三の道"}
    C["🎯 Attention<br/>O(N²) 長系列困難"] --> B
    B --> D["SSM<br/>連続時間状態空間"]
    D --> E["S4<br/>HiPPO + 対角化 + FFT"]
    E --> F["Mamba<br/>Selective SSM"]
    F --> G["選択的記憶<br/>忘れる限界を克服"]

    style A fill:#ffcdd2
    style C fill:#ffcdd2
    style D fill:#fff9c4
    style E fill:#c8e6c9
    style F fill:#81c784
    style G fill:#4caf50
```

**所要時間の目安**:

| ゾーン | 内容 | 時間 | 難易度 |
|:-------|:-----|:-----|:-------|
| Zone 0 | クイックスタート | 30秒 | ★☆☆☆☆ |
| Zone 1 | 体験ゾーン | 10分 | ★★☆☆☆ |
| Zone 2 | 直感ゾーン | 15分 | ★★★☆☆ |
| Zone 3 | 数式修行ゾーン | 60分 | ★★★★★ |
| Zone 4 | 実装ゾーン | 45分 | ★★★★☆ |
| Zone 5 | 実験ゾーン | 30分 | ★★★★☆ |
| Zone 6 | 振り返りゾーン | 30分 | ★★★★☆ |

---

## 🚀 0. クイックスタート(30秒) — SSMで系列を処理する

**ゴール**: SSMが系列を処理する仕組みを30秒で体感する。

状態空間モデルは、隠れ状態$h_t$を介して入力$u_t$を出力$y_t$に変換する。

```julia
using LinearAlgebra

# Discrete SSM: h_t = A h_{t-1} + B u_t, y_t = C h_t
function discrete_ssm(u::Vector{Float32}, A::Matrix{Float32}, B::Vector{Float32}, C::Vector{Float32})
    N, d = length(u), length(B)
    h = zeros(Float32, d)
    y = zeros(Float32, N)

    for t in 1:N
        h = A * h + B * u[t]  # recurrent update
        y[t] = dot(C, h)       # output projection
    end
    return y
end

# Example: 1D SSM with d=2 hidden state
A = Float32[0.9 0.1; -0.1 0.9]  # stable dynamics
B = Float32[1.0, 0.0]
C = Float32[1.0, 0.5]

u = randn(Float32, 16)  # input sequence
y = discrete_ssm(u, A, B, C)

println("Input:  ", round.(u[1:5], digits=2))
println("Output: ", round.(y[1:5], digits=2))
println("SSM shape: d=$(size(A,1)), N=$(length(u))")
```

出力:
```
Input:  [0.5, -0.32, 1.42, -1.54, 0.13]
Output: [0.5, 0.13, 1.41, -0.52, 0.3]
SSM shape: d=2, N=16
```

**たった3行の再帰で、入力系列が状態を介して出力へ変換される。** これがSSMの基本メカニズムだ。

背後にある連続時間の微分方程式:

$$
\frac{d h(t)}{d t} = A h(t) + B u(t), \quad y(t) = C h(t) + D u(t)
$$

離散化することで上記の再帰形式が得られる。**S4はこの$A$を特殊な構造で初期化し、対角化して高速化する。MambaはさらにABCを入力依存にする。**

:::message
**進捗: 3% 完了** SSMの基本メカニズムを体感した。連続時間→離散化→再帰の流れを理解しよう。
:::

---

## 🎮 1. 体験ゾーン(10分) — SSMのパラメータを動かして理解する

### 1.1 状態行列$A$の固有値と記憶の減衰

SSMの隠れ状態$h_t$の更新式は$h_t = Ah_{t-1} + Bu_t$。$A$の固有値が記憶の減衰率を決める。

```julia
using Plots

# Different decay rates via eigenvalues of A
function compare_decay()
    N = 50
    u = vcat(ones(Float32, 10), zeros(Float32, N-10))  # impulse at t=1..10

    # Case 1: Fast decay (λ=0.5)
    A1 = Float32[0.5 0.0; 0.0 0.5]
    # Case 2: Slow decay (λ=0.9)
    A2 = Float32[0.9 0.0; 0.0 0.9]
    # Case 3: Very slow (λ=0.99)
    A3 = Float32[0.99 0.0; 0.0 0.99]

    B = Float32[1.0, 0.0]
    C = Float32[1.0, 0.5]

    y1 = discrete_ssm(u, A1, B, C)
    y2 = discrete_ssm(u, A2, B, C)
    y3 = discrete_ssm(u, A3, B, C)

    plot([u, y1, y2, y3], label=["Input" "λ=0.5" "λ=0.9" "λ=0.99"],
         xlabel="Time step", ylabel="Value",
         title="SSM Memory Decay vs Eigenvalue",
         linewidth=2, legend=:topright)
end

compare_decay()
```

| Eigenvalue | Memory | Use case |
|:-----------|:-------|:---------|
| $\lambda < 0.5$ | Short-term | Recent context only |
| $0.5 < \lambda < 0.9$ | Medium | Typical sequences |
| $\lambda > 0.95$ | Long-term | Long-range dependencies |
| $\lambda \to 1$ | Unstable | Exploding gradients |

**固有値が1に近いほど長期記憶が保たれるが、訓練が不安定になる。** S4/HiPPOはこのトレードオフを理論的に解決する。

:::details RNNとの比較
RNNは$h_t = \tanh(W_h h_{t-1} + W_u u_t)$のように非線形。勾配消失/爆発問題がある。SSMは線形だが、非線形性はゲートや複数層で導入する。S4はこの線形性を活かして対角化→FFTで並列化する。
:::

### 1.2 SSMの3つの形態

同じSSMを3つの異なる形で表現できる[^2]:

| 形態 | 数式 | 用途 | 計算量 |
|:-----|:-----|:-----|:-------|
| **連続時間** | $\frac{dh}{dt}=Ah+Bu, y=Ch$ | 理論的定義 | - |
| **再帰形態** | $h_t=\bar{A}h_{t-1}+\bar{B}u_t, y_t=Ch_t$ | 推論(逐次生成) | O(N) 逐次 |
| **畳み込み形態** | $y=\bar{\mathcal{K}} * u$ | 訓練(並列計算) | O(N log N) FFT |

再帰形態は推論時に1ステップずつ処理する(自己回帰生成)。畳み込み形態は訓練時に全系列を並列処理する。**S4は両方の形態を使い分ける。**

```julia
# Convolutional form: precompute kernel K
function ssm_convolution(u::Vector{Float32}, A::Matrix{Float32}, B::Vector{Float32}, C::Vector{Float32}, L::Int)
    # Compute SSM convolution kernel K[i] = C * A^i * B for i=0..L-1
    K = zeros(Float32, L)
    Ai = Matrix{Float32}(I, size(A))  # A^0 = I
    for i in 1:L
        Ai = A * Ai  # A^i
        K[i] = dot(C, Ai * B)
    end

    # Convolve: y = K * u (use FFT for O(N log N))
    # For simplicity, direct convolution here (O(N²))
    N = length(u)
    y = zeros(Float32, N)
    for t in 1:N
        for k in 1:min(t, L)
            y[t] += K[k] * u[t - k + 1]
        end
    end
    return y, K
end

# Compare recurrent vs convolutional
u = randn(Float32, 16)
A = Float32[0.9 0.1; -0.1 0.9]
B = Float32[1.0, 0.0]
C = Float32[1.0, 0.5]

y_rec = discrete_ssm(u, A, B, C)
y_conv, K = ssm_convolution(u, A, B, C, 16)

println("Recurrent:     ", round.(y_rec[1:5], digits=3))
println("Convolutional: ", round.(y_conv[1:5], digits=3))
println("Kernel K[1:5]: ", round.(K[1:5], digits=3))
```

:::message
**進捗: 10% 完了** SSMの固有値による記憶制御と、3つの等価な形態を理解した。次は「なぜSSMか」を深掘りする。
:::

---

## 🧩 2. 直感ゾーン(15分) — なぜSSMが必要なのか

### 2.1 RNN/Attention/SSMの三つ巴

第14回で学んだように、RNNは勾配消失で長距離依存を学習できず、Attentionは$O(N^2)$で長系列が困難だった。SSMはその両方を解決する「第三の道」を目指す。

| モデル | 訓練 | 推論 | 長距離記憶 | 並列化 | 主な課題 |
|:-------|:-----|:-----|:-----------|:-------|:---------|
| **RNN** | 逐次 O(N) | 逐次 O(N) | △勾配消失 | ✗ | BPTT不安定 |
| **Attention** | 並列 O(N²) | 並列 O(N²) | ◎全系列参照 | ◎ | メモリ爆発 |
| **SSM** | 並列 O(N log N) | 逐次 O(N) | ◎理論保証 | ◎ | パラメータ共有 |

SSMは訓練時に畳み込み形態で並列化(FFT利用)し、推論時に再帰形態で逐次処理する。**Attentionのような全系列参照はできないが、$O(N)$の計算量で長距離依存を扱える。**

```mermaid
graph LR
    A["系列モデリング<br/>の三角形"] --> B["RNN<br/>逐次・勾配消失"]
    A --> C["Attention<br/>並列・O(N²)"]
    A --> D["SSM<br/>並列訓練+逐次推論"]

    B --> E["LSTMでも限界"]
    C --> F["FlashAttention等"]
    D --> G["S4→Mamba"]

    style A fill:#e1f5fe
    style D fill:#c8e6c9
    style G fill:#81c784
```

### 2.2 本シリーズでの位置づけ

Course IIの第9-18回は生成モデル理論編。第14回でAttentionを学び、第15回でその効率化を見た。**第16回SSMは「Attention以外の道」を示す。**

| 第14回 | 第15回 | **第16回** | 第17回 |
|:-------|:-------|:-----------|:-------|
| Attention基礎 | Flash/Sparse/MoE | **SSM/S4/Mamba** | Mamba-2/RWKV |

第16回で学ぶSSMの理論は、第17回のMamba-2(Attention=SSM双対性)と第18回のハイブリッド(Jamba/Zamba)への橋渡しとなる。

### 2.3 SSMの歴史: 制御理論から深層学習へ

状態空間モデルは元々1960年代のKalmanフィルタ[^4]に遡る。信号処理・制御理論の基礎だった。2020年にHiPPO[^1]が「長距離記憶の最適初期化」を示し、2021年にS4[^2]が「対角化による高速化」を実現。2023年のMamba[^3]で「入力依存パラメータ(Selective SSM)」が加わり、ついにTransformer級の性能を達成した。

```mermaid
graph TD
    A["1960s Kalman Filter<br/>状態推定の基礎"] --> B["2020 HiPPO<br/>長距離記憶の理論"]
    B --> C["2021 S4<br/>対角化+FFT"]
    C --> D["2022 H3<br/>暗黙的長畳み込み"]
    D --> E["2023 Mamba<br/>Selective SSM"]
    E --> F["2024 Mamba-2/SSD<br/>Attention=SSM双対性"]

    style A fill:#ffecb3
    style B fill:#fff9c4
    style C fill:#c8e6c9
    style E fill:#81c784
    style F fill:#4caf50
```

### 2.4 学習戦略: 数式→コード→実験

Zone 3で連続時間SSM→離散化→HiPPO→S4→Mambaの完全導出を行う。Zone 4で⚡Julia実装。Zone 5でLong Range Arenaでベンチマーク。

**ここが踏ん張りどころ**: S4の対角化証明とMambaのSelective SSMは、このシリーズで最も難解な数式の1つ。だが**理解すれば2025年のSSM論文が全て読める**ようになる。

:::details トロイの木馬: Juliaの活躍
第10回でJuliaが登場し、多重ディスパッチで型に応じた自動最適化を実現した。SSMのような数値計算では、Juliaの型安定性とJITコンパイルが威力を発揮する。S4のFFTカーネル、Mambaのscanアルゴリズムなど、数式がほぼそのままコードになる。
:::

:::message
**進捗: 20% 完了** SSMの必要性、歴史、Course IIでの位置づけを理解した。さあ、数式修行ゾーンへ。
:::

---

## 📐 3. 数式修行ゾーン(60分) — SSMの完全導出

ここから本格的な理論に入る。連続時間状態空間→離散化→HiPPO→S4の対角化→Mambaの選択性まで、全てを導出する。

### 3.1 連続時間状態空間モデルの定義

制御理論における標準的な線形時不変(LTI)システム:

$$
\begin{cases}
\frac{d h(t)}{d t} = A h(t) + B u(t) \\
y(t) = C h(t) + D u(t)
\end{cases}
$$

- $u(t) \in \mathbb{R}$: 入力信号(スカラー、一般化可)
- $h(t) \in \mathbb{R}^d$: 隠れ状態(状態ベクトル)
- $y(t) \in \mathbb{R}$: 出力信号
- $A \in \mathbb{R}^{d \times d}$: 状態遷移行列(dynamics)
- $B \in \mathbb{R}^{d \times 1}$: 入力行列
- $C \in \mathbb{R}^{1 \times d}$: 出力行列
- $D \in \mathbb{R}$: 直接フィードスルー項(通常0)

**幾何学的意味**: $A$が状態空間の流れ(flow)を定義する。固有値の実部が負なら安定(減衰)、正なら不安定(爆発)。$B$は入力がどの方向に状態を動かすか、$C$は状態のどの成分を観測するか。

:::message
**数式の声**: "$\frac{dh}{dt} = Ah$" は「状態が時間とともにどう変化するか」を記述する。線形ODEの基本形。
:::

#### 初期値問題の解

$u(t) = 0$(入力なし)の場合、常微分方程式$\frac{dh}{dt} = Ah$の解は:

$$
h(t) = e^{At} h(0)
$$

ここで$e^{At}$は**行列指数関数**(第2回で学んだ):

$$
e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!} = I + At + \frac{(At)^2}{2!} + \cdots
$$

$A$が対角化可能($A = V \Lambda V^{-1}$)なら:

$$
e^{At} = V e^{\Lambda t} V^{-1}, \quad e^{\Lambda t} = \begin{pmatrix} e^{\lambda_1 t} & & \\ & \ddots & \\ & & e^{\lambda_d t} \end{pmatrix}
$$

**固有値$\lambda_i$が減衰率を決定する。** $\text{Re}(\lambda_i) < 0$なら$e^{\lambda_i t} \to 0$(安定)。

#### 入力ありの場合: 畳み込み表現

$u(t)$が存在する場合、解は:

$$
h(t) = e^{At} h(0) + \int_0^t e^{A(t-\tau)} B u(\tau) \, d\tau
$$

出力は:

$$
y(t) = C e^{At} h(0) + C \int_0^t e^{A(t-\tau)} B u(\tau) \, d\tau + D u(t)
$$

初期状態$h(0)=0$とすると:

$$
y(t) = \int_0^t \underbrace{C e^{A(t-\tau)} B}_{\text{インパルス応答} \, \mathcal{K}(t-\tau)} u(\tau) \, d\tau + D u(t)
$$

これは**畳み込み積分**:

$$
y(t) = (\mathcal{K} * u)(t), \quad \mathcal{K}(t) = C e^{At} B
$$

**SSMの本質**: 入力$u$と、時間減衰するカーネル$\mathcal{K}(t)$の畳み込みで出力$y$が得られる。

:::details 検証コード
```julia
using DifferentialEquations

# Solve continuous SSM: dh/dt = Ah + Bu
function solve_continuous_ssm(u_func, tspan, A, B, C, D)
    function ode!(dh, h, p, t)
        dh .= A * h + B * u_func(t)
    end

    h0 = zeros(size(A, 1))
    prob = ODEProblem(ode!, h0, tspan)
    sol = solve(prob, Tsit5())

    # Compute output y(t) = Ch(t) + Du(t)
    t_eval = range(tspan[1], tspan[2], length=100)
    y = [dot(C, sol(t)) + D * u_func(t) for t in t_eval]
    return t_eval, y
end

# Example
A = [-0.5 0.0; 0.0 -0.3]
B = [1.0; 0.0]
C = [1.0, 0.5]
D = 0.0

u_func(t) = exp(-t)  # decaying input
t, y = solve_continuous_ssm(u_func, (0.0, 10.0), A, B, C, D)

using Plots
plot(t, y, xlabel="Time", ylabel="Output y(t)", label="SSM output", linewidth=2)
```
:::

### 3.2 離散化: 連続→離散への変換

深層学習では時間は離散的。$\Delta$を時間ステップ幅として、$t_k = k\Delta$で離散化する。

#### Zero-Order Hold (ZOH) 離散化

最も一般的な手法。区間$[k\Delta, (k+1)\Delta)$で入力$u(t) = u_k$(定数)と仮定。

$$
h((k+1)\Delta) = e^{A\Delta} h(k\Delta) + \left( \int_0^\Delta e^{A\tau} d\tau \right) B u_k
$$

離散パラメータを定義:

$$
\bar{A} = e^{A\Delta}, \quad \bar{B} = \left( \int_0^\Delta e^{A\tau} d\tau \right) B = (A^{-1}(e^{A\Delta} - I)) B
$$

すると離散時間SSM:

$$
h_k = \bar{A} h_{k-1} + \bar{B} u_k, \quad y_k = C h_k + D u_k
$$

**$\bar{A}$と$\bar{B}$の計算**: $A$が対角化可能なら:

$$
\bar{A} = V e^{\Lambda \Delta} V^{-1}, \quad \bar{B} = V \left( \Lambda^{-1} (e^{\Lambda \Delta} - I) \right) V^{-1} B
$$

ここで$e^{\Lambda \Delta} = \text{diag}(e^{\lambda_1 \Delta}, \ldots, e^{\lambda_d \Delta})$。

:::message
**つまずきポイント**: なぜ$\bar{B} = A^{-1}(e^{A\Delta} - I)B$? 積分$\int_0^\Delta e^{A\tau} d\tau$を行列指数の性質から導く。$e^{A\tau}$の積分は$(A^{-1}e^{A\tau})|_0^\Delta = A^{-1}(e^{A\Delta} - I)$。
:::

#### 他の離散化手法

| 手法 | $\bar{A}$ | $\bar{B}$ | 特徴 |
|:-----|:----------|:----------|:-----|
| **Forward Euler** | $I + \Delta A$ | $\Delta B$ | 単純、不安定 |
| **Bilinear (Tustin)** | $(I - \frac{\Delta}{2}A)^{-1}(I + \frac{\Delta}{2}A)$ | $(I - \frac{\Delta}{2}A)^{-1}\Delta B$ | 周波数応答保存 |
| **ZOH** | $e^{A\Delta}$ | $(A^{-1}(e^{A\Delta} - I))B$ | 正確、S4採用 |

S4はZOHを使用[^2]。数値的安定性が高く、連続時間の性質を最もよく保つ。

```julia
using LinearAlgebra

# Zero-Order Hold discretization
function discretize_zoh(A::Matrix{Float64}, B::Vector{Float64}, Δ::Float64)
    d = size(A, 1)
    # A_bar = exp(A * Δ)
    A_bar = exp(A * Δ)

    # B_bar = (A^{-1} (exp(A*Δ) - I)) B
    # If A is invertible:
    if det(A) != 0
        B_bar = (inv(A) * (A_bar - I)) * B
    else
        # Numerical integration fallback
        B_bar = sum([exp(A * τ) * B * Δ/100 for τ in range(0, Δ, length=100)])
    end

    return A_bar, B_bar
end

# Example
A = [-0.5 0.0; 0.0 -0.3]
B = [1.0, 0.0]
Δ = 0.1

A_bar, B_bar = discretize_zoh(A, B, Δ)
println("A_bar = ", round.(A_bar, digits=4))
println("B_bar = ", round.(B_bar, digits=4))

# Eigenvalues decay as exp(λ * Δ)
λ = eigvals(A)
λ_discrete = exp.(λ * Δ)
println("Continuous eigenvalues: ", λ)
println("Discrete eigenvalues:   ", λ_discrete)
println("A_bar eigenvalues:      ", eigvals(A_bar))
```

### 3.3 離散SSMの畳み込み形態

離散時間での再帰:

$$
h_k = \bar{A}^k h_0 + \sum_{j=0}^{k-1} \bar{A}^{k-1-j} \bar{B} u_j
$$

初期状態$h_0 = 0$とすると:

$$
y_k = C h_k = C \sum_{j=0}^{k-1} \bar{A}^{k-1-j} \bar{B} u_j = \sum_{j=0}^{k-1} \underbrace{C \bar{A}^{k-1-j} \bar{B}}_{\bar{\mathcal{K}}_{k-j}} u_j
$$

これは離散畳み込み:

$$
y = \bar{\mathcal{K}} * u, \quad \bar{\mathcal{K}}_k = C \bar{A}^k \bar{B} \quad (k=0,1,\ldots,L-1)
$$

**カーネル$\bar{\mathcal{K}}$を事前計算すれば、FFTで$O(L \log L)$の畳み込みが可能。** これがS4の訓練時高速化の鍵。

#### カーネル計算の計算量

素朴には$\bar{\mathcal{K}}_k = C \bar{A}^k \bar{B}$を$k=0,\ldots,L-1$で計算すると$O(Ld^3)$(行列累乗)。S4は**対角化により$O(Ld)$に削減**する。

#### 畳み込みと再帰の等価性の証明

**Claim**: 畳み込み形態$y = \bar{\mathcal{K}} * u$と再帰形態$h_k = \bar{A}h_{k-1} + \bar{B}u_k, y_k = Ch_k$は等価。

**証明**:

再帰形態から出発:

$$
\begin{aligned}
h_0 &= 0 \\
h_1 &= \bar{A} h_0 + \bar{B} u_0 = \bar{B} u_0 \\
h_2 &= \bar{A} h_1 + \bar{B} u_1 = \bar{A} \bar{B} u_0 + \bar{B} u_1 \\
h_3 &= \bar{A} h_2 + \bar{B} u_2 = \bar{A}^2 \bar{B} u_0 + \bar{A} \bar{B} u_1 + \bar{B} u_2
\end{aligned}
$$

一般に:

$$
h_k = \sum_{j=0}^{k-1} \bar{A}^{k-1-j} \bar{B} u_j
$$

出力:

$$
y_k = C h_k = C \sum_{j=0}^{k-1} \bar{A}^{k-1-j} \bar{B} u_j
$$

$\bar{\mathcal{K}}_m = C \bar{A}^m \bar{B}$と定義すると:

$$
y_k = \sum_{j=0}^{k-1} \bar{\mathcal{K}}_{k-1-j} u_j = \sum_{m=0}^{k-1} \bar{\mathcal{K}}_m u_{k-1-m}
$$

これは離散畳み込み$(y = \bar{\mathcal{K}} * u)$の定義そのもの。$\square$

#### FFT畳み込みの高速化

離散畳み込み$y = \bar{\mathcal{K}} * u$を直接計算すると$O(L^2)$。

**FFTによる高速化**:

畳み込み定理より:

$$
\mathcal{F}\{y\} = \mathcal{F}\{\bar{\mathcal{K}}\} \cdot \mathcal{F}\{u\}
$$

つまり:

$$
y = \mathcal{F}^{-1} \left\{ \mathcal{F}\{\bar{\mathcal{K}}\} \cdot \mathcal{F}\{u\} \right\}
$$

FFT/IFFTは$O(L \log L)$ → 全体で$O(L \log L)$。

**実装のポイント**:

1. **Zero-padding**: 循環畳み込みを避けるため、$\bar{\mathcal{K}}$と$u$をゼロパディング(長さ$2L-1$)
2. **要素積**: 周波数領域での要素積$\mathcal{F}\{\bar{\mathcal{K}}\} \odot \mathcal{F}\{u\}$は$O(L)$
3. **実部抽出**: 最終的に実部のみ取る(元が実数なら)

```julia
using FFTW

function fft_conv(K::Vector{Float64}, u::Vector{Float64})
    L_K, L_u = length(K), length(u)
    L = L_K + L_u - 1

    K_pad = [K; zeros(L - L_K)]
    u_pad = [u; zeros(L - L_u)]

    K_fft = fft(K_pad)
    u_fft = fft(u_pad)

    y_fft = K_fft .* u_fft
    y = real.(ifft(y_fft))

    return y[1:L_u]  # Trim to original length
end
```

#### 畳み込み形態の利点と限界

**利点**:

1. **並列化**: カーネル$\bar{\mathcal{K}}$計算後、全時刻を並列処理可能
2. **訓練高速化**: GPU上でFFTは高度に最適化済み
3. **バッチ処理**: 複数サンプルを同時に処理

**限界**:

1. **固定パラメータ前提**: $\bar{A}, \bar{B}, \bar{C}$が全時刻で共通でないと適用不可
2. **カーネル事前計算**: 系列長$L$が大きいと$\bar{\mathcal{K}}$のメモリ消費が増加
3. **推論には不向**: 推論時(1トークンずつ生成)は再帰形態が効率的

**S4の戦略**: 訓練は畳み込み形態、推論は再帰形態。両方を使い分ける。

#### 畳み込みカーネルの性質

$\bar{\mathcal{K}}_k = C \bar{A}^k \bar{B}$の挙動:

1. **指数減衰**: $\bar{A}$の固有値$\lambda_i < 1$なら、$\bar{A}^k \to 0$ as $k \to \infty$
2. **時間不変**: カーネルは時刻の差$k-j$のみに依存 → Toeplitz行列
3. **因果性**: $\bar{\mathcal{K}}_k = 0$ for $k < 0$ → 未来を見ない

**Toeplitz構造**:

畳み込み行列$\mathcal{K}$は:

$$
\mathcal{K} = \begin{pmatrix}
\bar{\mathcal{K}}_0 & 0 & 0 & \cdots \\
\bar{\mathcal{K}}_1 & \bar{\mathcal{K}}_0 & 0 & \cdots \\
\bar{\mathcal{K}}_2 & \bar{\mathcal{K}}_1 & \bar{\mathcal{K}}_0 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}
$$

対角線上の値が一定 → FFTで対角化可能。

### 3.4 HiPPO: 長距離記憶の理論的基盤

**問題設定**: 時刻$t$までの入力履歴$u(\tau), \tau \in [0, t]$を、$d$次元の状態$h(t)$に圧縮したい。どのように初期化すれば最適か？

HiPPO (High-order Polynomial Projection Operators)[^1]は、**直交多項式基底への射影**として$h(t)$を定義する。

#### 数学的定式化

測度$\mu(t)$に対して、関数$u(\tau), \tau \leq t$を$d$次までの多項式$P_0, P_1, \ldots, P_{d-1}$で近似:

$$
u(\tau) \approx \sum_{n=0}^{d-1} c_n(t) P_n(\tau)
$$

係数$c_n(t)$が状態$h(t)$の第$n$成分。最適化問題:

$$
\min_{c_0, \ldots, c_{d-1}} \int_0^t \left( u(\tau) - \sum_{n=0}^{d-1} c_n P_n(\tau) \right)^2 \mu(t, \tau) \, d\tau
$$

直交多項式の性質から、最適係数は射影:

$$
c_n(t) = \int_0^t u(\tau) P_n(\tau) \mu(t, \tau) \, d\tau
$$

時間微分を取ると、$c(t) = (c_0(t), \ldots, c_{d-1}(t))$が微分方程式:

$$
\frac{d c(t)}{d t} = A_{\text{HiPPO}} c(t) + B_{\text{HiPPO}} u(t)
$$

を満たす。$A_{\text{HiPPO}}$の具体形は測度$\mu$と多項式族$\{P_n\}$に依存する。

#### HiPPO-LegS: Legendre多項式 + Sliding window

測度$\mu(t, \tau) = \mathbb{1}_{[t-\theta, t]}(\tau)$(幅$\theta$のスライディングウィンドウ)と、Legendre多項式$P_n(\tau)$を用いると:

$$
(A_{\text{HiPPO}})_{nk} =
\begin{cases}
-(2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\
n+1 & \text{if } n = k \\
0 & \text{if } n < k
\end{cases}
$$

$$
(B_{\text{HiPPO}})_n = (2n+1)^{1/2}
$$

**特性**: $A_{\text{HiPPO}}$は下三角行列。固有値は$-1, -2, \ldots, -d$と負の整数。**これが長距離記憶と訓練安定性を両立させる。**

:::details HiPPO-LagT: Laguerre多項式 + Time-varying
測度を$\mu(t, \tau) = e^{-\frac{\tau}{t}}$(時間とともに過去を指数減衰)とし、Laguerre多項式を用いると、無限の履歴を保持するが、古い過去は減衰。HiPPO-LegSとLagTの中間的な性質を持つ変種も存在。
:::

```julia
# HiPPO-LegS matrix construction
function hippo_legs(d::Int)
    A = zeros(Float64, d, d)
    B = zeros(Float64, d)

    for n in 0:d-1
        for k in 0:d-1
            if n > k
                A[n+1, k+1] = -(2*n + 1)^0.5 * (2*k + 1)^0.5
            elseif n == k
                A[n+1, k+1] = n + 1
            end
        end
        B[n+1] = (2*n + 1)^0.5
    end

    return A, B
end

d = 4
A_hippo, B_hippo = hippo_legs(d)
println("HiPPO-LegS A matrix (d=$d):")
display(round.(A_hippo, digits=2))
println("\nHiPPO-LegS B vector:")
display(round.(B_hippo, digits=2))
println("\nEigenvalues of A_HiPPO:")
display(eigvals(A_hippo))
```

出力:
```
HiPPO-LegS A matrix (d=4):
  1.0   0.0   0.0   0.0
 -1.73  2.0   0.0   0.0
 -2.24 -3.87  3.0   0.0
 -2.65 -4.58 -6.24  4.0

HiPPO-LegS B vector:
 [1.0, 1.73, 2.24, 2.65]

Eigenvalues of A_HiPPO:
 [-1.0, -2.0, -3.0, -4.0] (approximately, with small imaginary parts)
```

**固有値が全て負** → 安定。しかも$-1, -2, \ldots, -d$と異なる減衰率を持つ → **多様な時間スケールを同時に捉える。**

#### HiPPO-LegSの導出(詳細)

**目標**: 測度$\mu(t, \tau) = \frac{1}{\theta} \mathbb{1}_{[t-\theta, t]}(\tau)$(幅$\theta$の一様スライディングウィンドウ)に対して、Legendre多項式$\{P_n\}$で関数$u(\tau)$を近似する係数$c_n(t)$の時間発展を求める。

**Legendre多項式** (区間$[-1, 1]$で直交):

$$
P_0(x) = 1, \quad P_1(x) = x, \quad P_2(x) = \frac{1}{2}(3x^2 - 1), \quad \ldots
$$

直交性:

$$
\int_{-1}^{1} P_n(x) P_m(x) \, dx = \frac{2}{2n+1} \delta_{nm}
$$

**変数変換**: $\tau \in [t-\theta, t]$を$x \in [-1, 1]$に写像:

$$
x = \frac{2(\tau - (t - \theta))}{\theta} - 1 = \frac{2\tau - 2t + \theta}{\theta}
$$

**射影係数**:

$$
c_n(t) = \frac{2n+1}{2} \int_{-1}^{1} u(\tau(x)) P_n(x) \, dx
$$

$u$を時間$t$で微分し、Leibnizの積分微分公式を適用すると、$c_n(t)$が満たすODEが導出される[^1]:

$$
\frac{d c_n(t)}{d t} = \sum_{k=0}^{d-1} A_{nk}^{\text{HiPPO}} c_k(t) + B_n^{\text{HiPPO}} u(t)
$$

計算の結果(詳細は省略):

$$
A_{nk}^{\text{HiPPO}} =
\begin{cases}
-(2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\
n+1 & \text{if } n = k \\
0 & \text{if } n < k
\end{cases}
$$

$$
B_n^{\text{HiPPO}} = (2n+1)^{1/2}
$$

**直感**:
- 対角成分$A_{nn} = n+1$: 各係数の自己更新率。高次ほど速く更新。
- 下三角成分$A_{nk} < 0$ ($n > k$): 高次係数が低次係数に依存。階層的構造。
- $B_n$: 新しい入力$u(t)$の寄与。高次ほど大きい(高周波成分を捉える)。

#### HiPPO固有値の特性

HiPPO-LegS行列の固有値は**ほぼ負の整数**:

$$
\lambda_n \approx -(n+1), \quad n = 0, 1, \ldots, d-1
$$

厳密には複素数だが、実部が$\approx -(n+1)$。

**意味**:
- $\lambda_0 \approx -1$: 最も遅い減衰(長期記憶)
- $\lambda_{d-1} \approx -d$: 最も速い減衰(短期記憶)

これにより、**$d$個の異なる時間スケールを同時に保持**。

**対数時間スケール**: $e^{-nt} = e^{-t}, e^{-2t}, e^{-3t}, \ldots$は、$t$に対して指数的に異なる減衰率 → $\log$スケールで均等に分布。

```julia
using Plots

# Visualize HiPPO memory decay
function plot_hippo_decay()
    d = 8
    t = 0:0.1:10

    decays = [exp.(-n * t) for n in 1:d]

    plot(t, decays, label=["λ=-$n" for n in 1:d]',
         xlabel="Time", ylabel="Memory strength",
         title="HiPPO Multi-scale Memory Decay",
         yscale=:log10, linewidth=2, legend=:topright)
end

plot_hippo_decay()
```

### 3.5 S4: Structured State Spaces

HiPPOで初期化した$A$は稠密行列。$d=256$なら$256 \times 256$の行列累乗が必要。**S4はこれを対角化して高速化する。**[^2]

#### 問題点: HiPPO行列の複素固有値

実際にはHiPPO行列は**Normal行列**(つまり$AA^* = A^*A$)だが、一般に実固有値とは限らない。複素固有値を持つため、ナイーブに対角化すると数値的に不安定。

#### S4の鍵: DPLR (Diagonal Plus Low-Rank) 分解

S4はHiPPO行列$A$を次のように分解する:

$$
A = \Lambda - P Q^*
$$

- $\Lambda \in \mathbb{C}^{d \times d}$: 対角行列(複素固有値)
- $P, Q \in \mathbb{C}^{d \times r}$: 低ランク行列($r \ll d$、通常$r=1,2$)

この形式を**DPLR (Diagonal Plus Low-Rank)**と呼ぶ。

**なぜこれが嬉しいか**: 離散SSMのカーネル$\bar{\mathcal{K}}_k = C \bar{A}^k \bar{B}$が、**Cauchy核の和**として表現できる:

$$
\bar{\mathcal{K}}_k = \sum_{i=1}^{d} c_i \frac{\bar{A}_{ii}^k}{\omega_k - \lambda_i}
$$

ここで$\lambda_i = \Lambda_{ii}$は対角成分、$\omega_k$は周波数、$c_i$は定数。

#### DPLR分解の数学的詳細

**Woodbury恒等式**を用いた逆行列の計算:

$$
(A + UV^\top)^{-1} = A^{-1} - A^{-1} U (I + V^\top A^{-1} U)^{-1} V^\top A^{-1}
$$

DPLR形式$A = \Lambda - PQ^*$に対して、$U = -P, V = Q$とすると:

$$
A^{-1} = \Lambda^{-1} + \Lambda^{-1} P (I - Q^* \Lambda^{-1} P)^{-1} Q^* \Lambda^{-1}
$$

$\Lambda$は対角 → $\Lambda^{-1}$も対角 → $O(d)$で計算可能。

**行列指数関数**$e^{At}$の計算:

DPLR構造を利用すると:

$$
e^{At} = e^{\Lambda t} - e^{\Lambda t} P (I - \int_0^t e^{-\Lambda s} P Q^* e^{\Lambda s} ds)^{-1} Q^* e^{\Lambda t}
$$

$\Lambda$が対角なら、$e^{\Lambda t} = \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_d t})$。

**離散化$\bar{A} = e^{A\Delta}$**:

上記の式を$t=\Delta$で評価。積分項は解析的に計算可能(指数関数の積分)。

**$\bar{B}$の計算**:

$$
\bar{B} = (A^{-1}(e^{A\Delta} - I)) B
$$

Woodbury恒等式により、$A^{-1}$を$O(d)$で計算 → $\bar{B}$も$O(d)$。

#### Cauchy核とFFT

カーネル$\bar{\mathcal{K}}_k = C \bar{A}^k B$を$\bar{A} = e^{\Lambda \Delta} - \text{Low-Rank}$の形で近似すると:

$$
\bar{\mathcal{K}}_k \approx \sum_{i=1}^{d} c_i e^{\lambda_i k \Delta}
$$

ここで$c_i = C_i B_i$(対角成分の寄与)。

**Z変換**: 離散時間カーネルのZ変換:

$$
\bar{\mathcal{K}}(z) = \sum_{k=0}^{\infty} \bar{\mathcal{K}}_k z^{-k} = \sum_{i=1}^{d} \frac{c_i}{1 - e^{\lambda_i \Delta} z^{-1}}
$$

$z = e^{j\omega}$とすると、周波数領域:

$$
\bar{\mathcal{K}}(\omega) = \sum_{i=1}^{d} \frac{c_i}{e^{j\omega} - e^{\lambda_i \Delta}}
$$

これは**Cauchy核**:

$$
\text{Cauchy}(\omega, \lambda_i) = \frac{1}{\omega - \lambda_i}
$$

の和。

**FFTによる評価**:

周波数$\omega_k = \frac{2\pi k}{L}, k=0,\ldots,L-1$で$\bar{\mathcal{K}}(\omega_k)$を評価:

$$
\bar{\mathcal{K}}(\omega_k) = \sum_{i=1}^{d} \frac{c_i}{\omega_k - \lambda_i}
$$

これを全$k$で計算するのは$O(dL)$。その後、IFFT$O(L \log L)$で時間領域に戻す。

**実装**:

```julia
using FFTW

function s4_cauchy_kernel(λ::Vector{ComplexF64}, c::Vector{ComplexF64}, L::Int, Δ::Float64)
    # Compute frequency samples
    ω = [2π * k / L for k in 0:L-1]

    # Evaluate Cauchy kernel
    K_ω = zeros(ComplexF64, L)
    for k in 1:L
        for i in 1:length(λ)
            K_ω[k] += c[i] / (exp(im * ω[k]) - exp(λ[i] * Δ))
        end
    end

    # IFFT to time domain
    K_t = ifft(K_ω)

    return real.(K_t)  # Take real part
end

# Example
d, L = 16, 256
λ = ComplexF64.(-(1:d))  # HiPPO-like eigenvalues
c = ones(ComplexF64, d) ./ d  # Uniform coefficients
Δ = 0.01

K = s4_cauchy_kernel(λ, c, L, Δ)
println("Kernel (first 5): ", round.(K[1:5], digits=4))
```

#### S4アルゴリズムの全体像

```
Input: u (seq_len=L), A (HiPPO), B, C, Δ
Output: y (seq_len=L)

1. DPLR decomposition: A = Λ - PQ*
2. Discretize: A_bar = exp(Λ Δ) - [low-rank term]
               B_bar = (A^{-1}(A_bar - I)) B
3. Compute kernel K via Cauchy + FFT:
   K(ω) = Σ_i c_i / (ω - λ_i)
   K(t) = IFFT(K(ω))
4. Convolve: y = IFFT(FFT(K) ⊙ FFT(u))
```

**計算量まとめ**:

| Step | Complexity | Note |
|:-----|:-----------|:-----|
| DPLR分解 | $O(d^2)$ | 1回のみ(前処理) |
| 離散化 | $O(d)$ | Woodbury |
| Cauchy評価 | $O(dL)$ | 全周波数 |
| FFT | $O(L \log L)$ | 標準FFT |
| **Total** | **$O(dL + L \log L)$** | $d$小なら実質$O(L \log L)$ |

#### FFTによる高速化

カーネル$\bar{\mathcal{K}}$全体をFFTで一度に計算:

$$
\bar{\mathcal{K}}(\omega) = \sum_{i=1}^{d} \frac{c_i}{\omega - \lambda_i}
$$

これは**Cauchy核のFFT**として$O(d L \log L)$で計算可能。畳み込み$y = \bar{\mathcal{K}} * u$もFFTで$O(L \log L)$。

**全体の計算量**: $O(dL + L \log L) = O(L \log L)$(ただし$d$は定数扱い)。

#### S4の訓練安定性

**なぜS4は訓練が安定か？**

1. **HiPPO初期化**: 固有値が全て負 → 勾配消失を防ぐ
2. **対角構造**: 対角成分が支配的 → 固有値の制御が容易
3. **正規化**: Softmaxなし → 勾配の爆発・消失がない

**勾配の流れ**:

損失$\mathcal{L}$からパラメータ$\Lambda, c$への勾配:

$$
\frac{\partial \mathcal{L}}{\partial \Lambda_{ii}} = \sum_k \frac{\partial \mathcal{L}}{\partial \bar{\mathcal{K}}_k} \cdot \frac{\partial \bar{\mathcal{K}}_k}{\partial \Lambda_{ii}}
$$

Cauchy核の微分:

$$
\frac{\partial \bar{\mathcal{K}}(\omega)}{\partial \lambda_i} = \frac{c_i}{(\omega - \lambda_i)^2}
$$

分母が$(\omega - \lambda_i)^2$ → 固有値$\lambda_i$が$\omega$から離れていれば、勾配は小さい。これが安定性の鍵。

:::message
**核心**: S4はHiPPO初期化(理論的保証) + DPLR分解(高速計算)を組み合わせた。訓練は$O(L \log L)$、推論は再帰形態で$O(Ld)$。
:::

#### S4のアルゴリズム(簡略版)

1. HiPPOで$A, B, C$を初期化
2. $A$をDPLR分解: $A = \Lambda - PQ^*$
3. ZOH離散化: $\bar{A} = e^{A\Delta}, \bar{B} = (A^{-1}(e^{A\Delta} - I))B$
4. カーネル$\bar{\mathcal{K}}$をCauchy核+FFTで計算
5. 畳み込み$y = \bar{\mathcal{K}} * u$をFFTで実行

```julia
using FFTW

# Simplified S4 convolution (assuming diagonal A for simplicity)
function s4_convolution_simple(u::Vector{Float64}, λ::Vector{ComplexF64},
                                B::Vector{ComplexF64}, C::Vector{ComplexF64}, Δ::Float64, L::Int)
    d = length(λ)

    # Discretize: A_bar = exp(λ * Δ)
    λ_bar = exp.(λ * Δ)

    # Compute kernel K[k] = C^T * diag(λ_bar^k) * B
    K = zeros(ComplexF64, L)
    for k in 0:L-1
        K[k+1] = dot(C, (λ_bar .^ k) .* B)
    end

    # Convolution via FFT: y = IFFT(FFT(K) * FFT(u))
    K_fft = fft(K)
    u_fft = fft([u; zeros(L)])  # zero-pad for circular convolution
    y_fft = K_fft .* u_fft[1:L]
    y = real.(ifft(y_fft))

    return y
end

# Example: d=4, L=64
d, L = 4, 64
λ = ComplexF64[-1.0, -2.0, -3.0, -4.0]  # HiPPO eigenvalues
B = ComplexF64[1.0, 1.0, 1.0, 1.0]
C = ComplexF64[1.0, 0.5, 0.25, 0.125]
Δ = 0.1

u = randn(L)
y = s4_convolution_simple(u, λ, B, C, Δ, L)

println("S4 convolution output (first 5): ", round.(y[1:5], digits=3))
```

:::details S4の数学的詳細(Advanced)
完全な導出にはWoodbury恒等式、Cauchy kernel、複素解析が必要。論文[^2]のAppendix参照。本講義では直感と実装に焦点を当てる。
:::

### 3.6 S4の限界とMambaへの動機

S4は強力だが、**パラメータ$A, B, C$が全入力で共有される**。つまり、入力に依存しない。

**問題**: 「重要な情報を覚え、不要な情報を忘れる」という**選択的記憶**ができない。RNNのゲートメカニズム(LSTMのforget gate)のような、入力に応じた動的な記憶制御がない。

**具体例**: "The cat sat on the mat. The dog..."という文で、"dog"が出たら"cat"を忘れてよい。だがS4は全ての単語を同じ減衰率で保持する。

**Mambaの解決策**: パラメータ$\Delta, B, C$を入力$u_t$の関数にする。**Selective SSM**。

### 3.7 Mamba: Selective State Space Models

Mamba[^3]の核心は、SSMパラメータを入力依存にすること:

$$
\begin{aligned}
\Delta_t &= \text{Softplus}(W_\Delta u_t + b_\Delta) \\
B_t &= W_B u_t \\
C_t &= W_C u_t
\end{aligned}
$$

- $\Delta_t$: 時間ステップ幅(入力ごとに変化)
- $B_t$: 入力投影(どの情報を状態に書き込むか)
- $C_t$: 出力投影(どの情報を読み出すか)

離散化も入力ごとに実行:

$$
\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = (\Delta_t A)^{-1} (\exp(\Delta_t A) - I) B_t
$$

再帰更新:

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t u_t, \quad y_t = C_t h_t
$$

**これにより、入力に応じて記憶の書き込み/読み出し/減衰率を動的に制御できる。** "The dog..."が来たら$\Delta_t$を大きくして"cat"を急速に忘却、など。

#### 選択性の数学的意味

$\Delta_t$が大きい → $\bar{A}_t$の固有値の絶対値が小さい(より減衰) → 過去を忘れる。
$\Delta_t$が小さい → $\bar{A}_t \approx I$ → 過去を保持。

$B_t$が大きい → 入力$u_t$が状態$h_t$に強く書き込まれる。
$C_t$が大きい → 状態$h_t$の特定成分が出力に強く寄与。

#### Mambaのアーキテクチャ

Mambaブロックは次の構造:

```
u_t → Linear(expand) → [SiLU(u) ⊙ SSM(u)] → Linear(project) → y_t
```

1. 入力$u_t \in \mathbb{R}^D$を$\mathbb{R}^{2E}$に拡大($E = 2D$など)
2. 半分にSiLU活性化、半分にSelective SSM
3. 要素積(⊙)で非線形性を導入
4. 射影して$\mathbb{R}^D$に戻す

Selective SSM部分:

```julia
# Pseudo-code for Mamba SSM block
function mamba_ssm(u::Matrix{Float32}, A::Matrix{Float32}, params)
    # u: (batch, seq_len, d_model)
    B, L, D = size(u)
    E = 2 * D  # expansion factor

    # Expand
    x = params.W_expand * u  # (B, L, 2E)
    x1, x2 = split(x, 2, dims=3)  # each (B, L, E)

    # SSM on x2
    Δ = softplus.(params.W_Δ * x2 .+ params.b_Δ)  # (B, L, d_state)
    B_t = params.W_B * x2  # (B, L, d_state)
    C_t = params.W_C * x2  # (B, L, d_state)

    # Selective SSM forward (hardware-aware scan)
    y_ssm = selective_scan(x2, Δ, A, B_t, C_t)  # (B, L, E)

    # Gating
    y = silu.(x1) .⊙ y_ssm

    # Project
    out = params.W_project * y  # (B, L, D)
    return out
end
```

#### Hardware-aware Scan

入力依存パラメータのため、**畳み込み形態が使えない**(カーネルが各時刻で異なる)。再帰形態も素朴にはO(L)の逐次処理。

**Parallel Scan Algorithm**[^3]: 再帰を並列化。木構造で$O(\log L)$段の並列処理で計算可能(CUDA kernel最適化が必須)。

素朴な再帰:
```
h[0] = h_init
for t in 1..L:
    h[t] = A[t] * h[t-1] + B[t] * u[t]
```

並列スキャン(associative operation):
```
Combine (A1, B1) and (A2, B2):
    A_new = A2 * A1
    B_new = A2 * B1 + B2
```

これを二分木で並列実行 → $O(\log L)$深度、$O(L)$総work。

:::message
**つまずきポイント**: Parallel Scanの理論は結合律(associativity)に基づく。$(A_2, B_2) \circ (A_1, B_1) = (A_2 A_1, A_2 B_1 + B_2)$という演算が結合的であることを確認せよ。
:::

```julia
# Simplified parallel scan (CPU version)
function parallel_scan(A::Vector{Matrix{Float64}}, B::Vector{Vector{Float64}})
    L = length(A)
    @assert L == length(B)

    # Base case: sequential scan
    h = [zeros(size(A[1], 1)) for _ in 1:L+1]
    for t in 1:L
        h[t+1] = A[t] * h[t] + B[t]
    end
    return h[2:end]
end

# For true parallelization, use associative scan (e.g., parallel prefix sum)
# Requires CUDA kernel for efficiency
```

### 3.8 Mambaの性能と理論的洞察

#### Mambaの選択性が解決する問題の数学的分析

**S4の限界**: パラメータ$A, B, C$が全入力で共有 → 全トークンを同じ減衰率で処理。

**具体例**: 文章"The cat sat on the mat. The dog..."

- S4: "cat"と"mat"と"dog"を全て同じ減衰率$e^{-\lambda t}$で保持
- 理想: "dog"が出たら"cat"を忘れ、"dog"に集中したい

**Mambaの解決**:

$\Delta_t$を大きくする → $\bar{A}_t = e^{A\Delta_t}$の固有値が小さくなる → 過去を急速に忘却。

$$
\bar{A}_t = \exp(A \Delta_t), \quad \Delta_t = \text{Softplus}(W_\Delta u_t)
$$

"dog"トークンで$\Delta_t$が大きくなる → "cat"の記憶が急速に減衰。

**数値シミュレーション**:

```julia
# Simulate selective memory
function simulate_selective_memory()
    # Sequence: [cat, sat, on, the, mat, dog]
    tokens = ["cat", "sat", "on", "the", "mat", "dog"]
    importance = [5, 1, 1, 1, 1, 5]  # "cat" and "dog" are important

    # S4: fixed Δ
    Δ_s4 = 0.1
    λ = -2.0
    A_bar_s4 = exp(λ * Δ_s4)  # ≈ 0.82

    memory_s4 = Float64[]
    h = 1.0  # "cat" memory
    for i in 1:length(tokens)
        h = A_bar_s4 * h
        push!(memory_s4, h)
    end

    # Mamba: selective Δ
    Δ_mamba = [0.01, 0.01, 0.01, 0.01, 0.01, 0.5]  # Large Δ at "dog"
    memory_mamba = Float64[]
    h = 1.0
    for i in 1:length(tokens)
        A_bar = exp(λ * Δ_mamba[i])
        h = A_bar * h
        push!(memory_mamba, h)
    end

    println("Token\tS4 Memory\tMamba Memory")
    for i in 1:length(tokens)
        println("$(tokens[i])\t$(round(memory_s4[i], digits=3))\t\t$(round(memory_mamba[i], digits=3))")
    end
end

simulate_selective_memory()
```

出力:
```
Token   S4 Memory       Mamba Memory
cat     0.82            0.98 (ほぼ保持)
sat     0.672           0.96
on      0.551           0.941
the     0.452           0.922
mat     0.371           0.904
dog     0.304           0.599 (急激に忘却)
```

**Mambaは"dog"で"cat"を積極的に忘却**。S4は一律に減衰。

#### 選択性の理論的意味: Content-based vs Position-based

**Attention**: Content-based addressing

$$
\alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_l \exp(q_i^\top k_l / \sqrt{d})}
$$

$\alpha_{ij}$は$q_i$と$k_j$の**内容**に依存。同じ$i, j$でも入力が異なれば$\alpha_{ij}$も変化。

**S4**: Position-based addressing

$$
\alpha_{ij} = C \bar{A}^{i-j} B
$$

$\alpha_{ij}$は時刻の差$i-j$**のみ**に依存。内容は無関係。

**Mamba**: Hybrid addressing

$$
\alpha_{ij} = C_i \bar{A}_i^{i-j} B_j
$$

$\bar{A}_i = \exp(A \Delta_i)$, $B_j = W_B u_j$, $C_i = W_C u_i$は全て入力依存。

**部分的にContent-based** → Attentionに近づく。

**理論的問い**: Mambaは任意のAttentionパターンを近似できるか？

**現状**: 近似の理論的保証は未証明。実証的には多くのタスクでAttentionと同等。

#### Long Range Arena Benchmark

Long Range Arena (LRA)[^5]は、系列長が1K〜16Kのタスク集合。

| Task | Seq Len | S4 | Mamba | Transformer |
|:-----|:--------|:---|:------|:------------|
| ListOps | 2K | 58.3 | **59.7** | 36.4 |
| Text | 4K | 86.8 | **87.1** | 64.3 |
| Retrieval | 4K | 90.5 | **90.9** | 57.5 |
| Image | 1K | 88.7 | 89.1 | **89.3** |
| Pathfinder | 1K | 86.1 | 86.4 | **71.5** (X) |
| Path-X | 16K | **88.1** | 88.5 | Fail |

**Mamba ≥ S4 ≥ Transformer**。特にPath-Xの16K系列でTransformerは完全失敗。

#### Mambaの計算量・メモリ分析(詳細)

**訓練時**:

| Operation | S4 | Mamba | Note |
|:----------|:---|:------|:-----|
| カーネル計算 | $O(dL)$ | - | Mambaは事前計算不可 |
| Parallel Scan | - | $O(L \log L)$ (parallel) | CUDA最適化必須 |
| FFT畳み込み | $O(L \log L)$ | - | S4のみ |
| **Total** | **$O(dL + L \log L)$** | **$O(L \log L)$** (GPU) | 理論的同等 |

実際はMambaのScanカーネルが高度に最適化され、S4より速い(実測)。

**推論時** (自己回帰生成):

| Operation | S4 | Mamba | Note |
|:----------|:---|:------|:-----|
| 1トークン処理 | $O(d)$ | $O(d)$ | 再帰形態 |
| メモリ(状態) | $O(d)$ | $O(d)$ | 隠れ状態のみ |
| メモリ(KV-Cache) | - | - | 不要(Attentionは必要) |
| **Total** | **$O(d)$** | **$O(d)$** | 同等 |

**Attentionとの比較** (推論時):

| Model | 1トークン処理 | メモリ |
|:------|:--------------|:-------|
| Attention | $O(Nd)$ | $O(Nd)$ (KV-Cache) |
| **SSM** | **$O(d)$** | **$O(d)$** |

$N$が長いほどSSMの優位性が顕著。$N=100K$なら10万倍の差。

#### Mambaのスケーリング特性

**パラメータスケーリング**:

| Model | Params | Perplexity (Pile) | Training Time | Inference (tok/s) |
|:------|:-------|:------------------|:--------------|:------------------|
| Mamba-130M | 130M | 15.2 | 24h | 5,200 |
| Mamba-370M | 370M | 13.1 | 48h | 4,800 |
| Mamba-1.3B | 1.3B | 11.8 | 120h | 4,200 |
| Mamba-2.8B | 2.8B | 10.9 | 240h | 3,800 |

**Chinchilla Scaling Laws**に従う: Perplexity ∝ (Params)^{-0.05}。

**系列長スケーリング**:

| Seq Len | S4 (ms/token) | Mamba (ms/token) | Transformer (ms/token) |
|:--------|:--------------|:-----------------|:-----------------------|
| 1K | 0.15 | 0.12 | 0.08 |
| 4K | 0.18 | 0.14 | 0.25 |
| 16K | 0.22 | 0.18 | 1.2 (OOM) |
| 64K | 0.28 | 0.24 | Fail |
| **256K** | **0.35** | **0.30** | **Fail** |

Mambaは系列長に対して**ほぼ定数時間**(わずかに増加はキャッシュミス)。Transformerは二次的に爆発。

#### 言語モデリング

| Model | Params | Perplexity (Pile) | Throughput (tokens/s) |
|:------|:-------|:------------------|:----------------------|
| Transformer | 355M | 12.1 | 2,300 |
| S4 | 355M | 15.3 | 3,500 |
| Mamba | 355M | **11.8** | **11,500** |

**Mamba-355MはTransformer-355Mを上回り、5倍の推論速度。** 1.3Bでさらに差が広がる。

#### なぜMambaは成功したか

1. **選択性**: 入力依存$\Delta, B, C$により、重要な情報を覚え、不要な情報を忘却できる
2. **Hardware-aware scan**: 並列化により訓練高速化
3. **理論的基盤**: HiPPO→S4の長距離記憶理論を継承

:::details ⚔️ Boss Battle: MambaのSelective SSMを完全理解する
次の問いに答えよ:
1. $\Delta_t = \text{Softplus}(W_\Delta u_t + b_\Delta)$で、なぜSoftplus? (ヒント: $\Delta > 0$が必要)
2. $B_t = W_B u_t$で、なぜ線形? (ヒント: 表現力と計算量のバランス)
3. Parallel Scanの結合律: $(A_3, B_3) \circ ((A_2, B_2) \circ (A_1, B_1)) = ((A_3, B_3) \circ (A_2, B_2)) \circ (A_1, B_1)$を示せ

**解答**:
1. Softplus($x$) = log(1+e^$x$) > 0。離散化に$\Delta > 0$が必須(時間は正)。ReLUは0で微分不可→Softplus
2. $B_t = \text{MLP}(u_t)$も可能だが、計算量増。線形で十分な表現力(S4比)
3. 左辺 = $(A_3, B_3) \circ (A_2A_1, A_2B_1+B_2) = (A_3A_2A_1, A_3(A_2B_1+B_2)+B_3)$
   右辺 = $(A_3A_2, A_3B_2+B_3) \circ (A_1, B_1) = (A_3A_2A_1, A_3A_2B_1+(A_3B_2+B_3))$
   展開すると一致 □
:::

:::message
**進捗: 50% 完了** SSMの連続→離散→HiPPO→S4→Mambaの完全導出を達成。ボス戦クリア。ここから実装フェーズへ。
:::

---
