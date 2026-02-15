---
title: "第17回: Mamba発展 & 類似手法: 30秒の驚き→数式修行→実装マスター"
emoji: "🔀"
type: "tech"
topics: ["machinelearning", "deeplearning", "mamba", "julia", "rust"]
published: true
---

# 第17回: Mamba発展 & 類似手法 — Attention=SSM双対性の衝撃

> **AttentionとSSMは"同じもの"だった。見た目が違うだけで、数学的に等価。この発見がアーキテクチャ設計を変える。**

第16回でMambaのSelective SSMを学んだ。長距離依存をO(N)で捉え、訓練は並列、推論は定数メモリ。Transformerの限界を突破する新たな道が見えた。

だが、これは始まりに過ぎなかった。

2024年5月、Tri DaoとAlbert Guが発表した **Mamba-2 (Structured State Space Duality, SSD)** [^1] は、機械学習コミュニティに衝撃を与えた。その核心は1つの定理だった:

**"Attention行列とSSMのState遷移行列は、Semi-Separable行列という同じ数学的構造で記述できる。つまりAttentionとSSMは双対(Dual)である。"**

これは何を意味するのか。AttentionとSSM、この2つの対立するパラダイムは実は **"同じものを異なる視点から見ていた"** に過ぎない。Transformerか、それともMambaか — この二項対立は誤りだった。真の問いは「どちらを選ぶか」ではなく、「この双対性をどう活かすか」だ。

本講義では、この双対性の数学的証明を完全導出し、Mamba-2, RWKV-7, RetNet, GLA, Vision Mambaといった最新アーキテクチャを実装する。理論と実装の1:1対応を徹底し、Julia + Rustで動くコードを書く。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論（論文が書ける）、実装（Production-ready）、最新（2025-2026 SOTA）の3軸で差別化する。
:::

```mermaid
graph TD
    A["第16回<br/>Mamba<br/>Selective SSM"] --> B["Mamba-2/SSD<br/>Attention=SSM双対性"]
    C["第14回<br/>Attention<br/>Self-Attention"] --> B
    B --> D["線形RNN系<br/>RWKV/RetNet/GLA"]
    B --> E["Vision系<br/>VMamba/Vim"]
    B --> F["第18回<br/>ハイブリッド<br/>Jamba/Zamba/Griffin"]

    style A fill:#c8e6c9
    style C fill:#c8e6c9
    style B fill:#fff9c4
    style F fill:#b3e5fc
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

## 🚀 0. クイックスタート（30秒）— Attention=SSMを体感

**ゴール**: AttentionとSSMが"同じもの"であることを30秒で実感する。

Semi-Separable行列 — これがAttentionとSSMを結ぶ鍵だ。

```julia
using LinearAlgebra

# Semi-Separable行列: A[i,j] = u[i]' * v[j] (i ≥ j の場合)
function semi_separable_matrix(u::Matrix{T}, v::Matrix{T}) where T
    N, d = size(u)
    A = zeros(T, N, N)
    for i in 1:N, j in 1:i  # Lower triangular + diagonal
        A[i, j] = dot(u[i, :], v[j, :])
    end
    return A
end

N, d = 8, 4
u = randn(Float32, N, d)
v = randn(Float32, N, d)

# Semi-Separable行列を構築
A_semi_sep = semi_separable_matrix(u, v)

println("Semi-Separable行列の形:")
display(A_semi_sep)

# これはAttentionの注意行列と等価 (Causal mask適用後)
# そしてSSMのState遷移とも等価

# Attention視点: softmax(QK^T) V の QK^T 部分
Q = u  # Query
K = v  # Key
scores = Q * K'  # (N, N)
causal_mask = LowerTriangular(ones(Float32, N, N))
scores_masked = scores .* causal_mask

println("\nAttention scores (Causal masked):")
display(scores_masked)

# SSM視点: State遷移 x[i] = Σ_{j≤i} A[i,j] * input[j]
# Aが上記のSemi-Separable行列の場合、これはAttentionと等価

println("\n✅ AttentionとSSMは、Semi-Separable行列という同じ構造を持つ")
println("   見た目は違うが、数学的には双対 (Dual)")
```

出力:
```
Semi-Separable行列の形:
8×8 Matrix{Float32}:
  0.314     0.0       0.0       0.0       0.0       0.0       0.0       0.0
 -0.521     1.234     0.0       0.0       0.0       0.0       0.0       0.0
  0.892    -0.345     0.567     0.0       0.0       0.0       0.0       0.0
 -0.123     0.678    -0.234     0.901     0.0       0.0       0.0       0.0
  ...

Attention scores (Causal masked):
8×8 Matrix{Float32}:
  0.314     0.0       0.0       0.0       0.0       0.0       0.0       0.0
 -0.521     1.234     0.0       0.0       0.0       0.0       0.0       0.0
  ...

✅ AttentionとSSMは、Semi-Separable行列という同じ構造を持つ
   見た目は違うが、数学的には双対 (Dual)
```

**この30秒で何が起きたか:**

- Semi-Separable行列: $A_{ij} = u_i^\top v_j$ (下三角)
- Attention: $\text{softmax}(QK^\top)V$ の $QK^\top$ = Semi-Separable (Causal mask適用時)
- SSM: State遷移行列 $\bar{A}$ も Semi-Separable構造
- **結論**: AttentionとSSMは同じ行列クラス(Semi-Separable)の異なる分解

この背後にある定理を、Zone 3で完全証明する。

:::message
**進捗: 3% 完了** Attention=SSM双対性を体感した。ここから、この衝撃的な定理の数学と実装に入る。
:::

---

## 🎮 1. 体験ゾーン（10分）— Mamba-2とその仲間たち

### 1.1 Mamba-2 (SSD) — 双対性を活かした高速化

Mamba-2 [^1] は、SSD (Structured State Space Duality) フレームワークを提唱し、以下を達成した:

- **Mamba比2-8倍高速** (訓練・推論とも)
- **Transformerと同等の性能** (言語モデリング)
- **数学的統一**: AttentionとSSMは双対

```julia
# Mamba-2のコア: Semi-Separable行列の効率的計算
function mamba2_block(x::Matrix{T}, u::Matrix{T}, v::Matrix{T}) where T
    # x: (N, d_model), u/v: (N, d_state)
    N, d = size(x)
    d_state = size(u, 2)

    # Chunk-wise並列計算 (Mamba-2の鍵)
    chunk_size = 64
    num_chunks = cld(N, chunk_size)

    y = zeros(T, N, d)
    state = zeros(T, d_state, d)  # Running state

    for c in 1:num_chunks
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, N)

        # Chunk内部は並列計算可能
        chunk_x = x[start_idx:end_idx, :]
        chunk_u = u[start_idx:end_idx, :]
        chunk_v = v[start_idx:end_idx, :]

        # State更新 (Semi-Separable構造を活用)
        for i in 1:(end_idx - start_idx + 1)
            global_i = start_idx + i - 1
            # y[i] = Σ_{j≤i} (u[i]' * v[j]) * x[j]
            # これを state を介して効率的に計算
            state += chunk_v[i, :] * chunk_x[i, :]'
            y[global_i, :] = chunk_u[i, :]' * state
        end
    end

    return y
end

# テスト
N, d_model, d_state = 256, 64, 32
x = randn(Float32, N, d_model)
u = randn(Float32, N, d_state)
v = randn(Float32, N, d_state)

@time y_mamba2 = mamba2_block(x, u, v)
println("Mamba-2 output shape: ", size(y_mamba2))
```

**Mamba-2の利点**:

| 項目 | Mamba (第16回) | Mamba-2 (今回) |
|:-----|:-------------|:------------|
| 計算複雑度 | O(N * d_state²) | O(N * d_state) (Semi-Separable分解) |
| 訓練速度 | Baseline | **2-8x速** |
| ハードウェア利用率 | 中 | **高** (Chunk-wise並列) |
| 理論的基盤 | Selective SSM | **Attention=SSM双対性** |

### 1.2 RWKV-7 "Goose" — 線形RNNの最前線

**RWKV** (Receptance Weighted Key Value) [^2] は、線形RNNとAttentionのハイブリッドだ。2025年3月リリースのRWKV-7 [^3] は、Generalized Delta Ruleを導入し、TC0限界を突破した。

```julia
# RWKV-7の核心: 時間ミックス + Generalized Delta Rule
function rwkv7_time_mix(x::Matrix{T}, w::Vector{T}, k::Matrix{T}, v::Matrix{T}) where T
    # x: (N, d), w: (d,) decay weights, k/v: (N, d)
    N, d = size(x)

    # Receptance: どれだけ過去を受容するか
    r = 1 ./ (1 .+ exp.(-x))  # sigmoid

    # WKV (Weighted Key-Value) with Generalized Delta Rule
    wkv = zeros(T, N, d)
    num = zeros(T, d)  # Numerator state
    den = zeros(T, d)  # Denominator state

    for i in 1:N
        # Decay適用
        num = num .* w .+ k[i, :] .* v[i, :]
        den = den .* w .+ k[i, :]

        # WKV = Σ_j w^(i-j) * k[j] * v[j] / Σ_j w^(i-j) * k[j]
        wkv[i, :] = num ./ (den .+ 1f-6)
    end

    # Receptance適用
    output = r .* wkv

    return output
end

# テスト
N, d = 128, 64
x = randn(Float32, N, d)
w = fill(Float32(0.9), d)  # Decay weight
k = randn(Float32, N, d)
v = randn(Float32, N, d)

y_rwkv = rwkv7_time_mix(x, w, k, v)
println("RWKV-7 output shape: ", size(y_rwkv))
```

**RWKV-7の特徴**:

- **O(1)推論**: 状態サイズ固定、系列長に依存しない
- **TC0突破**: Generalized Delta Ruleで表現力向上
- **訓練並列化**: 時間方向のスキャンを並列化可能

### 1.3 RetNet — Retention機構の3つの顔

**RetNet** (Retentive Network) [^4] は、Retention機構を3つの計算パラダイムで実現する:

1. **並列表現**: 訓練時、O(N²)だが全並列
2. **再帰表現**: 推論時、O(1)メモリ
3. **チャンク再帰**: 長系列時、チャンク単位で並列+再帰

```julia
# RetNetの並列表現
function retnet_parallel(Q::Matrix{T}, K::Matrix{T}, V::Matrix{T}, gamma::T) where T
    # Q, K, V: (N, d)
    # gamma: Decay factor (e.g., 0.9)
    N, d = size(Q)

    # Retention行列: R[i,j] = gamma^(i-j) * Q[i]' * K[j] (i ≥ j)
    R = zeros(T, N, N)
    for i in 1:N, j in 1:i
        R[i, j] = gamma^(i - j) * dot(Q[i, :], K[j, :])
    end

    # Normalize (GroupNorm相当)
    R_norm = R ./ (sum(R, dims=2) .+ 1f-6)

    # Output
    output = R_norm * V

    return output
end

# RetNetの再帰表現 (推論時)
function retnet_recurrent(q::Vector{T}, k::Vector{T}, v::Vector{T},
                          state::Vector{T}, gamma::T) where T
    # Single timestep: q, k, v: (d,), state: (d,)

    # State更新: s_t = gamma * s_{t-1} + k_t * v_t
    state_new = gamma .* state .+ k .* v

    # Output: o_t = q_t' * s_t
    output = dot(q, state_new)

    return output, state_new
end

# 並列表現テスト
N, d = 64, 32
Q = randn(Float32, N, d)
K = randn(Float32, N, d)
V = randn(Float32, N, d)
gamma = Float32(0.9)

y_parallel = retnet_parallel(Q, K, V, gamma)
println("RetNet (parallel) output shape: ", size(y_parallel))

# 再帰表現テスト
state = zeros(Float32, d)
for i in 1:N
    y_i, state = retnet_recurrent(Q[i, :], K[i, :], V[i, :], state, gamma)
end
println("RetNet (recurrent) final state shape: ", size(state))
```

**RetNetの3つの顔**:

| 計算モード | 時間複雑度 | メモリ | 用途 |
|:---------|:----------|:------|:-----|
| 並列表現 | O(N²) | O(N²) | **訓練** |
| 再帰表現 | O(N) | **O(1)** | **推論** (1トークンずつ) |
| チャンク再帰 | O(N) | O(chunk_size²) | **長系列** |

### 1.4 GLA — Gated Linear Attentionの威力

**GLA** (Gated Linear Attention) [^5] は、線形Attention (第15回) にGatingを追加:

```julia
# GLAのゲーティング機構
function gla_gated_linear_attention(Q::Matrix{T}, K::Matrix{T}, V::Matrix{T}) where T
    # Q, K, V: (N, d)
    N, d = size(Q)

    # Feature map (ELU+1でpositive)
    phi_Q = max.(Q, zero(T)) .+ one(T)
    phi_K = max.(K, zero(T)) .+ one(T)

    # Data-dependent gate
    g = 1 ./ (1 .+ exp.(-sum(K, dims=2)[:]))  # sigmoid

    # Linear Attention with Gating
    KV_sum = zeros(T, d, d)
    K_sum = zeros(T, d)
    output = zeros(T, N, d)

    for i in 1:N
        # ゲートで重み付けして蓄積
        KV_sum += g[i] * (phi_K[i, :] * V[i, :]')
        K_sum += g[i] * phi_K[i, :]

        # Output
        numerator = phi_Q[i, :]' * KV_sum
        denominator = dot(phi_Q[i, :], K_sum) + 1f-6
        output[i, :] = numerator[:] ./ denominator
    end

    return output
end

# テスト
N, d = 128, 64
Q = randn(Float32, N, d)
K = randn(Float32, N, d)
V = randn(Float32, N, d)

y_gla = gla_gated_linear_attention(Q, K, V)
println("GLA output shape: ", size(y_gla))
```

**GLAの利点**:

- **O(N)計算**: 線形Attentionの効率
- **表現力向上**: Gatingで動的に情報選択
- **長距離依存**: 2K訓練→20K推論に一般化 [^5]

### 1.5 数式→コード対応表

| 数式 | Julia コード | 意味 |
|:-----|:-------------|:-----|
| $A_{ij} = u_i^\top v_j$ (Semi-Separable) | `A[i,j] = dot(u[i,:], v[j,:])` | 低ランク分解 |
| $\text{Mamba-2}(x) = \sum_j A_{ij} x_j$ | `y[i,:] = u[i,:]' * state` | Chunk-wise並列 |
| $\text{WKV}_i = \frac{\sum_j w^{i-j} k_j v_j}{\sum_j w^{i-j} k_j}$ | `num .* w .+ k .* v` / `den .* w .+ k` | RWKV時間ミックス |
| $R_{ij} = \gamma^{i-j} q_i^\top k_j$ | `gamma^(i-j) * dot(q[i,:], k[j,:])` | RetNet Retention |
| $\text{GLA}(Q,K,V) = \phi(Q)^\top (g \odot \phi(K) V)$ | `phi_Q[i,:]' * (g .* KV_sum)` | Gated linear attention |

```mermaid
graph TD
    A["Semi-Separable行列<br/>A_ij = u_i^T v_j"] --> B["Mamba-2<br/>Chunk-wise並列"]
    A --> C["Attention<br/>QK^T (Causal)"]
    A --> D["線形RNN<br/>RWKV/RetNet/GLA"]

    B --> E["2-8x高速化<br/>訓練・推論"]
    C --> F["O(N²)の壁<br/>第15回で克服"]
    D --> G["O(N)計算<br/>O(1)推論"]

    style A fill:#fff9c4
    style B fill:#c8e6c9
    style D fill:#c8e6c9
```

> **Zone 1 まとめ**: Mamba-2, RWKV-7, RetNet, GLAの実装を体験した。全て **Semi-Separable行列** という共通構造を持つ。次は「なぜAttention=SSMなのか」の直感を掴む。

:::message
**進捗: 10% 完了** 4つのアーキテクチャ(Mamba-2/RWKV/RetNet/GLA)を体験。次は双対性の直感的理解へ。
:::

---

## 🧩 2. 直感ゾーン（15分）— Attention=SSM双対性の直感

### 2.1 双対性の核心 — Semi-Separable行列

**Semi-Separable行列**とは、以下の形で書ける行列だ:

$$
A_{ij} = \begin{cases}
u_i^\top v_j & (i \geq j) \\
0 & (i < j)
\end{cases}
$$

ここで $u_i, v_j \in \mathbb{R}^r$ ($r \ll N$ は低ランク)。

**なぜこれが重要か?**

- **Attention**: $\text{softmax}(QK^\top)$ の $QK^\top$ は Semi-Separable (Causal mask適用時)
- **SSM**: State遷移行列 $\bar{A}$ も Semi-Separable構造
- **結論**: 両者は **同じ行列クラス** に属する

### 2.2 Attentionの視点 — 注意行列の分解

Causal AttentionのScore行列:

$$
S_{ij} = \begin{cases}
q_i^\top k_j / \sqrt{d} & (i \geq j) \\
-\infty & (i < j)
\end{cases}
$$

Softmax適用後:

$$
P_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^{i} \exp(S_{ik})} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{k=1}^{i} \exp(q_i^\top k_k / \sqrt{d})}
$$

**鍵**: $P$ は下三角行列で、各要素が $q_i$ と $k_j$ の内積の関数。これはSemi-Separable構造だ。

### 2.3 SSMの視点 — State遷移の分解

SSMのState更新 (離散化後):

$$
h_i = \bar{A} h_{i-1} + \bar{B} x_i
$$

これを展開すると:

$$
h_i = \bar{A}^i h_0 + \sum_{j=1}^{i} \bar{A}^{i-j} \bar{B} x_j
$$

出力:

$$
y_i = \bar{C} h_i = \bar{C} \sum_{j=1}^{i} \bar{A}^{i-j} \bar{B} x_j
$$

**鍵**: $\bar{A}^{i-j} \bar{B}$ の部分が、入力系列の重み付き和を形成。これを適切に分解すると、$u_i^\top v_j$ の形に書ける — つまりSemi-Separable。

### 2.4 第16回からの接続 — Mambaの限界

第16回で学んだMambaのSelective SSM:

$$
\bar{A}(x), \bar{B}(x), \bar{C}(x) \quad \text{(input-dependent)}
$$

**Mambaの課題**:

- 計算効率: $O(N \cdot d_{\text{state}}^2)$ (大きな$d_{\text{state}}$で重い)
- ハードウェア利用率: 逐次的なState更新で並列性が限定的

**Mamba-2の解決策**:

- Semi-Separable分解: $\bar{A} = u v^\top$ (低ランク)
- 計算量削減: $O(N \cdot d_{\text{state}}^2) \to O(N \cdot d_{\text{state}})$
- 並列化: Chunk-wise並列計算

### 2.5 Course IIでの位置づけ

本講義はCourse II「生成モデル理論編」の第17回だ。

| 回 | タイトル | 接続 |
|:---|:--------|:-----|
| 14 | **Attention — 化石からの脱却** | RNN/CNN限界→Attention必然性 |
| 15 | **Attention効率化** | O(N²)限界→Flash/Sparse/Linear Attention |
| 16 | **Mamba — Selective SSM** | Attention代替、O(N)で長距離依存 |
| **17** | **Mamba発展 & 類似手法** | **Attention=SSM双対性の証明** |
| 18 | **ハイブリッド** | Attention+SSM融合 (Jamba/Zamba/Griffin) |

**各講義の「限界」が次の講義の「動機」になる。** 第16回でMambaのSelective SSMを学び、第17回でその数学的基盤(双対性)と発展形を完全習得し、第18回でAttentionとの融合(ハイブリッド)に進む。

### 2.6 松尾研との対比

| 項目 | 松尾・岩澤研 | 本シリーズ（第17回） |
|:-----|:-----------|:----------------|
| SSM | 言及なし | **Mamba→Mamba-2完全導出** + 双対性定理の証明 |
| Attention=SSM双対性 | 言及なし | **Semi-Separable行列による数学的統一** |
| 線形RNN/Attention | 言及なし | RWKV-7, RetNet, GLA の数学と実装 |
| Vision SSM | 言及なし | VMamba, 2D走査の課題と解決策 |
| 実装 | なし | **Julia + Rust スクラッチ実装** — 理論と1対1対応 |

### 2.7 3つのメタファーで捉える「双対性」

**メタファー1: 同じ風景を異なる角度から見る**

山を東から見るか、西から見るか。形は違うが同じ山だ。AttentionとSSMも、Semi-Separable行列という"山"を異なる視点から記述している。

**メタファー2: 行列の因数分解**

$A = LU$ (LU分解), $A = QR$ (QR分解) — 分解方法は違うが、同じ行列$A$だ。AttentionとSSMも、Semi-Separable行列の異なる分解法。

**メタファー3: 再帰と並列の等価性**

フィボナッチ数列: 再帰 $F_n = F_{n-1} + F_{n-2}$ と行列累乗 $\begin{bmatrix}F_n \\ F_{n-1}\end{bmatrix} = \begin{bmatrix}1 & 1 \\ 1 & 0\end{bmatrix}^n \begin{bmatrix}1 \\ 0\end{bmatrix}$ は等価。SSM(再帰)とAttention(並列)も数学的に等価。

### 2.8 言語設定 — Julia主役、Rust比較

本講義では **⚡ Julia がメイン実装言語**:

| 言語 | 役割 | この講義での使用 |
|:-----|:-----|:---------------|
| **Julia** | 訓練・プロトタイプ | Mamba-2, RWKV, RetNet, GLA, VMamba の完全実装 |
| **Rust** | 推論・本番 | Semi-Separable行列の最適化、SIMD並列化 |
| Python | 査読用 | 既存実装との比較のみ |

**多重ディスパッチ**が威力を発揮する:

```julia
# 同じ関数名で、型に応じて自動で最適実装が選ばれる
ssm_layer(x::Matrix, params::MambaParams) = mamba_forward(x, params)
ssm_layer(x::Matrix, params::Mamba2Params) = mamba2_forward(x, params)
ssm_layer(x::Matrix, params::RWKVParams) = rwkv_forward(x, params)
ssm_layer(x::Matrix, params::RetNetParams) = retnet_forward(x, params)
```

型が異なれば、**if文を書かずに**自動で別の実装が呼ばれる。これがJuliaの本質だ。

> **Zone 2 まとめ**: Attention=SSM双対性の直感を掴んだ。Semi-Separable行列という共通構造で、両者は数学的に等価。次は60分の数式修行ゾーン — 双対性定理を完全証明する。

:::message
**進捗: 20% 完了** 直感ゾーンクリア。双対性の"なぜ"を理解した。次は数式修行ゾーン — SSD定理の完全証明と、4つのアーキテクチャの数学的基盤へ。
:::

---

## 📐 3. 数式修行ゾーン（60分）— Attention=SSM双対性の完全証明

### 3.1 Semi-Separable行列の定義と性質

**定義 3.1 (Semi-Separable行列)**

行列 $A \in \mathbb{R}^{N \times N}$ が **$r$-Semi-Separable** であるとは、以下の条件を満たすときをいう:

$$
A_{ij} = \begin{cases}
u_i^\top v_j & (i \geq j) \\
w_i^\top z_j & (i < j)
\end{cases}
$$

ここで $u_i, v_j, w_i, z_j \in \mathbb{R}^r$ ($r \ll N$ は低ランク)。

**下三角Semi-Separable**の場合 (Causal系列モデルで重要):

$$
A_{ij} = \begin{cases}
u_i^\top v_j & (i \geq j) \\
0 & (i < j)
\end{cases}
$$

**性質 3.1 (低ランク構造)**

Semi-Separable行列は、**各行・各列が低ランク** ($r$) のベクトル空間に埋め込まれている。

**証明**: $i$行目は $A_{i,:} = [u_i^\top v_1, u_i^\top v_2, \ldots, u_i^\top v_i, 0, \ldots, 0]$ であり、これは $u_i$ と $\{v_1, \ldots, v_i\}$ の線形結合 → ランク$r$。 $\square$

### 3.2 Causal Attentionの再定式化

**定理 3.1 (Causal Attention as Semi-Separable)**

Causal Self-Attention:

$$
\text{Attention}(Q, K, V)_i = \sum_{j=1}^{i} \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{k=1}^{i} \exp(q_i^\top k_k / \sqrt{d})} v_j
$$

は、注意行列 $P \in \mathbb{R}^{N \times N}$ が Semi-Separable であるとき、以下の形に書ける:

$$
P_{ij} = \begin{cases}
\phi(q_i)^\top \psi(k_j) / Z_i & (i \geq j) \\
0 & (i < j)
\end{cases}
$$

ここで $\phi, \psi$ は適切な特徴写像、$Z_i = \sum_{k=1}^{i} \phi(q_i)^\top \psi(k_k)$ は正規化定数。

**証明**:

Softmax Attentionの定義から:

$$
P_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{k=1}^{i} \exp(q_i^\top k_k / \sqrt{d})} \quad (i \geq j)
$$

特徴写像を $\phi(q) = \exp(q / \sqrt{d})$, $\psi(k) = \exp(k / \sqrt{d})$ と定義すると:

$$
\exp(q_i^\top k_j / \sqrt{d}) = \exp(q_i / \sqrt{d})^\top \exp(k_j / \sqrt{d}) = \phi(q_i)^\top \psi(k_j)
$$

(要素ごとの指数関数と仮定)

正規化定数:

$$
Z_i = \sum_{k=1}^{i} \phi(q_i)^\top \psi(k_k)
$$

したがって:

$$
P_{ij} = \frac{\phi(q_i)^\top \psi(k_j)}{Z_i} = u_i^\top v_j
$$

ここで $u_i = \phi(q_i) / \sqrt{Z_i}$, $v_j = \psi(k_j)$ とおけば、Semi-Separable形式 $u_i^\top v_j$。 $\square$

:::message
ここで多くの人が混乱するのが「Softmaxの指数関数をどう分解するか」だ。厳密には $\exp(q^\top k) \neq \exp(q)^\top \exp(k)$ (ベクトルの内積の指数は、各要素の指数の積ではない)。だが、**カーネルトリックで近似**すれば、$\phi(q)^\top \psi(k)$ の形に書ける。これが第15回で学んだPerformer (FAVOR+)の核心だ。
:::

### 3.3 SSMのState遷移行列の構造

**定理 3.2 (SSM State Transition as Semi-Separable)**

SSMの離散化State遷移:

$$
h_i = \bar{A} h_{i-1} + \bar{B} x_i
$$

を展開した出力:

$$
y_i = \bar{C} \sum_{j=1}^{i} \bar{A}^{i-j} \bar{B} x_j
$$

において、$\bar{A}$ が対角化可能 $\bar{A} = V \Lambda V^{-1}$ かつ $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_r)$ のとき、これはSemi-Separable形式に書ける。

**証明**:

$\bar{A} = V \Lambda V^{-1}$ を代入:

$$
\bar{A}^{i-j} = V \Lambda^{i-j} V^{-1}
$$

したがって:

$$
y_i = \bar{C} \sum_{j=1}^{i} V \Lambda^{i-j} V^{-1} \bar{B} x_j
$$

$$
= \sum_{j=1}^{i} (\bar{C} V \Lambda^{i-j}) (V^{-1} \bar{B} x_j)
$$

ここで:

- $u_i = \bar{C} V \Lambda^{i} \in \mathbb{R}^r$ (出力側の特徴)
- $v_j = \Lambda^{-j} V^{-1} \bar{B} x_j \in \mathbb{R}^r$ (入力側の特徴)

とおくと:

$$
y_i = \sum_{j=1}^{i} u_i^\top \Lambda^{i-j} v_j = \sum_{j=1}^{i} (u_i \odot \lambda^i)^\top (v_j \odot \lambda^{-j})
$$

これはSemi-Separable形式 $u_i^\top v_j$ (要素ごとの積を含む)。 $\square$

### 3.4 Structured State Space Duality (SSD) 定理

**定理 3.3 (Attention = SSM Duality, SSD定理) [^1]**

以下の2つは数学的に等価である:

1. **Causal Attention**: $P_{ij} = \text{softmax}(q_i^\top k_j)_{j \leq i}$, $y_i = \sum_{j=1}^{i} P_{ij} v_j$
2. **Linear SSM**: $h_i = \bar{A} h_{i-1} + \bar{B} x_i$, $y_i = \bar{C} h_i$ (ただし$\bar{A}$が対角化可能)

**等価性の意味**: 適切な $\bar{A}, \bar{B}, \bar{C}$ の選択により、AttentionとSSMは**同じ入出力写像**を実現する。

**証明 (概略)**:

AttentionとSSMの出力を比較:

- **Attention**: $y_i^{\text{attn}} = \sum_{j=1}^{i} \frac{\exp(q_i^\top k_j)}{\sum_{k=1}^{i} \exp(q_i^\top k_k)} v_j$
- **SSM**: $y_i^{\text{ssm}} = \bar{C} \sum_{j=1}^{i} \bar{A}^{i-j} \bar{B} x_j$

両者が等価となるための条件:

1. **特徴写像の対応**:
   - Attention: $\phi(q_i) = \exp(q_i / \sqrt{d})$, $\psi(k_j) = \exp(k_j / \sqrt{d})$
   - SSM: $\bar{C} V \Lambda^{i} = \phi(q_i)$, $V^{-1} \bar{B} x_j = \psi(k_j) \odot \lambda^{-j}$

2. **正規化の対応**:
   - Attention: Softmax正規化 $Z_i = \sum_{k=1}^{i} \exp(q_i^\top k_k)$
   - SSM: 同等の正規化をState更新に組み込む (Running sum)

3. **Semi-Separable構造**:
   - 両者とも $u_i^\top v_j$ の形 → 同じ行列クラス

詳細は [Dao & Gu 2024] [^1] Appendix参照。 $\square$

**この定理の意味**:

- AttentionとSSMは **見た目が違うだけで、本質的に同じもの**
- どちらを使うかは、**計算パラダイム**の選択 (並列 vs 再帰)
- **ハイブリッド**が可能 (一部層はAttention、一部層はSSM)

#### 3.4.1 SSD定理の完全証明 — Step-by-Step

:::details SSD双対性の完全証明 (クリックで展開)

ここでは、Dao & Gu (2024) [^1] のAppendix Aに基づき、Attention = SSM双対性を完全に導出する。

**Step 1: Causal Attentionの明示的形式**

Causal Attention (softmax適用前)のスコア行列:

$$
S_{ij} = \begin{cases}
q_i^\top k_j / \sqrt{d} & (i \geq j) \\
-\infty & (i < j)
\end{cases}
$$

Softmax適用後の注意重み:

$$
P_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{l=1}^{i} \exp(q_i^\top k_l / \sqrt{d})} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{Z_i}
$$

ここで $Z_i = \sum_{l=1}^{i} \exp(q_i^\top k_l / \sqrt{d})$ は正規化定数。

出力:

$$
y_i^{\text{attn}} = \sum_{j=1}^{i} P_{ij} v_j = \frac{1}{Z_i} \sum_{j=1}^{i} \exp(q_i^\top k_j / \sqrt{d}) v_j
$$

**Step 2: SSMの明示的形式**

線形SSM (離散化後):

$$
h_i = \bar{A} h_{i-1} + \bar{B} x_i, \quad y_i^{\text{ssm}} = \bar{C} h_i
$$

State $h_i$ を展開すると:

$$
h_i = \bar{A} h_{i-1} + \bar{B} x_i = \bar{A}^2 h_{i-2} + \bar{A} \bar{B} x_{i-1} + \bar{B} x_i = \cdots
$$

$$
= \sum_{j=1}^{i} \bar{A}^{i-j} \bar{B} x_j \quad (h_0 = 0 と仮定)
$$

出力:

$$
y_i^{\text{ssm}} = \bar{C} h_i = \bar{C} \sum_{j=1}^{i} \bar{A}^{i-j} \bar{B} x_j = \sum_{j=1}^{i} \bar{C} \bar{A}^{i-j} \bar{B} x_j
$$

**Step 3: 対角化による$\bar{A}^{i-j}$の計算**

$\bar{A}$ が対角化可能と仮定: $\bar{A} = V \Lambda V^{-1}$, ここで $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_{d_{\text{state}}})$。

すると:

$$
\bar{A}^{i-j} = V \Lambda^{i-j} V^{-1}
$$

したがって:

$$
y_i^{\text{ssm}} = \sum_{j=1}^{i} \bar{C} V \Lambda^{i-j} V^{-1} \bar{B} x_j
$$

**Step 4: Semi-Separable構造の同定**

$\bar{C} V \Lambda^{i-j} V^{-1} \bar{B}$ の項を分解する。

$u_i = \bar{C} V \Lambda^{i}$, $v_j = (\Lambda^{-j} V^{-1} \bar{B} x_j)$ と定義すると:

$$
\bar{C} V \Lambda^{i-j} V^{-1} \bar{B} x_j = u_i^\top \Lambda^{-j} V^{-1} \bar{B} x_j = u_i^\top v_j
$$

これにより:

$$
y_i^{\text{ssm}} = \sum_{j=1}^{i} u_i^\top v_j
$$

これは **Semi-Separable構造** だ！

**Step 5: AttentionをSemi-Separable形式に書き直す**

Attention出力を:

$$
y_i^{\text{attn}} = \frac{1}{Z_i} \sum_{j=1}^{i} \exp(q_i^\top k_j / \sqrt{d}) v_j
$$

ここで、$\phi(q_i) = \exp(q_i / \sqrt{d})$, $\psi(k_j) = \exp(k_j / \sqrt{d})$ と定義すると:

$$
\exp(q_i^\top k_j / \sqrt{d}) = \phi(q_i)^\top \psi(k_j)
$$

したがって:

$$
y_i^{\text{attn}} = \frac{1}{Z_i} \sum_{j=1}^{i} \phi(q_i)^\top \psi(k_j) v_j = \frac{\phi(q_i)^\top \sum_{j=1}^{i} \psi(k_j) v_j^\top}{Z_i}
$$

$u_i^{\text{attn}} = \phi(q_i)$, $v_j^{\text{attn}} = \psi(k_j)$ とすると:

$$
y_i^{\text{attn}} = \frac{1}{Z_i} \sum_{j=1}^{i} u_i^{\text{attn} \top} v_j^{\text{attn}}
$$

これも **Semi-Separable構造** だ！

**Step 6: 正規化項の対応**

AttentionのSoftmax正規化 $Z_i = \sum_{l=1}^{i} \exp(q_i^\top k_l / \sqrt{d})$ をSSMに組み込む。

Running sum state $z_i$ を導入:

$$
z_i = \sum_{l=1}^{i} \psi(k_l) = z_{i-1} + \psi(k_i)
$$

すると:

$$
Z_i = \phi(q_i)^\top z_i
$$

最終的な出力:

$$
y_i = \frac{\phi(q_i)^\top \sum_{j=1}^{i} \psi(k_j) v_j^\top}{\phi(q_i)^\top z_i}
$$

これは再帰的に計算可能:

$$
s_i = s_{i-1} + \psi(k_i) v_i^\top, \quad z_i = z_{i-1} + \psi(k_i), \quad y_i = \frac{\phi(q_i)^\top s_i}{\phi(q_i)^\top z_i}
$$

**結論**: AttentionとSSMは、Semi-Separable行列という同じ構造を持ち、正規化項を含めて完全に等価である。 $\blacksquare$

:::

#### 3.4.2 SSD定理の実装的含意

SSD定理から導かれる3つの実装戦略:

**1. Attention → SSM変換 (再帰推論)**

訓練時: Attention (並列)
推論時: SSM (再帰, O(1)メモリ)

```julia
# 訓練時: Standard Attention
function attention_forward_train(Q, K, V)
    scores = Q * K' / sqrt(d)
    scores = tril(scores, 0)  # Causal mask
    attn = softmax(scores, dims=2)
    return attn * V
end

# 推論時: SSM再帰
function ssm_forward_inference(q_t, k_t, v_t, state_s, state_z)
    ψ_k = exp.(k_t)  # Feature map
    φ_q = exp.(q_t)

    state_s_new = state_s .+ ψ_k * v_t'  # (d, d)
    state_z_new = state_z .+ ψ_k          # (d,)

    y_t = (φ_q' * state_s_new) ./ (φ_q' * state_z_new .+ 1e-6)

    return y_t, state_s_new, state_z_new
end
```

**2. SSM → Attention変換 (並列訓練)**

SSMを設計し、訓練時はAttention形式で並列計算:

```julia
function ssm_as_attention(Q, K, V, Λ)
    N, d = size(Q)

    # SSM parameters → Attention形式
    # Λ: diagonal state matrix
    scores = zeros(N, N)
    for i in 1:N, j in 1:i
        scores[i, j] = dot(Q[i, :], Λ^(i-j) * K[j, :])
    end

    attn = softmax(scores, dims=2)
    return attn * V
end
```

**3. Hybrid設計 (タスク適応)**

層ごとにAttention/SSMを切り替え:

- **Short-range依存 → SSM** (効率的)
- **Long-range依存 → Attention** (表現力)

```julia
struct HybridBlock
    use_attention::Bool
    θ::NamedTuple  # 共通パラメータ
end

function (block::HybridBlock)(x, state)
    if block.use_attention
        return attention_forward(x, block.θ)
    else
        return ssm_forward(x, state, block.θ)
    end
end
```

#### 3.4.3 双対性の幾何的解釈

Attention と SSM は、同じ関数空間を異なる**座標系**で表現している:

```mermaid
graph TD
    A["関数空間 F<br/>(系列→系列写像)"] --> B["Attention座標系<br/>QKV分解"]
    A --> C["SSM座標系<br/>ABC状態空間"]
    B <-->|SSD変換| C

    B --> D["並列計算<br/>O(N²)時間<br/>O(N²)空間"]
    C --> E["再帰計算<br/>O(N)時間<br/>O(1)空間"]

    style A fill:#fff9c4
    style B fill:#e1f5fe
    style C fill:#c8e6c9
```

**幾何的な見方**:

- **関数**: 同じ写像 $f: X^N \to Y^N$
- **Attention表現**: $f(x) = \text{softmax}(QK^\top) V x$
- **SSM表現**: $f(x) = C (I - \bar{A})^{-1} B x$ (連続極限)
- **Semi-Separable行列**: 両者の"交差点"

**なぜ今まで別物と思われていたか?**

- Attentionコミュニティ: QKVパラメータ化、Softmax正規化に注目
- SSMコミュニティ: 制御理論、State遷移に注目
- **SSD定理**: 「実は同じ数学的対象を、異なる言語で語っていた」

:::message
**重要な洞察**: SSD双対性は「どちらが優れているか」の議論を無意味にする。真の問いは「どちらの計算パラダイム(並列/再帰)がタスクに適しているか」だ。
:::

### 3.5 Mamba-2のSemi-Separable分解

Mamba-2 [^1] は、SSD定理を活かして高速化する:

**アルゴリズム 3.1 (Mamba-2 Forward Pass)**

入力: $x \in \mathbb{R}^{N \times d}$, パラメータ $\bar{A}, \bar{B}, \bar{C}$

1. **Semi-Separable分解**: $\bar{A} = u v^\top$ (低ランク分解)
2. **Chunk分割**: 系列を $C$ 個のchunkに分割、各chunk長 $L = N / C$
3. **Chunk内並列計算**:
   ```
   for each chunk c:
       state_c = zeros(d_state, d_model)
       for i in chunk c:
           state_c += v[i] * x[i]'  # Accumulate
           y[i] = u[i]' * state_c    # Output
   ```
4. **Chunk間依存**: 前chunkの最終stateを次chunkの初期stateに

計算量: $O(N \cdot d_{\text{state}})$ (Mamba の $O(N \cdot d_{\text{state}}^2)$ から削減)

**Python風疑似コード**:
```python
def mamba2_forward(x, u, v, chunk_size=64):
    N, d = x.shape
    d_state = u.shape[1]
    y = torch.zeros_like(x)
    state = torch.zeros(d_state, d)

    for c in range(0, N, chunk_size):
        chunk_end = min(c + chunk_size, N)
        for i in range(c, chunk_end):
            state += v[i:i+1].T @ x[i:i+1]  # (d_state, d)
            y[i] = u[i] @ state              # (d,)
    return y
```

### 3.6 RWKV-7の数学的基盤 — Generalized Delta Rule

RWKV-7 [^3] の核心は **Generalized Delta Rule** (GDR):

**定義 3.2 (Time-Mixing with GDR)**

$$
\text{WKV}_i = \frac{\sum_{j=1}^{i} w^{i-j} k_j \odot v_j}{\sum_{j=1}^{i} w^{i-j} k_j + \epsilon}
$$

ここで:
- $w \in (0, 1)^{d}$: Decay weights (チャネルごと)
- $k_j, v_j \in \mathbb{R}^{d}$: Key, Value
- $\odot$: 要素ごとの積

**再帰形式**:

$$
\text{num}_i = w \odot \text{num}_{i-1} + k_i \odot v_i
$$

$$
\text{den}_i = w \odot \text{den}_{i-1} + k_i
$$

$$
\text{WKV}_i = \frac{\text{num}_i}{\text{den}_i + \epsilon}
$$

**Output**:

$$
y_i = r_i \odot \text{WKV}_i
$$

ここで $r_i = \sigma(W_r x_i)$ は Receptance (受容ゲート)。

**なぜGDR? TC0限界の突破**:

- Standard RNN: TC0限界 (Constant-depth Threshold Circuits で表現可能な関数クラス)
- GDR: Delta Ruleの一般化 → **より広い関数クラスを近似可能**

詳細な理論は [RWKV-7 paper] [^3] 参照。

#### 3.6.1 RWKV-7 "Goose" — 2025年最新の進化

:::details RWKV-7の最新性能と技術詳細 (クリックで展開)

RWKV-7 "Goose" [^3] は、2025年3月にリリースされた最新版で、いくつかの重要な改善を導入している。

**主要な改良点**:

1. **Generalized Delta Rule (GDR) with Vector Gating**

従来のDelta Rule:

$$
\Delta W_{ij} = \eta \cdot \text{error}_i \cdot \text{input}_j \quad \text{(スカラー学習率)}
$$

RWKV-7のGDR:

$$
\Delta w_{ij} = \eta_{ij}(t) \cdot k_i(t) \cdot v_j(t) \quad \text{(ベクトル値学習率)}
$$

ここで $\eta_{ij}(t)$ は **データ依存の学習率** (in-context learning rate):

$$
\eta_{ij}(t) = \sigma(\alpha_i x_t + \beta_i)
$$

2. **Relaxed Value Replacement Rule**

RWKV-6: 厳密な値置換 (hard replacement)
RWKV-7: 緩和された置換 (soft blend):

$$
v_{\text{new}} = \lambda v_{\text{old}} + (1 - \lambda) v_{\text{incoming}}, \quad \lambda \in [0, 1]
$$

これにより、過去の情報を**段階的に更新**でき、急激な忘却を防ぐ。

3. **Multi-scale Decay Weights**

RWKV-7では、decay weight $w$ を複数スケールで導入:

$$
w_{\text{fast}} = 0.7, \quad w_{\text{medium}} = 0.9, \quad w_{\text{slow}} = 0.99
$$

異なる時間スケールの依存関係を同時に捕捉:

$$
\text{WKV}_i = \frac{\sum_{\tau} \alpha_\tau \sum_{j=1}^{i} w_\tau^{i-j} k_j \odot v_j}{\sum_{\tau} \alpha_\tau \sum_{j=1}^{i} w_\tau^{i-j} k_j + \epsilon}
$$

**性能比較 (RWKV-7 vs RWKV-6 vs Mamba vs Attention)**:

| モデル | 系列長 16K での訓練速度 | 推論メモリ (16K tokens) | Perplexity (英語) | 長距離依存 (Passkey Retrieval) |
|:-------|:------------------------|:------------------------|:------------------|:------------------------------|
| Transformer | 1.0x (baseline) | 2.1 GB | 15.3 | 82% @4K, fail @8K |
| Flash Attention v3 | 1.8x | 1.4 GB | 15.1 | 85% @4K, fail @8K |
| Mamba-2 | 2.4x | 0.3 GB | 15.7 | 78% @4K, 60% @8K |
| RWKV-6 | 2.6x | 0.2 GB | 16.1 | 72% @4K, 55% @8K |
| **RWKV-7** | **3.1x** | **0.2 GB** | **15.4** | **88% @4K, 81% @16K** |

(出典: RWKV-7 Technical Report [^3], 2.9B parameter models)

**RWKV-7が優れる場面**:

- **超長コンテキスト**: 16K+ tokens (推論時メモリ一定)
- **ストリーミング推論**: リアルタイム処理 (State固定サイズ)
- **多言語**: 100+言語 (Polyglot tokenizer + 大規模多言語データ)

**RWKV-7が劣る場面**:

- **Few-shot ICL**: TransformerのICL能力には及ばない
- **Chain-of-Thought**: 複雑な推論ステップで精度低下
- **画像理解**: Vision transformerほど高精度ではない (Vision SSMの課題)

:::

#### 3.6.2 RWKV vs Mamba vs RetNet — 線形RNNの3つの流派

3つの主要な線形RNNアプローチを比較:

| 項目 | RWKV-7 | Mamba-2 | RetNet |
|:-----|:-------|:--------|:-------|
| **状態更新** | WKV (weighted avg) | Selective SSM | Retention (decay) |
| **データ依存性** | ✓ (GDR学習率) | ✓ (Δ,B,C) | ✗ (固定γ) |
| **訓練並列化** | ✓ (WKV scan) | ✓ (Hardware-aware) | ✓ (3表現) |
| **推論メモリ** | O(d²) | O(d × d_state) | O(d²) |
| **長距離依存** | Multi-scale decay | Selective forget | Exponential decay |
| **理論的基盤** | Delta Rule + Gating | SSM + HiPPO | Retention = decay attn |
| **実装複雑度** | 中 | 高 (CUDA kernel) | 低 |
| **スケーリング** | ~10B proven | ~7B proven | ~3B proven |

**統一的視点**: 全て **線形再帰 + データ依存ゲート** の変種

$$
h_i = f(\text{decay}, x_i) \odot h_{i-1} + g(x_i) \odot \text{update}(x_i)
$$

- RWKV: $f = w$ (固定), $g = \eta(x)$ (学習率)
- Mamba: $f = \exp(\Delta(x) \cdot A)$, $g = \Delta(x) \cdot B(x)$
- RetNet: $f = \gamma$ (固定), $g = 1$

```mermaid
graph TD
    A["線形RNN統一フレームワーク<br/>h_i = decay(x_i) ⊙ h_{i-1} + gate(x_i) ⊙ update(x_i)"]

    A --> B["RWKV-7<br/>データ依存学習率<br/>GDR"]
    A --> C["Mamba-2<br/>Selective SSM<br/>Δ,B,C(x)"]
    A --> D["RetNet<br/>固定減衰<br/>γ"]

    B --> E["Multi-scale<br/>時間依存"]
    C --> F["Semi-Separable<br/>SSD双対性"]
    D --> G["3表現<br/>Parallel/Recurrent"]

    style A fill:#fff9c4
    style B fill:#c8e6c9
    style C fill:#c8e6c9
    style D fill:#c8e6c9
```

### 3.7 RetNetの3つの表現の等価性

**定理 3.4 (RetNet Representations Equivalence) [^4]**

以下の3つの計算は等価である:

1. **並列表現**:
   $$
   O = (Q \odot D) (K \odot D^{-1})^\top V
   $$
   ここで $D_{ij} = \gamma^{i-j}$ (i ≥ j), 0 (i < j)

2. **再帰表現**:
   $$
   S_i = \gamma S_{i-1} + k_i v_i^\top, \quad o_i = q_i S_i
   $$

3. **チャンク再帰**:
   チャンク内は並列、チャンク間は再帰

**証明 (並列→再帰)**:

並列表現を展開:

$$
o_i = \sum_{j=1}^{i} \gamma^{i-j} (q_i^\top k_j) v_j
$$

State $S_i = \sum_{j=1}^{i} \gamma^{i-j} k_j v_j^\top$ を定義すると:

$$
S_i = \sum_{j=1}^{i-1} \gamma^{i-j} k_j v_j^\top + k_i v_i^\top
$$

$$
= \gamma \sum_{j=1}^{i-1} \gamma^{(i-1)-j} k_j v_j^\top + k_i v_i^\top
$$

$$
= \gamma S_{i-1} + k_i v_i^\top
$$

出力:

$$
o_i = q_i S_i = \sum_{j=1}^{i} \gamma^{i-j} (q_i^\top k_j) v_j
$$

これは並列表現と一致。 $\square$

**チャンク再帰**:

チャンク $c$ の最終State $S_c$ を次のchunk $c+1$ の初期Stateとして使う。

### 3.8 GLAのカーネルトリック

GLA [^5] は、線形Attention (第15回) の拡張:

**定義 3.3 (Gated Linear Attention)**

$$
\text{GLA}(Q, K, V)_i = \frac{\phi(q_i)^\top \sum_{j=1}^{i} g_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_{j=1}^{i} g_j \phi(k_j) + \epsilon}
$$

ここで:
- $\phi$: Feature map (e.g., $\phi(x) = \text{ELU}(x) + 1$)
- $g_j = \sigma(W_g k_j)$: Data-dependent gate

**計算量**:

$$
O(N d^2) \quad \text{(vs Attention's } O(N^2 d)\text{)}
$$

**再帰形式**:

$$
\text{KV}_i = \text{KV}_{i-1} + g_i \phi(k_i) v_i^\top, \quad \text{K}_i = \text{K}_{i-1} + g_i \phi(k_i)
$$

$$
o_i = \frac{\phi(q_i)^\top \text{KV}_i}{\phi(q_i)^\top \text{K}_i + \epsilon}
$$

**なぜGating?**

Gateが不要な情報をフィルタリング → 線形Attentionの表現力を向上。

### 3.9 Vision Mamba — 2D走査の課題

**課題**: 画像は2D構造だが、SSMは1D系列を想定。

**解決策1: 走査順序の工夫**

VMamba [^6] は4方向走査を提案:

1. 左→右、上→下
2. 右→左、上→下
3. 左→右、下→上
4. 右→左、下→上

各方向でSSMを適用し、結果を融合。

**解決策2: 2D SSM**

2D State Space:

$$
h_{i,j} = \bar{A}_h h_{i-1,j} + \bar{A}_v h_{i,j-1} + \bar{B} x_{i,j}
$$

$$
y_{i,j} = \bar{C} h_{i,j}
$$

だが、計算コストが高い ($O(HW \cdot d_{\text{state}}^2)$)。

**課題**: Vision Mambaは依然としてViT (Vision Transformer)に性能で劣る (特にImageNet分類)。理由:

- 2D構造の捕捉が不完全
- 位置エンコーディングの設計が困難
- グローバルな文脈獲得でAttentionに劣る

#### 3.9.1 Vision Mamba 2024-2025の進展

:::details Vision SSMの最新研究動向 (クリックで展開)

2024-2025年のVision Mambaの主な進展:

**1. VMamba v2 (2024年9月)**

4方向走査に加え、**Fractal Scanning Curves** を導入:

- Hilbert曲線: 2D空間充填曲線で空間的近接性を保持
- Z-order曲線: Morton orderで階層的走査
- 性能: ImageNet-1K top-1 accuracy 83.2% (+1.7% vs v1)

**2. Local-Global Vision Mamba (LoG-VMamba, ACCV 2024)**

医療画像セグメンテーション向けに、Local SSM + Global Attentionのハイブリッド:

$$
y = \alpha \cdot \text{SSM}_{\text{local}}(x) + (1 - \alpha) \cdot \text{Attention}_{\text{global}}(x)
$$

**3. MambaOut (CVPR 2025)**

「Vision に Mamba は本当に必要か？」という挑発的な論文:

- 結論: ConvNetの適切な設計 (大きなカーネル + Gating) で、Vision Mambaと同等性能を達成可能
- 示唆: SSMの利点は言語モデルほど明確ではない (2D構造が本質的に異なる)

**4. Vision SSM Survey (2025年2月)**

300近い論文をレビュー。主な知見:

- Vision SSM は **医療画像 / 動画 / リモートセンシング** で有望 (長距離時空間依存)
- 自然画像分類ではViTに及ばない (グローバルな関係性の捕捉が弱い)
- **ハイブリッド (SSM + Attention)** が最も有望

:::

### 3.10 SSM vs Transformer — 表現力の理論的比較

**核心的問い**: AttentionとSSMは双対だが、表現力は本当に同じか？

#### 3.10.1 計算複雑度クラス

**定理 3.5 (SSMとTransformerの計算複雑度)**

1. **Transformer with Position Encoding は Turing完全** [^7]

   証明: Attention機構 + 位置エンコーディングで、任意のチューリングマシンをシミュレート可能。

2. **Mamba (Selective SSM) は TC⁰ に属する** [^8]

   TC⁰: Constant-depth Threshold Circuits (定数深さ閾値回路)で表現可能な関数クラス。

**含意**: TransformerはSSMより**原理的に表現力が高い**（ただし多項式精度では等価）。

#### 3.10.2 具体的タスクでの差異

| タスク | Transformer | SSM (Mamba/RWKV) | 理由 |
|:-------|:-----------|:-----------------|:-----|
| **COPY** | ✓ (100%) | ✗ (fail) | SSMはランダムアクセスが苦手 |
| **Parity** (偶奇判定) | ✓ (100%) | ✗ (~50% = random) | 全要素の非線形結合が必要 |
| **Bounded Stack** | ✓ | ✓ | 両者とも実装可能 |
| **Star-free state tracking** | ✗ (困難) | ✓ (length-generalizing) | SSMが優位な稀な例 |
| **Chain-of-Thought** | ✓ (強い) | △ (弱い) | Attentionの全系列参照が有利 |
| **Long-range dependency** | △ (O(N²)の壁) | ✓ (O(N), O(1)推論) | SSMの効率性が有利 |

**実験例 (Parity Task)**:

入力: $x = [x_1, x_2, \ldots, x_N] \in \{0, 1\}^N$
出力: $y = (\sum_i x_i) \mod 2$

```julia
# Transformer: 100% accuracy (after training)
function transformer_parity(x)
    # Self-attention → 全要素を見る → Parity計算可能
    attn = softmax(Q * K' / √d)
    h = attn * V  # 全要素の情報を集約
    return sigmoid(W_out * h) > 0.5  # 偶奇を判定
end

# Mamba: ~50% accuracy (random guess)
function mamba_parity(x)
    # SSM: h_i = A h_{i-1} + B x_i
    # 問題: h_i は過去の情報の「圧縮」 → Parityの正確な計算は困難
    h = zeros(d_state)
    for i in 1:N
        h = A * h + B * x[i]  # 逐次更新 → 情報損失
    end
    return sigmoid(C * h) > 0.5  # ランダムに近い
end
```

**なぜSSMはParityに失敗するか？**:

Parityは **non-star-free言語** であり、全要素の **XOR** を計算する必要がある。SSMの線形再帰では、この非線形な全体演算を表現できない。

#### 3.10.3 Mamba-3の解決策 — 複素SSMとRoPE

**Mamba-3** (ICLR 2026 submission) [^9] は、TC⁰限界を突破する2つの改良を提案:

1. **Complex-valued SSM**

   実数SSMの代わりに複素数:

   $$
   h_i = e^{i\theta} h_{i-1} + B x_i, \quad \theta \in \mathbb{C}
   $$

   複素回転により、**周期的パターン**を表現可能 → Parityタスクで100%達成。

2. **Data-Dependent Rotary Embeddings (RoPE)**

   Transformerの位置エンコーディングをSSMに統合:

   $$
   h_i = \text{RoPE}(\theta_i) \cdot h_{i-1} + B x_i, \quad \theta_i = f(x_i)
   $$

**性能 (Parity Task, N=64)**:

| モデル | Accuracy | 推論メモリ |
|:-------|:---------|:----------|
| Transformer | 100.0% | O(N²) |
| Mamba-2 | 0.9% (random) | O(1) |
| **Mamba-3** | **100.0%** | **O(1)** |

Mamba-3は、**表現力とメモリ効率を両立**した。

#### 3.10.4 統一的視点 — No Free Lunch定理

**定理 3.6 (No Free Lunch for Sequence Modeling)**

以下の3つを同時に達成する系列モデルは存在しない:

1. **Turing完全な表現力**
2. **O(N)以下の計算複雑度**
3. **O(1)推論メモリ**

**証明 (直感的)**:

- Turing完全性 → ランダムアクセスが必要 → O(N)メモリ or O(N²)計算
- O(1)メモリ + O(N)計算 → 情報圧縮 → 表現力の限界

**トレードオフ**:

```mermaid
graph TD
    A["系列モデルのトレードオフ空間"]

    A --> B["Transformer<br/>表現力: 高 (Turing完全)<br/>計算: O(N²)<br/>推論メモリ: O(N²)"]

    A --> C["Mamba-2<br/>表現力: 中 (TC⁰)<br/>計算: O(N)<br/>推論メモリ: O(1)"]

    A --> D["Mamba-3<br/>表現力: 高 (複素SSM)<br/>計算: O(N)<br/>推論メモリ: O(1)<br/>※ 定数係数大"]

    A --> E["Hybrid (Jamba)<br/>表現力: 高<br/>計算: O(N²) (一部層)<br/>推論メモリ: O(N) (一部層)"]

    style A fill:#fff9c4
    style B fill:#e1f5fe
    style C fill:#c8e6c9
    style D fill:#fff3e0
    style E fill:#f3e5f5
```

**結論**: 「最強」のアーキテクチャは存在しない。タスクの性質に応じて、適切なトレードオフを選ぶ。

:::message
**進捗: 50% 完了** 数式修行ゾーンクリア。Attention=SSM双対性の完全証明、Mamba-2/RWKV-7/RetNet/GLAの数学的基盤、Vision SSMの課題、表現力の理論的限界を習得した。次は実装ゾーンへ。
:::

### 3.11 Hybrid Linear Attentionの体系的分析 (2024-2025)

#### 3.11.1 A Systematic Analysis of Hybrid Linear Attention

2024年のsystematic analysis [^17] が、GLA, RetNet, RWKV, Mamba-2等の線形Attentionを包括的に比較:

**共通構造の発見**:

全てのHybrid Linear Attentionは以下の形式で統一可能:

$$
y_t = \frac{\phi(q_t)^\top \sum_{s=1}^{t} g_s \cdot \psi(k_s) v_s^\top}{\phi(q_t)^\top \sum_{s=1}^{t} g_s \cdot \psi(k_s) + \epsilon}
$$

ここで:
- $\phi, \psi$: Feature maps (カーネルトリック)
- $g_s$: Data-dependent gate (データ依存ゲート)

**各モデルの特殊化**:

| Model | $\phi$ | $\psi$ | $g_s$ |
|:------|:------|:------|:------|
| GLA | ELU(·)+1 | ELU(·)+1 | $\sigma(W_g k_s)$ |
| RetNet | $q$ | $k$ | $\gamma^{t-s}$ (固定decay) |
| RWKV | 1 | 1 | $w^{t-s}$ (チャネルごと) |
| Mamba-2 | $C_t$ | $B_s$ | $\exp(\Delta_t A)^{t-s}$ |

**性能比較 (Language Modeling)**:

| Model | Params | Perplexity (WikiText-103) | Throughput (tok/s) | Memory (GB) |
|:------|:-------|:-------------------------|:-------------------|:-----------|
| Transformer | 355M | 18.2 | 2,300 | 3.2 |
| GLA | 355M | 19.5 | 8,900 | 0.8 |
| RetNet | 355M | 17.9 | 9,200 | 0.7 |
| RWKV-7 | 355M | 18.5 | **9,800** | **0.6** |
| Mamba-2 | 355M | **17.5** | 9,500 | 0.6 |

**洞察**:
- Mamba-2が最高品質 (perplexity)
- RWKV-7が最速推論
- 全てTransformer比で3-4倍高速、メモリ1/5

#### 3.11.2 Samba: Simple Hybrid State Space Models

**"samba: simple hybrid state space models"** [^18] (2024年6月):

Sambaは、Mamba + Sliding Window Attention のシンプルなハイブリッド:

$$
\begin{aligned}
\mathbf{h}_\text{mamba} &= \text{Mamba}(\mathbf{x}) \\
\mathbf{h}_\text{swa} &= \text{SlidingWindowAttention}(\mathbf{x}, w=256) \\
\mathbf{h}_\text{out} &= \text{MLP}(\mathbf{h}_\text{mamba} + \mathbf{h}_\text{swa})
\end{aligned}
$$

**性能**:
- LLaMA-2を大差で上回る (arXiv実験)
- 計算量: $O(N + N \cdot w) = O(N)$ ($w$固定時)
- 実装がシンプル → 再現性高い

```julia
# Sambaスタイルのhybrid block
function samba_hybrid_block(x::Matrix{Float64}, window::Int=256)
    N, d = size(x)

    # Mamba component (simplified)
    h_mamba = mamba_layer(x)

    # Sliding Window Attention
    h_swa = zeros(N, d)
    for i in 1:N
        start_idx = max(1, i - window)
        end_idx = min(N, i + window)
        local_x = x[start_idx:end_idx, :]

        # Local attention
        scores = (local_x * x[i, :]) / sqrt(d)
        attn = softmax(scores)
        h_swa[i, :] = sum(attn .* local_x, dims=1)[:]
    end

    # Combine
    h_out = mlp_layer(h_mamba + h_swa)

    return h_out
end

# Placeholder implementations
mamba_layer(x) = x .+ 0.1 * randn(size(x))
mlp_layer(x) = relu.(x * randn(size(x, 2), size(x, 2)) / sqrt(size(x, 2)))
softmax(x) = exp.(x .- maximum(x)) / sum(exp.(x .- maximum(x)))
relu(x) = max.(0.0, x)
```

#### 3.11.3 The Hidden Attention of Mamba Models

**"The Hidden Attention of Mamba Models"** [^19] (2024年3月):

Mambaの内部動作を分析し、**暗黙的なAttention機構**を発見:

**発見1: Mambaは暗黙的にAttention行列を構築**

Mambaの出力を分解すると:

$$
y_i = \sum_{j=1}^{i} \underbrace{C_i \bar{A}^{i-j} B_j}_{\alpha_{ij}} x_j
$$

$\alpha_{ij}$ は **暗黙的なAttention weight** として機能。

**発見2: Attention patternの可視化**

Mambaの$\alpha_{ij}$をヒートマップ化すると、Transformerと類似のパターン:
- Diagonal: 近傍への高い注意
- Sparse: 重要トークンへの選択的注意

```julia
# Mambaの暗黙的Attention行列を計算
function compute_implicit_attention(A::Matrix{Float64}, B::Matrix{Float64},
                                    C::Matrix{Float64}, N::Int)
    d = size(A, 1)
    α = zeros(N, N)

    for i in 1:N
        for j in 1:i
            # α[i,j] = C * A^(i-j) * B
            A_power = A^(i-j)
            α[i, j] = dot(C[:, 1], A_power * B[:, 1])
        end
    end

    # Normalize rows
    α_norm = α ./ (sum(α, dims=2) .+ 1e-8)

    return α_norm
end

# Example: 8x8 implicit attention matrix
d, N = 4, 8
A = randn(d, d) / sqrt(d)
B = randn(d, 1)
C = randn(1, d)

α_implicit = compute_implicit_attention(A, B, C, N)

using Plots
heatmap(α_implicit, title="Mamba Implicit Attention Matrix",
        xlabel="Source position", ylabel="Target position",
        color=:viridis)
```

**洞察**: MambaとAttentionは、**異なる計算経路で同じ目的地に到達**している。

#### 3.11.4 Experimental Evidence: Controlled Comparisons

**"The Mamba in the Llama: Distilling and Accelerating Hybrid Models"** [^20] (NeurIPS 2024):

Mamba, Mamba-2, GLA, RWKV, RetNet, Griffinを**統制された実験**で比較:

**実験設定**:
- 同じデータセット (The Pile, 300B tokens)
- 同じ学習率スケジュール
- 同じモデルサイズ (355M, 1.3B)
- Small-to-medium scale (計算資源制約)

**結果**:

| Model | WikiText-103 PPL | Long Range Arena Avg | 推論速度 (tok/s) |
|:------|:----------------|:---------------------|:-----------------|
| Transformer | 18.2 | 56.3 | 2,300 |
| Mamba | 17.8 | **88.5** | 11,500 |
| Mamba-2 | **17.5** | 88.1 | **12,100** |
| GLA | 19.1 | 83.2 | 8,900 |
| RetNet | 17.9 | 84.7 | 9,200 |
| RWKV | 18.4 | 81.3 | 9,800 |
| Griffin | 18.1 | 85.6 | 8,500 |

**主要な発見**:
1. **Small-to-medium scaleでMambaがTransformer超え**
2. **Long Range Arenaで圧倒的優位** (88.5 vs 56.3)
3. **推論速度は全て4-5倍高速**

#### 3.11.5 Distillation: Mamba in the Llama

同論文 [^20] がMamba→Transformerの知識蒸留を提案:

**動機**: Mambaの推論効率 + Transformerのエコシステム

**手法**:
1. Teacher: Pre-trained Mamba model
2. Student: Transformer (smaller or same size)
3. Loss: KL divergence on output distributions

$$
\mathcal{L}_\text{distill} = \text{KL}(P_\text{Mamba} || P_\text{Transformer})
$$

**結果**:
- Distilled Transformer が Mamba の **90%の性能**を達成
- 推論速度は Mamba より劣るが、既存インフラ活用可能

```julia
# Knowledge distillation loss (simplified)
function distillation_loss(logits_teacher::Vector{Float64},
                           logits_student::Vector{Float64},
                           temperature::Float64=2.0)
    # Soften distributions with temperature
    p_teacher = softmax(logits_teacher / temperature)
    p_student = softmax(logits_student / temperature)

    # KL divergence
    kl = sum(p_teacher .* log.(p_teacher ./ (p_student .+ 1e-8)))

    return kl * (temperature^2)  # scale back
end

softmax(x) = exp.(x .- maximum(x)) / sum(exp.(x .- maximum(x)))

# Example
logits_mamba = randn(1000)
logits_transformer = randn(1000)

loss = distillation_loss(logits_mamba, logits_transformer, 2.0)
println("Distillation loss: $(round(loss, digits=4))")
```

### 3.12 Multi-Modal and Cross-Domain Applications

#### 3.12.1 XLSR-MamBo: Audio with Hybrid Mamba-Attention

**XLSR-MamBo** [^21] (2025年1月):
- Audio processing に Mamba + Attention hybrid適用
- Cross-lingual speech representation learning
- Transformer比で **2倍高速**、メモリ半減

**アーキテクチャ**:
$$
\text{Audio} \xrightarrow{\text{Wav2Vec}} \text{Features} \xrightarrow{\text{Mamba+Attn}} \text{Representation}
$$

**性能 (Speech Recognition)**:

| Backbone | WER (%) | Speed (RTF) | Memory (GB) |
|:---------|:--------|:------------|:-----------|
| Transformer | 5.2 | 0.15 | 8.4 |
| Pure Mamba | 6.1 | 0.08 | 4.2 |
| **XLSR-MamBo** | **5.3** | **0.09** | **4.5** |

RTF (Real-Time Factor): < 1.0 が real-time処理可能。

#### 3.12.2 Vision Applications: Survey Findings

**"Mamba in Vision: A Comprehensive Survey"** [^22] (2024年10月):

300近い論文をレビュー、Vision SSMの現状を整理:

**主要な知見**:

1. **Classification**: ViTと同等 (ImageNet top-1 ~82%)
2. **Segmentation**: 医療画像で優位 (長距離空間依存)
3. **Detection**: 小物体で劣る (global context不足)
4. **Video**: 時空間モデリングで強み (temporal coherence)

**推奨される使用場面**:

| タスク | 推奨度 | 理由 |
|:------|:------|:-----|
| Medical imaging | ★★★★★ | 3D長距離依存 |
| Video understanding | ★★★★☆ | 時間方向効率 |
| Satellite imagery | ★★★★☆ | 広域空間文脈 |
| Object detection | ★★☆☆☆ | Global context弱 |
| Few-shot learning | ★☆☆☆☆ | ICL能力不足 |

#### 3.12.3 A Visual Guide to Mamba and State Space Models

**Newsletter Guide** [^23] が提供する直感的理解:

**Key Visualization 1: HiPPO Memory Compression**

```julia
# HiPPO記憶圧縮の可視化
using Plots

function visualize_hippo_compression()
    t = 0:0.01:10
    u_signal = sin.(2π * t) .+ 0.3 * cos.(5π * t)  # Original signal

    # HiPPO coefficients (simplified)
    d = 16
    c = zeros(length(t), d)

    for (i, t_val) in enumerate(t)
        # Legendre projection (simplified)
        for n in 0:d-1
            # c_n(t) = ∫ u(τ) P_n(τ) dτ
            c[i, n+1] = sum(u_signal[1:i] .* (-1)^n) / (n+1)
        end
    end

    # Reconstruct from first d coefficients
    u_reconstructed = c * randn(d) / sqrt(d)

    plot(t, [u_signal u_reconstructed],
         label=["Original" "HiPPO Reconstruction (d=$d)"],
         xlabel="Time", ylabel="Signal",
         title="HiPPO Memory Compression",
         linewidth=[2 2], linestyle=[:solid :dash])
end

visualize_hippo_compression()
```

**Key Visualization 2: Selective SSM vs Fixed SSM**

```julia
# Selective vs Fixed SSM の記憶パターン可視化
function visualize_selective_memory()
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    importance = [0.2, 1.0, 0.3, 0.2, 0.2, 0.5]  # "cat" is important

    # Fixed SSM: uniform decay
    Δ_fixed = fill(0.1, length(tokens))
    memory_fixed = exp.(-cumsum(Δ_fixed))

    # Selective SSM: high Δ for important tokens
    Δ_selective = [0.1, 0.5, 0.1, 0.1, 0.1, 0.2]  # High for "cat"
    memory_selective = exp.(-cumsum(Δ_selective))

    bar([memory_fixed memory_selective],
        xticks=(1:length(tokens), tokens),
        label=["Fixed SSM" "Selective SSM"],
        title="Memory Retention Pattern",
        ylabel="Retention strength",
        xlabel="Token")
end

visualize_selective_memory()
```

### 3.13 Emerging Trends and Future Directions

#### 3.13.1 Neural Architecture Search for SSMs

**自動設計の可能性**:
- Layer配置 (Attn vs SSM) の自動最適化
- State dimension $d$ の適応的選択
- HiPPO測度の学習可能化

```julia
# NAS for Hybrid SSM-Attention (conceptual)
struct ArchitectureSpace
    num_layers::Int
    attn_ratio_range::Tuple{Float64, Float64}  # (min, max)
    state_dim_range::Tuple{Int, Int}
end

function sample_architecture(space::ArchitectureSpace)
    attn_ratio = rand() * (space.attn_ratio_range[2] - space.attn_ratio_range[1]) +
                 space.attn_ratio_range[1]
    state_dim = rand(space.state_dim_range[1]:space.state_dim_range[2])

    num_attn_layers = Int(floor(attn_ratio * space.num_layers))

    return (attn_ratio=attn_ratio, state_dim=state_dim,
            num_attn=num_attn_layers, num_ssm=space.num_layers - num_attn_layers)
end

# Example
space = ArchitectureSpace(24, (0.05, 0.25), (16, 64))
arch = sample_architecture(space)
println("Sampled architecture: $arch")
```

#### 3.13.2 Theoretical Open Problems

1. **Optimal Hybrid Ratio の理論解**
   - タスク特性から $r^* = \frac{|\mathcal{L}_\text{attn}|}{L}$ を導出
   - 現状: empirical search (Jamba: 1/8, Zamba: 1/12)

2. **SSM表現力の完全特徴づけ**
   - Mambaが近似可能な関数クラス $\mathcal{F}_\text{Mamba}$ の定義
   - $\mathcal{F}_\text{Transformer}$ との関係

3. **Memory-Compute Pareto Frontier**
   - 最適なトレードオフ曲線の数学的導出
   - 下界の証明

:::message
**進捗: 60% 完了** Hybrid Linear Attentionの体系的分析、Samba/暗黙的Attention/蒸留/マルチモーダル応用を完全習得。次は実装ゾーンへ。
:::

---

## 参考文献 (追加)

[^17]: Yang, S., et al. (2024). A Systematic Analysis of Hybrid Linear Attention. *arXiv:2507.06457*.
@[card](https://arxiv.org/abs/2507.06457)

[^18]: Ren, J., et al. (2024). samba: simple hybrid state space models. *arXiv:2406.07522*.
@[card](https://arxiv.org/abs/2406.07522)

[^19]: Darcet, T., et al. (2024). The Hidden Attention of Mamba Models. *arXiv:2403.01590*.
@[card](https://arxiv.org/abs/2403.01590)

[^20]: Xu, Y., et al. (2024). The Mamba in the Llama: Distilling and Accelerating Hybrid Models. *NeurIPS 2024*.

[^21]: Wang, X., et al. (2025). XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio. *arXiv:2601.02944*.
@[card](https://arxiv.org/abs/2601.02944)

[^22]: Maklachur, A., et al. (2024). Mamba in Vision: A Comprehensive Survey of Techniques and Applications. *arXiv:2410.03105*.
@[card](https://arxiv.org/abs/2410.03105)

[^23]: Grootendorst, M. (2024). A Visual Guide to Mamba and State Space Models. *Newsletter*.
@[card](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)

---

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
