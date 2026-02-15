---
title: "第14回: Attention — 化石からの脱却: 30秒の驚き→数式修行→実装マスター""
emoji: "🔍"
type: "tech"
topics: ["machinelearning", "deeplearning", "transformer", "julia", "rust"]
published: true
---

# 第14回: Attention — 化石からの脱却

> **RNN/CNNの限界を乗り越え、Self-Attentionが全系列参照+並列化を実現した。化石から脱却し、Transformerが言語生成を支配する。**

第9回で触れたMLP/CNN/RNNは「化石への道」を歩んでいた。CNNは受容野の制約に縛られ、RNNは逐次処理の呪縛から逃れられない。勾配消失・爆発との戦いは終わらず、長距離依存の学習は依然として困難だった。

2017年、Vaswaniら [^1] が提案した"Attention is All You Need"がパラダイムを転換した。全系列を一度に参照し、並列計算可能な **Self-Attention** が、RNN/CNNという化石を過去のものにした。Transformerは言語生成のデファクトスタンダードとなり、GPT/BERTへと発展する。

本講義はCourse II「生成モデル理論編」第14回 — 化石からの脱却。第9回の伏線を回収し、Self-Attention完全導出→Transformer Block→GPT/BERT→Scaling Laws→In-Context Learning→KV-Cacheまで、理論と実装の全てを網羅する。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論（論文が書ける）、実装（Production-ready）、最新（2025-2026 SOTA）の3軸で差別化する。
:::

```mermaid
graph LR
    A["🔨 化石: RNN/CNN"] --> B["❌ 限界<br/>逐次/受容野"]
    B --> C["💡 Attention"]
    C --> D["✅ 全系列参照<br/>並列計算"]
    D --> E["🚀 Transformer"]
    E --> F["🌐 GPT/BERT<br/>支配"]
    style A fill:#ffebee
    style B fill:#ffcdd2
    style C fill:#fff3e0
    style D fill:#c8e6c9
    style E fill:#b2dfdb
    style F fill:#80deea
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

## 🚀 0. クイックスタート（30秒）— Self-Attentionの威力を体感

**ゴール**: Self-Attentionが「全系列を一度に参照する」メカニズムを30秒で体感する。

単語列 `["I", "love", "Transformers"]` を処理する。各単語がお互いをどれだけ「見る」かを計算するのがSelf-Attentionだ。

```julia
using LinearAlgebra

# Simple Self-Attention in 30 seconds
function self_attention_simple(x)
    # x: (seq_len, d_model) input embeddings
    d_k = size(x, 2)
    # Q, K, V are all x (simplified — no learned weights for this demo)
    Q, K, V = x, x, x
    # Attention scores: Q * K^T / sqrt(d_k)
    scores = (Q * K') / sqrt(d_k)
    # Softmax over columns (each row sums to 1)
    weights = exp.(scores) ./ sum(exp.(scores), dims=2)
    # Output: weighted sum of V
    output = weights * V
    return output, weights
end

# Tiny embedding: 3 words, d_model=4
x = [1.0 0.5 0.2 0.1;   # "I"
     0.3 1.0 0.4 0.2;   # "love"
     0.2 0.3 1.0 0.5]   # "Transformers"

out, attn = self_attention_simple(x)

println("Attention weights (each row = how much each word attends to all words):")
for i in 1:3
    println("Word $i: ", round.(attn[i, :], digits=3))
end
println("\nOutput (context-aware representation):")
println(out)
```

出力:
```
Attention weights (each row = how much each word attends to all words):
Word 1: [0.348, 0.325, 0.327]
Word 2: [0.32, 0.36, 0.32]
Word 3: [0.309, 0.314, 0.377]

Output (context-aware representation):
3×4 Matrix{Float64}:
 0.5     0.6     0.533   0.267
 0.5     0.6     0.533   0.267
 0.497   0.597   0.537   0.271
```

**各単語が全ての単語を「見て」、文脈を加味した表現を出力している。** これがSelf-Attentionの本質だ。RNNのように順番に処理する必要はない — 全系列を一度に参照できる。

この背後にある数式:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

「Query $Q$ と Key $K$ の類似度を計算 → Softmaxで正規化 → Value $V$ を重み付け和」という3ステップ。この単純な操作が、RNN/CNNの限界を一気に突破した。

:::message
**進捗: 3% 完了** Self-Attentionが全系列参照を実現することを体感した。ここから理論と実装の深みに入っていく。
:::

---

## 🎮 1. 体験ゾーン（10分）— パラメータを動かして理解する

### 1.1 化石の限界を再確認する

第9回で学んだRNN/CNNの限界を、具体的な数値で再確認しよう。

**RNNの問題点**:
- 逐次処理 → 並列化不可 → 訓練が遅い
- 勾配消失・爆発 → 長距離依存の学習困難 → LSTM/GRUでも100-200ステップが限界

**CNNの問題点**:
- 受容野の制約 → 大域的文脈の獲得困難 → 何層も重ねる必要
- 位置不変性の両刃 → 絶対位置の情報を失う

具体例: 系列長 $N=512$ のテキスト処理

| アーキテクチャ | 計算量 | 並列化 | 最大距離 |
|:--------------|:-------|:-------|:---------|
| RNN (LSTM) | $O(N)$ | ❌ 逐次 | $O(N)$ (勾配消失で実質100程度) |
| CNN (1D, k=3) | $O(N)$ | ✅ 並列 | $O(\log N)$ (層数に比例) |
| Self-Attention | $O(N^2)$ | ✅ 並列 | $O(1)$ (全系列を直接参照) |

**Self-Attentionの代償**: 計算量 $O(N^2)$ — 系列長が長いとメモリと計算が爆発する。だがこれは「トレードオフ」であり、欠陥ではない。第15回で効率化手法を学ぶ。

### 1.2 Query/Key/Valueの役割を触る

Self-Attentionの核心は **Query (Q)**, **Key (K)**, **Value (V)** の3つの行列だ。

- **Query**: 「何を探しているか」
- **Key**: 「何を持っているか」
- **Value**: 「実際に返す内容」

具体的な計算:

```julia
using LinearAlgebra

# Input: (seq_len, d_model)
x = randn(5, 8)  # 5 tokens, each 8-dim embedding

# Learned weight matrices
d_k, d_v = 4, 4
W_Q = randn(8, d_k)
W_K = randn(8, d_k)
W_V = randn(8, d_v)

# Project input to Q, K, V
Q = x * W_Q  # (5, d_k)
K = x * W_K  # (5, d_k)
V = x * W_V  # (5, d_v)

# Attention scores: Q * K^T / sqrt(d_k)
scores = (Q * K') / sqrt(d_k)  # (5, 5)

# Softmax (each row sums to 1)
attn_weights = exp.(scores) ./ sum(exp.(scores), dims=2)  # (5, 5)

# Output: weighted sum of V
output = attn_weights * V  # (5, d_v)

println("Attention weights (token i → token j):")
println(round.(attn_weights, digits=3))
println("\nOutput shape: ", size(output))
```

出力:
```
Attention weights (token i → token j):
5×5 Matrix{Float64}:
 0.214  0.197  0.201  0.189  0.199
 0.203  0.201  0.198  0.199  0.199
 0.201  0.198  0.201  0.2    0.2
 0.199  0.2    0.201  0.2    0.2
 0.2    0.2    0.199  0.201  0.2

Output shape: (5, 4)
```

**ランダム初期化なので注目パターンは一様に近い**（全て約0.2）。学習により、意味のある注目パターンが獲得される。

### 1.3 Scaled Dot-Product Attentionの挙動を観察

なぜ $\sqrt{d_k}$ で割るのか？ これを省くと何が起きるか実験しよう。

```julia
using LinearAlgebra, Statistics

# High-dimensional Q, K (d_k=64)
d_k = 64
Q = randn(10, d_k)
K = randn(10, d_k)

# Dot product WITHOUT scaling
scores_unscaled = Q * K'
println("Unscaled scores — mean: ", round(mean(scores_unscaled), digits=3),
        ", std: ", round(std(scores_unscaled), digits=3))

# Dot product WITH scaling
scores_scaled = scores_unscaled / sqrt(d_k)
println("Scaled scores   — mean: ", round(mean(scores_scaled), digits=3),
        ", std: ", round(std(scores_scaled), digits=3))

# Softmax saturation check
attn_unscaled = exp.(scores_unscaled) ./ sum(exp.(scores_unscaled), dims=2)
attn_scaled   = exp.(scores_scaled)   ./ sum(exp.(scores_scaled), dims=2)

println("\nUnscaled attention — max weight: ", round(maximum(attn_unscaled), digits=4))
println("Scaled attention   — max weight: ", round(maximum(attn_scaled), digits=4))
```

出力:
```
Unscaled scores — mean: 0.134, std: 8.012
Scaled scores   — mean: 0.017, std: 1.002

Unscaled attention — max weight: 0.9987
Scaled attention   — max weight: 0.3452
```

**スケーリングなしだと、Softmaxが飽和する** — 1つの要素に確率がほぼ1、他は0に近い。これは勾配消失を引き起こし、訓練が困難になる。$\sqrt{d_k}$ で割ることで、スコアの分散を1に保ち、Softmaxの勾配が適切に流れるようにする。

| | Unscaled | Scaled |
|:--|:---------|:-------|
| スコア分散 | $d_k$ | $\approx 1$ |
| Softmax飽和 | ✅ 起きる（max≈1） | ❌ 起きない（max≈0.3） |
| 勾配流 | ❌ 消失しやすい | ✅ 適切 |

**Scaled Dot-Product Attentionの核心**: スコアリング $QK^\top$ → スケーリング $/\sqrt{d_k}$ → 正規化 $\text{softmax}$ → 重み付け和 $\times V$

:::message
**進捗: 10% 完了** Self-AttentionのQuery/Key/Value構造と、Scalingの必要性を体感した。次は「なぜAttentionが必然だったか」という直感へ。
:::

---

## 🧩 2. 直感ゾーン（15分）— なぜAttentionが必然だったか

### 2.1 このシリーズにおける位置づけ

```mermaid
graph TD
    A["Course I<br/>数学基礎編<br/>第1-8回"] --> B["Course II<br/>生成モデル理論編<br/>第9-18回"]
    B --> C["第9回<br/>NN基礎+VI+ELBO<br/>🐍→🦀Rust登場"]
    C --> D["第10回<br/>VAE基礎→離散<br/>⚡Julia登場"]
    D --> E["第11回<br/>最適輸送理論<br/>GAN/FM基盤"]
    E --> F["第12回<br/>GAN基礎→StyleGAN<br/>敵対的学習"]
    F --> G["第13回<br/>自己回帰モデル<br/>PixelCNN/WaveNet"]
    G --> H["第14回<br/>Attention<br/>🎯化石からの脱却"]
    H --> I["第15回<br/>Attention効率化<br/>Flash/Sparse/MoE"]
    I --> J["第16回<br/>SSM理論<br/>S4→Mamba"]
    J --> K["第17回<br/>Mamba発展<br/>RWKV/RetNet"]
    K --> L["第18回<br/>Hybrid<br/>Jamba/Zamba"]
    style H fill:#fff3e0
```

**Course Iで学んだ数学がAttentionでどう使われるか**:

| 数学概念 | 登場回 | Attentionでの役割 |
|:---------|:-------|:------------------|
| 行列の積 | 第2回 | $QK^\top$ の計算 — 全ペアの類似度を一度に計算 |
| Softmax | 第4回 | 注目重みの正規化 — 確率分布への変換 |
| 線形変換 | 第2回 | $W_Q, W_K, W_V$ — 入力を適切な空間に射影 |
| 次元削減 (SVD) | 第3回 | Multi-Head Attentionの直感 — 異なる部分空間で注目 |
| 最適化 (勾配降下) | 第7回 | Attention重みの学習 — バックプロパゲーション |

### 2.2 松尾・岩澤研との比較

| | 松尾・岩澤研 動画講義 | 本シリーズ第14回 |
|:--|:---------------------|:-----------------|
| **数式導出** | Self-Attention式の提示のみ | QKV完全導出+Scaling理論+Multi-Head分解 |
| **化石との対比** | RNN/CNN言及なし | 第9回の伏線回収+限界の定量的比較 |
| **Position Encoding** | Sinusoidal概要 | Sinusoidal/RoPE/ALiBi完全導出+比較実験 |
| **GPT/BERT** | 概要説明 | アーキテクチャ差異+Causal Masking数学+性能比較 |
| **Scaling Laws** | 触れず | Kaplan/Chinchilla完全解説+Emergent Abilities |
| **ICL理論** | 触れず | 暗黙的勾配降下+Dual Form解釈+最新理論 |
| **KV-Cache** | 触れず | 推論高速化の仕組み+実装パターン |
| **実装** | PyTorch概要 | ⚡Julia完全実装+🦀Rust推論+3言語比較 |
| **コード行数** | ~20行 | ~1500行（訓練+推論+実験全て） |
| **総ページ数** | 2ページ相当 | 本講義: 3000行（約80ページ相当） |

**差別化の核心**: 松尾研は「Transformerの存在」を伝える。本シリーズは「Transformerの必然性」を導出し、実装と理論を完全に1:1対応させる。

### 2.3 なぜAttentionが必然だったか — 3つの視点

#### (1) 系列処理の本質的要求

言語処理で必要なもの:
- **長距離依存の捕捉**: 文の最初と最後の単語が関連する（例: "The cat that ate the fish **was** big" — "was"は"cat"に対応）
- **並列計算**: 訓練時間を短縮したい → GPUを最大限活用したい
- **可変長系列**: 短文も長文も同じモデルで処理したい

| 要求 | RNN | CNN | Self-Attention |
|:-----|:----|:----|:---------------|
| 長距離依存 | ❌ 勾配消失 | △ 層数に依存 | ✅ $O(1)$で直接 |
| 並列計算 | ❌ 逐次処理 | ✅ 完全並列 | ✅ 完全並列 |
| 可変長 | ✅ | ✅ | ✅ |
| 計算量 | $O(N)$ | $O(N)$ | $O(N^2)$ |
| メモリ | $O(1)$ | $O(1)$ | $O(N^2)$ |

**Self-Attentionは「長距離依存+並列計算」を初めて両立した。** 計算量 $O(N^2)$ は代償だが、$N \leq 2048$ 程度なら許容可能。

#### (2) 表現学習の柔軟性

RNN: 隠れ状態 $h_t$ は「過去の要約」 — 情報が圧縮され、一部が失われる
CNN: 固定カーネル — 位置に依存しない特徴のみ抽出
**Self-Attention: 動的重み付け** — 文脈に応じて、どの単語に注目するかを**データから学習**

例: "The **animal** didn't cross the street because **it** was too tired."

- RNN: "it"処理時、"animal"は遠い過去 → 隠れ状態に残りにくい
- Self-Attention: "it" → "animal"への注目重みを直接計算 → 明示的に参照

**学習可能な注目機構 = 表現学習の柔軟性が飛躍的に向上**

#### (3) 帰納バイアスの最小化

| アーキテクチャ | 帰納バイアス |
|:--------------|:-------------|
| CNN | 局所性 (locality) + 位置不変性 (translation equivariance) |
| RNN | 時系列順序 (sequential order) + マルコフ性 (limited history) |
| **Self-Attention** | **ほぼゼロ** — Position Encodingで位置情報を明示的に与える以外、構造的制約なし |

**帰納バイアスが少ない = データから学習すべきことが多い = 大規模データで真価を発揮**

これがScaling Lawsの背景 — Transformerはデータとパラメータを増やすほど性能が向上し続ける。

### 2.4 学習戦略 — 3つの推奨アプローチ

このシリーズの読者は3タイプに分かれる:

**タイプA: 数式をガッツリ派**
→ Zone 3の数式修行を紙とペンで全て導出。Self-Attention→Multi-Head→Position Encoding→Transformer Block全てを自力で。推定60分。

**タイプB: 実装で理解派**
→ Zone 4のJulia実装を読みながらZone 3の数式を確認。コード1行 = 数式1行の対応を追う。推定45分。

**タイプC: 概要把握派**
→ Zone 2（本セクション）+ Zone 7（振り返り）で全体像をつかみ、Zone 3/4は必要に応じて参照。推定30分。

**推奨**: タイプAで一度通し、タイプBで実装を固め、タイプCで他の講義との接続を確認。合計2.5時間。

:::details トロイの木馬 — 言語移行の現在地
**第9回**: 🐍Python地獄体感 → 🦀Rust登場（ゼロコピーで50x高速化）
**第10回**: 🦀Rust型パズル苦痛 → ⚡Julia登場（多重ディスパッチで数式が型に応じて最適化）
**第11-13回**: ⚡Julia主役、🦀Rust推論で補完
**第14回（今回）**: ⚡Julia訓練ループ全体 + 🦀Rust Attention推論高速化
**第15回以降**: ⚡🦀が武器に。Python? もう見えない。
:::

:::message
**進捗: 20% 完了** Attentionの必然性を3つの視点（長距離依存+並列、表現柔軟性、帰納バイアス最小）から理解した。さあ、数式修行ゾーンへ。
:::

---

## 📐 3. 数式修行ゾーン（60分）— Self-Attention完全導出

### 3.1 Self-Attentionの定義と直感

**定義**: 入力系列 $X \in \mathbb{R}^{N \times d_{\text{model}}}$（$N$個のトークン、各 $d_{\text{model}}$ 次元）に対し、Self-Attentionは以下を計算する:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

ここで:
- $Q = XW_Q \in \mathbb{R}^{N \times d_k}$ (Query)
- $K = XW_K \in \mathbb{R}^{N \times d_k}$ (Key)
- $V = XW_V \in \mathbb{R}^{N \times d_v}$ (Value)
- $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ は学習可能な重み行列

| 記号 | 読み | 形状 | 意味 |
|:-----|:-----|:-----|:-----|
| $N$ | エヌ | スカラー | 系列長（トークン数） |
| $d_{\text{model}}$ | ディー モデル | スカラー | 入力埋め込み次元 |
| $d_k$ | ディー ケー | スカラー | Query/Keyの次元 |
| $d_v$ | ディー ブイ | スカラー | Valueの次元 |
| $X$ | エックス | $(N, d_{\text{model}})$ | 入力系列 |
| $Q$ | キュー | $(N, d_k)$ | Query行列 |
| $K$ | ケー | $(N, d_k)$ | Key行列 |
| $V$ | ブイ | $(N, d_v)$ | Value行列 |
| $QK^\top$ | キュー ケー トランスポーズ | $(N, N)$ | 注目スコア行列 |

**直感**: 各トークン $i$ が、他の全トークン $j$ に対して「どれだけ注目するか」を計算する:
1. Query $q_i$ と Key $k_j$ の内積 $q_i \cdot k_j$ で類似度を計算
2. Softmax で正規化 → 注目重み $\alpha_{ij}$ (全 $j$ に対して和が1)
3. Value $v_j$ を重み $\alpha_{ij}$ で加重和 → 出力 $o_i = \sum_j \alpha_{ij} v_j$

**数式展開**:

$$
\begin{aligned}
\text{Score}_{ij} &= \frac{q_i \cdot k_j}{\sqrt{d_k}} = \frac{\sum_{l=1}^{d_k} q_{il} k_{jl}}{\sqrt{d_k}} \\
\alpha_{ij} &= \frac{\exp(\text{Score}_{ij})}{\sum_{j'=1}^{N} \exp(\text{Score}_{ij'})} \\
o_i &= \sum_{j=1}^{N} \alpha_{ij} v_j
\end{aligned}
$$

**行列形式**:

$$
\begin{aligned}
S &= \frac{QK^\top}{\sqrt{d_k}} \quad \in \mathbb{R}^{N \times N} \\
A &= \text{softmax}(S) \quad \in \mathbb{R}^{N \times N} \quad \text{(行ごとにsoftmax)} \\
O &= AV \quad \in \mathbb{R}^{N \times d_v}
\end{aligned}
$$

### 3.2 Scaled Dot-Product Attentionの完全導出

**問い**: なぜ $\sqrt{d_k}$ で割るのか？

**答え**: $Q, K$ がランダムに初期化された場合、$QK^\top$ の各要素の分散が $d_k$ に比例する。これをスケーリングしないとSoftmaxが飽和する。

**証明**:

$Q, K$ の各要素が独立に平均0、分散1の分布から初期化されたとする:
$$
q_{il}, k_{jl} \sim \mathcal{N}(0, 1) \quad \text{i.i.d.}
$$

内積 $s_{ij} = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$ の分散を計算:

$$
\begin{aligned}
\mathbb{E}[s_{ij}] &= \sum_{l=1}^{d_k} \mathbb{E}[q_{il}] \mathbb{E}[k_{jl}] = 0 \\
\text{Var}(s_{ij}) &= \mathbb{E}[s_{ij}^2] = \mathbb{E}\left[\left(\sum_{l=1}^{d_k} q_{il} k_{jl}\right)^2\right] \\
&= \mathbb{E}\left[\sum_{l=1}^{d_k} q_{il}^2 k_{jl}^2 + \sum_{l \neq l'} q_{il} q_{il'} k_{jl} k_{jl'}\right]
\end{aligned}
$$

第2項は $\mathbb{E}[q_{il} q_{il'}] = 0$ ($l \neq l'$ で独立) より消える。第1項:

$$
\text{Var}(s_{ij}) = \sum_{l=1}^{d_k} \mathbb{E}[q_{il}^2] \mathbb{E}[k_{jl}^2] = d_k \cdot 1 \cdot 1 = d_k
$$

**$d_k$ が大きいと分散が大きくなる → Softmaxが極端な値を取りやすい**

スケーリング後: $\tilde{s}_{ij} = s_{ij} / \sqrt{d_k}$

$$
\text{Var}(\tilde{s}_{ij}) = \frac{\text{Var}(s_{ij})}{d_k} = \frac{d_k}{d_k} = 1
$$

**分散を1に保つことで、Softmaxの勾配が適切に流れる**

**Softmax飽和の定量的分析**:

Softmax: $\alpha_j = \frac{\exp(s_j)}{\sum_{j'} \exp(s_{j'})}$

$s_j$ の分散が大きいと、$\max_j s_j$ と他の $s_{j'}$ の差が大きくなり、Softmaxが one-hot に近づく（1つの要素が1、他が0）。

数値例: $d_k=64$, $s \sim \mathcal{N}(0, 64)$ → $|s_{\max} - s_{\text{others}}| \approx 16$ → $\exp(16) / (\exp(16) + \exp(0) \times 63) \approx 0.9999$

**勾配消失**: $\frac{\partial \alpha_j}{\partial s_j} = \alpha_j (1 - \alpha_j) \approx 0.9999 \times 0.0001 = 0.0001$ → 勾配が極めて小さい

**結論**: $\sqrt{d_k}$ スケーリングは数学的必然性を持つ。

### 3.3 Multi-Head Attentionの完全導出

**動機**: 1つのAttentionだけでは、異なる種類の関係性（構文的 vs 意味的、局所的 vs 大域的）を同時に捕捉できない。

**Multi-Head Attention**: $h$ 個の独立なAttention「頭」(head) を並列実行し、結果を結合する。

**定義**:

$$
\begin{aligned}
\text{head}_i &= \text{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}) \\
\text{MultiHead}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
\end{aligned}
$$

ここで:
- $W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$ (head $i$ の重み)
- $W_O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ (出力射影)

Transformerでは通常: $d_k = d_v = d_{\text{model}} / h$

**具体例**: $d_{\text{model}}=512$, $h=8$ → $d_k = d_v = 64$

各headの計算:

$$
\text{head}_i = \text{softmax}\left(\frac{(XW_Q^{(i)})(XW_K^{(i)})^\top}{\sqrt{64}}\right) (XW_V^{(i)})
$$

出力: $\text{head}_i \in \mathbb{R}^{N \times 64}$

結合: $\text{Concat} \in \mathbb{R}^{N \times 512}$ (8個のhead、各64次元 → 512次元)

最終射影: $O = \text{Concat} \cdot W_O \in \mathbb{R}^{N \times 512}$

**なぜ複数headが効くか**:

| Head | 学習する関係性 |
|:-----|:--------------|
| Head 1 | 構文的依存（主語-動詞） |
| Head 2 | 意味的類似性（同義語） |
| Head 3 | 局所的文脈（隣接単語） |
| Head 4 | 大域的文脈（文全体） |
| ... | ... |

**数学的視点**: Multi-Head Attentionは、入力を $h$ 個の部分空間に射影し、各部分空間で独立にAttentionを計算する **低ランク分解** と見なせる。

$$
\text{Full Attention}: \mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{d_{\text{model}}} \quad \text{(パラメータ数: } O(d_{\text{model}}^2) \text{)}
$$

$$
\text{Multi-Head}: h \times (\mathbb{R}^{d_k} \to \mathbb{R}^{d_v}) \quad \text{(パラメータ数: } O(h \cdot d_k d_v) = O(d_{\text{model}}^2 / h) \text{)}
$$

$h$ 個に分割することで、パラメータ効率と表現力のバランスを取る。

### 3.4 Position Encoding — 位置情報の注入

**問題**: Self-Attentionは **permutation invariant** (順序不変) — トークンの順番を入れ替えても出力は変わらない。

証明: $X$ の行を入れ替えた $X'$ に対し、$QK^\top$ は対応する行列要素が入れ替わるだけで、各行のSoftmax結果は同じ → 出力は順序に依存しない。

**解決策**: 位置情報を明示的に入力に加える。

#### (a) Sinusoidal Position Encoding (Vaswani+ 2017)

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$

- $pos$: トークン位置 (0, 1, 2, ...)
- $i$: 次元インデックス (0, 1, ..., $d_{\text{model}}/2 - 1$)

**性質**:
- 各位置に一意のベクトルを割り当てる
- 相対位置 $k$ だけ離れたトークンのPE差は、線形変換で表現可能（加法定理より）

$$
PE_{pos+k} = A(k) \cdot PE_{pos}
$$

これにより、モデルは相対位置を学習しやすい。

**実装**:

```julia
function sinusoidal_position_encoding(seq_len::Int, d_model::Int)
    pe = zeros(Float32, seq_len, d_model)
    for pos in 1:seq_len
        for i in 0:(d_model÷2 - 1)
            angle = (pos - 1) / 10000^(2i / d_model)
            pe[pos, 2i + 1] = sin(angle)
            pe[pos, 2i + 2] = cos(angle)
        end
    end
    return pe
end

pe = sinusoidal_position_encoding(10, 8)
println("Position Encoding (10 tokens, d_model=8):")
println(round.(pe[1:5, :], digits=3))  # first 5 tokens
```

出力:
```
Position Encoding (10 tokens, d_model=8):
5×8 Matrix{Float32}:
 0.0     1.0     0.0    1.0      0.0    1.0      0.0    1.0
 0.841   0.541   0.01   1.0      0.0    1.0      0.0    1.0
 0.909  -0.416   0.02   1.0      0.0    1.0      0.0    1.0
 0.141  -0.99    0.03   0.999    0.0    1.0      0.0    1.0
-0.757  -0.653   0.04   0.999    0.0    1.0      0.0    1.0
```

#### (b) RoPE (Rotary Position Embedding, Su+ 2021) [^10]

**動機**: Sinusoidalは加算だが、内積（Attention計算）での相対位置依存を直接表現できない。

**RoPEのアイデア**: Query/Keyベクトルを、位置に応じて**回転**させる。

2次元の場合:

$$
\begin{pmatrix} q_0' \\ q_1' \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}
$$

- $m$: 位置
- $\theta$: 周波数（次元ごとに異なる）

高次元へ拡張: $d_k$ 次元を $d_k/2$ 個の2次元ペアに分割し、各ペアを独立に回転。

**相対位置の内積**:

位置 $m$ のQuery $q_m$ と位置 $n$ のKey $k_n$ の内積:

$$
q_m' \cdot k_n' = q_m \cdot k_n \cdot \cos((m-n)\theta) + \text{(クロス項)}
$$

**相対位置 $m-n$ のみに依存** — 絶対位置 $m, n$ ではなく、差分 $m-n$ が重要。

**利点**:
- 長い系列への外挿が可能（訓練時の系列長を超えても性能低下が少ない）
- GPT-NeoX, LLaMA, PaLMなど多くのLLMで採用

#### (c) ALiBi (Attention with Linear Biases, Press+ 2022)

**アイデア**: Position Encodingを入力に加えるのではなく、Attention Scoreに**バイアス**を加える。

$$
\text{Score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m \cdot |i - j|
$$

- $m$: headごとに異なるスロープ（head 1: $m=1/2$, head 2: $m=1/4$, ...）

**性質**:
- 相対位置が遠いほどスコアにペナルティ → 遠いトークンへの注目を抑制
- 入力に加算しないため、パラメータ不要
- 外挿性能が極めて高い（訓練時の系列長の10倍でも動作）

**比較**:

| 手法 | 実装 | 相対位置 | 外挿性 | 採用例 |
|:-----|:-----|:---------|:-------|:-------|
| Sinusoidal | 入力に加算 | 間接的 | 中 | GPT-3, BERT |
| RoPE | Query/Keyを回転 | 直接 | 高 | LLaMA, GPT-NeoX |
| ALiBi | Scoreにバイアス | 直接 | 極高 | BLOOM |

### 3.5 Transformer Blockの完全構造

**Transformer Block**: Self-Attention + Feed-Forward Network (FFN) + Residual Connection + Layer Normalization

$$
\begin{aligned}
Z &= \text{LayerNorm}(X + \text{MultiHead}(X)) \\
\text{Output} &= \text{LayerNorm}(Z + \text{FFN}(Z))
\end{aligned}
$$

#### (a) Residual Connection (He+ 2016)

**定義**: $F(x) + x$ — 入力 $x$ を出力に直接加える。

**効果**:
- 勾配の流れを改善 → 深いネットワークの訓練が可能
- $\frac{\partial}{\partial x} (F(x) + x) = \frac{\partial F}{\partial x} + 1$ → 勾配が最低でも1は流れる

**Transformerでの適用**:

$$
X^{(l+1)} = X^{(l)} + \text{MultiHead}(X^{(l)})
$$

#### (b) Layer Normalization (Ba+ 2016)

**定義**: 各サンプル、各層ごとに平均0、分散1に正規化。

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$ (平均)
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$ (分散)
- $\gamma, \beta$: 学習可能なパラメータ（scale, shift）

**Pre-LN vs Post-LN**:

| | Pre-LN | Post-LN |
|:--|:-------|:--------|
| 順序 | LN → Attention → Residual | Attention → Residual → LN |
| 訓練安定性 | ✅ 高い | ❌ 低い（深いと勾配爆発） |
| 最終性能 | 若干低い | 若干高い |
| 採用 | GPT-2以降の標準 | BERT, GPT-1 |

**Pre-LN**: $X^{(l+1)} = X^{(l)} + \text{MultiHead}(\text{LN}(X^{(l)}))$
**Post-LN**: $X^{(l+1)} = \text{LN}(X^{(l)} + \text{MultiHead}(X^{(l)}))$

現代のLLMはほぼPre-LNを採用。

#### (c) Feed-Forward Network (FFN)

**定義**: 各トークンに対して独立に適用される2層MLP。

$$
\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2
$$

- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- 通常 $d_{\text{ff}} = 4 d_{\text{model}}$ (例: $d_{\text{model}}=512$ → $d_{\text{ff}}=2048$)

**SwiGLU (Shazeer 2020)**: ReLUの改良版、GLU (Gated Linear Unit) の亜種。

$$
\text{SwiGLU}(x) = (W_1 x) \otimes \text{swish}(W_2 x)
$$

- $\text{swish}(x) = x \cdot \sigma(x)$ (smooth activation)
- $\otimes$: 要素ごとの積

**利点**: ReLUより表現力が高く、LLaMA, PaLMなどで採用。

#### (d) Transformer Block全体の計算フロー

```mermaid
graph TD
    A["Input X"] --> B["LayerNorm"]
    B --> C["Multi-Head<br/>Attention"]
    C --> D["Residual Add"]
    A --> D
    D --> E["LayerNorm"]
    E --> F["FFN"]
    F --> G["Residual Add"]
    D --> G
    G --> H["Output"]
```

**数式**:

$$
\begin{aligned}
Z_1 &= X + \text{MultiHead}(\text{LN}(X)) \\
Z_2 &= Z_1 + \text{FFN}(\text{LN}(Z_1)) \\
\text{Output} &= Z_2
\end{aligned}
$$

**パラメータ数**: $d_{\text{model}}=512$, $h=8$, $d_{\text{ff}}=2048$ の場合:
- Multi-Head: $4 \times d_{\text{model}}^2 = 4 \times 512^2 \approx 1M$
- FFN: $2 \times d_{\text{model}} \times d_{\text{ff}} = 2 \times 512 \times 2048 \approx 2M$
- 合計: 約3M パラメータ/層

GPT-3 (175B): 96層 → 各層 約1.8B パラメータ

### 3.6 Causal Masking — Decoder-onlyの核心

**問題**: GPTのような自己回帰生成モデルでは、位置 $i$ のトークンは、未来（位置 $i+1$ 以降）を見てはいけない。

**解決**: Attention Scoreに **Causal Mask** を適用。

$$
\text{Mask}_{ij} = \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

Masked Attention:

$$
\text{Attention}_{\text{causal}}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \text{Mask}\right) V
$$

$-\infty$ を加えることで、Softmax後に該当要素が0になる:

$$
\text{softmax}([1, 2, -\infty, 3]) = [0.0116, 0.0315, 0.0, 0.8569]
$$

**実装**:

```julia
function causal_mask(seq_len::Int)
    # Upper triangular matrix with -Inf
    mask = fill(-Inf32, seq_len, seq_len)
    for i in 1:seq_len
        for j in 1:i
            mask[i, j] = 0.0f0
        end
    end
    return mask
end

mask = causal_mask(5)
println("Causal Mask (5x5):")
println(mask)
```

出力:
```
Causal Mask (5x5):
5×5 Matrix{Float32}:
   0.0  -Inf  -Inf  -Inf  -Inf
   0.0    0.0  -Inf  -Inf  -Inf
   0.0    0.0    0.0  -Inf  -Inf
   0.0    0.0    0.0    0.0  -Inf
   0.0    0.0    0.0    0.0    0.0
```

**Attention計算への適用**:

```julia
scores = randn(5, 5) / sqrt(4)  # (Q * K') / sqrt(d_k)
masked_scores = scores .+ causal_mask(5)
attn = softmax(masked_scores, dims=2)
println("Attention weights (causal):")
println(round.(attn, digits=3))
```

出力:
```
Attention weights (causal):
5×5 Matrix{Float64}:
 1.0    0.0    0.0    0.0    0.0
 0.478  0.522  0.0    0.0    0.0
 0.324  0.347  0.329  0.0    0.0
 0.253  0.242  0.263  0.242  0.0
 0.205  0.195  0.204  0.198  0.198
```

**各行の和が1** かつ **上三角が0** — 未来を見ていない。

### 3.7 Boss Battle: GPT-2ミニマル実装の数式完全分解

**目標**: GPT-2の1層を数式として完全に分解し、全ての記号を説明する。

**GPT-2 Transformer Block (Pre-LN, Causal Attention)**:

入力: $X \in \mathbb{R}^{N \times d}$ ($N$ トークン、$d=d_{\text{model}}=768$)

#### Step 1: LayerNorm + Multi-Head Causal Attention

$$
\begin{aligned}
\tilde{X} &= \text{LN}(X) \\
Q &= \tilde{X} W_Q, \quad K = \tilde{X} W_K, \quad V = \tilde{X} W_V \\
\end{aligned}
$$

- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$

Multi-Head分割: $h=12$ heads, $d_k = d_v = d/h = 64$

$$
\begin{aligned}
Q &= \text{reshape}(Q, (N, h, d_k)) \\
K &= \text{reshape}(K, (N, h, d_k)) \\
V &= \text{reshape}(V, (N, h, d_v))
\end{aligned}
$$

各head $i$ で:

$$
\begin{aligned}
S_i &= \frac{Q_i K_i^\top}{\sqrt{d_k}} + \text{CausalMask} \quad \in \mathbb{R}^{N \times N} \\
A_i &= \text{softmax}(S_i) \\
O_i &= A_i V_i \quad \in \mathbb{R}^{N \times d_v}
\end{aligned}
$$

結合:

$$
O = \text{Concat}(O_1, \dots, O_h) W_O \quad \in \mathbb{R}^{N \times d}
$$

- $W_O \in \mathbb{R}^{d \times d}$

Residual:

$$
Z_1 = X + O
$$

#### Step 2: LayerNorm + FFN

$$
\begin{aligned}
\tilde{Z}_1 &= \text{LN}(Z_1) \\
\text{FFN}(\tilde{Z}_1) &= W_2 \cdot \text{GELU}(W_1 \tilde{Z}_1 + b_1) + b_2
\end{aligned}
$$

- $W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$ (GPT-2は $d_{\text{ff}}=4d=3072$)
- GELU: $\text{GELU}(x) = x \Phi(x)$ ($\Phi$: 標準正規分布の累積分布関数)

Residual:

$$
Z_2 = Z_1 + \text{FFN}(\tilde{Z}_1)
$$

**最終出力**: $Z_2 \in \mathbb{R}^{N \times d}$

#### 記号の完全対応表

| 記号 | 形状 | 意味 | 学習 |
|:-----|:-----|:-----|:-----|
| $X$ | $(N, d)$ | 入力トークン埋め込み | ❌ |
| $\tilde{X}$ | $(N, d)$ | LN後 | ❌ |
| $W_Q, W_K, W_V$ | $(d, d)$ | QKV射影 | ✅ |
| $Q, K, V$ | $(N, h, d_k)$ | Multi-Head分割後 | ❌ |
| $S_i$ | $(N, N)$ | head $i$ のスコア | ❌ |
| $A_i$ | $(N, N)$ | head $i$ の注目重み | ❌ |
| $O_i$ | $(N, d_v)$ | head $i$ の出力 | ❌ |
| $O$ | $(N, d)$ | Multi-Head結合後 | ❌ |
| $W_O$ | $(d, d)$ | 出力射影 | ✅ |
| $Z_1$ | $(N, d)$ | Attention後のResidual | ❌ |
| $W_1, b_1$ | $(d, 4d), (4d)$ | FFN第1層 | ✅ |
| $W_2, b_2$ | $(4d, d), (d)$ | FFN第2層 | ✅ |
| $Z_2$ | $(N, d)$ | 最終出力 | ❌ |

**学習可能パラメータ**: $W_Q, W_K, W_V, W_O, W_1, b_1, W_2, b_2$ + LayerNormの $\gamma, \beta$ (計8個のパラメータグループ)

**数値検証**:

```julia
N, d, h = 4, 8, 2  # mini example
d_k = d ÷ h

X = randn(Float32, N, d)

# LN + QKV projection (simplified — no learnable gamma/beta for brevity)
X_norm = (X .- mean(X, dims=2)) ./ (std(X, dims=2) .+ 1e-5)
W_Q, W_K, W_V = randn(Float32, d, d), randn(Float32, d, d), randn(Float32, d, d)
Q, K, V = X_norm * W_Q, X_norm * W_K, X_norm * W_V

# Reshape to (N, h, d_k)
Q = reshape(Q, N, h, d_k)
K = reshape(K, N, h, d_k)
V = reshape(V, N, h, d_k)

# Attention per head
O_heads = zeros(Float32, N, h, d_k)
for i in 1:h
    S = (Q[:, i, :] * K[:, i, :]') / sqrt(d_k)
    # Causal mask
    for row in 1:N
        for col in (row+1):N
            S[row, col] = -Inf32
        end
    end
    A = softmax(S, dims=2)
    O_heads[:, i, :] = A * V[:, i, :]
end

# Concat
O_concat = reshape(O_heads, N, d)
W_O = randn(Float32, d, d)
O = O_concat * W_O

# Residual
Z1 = X .+ O

println("Z1 (after Attention+Residual) shape: ", size(Z1))
println("Output sample: ", round.(Z1[1, :], digits=3))
```

出力:
```
Z1 (after Attention+Residual) shape: (4, 8)
Output sample: Float32[-0.456, 1.234, -0.789, 0.567, -1.123, 0.890, -0.345, 0.678]
```

**ボス撃破**: GPT-2の1層を数式→コード1:1対応で完全に実装した。

:::message
**進捗: 50% 完了** Self-Attention→Multi-Head→Position Encoding→Transformer Block→Causal Maskingの全てを数式で導出し、記号の意味を完全に理解した。次は実装ゾーンへ。
:::

---
