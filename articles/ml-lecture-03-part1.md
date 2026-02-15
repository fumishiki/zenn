---
title: "第3回: 線形代数 II: SVD・行列微分・テンソル — 万能ナイフSVDと逆伝播の数学 【前編】理論編"
emoji: "🔬"
type: "tech"
topics: ["machinelearning", "deeplearning", "linearalgebra", "python"]
published: true
---


# 第3回: 線形代数 II — SVD・行列微分・テンソル

> **SVDは万能ナイフだ。画像圧縮もPCAも推薦も、全て「同じ計算」に帰着する。**

第2回で線形代数の基盤を築いた。ベクトル空間の公理、行列演算、固有値分解、正定値行列、射影 — これらは全て「正方行列」の世界の話だった。

だが、現実のデータは正方行列ではない。画像は $3 \times 224 \times 224$ のテンソルだ。言語モデルの重み行列は $d_{\text{model}} \times d_{\text{ff}}$ の長方形行列だ。バッチ処理されたAttentionスコアは $B \times H \times T \times T$ の4階テンソルだ。

**正方行列の外の世界**を扱うために、3つの道具が必要になる:

1. **SVD**（特異値分解）— 任意の行列を分解する「万能ナイフ」
2. **行列微分** — ニューラルネットワーク学習の数学的基盤
3. **テンソル演算** — 多次元配列を数学的に扱う言語

この3つを本講義で完全武装する。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論（論文が書ける）、実装（Production-ready）、最新（2025-2026 SOTA）の3軸で差別化する。
:::

```mermaid
graph LR
    A["🔬 SVD<br/>特異値分解"] --> B["📉 低ランク近似<br/>Eckart-Young"]
    B --> C["📊 PCA (SVD版)<br/>分散最大化"]
    A --> D["📐 擬似逆行列<br/>Moore-Penrose"]
    E["✏️ 行列微分<br/>ヤコビアン・ヘシアン"] --> F["⛓️ 連鎖律<br/>計算グラフ"]
    F --> G["🔄 自動微分<br/>Forward/Reverse"]
    style A fill:#e1f5fe
    style G fill:#c8e6c9
```

**所要時間の目安**:

| ゾーン | 内容 | 時間 | 難易度 |
|:-------|:-----|:-----|:-------|
| Zone 0 | クイックスタート | 30秒 | ★☆☆☆☆ |
| Zone 1 | 体験ゾーン | 10分 | ★★☆☆☆ |
| Zone 2 | 直感ゾーン | 15分 | ★★☆☆☆ |
| Zone 3 | 数式修行ゾーン | 60分 | ★★★★★ |
| Zone 4 | 実装ゾーン | 45分 | ★★★☆☆ |
| Zone 5 | 実験ゾーン | 30分 | ★★★☆☆ |
| Zone 6 | 振り返りゾーン | 30分 | ★★★★☆ |

---

## 🚀 0. クイックスタート（30秒）— SVDで画像を圧縮する

**ゴール**: SVDが「データの本質的な構造を抽出する道具」であることを30秒で体感する。

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# grayscale image as matrix
np.random.seed(42)
A = np.random.randn(100, 80)  # 100×80 matrix (like a small grayscale image)

# SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Rank-5 approximation
k = 5
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Compression ratio
original_params = A.shape[0] * A.shape[1]  # 8000
compressed_params = k * (A.shape[0] + A.shape[1] + 1)  # 5 * 181 = 905
print(f"Original:    {original_params} parameters")
print(f"Compressed:  {compressed_params} parameters (rank-{k})")
print(f"Compression: {compressed_params/original_params:.1%}")
print(f"Error:       {np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'):.4f}")
```

出力:
```
Original:    8000 parameters
Compressed:  905 parameters (rank-5)
Compression: 11.3%
Error:       0.8716
```

**この5行の裏にある数学**:

$$
A = U \Sigma V^\top = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

任意の行列 $A \in \mathbb{R}^{m \times n}$ を、直交行列 $U$、対角行列 $\Sigma$、直交行列 $V^\top$ の積に分解する。上位 $k$ 個の特異値だけを残せば、**最適な** rank-$k$ 近似が得られる[^3]。「最適」の意味はEckart-Young定理が保証する。

:::message
**進捗: 3% 完了** SVDで行列を圧縮できることを体感した。残り7ゾーンの冒険が待っている。
:::

---

## 🎮 1. 体験ゾーン（10分）— SVDと行列微分を「触って」理解する

### 1.1 SVDの幾何学 — 行列は「回転→拡大→回転」

第2回で「行列は線形変換」と言った。SVDは、その変換を3つの基本操作に分解する。

$$
A = U \Sigma V^\top
$$

| 成分 | 幾何学的意味 | 行列の型 |
|:-----|:-----------|:---------|
| $V^\top$ | 入力空間での回転（直交変換） | $n \times n$ 直交行列 |
| $\Sigma$ | 各軸方向の拡大（スケーリング） | $m \times n$ 対角行列 |
| $U$ | 出力空間での回転（直交変換） | $m \times m$ 直交行列 |

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D example: matrix transforms a unit circle
A = np.array([[3, 1],
              [1, 2]])

# SVD
U, s, Vt = np.linalg.svd(A)
print(f"U = \n{np.round(U, 4)}")
print(f"Singular values = {np.round(s, 4)}")
print(f"Vt = \n{np.round(Vt, 4)}")

# Unit circle
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Apply each SVD step
step1 = Vt @ circle         # V^T: rotate in input space
step2 = np.diag(s) @ step1  # Sigma: scale
step3 = U @ step2           # U: rotate in output space

# Verify: A @ circle == U @ Sigma @ Vt @ circle
direct = A @ circle
print(f"\nSVD reconstruction matches: {np.allclose(step3, direct)}")
```

**核心**: どんな行列による変換も「回転 → 拡大 → 回転」に分解できる。特異値 $\sigma_1, \sigma_2, \ldots$ は拡大率を表し、降順にソートされている。

```mermaid
graph LR
    IN["🔵 単位円<br/>(入力)"] -->|"V^T<br/>回転"| R1["🔵 回転された円"]
    R1 -->|"Σ<br/>拡大"| EL["🔴 楕円"]
    EL -->|"U<br/>回転"| OUT["🔴 回転された楕円<br/>(出力)"]

    style IN fill:#e3f2fd
    style OUT fill:#ffcdd2
```

### 1.2 特異値の減衰 — なぜ低ランク近似が有効なのか

実データの行列は、特異値が急速に減衰する。これが低ランク近似やPCA[^5][^6]が有効な理由だ。

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: create a matrix with rapid singular value decay
np.random.seed(42)
# Low-rank structure + noise
rank_true = 5
m, n = 100, 80
U_true = np.linalg.qr(np.random.randn(m, rank_true))[0]
V_true = np.linalg.qr(np.random.randn(n, rank_true))[0]
s_true = np.array([10, 5, 2, 1, 0.5])
A_clean = U_true @ np.diag(s_true) @ V_true.T
A_noisy = A_clean + 0.1 * np.random.randn(m, n)

# SVD of noisy matrix
U, s, Vt = np.linalg.svd(A_noisy, full_matrices=False)

print("Top 10 singular values:")
for i, sv in enumerate(s[:10]):
    bar = "█" * int(sv * 3)
    print(f"  σ_{i+1:2d} = {sv:8.4f}  {bar}")

# Cumulative energy
energy = np.cumsum(s**2) / np.sum(s**2)
print(f"\nCumulative energy:")
for k in [1, 2, 3, 5, 10, 20]:
    if k <= len(energy):
        print(f"  rank-{k:2d}: {energy[k-1]:.4f} ({energy[k-1]*100:.1f}%)")
```

**重要な洞察**: 上位5個の特異値だけで元の行列のエネルギー（Frobenius ノルムの二乗）の99%以上を捕捉できる。これは元の行列が「本質的に rank-5」であることを意味する。

### 1.3 勾配を「見る」— 損失関数の地形

Backpropagation[^2]の核心は勾配の計算だ。勾配とは「損失関数がどの方向にどれだけ変化するか」を表すベクトル。

```python
import numpy as np

# Simple loss: L(w) = (y - w^T x)^2
x = np.array([1.0, 2.0, 3.0])
y = 10.0
w = np.array([1.0, 1.0, 1.0])

# Forward pass
y_pred = w @ x  # w^T x = 6
loss = (y - y_pred) ** 2  # (10 - 6)^2 = 16
print(f"y_pred = {y_pred}, loss = {loss}")

# Gradient: dL/dw = -2(y - w^T x) * x
grad = -2 * (y - y_pred) * x
print(f"gradient = {grad}")

# Gradient descent step
lr = 0.01
w_new = w - lr * grad
y_pred_new = w_new @ x
loss_new = (y - y_pred_new) ** 2
print(f"After update: y_pred = {y_pred_new:.4f}, loss = {loss_new:.4f}")
print(f"Loss decreased: {loss:.4f} → {loss_new:.4f}")
```

**勾配 $\nabla_{\mathbf{w}} L$ は「損失を最も速く減少させる方向」**の逆方向だ。$-\nabla L$ の方向にパラメータを動かすのが勾配降下法。

### 1.4 ヤコビアンを「見る」— ベクトル→ベクトル関数の微分

スカラー関数の勾配は「ベクトル」だった。では、ベクトルからベクトルへの関数の微分は？ — それが**ヤコビアン**（Jacobian matrix）。

$$
\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m, \quad J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}
$$

```python
import numpy as np

# f: R^2 -> R^2, f(x) = [x1^2 + x2, x1 * x2]
def f(x):
    return np.array([x[0]**2 + x[1], x[0] * x[1]])

# Analytical Jacobian
def jacobian(x):
    return np.array([
        [2 * x[0], 1],        # df1/dx1, df1/dx2
        [x[1],     x[0]]      # df2/dx1, df2/dx2
    ])

# Numerical Jacobian (finite differences)
def numerical_jacobian(f, x, eps=1e-7):
    n = len(x)
    m = len(f(x))
    J = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        x_minus = x.copy()
        x_minus[j] -= eps
        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return J

x = np.array([2.0, 3.0])
J_analytical = jacobian(x)
J_numerical = numerical_jacobian(f, x)

print(f"Analytical Jacobian:\n{J_analytical}")
print(f"Numerical Jacobian:\n{np.round(J_numerical, 6)}")
print(f"Match: {np.allclose(J_analytical, J_numerical, atol=1e-5)}")
```

**ヤコビアンの各行は、出力の各成分の勾配**。ヤコビアンの行列式 $\det(J)$ は「変換による体積の変化率」を表し、Normalizing Flow[^13]の核心的な計算量のボトルネックになる。

### 1.5 自動微分の威力 — PyTorchの `backward()` が内部でやっていること

```python
# PyTorch-style automatic differentiation (manual implementation)
import numpy as np

class Var:
    """Simple autograd variable for demonstration"""
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(_children)
        self._op = _op

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def backward(self):
        # topological sort
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

# Demo: f(a, b) = a*b + b
a = Var(2.0)
b = Var(3.0)
c = a * b      # c = 6
d = c + b      # d = 9
d.backward()

print(f"d = {d.data}")        # 9.0
print(f"dd/da = {a.grad}")    # b = 3.0 (correct: d(ab+b)/da = b)
print(f"dd/db = {b.grad}")    # a+1 = 3.0 (correct: d(ab+b)/db = a+1)
```

このたった50行のコードが、PyTorchの `loss.backward()` の本質だ[^7][^8]。計算の「記録」を逆順に辿って勾配を伝播する — これが**Reverse Mode 自動微分**であり、Backpropagation[^2]の正体だ。

:::message
**進捗: 15% 完了** SVDの幾何学、特異値の減衰、勾配、ヤコビアン、自動微分の基本を体験した。ここから直感を深めてZone 3の数式修行に備える。
:::

---

## 🧩 2. 直感ゾーン（15分）— SVDと自動微分がAIを支える理由

### 2.1 第3回の「地図」

第2回で線形代数の「文法」を学んだ。第3回では「修辞法」を学ぶ。

| 道具 | 比喩 | 機械学習での役割 |
|:-----|:-----|:--------------|
| **SVD** | 万能ナイフ | データの本質的構造を抽出（PCA, LoRA[^10], 推薦） |
| **行列微分** | 羅針盤 | 損失関数の勾配方向を示す |
| **連鎖律** | 連鎖反応 | 多層ネットワークの全パラメータの勾配を一括計算 |
| **自動微分** | 自動翻訳機 | 数式→勾配計算コードの自動変換 |
| **テンソル演算** | 多次元の文法 | バッチ・ヘッド・シーケンスの一括処理 |

### 2.2 Course I の中での位置づけ

```mermaid
graph TD
    L1["第1回: 概論<br/>数式と論文の読み方"]
    L2["第2回: 線形代数 I<br/>ベクトル・行列・固有値"]
    L3["第3回: 線形代数 II<br/>SVD・行列微分・テンソル<br/>🎯 Backprop完全導出"]
    L4["第4回: 確率論・統計学<br/>分布・ベイズ推論"]

    L1 -->|"数式が読めた"| L2
    L2 -->|"行列を扱えた"| L3
    L3 -->|"微分もできた"| L4

    style L3 fill:#ffeb3b
```

| 回 | テーマ | LLM/Transformerとの接点 |
|:---|:------|:----------------------|
| 第2回 | 線形代数 I | $QK^\top$ の内積、固有値→PCA→埋め込み |
| **第3回** | **線形代数 II** | **ヤコビアン→Flow Model、勾配→Backprop、連鎖律→Transformer各層** |
| 第4回 | 確率論・統計学 | $p(x_t \mid x_{<t})$ 自己回帰、Softmax分布 |

**第2回→第3回の接続**: 第2回で固有値分解を学んだ。だが固有値分解は正方行列にしか使えない。SVDはその制約を取り払い、**任意の長方形行列**を分解できる万能ツールだ。

### 2.3 松尾研との差別化

| 松尾研の前提 | 実際の壁 | 本講義の対策 |
|:------------|:--------|:-----------|
| 「SVDは知ってるよね」 | Eckart-Young定理[^3]の意味が説明できない | 存在定理→幾何学→最適性を全導出 |
| 「Backpropは理解してるよね」 | 行列微分の連鎖律が書けない | ヤコビアン→連鎖律→Backpropを一から導出 |
| 「自動微分は PyTorch に任せて」 | Forward/Reverse の計算量の差がわからない | Wengert list から Forward/Reverse を手動実装 |
| 「テンソルはNumPyの配列」 | 添字の縮約規則が読めない | Einstein記法→einsum完全版 |

### 2.4 LLMの中のSVDと行列微分

LLMの学習と推論の両方で、SVDと行列微分が使われている。

```mermaid
graph TD
    subgraph "🎯 推論（Forward Pass）"
        EMB["埋め込み<br/>E[x_t]"] --> ATTN["Attention<br/>QK^T/√d"]
        ATTN --> FFN["FFN<br/>W_2 σ(W_1 h)"]
        FFN --> OUT["出力<br/>logits"]
    end

    subgraph "🔄 学習（Backward Pass）"
        LOSS["Loss = -log p(x_t)"] --> GRAD_OUT["∂L/∂logits<br/>ヤコビアン"]
        GRAD_OUT --> GRAD_FFN["∂L/∂W_1, ∂L/∂W_2<br/>連鎖律"]
        GRAD_FFN --> GRAD_ATTN["∂L/∂W_Q, ∂L/∂W_K, ∂L/∂W_V<br/>連鎖律"]
        GRAD_ATTN --> UPDATE["Adam更新<br/>W ← W - lr·m̂/(√v̂+ε)"]
    end

    subgraph "🔧 効率化"
        LORA["LoRA: ΔW = BA<br/>低ランク近似 (SVD的発想)"]
        PRUNE["構造化枝刈り<br/>SVDで重要度判定"]
    end

    OUT -.->|"Cross-Entropy"| LOSS
    UPDATE -.->|"重み更新"| EMB
    LORA -.->|"パラメータ効率化"| UPDATE
    PRUNE -.->|"モデル圧縮"| FFN

    style LOSS fill:#ffcdd2
    style LORA fill:#c8e6c9
```

| LLMの操作 | 第3回の対応セクション | なぜ必要か |
|:----------|:-------------------|:---------|
| Forward pass | 3.7 連鎖律 | 各層の出力を順に計算 |
| Backward pass | 3.7 連鎖律 + 3.8 Backprop | 全パラメータの勾配を逆順に計算 |
| LoRA | 3.3 低ランク近似 | 重み更新を rank-$r$ で近似 |
| Adam optimizer | 3.6 勾配 | 一次・二次モーメントの推定 |
| 勾配クリッピング | 3.6 ヤコビアン | 勾配爆発の防止 |

### 2.5 3つの比喩で捉える本講義の本質

**比喩1: SVDは「顕微鏡」**

行列の「微細構造」を特異値という数値で読み取る。大きい特異値 = 重要な構造、小さい特異値 = ノイズ。顕微鏡の倍率を変えるように、残す特異値の数（ランク $k$）を変えることで、粗い構造から精密な構造まで見える。

**比喩2: 行列微分は「高次元の傾き」**

2次元で $y = f(x)$ の傾きが $f'(x)$ だったように、高次元で $\mathbf{y} = \mathbf{f}(\mathbf{x})$ の「傾き」がヤコビアン $J$ だ。ヤコビアンは「入力の微小変化が出力にどう伝播するか」を行列として表現する。

**比喩3: 自動微分は「計算の録画と巻き戻し」**

Forward passで計算を「録画」し、Backward passで「巻き戻し」ながら勾配を計算する。VHSテープの巻き戻しと同じで、最後に計算した部分から順に勾配が求まる。

### 2.6 学習戦略

この講義は第2回よりもさらに数式が多い。心構え:

1. **Zone 3 が最重要**。90分を惜しまない
2. **SVD → 行列微分 → 自動微分** の順で学ぶ（各トピックが前のトピックに依存する）
3. **数値検証を怠らない**: 解析的な結果は必ずコードで確認する
4. **紙に書く**: 2×2行列のSVDを手計算で1回やると理解が段違いに深まる
5. **Zone 5 で腕試し**: SVD画像圧縮と自動微分の手動実装が、理解度の最良のテスト

### 2.7 SVD・行列微分の機械学習における位置づけ

```mermaid
graph TD
    SVD["SVD<br/>特異値分解"]
    AD["自動微分<br/>Forward/Reverse"]
    CALC["行列微分<br/>ヤコビアン・ヘシアン"]

    SVD --> PCA_["PCA<br/>次元削減"]
    SVD --> LORA["LoRA<br/>パラメータ効率化"]
    SVD --> REC["推薦システム<br/>協調フィルタリング"]
    SVD --> COMPRESS["モデル圧縮<br/>低ランク近似"]
    SVD --> NMF["NMF<br/>非負行列分解"]

    AD --> BACKPROP["Backpropagation<br/>ニューラルネット学習"]
    AD --> JAX_["JAX<br/>関数変換"]
    AD --> PHYSICS["Physics-Informed NN<br/>微分方程式"]

    CALC --> NF["Normalizing Flow<br/>ヤコビアン行列式"]
    CALC --> NATURAL["Natural Gradient<br/>Fisher情報行列"]
    CALC --> HESSIAN["二次最適化<br/>Newton法"]

    style SVD fill:#e3f2fd
    style AD fill:#c8e6c9
    style CALC fill:#fff9c4
```

| 技術 | 関連する数学 | 応用 | 講義 |
|:-----|:-----------|:-----|:-----|
| LoRA[^10] | SVD + 低ランク近似 | LLMのファインチューニング | 本講義 |
| FlashAttention[^12] | 行列のブロック分割 | Attention高速化 | 第2回 |
| Normalizing Flow[^13] | ヤコビアン行列式 | 確率密度変換 | 第25回 |
| Natural Gradient | Fisher情報行列 | 最適化の幾何学 | 第27回 |
| Neural ODE | 自動微分 + ODE | 連続深度モデル | 第26回 |
| Spectral Normalization | SVDの最大特異値 | GAN安定化 | 第14回 |

### 2.8 自動微分フレームワークの進化

```mermaid
timeline
    title 自動微分の歴史
    1964 : Wengert が AD の基本アイデアを発表
    1970 : Linnainmaa が Reverse Mode を定式化
    1986 : Rumelhart-Hinton が Backpropagation として再発見
    2007 : Theano — 記号微分 + コンパイル
    2015 : TensorFlow 1.x — 静的計算グラフ
    2016 : PyTorch — 動的計算グラフ (Define-by-Run)
    2018 : JAX — 関数変換 (grad, jit, vmap, pmap)
    2020 : PyTorch 2.0 — torch.compile (動的+静的の融合)
    2024 : Reactant.jl — Julia + XLA コンパイル
```

| フレームワーク | AD方式 | 特徴 | 長所 |
|:-------------|:------|:-----|:-----|
| PyTorch | Reverse (tape-based) | Define-by-Run | 柔軟、デバッグしやすい |
| JAX | Forward + Reverse (tracing) | 関数変換 | `grad`, `vmap`, `jit` の合成 |
| TensorFlow | Reverse (graph-based) | 静的最適化 | デプロイに強い |
| Zygote.jl | Source-to-source | Julia AST変換 | 任意のJuliaコードに適用可能 |
| Enzyme | LLVM IR レベル | コンパイラ統合 | 言語非依存 |

:::details JAX の関数変換: grad, jit, vmap
JAXの革新は、自動微分を「関数変換」として扱うこと。

```python
# JAX-style function transforms (conceptual)
# grad: f → ∇f
# jit: f → compiled f
# vmap: f → batched f

# Real JAX code would look like:
# import jax
# import jax.numpy as jnp
#
# def loss(params, x, y):
#     pred = params @ x
#     return jnp.sum((pred - y)**2)
#
# grad_fn = jax.grad(loss)        # returns gradient function
# fast_grad = jax.jit(grad_fn)    # compile for speed
# batch_grad = jax.vmap(grad_fn)  # vectorize over batch
```

`grad` が返すのは**関数**。これにより「勾配の勾配」（ヘシアン）も簡単に計算できる:

```python
# hessian = jax.hessian(loss)  # ∇²f
# jvp = jax.jvp(f, primals, tangents)  # Forward Mode
# vjp = jax.vjp(f, primals)  # Reverse Mode
```
:::

:::message
**進捗: 20% 完了** SVD・行列微分・自動微分の全体像を掴んだ。ここからZone 3「数式修行ゾーン」— 本講義最大の山場だ。
:::

---

## 📐 3. 数式修行ゾーン（60分）— SVDから自動微分まで

> **目標**: SVDの存在定理と最適性、行列微分の体系、連鎖律、自動微分の理論を導出し、Backpropagationの数学的基盤を完全理解する。

本シリーズで最も数式密度が高いゾーンだ。だが、ここで学ぶ全ての概念は、第9回以降の生成モデルで繰り返し登場する。一つずつ、確実に理解していこう。

### 3.1 SVD（特異値分解）の定義と存在定理

#### 定義

**定理** (特異値分解): 任意の行列 $A \in \mathbb{R}^{m \times n}$ に対して、以下の分解が存在する:

$$
A = U \Sigma V^\top
$$

ここで:
- $U \in \mathbb{R}^{m \times m}$: 直交行列（$U^\top U = I_m$）— **左特異ベクトル**
- $\Sigma \in \mathbb{R}^{m \times n}$: 対角行列（$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$）— **特異値**
- $V \in \mathbb{R}^{n \times n}$: 直交行列（$V^\top V = I_n$）— **右特異ベクトル**
- $r = \text{rank}(A)$

#### 固有値分解との関係

SVDの存在は、固有値分解から導ける。

$A^\top A$ は $n \times n$ の半正定値対称行列なので、スペクトル定理より直交対角化可能:

$$
A^\top A = V \Lambda V^\top, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n), \quad \lambda_1 \geq \cdots \geq \lambda_n \geq 0
$$

特異値を $\sigma_i = \sqrt{\lambda_i}$ と定義する。$\sigma_i > 0$ の個数が $r = \text{rank}(A)$。

左特異ベクトルは:

$$
\mathbf{u}_i = \frac{A \mathbf{v}_i}{\sigma_i} \quad (i = 1, \ldots, r)
$$

**検証**:

$$
A = U \Sigma V^\top \implies A^\top A = V \Sigma^\top U^\top U \Sigma V^\top = V \Sigma^\top \Sigma V^\top = V \Lambda V^\top \quad \checkmark
$$

同様に $AA^\top = U \Lambda' U^\top$（$\Lambda'$ の非ゼロ対角要素は $\Lambda$ と同じ）。

```python
import numpy as np

# Verify SVD via eigendecomposition
A = np.array([[3, 2, 2],
              [2, 3, -2]])

# Method 1: np.linalg.svd
U, s, Vt = np.linalg.svd(A)
print("SVD:")
print(f"  Singular values: {np.round(s, 4)}")

# Method 2: eigendecomposition of A^T A
AtA = A.T @ A
eigenvalues, V_eig = np.linalg.eigh(AtA)
# eigh returns ascending order, reverse for descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
V_eig = V_eig[:, idx]

print(f"\nEigenvalues of A^T A: {np.round(eigenvalues, 4)}")
print(f"Singular values (sqrt): {np.round(np.sqrt(np.maximum(eigenvalues, 0)), 4)}")
print(f"Match: {np.allclose(s, np.sqrt(np.maximum(eigenvalues, 0))[:len(s)])}")
```

#### Compact SVD と Economy SVD

Full SVDは計算量が無駄になることが多い。実用上は以下を使う:

| 名称 | 定義 | サイズ | 用途 |
|:-----|:-----|:------|:-----|
| Full SVD | $A = U \Sigma V^\top$ | $U: m \times m, \Sigma: m \times n, V: n \times n$ | 理論 |
| Compact SVD | $A = U_r \Sigma_r V_r^\top$ | $U_r: m \times r, \Sigma_r: r \times r, V_r: r \times n$ | $\text{rank}(A) = r \ll \min(m,n)$ |
| Truncated SVD | $A_k = U_k \Sigma_k V_k^\top$ | $U_k: m \times k, \Sigma_k: k \times k, V_k: k \times n$ | 低ランク近似 |

```python
import numpy as np

A = np.random.randn(100, 50)

# Full SVD
U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=True)
print(f"Full SVD: U={U_full.shape}, s={s_full.shape}, Vt={Vt_full.shape}")

# Economy SVD (full_matrices=False)
U_econ, s_econ, Vt_econ = np.linalg.svd(A, full_matrices=False)
print(f"Economy SVD: U={U_econ.shape}, s={s_econ.shape}, Vt={Vt_econ.shape}")

# Truncated SVD (rank-k)
k = 10
U_k = U_econ[:, :k]
s_k = s_econ[:k]
Vt_k = Vt_econ[:k, :]
A_k = U_k @ np.diag(s_k) @ Vt_k
print(f"Truncated SVD (k={k}): error = {np.linalg.norm(A - A_k, 'fro'):.4f}")
```

### 3.2 Eckart-Young定理 — 低ランク近似の最適性

#### 定理

**定理** (Eckart-Young-Mirsky[^3]): $A \in \mathbb{R}^{m \times n}$ の SVD を $A = U \Sigma V^\top$ とし、$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ を特異値とする。任意の rank-$k$ 行列 $B$ に対して:

$$
\min_{\text{rank}(B) \leq k} \|A - B\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}
$$

この最小値を達成する $B$ は:

$$
A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = U_k \Sigma_k V_k^\top
$$

スペクトルノルムについても:

$$
\min_{\text{rank}(B) \leq k} \|A - B\|_2 = \sigma_{k+1}
$$

#### 証明のスケッチ

$B$ を任意の rank-$k$ 行列とする。$\ker(B)$ の次元は $n - k$ 以上。一方、$V_1, \ldots, V_{k+1}$ が張る部分空間は $k+1$ 次元。次元の引数（dimension argument）より、$\ker(B)$ と $\text{span}\{V_1, \ldots, V_{k+1}\}$ は非自明な交わりを持つ。

$\mathbf{w} \neq \mathbf{0}$ をこの交わりの要素とすると:

$$
\|A - B\|_F^2 \geq \|(A-B)\mathbf{w}\|^2 / \|\mathbf{w}\|^2 = \|A\mathbf{w}\|^2 / \|\mathbf{w}\|^2
$$

$\mathbf{w} \in \text{span}\{V_1, \ldots, V_{k+1}\}$ より、$\mathbf{w} = \sum_{i=1}^{k+1} c_i \mathbf{v}_i$ と書ける。

$$
\|A\mathbf{w}\|^2 = \sum_{i=1}^{k+1} c_i^2 \sigma_i^2 \geq \sigma_{k+1}^2 \sum_{i=1}^{k+1} c_i^2 = \sigma_{k+1}^2 \|\mathbf{w}\|^2
$$

したがって $\|A - B\|_2 \geq \sigma_{k+1}$。$A_k$ がこの下界を達成することは直接計算で確認できる。$\square$

:::message alert
上記の証明スケッチはスペクトルノルム版です。フロベニウスノルム版の最適性は $\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$ の直接計算で示されます（Fan-Hoffman不等式）。
:::

```python
import numpy as np

# Verify Eckart-Young theorem
A = np.random.randn(50, 30)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

for k in [1, 3, 5, 10, 20]:
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    error_F = np.linalg.norm(A - A_k, 'fro')
    theoretical = np.sqrt(np.sum(s[k:]**2))
    print(f"rank-{k:2d}: ||A-A_k||_F = {error_F:.6f}, "
          f"theoretical = {theoretical:.6f}, "
          f"match = {np.isclose(error_F, theoretical)}")
```

:::message
**LoRAへの接続**: LoRA[^10]は、ファインチューニング時の重み更新 $\Delta W$ を低ランク行列 $BA$ で近似する。$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$ で $r \ll d$。Eckart-Young定理は「低ランク近似は最適」を保証するが、LoRAの場合は学習で $B, A$ を最適化するため、SVDとは異なるアプローチ。だが、学習後の $\Delta W = BA$ をSVDで分析すると、確かに少数の特異値が支配的であることが確認される。
:::

### 3.3 低ランク近似の応用 — 画像圧縮・推薦・LoRA

#### 画像圧縮

```python
import numpy as np

# Create a test image-like matrix (smooth gradients + structure)
m, n = 200, 150
x = np.linspace(0, 4*np.pi, m)
y = np.linspace(0, 3*np.pi, n)
X, Y = np.meshgrid(y, x)
A = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X + Y)  # structured image

U, s, Vt = np.linalg.svd(A, full_matrices=False)

print("Singular value decay:")
for k in [1, 5, 10, 20, 50]:
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    rel_error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
    storage_original = m * n
    storage_compressed = k * (m + n + 1)
    ratio = storage_compressed / storage_original
    print(f"  rank-{k:2d}: error={rel_error:.6f}, "
          f"storage={ratio:.1%} ({storage_compressed}/{storage_original})")
```

#### 推薦システム（協調フィルタリング）

ユーザー×アイテムの評価行列 $R$ は大部分が欠損（未評価）。低ランク近似 $R \approx U_k \Sigma_k V_k^\top$ で欠損値を予測できる。

$$
\hat{r}_{ij} = \sum_{l=1}^{k} \sigma_l u_{il} v_{jl}
$$

```python
import numpy as np

# Toy recommendation: 5 users × 4 items
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
], dtype=float)

# Replace 0 (unknown) with mean for SVD
mask = R > 0
R_filled = R.copy()
R_filled[~mask] = np.mean(R[mask])

U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)

# Rank-2 approximation
k = 2
R_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

print("Original (0 = unknown):")
print(R.astype(int))
print(f"\nRank-{k} approximation (predictions for unknowns):")
print(np.round(R_approx, 1))
print(f"\nPredicted ratings for unknown entries:")
for i, j in zip(*np.where(~mask)):
    print(f"  User {i+1}, Item {j+1}: {R_approx[i,j]:.1f}")
```

### 3.4 擬似逆行列（Moore-Penrose）

#### 定義

$A \in \mathbb{R}^{m \times n}$ の **Moore-Penrose 擬似逆行列** $A^+ \in \mathbb{R}^{n \times m}$ は以下の4条件を満たす唯一の行列:

1. $A A^+ A = A$
2. $A^+ A A^+ = A^+$
3. $(A A^+)^\top = A A^+$
4. $(A^+ A)^\top = A^+ A$

#### SVDによる構成

$A = U \Sigma V^\top$ ならば:

$$
A^+ = V \Sigma^+ U^\top
$$

ここで $\Sigma^+ = \text{diag}(1/\sigma_1, \ldots, 1/\sigma_r, 0, \ldots, 0)$。

**直感**: 特異値の逆数を取る。ただし $\sigma_i = 0$ の成分は無視する。

```python
import numpy as np

# Pseudoinverse via SVD
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Method 1: np.linalg.pinv
A_pinv = np.linalg.pinv(A)

# Method 2: manual SVD construction
U, s, Vt = np.linalg.svd(A, full_matrices=False)
S_pinv = np.diag(1.0 / s)
A_pinv_manual = Vt.T @ S_pinv @ U.T

print(f"A (shape {A.shape}):")
print(A)
print(f"\nA+ (shape {A_pinv.shape}):")
print(np.round(A_pinv, 4))
print(f"\nManual matches: {np.allclose(A_pinv, A_pinv_manual)}")

# Verify Moore-Penrose conditions
print(f"\nMoore-Penrose conditions:")
print(f"  A A+ A = A: {np.allclose(A @ A_pinv @ A, A)}")
print(f"  A+ A A+ = A+: {np.allclose(A_pinv @ A @ A_pinv, A_pinv)}")
print(f"  (A A+)^T = A A+: {np.allclose((A @ A_pinv).T, A @ A_pinv)}")
print(f"  (A+ A)^T = A+ A: {np.allclose((A_pinv @ A).T, A_pinv @ A)}")
```

#### 最小二乗法との関係

過剰決定系 $A\mathbf{x} = \mathbf{b}$（$m > n$, 解なし）の最小二乗解は:

$$
\hat{\mathbf{x}} = A^+ \mathbf{b} = V \Sigma^+ U^\top \mathbf{b}
$$

第2回の正規方程式 $A^\top A \hat{\mathbf{x}} = A^\top \mathbf{b}$ と同じ解を与えるが、SVD版は $A^\top A$ が特異な場合でも数値的に安定。

#### Tikhonov正則化（Ridge回帰）

条件数が大きい場合、擬似逆行列は数値的に不安定。正則化パラメータ $\lambda > 0$ を加える:

$$
\hat{\mathbf{x}}_\lambda = (A^\top A + \lambda I)^{-1} A^\top \mathbf{b} = \sum_{i=1}^{r} \frac{\sigma_i}{\sigma_i^2 + \lambda} \mathbf{v}_i (\mathbf{u}_i^\top \mathbf{b})
$$

$\lambda$ が大きいほど、小さな特異値の影響が抑制される。これは**Ridge回帰**と等価。

```python
import numpy as np

# Ill-conditioned system
np.random.seed(42)
A = np.random.randn(20, 10)
A[:, -1] = A[:, 0] + 1e-8 * np.random.randn(20)  # nearly collinear
b = np.random.randn(20)

print(f"Condition number: {np.linalg.cond(A):.2e}")

# Pseudoinverse (unstable)
x_pinv = np.linalg.pinv(A) @ b
print(f"||x_pinv|| = {np.linalg.norm(x_pinv):.4f}")

# Tikhonov regularization
for lam in [0.001, 0.01, 0.1, 1.0]:
    x_ridge = np.linalg.solve(A.T @ A + lam * np.eye(10), A.T @ b)
    residual = np.linalg.norm(A @ x_ridge - b)
    print(f"λ={lam:.3f}: ||x||={np.linalg.norm(x_ridge):.4f}, "
          f"residual={residual:.4f}")
```

### 3.5 PCA の SVD による導出

第2回では固有値分解によるPCA[^5][^6]を導出した。ここではSVDによるPCAを導出し、両者の等価性を示す。

#### データ行列からの導出

データ行列 $X \in \mathbb{R}^{n \times d}$（$n$ サンプル、$d$ 次元）を中心化（各列の平均を引く）したものを $\tilde{X}$ とする。

共分散行列:

$$
C = \frac{1}{n-1} \tilde{X}^\top \tilde{X}
$$

$\tilde{X}$ の SVD を $\tilde{X} = U \Sigma V^\top$ とすると:

$$
C = \frac{1}{n-1} V \Sigma^\top U^\top U \Sigma V^\top = \frac{1}{n-1} V \Sigma^2 V^\top
$$

これは $C$ の固有値分解そのものだ。つまり:
- **PCAの主成分方向** = $\tilde{X}$ の右特異ベクトル $V$ の列
- **PCAの主成分の分散** = $\sigma_i^2 / (n-1)$

#### 分散最大化 ↔ 再構成誤差最小化の等価性

**分散最大化**: 第1主成分は $\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \text{Var}(\tilde{X}\mathbf{w})$

**再構成誤差最小化**: rank-$k$ 近似 $\hat{X} = \tilde{X} V_k V_k^\top$ が $\|\tilde{X} - \hat{X}\|_F^2$ を最小化

この2つは**等価**:

$$
\|\tilde{X} - \hat{X}\|_F^2 = \|\tilde{X}\|_F^2 - \|\tilde{X} V_k\|_F^2 = \sum_{i=1}^{r} \sigma_i^2 - \sum_{i=1}^{k} \sigma_i^2 = \sum_{i=k+1}^{r} \sigma_i^2
$$

再構成誤差を最小化するには $\sum_{i=1}^{k} \sigma_i^2$（= 射影後の分散の合計）を最大化すればよい。これはEckart-Young定理[^3]の直接的な帰結。

```python
import numpy as np

# PCA via SVD vs eigendecomposition
np.random.seed(42)
n, d = 200, 5
X = np.random.randn(n, d) @ np.diag([5, 3, 1, 0.5, 0.1])  # structured data

# Center the data
X_centered = X - X.mean(axis=0)

# Method 1: PCA via eigendecomposition of covariance
C = X_centered.T @ X_centered / (n - 1)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Method 2: PCA via SVD
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
pca_variance = s**2 / (n - 1)

print("PCA via Eigendecomposition vs SVD:")
print(f"  Eigenvalues: {np.round(eigvals, 4)}")
print(f"  s^2/(n-1):   {np.round(pca_variance, 4)}")
print(f"  Match: {np.allclose(eigvals, pca_variance)}")

# Principal components
PC_eig = X_centered @ eigvecs[:, :2]  # project onto top-2
PC_svd = U[:, :2] * s[:2]             # equivalent via SVD
print(f"\nPrincipal components match: {np.allclose(np.abs(PC_eig), np.abs(PC_svd))}")

# Explained variance ratio
total_var = np.sum(pca_variance)
for k in range(1, 6):
    ratio = np.sum(pca_variance[:k]) / total_var
    print(f"  Top-{k}: {ratio:.4f} ({ratio*100:.1f}%)")
```

### 3.6 テンソル演算と Einstein記法

#### テンソルとは

テンソルは多次元配列の数学的な一般化。機械学習では「多次元配列」と同義で使うことが多い。

| 階数 | 数学的名称 | 例 | NumPy |
|:-----|:---------|:---|:------|
| 0 | スカラー | 損失値 $L$ | `np.float64` |
| 1 | ベクトル | 埋め込み $\mathbf{e} \in \mathbb{R}^d$ | `shape=(d,)` |
| 2 | 行列 | 重み $W \in \mathbb{R}^{m \times n}$ | `shape=(m, n)` |
| 3 | 3階テンソル | バッチ入力 $X \in \mathbb{R}^{B \times T \times d}$ | `shape=(B, T, d)` |
| 4 | 4階テンソル | Multi-Head Attention $\in \mathbb{R}^{B \times H \times T \times T}$ | `shape=(B, H, T, T)` |

#### Kronecker積

行列微分をベクトル化する際に不可欠な道具として、Kronecker積を導入します。

行列 $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times q}$ の **Kronecker積**:

$$
A \otimes B = \begin{pmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{pmatrix} \in \mathbb{R}^{mp \times nq}
$$

重要な性質:
- $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$
- $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$
- $\text{vec}(AXB) = (B^\top \otimes A) \text{vec}(X)$

最後の性質は行列方程式のベクトル化に不可欠:

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Kronecker product
K = np.kron(A, B)
print(f"A ⊗ B (shape {K.shape}):")
print(K)

# vec(AXB) = (B^T ⊗ A) vec(X)
X = np.array([[1, 0], [0, 1]])
AXB = A @ X @ B
vec_AXB = AXB.flatten('F')  # column-major vectorization
kron_vec = np.kron(B.T, A) @ X.flatten('F')
print(f"\nvec(AXB) = {vec_AXB}")
print(f"(B^T ⊗ A)vec(X) = {kron_vec}")
print(f"Match: {np.allclose(vec_AXB, kron_vec)}")
```

#### Einstein記法（完全版）

Einstein記法は、テンソル演算を添字の規則だけで記述する強力な記法。NumPyの `einsum` はこの記法を直接実装している。

**規則**: 繰り返される添字は**暗黙に総和**される（縮約）。

| 演算 | 数式 | einsum | 説明 |
|:-----|:-----|:-------|:-----|
| 内積 | $c = \sum_i a_i b_i$ | `'i,i->'` | ベクトル内積 |
| 外積 | $C_{ij} = a_i b_j$ | `'i,j->ij'` | ランク1行列 |
| 行列積 | $C_{ij} = \sum_k A_{ik} B_{kj}$ | `'ik,kj->ij'` | 標準的な行列積 |
| 行列のトレース | $t = \sum_i A_{ii}$ | `'ii->'` | 対角要素の和 |
| 転置 | $B_{ji} = A_{ij}$ | `'ij->ji'` | 行列の転置 |
| 対角抽出 | $d_i = A_{ii}$ | `'ii->i'` | 対角成分 |
| バッチ行列積 | $C_{bij} = \sum_k A_{bik} B_{bkj}$ | `'bik,bkj->bij'` | バッチ処理 |
| Multi-Head Attention | $S_{bhij} = \sum_k Q_{bhik} K_{bhjk}$ | `'bhik,bhjk->bhij'` | $QK^\top$ per head |
| 二重縮約 | $s = \sum_{ij} A_{ij} B_{ij}$ | `'ij,ij->'` | Frobenius内積 |
| テンソル縮約 | $C_{ik} = \sum_j A_{ij} B_{jk}$ | `'ij,jk->ik'` | 一般縮約 |

```python
import numpy as np

# einsum examples
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
v = np.random.randn(4)

# Matrix multiplication
C1 = A @ B
C2 = np.einsum('ik,kj->ij', A, B)
print(f"Matrix mul match: {np.allclose(C1, C2)}")

# Trace
t1 = np.trace(A[:3, :3])
# need square submatrix for trace
A_sq = np.random.randn(4, 4)
t1 = np.trace(A_sq)
t2 = np.einsum('ii->', A_sq)
print(f"Trace match: {np.allclose(t1, t2)}")

# Batch matrix multiplication (Attention-style)
B_size, H, T, d = 2, 4, 8, 16
Q = np.random.randn(B_size, H, T, d)
K = np.random.randn(B_size, H, T, d)

# QK^T per head
scores1 = Q @ K.transpose(0, 1, 3, 2)  # using @ and transpose
scores2 = np.einsum('bhik,bhjk->bhij', Q, K)  # using einsum
print(f"Batch attention match: {np.allclose(scores1, scores2)}")
print(f"Scores shape: {scores1.shape}")  # (2, 4, 8, 8)
```

:::details einsum の計算グラフと最適化
`np.einsum` は添字の縮約順序を最適化できる。`optimize=True` を指定すると、中間テンソルのサイズを最小化する縮約順序を自動的に選択する。

```python
import numpy as np

# Three-tensor contraction: different orders have different costs
A = np.random.randn(100, 50)
B = np.random.randn(50, 200)
C = np.random.randn(200, 100)

# Without optimization: may choose suboptimal contraction order
result1 = np.einsum('ij,jk,kl->il', A, B, C, optimize=False)

# With optimization: chooses optimal contraction order
result2 = np.einsum('ij,jk,kl->il', A, B, C, optimize=True)
print(f"Results match: {np.allclose(result1, result2)}")

# Check optimal contraction path
path, info = np.einsum_path('ij,jk,kl->il', A, B, C, optimize='optimal')
print(f"Optimal path: {path}")
print(info)
```
:::

### 3.7 多変数微分 — 勾配・ヤコビアン・ヘシアン

#### 勾配（Gradient）

スカラー関数 $f: \mathbb{R}^n \to \mathbb{R}$ の**勾配**:

$$
\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix} \in \mathbb{R}^n
$$

勾配は $f$ が最も急に増加する方向を指す。$-\nabla f$ が最急降下方向。

#### ヤコビアン（Jacobian）

ベクトル関数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ の**ヤコビアン**:

$$
J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}, \quad J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

ヤコビアンの各行は $f_i$ の勾配 $\nabla f_i^\top$。$m = 1$ のとき、ヤコビアンは勾配の転置 $\nabla f^\top$。

**幾何学的意味**: $\mathbf{x}$ の近傍で、$\mathbf{f}(\mathbf{x} + \boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + J \boldsymbol{\delta}$（線形近似）。

**体積変化**: $\det(J)$ は変換 $\mathbf{f}$ による局所的な体積の拡大率。Normalizing Flow[^13]では:

$$
p_Y(\mathbf{y}) = p_X(\mathbf{f}^{-1}(\mathbf{y})) \cdot |\det(J_{\mathbf{f}^{-1}}(\mathbf{y}))|
$$

#### ヘシアン（Hessian）

スカラー関数 $f: \mathbb{R}^n \to \mathbb{R}$ の**ヘシアン**:

$$
H = \nabla^2 f(\mathbf{x}) \in \mathbb{R}^{n \times n}, \quad H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

ヘシアンは対称行列（$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$、Schwarzの定理）。

| ヘシアンの性質 | 意味 |
|:-------------|:-----|
| $H \succ 0$（正定値） | $\mathbf{x}$ は極小点 |
| $H \prec 0$（負定値） | $\mathbf{x}$ は極大点 |
| $H$ が不定 | $\mathbf{x}$ は鞍点（saddle point） |

```python
import numpy as np

# Example: f(x, y) = x^2 + 3*y^2 + 2*x*y
# Gradient: [2x + 2y, 6y + 2x]
# Hessian: [[2, 2], [2, 6]]

def f(xy):
    x, y = xy
    return x**2 + 3*y**2 + 2*x*y

def grad_f(xy):
    x, y = xy
    return np.array([2*x + 2*y, 6*y + 2*x])

H = np.array([[2, 2], [2, 6]])  # constant Hessian

# Check positive definiteness
eigvals = np.linalg.eigvalsh(H)
print(f"Hessian eigenvalues: {eigvals}")
print(f"Positive definite: {np.all(eigvals > 0)}")  # True → minimum exists

# Find minimum: grad = 0 → x=0, y=0
x_min = np.array([0.0, 0.0])
print(f"Minimum at: {x_min}, f = {f(x_min)}")

# Newton's method: x_new = x - H^{-1} grad(x)
x = np.array([5.0, 3.0])
for i in range(5):
    g = grad_f(x)
    x = x - np.linalg.solve(H, g)
    print(f"Step {i+1}: x = {np.round(x, 6)}, f = {f(x):.6f}")
```

### 3.8 行列微分（Matrix Calculus）

#### 基本的な微分公式

スカラー関数 $L$ の行列 $W \in \mathbb{R}^{m \times n}$ に関する微分:

$$
\frac{\partial L}{\partial W} \in \mathbb{R}^{m \times n}, \quad \left(\frac{\partial L}{\partial W}\right)_{ij} = \frac{\partial L}{\partial W_{ij}}
$$

**Matrix Cookbook[^9] 主要公式15選**:

| # | 公式 | 条件 |
|:--|:-----|:-----|
| 1 | $\frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}$ | |
| 2 | $\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^\top A \mathbf{x}) = (A + A^\top) \mathbf{x}$ | |
| 3 | $\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^\top A \mathbf{x}) = 2A\mathbf{x}$ | $A$ 対称 |
| 4 | $\frac{\partial}{\partial X} \text{tr}(AX) = A^\top$ | |
| 5 | $\frac{\partial}{\partial X} \text{tr}(X^\top A) = A$ | |
| 6 | $\frac{\partial}{\partial X} \text{tr}(AXB) = A^\top B^\top$ | |
| 7 | $\frac{\partial}{\partial X} \text{tr}(X^\top AX) = (A + A^\top)X$ | |
| 8 | $\frac{\partial}{\partial X} \|X\|_F^2 = 2X$ | |
| 9 | $\frac{\partial}{\partial X} \ln \det(X) = X^{-\top}$ | $X$ 正則 |
| 10 | $\frac{\partial}{\partial X} \det(X) = \det(X) X^{-\top}$ | $X$ 正則 |
| 11 | $\frac{\partial}{\partial \mathbf{x}} \|\mathbf{x}\|^2 = 2\mathbf{x}$ | |
| 12 | $\frac{\partial}{\partial \mathbf{x}} (A\mathbf{x} - \mathbf{b})^\top (A\mathbf{x} - \mathbf{b}) = 2A^\top(A\mathbf{x} - \mathbf{b})$ | |
| 13 | $\frac{\partial}{\partial A} \text{tr}(A^{-1}B) = -(A^{-1}BA^{-1})^\top$ | $A$ 正則 |
| 14 | $\frac{\partial}{\partial \mathbf{x}} \sigma(\mathbf{x}) = \sigma(\mathbf{x}) \odot (1 - \sigma(\mathbf{x}))$ | $\sigma$ = sigmoid |
| 15 | $\frac{\partial}{\partial \mathbf{x}} \text{softmax}(\mathbf{x})_i = s_i(\delta_{ij} - s_j)$ | $s = \text{softmax}(\mathbf{x})$ |

```python
import numpy as np

# Verify formula 3: d/dx (x^T A x) = 2Ax for symmetric A
def verify_matrix_derivative(A, x, eps=1e-7):
    n = len(x)
    # Analytical gradient
    grad_analytical = 2 * A @ x

    # Numerical gradient
    grad_numerical = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad_numerical[i] = (x_plus @ A @ x_plus - x_minus @ A @ x_minus) / (2 * eps)

    return grad_analytical, grad_numerical

A = np.array([[2, 1], [1, 3]], dtype=float)  # symmetric
x = np.array([1.0, 2.0])

grad_a, grad_n = verify_matrix_derivative(A, x)
print(f"Analytical: {grad_a}")
print(f"Numerical:  {np.round(grad_n, 6)}")
print(f"Match: {np.allclose(grad_a, grad_n)}")

# Verify formula 9: d/dX ln det(X) = X^{-T}
X = np.array([[2.0, 0.5], [0.5, 3.0]])
grad_analytical = np.linalg.inv(X).T
print(f"\nd/dX ln det(X) = X^{{-T}}:")
print(f"  Analytical:\n{np.round(grad_analytical, 4)}")

# Numerical verification
eps = 1e-7
grad_numerical = np.zeros_like(X)
for i in range(2):
    for j in range(2):
        X_plus = X.copy()
        X_plus[i, j] += eps
        X_minus = X.copy()
        X_minus[i, j] -= eps
        grad_numerical[i, j] = (np.log(np.linalg.det(X_plus)) -
                                  np.log(np.linalg.det(X_minus))) / (2 * eps)
print(f"  Numerical:\n{np.round(grad_numerical, 4)}")
print(f"  Match: {np.allclose(grad_analytical, grad_numerical)}")
```

### 3.9 連鎖律 — Backpropagationの数学的基盤

#### スカラーの連鎖律

$y = f(g(x))$ のとき:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

#### ベクトルの連鎖律

$\mathbf{y} = \mathbf{f}(\mathbf{g}(\mathbf{x}))$、$\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^p$、$\mathbf{f}: \mathbb{R}^p \to \mathbb{R}^m$ のとき:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{f}}{\partial \mathbf{g}} \cdot \frac{\partial \mathbf{g}}{\partial \mathbf{x}} = J_{\mathbf{f}} J_{\mathbf{g}} \in \mathbb{R}^{m \times n}
$$

**ヤコビアンの積**。これが連鎖律の行列版。

#### 多層ネットワークへの適用

$L$ 層のニューラルネットワーク:

$$
\mathbf{h}_0 = \mathbf{x}, \quad \mathbf{h}_l = f_l(W_l \mathbf{h}_{l-1} + \mathbf{b}_l), \quad L = \ell(\mathbf{h}_L, \mathbf{y})
$$

損失 $L$ のパラメータ $W_l$ に関する勾配:

$$
\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial \mathbf{h}_L} \cdot \frac{\partial \mathbf{h}_L}{\partial \mathbf{h}_{L-1}} \cdots \frac{\partial \mathbf{h}_{l+1}}{\partial \mathbf{h}_l} \cdot \frac{\partial \mathbf{h}_l}{\partial W_l}
$$

```mermaid
graph LR
    X["x"] --> H1["h_1 = f(W_1 x + b_1)"]
    H1 --> H2["h_2 = f(W_2 h_1 + b_2)"]
    H2 --> HL["h_L = f(W_L h_{L-1} + b_L)"]
    HL --> LOSS["L = ℓ(h_L, y)"]

    LOSS -->|"∂L/∂h_L"| HL
    HL -->|"∂h_L/∂h_{L-1}"| H2
    H2 -->|"∂h_2/∂h_1"| H1
    H1 -->|"∂h_1/∂x"| X

    style LOSS fill:#ffcdd2
    style X fill:#e3f2fd
```

**Forward pass**: $\mathbf{x} \to \mathbf{h}_1 \to \cdots \to \mathbf{h}_L \to L$（左→右）

**Backward pass**: $\frac{\partial L}{\partial \mathbf{h}_L} \to \frac{\partial L}{\partial \mathbf{h}_{L-1}} \to \cdots \to \frac{\partial L}{\partial W_l}$（右→左）

#### Backpropagation の完全導出

1層の線形変換 + 活性化: $\mathbf{h}_l = \sigma(\mathbf{z}_l)$, $\mathbf{z}_l = W_l \mathbf{h}_{l-1} + \mathbf{b}_l$

**誤差信号** $\boldsymbol{\delta}_l = \frac{\partial L}{\partial \mathbf{z}_l}$ を定義する。

出力層 ($l = L$):

$$
\boldsymbol{\delta}_L = \frac{\partial L}{\partial \mathbf{z}_L} = \frac{\partial L}{\partial \mathbf{h}_L} \odot \sigma'(\mathbf{z}_L)
$$

隠れ層 ($l < L$、逆伝播の本体）:

$$
\boldsymbol{\delta}_l = (W_{l+1}^\top \boldsymbol{\delta}_{l+1}) \odot \sigma'(\mathbf{z}_l)
$$

パラメータの勾配:

$$
\frac{\partial L}{\partial W_l} = \boldsymbol{\delta}_l \mathbf{h}_{l-1}^\top, \quad \frac{\partial L}{\partial \mathbf{b}_l} = \boldsymbol{\delta}_l
$$

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Simple 3-layer network: 3 -> 4 -> 2 -> 1
np.random.seed(42)
W1 = np.random.randn(4, 3) * 0.5
b1 = np.zeros(4)
W2 = np.random.randn(2, 4) * 0.5
b2 = np.zeros(2)
W3 = np.random.randn(1, 2) * 0.5
b3 = np.zeros(1)

# Input and target
x = np.array([1.0, 0.5, -0.5])
y = np.array([1.0])

# === Forward pass ===
z1 = W1 @ x + b1
h1 = sigmoid(z1)
z2 = W2 @ h1 + b2
h2 = sigmoid(z2)
z3 = W3 @ h2 + b3
h3 = sigmoid(z3)
loss = 0.5 * np.sum((h3 - y)**2)
print(f"Forward: loss = {loss:.6f}")

# === Backward pass (manual backpropagation) ===
# Output layer
dL_dh3 = h3 - y                       # dL/dh3
delta3 = dL_dh3 * sigmoid_deriv(z3)   # delta_3

# Hidden layer 2
delta2 = (W3.T @ delta3) * sigmoid_deriv(z2)

# Hidden layer 1
delta1 = (W2.T @ delta2) * sigmoid_deriv(z1)

# Parameter gradients
dL_dW3 = np.outer(delta3, h2)
dL_db3 = delta3
dL_dW2 = np.outer(delta2, h1)
dL_db2 = delta2
dL_dW1 = np.outer(delta1, x)
dL_db1 = delta1

print(f"\nGradients:")
print(f"  dL/dW3 shape: {dL_dW3.shape}, norm: {np.linalg.norm(dL_dW3):.6f}")
print(f"  dL/dW2 shape: {dL_dW2.shape}, norm: {np.linalg.norm(dL_dW2):.6f}")
print(f"  dL/dW1 shape: {dL_dW1.shape}, norm: {np.linalg.norm(dL_dW1):.6f}")

# === Numerical verification ===
def compute_loss(W1, b1, W2, b2, W3, b3, x, y):
    h1 = sigmoid(W1 @ x + b1)
    h2 = sigmoid(W2 @ h1 + b2)
    h3 = sigmoid(W3 @ h2 + b3)
    return 0.5 * np.sum((h3 - y)**2)

# Verify dL/dW1[0,0]
eps = 1e-7
W1_plus = W1.copy()
W1_plus[0, 0] += eps
W1_minus = W1.copy()
W1_minus[0, 0] -= eps
numerical = (compute_loss(W1_plus, b1, W2, b2, W3, b3, x, y) -
             compute_loss(W1_minus, b1, W2, b2, W3, b3, x, y)) / (2 * eps)
print(f"\nNumerical check dL/dW1[0,0]:")
print(f"  Analytical: {dL_dW1[0,0]:.8f}")
print(f"  Numerical:  {numerical:.8f}")
print(f"  Match: {np.isclose(dL_dW1[0,0], numerical, rtol=1e-4)}")
```

:::message
**これがBackpropagation[^2]の全てだ。** 「連鎖律でヤコビアンを逆順に掛けて、各層のパラメータ勾配を計算する」— この一文に全てが凝縮されている。1986年にRumelhart, Hinton, Williamsが発表したこのアルゴリズムが、深層学習の計算的基盤を築いた。
:::

### 3.10 自動微分の理論 — Forward Mode と Reverse Mode

自動微分（Automatic Differentiation, AD）[^7][^8]は、数値微分でも記号微分でもない、第3の微分法だ。

#### 3つの微分法の比較

| 方法 | 精度 | 計算量 | 長所 | 短所 |
|:-----|:-----|:------|:-----|:-----|
| 数値微分 | $O(\epsilon)$ 誤差 | $O(n)$ 回の関数評価 | 実装が簡単 | 遅い、不正確 |
| 記号微分 | 厳密 | 式膨張（expression swell） | 数学的に正確 | 式が巨大に |
| 自動微分 | 機械精度 | $O(1)$ 倍（reverse mode） | 速い、正確 | 実装が複雑 |

#### Wengert List（計算トレース）

自動微分の核心は、計算をプリミティブ操作の列（Wengert list）として記録すること。

例: $f(x_1, x_2) = x_1 x_2 + \sin(x_1)$

| Step | 演算 | 値 ($x_1=2, x_2=3$) |
|:-----|:-----|:---------------------|
| $v_1 = x_1$ | 入力 | $2$ |
| $v_2 = x_2$ | 入力 | $3$ |
| $v_3 = v_1 \cdot v_2$ | 乗算 | $6$ |
| $v_4 = \sin(v_1)$ | sin | $0.9093$ |
| $v_5 = v_3 + v_4$ | 加算 | $6.9093$ |

#### Forward Mode AD

入力に対する微分 $\dot{v}_i = \frac{\partial v_i}{\partial x_j}$ を**前向き**に伝播:

| Step | 値 | $\dot{v}_i = \partial v_i / \partial x_1$ |
|:-----|:---|:----------------------------------------|
| $v_1 = x_1$ | $2$ | $\dot{v}_1 = 1$ |
| $v_2 = x_2$ | $3$ | $\dot{v}_2 = 0$ |
| $v_3 = v_1 v_2$ | $6$ | $\dot{v}_3 = \dot{v}_1 v_2 + v_1 \dot{v}_2 = 3$ |
| $v_4 = \sin(v_1)$ | $0.909$ | $\dot{v}_4 = \cos(v_1) \dot{v}_1 = -0.416$ |
| $v_5 = v_3 + v_4$ | $6.909$ | $\dot{v}_5 = \dot{v}_3 + \dot{v}_4 = 2.584$ |

$\frac{\partial f}{\partial x_1} = 2.584$。正しい（$\frac{\partial}{\partial x_1}(x_1 x_2 + \sin x_1) = x_2 + \cos x_1 = 3 + \cos 2 = 2.584$）。

**計算量**: 1回の Forward Mode で、1つの入力変数に対する微分が得られる。$n$ 個の入力変数の勾配を求めるには $n$ 回の Forward pass が必要。

#### Reverse Mode AD（= Backpropagation）

出力に対する微分 $\bar{v}_i = \frac{\partial f}{\partial v_i}$ を**逆向き**に伝播:

| Step (逆順) | $\bar{v}_i = \partial f / \partial v_i$ |
|:-----------|:---------------------------------------|
| $\bar{v}_5 = 1$ | 出力に対する微分は1 |
| $\bar{v}_3 = \bar{v}_5 \cdot 1 = 1$ | $v_5 = v_3 + v_4$ の $v_3$ に対する偏微分 |
| $\bar{v}_4 = \bar{v}_5 \cdot 1 = 1$ | $v_5 = v_3 + v_4$ の $v_4$ に対する偏微分 |
| $\bar{v}_1 = \bar{v}_3 \cdot v_2 + \bar{v}_4 \cdot \cos(v_1) = 2.584$ | 積の規則 + sin微分 |
| $\bar{v}_2 = \bar{v}_3 \cdot v_1 = 2$ | |

$\frac{\partial f}{\partial x_1} = 2.584$, $\frac{\partial f}{\partial x_2} = 2.0$。**1回の Reverse pass で全入力変数の勾配が得られる**。

#### Forward vs Reverse: 計算量の比較

| | Forward Mode | Reverse Mode |
|:--|:------------|:------------|
| 1回のpassで得られる | 1つの入力に対する勾配 | 1つの出力に対する全入力の勾配 |
| $n$ 入力, $m$ 出力の勾配 | $n$ 回のpass | $m$ 回のpass |
| 最適な場合 | $n \ll m$（ヤコビアンが「横長」） | $m \ll n$（ヤコビアンが「縦長」） |
| 機械学習での典型 | — | $m = 1$（損失はスカラー）→ **1回のpassで全勾配** |

**だからBackpropはReverse Mode ADなのだ。** 損失関数はスカラー値（$m = 1$）、パラメータは数十億（$n \sim 10^9$）。Reverse modeなら1回のbackward passで全パラメータの勾配が得られる。Forward modeでは $10^9$ 回のforward passが必要。

```python
import numpy as np

# Implementing Forward Mode AD with dual numbers
class Dual:
    """Dual number: a + bε where ε^2 = 0"""
    def __init__(self, val, deriv=0.0):
        self.val = val      # primal value
        self.deriv = deriv   # tangent (derivative)

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val + other.val, self.deriv + other.deriv)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val * other.val,
                    self.val * other.deriv + self.deriv * other.val)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Dual({self.val:.4f}, {self.deriv:.4f})"

def sin_dual(x):
    return Dual(np.sin(x.val), np.cos(x.val) * x.deriv)

# f(x1, x2) = x1*x2 + sin(x1)
def f_dual(x1, x2):
    return x1 * x2 + sin_dual(x1)

# df/dx1 at (2, 3): set x1.deriv = 1
x1 = Dual(2.0, 1.0)  # seed: dx1/dx1 = 1
x2 = Dual(3.0, 0.0)  # seed: dx2/dx1 = 0
result = f_dual(x1, x2)
print(f"f(2, 3) = {result.val:.4f}")
print(f"df/dx1  = {result.deriv:.4f}")
print(f"Expected: {3 + np.cos(2):.4f}")

# df/dx2 at (2, 3): set x2.deriv = 1
x1 = Dual(2.0, 0.0)
x2 = Dual(3.0, 1.0)
result = f_dual(x1, x2)
print(f"df/dx2  = {result.deriv:.4f}")
print(f"Expected: {2.0:.4f}")
```

### 3.11 テイラー展開と二次近似

多変数のテイラー展開は、最適化理論の基盤:

$$
f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{\delta} + \frac{1}{2} \boldsymbol{\delta}^\top H(\mathbf{x}) \boldsymbol{\delta}
$$

**Newton法**: 二次近似を最小化する $\boldsymbol{\delta}$ を求める:

$$
\nabla f + H \boldsymbol{\delta} = 0 \implies \boldsymbol{\delta}^* = -H^{-1} \nabla f
$$

```python
import numpy as np

# Rosenbrock function (classic optimization test)
def rosenbrock(xy):
    x, y = xy
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(xy):
    x, y = xy
    dx = -2*(1-x) + 100*2*(y-x**2)*(-2*x)
    dy = 100*2*(y-x**2)
    return np.array([dx, dy])

def rosenbrock_hessian(xy):
    x, y = xy
    dxx = 2 - 400*(y - x**2) + 800*x**2
    dxy = -400*x
    dyy = 200.0
    return np.array([[dxx, dxy], [dxy, dyy]])

# Newton's method
x = np.array([-1.0, 1.0])
print(f"Newton's method on Rosenbrock:")
for i in range(10):
    g = rosenbrock_grad(x)
    H = rosenbrock_hessian(x)
    delta = -np.linalg.solve(H, g)
    x = x + delta
    f_val = rosenbrock(x)
    print(f"  Step {i+1}: x={np.round(x, 6)}, f={f_val:.8f}")
    if f_val < 1e-14:
        print(f"  Converged in {i+1} steps!")
        break
```

### 3.12 Softmaxの微分 — Attention学習の鍵

Softmaxの微分はTransformerの学習で最も頻繁に現れる計算の一つ。

#### Softmaxの定義と性質

$$
s_i = \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

性質:
- $s_i > 0$ かつ $\sum_i s_i = 1$（確率分布）
- $\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$

#### ヤコビアンの導出

$i = j$ のとき:

$$
\frac{\partial s_i}{\partial z_i} = s_i(1 - s_i)
$$

$i \neq j$ のとき:

$$
\frac{\partial s_i}{\partial z_j} = -s_i s_j
$$

まとめると:

$$
\frac{\partial \mathbf{s}}{\partial \mathbf{z}} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top
$$

```python
import numpy as np

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

def softmax_jacobian(z):
    """Analytical Jacobian of softmax"""
    s = softmax(z)
    return np.diag(s) - np.outer(s, s)

# Verify with numerical differentiation
z = np.array([2.0, 1.0, 0.1])
J_analytical = softmax_jacobian(z)

eps = 1e-7
n = len(z)
J_numerical = np.zeros((n, n))
for j in range(n):
    z_plus = z.copy(); z_plus[j] += eps
    z_minus = z.copy(); z_minus[j] -= eps
    J_numerical[:, j] = (softmax(z_plus) - softmax(z_minus)) / (2 * eps)

print("Softmax Jacobian (analytical):")
print(np.round(J_analytical, 6))
print(f"\nMatch numerical: {np.allclose(J_analytical, J_numerical)}")

# Key property: each row sums to 0
print(f"Row sums: {np.round(J_analytical.sum(axis=1), 10)}")
```

#### Cross-Entropy損失のSoftmax微分

Cross-Entropy損失 $L = -\sum_i y_i \log s_i$ のSoftmax入力 $\mathbf{z}$ に関する勾配:

$$
\frac{\partial L}{\partial \mathbf{z}} = \mathbf{s} - \mathbf{y}
$$

この結果は驚くほどシンプル。導出:

$$
\frac{\partial L}{\partial z_j} = -\sum_i y_i \frac{1}{s_i} \frac{\partial s_i}{\partial z_j} = -\sum_i y_i \frac{s_i(\delta_{ij} - s_j)}{s_i} = -y_j + s_j \sum_i y_i = s_j - y_j
$$

（$\sum_i y_i = 1$ を使った）

```python
import numpy as np

z = np.array([2.0, 1.0, 0.1])
s = softmax(z)
y = np.array([1.0, 0.0, 0.0])  # one-hot target

# Analytical gradient: s - y
grad_analytical = s - y

# Numerical gradient
def cross_entropy_loss(z, y):
    s = softmax(z)
    return -np.sum(y * np.log(s + 1e-12))

eps = 1e-7
grad_numerical = np.zeros(len(z))
for j in range(len(z)):
    z_plus = z.copy(); z_plus[j] += eps
    z_minus = z.copy(); z_minus[j] -= eps
    grad_numerical[j] = (cross_entropy_loss(z_plus, y) - cross_entropy_loss(z_minus, y)) / (2 * eps)

print(f"Analytical: {np.round(grad_analytical, 6)}")
print(f"Numerical:  {np.round(grad_numerical, 6)}")
print(f"Match: {np.allclose(grad_analytical, grad_numerical)}")
```

:::message
**LLMへの接続**: GPT系モデルの学習では、各トークン位置で Softmax + Cross-Entropy の勾配 $\mathbf{s} - \mathbf{y}$ を計算する。語彙サイズが50,000以上のとき、この計算が学習のボトルネックの一つになる。
:::

### 3.13 変分法入門 — 変分推論への予告

変分法は「関数の関数」（汎関数）を最適化する。VAE（第15回）で使う変分推論の数学的基盤。

#### 汎関数と変分

**汎関数**: 関数を入力として受け取り、スカラーを返す写像。

$$
F[f] = \int_a^b L(x, f(x), f'(x)) \, dx
$$

例: 曲線の長さ $F[f] = \int_a^b \sqrt{1 + f'(x)^2} \, dx$

#### Euler-Lagrange方程式

$F[f]$ を最小化する $f$ は以下を満たす:

$$
\frac{\partial L}{\partial f} - \frac{d}{dx} \frac{\partial L}{\partial f'} = 0
$$

#### 変分推論との接続（予告）

VAEでは、真の事後分布 $p(\mathbf{z} \mid \mathbf{x})$ を近似する分布 $q(\mathbf{z} \mid \mathbf{x})$ を見つけたい。これは「KLダイバージェンスという汎関数を、分布の空間上で最小化する」問題:

$$
q^* = \arg\min_q \text{KL}(q(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z} \mid \mathbf{x}))
$$

この最適化問題を解くのが変分推論。その理論的基盤が変分法だ。詳細は第15回（VAE）で扱う。

### 3.14 Boss Battle: Transformer 1層の完全微分

Transformer[^1]の1層における Forward + Backward を行列微分で完全に記述する。

#### Forward Pass

入力 $H \in \mathbb{R}^{T \times d}$（$T$ トークン、$d$ 次元）に対して:

$$
Q = HW_Q, \quad K = HW_K, \quad V = HW_V
$$
$$
S = \frac{QK^\top}{\sqrt{d_k}}, \quad A = \text{softmax}(S), \quad O = AV
$$
$$
\text{output} = OW_O + H \quad \text{(residual connection)}
$$

#### Backward Pass（$\frac{\partial L}{\partial W_Q}$ の導出）

$L$ をスカラー損失とし、$\frac{\partial L}{\partial O}$ が既知とする。

$$
\frac{\partial L}{\partial W_Q} = H^\top \frac{\partial L}{\partial Q}
$$

ここで $\frac{\partial L}{\partial Q}$ は連鎖律で:

$$
\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial S} \cdot \frac{\partial S}{\partial Q}
$$

$S = QK^\top / \sqrt{d_k}$ より $\frac{\partial S}{\partial Q} = K / \sqrt{d_k}$、つまり:

$$
\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} K
$$

Softmax の微分は:

$$
\frac{\partial L}{\partial S_{ij}} = \sum_k \frac{\partial L}{\partial A_{ik}} A_{ik} (\delta_{jk} - A_{ij})
$$

```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

# Transformer single layer forward + backward
np.random.seed(42)
T, d, dk = 4, 8, 8  # 4 tokens, 8 dims
H = np.random.randn(T, d)
W_Q = np.random.randn(d, dk) * 0.1
W_K = np.random.randn(d, dk) * 0.1
W_V = np.random.randn(d, dk) * 0.1
W_O = np.random.randn(dk, d) * 0.1

# Forward
Q = H @ W_Q
K = H @ W_K
V = H @ W_V
S = Q @ K.T / np.sqrt(dk)
A = softmax(S)
O = A @ V
output = O @ W_O + H  # residual

# Backward (assume dL/doutput = random for demo)
dL_doutput = np.random.randn(T, d)

# dL/dO
dL_dO = dL_doutput @ W_O.T

# dL/dA (from O = AV)
dL_dA = dL_dO @ V.T

# dL/dS (softmax backward)
dL_dS = np.zeros_like(S)
for i in range(T):
    a = A[i, :]  # (T,)
    dL_da = dL_dA[i, :]  # (T,)
    # Jacobian of softmax: diag(a) - a a^T
    J_softmax = np.diag(a) - np.outer(a, a)
    dL_dS[i, :] = J_softmax @ dL_da

# dL/dQ, dL/dK
dL_dQ = dL_dS @ K / np.sqrt(dk)
dL_dK = dL_dS.T @ Q / np.sqrt(dk)

# dL/dW_Q, dL/dW_K, dL/dW_V
dL_dW_Q = H.T @ dL_dQ
dL_dW_K = H.T @ dL_dK
dL_dW_V = H.T @ (A.T @ dL_dO)
dL_dW_O = O.T @ dL_doutput

print("Gradient norms:")
print(f"  dL/dW_Q: {np.linalg.norm(dL_dW_Q):.6f}")
print(f"  dL/dW_K: {np.linalg.norm(dL_dW_K):.6f}")
print(f"  dL/dW_V: {np.linalg.norm(dL_dW_V):.6f}")
print(f"  dL/dW_O: {np.linalg.norm(dL_dW_O):.6f}")

# Numerical verification for dL/dW_Q[0,0]
eps = 1e-5
def forward_loss(W_Q_):
    Q_ = H @ W_Q_
    S_ = Q_ @ K.T / np.sqrt(dk)
    A_ = softmax(S_)
    O_ = A_ @ V
    out_ = O_ @ W_O + H
    return np.sum(out_ * dL_doutput)  # proxy loss

W_Q_plus = W_Q.copy(); W_Q_plus[0, 0] += eps
W_Q_minus = W_Q.copy(); W_Q_minus[0, 0] -= eps
numerical = (forward_loss(W_Q_plus) - forward_loss(W_Q_minus)) / (2 * eps)
print(f"\nNumerical check dL/dW_Q[0,0]:")
print(f"  Analytical: {dL_dW_Q[0,0]:.8f}")
print(f"  Numerical:  {numerical:.8f}")
print(f"  Match: {np.isclose(dL_dW_Q[0,0], numerical, rtol=1e-3)}")
```

:::message
**進捗: 70% 完了** SVDの理論（存在定理・Eckart-Young・擬似逆行列・PCA）、テンソル演算・Einstein記法、行列微分、連鎖律、Backpropagation、自動微分、Transformer 1層の完全微分を導出した。
:::

---

## 参考文献

### 主要論文

[^1]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*.
@[card](https://arxiv.org/abs/1706.03762)

[^2]: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533-536.
@[card](https://doi.org/10.1038/323533a0)

[^3]: Eckart, C. & Young, G. (1936). The Approximation of One Matrix by Another of Lower Rank. *Psychometrika*, 1, 211-218.
@[card](https://doi.org/10.1007/BF02288367)

[^5]: Pearson, K. (1901). On Lines and Planes of Closest Fit to Systems of Points in Space. *Philosophical Magazine*, 2(11), 559-572.
@[card](https://doi.org/10.1080/14786440109462720)

[^6]: Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417-441.
@[card](https://doi.org/10.1037/h0071325)

[^7]: Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic Differentiation in Machine Learning: a Survey. *JMLR*, 18(153), 1-43.
@[card](https://arxiv.org/abs/1502.05767)

[^10]: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
@[card](https://arxiv.org/abs/2106.09685)

[^12]: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.
@[card](https://arxiv.org/abs/2205.14135)

[^13]: Rezende, D. J. & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML 2015*.
@[card](https://arxiv.org/abs/1505.05770)

### 教科書

[^8]: Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM.

[^9]: Petersen, K. B. & Pedersen, M. S. (2012). *The Matrix Cookbook*. Technical Report, DTU.

---

## 記法規約

| 記号 | 意味 | 初出 |
|:-----|:-----|:-----|
| $A, B, W$ | 行列（大文字） | 3.1 |
| $\mathbf{x}, \mathbf{v}$ | ベクトル（太字小文字） | 3.7 |
| $\sigma_i$ | 特異値 | 3.1 |
| $\mathbf{u}_i, \mathbf{v}_i$ | 左/右特異ベクトル | 3.1 |
| $U, \Sigma, V$ | SVDの構成行列 | 3.1 |
| $A^+$ | Moore-Penrose擬似逆行列 | 3.4 |
| $A_k$ | rank-$k$ 截断SVD | 3.2 |
| $\nabla f$ | 勾配 | 3.7 |
| $J$ | ヤコビアン | 3.7 |
| $H$ | ヘシアン | 3.7 |
| $\boldsymbol{\delta}_l$ | 第$l$層の誤差信号 | 3.9 |
| $\otimes$ | Kronecker積 | 3.6 |
| $\odot$ | Hadamard積（要素ごとの積） | 3.9 |
| $\text{vec}(A)$ | 行列のベクトル化 | 3.6 |
| $\kappa(A)$ | 条件数 | 4.4 |

---
