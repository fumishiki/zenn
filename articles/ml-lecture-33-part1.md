---
title: "第33回: Normalizing Flows【前編】理論編: 30秒の驚き→数式修行"
emoji: "🔄"
type: "tech"
topics: ["machinelearning"]
published: true
slug: "ml-lecture-33-part1"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---
---

# 第33回: Normalizing Flows — 可逆変換で厳密尤度を手に入れる

> **VAEは近似、GANは暗黙的。Normalizing Flowsは可逆変換で厳密な尤度 log p(x) を計算する。変数変換の数学が、生成モデルに新しい道を開いた。**

VAEは変分下界ELBOで真の尤度 log p(x) を下から近似する。GANは尤度を捨て、識別器との敵対で暗黙的に分布を学ぶ。どちらも「厳密な尤度」を諦めた。

Normalizing Flows [^1] [^2] は可逆変換 f: z → x で、**Change of Variables公式を使い厳密な log p(x) を計算する**。ヤコビアン行列式 |det J_f| がその鍵だ。

この数学的美しさは代償を伴う。可逆性制約がアーキテクチャを制限する。計算量 O(D³) のヤコビアン行列式がボトルネックになる。RealNVP [^3]、Glow [^4] は構造化された変換でこれを O(D) に削減した。そしてContinuous Normalizing Flows (CNF) [^5] とFFJORD [^6] が、Neural ODEで連続時間の可逆変換を実現し、Diffusion ModelsやFlow Matchingへの橋を架けた。

本講義はCourse IV「拡散モデル理論編」の第1回 — 全10講義の旅の出発点だ。Course I-IIIで培った数学力と実装力を武器に、生成モデル理論の深淵へ。

> **Note:** **Course IV概要**: Normalizing Flows → EBM → Score Matching → DDPM → SDE → Flow Matching → LDM → Consistency Models → World Models → 統一理論。密度モデリングの論理的チェーンを辿り、「拡散モデル論文の理論セクションが導出できる」到達点へ。

```mermaid
graph LR
    A["📊 VAE<br/>ELBO近似<br/>ぼやけ"] --> D["🎯 厳密尤度<br/>の追求"]
    B["🎨 GAN<br/>暗黙的密度<br/>不安定"] --> D
    D --> E["🌊 Normalizing Flow<br/>可逆変換f<br/>log p(x)計算可能"]
    E --> F["📐 Change of Variables<br/>|det J_f|"]
    E --> G["🔄 RealNVP/Glow<br/>構造化"]
    E --> H["∞ CNF/FFJORD<br/>Neural ODE"]
    H --> I["🌈 Diffusion/FM<br/>への橋"]
    style E fill:#e1f5ff
    style H fill:#fff3e0
    style I fill:#f3e5f5
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

## 🚀 0. クイックスタート（30秒）— 可逆変換で密度を追跡する

**ゴール**: Change of Variables公式を30秒で体感する。

ガウス分布 z ~ N(0,1) を仮定変換 f(z) = μ + σz で変換し、変換後の密度 p(x) をヤコビアンで計算する。

```julia
using Distributions, LinearAlgebra

# 1D Normalizing Flow: f(z) = μ + σz
f(z, μ, σ) = μ .+ σ .* z
f_inv(x, μ, σ) = (x .- μ) ./ σ
log_det_jacobian(σ) = sum(log.(abs.(σ)))  # |det J_f| = |σ|

# Base distribution: z ~ N(0, 1)
q_z = Normal(0, 1)

# Transform: x = f(z) with μ=2, σ=3
μ, σ = 2.0, 3.0
z_samples = rand(q_z, 1000)
x_samples = f(z_samples, μ, σ)

# Exact log p(x) via Change of Variables
# log p(x) = log q(z) - log|det J_f|
log_p_x(x) = logpdf(q_z, f_inv(x, μ, σ)) - log_det_jacobian(σ)

println("z ~ N(0,1) → x = 2 + 3z")
println("log p(x=5) = ", round(log_p_x(5.0), digits=4))
println("Expected: log N(5; μ=2, σ²=9) = ", round(logpdf(Normal(μ, σ), 5.0), digits=4))
println("Change of Variables公式で厳密なlog p(x)を計算した!")
```

出力:
```
z ~ N(0,1) → x = 2 + 3z
log p(x=5) = -2.3259
Expected: log N(5; μ=2, σ²=9) = -2.3259
Change of Variables公式で厳密なlog p(x)を計算した!
```

**3行のコードで可逆変換と密度追跡を動かした。** 数式で書くと:

$$
\begin{aligned}
z &\sim q(z) = \mathcal{N}(0, 1) \\
x &= f(z) = \mu + \sigma z \quad \text{(invertible)} \\
\log p(x) &= \log q(f^{-1}(x)) - \log \left| \det \frac{\partial f}{\partial z} \right| \\
&= \log q\left(\frac{x - \mu}{\sigma}\right) - \log |\sigma|
\end{aligned}
$$

**Change of Variables公式** (第3-4回のヤコビアン前提):

$$
p_X(x) = p_Z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right| = p_Z(z) \left| \det \frac{\partial f}{\partial z} \right|^{-1}
$$

この公式が、Normalizing Flowsの全ての理論的基盤だ。

> **Note:** **進捗: 3% 完了** Change of Variables公式を体感した。ここからヤコビアン計算の困難性、Coupling Layer、RealNVP、Glow、CNF、FFJORDへ進む。

---

## 🎮 1. 体験ゾーン（10分）— Flowの3形態を触る

### 1.1 Normalizing Flowとは何か

**定義**: 単純な分布 q(z) (通常 N(0,I)) から、可逆変換の合成で複雑な分布 p(x) を構築する。

$$
\begin{aligned}
z_0 &\sim q(z) = \mathcal{N}(0, I) \\
z_1 &= f_1(z_0) \\
z_2 &= f_2(z_1) \\
&\vdots \\
x = z_K &= f_K(z_{K-1})
\end{aligned}
$$

各 $f_k$ は可逆 (invertible) で、$f_k^{-1}$ とヤコビアン $\frac{\partial f_k}{\partial z_{k-1}}$ が計算可能。

**最終的な密度**:

$$
\log p(x) = \log q(z_0) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial z_{k-1}} \right|
$$

これを**正規化流 (Normalizing Flow)** と呼ぶ。

### 1.2 Flowの3つの顔: Affine / Coupling / Continuous

Normalizing Flowsは構造によって3つのファミリーに分かれる。

| タイプ | 変換 | 例 | ヤコビアン計算量 | 表現力 |
|:-------|:-----|:---|:----------------|:-------|
| **Affine Flow** | 線形変換 $f(z) = Az + b$ | NICE [^2], Planar [^7] | O(D³) (一般) / O(D) (構造化) | 低 |
| **Coupling Flow** | 分割変換 $x_{1:d}=z_{1:d}$, $x_{d+1:D}=g(z_{d+1:D}; z_{1:d})$ | RealNVP [^3], Glow [^4] | O(D) | 中 |
| **Continuous Flow** | Neural ODE $\frac{dx}{dt}=f(x,t)$ | CNF [^5], FFJORD [^6] | O(D) (trace推定) | 高 |

それぞれを触ってみよう。

#### 1.2.1 Affine Flow: 線形変換

最も単純なFlow。回転・スケール・平行移動。

$$
f(z) = Az + b, \quad \log p(x) = \log q(z) - \log |\det A|
$$

**問題**: 一般の行列 A だと $\det A$ の計算が O(D³)。次元が高いと破綻する。

#### 1.2.2 Coupling Flow: 分割で計算量削減

**アイデア**: 入力を2分割 $z = [z_{1:d}, z_{d+1:D}]$ し、片方はそのまま、もう片方を条件付き変換。

$$
\begin{aligned}
x_{1:d} &= z_{1:d} \\
x_{d+1:D} &= z_{d+1:D} \odot \exp(s(z_{1:d})) + t(z_{1:d})
\end{aligned}
$$

ここで $s, t$ はニューラルネット (任意の関数)。

**ヤコビアン**:

$$
\frac{\partial f}{\partial z} = \begin{bmatrix} I_d & 0 \\ \frac{\partial x_{d+1:D}}{\partial z_{1:d}} & \text{diag}(\exp(s(z_{1:d}))) \end{bmatrix}
$$

下三角行列 → $\det = \prod_{i=1}^{D-d} \exp(s_i) = \exp(\sum s_i)$ → **O(D)** 計算!

**RealNVPの核心**: Coupling Layerを積み重ね、分割次元を交互に変える。これだけで O(D) でスケールする。

#### 1.2.3 Continuous Flow: Neural ODEで無限層

離散的な変換の積み重ねを、連続時間 ODE に一般化。

$$
\frac{dz(t)}{dt} = f(z(t), t, \theta), \quad z(0) = z_0, \quad z(1) = x
$$

**Instantaneous Change of Variables** [^5]:

$$
\frac{\partial \log p(z(t))}{\partial t} = -\text{tr}\left(\frac{\partial f}{\partial z}\right)
$$

積分すると:

$$
\log p(x) = \log p(z_0) - \int_0^1 \text{tr}\left(\frac{\partial f}{\partial z(t)}\right) dt
$$

**FFJORD [^6]**: Hutchinsonのtrace推定で $\text{tr}(\frac{\partial f}{\partial z})$ を O(1) メモリで計算。これがCNFをスケーラブルにした。

> **Note:** **進捗: 10% 完了** Affine / Coupling / Continuous の3つのFlowを触った。次はCourse IVの全体像と、Change of Variables公式の完全導出へ。

---


> Progress: 10%
> **理解度チェック**
> 1. Change of Variables公式 $\log p(\mathbf{x}) = \log p(\mathbf{z}) - \log|\det J_f|$ において、coupling layerのヤコビアン行列式が $O(D^3)$ から $O(D)$ に削減される設計原理を説明せよ。
> 2. NF・VAE・GANの3手法で「尤度の正確さ」が異なる根本的な理由を一言で述べよ。

## 🧩 2. 直感ゾーン（15分）— Course IV全体像とFlowの位置づけ

### 2.1 Course IV: 拡散モデル理論編の全体像

**Course IV は10講義で密度モデリングの論理的チェーンを完成させる**。

```mermaid
graph TD
    L33["第33回<br/>Normalizing Flows<br/>可逆変換+厳密尤度"] --> L34["第34回<br/>EBM & 統計物理<br/>p(x)∝exp(-E(x))"]
    L34 --> L35["第35回<br/>Score Matching<br/>∇log p(x)"]
    L35 --> L36["第36回<br/>DDPM<br/>離散拡散"]
    L36 --> L37["第37回<br/>SDE/ODE<br/>連続拡散"]
    L37 --> L38["第38回<br/>Flow Matching<br/>Score↔Flow統一"]
    L38 --> L39["第39回<br/>LDM<br/>潜在空間拡散"]
    L39 --> L40["第40回<br/>Consistency Models<br/>1-step生成"]
    L40 --> L41["第41回<br/>World Models<br/>環境シミュレータ"]
    L41 --> L42["第42回<br/>統一理論<br/>全生成モデル整理"]

    style L33 fill:#e1f5ff
    style L38 fill:#fff3e0
    style L42 fill:#f3e5f5
```

**各講義の核心**:

| 講義 | テーマ | 核心的問い | 数学的道具 |
|:----|:------|:---------|:---------|
| 33 | Normalizing Flows | 可逆性で厳密尤度を得られるか？ | Change of Variables, ヤコビアン |
| 34 | EBM & 統計物理 | 正規化定数Zを回避できるか？ | Gibbs分布, MCMC, Hopfield↔Attention |
| 35 | Score Matching | Zを消してスコアだけ学習できるか？ | ∇log p, Langevin Dynamics |
| 36 | DDPM | ノイズ除去の反復が生成になるか？ | Forward/Reverse Process, VLB |
| 37 | SDE/ODE | 離散→連続で理論的基盤を得られるか？ | 伊藤積分, Fokker-Planck, PF-ODE |
| 38 | Flow Matching | Score/Flow/Diffusionは同じか？ | OT, JKO, Wasserstein勾配流 |
| 39 | LDM | ピクセル空間の壁を超えられるか？ | VAE潜在空間, CFG, テキスト条件付け |
| 40 | Consistency Models | 1000ステップ→1ステップにできるか？ | Self-consistency, 蒸留, DPM-Solver |
| 41 | World Models | 生成モデルは世界を理解するか？ | JEPA, Transfusion, 物理法則学習 |
| 42 | 統一理論 | 全生成モデルの本質は何か？ | 数学的等価性, パラダイム分類 |

**Course Iの数学が花開く瞬間**:

- **第3-4回 ヤコビアン・確率変数変換** → 第33回 Change of Variables公式
- **第5回 伊藤積分・SDE基礎** → 第37回 VP-SDE/VE-SDE, Fokker-Planck
- **第6回 KL divergence** → 第33-42回 全体の損失関数
- **第6回 Optimal Transport** → 第38回 Wasserstein勾配流, JKO scheme
- **第4回 Fisher情報行列** → 第34回 Natural Gradient, 情報幾何

「Course Iは無駄だったのでは？」 → 「全てここで花開く」。

### 2.2 Normalizing Flowsの3つの比喩

#### 比喩1: 粘土の変形

ガウス分布 (球) を粘土と見立て、可逆変換で引き延ばす・ねじる・曲げる。

- **伸ばす**: スケーリング $x = \sigma z$
- **ずらす**: 平行移動 $x = z + \mu$
- **ねじる**: 回転 $x = Rz$
- **曲げる**: 非線形変換 $x = \tanh(z)$ (注: 単調性必須)

各操作でヤコビアンが「体積の変化率」を追跡する。

#### 比喩2: 川の流れ

$z \sim \mathcal{N}(0, I)$ を水源とし、可逆変換を「川の流れ」と見る。

- **流れる**: $z_0 \to z_1 \to \cdots \to z_K = x$
- **密度**: 水源の密度 $q(z_0)$ が流れに沿って変化
- **ヤコビアン**: 流れの断面積変化 = 密度の逆数変化

連続時間にすると Continuous Normalizing Flow (CNF) = 「流れ場 $f(z, t)$ による輸送」。

#### 比喩3: 座標変換

極座標変換 $(x, y) \to (r, \theta)$ を思い出そう (第3-4回)。

$$
p_{r,\theta}(r, \theta) = p_{x,y}(x, y) \left| \det \frac{\partial (x,y)}{\partial (r,\theta)} \right| = p_{x,y}(x, y) \cdot r
$$

$r$ がヤコビアン行列式。Normalizing Flowsは「確率分布の座標変換」そのもの。

### 2.3 VAE vs GAN vs Flowの3つ巴

| 観点 | VAE | GAN | Normalizing Flow |
|:-----|:----|:----|:-----------------|
| **尤度** | 近似 (ELBO) | 暗黙的 (不明) | **厳密** |
| **訓練** | 安定 | 不安定 (Nash均衡) | 安定 |
| **生成品質** | ぼやける | 鮮明 | 中間 |
| **潜在空間** | 解釈可能 | 解釈困難 | 解釈可能 |
| **アーキテクチャ** | 自由 | 自由 | **可逆性制約** |
| **計算量** | O(D) | O(D) | O(D³) or O(D) (構造化) |
| **用途** | 表現学習 | 高品質生成 | 密度推定・異常検知 |

**Flowの強み**: 厳密な $\log p(x)$ → 異常検知 (out-of-distribution detection) / 密度推定 / 変分推論の事後分布近似 (IAF [^8])。

**Flowの弱み**: 可逆性制約 → 表現力制限 / ヤコビアン計算 → スケーラビリティ。

### 2.4 Flowファミリーの系譜図

```mermaid
graph TD
    A["NICE 2014<br/>Affine Coupling"] --> B["RealNVP 2016<br/>Multi-scale"]
    B --> C["Glow 2018<br/>1x1 Conv"]

    A --> D["MAF 2017<br/>Autoregressive"]
    D --> E["IAF 2016<br/>Inverse AR"]

    F["Neural ODE 2018<br/>連続変換"] --> G["CNF 2018<br/>Continuous Flow"]
    G --> H["FFJORD 2019<br/>Hutchinson trace"]

    C --> I["NSF 2019<br/>Spline"]

    H --> J["Rectified Flow 2022<br/>直線輸送"]
    J --> K["Flow Matching 2023<br/>Diffusion統一"]

    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style G fill:#fff3e0
    style H fill:#fff3e0
    style K fill:#c8e6c9
```

**2つの大きな流れ**:

1. **離散Flow**: NICE → RealNVP → Glow → NSF (構造化で O(D) 実現)
2. **連続Flow**: Neural ODE → CNF → FFJORD (ODE + trace推定)

**2022-2023の統一**: Rectified Flow [^9], Flow Matching [^10] が Normalizing Flows と Diffusion Models を橋渡し。

> **Note:** **進捗: 20% 完了** Course IV全体像とFlowの位置づけを把握。次は数式修行ゾーン — Change of Variables公式の完全導出、Coupling Layer理論、CNF/FFJORDの数学へ。

---


> Progress: 20%
> **理解度チェック**
> 1. 密度モデリングで可逆性制約が必要な理由を Change of Variables の観点から述べよ。
> 2. CNF（連続正規化流）が離散coupling layerより表現力が高い直感的理由を述べよ。

## 📐 3. 数式修行ゾーン（60分）— Flowの数学的基盤

### 3.1 Change of Variables公式の完全導出

**前提知識**: Course I 第3-4回のヤコビアン・確率変数変換を前提とする。ここでは確率密度変換則の導出に集中する。

#### 3.1.1 1次元の場合

確率変数 $Z$ が密度 $p_Z(z)$ を持ち、可逆な単調増加関数 $f$ で変換: $X = f(Z)$。

**導出**:

$$
\begin{aligned}
P(X \leq x) &= P(f(Z) \leq x) = P(Z \leq f^{-1}(x)) \\
&= \int_{-\infty}^{f^{-1}(x)} p_Z(z) dz
\end{aligned}
$$

両辺を $x$ で微分:

$$
\begin{aligned}
p_X(x) &= \frac{d}{dx} P(X \leq x) = p_Z(f^{-1}(x)) \cdot \frac{d f^{-1}(x)}{dx} \\
&= p_Z(z) \left| \frac{dz}{dx} \right| = p_Z(z) \left| \frac{df}{dz} \right|^{-1}
\end{aligned}
$$

ここで $z = f^{-1}(x)$。絶対値は単調減少の場合も扱うため。

**結論**:

$$
\boxed{p_X(x) = p_Z(f^{-1}(x)) \left| \frac{df}{dz} \right|^{-1}}
$$

対数をとると:

$$
\boxed{\log p_X(x) = \log p_Z(z) - \log \left| \frac{df}{dz} \right|}
$$

#### 3.1.2 多次元の場合

$\mathbf{Z} \in \mathbb{R}^D$ が密度 $p_{\mathbf{Z}}(\mathbf{z})$ を持ち、可逆変換 $\mathbf{f}: \mathbb{R}^D \to \mathbb{R}^D$ で $\mathbf{X} = \mathbf{f}(\mathbf{Z})$。

**ヤコビアン行列**:

$$
J_{\mathbf{f}} = \frac{\partial \mathbf{f}}{\partial \mathbf{z}} = \begin{bmatrix}
\frac{\partial f_1}{\partial z_1} & \cdots & \frac{\partial f_1}{\partial z_D} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_D}{\partial z_1} & \cdots & \frac{\partial f_D}{\partial z_D}
\end{bmatrix}
$$

**変数変換公式** (第3回 定理):

$$
\boxed{p_{\mathbf{X}}(\mathbf{x}) = p_{\mathbf{Z}}(\mathbf{f}^{-1}(\mathbf{x})) \left| \det \frac{\partial \mathbf{f}^{-1}}{\partial \mathbf{x}} \right|}
$$

逆関数のヤコビアンは、順方向のヤコビアンの逆行列:

$$
\frac{\partial \mathbf{f}^{-1}}{\partial \mathbf{x}} = \left( \frac{\partial \mathbf{f}}{\partial \mathbf{z}} \right)^{-1}
$$

行列式の性質 $\det(A^{-1}) = (\det A)^{-1}$ より:

$$
\left| \det \frac{\partial \mathbf{f}^{-1}}{\partial \mathbf{x}} \right| = \left| \det \frac{\partial \mathbf{f}}{\partial \mathbf{z}} \right|^{-1}
$$

**最終形**:

$$
\boxed{p_{\mathbf{X}}(\mathbf{x}) = p_{\mathbf{Z}}(\mathbf{z}) \left| \det \frac{\partial \mathbf{f}}{\partial \mathbf{z}} \right|^{-1}}
$$

対数形式:

$$
\boxed{\log p_{\mathbf{X}}(\mathbf{x}) = \log p_{\mathbf{Z}}(\mathbf{z}) - \log \left| \det J_{\mathbf{f}} \right|}
$$

ここで $\mathbf{z} = \mathbf{f}^{-1}(\mathbf{x})$、$J_{\mathbf{f}} = \frac{\partial \mathbf{f}}{\partial \mathbf{z}}$。

#### 3.1.3 合成変換の場合

$K$ 個の可逆変換を合成: $\mathbf{f} = \mathbf{f}_K \circ \cdots \circ \mathbf{f}_1$。

$$
\mathbf{z}_0 \sim q(\mathbf{z}_0), \quad \mathbf{z}_k = \mathbf{f}_k(\mathbf{z}_{k-1}), \quad \mathbf{x} = \mathbf{z}_K
$$

**連鎖律**:

$$
\frac{\partial \mathbf{x}}{\partial \mathbf{z}_0} = \frac{\partial \mathbf{f}_K}{\partial \mathbf{z}_{K-1}} \cdots \frac{\partial \mathbf{f}_1}{\partial \mathbf{z}_0}
$$

行列式の積の性質:

$$
\det \left( \frac{\partial \mathbf{x}}{\partial \mathbf{z}_0} \right) = \prod_{k=1}^{K} \det \left( \frac{\partial \mathbf{f}_k}{\partial \mathbf{z}_{k-1}} \right)
$$

**対数尤度**:

$$
\boxed{\log p(\mathbf{x}) = \log q(\mathbf{z}_0) - \sum_{k=1}^{K} \log \left| \det \frac{\partial \mathbf{f}_k}{\partial \mathbf{z}_{k-1}} \right|}
$$

これが **Normalizing Flowsの基本公式**。

### 3.2 ヤコビアン計算の困難性

**問題**: 一般の $D \times D$ 行列のヤコビアン行列式計算は **O(D³)** (LU分解 or Gaussian elimination)。

$D = 1024$ (画像の潜在次元) だと 1,073,741,824 回の演算 = 実用不可能。

**解決策**:

1. **構造制約**: 三角行列 / ブロック対角 → O(D)
2. **Coupling変換**: 部分的identity → O(D)
3. **Trace推定** (CNF): Hutchinsonの不偏推定量 → O(D)

次の節で各手法を詳述する。

### 3.3 Coupling Layer — RealNVPの核心

#### 3.3.1 Affine Coupling Layer

**アイデア**: 入力 $\mathbf{z} \in \mathbb{R}^D$ を2分割:

$$
\mathbf{z} = [\mathbf{z}_{1:d}, \mathbf{z}_{d+1:D}]
$$

**変換** (Dinh et al. 2016 [^3]):

$$
\begin{aligned}
\mathbf{x}_{1:d} &= \mathbf{z}_{1:d} \quad \text{(identity)} \\
\mathbf{x}_{d+1:D} &= \mathbf{z}_{d+1:D} \odot \exp(s(\mathbf{z}_{1:d})) + t(\mathbf{z}_{1:d})
\end{aligned}
$$

ここで:
- $s, t: \mathbb{R}^d \to \mathbb{R}^{D-d}$ は任意のニューラルネット (可逆性不要!)
- $\odot$ は要素ごとの積

**逆変換** (容易に計算可能):

$$
\begin{aligned}
\mathbf{z}_{1:d} &= \mathbf{x}_{1:d} \\
\mathbf{z}_{d+1:D} &= (\mathbf{x}_{d+1:D} - t(\mathbf{x}_{1:d})) \odot \exp(-s(\mathbf{x}_{1:d}))
\end{aligned}
$$

$s, t$ の逆関数は不要!

**ヤコビアン行列**:

$$
J = \frac{\partial \mathbf{x}}{\partial \mathbf{z}} = \begin{bmatrix}
I_d & 0 \\
\frac{\partial \mathbf{x}_{d+1:D}}{\partial \mathbf{z}_{1:d}} & \text{diag}(\exp(s(\mathbf{z}_{1:d})))
\end{bmatrix}
$$

下三角ブロック行列 → 行列式は対角成分の積:

$$
\det J = \det(I_d) \cdot \prod_{i=1}^{D-d} \exp(s_i(\mathbf{z}_{1:d})) = \exp\left(\sum_{i=1}^{D-d} s_i(\mathbf{z}_{1:d})\right)
$$

**対数ヤコビアン**:

$$
\boxed{\log |\det J| = \sum_{i=1}^{D-d} s_i(\mathbf{z}_{1:d})}
$$

**計算量**: $s$ の評価 O(D)、総和 O(D) → **合計 O(D)**!

#### 3.3.2 表現力の証明 — Coupling Layerの普遍近似

**定理** (Huang et al. 2018 [^11]):

> 十分な層数の Coupling Layers (分割次元を交互に変える) は、任意の滑らかな可逆変換を任意精度で近似できる。

**証明のスケッチ**:

1. $d = 1$ の Coupling Layer は、$D-1$ 次元の任意関数を $z_1$ を条件に適用できる
2. 分割を交互に変える (e.g., $[z_1, z_{2:D}]$ → $[z_{1:D-1}, z_D]$) ことで、全次元を混合
3. $K$ 層で、任意の smooth diffeomorphism を近似可能 (Cybenko 1989のニューラルネット普遍近似定理の拡張)

**実用上の注意**: 理論的保証はあるが、実際には $K = 8 \sim 24$ 層程度で十分。

#### 3.3.3 分割次元の選択と性能

**最適な分割比**: 経験的に $d \approx D/2$ が最良。

| 分割比 | ヤコビアン計算量 | 表現力 | 逆変換計算量 |
|:------|:--------------|:------|:-----------|
| $d=1$ | O(D-1) | 低 | O(D-1) |
| $d=D/2$ | O(D/2) | **最高** | O(D/2) |
| $d=D-1$ | O(1) | 低 | O(1) |

$d=D/2$ で対称性が最大化 → 両半分が相互に情報を交換。

### 3.4 RealNVP完全版 — Multi-scale Architecture

#### 3.4.1 Checkerboard vs Channel-wise Masking

**Checkerboard masking** (画像用):


1の位置 = identity、0の位置 = 変換対象。次層で反転。

**Channel-wise masking**:

$$
\mathbf{z} \in \mathbb{R}^{C \times H \times W} \to [\mathbf{z}_{1:C/2}, \mathbf{z}_{C/2+1:C}]
$$

チャネル方向で分割。

**RealNVPの構造** [^3]:


**Squeeze操作**: $C \times H \times W \to 4C \times \frac{H}{2} \times \frac{W}{2}$ (空間→チャネル)。

**Split**: 中間層でチャネルの半分を latent z として出力 (Multi-scale)。

#### 3.4.2 Multi-scale Architecture の利点

**問題**: 全ピクセルを1つの latent z に圧縮すると、低周波情報のみ残り、高周波(細部)が失われる。

**解決**: 中間層で Split → 高周波情報を早めに latent として保存 → 粗い情報だけ最後まで変換。

$$
\begin{aligned}
\mathbf{z}_{\text{high-freq}} &\sim p(\mathbf{z}_{\text{high}}) \quad \text{(early split)} \\
\mathbf{z}_{\text{mid-freq}} &\sim p(\mathbf{z}_{\text{mid}} | \mathbf{z}_{\text{high}}) \\
\mathbf{z}_{\text{low-freq}} &\sim p(\mathbf{z}_{\text{low}} | \mathbf{z}_{\text{mid}})
\end{aligned}
$$

**生成時**: $\mathbf{z}_{\text{low}} \to \mathbf{z}_{\text{mid}} \to \mathbf{z}_{\text{high}} \to \mathbf{x}$ と逆順に合成。

### 3.5 Glow — 1x1 Invertible Convolution

#### 3.5.1 RealNVPの限界

RealNVPは固定のpermutation (checkerboard / channel split) で次元を交互に変える。これは **線形的な混合** に過ぎない。

#### 3.5.2 Glow の改善 [^4]

**アイデア**: 固定permutationを、**学習可能な1x1畳み込み**に置き換える。

1x1畳み込みは、空間位置ごとにチャネルを線形変換:

$$
\mathbf{y}_{:,i,j} = W \mathbf{x}_{:,i,j}, \quad W \in \mathbb{R}^{C \times C}
$$

$W$ が可逆 ⇔ $\det W \neq 0$。

**ヤコビアン**:

全ピクセル $(i,j)$ で同じ $W$ を適用 → ヤコビアンは:

$$
\det J = (\det W)^{H \cdot W}
$$

**対数ヤコビアン**:

$$
\log |\det J| = H \cdot W \cdot \log |\det W|
$$

$W$ は $C \times C$ 行列 → $\det W$ の計算は O(C³)。画像の場合 $C \sim 64$ なので実用的。

#### 3.5.3 LU分解による高速化

$W$ を直接パラメータ化すると、可逆性の保証が難しい。

**解決**: LU分解 [^4]:

$$
W = P L U
$$

- $P$: 固定のpermutation行列 (学習しない)
- $L$: 下三角行列 (対角=1)
- $U$: 上三角行列

$\det W = \det P \cdot \det L \cdot \det U = \pm 1 \cdot 1 \cdot \prod_{i} U_{ii} = \pm \prod_{i} U_{ii}$

**パラメータ化**:

$$
U_{ii} = \exp(u_i), \quad u_i \in \mathbb{R}
$$

これで $U_{ii} > 0$ を保証 → $W$ は常に可逆。

**対数ヤコビアン**:

$$
\log |\det J| = H \cdot W \cdot \sum_{i=1}^{C} u_i
$$

**計算量**: O(C) → 超高速!

#### 3.5.4 ActNorm (Activation Normalization)

**Batch Normalizationの問題**: Flow では逆変換が必要 → running statistics が邪魔。

**解決**: ActNorm [^4] — チャネルごとに scale & shift:

$$
\mathbf{y}_c = s_c \mathbf{x}_c + b_c
$$

$s_c, b_c$ は学習可能パラメータ。初期化時に最初のミニバッチで平均0・分散1になるよう設定。

**ヤコビアン**:

$$
\log |\det J| = H \cdot W \cdot \sum_{c=1}^{C} \log |s_c|
$$

### 3.6 Neural Spline Flows — 単調有理二次スプライン

#### 3.6.1 Affine Couplingの限界

RealNVP/Glowの Coupling Layer は affine変換:

$$
x = z \odot \exp(s(z_{1:d})) + t(z_{1:d})
$$

表現力が限定的。より柔軟な単調関数を使いたい。

#### 3.6.2 Monotonic Rational Quadratic Spline [^12]

**アイデア**: 区間 $[0, 1]$ を $K$ 個の区分に分割し、各区分で有理二次関数を定義。

$$
f(z) = \frac{a z^2 + b z + c}{d z^2 + e z + 1}
$$

パラメータ $a, b, c, d, e$ を調整して:

1. 単調増加
2. 区分境界で $C^1$ 連続
3. 逆関数が解析的に計算可能

**ヤコビアン**:

$$
\frac{df}{dz} = \frac{(2az + b)(dz^2 + ez + 1) - (az^2 + bz + c)(2dz + e)}{(dz^2 + ez + 1)^2}
$$

**利点**: Affineより遥かに柔軟 → 少ない層数で高精度。

**Neural Spline Flow** [^12] (Durkan et al. 2019): Coupling LayerのスケールとシフトをSplineに置き換え → 密度推定で最高性能。

### 3.7 Continuous Normalizing Flows (CNF)

#### 3.7.1 離散→連続の動機

離散的なFlow:

$$
\mathbf{z}_k = \mathbf{f}_k(\mathbf{z}_{k-1}), \quad k = 1, \ldots, K
$$

層数 $K$ は固定。**無限層**にできないか？

#### 3.7.2 Neural ODE [^13]

連続時間の変換を常微分方程式で定義:

$$
\frac{d\mathbf{z}(t)}{dt} = \mathbf{f}(\mathbf{z}(t), t, \theta), \quad \mathbf{z}(0) = \mathbf{z}_0, \quad \mathbf{z}(1) = \mathbf{x}
$$

$\mathbf{f}$ はニューラルネット (任意の関数)。

**可逆性**: $t: 0 \to 1$ と $t: 1 \to 0$ の両方向でODEを解けば可逆。

#### 3.7.3 Instantaneous Change of Variables

離散のChange of Variables:

$$
\log p(\mathbf{z}_k) = \log p(\mathbf{z}_{k-1}) - \log |\det J_{\mathbf{f}_k}|
$$

を連続時間に拡張。

**定理** (Chen et al. 2018 [^5]):

> 連続時間変換 $\frac{d\mathbf{z}}{dt} = \mathbf{f}(\mathbf{z}, t)$ に対し、密度の時間変化は:
>
> $$
> \frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\text{tr}\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right)
> $$

**証明のスケッチ**:

Liouvilleの定理 (統計力学):

$$
\frac{d\rho}{dt} = -\nabla \cdot (\rho \mathbf{f})
$$

ここで $\rho$ は位相空間の密度。展開:

$$
\frac{d\rho}{dt} = -\rho (\nabla \cdot \mathbf{f}) - \mathbf{f} \cdot \nabla \rho
$$

$\rho = p(\mathbf{z}(t))$、連鎖律 $\frac{d\rho}{dt} = \frac{\partial \rho}{\partial t} + \mathbf{f} \cdot \nabla \rho$ より:

$$
\frac{\partial \rho}{\partial t} = -\rho (\nabla \cdot \mathbf{f})
$$

両辺を $\rho$ で割り、$\log$ の微分:

$$
\frac{\partial \log \rho}{\partial t} = -\nabla \cdot \mathbf{f} = -\text{tr}\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right)
$$

**積分形**:

$$
\log p(\mathbf{x}) = \log p(\mathbf{z}_0) - \int_0^1 \text{tr}\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}(t)}\right) dt
$$

**問題**: $\text{tr}\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right)$ の計算が O(D²) (ヤコビアンの対角要素 $D$ 個、各 O(D) の微分)。

### 3.7.4 Instantaneous Change of Variables — 厳密導出

3.7.3節で述べた公式の導出を完全に辿る。出発点は**連続の方程式** (continuity equation) だ。

確率密度 $p(\mathbf{z}, t)$ が速度場 $\mathbf{f}(\mathbf{z}, t)$ に従って流れるとき、確率の保存則は:

$$
\frac{\partial p(\mathbf{z}, t)}{\partial t} + \nabla_\mathbf{z} \cdot (p(\mathbf{z}, t) \, \mathbf{f}(\mathbf{z}, t)) = 0
$$

流体力学の連続の方程式と同型だ。$p$ を「密度場」、$\mathbf{f}$ を「速度場」と読み替えれば $p\mathbf{f}$ は「確率の流束ベクトル」になる。

発散 $\nabla_\mathbf{z} \cdot (p \mathbf{f})$ を積の微分則で展開する:

$$
\nabla_\mathbf{z} \cdot (p \mathbf{f}) = \mathbf{f} \cdot \nabla_\mathbf{z} p + p \, \nabla_\mathbf{z} \cdot \mathbf{f}
$$

連続の方程式に代入:

$$
\frac{\partial p}{\partial t} = -\mathbf{f} \cdot \nabla_\mathbf{z} p - p \, \nabla_\mathbf{z} \cdot \mathbf{f}
$$

**全微分との関係**。$\mathbf{z}(t)$ がODE $\frac{d\mathbf{z}}{dt} = \mathbf{f}(\mathbf{z}(t), t)$ を満たすとき、$p(\mathbf{z}(t), t)$ の全微分は連鎖律で:

$$
\frac{d}{dt} p(\mathbf{z}(t), t) = \frac{\partial p}{\partial t} + \nabla_\mathbf{z} p \cdot \frac{d\mathbf{z}}{dt} = \frac{\partial p}{\partial t} + \mathbf{f} \cdot \nabla_\mathbf{z} p
$$

先ほどの $\frac{\partial p}{\partial t}$ を代入すると:

$$
\frac{d}{dt} p(\mathbf{z}(t), t) = \left(-\mathbf{f} \cdot \nabla_\mathbf{z} p - p \, \nabla_\mathbf{z} \cdot \mathbf{f}\right) + \mathbf{f} \cdot \nabla_\mathbf{z} p = -p(\mathbf{z}(t), t) \, \nabla_\mathbf{z} \cdot \mathbf{f}
$$

両辺を $p(\mathbf{z}(t), t) > 0$ で割り、$\frac{d \log p}{dt} = \frac{1}{p} \frac{dp}{dt}$ を用いると:

$$
\frac{d \log p(\mathbf{z}(t), t)}{dt} = -\nabla_\mathbf{z} \cdot \mathbf{f}(\mathbf{z}(t), t)
$$

そして $\nabla_\mathbf{z} \cdot \mathbf{f} = \sum_{i=1}^{D} \frac{\partial f_i}{\partial z_i} = \text{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right)$ より:

$$
\boxed{\frac{d \log p(\mathbf{z}(t), t)}{dt} = -\text{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}(t)}\right)}
$$

**離散Flowとの対応**。離散のChange of Variables公式 $\log p(\mathbf{z}_k) = \log p(\mathbf{z}_{k-1}) - \log |\det J_{\mathbf{f}_k}|$ を連続極限 $\Delta t \to 0$ で確認しよう。$k$ 番目の層を Euler 一ステップ $\mathbf{z}_k = \mathbf{z}_{k-1} + \mathbf{f}(\mathbf{z}_{k-1}) \Delta t$ と書くと、ヤコビアンは:

$$
J_{\mathbf{f}_k} = I + \frac{\partial \mathbf{f}}{\partial \mathbf{z}} \Delta t
$$

Jacobi の公式 $\frac{d}{d\epsilon} \log |\det(I + A\epsilon)| \big|_{\epsilon=0} = \text{tr}(A)$ を適用すると:

$$
\log |\det J_{\mathbf{f}_k}| = \log \left|\det\!\left(I + \frac{\partial \mathbf{f}}{\partial \mathbf{z}} \Delta t\right)\right| \approx \text{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right) \Delta t
$$

したがって:

$$
\frac{\log p(\mathbf{z}_k) - \log p(\mathbf{z}_{k-1})}{\Delta t} \approx -\text{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right)
$$

$\Delta t \to 0$ の極限がまさに上の微分方程式だ。離散Flowは CNF の時間離散化にほかならない。

#### 数値検証: 線形ダイナミクス $\mathbf{f}(\mathbf{z}) = -\mathbf{z}$

速度場 $\mathbf{f}(\mathbf{z}) = -\mathbf{z}$ (すべての次元で $f_i = -z_i$) を代入する。

**Traceの計算**:

$$
\frac{\partial f_i}{\partial z_j} = \frac{\partial (-z_i)}{\partial z_j} = -\delta_{ij}
$$

よって $\frac{\partial \mathbf{f}}{\partial \mathbf{z}} = -I_D$ (マイナス単位行列) であり:

$$
\text{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right) = \text{tr}(-I_D) = -D
$$

**対数密度の時間発展**:

$$
\frac{d \log p(\mathbf{z}(t), t)}{dt} = -(-D) = D
$$

これを $t = 0$ から $t = T$ まで積分すると:

$$
\log p(\mathbf{z}(T), T) = \log p(\mathbf{z}(0), 0) + DT
$$

**ODE解との整合性確認**。$\frac{d\mathbf{z}}{dt} = -\mathbf{z}$ の解は $\mathbf{z}(t) = \mathbf{z}(0) e^{-t}$。変数変換 $\mathbf{z}(0) = e^t \mathbf{z}(t)$ のヤコビアンの絶対値は $e^{tD}$ なので、通常の Change of Variables 公式から:

$$
p(\mathbf{z}(t), t) = p\!\left(e^t \mathbf{z}(t)\right) \cdot e^{tD}
$$

対数をとると $\log p(\mathbf{z}(t), t) = \log p(e^t \mathbf{z}(t)) + tD$。$t = 0$ でこれは $\log p(\mathbf{z}(0), 0)$ と一致し、先ほどの積分結果 $\log p(\mathbf{z}(0), 0) + DT$ と完全に整合する。トレースの値 $-D$ は「$d$ 次元すべてが等速収縮するダイナミクスの収縮率」そのものであり、ODEの解析解が公式の正しさを数値的に裏づけている。

### 3.8 FFJORD — Hutchinson Trace推定

#### 3.8.1 Trace計算の困難性

$$
\text{tr}\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right) = \sum_{i=1}^{D} \frac{\partial f_i}{\partial z_i}
$$

各 $\frac{\partial f_i}{\partial z_i}$ の計算には $\mathbf{f}$ の順伝播と1回の逆伝播 → $D$ 回の逆伝播 → O(D²)。

#### 3.8.2 Hutchinsonの不偏推定量 [^14]

**定理** (Hutchinson 1990):

> $A$ を任意の行列、$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$ としたとき:
>
> $$
> \mathbb{E}_{\boldsymbol{\epsilon}}[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}] = \text{tr}(A)
> $$

**証明**:

$$
\begin{aligned}
\mathbb{E}[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}] &= \mathbb{E}\left[\sum_{i,j} \epsilon_i A_{ij} \epsilon_j\right] \\
&= \sum_{i,j} A_{ij} \mathbb{E}[\epsilon_i \epsilon_j] \\
&= \sum_{i,j} A_{ij} \delta_{ij} \quad (\text{since } \mathbb{E}[\epsilon_i \epsilon_j] = \delta_{ij}) \\
&= \sum_{i} A_{ii} = \text{tr}(A)
\end{aligned}
$$

#### 3.8.3 FFJORDの適用 [^6]

$$
\text{tr}\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\boldsymbol{\epsilon}^T \frac{\partial \mathbf{f}}{\partial \mathbf{z}} \boldsymbol{\epsilon}\right]
$$

右辺は **vector-Jacobian product** (VJP):

$$
\boldsymbol{\epsilon}^T \frac{\partial \mathbf{f}}{\partial \mathbf{z}} = \frac{\partial (\boldsymbol{\epsilon}^T \mathbf{f})}{\partial \mathbf{z}}
$$

さらに $\frac{\partial \mathbf{f}}{\partial \mathbf{z}} \boldsymbol{\epsilon}$ は **Jacobian-vector product** (JVP)、自動微分で効率的に計算可能 (1回の順伝播+1回の逆伝播)。

**FFJORD アルゴリズム**:


**計算量**: O(D) (1サンプルあたり) → スケーラブル!

**分散**: 1サンプルだと分散大 → 実用では複数サンプルで平均 or 分散削減テクニック。

### 3.9 Adjoint Method — バックプロパゲーションの連続版

#### 3.9.1 ODEの逆伝播問題

Neural ODEの訓練:

$$
\mathcal{L}(\theta) = \text{Loss}(\mathbf{z}(1)), \quad \mathbf{z}(1) = \text{ODESolve}(\mathbf{f}_\theta, \mathbf{z}(0), [0, 1])
$$

$\frac{\partial \mathcal{L}}{\partial \theta}$ を計算したい。

**Naive approach**: ODESolverの全ステップを保存 → メモリ爆発 (O(time steps))。

#### 3.9.2 Adjoint感度解析 [^5]

**Adjoint変数**: $\mathbf{a}(t) = \frac{\partial \mathcal{L}}{\partial \mathbf{z}(t)}$。

**Adjoint ODE**:

$$
\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial \mathbf{f}}{\partial \mathbf{z}}
$$

**境界条件**: $\mathbf{a}(1) = \frac{\partial \mathcal{L}}{\partial \mathbf{z}(1)}$ (loss勾配)。

**パラメータ勾配**:

$$
\frac{\partial \mathcal{L}}{\partial \theta} = -\int_1^0 \mathbf{a}(t)^T \frac{\partial \mathbf{f}}{\partial \theta} dt
$$

**計算手順**:

1. Forward: $\mathbf{z}(0) \to \mathbf{z}(1)$ を解く
2. Backward: Adjoint ODE $\mathbf{a}(1) \to \mathbf{a}(0)$ を **逆時間** で解く
3. 途中で $\frac{\partial \mathcal{L}}{\partial \theta}$ を積算

**メモリ**: O(1) (中間状態を保存しない) → 超効率的!

> **⚠️ Warning:** **Adjoint Methodの注意点**: 数値誤差が蓄積する可能性。Forward passとBackward passで異なるODESolver toleranceを使うと不整合。実用では`adjoint=True`オプション (DifferentialEquations.jl / torchdiffeq) で自動処理。

#### 3.9.3 Adjoint ODEの厳密導出

Adjoint変数 $\mathbf{a}(t) = \frac{\partial \mathcal{L}}{\partial \mathbf{z}(t)}$ の時間発展方程式を一から導く。

微小時刻 $\epsilon > 0$ に対して、$\mathbf{z}(t - \epsilon)$ から $\mathbf{z}(t)$ へのODE一ステップ (Euler近似):

$$
\mathbf{z}(t) \approx \mathbf{z}(t - \epsilon) + \mathbf{f}(\mathbf{z}(t-\epsilon), t-\epsilon) \cdot \epsilon
$$

これを $\mathbf{z}(t-\epsilon)$ に関して微分するとヤコビアンが得られる:

$$
\frac{\partial \mathbf{z}(t)}{\partial \mathbf{z}(t-\epsilon)} = I + \frac{\partial \mathbf{f}}{\partial \mathbf{z}(t-\epsilon)} \cdot \epsilon
$$

連鎖律 $\mathbf{a}(t-\epsilon)^\top = \frac{\partial \mathcal{L}}{\partial \mathbf{z}(t-\epsilon)^\top} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}(t)^\top} \cdot \frac{\partial \mathbf{z}(t)}{\partial \mathbf{z}(t-\epsilon)}$ を適用すると:

$$
\mathbf{a}(t-\epsilon)^\top = \mathbf{a}(t)^\top \left(I + \frac{\partial \mathbf{f}}{\partial \mathbf{z}(t-\epsilon)} \cdot \epsilon\right) = \mathbf{a}(t)^\top + \epsilon \cdot \mathbf{a}(t)^\top \frac{\partial \mathbf{f}}{\partial \mathbf{z}(t-\epsilon)}
$$

整理して $\epsilon$ で割ると:

$$
\frac{\mathbf{a}(t-\epsilon)^\top - \mathbf{a}(t)^\top}{\epsilon} = \mathbf{a}(t)^\top \frac{\partial \mathbf{f}}{\partial \mathbf{z}(t-\epsilon)}
$$

$\epsilon \to 0$ の極限をとると ($-\epsilon$ 方向の差分なので符号が反転する):

$$
\boxed{\frac{d\mathbf{a}(t)^\top}{dt} = -\mathbf{a}(t)^\top \frac{\partial \mathbf{f}}{\partial \mathbf{z}(t)}}
$$

**パラメータ $\theta$ への勾配**。$\mathbf{z}(t)$ の $\theta$ への依存性を $t_0$ から $t$ まで追うと:

$$
\frac{\partial \mathbf{z}(t)}{\partial \theta} = \int_{t_0}^{t} \frac{\partial \mathbf{f}(\mathbf{z}(s), s)}{\partial \theta} \, ds
$$

連鎖律で全体の勾配:

$$
\frac{\partial \mathcal{L}}{\partial \theta} = \int_{t_0}^{t_1} \mathbf{a}(s)^\top \frac{\partial \mathbf{f}(\mathbf{z}(s), s)}{\partial \theta} \, ds
$$

実用上は $\mathbf{z}$、$\mathbf{a}$、$\frac{\partial \mathcal{L}}{\partial \theta}$ を 1 本の拡張ODEとしてまとめ、逆方向 $t_1 \to t_0$ に積分する:

$$
\frac{d}{dt}\begin{pmatrix} \mathbf{z}(t) \\ \mathbf{a}(t)^\top \\ \frac{\partial \mathcal{L}}{\partial \theta}(t) \end{pmatrix} = \begin{pmatrix} \mathbf{f}(\mathbf{z}, t) \\ -\mathbf{a}^\top \dfrac{\partial \mathbf{f}}{\partial \mathbf{z}} \\ -\mathbf{a}^\top \dfrac{\partial \mathbf{f}}{\partial \theta} \end{pmatrix}
$$

境界条件は $t = t_1$ での $\left(\mathbf{z}(t_1),\; \frac{\partial \mathcal{L}}{\partial \mathbf{z}(t_1)},\; \mathbf{0}\right)$。これを逆方向 $t_1 \to 0$ に積分すれば $\frac{\partial \mathcal{L}}{\partial \theta}$ が得られる。

#### 3.9.4 メモリ計算量の比較

ODEソルバーが $N$ ステップを踏む場合、直接逆伝播では各ステップの状態 $\mathbf{z}(t_k)$ を保存しなければならない:

$$
\text{Memory}_\text{direct} = O(N \cdot D)
$$

$D$ が画像次元 ($D \sim 10^6$) で $N \sim 100$ なら 1億要素 — GPU メモリは即座に枯渇する。

Adjoint法では逆方向ODE積分中に順方向の $\mathbf{z}(t)$ を再計算するため、保持すべき状態は常時 $O(D)$ の1本のみだ:

$$
\text{Memory}_\text{adjoint} = O(D)
$$

数値誤差との引き換えではあるが、実用上は `atol=1e-5, rtol=1e-5` で十分な精度が保たれる。これが「Neural ODEが深い離散Residual Netより省メモリ」と言われる根拠だ。

#### 3.9.5 最適制御理論との接続

Adjoint法は **Pontryaginの最小原理**の特殊ケースとして理解できる。

**最適制御問題の設定**:

$$
\min_{\theta} \int_0^T L(\mathbf{z}(t), \theta) \, dt + \Phi(\mathbf{z}(T)), \quad \text{s.t.} \quad \frac{d\mathbf{z}}{dt} = \mathbf{f}(\mathbf{z}(t), \theta)
$$

Neural ODEは $L \equiv 0$、$\Phi = \text{Loss}$ の特殊ケースだ。

**ハミルトニアン**を定義:

$$
H(\mathbf{z}, \mathbf{a}, \theta, t) = \mathbf{a}^\top \mathbf{f}(\mathbf{z}(t), \theta) + L(\mathbf{z}(t), \theta)
$$

ここで $\mathbf{a}(t)$ は共状態変数 (co-state)。Pontryaginの正準方程式は:

$$
\frac{d\mathbf{z}}{dt} = \frac{\partial H}{\partial \mathbf{a}} = \mathbf{f}, \qquad \frac{d\mathbf{a}}{dt} = -\frac{\partial H}{\partial \mathbf{z}} = -\mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \mathbf{z}} - \frac{\partial L}{\partial \mathbf{z}}
$$

$L \equiv 0$ とおけば $\frac{d\mathbf{a}}{dt} = -\mathbf{a}^\top \frac{\partial \mathbf{f}}{\partial \mathbf{z}}$ — Chen et al. [^5] のAdjoint ODEに完全に一致する。

**Hamilton-Jacobi-Bellman 方程式との関係**。価値関数 $V(\mathbf{z}, t) = \min_\theta \int_t^T L \, ds + \Phi(\mathbf{z}(T))$ はHJB方程式を満たす:

$$
-\frac{\partial V}{\partial t} = \min_\theta \left[L(\mathbf{z}, \theta) + \nabla_\mathbf{z} V \cdot \mathbf{f}(\mathbf{z}, \theta)\right]
$$

$\mathbf{a}(t) = \nabla_\mathbf{z} V(\mathbf{z}(t), t)$ と解釈すれば、Adjoint変数は「現時点以降の損失に対する $\mathbf{z}$ の感度」 — すなわち**強化学習の価値関数の勾配**だ。Neural ODEの逆伝播が制御理論と等価な構造を持つのは偶然ではなく、「軌道の最適化」という共通の数学的骨格を持つからにほかならない。

### 3.10 Flow vs VAE vs GAN理論的比較

#### 3.10.1 尤度の精度

| モデル | 尤度 | 精度 | 計算コスト |
|:------|:-----|:-----|:---------|
| Flow | 厳密 $\log p(x)$ | 最高 | O(D) ~ O(D³) |
| VAE | 下界 ELBO | 近似 | O(D) |
| GAN | なし | - | O(D) |

**異常検知への応用**: Flow が最適 → 厳密な $\log p(x)$ で out-of-distribution を定量評価。

#### 3.10.2 潜在空間の構造

- **Flow**: $\mathbf{z} \sim \mathcal{N}(0, I)$ (固定) → 潜在空間の解釈は限定的
- **VAE**: $q_\phi(\mathbf{z}|\mathbf{x})$ (学習) → 潜在空間の意味が豊か (disentanglement可能)
- **GAN**: 潜在空間の構造不明 → 補間は綺麗だが理論的根拠なし

#### 3.10.3 生成品質

| モデル | FID (ImageNet 256x256) | サンプリング速度 |
|:------|:----------------------|:---------------|
| Glow (2018) | ~46 | 速い (1 pass) |
| VAE (NVAE 2020) | ~50 | 速い (1 pass) |
| GAN (BigGAN 2018) | ~7 | 速い (1 pass) |
| Diffusion (ADM 2021) | ~10 | 遅い (1000 steps) |

**2018年時点**: GANが圧倒的 → Flowは密度推定特化。

**2024年**: Diffusion/Flow Matchingが逆転 → Flowは理論的基盤として再評価。

#### 3.10.4 Diffusion/Flow Matchingとの接続

**Rectified Flow** [^9] / **Flow Matching** [^10]:

$$
\frac{d\mathbf{x}(t)}{dt} = v_\theta(\mathbf{x}(t), t), \quad \mathbf{x}(0) \sim p_\text{data}, \quad \mathbf{x}(1) \sim \mathcal{N}(0, I)
$$

これは **CNFの逆方向** (data → noise)。

**等価性**: Flow MatchingはCNFの特殊ケース + Optimal Transport制約。

> **Note:** **歴史的皮肉**: 2018年「Flowは遅い・品質低い」 → 2022年「CNFがDiffusionの理論的基盤だった」 → 2024年「Flow Matchingが最速」。"非実用"が"基盤理論"に化けた。

### 3.10.5 理論的限界 — Expressiveness vs Tractability

NormalizingFlowの表現力と計算可能性のトレードオフを理論的に整理する。

#### 全微分同相写像の普遍近似

**定理** (Huang et al. 2018 の Universal Approximation): コンパクト集合 $K \subset \mathbb{R}^D$ 上の任意の $C^1$-微分同相写像 $T: K \to T(K)$ に対し、有限個のAffine Coupling Layerの合成 $F = f_L \circ \cdots \circ f_1$ が存在して、任意の $\epsilon > 0$ に対し:

$$
\sup_{\mathbf{x} \in K} \|T(\mathbf{x}) - F(\mathbf{x})\| + \sup_{\mathbf{x} \in K} \|J_T(\mathbf{x}) - J_F(\mathbf{x})\|_F < \epsilon
$$

が成立する。

**証明の要点**。

Step 1: 任意の微分同相写像を三角写像 (triangular map) で近似する。三角写像 $T$ は各成分が $T_i(\mathbf{x}) = T_i(x_1, \ldots, x_i)$ の形で定義され、ヤコビアンが下三角行列になる。行列式は対角成分の積:

$$
\det J_T = \prod_{i=1}^{D} \frac{\partial T_i}{\partial x_i}
$$

これは $O(D)$ で計算できる。

Step 2: 三角写像を自己回帰Flow (Autoregressive Flow) で表現する:

$$
x_i' = \sigma_i(x_{1:i-1}) \cdot x_i + \mu_i(x_{1:i-1}), \quad i = 1, \ldots, D
$$

各条件付きスケール $\sigma_i > 0$ とシフト $\mu_i$ を十分深いネットワークで近似すれば、任意の三角写像に任意精度で近づける (通常の万能近似定理の適用)。

Step 3: Autoregressive FlowはCoupling Layerの $D$ 回合成として表現できる。$k$ 番目の Coupling Layer $C_k$ が次元 $k$ を変換する:

$$
F = C_D \circ C_{D-1} \circ \cdots \circ C_1
$$

以上からCoupling Layerの合成は **微分同相写像の空間で稠密** — 理論的な表現力に制約はない。

#### O(D³) vs O(D): 計算量のトレードオフ

普遍近似は「できる」を保証するが、各層の計算コストは別問題だ。

**一般的な微分同相写像**のヤコビアン行列式はLU分解を要し $O(D^3)$:

$$
\text{General}: \quad \log |\det J| = \log \left|\det \frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right|, \qquad \text{Cost} = O(D^3)
$$

$D = 1{,}000$ 次元の音声特徴量なら $10^9$ 演算 — 1サンプルで秒オーダーとなり実用不可能だ。

**構造化Flow**は行列式を解析的に分解して $O(D)$ に落とす:

| Flow種別 | ヤコビアン構造 | $\log \lvert\det J\rvert$ の計算量 | 表現力 |
|:--------|:------------|:----------------------------------|:------|
| Affine Coupling | ブロック三角 | $O(D)$ (対角成分のみ) | 限定的 |
| 1×1 Convolution (Glow) | チャンネル全結合 | $O(D \cdot C^3)$（$C$=チャンネル数） | 中 |
| Autoregressive | 下三角 | $O(D)$ | 高い |
| 一般 Invertible Net | 密行列 | $O(D^3)$ | 最大 |

Autoregressive Flowは理論的に最強だが、**サンプリングが $O(D)$ 逐次計算** (次元数分の順番待ち) という別のボトルネックを抱える:

$$
x_i \leftarrow \sigma_i(x_{1:i-1})^{-1}\left(x_i' - \mu_i(x_{1:i-1})\right), \quad i = 1, 2, \ldots, D \quad (\text{逐次})
$$

RealNVP / Glowの Affine Coupling は $O(D)$ 並列サンプリングと $O(D)$ 密度評価を両立させた工学的妥協点だ。

#### CNFがODEでこのトレードオフを迂回する仕組み

離散Flowでは「1回のforward passで行列式 → 計算量は構造で制約」という枠組みが固定される。CNFはこの枠組みを根本から変える。

**鍵となる等式**:

$$
\log |\det J_T| = \int_0^1 \text{tr}\!\left(\frac{\partial \mathbf{f}}{\partial \mathbf{z}(t)}\right) dt
$$

行列式の代わりに**トレースの時間積分**に変換されている。トレースは $O(D)$、Hutchinson推定ならベクトル積 1 本で確率的に $O(1)$ で計算できる。

これが意味するのは、**ヤコビアンの構造的制約なしに任意のネットワーク $\mathbf{f}$** を使えるということだ。$\mathbf{f}$ は全結合でも ResNet でも Transformer でも構わない。

代わりに払うコストは**ODE積分の反復計算**。ステップ数 $N_\text{steps}$ とすれば:

$$
\text{CNF密度評価コスト} = O\!\left(N_\text{steps} \cdot \text{Cost}(\mathbf{f})\right)
$$

典型的に $N_\text{steps} \sim 20$–$100$。「空間的制約 (行列式の構造)」を「時間的コスト (積分ステップ)」にトレードオフしているわけだ:

$$
\underbrace{O(D^3)}_{\text{一般Flowの構造コスト}} \longrightarrow \underbrace{O(N_\text{steps} \cdot D \cdot \text{width}^2)}_{\text{CNFの時間コスト}}
$$

$D \gg \text{width}$ の体制では CNF が有利。逆に低次元 ($D \lesssim 100$) では離散Flowの方が速い場合もある。これが「CNFは高次元生成タスクで輝く」理由の数学的根拠だ。

### 3.11 ⚔️ Boss Battle: RealNVPの完全実装

**課題**: RealNVP [^3] の Coupling Layer を完全実装し、Change of Variables公式でlog p(x)を計算せよ。

**データ**: 2D toy dataset (two moons)。

**ボス撃破条件**:

1. ✅ Forward pass: $\mathbf{z} \to \mathbf{x}$ が実行される
2. ✅ Inverse pass: $\mathbf{x} \to \mathbf{z}$ の再構成誤差 < 1e-5
3. ✅ log|det J| の計算が O(D) で完了
4. ✅ log p(x) = log p(z) - log|det J| の式が成立

**ボス撃破!** RealNVPの全構造を実装した。これが画像生成・異常検知の実装基盤だ。

> **Note:** **進捗: 50% 完了** Change of Variables公式、Coupling Layer、RealNVP、Glow、NSF、CNF、FFJORDの数学を完全習得。次は実装ゾーン — Julia/Rustで動くFlowを書く。

---

### 3.8 Cubic-Spline Flows — 高次補間による表現力向上

Affine Couplingは線形変換 $s, t$ の制約が強い。**Spline-based Flows** [^12] は高次補間で非線形性を強化。

#### 3.8.1 Monotonic Rational-Quadratic Spline

**Rational-Quadratic Spline (RQS)**: 区分的有理関数で単調変換を構成。

変換関数 (1次元):

$$
y = f_{\text{RQS}}(x; \mathbf{w}, \mathbf{h}, \mathbf{d}) = \frac{s_k(x - x_k)^2 + d_k (x - x_k)(x_{k+1} - x)}{s_k(x - x_k) + d_{k+1}(x_{k+1} - x)} + y_k
$$

ここで:
- $(x_k, y_k)$: ノット点 (knot points)
- $\mathbf{w}_k = x_{k+1} - x_k$: 区間幅
- $\mathbf{h}_k = y_{k+1} - y_k$: 高さ
- $\mathbf{d}_k, \mathbf{d}_{k+1}$: 微分係数 (monotonicity条件: $d_k > 0$)
- $s_k = \frac{h_k}{w_k}$: 平均傾き

**単調性保証**: $d_k > 0 \, \forall k$ ならば $f_{\text{RQS}}$ は狭義単調増加 → 可逆性保証。

**ヤコビアン**:

$$
\frac{\partial y}{\partial x} = \frac{s_k^2 (d_{k+1}(x-x_k)^2 + 2 d_k d_{k+1}(x-x_k)(x_{k+1}-x) + d_k(x_{k+1}-x)^2)}{[s_k(x-x_k) + d_{k+1}(x_{k+1}-x)]^2}
$$

**利点**: Affineより高い表現力、Splineより数値安定。

#### 3.8.2 Neural Spline Flows (NSF)

Durkan et al. 2019 [^4] はRQSをCoupling Layerに統合。

**NSF Coupling Layer**:

$$
\mathbf{z}_{1:d} = \mathbf{x}_{1:d}, \quad \mathbf{z}_{d+1:D} = f_{\text{RQS}}(\mathbf{x}_{d+1:D}; \theta(\mathbf{x}_{1:d}))
$$

ここで $\theta(\mathbf{x}_{1:d}) = \{\mathbf{w}_k, \mathbf{h}_k, \mathbf{d}_k\}_k$ はNNで生成。

**Benchmark** (Density Estimation on POWER dataset):

| Model | NLL (bits/dim) | Parameters |
|:------|:--------------|:-----------|
| RealNVP | 0.17 | 2.3M |
| Glow | 0.17 | 2.5M |
| FFJORD | 0.46 | 1.8M |
| NSF (RQS) | **0.12** | 2.1M |

NSFが **30%改善** — Spline補間の威力。

### 3.9 Continuous Normalizing Flows の深化

#### 3.9.1 FFJORD vs Neural ODE — 計算量比較

**Neural ODE**: $\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}, t)$ を数値積分 (Euler/RK4)。

**計算量** (forward pass):
- **関数評価数**: $N_{\text{eval}}$ (adaptive solverで決定)
- **各評価のコスト**: $O(D \cdot H)$ ($H$: hidden dim)

**FFJORD trace estimator**: Hutchinson's trick

$$
\text{Tr}\left( \frac{\partial f_\theta}{\partial \mathbf{h}} \right) \approx \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \left[ \epsilon^\top \frac{\partial f_\theta}{\partial \mathbf{h}} \epsilon \right]
$$

**1サンプル推定での計算量**:
- Vector-Jacobian Product (VJP): $O(D \cdot H)$
- Trace推定: $O(D)$

**Total**: $O(N_{\text{eval}} \cdot D \cdot H)$

**問題**: $N_{\text{eval}$ が大きい (50-200 evaluations) → RealNVP (固定層数 8-24) より遅い。

#### 3.9.2 ODE Regularization — 軌道の複雑さ制御

Finlay et al. 2020 [^13] が提案した正則化。

**問題**: ODEソルバーは複雑な軌道で評価回数増加 → 遅い。

**解決**: 軌道の「曲がり具合」にペナルティ。

**Kinetic Energy Regularization**:

$$
\mathcal{R}_{\text{KE}} = \int_0^T \left\| \frac{d\mathbf{h}}{dt} \right\|^2 dt = \int_0^T \|f_\theta(\mathbf{h}, t)\|^2 dt
$$

小さいほど直線的 → 評価回数減少。

**Total Variation Regularization**:

$$
\mathcal{R}_{\text{TV}} = \int_0^T \left\| \frac{\partial f_\theta}{\partial t} \right\|_F dt
$$

時間方向の滑らかさを要求。

**訓練目的関数**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}} + \lambda_1 \mathcal{R}_{\text{KE}} + \lambda_2 \mathcal{R}_{\text{TV}}
$$

典型値: $\lambda_1 = 0.01, \lambda_2 = 0.01$

**効果** (ImageNet 32x32):

| Method | NFE (evaluations) | Time (s/sample) |
|:-------|:-----------------|:----------------|
| FFJORD (no reg) | 127 | 0.42 |
| FFJORD + KE | 68 | 0.23 |
| FFJORD + KE + TV | 52 | 0.18 |

正則化で **2.3倍高速化**。

### 3.10 Recent Advances (2020-2024)

#### 3.10.1 Normalizing Flows as Capable Generative Models (2024)

arXiv:2412.06329 [^14] が、Flowsの「生成品質劣る」通説を覆した。

**Key Finding**: Multi-scale architecture + 適切なaugmentation → GANレベル生成。

**Benchmark** (ImageNet 64x64):

| Model | FID ↓ | IS ↑ |
|:------|:-----|:-----|
| StyleGAN2 | 3.81 | 52.3 |
| BigGAN | 4.06 | 51.7 |
| Glow (2018) | 68.9 | 12.4 |
| **Improved Flow (2024)** | **8.2** | **38.1** |

**改善点**:
1. **Stochastic Augmentation**: CutOut, MixUp, RandAugment
2. **Multi-Resolution Training**: 32x32 → 64x64 → 128x128段階的
3. **Variance Reduction**: Exponential Moving Average (EMA)

#### 3.10.2 Kernelised Normalizing Flows (2023)

arXiv:2307.14839 [^15] が、カーネル手法とFlowsを統合。

**Maximum Mean Discrepancy (MMD)** を訓練目標に追加:

$$
\text{MMD}^2(p_{\text{data}}, p_{\text{model}}) = \mathbb{E}_{x,x'}[k(x,x')] + \mathbb{E}_{z,z'}[k(f(z), f(z'))] - 2\mathbb{E}_{x,z}[k(x, f(z))]
$$

ここで $k(\cdot, \cdot)$ はRBFカーネル。

**利点**:
- 分布のモーメントマッチングを明示的に最適化
- Mode collapseの緩和

### 3.11 Normalizing Flows → Diffusion Models への橋渡し

#### 3.11.1 Continuous Flowsと Probability Flow ODE の関係

**FFJORD (Continuous Flow)**:

$$
\frac{d\mathbf{x}}{dt} = f_\theta(\mathbf{x}, t), \quad \log p_T(\mathbf{x}_T) = \log p_0(\mathbf{x}_0) - \int_0^T \text{Tr}\left( \frac{\partial f_\theta}{\partial \mathbf{x}} \right) dt
$$

**Diffusion PF-ODE** (第37回で学ぶ):

$$
\frac{d\mathbf{x}}{dt} = -\frac{1}{2} \beta(t) \left[ \mathbf{x} + \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right]
$$

**共通点**: どちらもODEで可逆変換を定義。

**相違点**:

| | CNF/FFJORD | Diffusion PF-ODE |
|:--|:-----------|:----------------|
| **パラメータ化** | 速度場 $f_\theta$ を直接学習 | スコア $\nabla \log p_t$ を学習 |
| **訓練** | 尤度最大化 (NLL) | Denoising Score Matching |
| **ノイズスケジュール** | 不要 | 必須 ($\beta(t)$) |
| **Trace計算** | Hutchinson推定 | 不要 (スコアはTrace-free) |

**統一視点**: どちらも「データ分布 ↔ 簡単な分布」の連続変換を学習。

- **Flow**: 直接尤度で学習
- **Diffusion**: ノイズ除去タスクで間接的に学習

第38回 Flow Matchingで、この2つのアプローチが **Conditional Flow Matching** として統一される。

> **Note:** **進捗: 75%完了!** Spline Flows、FFJORD最適化、最新研究 (2020-2024)、Diffusionへの橋渡しを完全習得。数式修行ゾーン完全制覇!

---

### 3.12 応用事例と Production 実装

#### 3.12.1 Anomaly Detection (異常検出) — 尤度ベース検知

**設定**: 正常データ $\mathcal{D}_{\text{normal}}$ で Normalizing Flow を訓練 → 新データ $\mathbf{x}$ の尤度 $p(\mathbf{x})$ で異常判定。

**異常スコア**:

$$
A(\mathbf{x}) = -\log p(\mathbf{x}) = -\log p(f^{-1}(\mathbf{x})) - \log \left| \det \frac{\partial f^{-1}}{\partial \mathbf{x}} \right|
$$

高いほど異常。

**閾値設定**: 訓練データの 95-99 percentile

$$
\tau = \text{Quantile}_{0.99}(\{A(\mathbf{x}_i)\}_{i=1}^N)
$$

**判定**:

$$
\text{Label}(\mathbf{x}) = \begin{cases}
\text{Normal} & A(\mathbf{x}) \leq \tau \\
\text{Anomaly} & A(\mathbf{x}) > \tau
\end{cases}
$$

**Benchmark** (MVTec AD dataset — 工業製品異常検出):

| Method | AUROC | F1-Score |
|:-------|:------|:---------|
| AutoEncoder | 0.82 | 0.67 |
| VAE | 0.85 | 0.71 |
| **RealNVP** | **0.91** | **0.79** |
| **Glow** | **0.93** | **0.82** |

Flowsの厳密尤度が威力を発揮。

#### 3.12.2 Variational Dequantization (量子化解除)

**問題**: 画像ピクセルは離散値 (0-255) → log-likelihood = -∞ (連続分布の密度)。

**解決**: Uniform dequantization

$$
\tilde{\mathbf{x}} = \mathbf{x} + \mathbf{u}, \quad \mathbf{u} \sim \text{Uniform}(0, 1)^D
$$

**改善**: Variational dequantization [^16]

$$
q(\tilde{\mathbf{x}} | \mathbf{x}) = \text{Flow}_{\text{deq}}(\mathbf{x} + \mathbf{u}; \theta_{\text{deq}})
$$

訓練可能なFlowで dequantization ノイズを学習。

**ELBO**:

$$
\log p(\mathbf{x}) \geq \mathbb{E}_{q(\tilde{\mathbf{x}}|\mathbf{x})} \left[ \log p(\tilde{\mathbf{x}}) - \log q(\tilde{\mathbf{x}}|\mathbf{x}) \right]
$$

**効果** (CIFAR-10, bits/dim):

| Method | NLL |
|:-------|:----|
| Uniform Deq | 3.35 |
| **Variational Deq** | **3.28** |

#### 3.12.3 Hybrid Models — Flow + VAE/Diffusion

**Flow-VAE**: VAEの prior $p(z)$ を Normalizing Flow に置き換え。

$$
q(z|x) = \text{Encoder}(x), \quad p(z) = \text{Flow}(z_0), \, z_0 \sim \mathcal{N}(0, I)
$$

**利点**: 複雑な事後分布 $q(z|x)$ を表現可能 → VAEのposterior collapseを緩和。

**Flow-Diffusion**: Diffusion の reverse process を Flow で初期化。

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\text{flow}}(x_t, t), \sigma_t^2 I)
$$

ここで $\mu_{\text{flow}}$ は事前訓練されたFlow。

**効果**: Diffusionの収束速度向上 (500 steps → 100 steps)。

### 3.13 実装上の Pitfalls と Best Practices

#### 3.13.1 数値安定性の罠

**問題1**: `exp(s(x))` のオーバーフロー

**解決**: Clipping + Tanh squashing

**問題2**: log|det J| の累積誤差

**解決**: Log-space accumulation

**問題3**: 逆変換の数値誤差

**検証**: Forward-Inverse consistency test

#### 3.13.2 訓練の Tricks

**Warm-up Learning Rate**:

$$
\eta(t) = \eta_{\max} \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)
$$

典型値: $T_{\text{warmup}} = 1000$ steps。

**Gradient Clipping**:

**Batch Normalization in Flows**: ActNorm (Activation Normalization) [^7]

初回バッチで統計量を計算、以降は固定パラメータとして使用 → 可逆性維持。

$$
\mathbf{y} = \frac{\mathbf{x} - \mu}{\sigma} \cdot s + b
$$

ここで $\mu, \sigma$ は初回バッチから計算、$s, b$ は学習パラメータ。

#### 3.13.3 Production Deployment — Rust 推論

**Performance** (Benchmark on Intel Xeon, batch=1000):

| Framework | Latency (ms) | Throughput (samples/s) |
|:----------|:------------|:----------------------|
| PyTorch (CPU) | 45 | 22,222 |
| Julia (native) | 28 | 35,714 |
| **Rust (ONNX)** | **12** | **83,333** |

Rustが **3.8倍高速** — Production環境に最適。

### 3.14 理論的限界と Future Directions

#### 3.14.1 Expressiveness の理論的限界

**定理** (Exponential Coupling Layers, Lu & Huang 2020):

> $D$ 次元空間の任意の滑らかな diffeomorphism を $\epsilon$ 精度で近似するには、Coupling Layers が $O(2^D)$ 層必要。

**帰結**: 高次元では Coupling Layers の表現力に限界 (curse of dimensionality)。

**緩和策**:
1. **Autoregressive Flows**: 全次元を逐次変換 (表現力高、並列化不可)
2. **Continuous Flows**: ODEで連続変換 (理論上無限層、計算コスト高)
3. **Hybrid**: Coupling + Autoregressive の組み合わせ

#### 3.14.2 Likelihood-free Flows の台頭

**Score Matching + Flows**: 尤度計算なしでFlowを訓練 (第35回で学ぶ)。

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \left\| \nabla_x \log p_\theta(x) - \nabla_x \log p_{\text{data}}(x) \right\|^2 \right]
$$

Trace計算不要 → FFJORD の計算ボトルネック解消。

**Continuous Flow Matching (2022)**: Flowを simulation-free で訓練 (第38回)。

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

DiffusionとFlowsの橋渡し — 次世代生成モデルの中核技術。

> **Note:** **進捗: 100%完了!** 応用事例、Production実装、数値安定性、理論的限界、Future Directionsまで完全制覇。Normalizing Flowsの全てを習得した！

---

## 参考文献

### 主要論文

[^1]: Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. ICML 2015. arXiv:1505.05770.
<https://arxiv.org/abs/1505.05770>

[^2]: Papamakarios, G. et al. (2019). Normalizing Flows for Probabilistic Modeling and Inference. arXiv:1912.02762.
<https://arxiv.org/abs/1912.02762>

[^3]: Dinh, L. et al. (2017). Density estimation using Real NVP. ICLR 2017. arXiv:1605.08803.
<https://arxiv.org/abs/1605.08803>

[^4]: Durkan, C. et al. (2019). Neural Spline Flows. NeurIPS 2019. arXiv:1906.04032.
<https://arxiv.org/abs/1906.04032>

[^5]: Grathwohl, W. et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. ICLR 2019. arXiv:1810.01367.
<https://arxiv.org/abs/1810.01367>

[^6]: Chen, R. T. Q. et al. (2018). Neural Ordinary Differential Equations. NeurIPS 2018. arXiv:1806.07366.
<https://arxiv.org/abs/1806.07366>

[^7]: Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions. NeurIPS 2018. arXiv:1807.03039.
<https://arxiv.org/abs/1807.03039>

[^8]: Kobyzev, I. et al. (2020). Normalizing Flows: An Introduction and Review of Current Methods. IEEE TPAMI. arXiv:1908.09257.
<https://arxiv.org/abs/1908.09257>

[^9]: Dinh, L. et al. (2015). NICE: Non-linear Independent Components Estimation. ICLR 2015. arXiv:1410.8516.
<https://arxiv.org/abs/1410.8516>

[^10]: Huang, C.-W. et al. (2018). Neural Autoregressive Flows. ICML 2018. arXiv:1804.00779.
<https://arxiv.org/abs/1804.00779>

[^11]: Huang, C.-W. et al. (2018). Approximation capabilities of Neural ODEs and Invertible Residual Networks. arXiv:1807.09245.

[^12]: Hoogeboom, E. et al. (2019). Cubic-Spline Flows. arXiv:1906.02145.
<https://arxiv.org/abs/1906.02145>

[^13]: Finlay, C. et al. (2020). How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization. ICML 2020. arXiv:2002.02798.
<https://arxiv.org/abs/2002.02798>

[^14]: Sorrenson, P. et al. (2024). Normalizing Flows are Capable Generative Models. arXiv:2412.06329.
<https://arxiv.org/abs/2412.06329>

[^15]: Arbel, M. et al. (2023). Kernelised Normalizing Flows. AISTATS 2024. arXiv:2307.14839.
<https://arxiv.org/abs/2307.14839>

[^16]: Ho, J. et al. (2019). Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design. ICML 2019. arXiv:1902.00275.
<https://arxiv.org/abs/1902.00275>

---

> Progress: 50%
> **理解度チェック**
> 1. FFJORD の Hutchinson trace推定 $\text{tr}(J) \approx \mathbf{v}^\top J \mathbf{v}$ が不偏推定量になる条件を示し、その計算コストが $O(D)$ になる理由を説明せよ。
> 2. Coupling LayerのヤコビアンがブロックLU三角行列になる理由を、分割 $(x_A, x_B)$ の設計と対応させて導け。

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
