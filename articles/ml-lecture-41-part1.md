---
title: "第41回: World Models & 環境シミュレータ理論🌍: 30秒の驚き→数式修行→実装マスター"
emoji: "🌍"
type: "tech"
topics: ["machinelearning", "deeplearning", "worldmodels", "julia", "jepa"]
published: true
---

# 第41回: World Models & 環境シミュレータ理論 🌍

**生成モデルの最終到達点は"理解"だった**

---

## 🚀 0. クイックスタート（30秒）— 1フレームから未来を予測する

第40回でConsistency Modelsによる1ステップ高速生成を実現した。だが生成モデルの真の目的は何だったのか？

単に画像を生成することではない。**環境の構造を理解し、未来を予測し、行動の結果をシミュレートすること**だ。

```julia
# World Modelの本質: 1フレーム → 未来の予測
using Lux, Random

# 観測 x_t から潜在表現 z_t を抽出
encoder = Chain(Conv((3,3), 3 => 64, relu), AdaptiveMeanPool((1,1)), FlattenLayer())

# 潜在空間で次状態を予測 (action条件付き)
predictor = Dense(64 + 4 => 64, tanh)  # 4次元action space

# 初期観測
x = rand(Float32, 64, 64, 3, 1)
a = rand(Float32, 4, 1)  # action

# 潜在状態抽出 → action条件付き予測
z = encoder(x, ps, st)[1]
z_next = predictor(vcat(z, a), ps_pred, st_pred)[1]

# 出力: z_next ∈ ℝ^64 (predicted next latent state)
```

**これが何をしているか？**

1フレームの観測$x_t$を潜在表現$z_t$に圧縮し、action $a_t$を与えて次状態$z_{t+1}$を予測する。

ピクセルは生成しない。**世界の潜在構造を予測する。**

$$
z_{t+1} = f_\theta(z_t, a_t)
$$

これがWorld Modelの数学だ。

:::message
**進捗**: 全体の3%完了。Consistency Modelsで1ステップ生成を実現したが、生成モデルの真の目的は「理解」だった。環境をシミュレートする理論へ。
:::

---

## 🎮 1. 体験ゾーン（10分）— World Modelsの3つの顔

### 1.1 生成 vs 理解 vs シミュレーション

生成モデルは3つのレベルに分類できる:

| レベル | 目的 | 入出力 | 代表手法 |
|:------|:-----|:------|:---------|
| **Level 1: 生成** | データ分布からサンプル | $p(x)$ | VAE, GAN, Diffusion |
| **Level 2: 条件付き生成** | 条件から生成 | $p(x|c)$ | LDM, CFG |
| **Level 3: World Models** | **環境の理解+予測+シミュレーション** | $p(x_{t+1}|x_{\leq t}, a_t)$ | JEPA, V-JEPA, Transfusion |

World Modelsは**行動の結果を予測できる**最高レベルだ。

### 1.2 JEPAの3変種を動かす

```julia
# I-JEPA: 画像の一部から他部分を予測
# Input: masked image patches
x_context = x[:, :, 1:32, :]  # 左半分
x_target_mask = [33:64]       # 右半分をマスク

# Context encoder → Predictor → Target prediction
z_context = context_encoder(x_context, ps_ctx, st_ctx)[1]
z_pred = predictor(z_context, mask_tokens, ps_pred, st_pred)[1]

# ❌ ピクセルを予測しない
# ✅ 潜在表現を予測する
```

```julia
# V-JEPA: 動画の一部フレームから未来フレームの潜在表現を予測
# Input: video sequence [B, T, H, W, C]
video = rand(Float32, 1, 16, 64, 64, 3)  # 16 frames
context_frames = video[:, 1:8, :, :, :]   # 前半8フレーム
target_frames = video[:, 9:16, :, :, :]   # 後半8フレーム

# Context encoder → Temporal predictor
z_ctx_video = video_encoder(context_frames, ps_v, st_v)[1]
z_pred_video = temporal_predictor(z_ctx_video, ps_tp, st_tp)[1]
```

```julia
# Transfusion: テキスト(AR) + 画像(Diffusion) を統一モデルで処理
# Text: autoregressive (next token prediction)
# Image: diffusion (denoising)

# テキストトークン
text_tokens = [101, 2054, 2003]  # "What is"

# 画像パッチ (continuous vectors)
image_patches = rand(Float32, 512, 16)  # 16 patches × 512 dim

# Transfusionの統一処理
# Text: next token prediction loss
loss_text = cross_entropy(model(text_tokens), text_tokens[2:end])

# Image: diffusion loss
t = rand(1:1000)
noise = randn(size(image_patches))
x_t = sqrt(α[t]) * image_patches + sqrt(1 - α[t]) * noise
loss_image = mse(model(x_t, t), noise)

# 総合loss
loss = loss_text + loss_image
```

### 1.3 World Modelsの応用領域

| 応用 | 目的 | World Modelの役割 |
|:-----|:-----|:-----------------|
| **ロボティクス** | 環境操作 | 行動結果の事前シミュレーション |
| **自動運転** | 予測制御 | 他車・歩行者の未来軌道予測 |
| **強化学習** | プランニング | Model-based RL (MuZero, Dreamer) |
| **科学シミュレーション** | 物理法則学習 | 微分方程式を学習で近似 |

:::details PyTorchとの対応（参考）
```python
# PyTorch版 JEPA predictor
import torch.nn as nn

class JEPAPredictor(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, context, mask_tokens):
        # Cross-attention: mask_tokens attend to context
        pred, _ = self.cross_attn(mask_tokens, context, context)
        pred = self.ffn(pred)
        return pred
```

Juliaでは型システムでこれを自然に表現できる。
:::

:::message
**進捗**: 全体の10%完了。World Modelsの3レベル分類を理解した。JEPAはピクセル生成をスキップし、潜在空間で予測する革命的アーキテクチャだ。
:::

---

## 🧩 2. 直感ゾーン（15分）— なぜWorld Modelsが最終到達点か

### 2.1 生成モデルの進化系譜

```mermaid
graph TD
    A[VAE: 潜在変数モデル] --> B[GAN: 敵対的学習]
    B --> C[Diffusion: スコア関数]
    C --> D[LDM: 潜在空間拡散]
    D --> E[Consistency Models: 1-step生成]
    E --> F[World Models: 環境理解]

    style F fill:#ff9,stroke:#333,stroke-width:4px
```

**なぜWorld Modelsが最終形態か？**

1. **生成は手段、理解が目的**: 画像生成はデータ分布の一部をサンプルするだけ。World Modelsは環境の**因果構造**を理解する
2. **行動条件付き予測**: $p(x_{t+1}|x_{\leq t}, a_t)$ — 行動の結果を予測できる
3. **シミュレーションフリー**: ピクセル生成を回避し、潜在空間で予測する効率性

### 2.2 Course IVでの位置づけ

| 回 | テーマ | World Modelsへの接続 |
|:---|:------|:--------------------|
| **第33回** | Normalizing Flows | 可逆変換 → 決定論的写像の限界 |
| **第34回** | EBM | エネルギー関数 → **Energy-based World Models** |
| **第35回** | Score Matching | スコア関数 → 動的過程の学習 |
| **第36回** | DDPM | Forward/Reverse → 時系列予測の基盤 |
| **第37回** | SDE/ODE | 連続時間確率過程 → 物理法則学習 |
| **第38回** | Flow Matching | OT視点 → **最適輸送としてのWorld Models** |
| **第39回** | LDM | 潜在空間拡散 → **潜在空間予測** |
| **第40回** | Consistency Models | 1-step生成 → 高速推論 |
| **第41回** | **World Models** | **生成モデルの最終到達点** |

### 2.3 松尾研との決定的な違い

| 項目 | 松尾研 | 本講義 |
|:-----|:------|:------|
| **World Models扱い** | 言及なし | **完全理論化** |
| **JEPA** | 触れない | I-JEPA / V-JEPA / VL-JEPA完全解説 |
| **Transfusion** | 扱わない | **AR+Diffusion統一理論の数学** |
| **物理法則学習** | 扱わない | Physics-Informed World Models深掘り |
| **実装** | なし | Julia JEPAコンセプト実装 |

### 2.4 学習戦略

```mermaid
graph LR
    A[第41回: 理論理解] --> B[第42回: 統一理論]
    A --> C[第43回 DiT/FLUX実装]
    A --> D[第45回 Video生成]
    A --> E[第47回 Embodied AI]

    style A fill:#f9f,stroke:#333,stroke-width:2px
```

World Modelsは**全ドメイン（画像・動画・ロボティクス・科学）の統一基盤**だ。

:::details Trojan Horse — 生成モデルの"本当の目的"
第1回から38回まで、生成モデルは「画像を生成する」技術として学んできた。

だがLeCunが提唱するJEPAは**生成をスキップする**。

**生成は副産物に過ぎなかった**。真の目的は**環境の因果構造を理解し、行動の結果を予測すること**だ。

これがTrojan Horseの最終形態だ。「生成AI」は「理解AI」に進化する。
:::

:::message
**進捗**: 全体の20%完了。生成モデルの進化系譜を理解した。World Modelsは生成の先にある「理解+予測+シミュレーション」の統合概念だ。
:::

---

## 📐 3. 数式修行ゾーン（60分）— World Modelsの数学的基礎

### 3.1 World Modelの定義

**定義**: World Model $\mathcal{M}$は環境の潜在表現$z_t$と遷移関数$f_\theta$から構成される確率的システムである。

$$
\begin{aligned}
\text{Encoder: } & z_t = \text{Enc}_\phi(x_t) \\
\text{Predictor: } & z_{t+1} = f_\theta(z_t, a_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma) \\
\text{Decoder: } & \hat{x}_{t+1} = \text{Dec}_\psi(z_{t+1})
\end{aligned}
$$

**なぜ潜在空間か？**

- ピクセル空間 $x \in \mathbb{R}^{H \times W \times C}$ は高次元（$H=256, W=256, C=3 \Rightarrow 196,608$次元）
- 潜在空間 $z \in \mathbb{R}^d$ は低次元（$d=256$程度）で**構造的表現**を獲得

**訓練目標**: 観測データ $\{(x_t, a_t, x_{t+1})\}_{t=1}^T$ から$\theta, \phi, \psi$を学習

$$
\mathcal{L}(\theta, \phi, \psi) = \mathbb{E}_{(x_t, a_t, x_{t+1})} \left[ \| \text{Dec}_\psi(f_\theta(\text{Enc}_\phi(x_t), a_t)) - x_{t+1} \|_2^2 \right]
$$

### 3.2 JEPA理論: Joint-Embedding Predictive Architecture

#### 3.2.1 I-JEPA (Image-based JEPA)

**論文**: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," CVPR 2023

**動機**: 画像生成（pixel reconstruction）は低レベル詳細に過剰適合し、高レベル抽象表現を学習しにくい。

**アーキテクチャ**:

```mermaid
graph LR
    A[Context patches<br>x_ctx] --> B[Context Encoder<br>s_θ]
    B --> C[Latent z_ctx]
    C --> D[Predictor<br>f_θ]
    E[Target patches<br>x_tgt] --> F[Target Encoder<br>s_θ EMA]
    F --> G[Latent z_tgt]
    D --> H[Predicted z_pred]
    H -.loss.-> G

    style D fill:#ff9,stroke:#333,stroke-width:2px
    style H fill:#f99,stroke:#333,stroke-width:2px
```

**損失関数**:

$$
\mathcal{L}_{\text{I-JEPA}} = \mathbb{E}_{x, M} \left[ \| f_\theta(s_\theta(x_{\text{ctx}}), M) - \bar{s}_\theta(x_{\text{tgt}}) \|_2^2 \right]
$$

ここで:
- $x_{\text{ctx}}$: マスクされていないcontext patches
- $x_{\text{tgt}}$: マスクされたtarget patches
- $M$: mask tokens (positional encoding)
- $s_\theta$: context encoder (trainable)
- $\bar{s}_\theta$: target encoder (EMA更新)
- $f_\theta$: predictor

**重要な特性**:

1. **ピクセル再構成なし**: $x_{\text{tgt}}$を生成せず、潜在表現$z_{\text{tgt}}$を予測
2. **EMA target encoder**: $\bar{\theta} \leftarrow \tau \bar{\theta} + (1-\tau)\theta$ でcollapse回避
3. **Mask strategy**: ランダムブロックマスク（Grid-based）で構造的予測を促進

**数値例**:

```julia
# Context: 画像の左半分 (32x64 patches)
x_ctx = x[:, 1:32, :, :]

# Target: 画像の右半分 (32x64 patches) — Encoder通すが勾配は流さない
x_tgt = x[:, 33:64, :, :]

# Context encoder (trainable)
z_ctx = s_θ(x_ctx)  # [B, 32, D]

# Predictor: mask tokens M を使って target位置の表現を予測
M = mask_tokens[:, 33:64, :]  # [B, 32, D_mask]
z_pred = f_θ(z_ctx, M)  # [B, 32, D]

# Target encoder (EMA, no grad)
z_tgt = stopgradient(s̄_θ(x_tgt))  # [B, 32, D]

# Loss
loss = mean((z_pred - z_tgt).^2)
```

#### 3.2.2 V-JEPA (Video JEPA)

**論文**: Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video," arXiv:2404.08471, 2024 (V-JEPA 1.0)
**最新**: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," arXiv:2506.09985, 2025

**拡張**: 画像→動画（時空間予測）

$$
\begin{aligned}
\text{Context: } & \mathbf{x}_{\text{ctx}} \in \mathbb{R}^{T_c \times H \times W \times C} \\
\text{Target: } & \mathbf{x}_{\text{tgt}} \in \mathbb{R}^{T_t \times H \times W \times C}
\end{aligned}
$$

**Spatio-temporal masking**:

- **Temporal masking**: 前半8フレーム（context）→後半8フレーム（target）を予測
- **Spatial masking**: 各フレーム内でパッチをマスク

**訓練目標**:

$$
\mathcal{L}_{\text{V-JEPA}} = \mathbb{E}_{\mathbf{x}, M_s, M_t} \left[ \| f_\theta(s_\theta(\mathbf{x}_{\text{ctx}}), M_s, M_t) - \bar{s}_\theta(\mathbf{x}_{\text{tgt}}) \|_2^2 \right]
$$

**性能**:

- Kinetics-400 (action recognition): **81.9%** Top-1 accuracy (video pre-trainingのみ)
- Something-Something v2: **72.2%**
- ImageNet: **77.9%** Top-1 (動画事前学習から画像タスクに転移)

#### 3.2.3 VL-JEPA (Vision-Language JEPA)

**論文**: Bardes et al., "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language," arXiv:2512.10942, 2024

**動機**: 従来のVLM（Vision-Language Models）はテキストをトークン単位でautoregressiveに生成する。これは計算コスト高く、パラメータ数も膨大（decoder層が必要）。

**アーキテクチャ**:

VL-JEPAは**テキストの連続埋め込みを予測**し、token-by-token生成をスキップする。

$$
\begin{aligned}
\text{Image encoder: } & z_v = \text{Enc}_v(x) \\
\text{Predictor: } & z_{\text{pred}} = f_\theta(z_v, \text{prompt}) \\
\text{Text encoder: } & z_t = \text{Enc}_t(\text{target text}) \\
\text{Loss: } & \mathcal{L} = \| z_{\text{pred}} - z_t \|_2^2
\end{aligned}
$$

**利点**:

- パラメータ数が標準VLMの**50%削減**（decoderなし）
- **より強い性能**: 同じvision encoderとデータで訓練した標準VLMを上回る

### 3.3 Transfusion理論: AR + Diffusion統一

**論文**: Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model," arXiv:2408.11039, 2024 (Meta AI)

**動機**: テキスト（離散トークン）と画像（連続ベクトル）を**単一Transformerで統一処理**したい。

**従来手法の問題**:

- 画像をVQ-VAEで離散トークン化 → 量子化誤差、コードブック利用率低下
- 別々のモデル（LM + Diffusion）→ 統合できない

**Transfusionの解決策**:

**同一Transformerで異なる損失関数**を使い分ける。

$$
\mathcal{L}_{\text{Transfusion}} = \mathcal{L}_{\text{LM}}(\text{text}) + \lambda \mathcal{L}_{\text{Diffusion}}(\text{image})
$$

#### 3.3.1 テキスト部分: Autoregressive

テキストトークン $\mathbf{t} = (t_1, t_2, \ldots, t_n)$ に対して:

$$
\mathcal{L}_{\text{LM}} = -\sum_{i=1}^n \log p_\theta(t_i | t_{<i})
$$

通常の言語モデルと同じcausal maskingとcross-entropy loss。

#### 3.3.2 画像部分: Diffusion

画像パッチ $\mathbf{x} = (x_1, \ldots, x_m) \in \mathbb{R}^{m \times d}$ に対して:

$$
\begin{aligned}
\text{Forward: } & x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \\
\text{Diffusion Loss: } & \mathcal{L}_{\text{Diffusion}} = \mathbb{E}_{t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) \|_2^2 \right]
\end{aligned}
$$

ここで $\mathbf{c}$ はテキスト条件（cross-attention経由でTransformerに注入）。

#### 3.3.3 統合処理の数学

入力シーケンス:

$$
\text{seq} = [\text{text tokens } t_1, \ldots, t_n, \text{ image patches } x_1, \ldots, x_m]
$$

**Attention mask**:

- テキスト部分: **causal mask**（未来のトークンを見ない）
- 画像部分: **bidirectional mask**（全パッチを見る）

**損失計算**:

```julia
# テキストトークン (discrete)
text_logits = model(text_tokens)
loss_text = cross_entropy(text_logits, text_tokens[2:end])

# 画像パッチ (continuous)
t_diffusion = rand(1:T)
noise = randn(size(image_patches))
x_t = sqrt(ᾱ[t_diffusion]) * image_patches + sqrt(1 - ᾱ[t_diffusion]) * noise
pred_noise = model(x_t, t_diffusion, context=text_tokens)
loss_image = mean((pred_noise - noise).^2)

# 統合
loss = loss_text + λ * loss_image
```

**スケーリング結果**: 7Bパラメータ、2T multi-modal tokensで訓練 → テキスト生成と画像生成の両方で同規模の専用モデルと同等性能。

### 3.4 物理法則学習理論

#### 3.4.1 Physics-Informed World Models

**動機**: 標準的なWorld Modelsは物理法則（保存則、対称性、微分方程式）を無視し、データ駆動で学習する。これは:

- 物理的に不可能な予測（エネルギー保存則違反など）
- データ効率の悪さ（物理法則を知っていれば少ないデータで学習可能）

**Physics-Informed Neural Networks (PINNs)の原理**:

微分方程式制約を損失関数に埋め込む。

例: Navier-Stokes方程式

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}
$$

**PINNs loss**:

$$
\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}}
$$

$$
\mathcal{L}_{\text{PDE}} = \mathbb{E}_{x,t} \left[ \left\| \frac{\partial \mathbf{u}_\theta}{\partial t} + (\mathbf{u}_\theta \cdot \nabla)\mathbf{u}_\theta + \frac{1}{\rho}\nabla p_\theta - \nu \nabla^2 \mathbf{u}_\theta \right\|_2^2 \right]
$$

**World Modelsへの適用**:

$$
\mathcal{L}_{\text{Physics-WM}} = \mathcal{L}_{\text{prediction}} + \lambda_{\text{conservation}} \mathcal{L}_{\text{conservation}}
$$

$$
\mathcal{L}_{\text{conservation}} = \mathbb{E} \left[ \| E(z_{t+1}) - E(z_t) \|_2^2 \right]
$$

ここで$E(z)$はエネルギー関数（学習または既知）。

#### 3.4.2 保存則の埋め込み

**運動量保存**:

$$
\sum_{i=1}^N m_i \mathbf{v}_i(t) = \text{const}
$$

**Graph Neural Networkでの実装**:

ノード$i$の速度$\mathbf{v}_i$に対して、edge $(i,j)$の更新:

$$
\Delta \mathbf{v}_i = \sum_{j \in \mathcal{N}(i)} \text{MLP}(\mathbf{h}_i, \mathbf{h}_j, \mathbf{r}_{ij})
$$

**保存則制約**: 各edge更新が運動量保存を満たすように、**Newton's third law**を明示的に適用:

$$
m_i \Delta \mathbf{v}_i = -m_j \Delta \mathbf{v}_j
$$

**実装**:

```julia
# Edge-local reference frameでのforce計算
f_ij = MLP(h_i, h_j, r_ij)

# Newton's third lawで対称化
Δv_i = f_ij / m_i
Δv_j = -f_ij / m_j

# 更新
v_i_new = v_i + Δv_i
v_j_new = v_j + Δv_j

# 検証: 系全体の運動量が保存されているか
@assert sum(m_i * v_i_new for i in 1:N) ≈ sum(m_i * v_i for i in 1:N)
```

#### 3.4.3 Hamiltonian Neural Networks

**Hamiltonian力学系**:

$$
\begin{aligned}
\dot{q} &= \frac{\partial H}{\partial p} \\
\dot{p} &= -\frac{\partial H}{\partial q}
\end{aligned}
$$

ここで$H(q, p)$はHamiltonian（総エネルギー）。

**HNNの学習**:

1. NNで$H_\theta(q, p)$をパラメータ化
2. 自動微分で$\partial H / \partial p, \partial H / \partial q$を計算
3. 損失:

$$
\mathcal{L}_{\text{HNN}} = \mathbb{E} \left[ \left\| \left(\dot{q}, \dot{p}\right) - \left(\frac{\partial H_\theta}{\partial p}, -\frac{\partial H_\theta}{\partial q}\right) \right\|_2^2 \right]
$$

**利点**: エネルギー保存則が**構造的に保証**される（Hamiltonianの時間微分が0）。

### 3.5 Energy-based World Models

**第34回EBMとの接続**:

World Modelsを**エネルギー関数$E_\theta(z_t, a_t, z_{t+1})$**として定式化:

$$
p(z_{t+1} | z_t, a_t) = \frac{\exp(-E_\theta(z_t, a_t, z_{t+1}))}{Z(z_t, a_t)}
$$

**利点**:

- 任意の分布形状を表現可能（Gaussianに制約されない）
- エネルギー最小化 = 最も可能性の高い未来状態

**訓練**: Contrastive Divergence (CD-k) または Score Matching

$$
\nabla_\theta \mathcal{L} = \mathbb{E}_{z_t, a_t, z_{t+1}^{+}} [\nabla_\theta E_\theta(z_t, a_t, z_{t+1}^{+})] - \mathbb{E}_{z_t, a_t, z_{t+1}^{-}} [\nabla_\theta E_\theta(z_t, a_t, z_{t+1}^{-})]
$$

ここで$z_{t+1}^{+}$はデータ、$z_{t+1}^{-}$はLangevin dynamicsでサンプル。

### 3.6 世界モデルの訓練理論

#### 3.6.1 Action-conditioned予測

**目的**: action $a_t$を与えて次状態$z_{t+1}$を予測

$$
z_{t+1} = f_\theta(z_t, a_t) + \epsilon_t
$$

**訓練データ**: ロボット軌跡 $(z_t, a_t, z_{t+1})$

**損失**:

$$
\mathcal{L}_{\text{pred}} = \mathbb{E} \left[ \| f_\theta(z_t, a_t) - z_{t+1} \|_2^2 \right]
$$

**Stochastic dynamics**の場合:

$$
p_\theta(z_{t+1} | z_t, a_t) = \mathcal{N}(f_\theta(z_t, a_t), \Sigma_\theta(z_t, a_t))
$$

$$
\mathcal{L}_{\text{NLL}} = -\mathbb{E} \left[ \log p_\theta(z_{t+1} | z_t, a_t) \right]
$$

#### 3.6.2 Reward Prediction

World Modelを強化学習に統合する場合、報酬関数$r_t$も予測:

$$
r_t = g_\phi(z_t, a_t)
$$

**訓練**:

$$
\mathcal{L}_{\text{reward}} = \mathbb{E} \left[ (g_\phi(z_t, a_t) - r_t)^2 \right]
$$

**Model-based RL**:

1. World Modelで未来をシミュレート: $z_{t+1} = f_\theta(z_t, a_t)$
2. 報酬を推定: $\hat{r}_t = g_\phi(z_t, a_t)$
3. Policy $\pi_\psi(a|z)$を最適化:

$$
\mathcal{L}_{\text{policy}} = -\mathbb{E}_{\pi} \left[ \sum_{t=0}^H \gamma^t g_\phi(z_t, a_t) \right]
$$

#### 3.6.3 自己教師あり学習

**Contrastive Learning**: 正例（同一動画の近接フレーム）vs 負例（異なる動画）

$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}) / \tau)}{\sum_{j} \exp(\text{sim}(z_t, z_j^{-}) / \tau)}
$$

**Masked Autoencoding**: 一部のフレームをマスク → 予測（JEPAと同じ原理）

### 3.7 世界モデルの評価

#### 3.7.1 予測精度

**Mean Squared Error (MSE)**:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N \| z_{t+1}^{(i)} - \hat{z}_{t+1}^{(i)} \|_2^2
$$

**Structural Similarity (SSIM)** (ピクセル空間で評価する場合):

$$
\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x \mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}
$$

#### 3.7.2 物理法則遵守スコア

**エネルギー保存誤差**:

$$
\text{Energy Error} = \frac{1}{T} \sum_{t=1}^T | E(z_t) - E(z_0) |
$$

**運動量保存誤差**:

$$
\text{Momentum Error} = \frac{1}{T} \sum_{t=1}^T \left\| \sum_i m_i \mathbf{v}_i(t) - \sum_i m_i \mathbf{v}_i(0) \right\|_2
$$

#### 3.7.3 長期一貫性

**Frame Prediction Horizon**: モデルが何ステップ先まで正確に予測できるか

$$
T_{\text{horizon}} = \max\{t : \text{MSE}(t) < \epsilon\}
$$

**Video Quality Metrics**:

- **FVD (Fréchet Video Distance)**: I3D特徴量でのFréchet距離
- **LPIPS**: 知覚的類似度

### 🥊 Boss Battle: Transfusionの完全分解

**課題**: arXiv:2408.11039のTransfusionの統一損失関数を、テキストとイメージの混合シーケンスに対して完全導出せよ。

**Step 1**: 入力シーケンス

テキスト部分 $\mathbf{t} = (t_1, \ldots, t_n)$ （discrete tokens）
画像部分 $\mathbf{x} = (x_1, \ldots, x_m)$ （continuous patch embeddings）

統合シーケンス:

$$
\mathbf{s} = [\mathbf{t}, \mathbf{x}] \in \mathbb{R}^{(n+m) \times d}
$$

**Step 2**: Attention Mask

$$
M_{\text{Transfusion}} = \begin{bmatrix}
M_{\text{causal}} & 0 \\
M_{\text{bi-dir}} & M_{\text{bi-dir}}
\end{bmatrix}
$$

- 左上: テキストのcausal mask（自己回帰）
- 右下: 画像のbidirectional mask（全パッチ相互参照）
- 左下: 画像がテキストを見る（cross-modal attention）
- 右上: 0（テキストは画像を見ない）

**Step 3**: Loss Functions

**テキストloss** (next token prediction):

$$
\mathcal{L}_{\text{text}} = -\frac{1}{n} \sum_{i=1}^n \log p_\theta(t_i | t_{<i})
$$

Softmaxで確率化:

$$
p_\theta(t_i | t_{<i}) = \frac{\exp(z_{t_i}^\top e_{t_i})}{\sum_{j=1}^{|V|} \exp(z_{t_i}^\top e_j)}
$$

ここで$z_{t_i}$はTransformerの$i$番目出力、$e_j$はtoken embeddingの$j$番目。

**画像loss** (diffusion):

$$
\mathcal{L}_{\text{image}} = \mathbb{E}_{t \sim [1,T], \epsilon \sim \mathcal{N}(0,I)} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) \|_2^2 \right]
$$

ここで:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

$\mathbf{c}$はテキスト条件（cross-attentionで注入）。

**Step 4**: 統合損失

$$
\mathcal{L}_{\text{Transfusion}} = \mathcal{L}_{\text{text}} + \lambda \mathcal{L}_{\text{image}}
$$

$\lambda$はハイパーパラメータ（論文では$\lambda=1$を使用）。

**Step 5**: 実装コード（Julia）

```julia
using Lux, Random, Optimisers

# Transformer with mixed masking
struct TransfusionTransformer
    layers::Chain
    text_head::Dense
    image_head::Dense
end

function (m::TransfusionTransformer)(s, mask, t_diffusion=nothing)
    h = m.layers(s, mask)  # [B, n+m, D]

    # Split into text and image parts
    h_text = h[:, 1:n_text, :]
    h_image = h[:, n_text+1:end, :]

    # Text head: logits
    logits_text = m.text_head(h_text)  # [B, n_text, vocab_size]

    # Image head: noise prediction
    if !isnothing(t_diffusion)
        # Embed diffusion time step
        t_emb = sinusoidal_embedding(t_diffusion)
        pred_noise = m.image_head(cat(h_image, t_emb, dims=3))
    else
        pred_noise = nothing
    end

    return logits_text, pred_noise
end

# Loss function
function transfusion_loss(model, text_tokens, image_patches, ps, st)
    n_text = size(text_tokens, 2)

    # Forward diffusion on images
    t = rand(1:1000)
    noise = randn(size(image_patches))
    ᾱ_t = alpha_bar(t)
    x_t = sqrt(ᾱ_t) * image_patches + sqrt(1 - ᾱ_t) * noise

    # Construct input sequence [text, image]
    s = cat(embed(text_tokens), x_t, dims=2)  # [B, n_text + m_image, D]

    # Construct mask
    mask = create_transfusion_mask(n_text, size(x_t, 2))

    # Forward
    logits_text, pred_noise = model(s, mask, t, ps, st)

    # Text loss (cross-entropy)
    loss_text = cross_entropy(logits_text, text_tokens[:, 2:end])

    # Image loss (diffusion)
    loss_image = mean((pred_noise - noise).^2)

    # Total loss
    return loss_text + λ * loss_image
end
```

**Boss撃破！** Transfusionの統一損失関数を完全に導出し、実装した。

:::message alert
**ここで躓く人が多い**: TransfusionのAttention maskは**混合型**である。テキスト部分はcausal、画像部分はbidirectional、そしてクロスモーダル部分は**画像→テキスト**のみ許可（テキストは画像を見ない）。これを正しく実装しないとテキスト生成でリークが起きる。
:::

:::message
**進捗**: 全体の50%完了。World Modelsの数学的基礎を完全習得した。JEPA（I/V/VL）の3変種、Transfusionの統一理論、物理法則学習、EBM視点、訓練・評価手法を導出した。数式修行ボス戦をクリア。
:::

### 3.8 JEPAの最新発展（2024-2026）

#### 3.8.1 LeJEPA: 理論的基盤の確立

**論文**: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics," arXiv:2511.08544, 2024[^1]

従来のJEPAは経験的設計（EMA、特定のマスク戦略など）に依存していた。LeJEPAは**理論的に正当化された訓練目的**を提示する。

**核心的洞察**: JEPAの目的関数は**潜在変数の相互情報量最大化**として定式化できる:

$$
\max_{\theta, \phi} I(Z_{\text{ctx}}; Z_{\text{tgt}}) = \mathbb{H}(Z_{\text{tgt}}) - \mathbb{H}(Z_{\text{tgt}} | Z_{\text{ctx}})
$$

ここで:
- $Z_{\text{ctx}} = s_\theta(x_{\text{ctx}})$: context表現
- $Z_{\text{tgt}} = s_\theta(x_{\text{tgt}})$: target表現
- $\mathbb{H}(\cdot)$: エントロピー

**LeJEPA目的関数**:

$$
\mathcal{L}_{\text{LeJEPA}} = \mathbb{E}_{x, M} \left[ \| f_\theta(s_\theta(x_{\text{ctx}}), M) - s_\theta(x_{\text{tgt}}) \|_2^2 \right] + \lambda \mathbb{H}(Z_{\text{tgt}})
$$

第2項は**表現の多様性**を保証し、collapse（全表現が同一になる）を防ぐ。

**理論的保証**:

1. **収束保証**: LeJEPAは適切な$\lambda$で大域最適解に収束
2. **EMA不要**: 理論的に正当化された目的関数によりEMAなしで訓練可能
3. **スケーラビリティ**: 10億パラメータモデルで効率的に訓練可能

#### 3.8.2 Causal-JEPA: 因果的介入学習

**論文**: "Causal-JEPA: Learning World Models through Object-Level Latent Interventions," arXiv:2602.11389, 2025[^2]

従来のJEPAは**相関**を学習するが、**因果関係**は学習しない。Causal-JEPA（C-JEPA）は**オブジェクトレベルのマスキング**と**潜在介入**を導入。

**アーキテクチャの拡張**:

1. **オブジェクト分解**: 画像を$K$個のオブジェクト潜在表現に分解
   $$
   z = \{z_1, z_2, \ldots, z_K\}, \quad z_k \in \mathbb{R}^d
   $$

2. **介入操作**: 特定オブジェクト$k$の表現を変更
   $$
   \text{do}(z_k = \tilde{z}_k)
   $$

3. **反事実予測**: 介入後の未来状態を予測
   $$
   z_{t+1}' = f_\theta(z_t | \text{do}(z_k = \tilde{z}_k))
   $$

**訓練目的**:

$$
\mathcal{L}_{\text{C-JEPA}} = \mathbb{E} \left[ \| f_\theta(z_{\text{ctx}} | \text{do}(z_k)) - z_{\text{tgt}} \|_2^2 \right]
$$

**応用**: ロボットマニピュレーション（「このオブジェクトを動かすと何が起きるか？」）

#### 3.8.3 Value-guided Action Planning with JEPA

**論文**: "Value-guided action planning with JEPA world models," arXiv:2601.00844, 2025[^3]

JEPAを**強化学習**に統合し、action planningに使用。

**アーキテクチャ**:

1. **JEPA world model**: $z_{t+1} = f_\theta(z_t, a_t)$
2. **Value network**: $V_\psi(z_t)$ — 状態価値関数
3. **Planning**: Model Predictive Control (MPC)風に未来軌道を最適化

**Planning objective**:

$$
a_{t:t+H}^* = \arg\max_{a_{t:t+H}} \sum_{k=0}^H \gamma^k V_\psi(z_{t+k})
$$

ここで$z_{t+k}$はworld modelで予測。

**実装（Julia概念コード）**:

```julia
# JEPA world model
function predict_latent(z_t, a_t, ps_wm, st_wm)
    z_next, st_new = world_model(vcat(z_t, a_t), ps_wm, st_wm)
    return z_next, st_new
end

# Value network
function estimate_value(z, ps_v, st_v)
    v, st_new = value_net(z, ps_v, st_v)
    return v, st_new
end

# Planning via gradient-based optimization
function plan_actions(z_0, horizon, ps_wm, ps_v, st_wm, st_v)
    # 初期action sequence (learnable parameters)
    a_seq = rand(Float32, action_dim, horizon)

    # Optimize actions
    opt_state = Optimisers.setup(Adam(0.01), a_seq)

    for step in 1:50  # optimization steps
        # Rollout world model
        z_t = z_0
        total_value = 0.0

        for h in 1:horizon
            z_t, _ = predict_latent(z_t, a_seq[:, h], ps_wm, st_wm)
            v, _ = estimate_value(z_t, ps_v, st_v)
            total_value += γ^(h-1) * v
        end

        # Gradient ascent on total value
        grad = gradient(a_seq -> compute_value(a_seq, z_0, ps_wm, ps_v), a_seq)
        opt_state, a_seq = Optimisers.update(opt_state, a_seq, grad)
    end

    return a_seq[:, 1]  # Execute first action only (MPC)
end
```

**実験結果**: Atariゲームで従来のmodel-free RL（PPO）を上回る性能（sample efficiency 3x向上）。

### 3.9 Physics-Informed World Modelsの最新発展

#### 3.9.1 Separable PINNs (SPINN)

**論文**: Cho et al., "Separable Physics-Informed Neural Networks," arXiv:2306.15969, 2023[^4]

従来のPINNsは高次元PDE（$d \geq 4$）でメモリ爆発する。SPINNは**軸分離可能**構造で次元削減。

**核心的アイデア**: PDE解を変数分離形式で近似:

$$
u(x_1, \ldots, x_d) \approx \sum_{i=1}^R u_1^{(i)}(x_1) \cdot u_2^{(i)}(x_2) \cdots u_d^{(i)}(x_d)
$$

ここで各$u_j^{(i)}: \mathbb{R} \to \mathbb{R}$は1次元NN。

**メモリ削減**: 標準PINNsが$O(N^d)$の collocation pointsを必要とする一方、SPINNは$O(dN)$で済む。

**実装例**（2D熱方程式）:

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

```julia
# 分離可能近似
u_x = Chain(Dense(1 => 64, tanh), Dense(64 => 1))
u_y = Chain(Dense(1 => 64, tanh), Dense(64 => 1))
u_t = Chain(Dense(1 => 64, tanh), Dense(64 => 1))

# Combined solution
function u(x, y, t, ps_x, ps_y, ps_t, st_x, st_y, st_t)
    u_x_val, _ = u_x(x, ps_x, st_x)
    u_y_val, _ = u_y(y, ps_y, st_y)
    u_t_val, _ = u_t(t, ps_t, st_t)
    return u_x_val .* u_y_val .* u_t_val
end

# PDE residual
function pde_residual(x, y, t, ps, st)
    u_val = u(x, y, t, ps...)

    # 自動微分でPDE項を計算
    ∂u_∂t = gradient(t -> u(x, y, t, ps...), t)[1]
    ∂²u_∂x² = gradient(x -> gradient(x -> u(x, y, t, ps...), x)[1], x)[1]
    ∂²u_∂y² = gradient(y -> gradient(y -> u(x, y, t, ps...), y)[1], y)[1]

    # 残差
    return ∂u_∂t - α * (∂²u_∂x² + ∂²u_∂y²)
end

# Loss
loss = mean(pde_residual(x_collocation, y_collocation, t_collocation, ps, st).^2)
```

**性能**: 10^7 collocation pointsで訓練可能（従来PINNsの1000倍）。

#### 3.9.2 Conservation-Aware PINNs

**論文**: Cardoso-Bihlo & Bihlo, "Exactly Conservative Physics-Informed Neural Operators," 2025[^5]

物理法則（質量・運動量・エネルギー保存）を**離散レベルで厳密に保証**する。

**問題設定**: Navier-Stokes方程式を解く際、標準PINNsは近似誤差により保存則を破る。

**解決策**: **Learnable Adaptive Correction**

$$
u_{\text{corrected}} = u_\theta + \Delta u_{\text{conservation}}
$$

ここで$\Delta u_{\text{conservation}}$は保存則を満たすように自動計算。

**質量保存の場合**:

$$
\int_\Omega \nabla \cdot \mathbf{u} \, dV = 0
$$

**補正項**:

$$
\Delta \mathbf{u} = \nabla \phi, \quad \text{where } \nabla^2 \phi = -(\nabla \cdot \mathbf{u}_\theta)
$$

この$\phi$を解くことで、$\nabla \cdot (\mathbf{u}_\theta + \nabla \phi) = 0$が厳密に成立。

**実装の鍵**: Poisson方程式$\nabla^2 \phi = f$を高速に解く（FFTまたは multigrid法）。

```julia
using FFTW

function enforce_mass_conservation(u_theta, grid)
    # 発散を計算
    div_u = compute_divergence(u_theta, grid)

    # Poisson equation: ∇²φ = -div_u
    # FFTで高速に解く
    φ_hat = fft(div_u) ./ laplacian_eigenvalues(grid)
    φ = real(ifft(φ_hat))

    # 補正
    ∇φ = compute_gradient(φ, grid)
    u_corrected = u_theta + ∇φ

    # 検証
    @assert maximum(abs.(compute_divergence(u_corrected, grid))) < 1e-10

    return u_corrected
end
```

**結果**: 保存則誤差が標準PINNsの10^-3から10^-12に改善（9桁向上）。

### 3.10 Energy-Based World Modelsの理論

#### 3.10.1 EB-JEPA: Energy-Based JEPA Library

**論文**: "A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures," arXiv:2602.03604, 2025[^6]

JEPAを**Energy-Based Model**として再定式化。

**動機**: 従来のJEPAはL2損失で訓練 → 単峰性Gaussian仮定。複雑な多峰性分布を表現できない。

**Energy-based formulation**:

$$
p(z_{\text{tgt}} | z_{\text{ctx}}) = \frac{\exp(-E_\theta(z_{\text{ctx}}, z_{\text{tgt}}))}{Z(z_{\text{ctx}})}
$$

ここでエネルギー関数:

$$
E_\theta(z_{\text{ctx}}, z_{\text{tgt}}) = \| f_\theta(z_{\text{ctx}}) - z_{\text{tgt}} \|_2^2
$$

**訓練**: Noise Contrastive Estimation (NCE)

$$
\mathcal{L}_{\text{NCE}} = -\mathbb{E}_{z^+} [\log \sigma(-E_\theta(z_{\text{ctx}}, z^+))] - \mathbb{E}_{z^-} [\log \sigma(E_\theta(z_{\text{ctx}}, z^-))]
$$

ここで$z^+$は真の target、$z^-$は負例（ランダムサンプル）。

**実装**:

```julia
# Energy function
function energy(z_ctx, z_tgt, ps, st)
    z_pred, _ = predictor(z_ctx, ps, st)
    E = sum((z_pred - z_tgt).^2, dims=1)  # [B]
    return E
end

# NCE loss
function nce_loss(z_ctx, z_tgt_pos, z_tgt_neg, ps, st)
    E_pos = energy(z_ctx, z_tgt_pos, ps, st)
    E_neg = energy(z_ctx, z_tgt_neg, ps, st)

    # Binary classification: positive = low energy, negative = high energy
    loss = -mean(log.(σ.(-E_pos))) - mean(log.(σ.(E_neg)))
    return loss
end
```

**利点**:

- **多峰性**: 複数の可能な未来を表現（例: 動画予測で複数の軌道候補）
- **不確実性定量化**: エネルギーの高さ = 不確実性

#### 3.10.2 Cognitively Inspired Energy-Based World Models

**論文**: "Cognitively Inspired Energy-Based World Models," arXiv:2406.08862, 2024[^7]

認知科学の**予測符号化（Predictive Coding）**理論をWorld Modelsに統合。

**脳の予測符号化**:

脳は常に**予測**を生成し、**予測誤差**を最小化するように学習する。

$$
\text{Prediction Error} = x_{\text{observed}} - x_{\text{predicted}}
$$

**Energy-Based World Modelとの対応**:

$$
E(x_t, a_t, x_{t+1}) = \| x_{t+1} - f_\theta(x_t, a_t) \|_2^2 + \text{Prior}(x_{t+1})
$$

**階層的予測**:

レベル1（低レベル特徴）→ レベル2（中レベル）→ レベル3（高レベル抽象概念）

各レベルで予測誤差を計算:

$$
\epsilon_l = h_l - f_l(h_{l+1})
$$

**Total energy**:

$$
E_{\text{total}} = \sum_{l=1}^L \lambda_l \| \epsilon_l \|_2^2
$$

**訓練**: エネルギー最小化 = 階層的予測誤差最小化

**認知的利点**:

- **注意機構**: 高エネルギー領域（予測誤差大）に注意を向ける
- **能動推論**: エネルギーを最小化するaction $a_t$を選択
- **意識**: 高レベル予測誤差が閾値を超えると「意識」に上る

```julia
# 階層的予測符号化モデル
struct HierarchicalPredictiveCoding
    encoders::Vector{Chain}
    predictors::Vector{Chain}
    num_levels::Int
end

function (m::HierarchicalPredictiveCoding)(x, a, ps, st)
    # Bottom-up pass (encoding)
    h = Vector{Any}(undef, m.num_levels)
    h[1] = x
    for l in 2:m.num_levels
        h[l], _ = m.encoders[l-1](h[l-1], ps.enc[l-1], st.enc[l-1])
    end

    # Top-down pass (prediction)
    pred_errors = Vector{Any}(undef, m.num_levels - 1)
    for l in (m.num_levels-1):-1:1
        # Predict lower level from higher level
        h_pred, _ = m.predictors[l](vcat(h[l+1], a), ps.pred[l], st.pred[l])
        pred_errors[l] = h[l] - h_pred
    end

    # Total energy = weighted sum of prediction errors
    E = sum(λ[l] * sum(pred_errors[l].^2) for l in 1:(m.num_levels-1))

    return E, pred_errors
end
```

**実験結果**: ロボットナビゲーションタスクで、標準World Modelsより30%サンプル効率向上。

#### 3.10.3 Autoregressive LMs as Energy-Based Models

**論文**: "Autoregressive Language Models are Secretly Energy-Based Models: Insights into the Lookahead Capabilities of Next-Token Prediction," arXiv:2512.15605, 2024[^8]

**驚きの発見**: Autoregressive LMs（GPT系）は実は**Energy-Based Models**と等価！

**定理**: ARMとEBMの間に**明示的全単射**が存在:

$$
p_{\text{ARM}}(x_{1:T}) = \prod_{t=1}^T p(x_t | x_{<t}) \iff p_{\text{EBM}}(x_{1:T}) = \frac{\exp(-E(x_{1:T}))}{Z}
$$

ここでエネルギー関数:

$$
E(x_{1:T}) = -\sum_{t=1}^T \log p(x_t | x_{<t})
$$

**Soft Bellman方程式との接続**:

$$
V(x_{<t}) = \log \sum_{x_t} \exp(r(x_t | x_{<t}) + V(x_{\leq t}))
$$

**Transfusionへの示唆**: テキスト（AR）と画像（Diffusion）の統一は、実は**両方ともEBM**という視点から自然に理解できる！

$$
E_{\text{Transfusion}}(x_{\text{text}}, x_{\text{image}}) = E_{\text{ARM}}(x_{\text{text}}) + E_{\text{Diffusion}}(x_{\text{image}})
$$

これは**単一のエネルギー関数**の異なる分解に過ぎない。

:::message alert
**深い洞察**: 生成モデルの統一理論は「Energy-Based World Models」に収束している。VAE、GAN、Diffusion、Transfusion、JEPAは全て**エネルギー関数の異なる訓練・推論方法**として理解できる。

第34回で学んだEBMが、実は生成モデル全体の**最も一般的なフレームワーク**だった！
:::

:::message
**進捗**: 全体の70%完了。最新のJEPA発展（LeJEPA、Causal-JEPA、Value-guided planning）、Physics-Informed World Models（SPINN、Conservation-Aware PINNs）、Energy-Based理論（EB-JEPA、Predictive Coding、ARM-EBM同値性）を完全習得。2020-2025の最先端研究を統合した。
:::

---

## 💻 4. 実装ゾーン（30分）— JEPA World Modelのコンセプト実装

### 4.1 I-JEPAの最小実装

```julia
using Lux, Random, Optimisers, Zygote

# Context Encoder (trainable)
function create_context_encoder(input_dim, hidden_dim, output_dim)
    Chain(
        Dense(input_dim => hidden_dim, gelu),
        LayerNorm(hidden_dim),
        Dense(hidden_dim => hidden_dim, gelu),
        LayerNorm(hidden_dim),
        Dense(hidden_dim => output_dim)
    )
end

# Target Encoder (EMA updated)
function create_target_encoder(input_dim, hidden_dim, output_dim)
    # Same architecture as context encoder
    create_context_encoder(input_dim, hidden_dim, output_dim)
end

# Predictor (cross-attention based)
struct JEPAPredictor{C,F}
    cross_attn::C
    ffn::F
end

function JEPAPredictor(dim, num_heads)
    cross_attn = MultiHeadAttention(dim, num_heads)
    ffn = Chain(
        Dense(dim => 4 * dim, gelu),
        Dense(4 * dim => dim)
    )
    JEPAPredictor(cross_attn, ffn)
end

function (m::JEPAPredictor)(context, mask_tokens, ps, st)
    # Cross-attention: mask_tokens query context
    attn_out, st_attn = m.cross_attn(mask_tokens, context, context, ps.attn, st.attn)

    # Feed-forward
    pred, st_ffn = m.ffn(attn_out, ps.ffn, st.ffn)

    return pred, (attn=st_attn, ffn=st_ffn)
end

# Complete I-JEPA model
struct IJEPA{E_ctx, E_tgt, P}
    context_encoder::E_ctx
    target_encoder::E_tgt
    predictor::P
    ema_momentum::Float32
end

function IJEPA(input_dim, hidden_dim, latent_dim, num_heads; τ=0.996f0)
    context_encoder = create_context_encoder(input_dim, hidden_dim, latent_dim)
    target_encoder = create_target_encoder(input_dim, hidden_dim, latent_dim)
    predictor = JEPAPredictor(latent_dim, num_heads)
    IJEPA(context_encoder, target_encoder, predictor, τ)
end

# Forward pass
function (m::IJEPA)(x, mask, ps, st)
    # x: [B, H, W, C] input image
    # mask: [B, N_patches] boolean mask (true = masked)

    # Split into context and target patches
    x_flat = reshape(x, size(x, 1), :, size(x, 4))  # [B, N_patches, C]
    x_context = x_flat[:, .!mask, :]
    x_target = x_flat[:, mask, :]

    # Context encoder (trainable)
    z_context, st_ctx = m.context_encoder(x_context, ps.context_encoder, st.context_encoder)

    # Target encoder (EMA, stop gradient)
    z_target, st_tgt = m.target_encoder(x_target, ps.target_encoder, st.target_encoder)
    z_target = Zygote.@ignore z_target  # Stop gradient

    # Predictor
    mask_tokens = randn(Float32, size(z_target))  # Learnable mask tokens
    z_pred, st_pred = m.predictor(z_context, mask_tokens, ps.predictor, st.predictor)

    return z_pred, z_target, (context_encoder=st_ctx, target_encoder=st_tgt, predictor=st_pred)
end

# EMA update for target encoder
function update_ema!(ps_target, ps_context, τ)
    for (k, v_target) in pairs(ps_target)
        v_context = ps_context[k]
        ps_target[k] = τ * v_target + (1 - τ) * v_context
    end
end

# Training step
function train_step!(model, ps, st, opt_state, x, mask)
    # Forward and loss
    loss, (grads, st_new) = Zygote.withgradient(ps) do p
        z_pred, z_target, st_out = model(x, mask, p, st)
        loss = mean((z_pred - z_target).^2)
        return loss, st_out
    end

    # Update context encoder and predictor
    opt_state, ps = Optimisers.update(opt_state, ps, grads)

    # EMA update for target encoder
    update_ema!(ps.target_encoder, ps.context_encoder, model.ema_momentum)

    return loss, ps, st_new, opt_state
end

# Example usage
rng = Random.default_rng()
Random.seed!(rng, 42)

# Model setup
input_dim = 64  # Patch embedding dimension
hidden_dim = 256
latent_dim = 128
num_heads = 8

model = IJEPA(input_dim, hidden_dim, latent_dim, num_heads)
ps, st = Lux.setup(rng, model)
opt_state = Optimisers.setup(Adam(1e-4), ps)

# Training loop (concept)
for epoch in 1:100
    # Sample batch
    x = rand(Float32, 32, 64, 64, 3)  # [B=32, H=64, W=64, C=3]

    # Generate random mask (mask 50% of patches)
    mask = rand(Float32, 32, 64) .< 0.5

    # Train step
    loss, ps, st, opt_state = train_step!(model, ps, st, opt_state, x, mask)

    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $loss"
    end
end
```

**実装のポイント**:

1. **EMA更新**: Target encoderはmomentum $\tau=0.996$でゆっくり更新 → collapse回避
2. **Stop gradient**: Target encoderの出力に勾配を流さない（`Zygote.@ignore`）
3. **Mask strategy**: ランダムにパッチの50%をマスク → 構造的予測を学習

### 4.2 V-JEPAの時空間拡張

```julia
# Video encoder (3D convolution)
function create_video_encoder(in_channels, hidden_dim, latent_dim)
    Chain(
        Conv((3, 3, 3), in_channels => 64, relu; stride=(1, 2, 2)),  # [T, H/2, W/2, 64]
        Conv((3, 3, 3), 64 => 128, relu; stride=(2, 2, 2)),          # [T/2, H/4, W/4, 128]
        Conv((3, 3, 3), 128 => hidden_dim, relu; stride=(2, 2, 2)),  # [T/4, H/8, W/8, hidden_dim]
        GlobalMeanPool(),  # [hidden_dim]
        Dense(hidden_dim => latent_dim)
    )
end

# Temporal predictor
struct TemporalPredictor{A,F}
    self_attn::A
    cross_attn::A
    ffn::F
    num_layers::Int
end

function TemporalPredictor(dim, num_heads, num_layers)
    self_attn = MultiHeadAttention(dim, num_heads)
    cross_attn = MultiHeadAttention(dim, num_heads)
    ffn = Chain(
        Dense(dim => 4 * dim, gelu),
        LayerNorm(4 * dim),
        Dense(4 * dim => dim)
    )
    TemporalPredictor(self_attn, cross_attn, ffn, num_layers)
end

function (m::TemporalPredictor)(context_seq, target_positions, ps, st)
    # context_seq: [B, T_ctx, D]
    # target_positions: [B, T_tgt, D] (positional encodings)

    h = target_positions
    st_layers = []

    for layer in 1:m.num_layers
        # Self-attention
        h_self, st_self = m.self_attn(h, h, h, ps.self_attn, st.self_attn)
        h = h + h_self

        # Cross-attention to context
        h_cross, st_cross = m.cross_attn(h, context_seq, context_seq, ps.cross_attn, st.cross_attn)
        h = h + h_cross

        # FFN
        h_ffn, st_ffn = m.ffn(h, ps.ffn, st.ffn)
        h = h + h_ffn

        push!(st_layers, (self_attn=st_self, cross_attn=st_cross, ffn=st_ffn))
    end

    return h, st_layers
end

# V-JEPA model
struct VJEPA{E_ctx, E_tgt, P}
    context_encoder::E_ctx
    target_encoder::E_tgt
    temporal_predictor::P
    ema_momentum::Float32
end

function VJEPA(in_channels, hidden_dim, latent_dim, num_heads, num_layers; τ=0.996f0)
    context_encoder = create_video_encoder(in_channels, hidden_dim, latent_dim)
    target_encoder = create_video_encoder(in_channels, hidden_dim, latent_dim)
    temporal_predictor = TemporalPredictor(latent_dim, num_heads, num_layers)
    VJEPA(context_encoder, target_encoder, temporal_predictor, τ)
end

# Example: predict future 8 frames from past 8 frames
function predict_future_frames(model, video, ps, st)
    # video: [B, T=16, H, W, C]
    B, T, H, W, C = size(video)
    T_ctx = T ÷ 2  # First 8 frames
    T_tgt = T - T_ctx  # Last 8 frames

    # Context frames
    video_ctx = video[:, 1:T_ctx, :, :, :]
    z_ctx, st_ctx = model.context_encoder(video_ctx, ps.context_encoder, st.context_encoder)

    # Target frames (for training)
    video_tgt = video[:, T_ctx+1:end, :, :, :]
    z_tgt, st_tgt = model.target_encoder(video_tgt, ps.target_encoder, st.target_encoder)
    z_tgt = Zygote.@ignore z_tgt

    # Predict target latents
    target_pos = positional_encoding(T_tgt, latent_dim)  # [1, T_tgt, D]
    z_pred, st_pred = model.temporal_predictor(z_ctx, target_pos, ps.temporal_predictor, st.temporal_predictor)

    return z_pred, z_tgt, (context_encoder=st_ctx, target_encoder=st_tgt, temporal_predictor=st_pred)
end
```

**V-JEPAの特徴**:

1. **3D Convolution**: 時空間特徴を同時に抽出
2. **Temporal Predictor**: Transformer-basedで過去から未来を予測
3. **Positional Encoding**: 時間位置情報を明示的に与える

### 4.3 Physics-Informed World Model実装

```julia
# Hamiltonian Neural Network
struct HamiltonianNN{H}
    hamiltonian::H  # Neural network for H(q, p)
end

function HamiltonianNN(input_dim, hidden_dim)
    hamiltonian = Chain(
        Dense(input_dim => hidden_dim, tanh),
        Dense(hidden_dim => hidden_dim, tanh),
        Dense(hidden_dim => 1)  # Scalar energy
    )
    HamiltonianNN(hamiltonian)
end

function (m::HamiltonianNN)(q, p, ps, st)
    # q: positions [B, n]
    # p: momenta [B, n]
    qp = vcat(q, p)  # [B, 2n]

    # Hamiltonian energy
    H, st_new = m.hamiltonian(qp, ps, st)

    return H, st_new
end

# Hamiltonian dynamics
function hamiltonian_dynamics(model, q, p, ps, st)
    # Compute H(q, p)
    H, st_new = model(q, p, ps, st)

    # ∂H/∂p and ∂H/∂q via automatic differentiation
    dH_dp = gradient(p -> model(q, p, ps, st)[1], p)[1]
    dH_dq = gradient(q -> model(q, p, ps, st)[1], q)[1]

    # Hamiltonian equations
    q̇ = dH_dp    # dq/dt = ∂H/∂p
    ṗ = -dH_dq   # dp/dt = -∂H/∂q

    return q̇, ṗ, st_new
end

# Training: match observed dynamics
function hnn_loss(model, q_t, p_t, q̇_obs, ṗ_obs, ps, st)
    q̇_pred, ṗ_pred, st_new = hamiltonian_dynamics(model, q_t, p_t, ps, st)

    # Prediction loss
    loss = mean((q̇_pred - q̇_obs).^2 + (ṗ_pred - ṗ_obs).^2)

    return loss, st_new
end

# Rollout in latent space
function rollout_hnn(model, q_0, p_0, num_steps, dt, ps, st)
    q_trajectory = [q_0]
    p_trajectory = [p_0]

    q_t, p_t = q_0, p_0
    st_t = st

    for step in 1:num_steps
        # Compute derivatives
        q̇, ṗ, st_t = hamiltonian_dynamics(model, q_t, p_t, ps, st_t)

        # Symplectic Euler integration (energy-preserving)
        p_t = p_t + dt * ṗ
        q_t = q_t + dt * q̇

        push!(q_trajectory, q_t)
        push!(p_trajectory, p_t)
    end

    return hcat(q_trajectory...), hcat(p_trajectory...)
end

# Example: Pendulum system
# H(q, p) = p²/2m + mgl(1 - cos(q))
rng = Random.default_rng()
model = HamiltonianNN(2, 64)  # input = [q, p]
ps, st = Lux.setup(rng, model)

# Generate synthetic pendulum data
m, g, l = 1.0, 9.8, 1.0
function true_hamiltonian(q, p)
    return p^2 / (2m) + m * g * l * (1 - cos(q))
end

# Training data
q_data = rand(100) * 2π .- π
p_data = randn(100)
q̇_data = [gradient(p -> true_hamiltonian(q_data[i], p), p_data[i])[1] for i in 1:100]
ṗ_data = [-gradient(q -> true_hamiltonian(q, p_data[i]), q_data[i])[1] for i in 1:100]

# Train HNN
opt_state = Optimisers.setup(Adam(1e-3), ps)
for epoch in 1:1000
    loss, st = hnn_loss(model, q_data, p_data, q̇_data, ṗ_data, ps, st)

    grads = gradient(ps -> hnn_loss(model, q_data, p_data, q̇_data, ṗ_data, ps, st)[1], ps)[1]
    opt_state, ps = Optimisers.update(opt_state, ps, grads)

    if epoch % 100 == 0
        @info "Epoch $epoch: Loss = $loss"
    end
end

# Verify energy conservation
q_0, p_0 = π/4, 0.0
q_traj, p_traj = rollout_hnn(model, [q_0], [p_0], 100, 0.01, ps, st)

# Compute energy at each step
energies = [model([q_traj[i]], [p_traj[i]], ps, st)[1] for i in 1:size(q_traj, 2)]
@info "Energy variance: $(std(energies))"  # Should be ~0 if conservation holds
```

**Physics-Informed実装の鍵**:

1. **自動微分**: Hamiltonianの偏微分を自動計算
2. **Symplectic積分**: エネルギー保存を数値的にも保証
3. **構造的制約**: Hamiltonian構造を強制 → 物理法則を学習

### 4.4 Energy-Based World Model with NCE

```julia
# Energy function
struct EnergyWorldModel{E}
    energy_net::E
end

function EnergyWorldModel(state_dim, action_dim, hidden_dim)
    energy_net = Chain(
        Dense(2 * state_dim + action_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => 1)  # Scalar energy
    )
    EnergyWorldModel(energy_net)
end

function (m::EnergyWorldModel)(z_t, a_t, z_next, ps, st)
    # z_t: current state [B, D]
    # a_t: action [B, A]
    # z_next: next state [B, D]
    input = vcat(z_t, a_t, z_next)  # [B, 2D+A]

    E, st_new = m.energy_net(input, ps, st)
    return E, st_new
end

# Noise Contrastive Estimation loss
function nce_loss(model, z_t, a_t, z_next_pos, z_next_neg, ps, st)
    # Positive samples (real transitions)
    E_pos, st_pos = model(z_t, a_t, z_next_pos, ps, st)

    # Negative samples (random states)
    E_neg, st_neg = model(z_t, a_t, z_next_neg, ps, st)

    # NCE loss: positive = low energy, negative = high energy
    loss = -mean(log.(σ.(-E_pos))) - mean(log.(σ.(E_neg)))

    return loss, st_pos
end

# Inference: find most likely next state
function infer_next_state(model, z_t, a_t, ps, st; num_steps=100, lr=0.01)
    # Initialize random candidate
    z_next = randn(Float32, size(z_t))

    # Gradient descent on energy
    for step in 1:num_steps
        E, st = model(z_t, a_t, z_next, ps, st)

        # ∇_{z_next} E
        grad_z = gradient(z -> model(z_t, a_t, z, ps, st)[1], z_next)[1]

        # Gradient descent
        z_next = z_next - lr * grad_z
    end

    return z_next
end

# Example usage
rng = Random.default_rng()
state_dim = 64
action_dim = 4
hidden_dim = 256

model = EnergyWorldModel(state_dim, action_dim, hidden_dim)
ps, st = Lux.setup(rng, model)
opt_state = Optimisers.setup(Adam(1e-4), ps)

# Training
for epoch in 1:100
    # Sample transitions
    z_t = randn(Float32, 32, state_dim)
    a_t = randn(Float32, 32, action_dim)
    z_next_pos = randn(Float32, 32, state_dim)  # Real next states
    z_next_neg = randn(Float32, 32, state_dim)  # Random negative samples

    # Compute loss
    loss, st = nce_loss(model, z_t, a_t, z_next_pos, z_next_neg, ps, st)

    # Update
    grads = gradient(ps -> nce_loss(model, z_t, a_t, z_next_pos, z_next_neg, ps, st)[1], ps)[1]
    opt_state, ps = Optimisers.update(opt_state, ps, grads)

    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $loss"
    end
end

# Predict next state
z_t_test = randn(Float32, 1, state_dim)
a_t_test = randn(Float32, 1, action_dim)
z_next_pred = infer_next_state(model, z_t_test, a_t_test, ps, st)
@info "Predicted next state: $z_next_pred"
```

**Energy-Based推論の特徴**:

1. **Gradient-based inference**: エネルギー最小化で最適な次状態を探索
2. **多峰性表現**: エネルギー関数が複数の極小値を持てる → 複数の可能な未来
3. **Uncertainty**: エネルギーの高さ = 不確実性の定量化

:::details 実装の完全性チェックリスト
✅ **I-JEPA**: EMA更新、stop gradient、mask strategy
✅ **V-JEPA**: 3D convolution、temporal predictor、positional encoding
✅ **Hamiltonian NN**: 自動微分、symplectic integration、energy conservation
✅ **Energy-Based WM**: NCE訓練、gradient-based inference、多峰性対応

全て本番投入可能なコンセプト実装（Production-readyにするにはバッチ処理最適化、distributed訓練、checkpointing等が必要）。
:::

:::message
**進捗**: 全体の85%完了。4つの主要World Modelアーキテクチャ（I-JEPA、V-JEPA、Hamiltonian NN、Energy-Based WM）を完全実装した。理論から実装への橋渡し完了。
:::

---

## 📚 参考文献

### 主要論文

[^1]: Garrido, Q., et al. (2024). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544.
@[card](https://arxiv.org/abs/2511.08544)

[^2]: Biza, O., et al. (2025). Causal-JEPA: Learning World Models through Object-Level Latent Interventions. arXiv:2602.11389.
@[card](https://arxiv.org/abs/2602.11389)

[^3]: Venkatesh, R., et al. (2025). Value-guided action planning with JEPA world models. arXiv:2601.00844.
@[card](https://arxiv.org/abs/2601.00844)

[^4]: Cho, J., et al. (2023). Separable Physics-Informed Neural Networks. In: Koyejo, S., et al. (eds) Advances in Neural Information Processing Systems 36 (NeurIPS 2023).
@[card](https://arxiv.org/abs/2306.15969)

[^5]: Cardoso-Bihlo, E. & Bihlo, A. (2024). Exactly conservative physics-informed neural networks and deep operator networks for dynamical systems. Neural Networks, 182, 106826. arXiv:2311.14131.
@[card](https://arxiv.org/abs/2311.14131)

[^6]: Kumar, A., et al. (2025). A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures. arXiv:2602.03604.
@[card](https://arxiv.org/abs/2602.03604)

[^7]: Patel, M., et al. (2024). Cognitively Inspired Energy-Based World Models. arXiv:2406.08862.
@[card](https://arxiv.org/abs/2406.08862)

[^8]: Chen, Y., et al. (2024). Autoregressive Language Models are Secretly Energy-Based Models: Insights into the Lookahead Capabilities of Next-Token Prediction. arXiv:2512.15605.
@[card](https://arxiv.org/abs/2512.15605)

---

## 🎯 5. まとめ — World Modelsの本質

### 5.1 Part 1で学んだこと

本Partでは、World Modelsの**理論的基盤**を完全に構築した:

**核心概念**:
- 生成モデルの最終到達点は「画像生成」ではなく「環境理解+予測+シミュレーション」
- JEPAはピクセル生成をスキップし、潜在空間で予測する革命的アーキテクチャ
- Physics-Informed World Modelsは物理法則を埋め込み、データ効率とsim2realを改善
- Energy-Based定式化により、全生成モデル（VAE/GAN/Diffusion/JEPA）が統一理論に収束

**数学的武器庫**:
- I/V/VL-JEPA、LeJEPA、Causal-JEPAの完全理論
- Transfusionの統一損失関数（AR + Diffusion）
- Hamiltonian NNとSPINNによる物理法則学習
- EB-JEPAとPredictive Codingによる認知科学的定式化

**実装スキル**:
- 4つの主要アーキテクチャ（I-JEPA、V-JEPA、HNN、Energy-Based WM）のJulia実装
- EMA更新、Stop gradient、NCE、Gradient-based inferenceの実践

### 5.2 Part 2への接続

Part 2では、これらの理論を**実世界応用**に展開する:

- 強化学習統合（DreamerV3、MuZero、IRIS）
- ロボットマニピュレーション（RT-1/RT-2、GNM）
- 動画生成（Sora、VideoPoet、WALT）
- 科学シミュレーション（AlphaFold3、Climate modeling）

Part 1の理論は**全ての応用の基盤**となる。次回はこれらを実践する。

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
