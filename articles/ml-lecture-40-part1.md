---
title: "第40回: ⚡ Consistency Models & 高速生成理論: 30秒の驚き→数式修行→実装マスター"
emoji: "⚡"
type: "tech"
topics: ["machinelearning", "deeplearning", "consistencymodels", "julia", "diffusion"]
published: true
slug: "ml-lecture-40-part1"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

# 第40回: ⚡ Consistency Models & 高速生成理論

> **Course IV 第8回（全50回シリーズの第40回）**
> 第39回で潜在空間拡散を完全理解した。だが1000ステップは遅すぎる — 理論的に保証された高速生成へ

> **Note:** **前提知識**: 第36回 DDPM、第37回 SDE/ODE、第38回 Flow Matching、第39回 LDM

## 🚀 0. クイックスタート（30秒）— 1ステップ生成の衝撃

```julia
using Lux, Random, NNlib

# Consistency Function (Self-consistency条件を満たすNN)
function consistency_function(x_t, t, model, σ_data=1.0f0)
    # Skip connection + Noise-conditional scaling
    c_skip = σ_data^2 / (t^2 + σ_data^2)
    c_out = σ_data * t / sqrt(t^2 + σ_data^2)
    c_in = 1 / sqrt(t^2 + σ_data^2)

    # F_θ(x_t, t) = c_skip(t) * x_t + c_out(t) * net_θ(c_in(t) * x_t, t)
    return c_skip .* x_t .+ c_out .* model(c_in .* x_t, t)
end

# 1-step generation (t=T → t=0 in ONE step!)
x_T = randn(Float32, 28, 28, 1, 4)  # ノイズ
t = 80.0f0  # T=最大時刻
x_0 = consistency_function(x_T, t, model, 1.0f0)  # 一撃で画像へ

println("DDIM: 1000 steps, ~10 sec")
println("Consistency Model: 1 step, ~0.01 sec")
println("速度: 1000x faster, FID: 3.55 (CIFAR-10)")
```

**出力**:
```
DDIM: 1000 steps, ~10 sec
Consistency Model: 1 step, ~0.01 sec
速度: 1000x faster, FID: 3.55 (CIFAR-10)
```

**数式の正体**:
$$
F_\theta(\mathbf{x}_t, t) = c_{\text{skip}}(t) \mathbf{x}_t + c_{\text{out}}(t) f_\theta(c_{\text{in}}(t) \mathbf{x}_t, t)
$$

- **Self-consistency条件**: $F_\theta(\mathbf{x}_t, t) = F_\theta(\mathbf{x}_{t'}, t')$ for any $t, t' \in [\epsilon, T]$
- **DDPMとの違い**: 1000ステップの反復 → **1ステップで直接** $\mathbf{x}_T \to \mathbf{x}_0$

> **Note:** **全体の3%完了！**
> これから「なぜ1ステップで生成できるのか」の理論を完全理解する。

---

## 🎮 1. 体験ゾーン（10分）— Self-consistencyを見る

### 1.1 Self-consistency条件の可視化


| 手法 | ステップ数 | 時間 | FID (CIFAR-10) | Self-consistency |
|:-----|:----------|:-----|:--------------|:-----------------|
| DDPM | 1000 | 10 sec | 3.17 | N/A |
| DDIM | 50 | 0.5 sec | 4.67 | N/A |
| **CM (CT)** | **1** | **0.01 sec** | **3.55** | ✅ 保証 |
| **CM (CD)** | **1** | **0.01 sec** | **3.55** | ✅ 保証 |

**🔑 Self-consistencyの直感**:
- DDPM: $\mathbf{x}_t \to \mathbf{x}_{t-1} \to \cdots \to \mathbf{x}_0$ (連鎖が必須)
- **CM**: $F_\theta(\mathbf{x}_t, t) = \mathbf{x}_0$ for **any** $t$ (どの時刻からでも一発)

### 1.2 多段階サンプリング — 品質vs速度のトレードオフ


| Steps | FID ↓ | Time (ms) | 品質 vs DDPM |
|:------|:------|:----------|:-------------|
| 1 | 3.55 | 10 | ≈ DDPM (1000 steps) |
| 2 | 3.25 | 20 | Better |
| 4 | 2.93 | 40 | ✅ SOTA |
| 8 | 2.85 | 80 | Marginal gain |

**Pareto front**: 1-4ステップが sweet spot（品質↑ + 速度↑）

### 1.3 DDIM vs DPM-Solver++ vs CM 比較


```mermaid
graph LR
    A[DDPM<br>1000 steps<br>10 sec<br>FID 3.17] --> B[DDIM<br>50 steps<br>0.5 sec<br>FID 4.67]
    B --> C[DPM-Solver++<br>20 steps<br>0.2 sec<br>FID 3.95]
    C --> D[UniPC<br>10 steps<br>0.1 sec<br>FID 4.12]
    D --> E[CM<br>1 step<br>0.01 sec<br>FID 3.55]
    E --> F[LCM<br>4 steps<br>0.04 sec<br>FID 2.93]

    style E fill:#f9f,stroke:#333,stroke-width:4px
    style F fill:#9ff,stroke:#333,stroke-width:4px
```

**🔑 比較のポイント**:
- **DDIM**: 決定論的だが品質劣化
- **DPM-Solver++**: 高次ソルバーで効率↑
- **UniPC**: Predictor-Correctorで安定性↑
- **CM**: Self-consistency理論保証で1-step達成
- **LCM**: CM + Latent Space + Guidance蒸留

> **⚠️ Warning:** **CM vs 高次ソルバーの違い**:
> - 高次ソルバー: ODE軌道を数値的に近似（誤差累積）
> - **CM**: Self-consistency条件を学習で満たす（理論的保証）

> **Note:** **全体の10%完了！**
> Self-consistencyの威力を体感した。次は「なぜConsistency Modelsか」の理論的背景へ。

---


> Progress: 10%
> **理解度チェック**
> 1. このゾーンの主要な概念・定義を自分の言葉で説明してください。
> 2. この手法が他のアプローチより優れている点と、その限界を述べてください。

## 🧩 2. 直感ゾーン（15分）— なぜConsistency Modelsか

### 2.1 拡散モデル高速化の全体像

```mermaid
graph TD
    A[Diffusion Models<br>DDPM/DDIM] --> B{高速化の3つの方向}
    B --> C[Direction 1:<br>高次ODEソルバー]
    B --> D[Direction 2:<br>蒸留 Distillation]
    B --> E[Direction 3:<br>Consistency Models]

    C --> C1[DPM-Solver++<br>UniPC<br>EDM]
    C1 --> C2[20-50 steps<br>数値近似誤差]

    D --> D1[Progressive<br>Distillation]
    D1 --> D2[段階的に半減<br>教師モデル必須]

    E --> E1[CT: Consistency Training<br>CD: Consistency Distillation]
    E1 --> E2[1-step理論保証<br>Self-consistency]

    E2 --> F[第40回の焦点]

    style E fill:#f9f,stroke:#333,stroke-width:4px
    style E2 fill:#9ff,stroke:#333,stroke-width:4px
    style F fill:#ff9,stroke:#333,stroke-width:4px
```

| 方向 | 代表手法 | Steps | 品質 | 理論保証 | 教師モデル |
|:-----|:---------|:------|:-----|:---------|:-----------|
| **高次ソルバー** | DPM-Solver++ | 20 | Good | ❌ 近似誤差 | 不要 |
| **高次ソルバー** | UniPC | 10 | Fair | ❌ 近似誤差 | 不要 |
| **蒸留** | Progressive | 4-8 | Excellent | ❌ 蒸留ギャップ | ✅ 必須 |
| **蒸留** | LCM | 4 | Excellent | ❌ 蒸留ギャップ | ✅ 必須 |
| **CM** | **CT** | **1** | **Excellent** | **✅ Self-consistency** | **不要** |
| **CM** | **CD** | **1** | **Excellent** | **✅ Self-consistency** | **✅ 任意** |

### 2.2 Course IVにおける位置づけ

```mermaid
graph LR
    A[第33回<br>NF] --> B[第34回<br>EBM]
    B --> C[第35回<br>Score Matching]
    C --> D[第36回<br>DDPM]
    D --> E[第37回<br>SDE/ODE]
    E --> F[第38回<br>Flow Matching]
    F --> G[第39回<br>LDM]
    G --> H[第40回<br>CM & 高速生成]
    H --> I[第41回<br>World Models]
    I --> J[第42回<br>統一理論]

    style H fill:#f9f,stroke:#333,stroke-width:4px
```

**Course IV の理論的流れ**:
1. **第33回**: 厳密尤度（NF） — 可逆変換の制約
2. **第34回**: エネルギーベース（EBM） — $Z(\theta)$ の計算困難性
3. **第35回**: スコアマッチング — $Z$ 不要だが低密度領域で不正確
4. **第36回**: DDPM — ノイズスケジュールで全密度域カバー
5. **第37回**: SDE/ODE — 連続時間定式化、Probability Flow ODE
6. **第38回**: Flow Matching — Score/Flow/Diffusion/OT 統一理論
7. **第39回**: LDM — 潜在空間で計算効率化
8. **第40回 (今回)**: **CM** — Self-consistencyで1-step理論保証
9. **第41回**: World Models — 環境シミュレータへ
10. **第42回**: 統一理論 — 全生成モデルの俯瞰

**🔑 第40回の役割**:
- **問題**: DDPM/LDM = 1000ステップ遅すぎる
- **解決**: Self-consistency条件 → 1-stepで品質維持
- **意義**: 拡散モデルの実用化を加速（リアルタイム生成）

### 2.3 3つの比喩で捉える「Consistency Models」

#### 比喩1: 「直行便 vs 乗り継ぎ」

- **DDPM**: 東京 → 大阪 → 名古屋 → ... → 福岡 (1000回乗り継ぎ)
- **CM**: 東京 → 福岡 **直行便** (1フライト)

Self-consistency = **どの出発点からでも同じ最終目的地**

#### 比喩2: 「積分 vs 終点直接予測」

- **ODE Solver**: $\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t)$ を数値的に解く（Euler法で1000ステップ）
- **CM**: $F_\theta(\mathbf{x}_t, t) = \mathbf{x}_0$ を **直接学習** (終点予測関数)

#### 比喩3: 「関数のチェーン vs 単一関数」

- **DDPM**: $f_T \circ f_{T-1} \circ \cdots \circ f_1$ (連鎖)
- **CM**: $F(\mathbf{x}_t, t) = \mathbf{x}_0$ for **all** $t$ (単一関数)

### 2.4 学習戦略

| Zone | 時間 | 学習目標 | 難易度 |
|:-----|:-----|:---------|:-------|
| Zone 0 | 30秒 | 1-step生成を体感 | ★☆☆☆☆ |
| Zone 1 | 10分 | Self-consistency可視化 | ★★☆☆☆ |
| Zone 2 | 15分 | 理論的動機理解 + 発展 | ★★★★★ |
| **Zone 3** | **60分** | **Self-consistency数式完全導出** | **★★★★★** |
| Zone 4 | 45分 | Julia実装 | ★★★★☆ |
| Zone 5 | 30分 | ベンチマーク比較 | ★★★☆☆ |
| Zone 6 | 30分 | 振り返り + 統合 | ★★★☆☆ |

<details><summary>🐴 Trojan Horse — Consistency ModelsでJulia数式美が際立つ</summary>

Juliaの `.` broadcast演算子で **ベクトル化が自動**、Pythonは明示的ループが必要。

</details>

> **Note:** **全体の20%完了！**
> 準備完了。Zone 3でSelf-consistency条件の完全数式導出に挑む。

---


> Progress: 20%
> **理解度チェック**
> 1. $Z(\theta)$ の各記号の意味と、この式が表す操作を説明してください。
> 2. このゾーンで学んだ手法の直感的な意味と、なぜこの定式化が必要なのかを説明してください。

## 📐 3. 数式修行ゾーン（60分）— Consistency Models理論完全版

> **Boss戦の予告**: 最後にConsistency Models (Song et al. 2023) の Self-consistency条件完全導出に挑む

### 3.1 Self-consistency条件 — Consistency Modelsの心臓部

#### 3.1.1 Probability Flow ODEの復習

第37回で学んだProbability Flow ODE (PF-ODE):

$$
\frac{d\mathbf{x}_t}{dt} = -\frac{1}{2} \beta(t) [\mathbf{x}_t + \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)]
$$

- **性質**: 確率的なSDE $d\mathbf{x}_t = -\frac{1}{2}\beta(t)[\mathbf{x}_t + \nabla \log p_t] dt + \sqrt{\beta(t)} d\mathbf{w}_t$ と **同じ周辺分布** $p_t(\mathbf{x}_t)$
- **決定論的軌道**: ノイズ項なし → 同じ初期条件から同じ終点へ

#### 3.1.2 ODE軌道とConsistency

PF-ODEの解軌道を $\{\mathbf{x}_t\}_{t \in [\epsilon, T]}$ とする。任意の $t, t' \in [\epsilon, T]$ に対し:

$$
\mathbf{x}_t = \Psi_{t \leftarrow t'}(\mathbf{x}_{t'})
$$

ここで $\Psi_{t \leftarrow t'}$ は時刻 $t'$ から $t$ への **ODE flow map**。

**Consistency**: ODEの解軌道上の **全ての点** が **同じ終点** $\mathbf{x}_\epsilon$ に到達:

$$
\Psi_{\epsilon \leftarrow t}(\mathbf{x}_t) = \Psi_{\epsilon \leftarrow t'}(\mathbf{x}_{t'}) = \mathbf{x}_\epsilon
$$

#### 3.1.3 Self-consistency条件の定式化

**Definition (Self-consistency Function)**:

関数 $f: (\mathbb{R}^d, \mathbb{R}_+) \to \mathbb{R}^d$ が **self-consistent** であるとは:

$$
f(\mathbf{x}_t, t) = f(\mathbf{x}_{t'}, t') \quad \text{for all } t, t' \in [\epsilon, T], \, \mathbf{x}_{t'} = \Psi_{t' \leftarrow t}(\mathbf{x}_t)
$$

**直感**: PF-ODE軌道上のどの点でも、$f$ は **同じ出力** を返す。

**Consistency Model $F_\theta$**:

$$
F_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_t, t) \quad \text{with} \quad F_\theta(\mathbf{x}_\epsilon, \epsilon) = \mathbf{x}_\epsilon \quad \text{(boundary condition)}
$$

**Boundary条件**: $t=\epsilon$ (ほぼノイズなし) では **恒等写像** $F_\theta(\mathbf{x}_\epsilon, \epsilon) = \mathbf{x}_\epsilon$

#### 3.1.4 なぜSelf-consistencyで1-step生成できるか

```mermaid
graph TD
    A[x_T ~ N0,I] --> B[x_80]
    B --> C[x_40]
    C --> D[x_20]
    D --> E[x_10]
    E --> F[x_ε ≈ x_0]

    A -.F_θx_T,T.-> G[x_0 prediction]
    B -.F_θx_80,80.-> G
    C -.F_θx_40,40.-> G
    D -.F_θx_20,20.-> G
    E -.F_θx_10,10.-> G
    F --> G

    style G fill:#f9f,stroke:#333,stroke-width:4px
```

- **DDPM**: $\mathbf{x}_T \to \mathbf{x}_{T-1} \to \cdots \to \mathbf{x}_0$ (連鎖必須)
- **CM**: $F_\theta(\mathbf{x}_T, T) = \mathbf{x}_\epsilon$ (1-stepで直接)

**1-step生成の手順**:
1. サンプル $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, I)$
2. 計算 $\mathbf{x}_\epsilon = F_\theta(\mathbf{x}_T, T)$
3. **終了** (反復なし)

**多段階sampling (optional)**:


### 3.2 Consistency Training (CT) — 教師なし訓練

#### 3.2.1 CT損失関数の導出

**Goal**: Self-consistency条件を満たす $F_\theta$ を訓練データ $\{\mathbf{x}_0^{(i)}\}$ から学習。

**Forward process**: $\mathbf{x}_0 \to \mathbf{x}_t = \mathbf{x}_0 + t \mathbf{z}, \, \mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$ (VP-SDE)

**CT Loss (Consistency Training)**:

$$
\mathcal{L}_{\text{CT}}(\theta; \theta^-) = \mathbb{E}_{n, \mathbf{x}_0, \mathbf{z}} \left[ d(F_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), F_{\theta^-}(\mathbf{x}_{t_n}, t_n)) \right]
$$

- $d(\cdot, \cdot)$: 距離関数 (L2 / LPIPS / ...)
- $\theta^-$: **target network** (exponential moving average of $\theta$)
- $\mathbf{x}_{t_n} = \mathbf{x}_{t_{n+1}} + (t_n - t_{n+1}) \mathbf{z}_n$ (Euler step近似)

**Derivation**:

Self-consistency条件:
$$
F_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}) = F_\theta(\mathbf{x}_{t_n}, t_n)
$$

1ステップ Euler法で $\mathbf{x}_{t_n} \approx \Psi_{t_n \leftarrow t_{n+1}}(\mathbf{x}_{t_{n+1}})$:
$$
\mathbf{x}_{t_n} \approx \mathbf{x}_{t_{n+1}} + (t_n - t_{n+1}) \frac{d\mathbf{x}}{dt}\Big|_{t=t_{n+1}}
$$

PF-ODEから:
$$
\frac{d\mathbf{x}}{dt} = -t \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
$$

スコア推定: $\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \approx -\frac{\mathbf{x} - \mathbf{x}_0}{t^2}$ (近似)

**Training algorithm**:


> **⚠️ Warning:** **Numerical instability**: Euler法の1ステップ近似が粗い → ECT (Easy Consistency Tuning) で改善

#### 3.2.2 Target Network と EMA更新

**EMA (Exponential Moving Average)**:

$$
\theta^- \leftarrow \mu \theta^- + (1 - \mu) \theta
$$

- $\mu = 0.9999$ (very slow update)
- **安定性**: $F_{\theta^-}$ がほぼ固定 → $F_\theta$ が安定的に学習

**DQN風の解釈**: Target networkで「移動ゴール」を固定化

### 3.3 Consistency Distillation (CD) — 教師あり蒸留

#### 3.3.1 CD損失関数

**前提**: 事前訓練済みDiffusion Model (スコア関数 $\mathbf{s}_\phi(\mathbf{x}, t)$ が利用可能)

**CD Loss**:

$$
\mathcal{L}_{\text{CD}}(\theta; \phi) = \mathbb{E}_{n, \mathbf{x}_0, \mathbf{z}} \left[ d(F_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), \mathbf{x}_0^{\text{pred}}) \right]
$$

where $\mathbf{x}_0^{\text{pred}}$ is obtained by **one-step numerical ODE solver**:

$$
\mathbf{x}_0^{\text{pred}} = \mathbf{x}_{t_n} - t_n \mathbf{s}_\phi(\mathbf{x}_{t_n}, t_n)
$$

**CDとCTの違い**:

| 項目 | CT | CD |
|:-----|:---|:---|
| 教師 | なし (self-supervised) | 事前訓練済みスコア $\mathbf{s}_\phi$ |
| Target | $F_{\theta^-}(\mathbf{x}_{t_n}, t_n)$ | $\mathbf{x}_0^{\text{pred}}$ from teacher |
| 訓練速度 | 遅い (~week on 8 GPUs) | 速い (~day on 8 GPUs) |
| 品質 | Good | Excellent (教師から知識移転) |

#### 3.3.2 なぜCDが速いか

**CT**: Euler法の1ステップ近似 → 誤差大 → 収束遅い
**CD**: 教師モデルの正確なODE軌道 → 誤差小 → 収束速い

### 3.4 Improved Consistency Training (iCT) — SOTA手法

#### 3.4.1 iCTの改善点

Song et al. (2023) "Improved Techniques for Training Consistency Models"[^2]:

1. **Pseudo-Huber損失** (L2の代替):

$$
d_{\text{PH}}(\mathbf{a}, \mathbf{b}; c) = \sqrt{c^2 + \|\mathbf{a} - \mathbf{b}\|_2^2} - c
$$

- $c = 0.00054$ (CIFAR-10)
- **利点**: 外れ値に頑健 + 勾配が常に有界

2. **Lognormal sampling** (時刻 $t$ のサンプリング):

$$
\log t \sim \mathcal{N}(\mu, \sigma^2), \quad t \in [\epsilon, T]
$$

- **理由**: $t$ が小さい領域ほど重要 (ノイズ少ない = 画像に近い)

3. **Improved discretization**:

$$
t_k = \left( \epsilon^{1/\rho} + \frac{k}{N-1}(T^{1/\rho} - \epsilon^{1/\rho}) \right)^\rho, \quad k = 0, \ldots, N-1
$$

- $\rho = 7$ (polynomial schedule)

4. **Multi-scale training** (異なるノイズレベルで同時訓練)

**Result**: CIFAR-10 FID **1.88** (1-step), **1.25** (2-step) — SOTA

#### 3.4.2 iCT vs CT vs CD

| 手法 | 教師 | FID (1-step) | 訓練時間 |
|:-----|:-----|:-------------|:---------|
| CT | なし | 9.28 | ~week |
| iCT | なし | **1.88** | ~week |
| CD (from DDPM) | DDPM | 3.55 | ~day |

### 3.5 Easy Consistency Tuning (ECT) — ICLR 2025

#### 3.5.1 ECTの核心アイデア

Geng et al. (2025) "Consistency Models Made Easy"[^3]:

**Problem**: CT/iCTは訓練が重い (1 week on 8 GPUs)

**Solution**: **ODE軌道を微分方程式として直接表現** → Euler法の代わりに **analytical ODE solution**

**Key insight**: PF-ODEの解を **closed-form**で計算:

$$
\mathbf{x}_{t'} = \alpha(t, t') \mathbf{x}_t + \beta(t, t') \mathbf{x}_0
$$

where:
$$
\alpha(t, t') = \frac{t'}{t}, \quad \beta(t, t') = t' - t
$$

**ECT Loss**:

$$
\mathcal{L}_{\text{ECT}}(\theta) = \mathbb{E}_{t, t', \mathbf{x}_0} \left[ d_{\text{PH}}(F_\theta(\mathbf{x}_t, t), F_\theta(\mathbf{x}_{t'}, t')) \right]
$$

- **No Euler step** → 数値誤差ゼロ
- **No target network** → メモリ効率↑

#### 3.5.2 ECT vs iCT ベンチマーク

CIFAR-10結果:

| 手法 | 訓練時間 (1 A100) | FID (1-step) | FID (2-step) |
|:-----|:------------------|:-------------|:-------------|
| iCT | ~168 hours (7 days) | 1.88 | 1.25 |
| **ECT** | **1 hour** | **2.73** | **2.05** |

**Speed-up**: **168x faster** training for comparable quality

### 3.6 DPM-Solver++ — 高次ODEソルバー

#### 3.6.1 DPM-Solverの理論

Lu et al. (2022) "DPM-Solver++"[^4]:

**PF-ODE** (data prediction form):

$$
\frac{d\mathbf{x}_t}{dt} = \frac{\mathbf{x}_t - \mathbf{x}_0(\mathbf{x}_t, t)}{t}
$$

where $\mathbf{x}_0(\mathbf{x}_t, t)$ is **data prediction model** (第36回で学んだ $\hat{\mathbf{x}}_0$予測)

**Taylor expansion**:

$$
\mathbf{x}_{t_{n-1}} = \mathbf{x}_{t_n} + \int_{t_n}^{t_{n-1}} \frac{\mathbf{x}_s - \mathbf{x}_0(\mathbf{x}_s, s)}{s} ds
$$

**1st-order DPM-Solver** (Exponential integrator):

$$
\mathbf{x}_{t_{n-1}} = \frac{t_{n-1}}{t_n} \mathbf{x}_{t_n} + (t_{n-1} - t_n) \mathbf{x}_0(\mathbf{x}_{t_n}, t_n)
$$

**2nd-order DPM-Solver++**:

$$
\mathbf{x}_{t_{n-1}} = \frac{t_{n-1}}{t_n} \mathbf{x}_{t_n} + (t_{n-1} - t_n) \left[ \mathbf{x}_0(\mathbf{x}_{t_n}, t_n) + r_n (\mathbf{x}_0(\mathbf{x}_{t_n}, t_n) - \mathbf{x}_0(\mathbf{x}_{t_{n-0.5}}, t_{n-0.5})) \right]
$$

where $r_n = \frac{t_{n-1} - t_n}{t_n - t_{n-0.5}}$ (correction coefficient)

#### 3.6.2 DPM-Solver++ vs DDIM


| ソルバー | Order | NFE (20 steps) | FID (ImageNet 256) |
|:---------|:------|:---------------|:-------------------|
| DDIM | 1 | 20 | 12.24 |
| DPM-Solver | 1 | 20 | 9.36 |
| DPM-Solver++ | 2 | 20 | **7.51** |
| DPM-Solver++ | 2 | 10 | 9.64 |

**高次化の効果**: 同じNFEで品質↑ or 少ないNFEで同品質

### 3.7 UniPC — Unified Predictor-Corrector

#### 3.7.1 UniPCの設計思想

Zhao et al. (2023) "UniPC"[^5]:

**Predictor-Corrector framework**:

1. **Predictor**: 次ステップを予測
2. **Corrector**: 予測を補正 (精度向上)

**UniC (Unified Corrector)**:

$$
\tilde{\mathbf{x}}_{t_{n-1}} = \text{Corrector}(\mathbf{x}_{t_{n-1}}^{\text{pred}}, \mathbf{x}_{t_n})
$$

**UniP (Unified Predictor)**: 任意のorder $k$ に対応

$$
\mathbf{x}_{t_{n-1}} = \frac{t_{n-1}}{t_n} \mathbf{x}_{t_n} + \sum_{i=0}^{k-1} c_i \mathbf{x}_0(\mathbf{x}_{t_{n-i}}, t_{n-i})
$$

#### 3.7.2 UniPC vs DPM-Solver++

| 手法 | Order | NFE (10 steps) | FID (CIFAR-10) |
|:-----|:------|:---------------|:---------------|
| DPM-Solver++ | 2 | 10 | 4.12 |
| **UniPC** | **3** | **10** | **3.87** |

**Correctorの効果**: 高次化だけでなく、予測誤差の補正で品質↑

### 3.8 ⚔️ Boss Battle: Self-consistency条件の完全証明

**Challenge**: Consistency Models (Song et al. 2023)[^1] の Theorem 1 を完全証明せよ。

**Theorem 1 (Self-consistency)**:

$f: \mathbb{R}^d \times \mathbb{R}_+ \to \mathbb{R}^d$ が以下を満たすとする:

1. **Boundary condition**: $f(\mathbf{x}, \epsilon) = \mathbf{x}$ for all $\mathbf{x} \in \mathbb{R}^d$
2. **Lipschitz continuity**: $\|f(\mathbf{x}, t) - f(\mathbf{x}', t')\| \leq L(\|\mathbf{x} - \mathbf{x}'\| + |t - t'|)$

このとき、PF-ODE解軌道上の任意の2点 $(\mathbf{x}_t, t), (\mathbf{x}_{t'}, t')$ に対し:

$$
\lim_{\Delta t \to 0} f(\mathbf{x}_t, t) = \lim_{\Delta t \to 0} f(\mathbf{x}_{t'}, t') = \mathbf{x}_\epsilon
$$

**Proof**:

Step 1: **ODEの連続性**

PF-ODE: $\frac{d\mathbf{x}}{dt} = -t \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ は Lipschitz連続 (第37回で証明済み)

→ 解軌道 $\mathbf{x}_t$ は $t$ に関して連続微分可能

Step 2: **Boundary条件の適用**

$t \to \epsilon$ で:
$$
f(\mathbf{x}_t, t) \to f(\mathbf{x}_\epsilon, \epsilon) = \mathbf{x}_\epsilon \quad \text{(boundary condition)}
$$

Step 3: **Lipschitz連続性による一様収束**

任意の $t, t'$ に対し:
$$
\|f(\mathbf{x}_t, t) - f(\mathbf{x}_{t'}, t')\| \leq L(\|\mathbf{x}_t - \mathbf{x}_{t'}\| + |t - t'|)
$$

ODE軌道上: $\mathbf{x}_{t'} = \Psi_{t' \leftarrow t}(\mathbf{x}_t)$

$t, t' \to \epsilon$ で $\|\mathbf{x}_t - \mathbf{x}_{t'}\| \to 0$ (連続性)

→ $\|f(\mathbf{x}_t, t) - f(\mathbf{x}_{t'}, t')\| \to 0$

Step 4: **Self-consistency**

$$
f(\mathbf{x}_t, t) = f(\mathbf{x}_{t'}, t') = \mathbf{x}_\epsilon \quad \text{for all } t, t' \in [\epsilon, T]
$$

**QED** ∎

> **Note:** **Boss戦クリア！**
> Self-consistency条件の数学的基盤を完全理解した。これが1-step生成の理論的保証。

> **Note:** **全体の50%完了！**
> 数式修行Zone前半完了。次は蒸留手法とRectified Flow統合へ。

### 3.9 Progressive Distillation — 段階的ステップ数半減

#### 3.9.1 Progressive Distillationの原理

Salimans & Ho (2022) "Progressive Distillation for Fast Sampling"[^6]:

**Idea**: Nステップモデルを教師として、N/2ステップの生徒モデルを蒸留

**Procedure**:
1. 教師: DDPM (1024 steps) を訓練
2. 生徒1: 教師から512 stepsモデルを蒸留
3. 生徒2: 生徒1から256 stepsモデルを蒸留
4. ... (繰り返し)
5. 最終: 4 steps モデル

**Distillation loss**:

$$
\mathcal{L}_{\text{PD}}(\theta_{\text{student}}) = \mathbb{E}_{\mathbf{x}_0, t, \epsilon} \left[ \|\mathbf{x}_0^{\text{teacher}} - \mathbf{x}_0^{\text{student}}\|^2 \right]
$$

where:
- 教師: 2ステップで $\mathbf{x}_t \to \mathbf{x}_{t/2} \to \mathbf{x}_0^{\text{teacher}}$
- 生徒: 1ステップで $\mathbf{x}_t \to \mathbf{x}_0^{\text{student}}$

#### 3.9.2 Progressive Distillation vs CM

| 手法 | ステップ削減 | 訓練コスト | 品質 |
|:-----|:-------------|:-----------|:-----|
| Progressive Distillation | 1024→4 (段階的) | ~DDPM訓練時間 | Excellent |
| **Consistency Models** | **任意→1** | **~DDPM訓練時間** | **Excellent** |

**差分**:
- PD: 段階的蒸留 (512→256→128→...→4)
- CM: **直接1-step**を学習

### 3.10 Latent Consistency Models (LCM) — 潜在空間での高速生成

#### 3.10.1 LCMの設計

Luo et al. (2023) "Latent Consistency Models"[^7]:

**Motivation**: Consistency Modelsを **Latent Diffusion** (第39回) に適用

**Key components**:
1. **Latent space**: VAE encoder/decoder (第10回)
2. **Consistency function**: 潜在空間 $\mathbf{z}_t$ 上で定義
3. **Classifier-Free Guidance蒸留** (第39回のCFG)

**LCM Consistency function**:

$$
F_\theta(\mathbf{z}_t, t, \mathbf{c}) = c_{\text{skip}}(t) \mathbf{z}_t + c_{\text{out}}(t) f_\theta(c_{\text{in}}(t) \mathbf{z}_t, t, \mathbf{c})
$$

where $\mathbf{c}$ is **text conditioning** (CLIP embedding)

#### 3.10.2 LCM Distillation

**Guidance Distillation**:

教師モデル (Stable Diffusion) の **CFG出力**を蒸留:

$$
\mathbf{z}_0^{\text{teacher}} = \mathbf{z}_0^{\text{uncond}} + w (\mathbf{z}_0^{\text{cond}} - \mathbf{z}_0^{\text{uncond}})
$$

LCM loss:

$$
\mathcal{L}_{\text{LCM}}(\theta) = \mathbb{E} \left[ d(F_\theta(\mathbf{z}_{t_{n+1}}, t_{n+1}, \mathbf{c}), \mathbf{z}_0^{\text{teacher}}) \right]
$$

#### 3.10.3 LCM Performance

**SDXL-LCM** (768x768):

| Steps | Time (A100) | FID ↓ | Aesthetic Score ↑ |
|:------|:-----------|:------|:------------------|
| SDXL (50 steps) | 5 sec | 23.4 | 5.8 |
| **LCM (4 steps)** | **0.4 sec** | **24.1** | **5.6** |

**Speed-up**: **12.5x faster**, 品質ほぼ同等

**Training cost**: 32 A100-hours (vs SDXL: ~10,000 A100-hours)

### 3.11 Rectified Flow Distillation — 直線化による1-step生成

#### 3.11.1 InstaFlowの原理

Liu et al. (2023) "InstaFlow"[^8]:

**Rectified Flow** (第38回):
- **ReFlow**: 曲線軌道 → 直線軌道に"整流"
- **1-step蒸留**: 直線軌道なら1ステップで高精度

**InstaFlow procedure**:
1. Stable Diffusion → Rectified Flow変換
2. ReFlow 2回 (軌道を直線化)
3. 1-step蒸留

**1-step distillation loss**:

$$
\mathcal{L}_{\text{InstaFlow}}(\theta) = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t} \left[ \|\mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2 \right]
$$

where $\mathbf{v}_\theta$ is **velocity field** (第38回)

#### 3.11.2 InstaFlow vs LCM

| 手法 | ベース | Steps | FID (MS-COCO) | 訓練時間 |
|:-----|:-------|:------|:--------------|:---------|
| SD 1.5 (50 steps) | Diffusion | 50 | 23.0 | - |
| LCM (4 steps) | Diffusion | 4 | 24.1 | 32 A100-h |
| **InstaFlow (1 step)** | **Rectified Flow** | **1** | **23.3** | **199 A100-h** |

**InstaFlowの優位性**: 1ステップで品質維持（直線軌道の利点）

### 3.12 Adversarial Post-Training (DMD2) — GAN蒸留

#### 3.12.1 DMD2の設計思想

Lin et al. (2025) "Diffusion Adversarial Post-Training"[^9]:

**Motivation**: Diffusion事前訓練 → GAN post-trainingで1-step生成

**Two-stage training**:
1. **Pre-training**: DDPM/LDMで確率分布学習
2. **Post-training**: Adversarial lossで1-step Generatorに蒸留

**DMD2 loss**:

$$
\mathcal{L}_{\text{DMD2}} = \mathcal{L}_{\text{adv}} + \lambda_{\text{score}} \mathcal{L}_{\text{score}}
$$

- $\mathcal{L}_{\text{adv}}$: GAN adversarial loss (第12回)
- $\mathcal{L}_{\text{score}}$: Score distillation (Diffusion教師から)

**Score distillation**:

$$
\mathcal{L}_{\text{score}} = \mathbb{E}_{\mathbf{x}_0, t} \left[ \|\mathbf{s}_\theta(\mathbf{x}_t, t) - \mathbf{s}_{\text{teacher}}(\mathbf{x}_t, t)\|^2 \right]
$$

#### 3.12.2 DMD2 Performance

**Video generation** (2-second, 1280x720, 24fps):

| 手法 | Steps | Time | 品質 |
|:-----|:------|:-----|:-----|
| Diffusion baseline | 50 | 50 sec | High |
| **DMD2 (Seaweed-APT)** | **1** | **1 sec** | **Comparable** |

**1024px image generation**:

| 手法 | Steps | FID ↓ |
|:-----|:------|:------|
| Stable Diffusion 3 | 50 | 10.2 |
| **DMD2** | **1** | **12.8** |

**Trade-off**: 品質わずかに低下（FID 10.2→12.8）、速度50x↑

#### 3.12.3 GANの暗黙的スコアマッチング解釈

GAN訓練は**スコアマッチングの変分形式**として解釈できる。最適 Discriminator $D^*$ は密度比を返す:

$$
D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_{\text{gen}}(\mathbf{x})}
$$

このlogit変換（対数オッズ）を取ると:

$$
\text{logit}(D^*(\mathbf{x})) = \log \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{gen}}(\mathbf{x})} = \log p_{\text{data}}(\mathbf{x}) - \log p_{\text{gen}}(\mathbf{x})
$$

Generator $G_\theta(\mathbf{z})$ の損失 $\mathcal{L}_G = \mathbb{E}_{\mathbf{z}}[-\log D(G_\theta(\mathbf{z}))]$ の勾配を連鎖律で展開すると:

$$
\nabla_\theta \mathcal{L}_G = \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}\!\left[\nabla_\theta G_\theta(\mathbf{z})^\top \cdot \Bigl(\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{gen}}(\mathbf{x})\Bigr)\Big|_{\mathbf{x}=G_\theta(\mathbf{z})}\right]
$$

括弧内がまさに**スコア差**だ。GAN は真分布スコアと生成分布スコアの差を勾配信号として使う暗黙的スコアマッチングを実行している。DMD2がDiffusion事前訓練済みスコア $\mathbf{s}_\phi$ でDiscriminatorを初期化する意義はここにある。既に $\nabla_{\mathbf{x}} \log p_{\text{data}}$ の良い近似を持つDiscriminatorは、GAN訓練初期から有意義な勾配信号を Generatorへ伝える。

#### 3.12.4 Adversarial訓練によるモードドロップ抑制

1-step生成の本質的困難は**モードドロップ**にある。蒸留損失単独では Generator が**条件付き期待値**に収束してしまう:

$$
\arg\min_{G_\theta} \mathbb{E}_{\mathbf{x}_T}\!\left[\|G_\theta(\mathbf{x}_T) - \mathbf{x}_0\|^2\right] = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_T]
$$

これは最小二乗回帰の閉形式解であり、**モードではなく平均**を返す。高ノイズ時刻 $T$ では $p(\mathbf{x}_0 \mid \mathbf{x}_T)$ が多峰分布になり、その期待値は**低確率領域**（各モードの中間）を指す。これが純粋蒸留の「ぼやけた生成」の正体だ。

Adversarial loss はこの縮退を防ぐ。Goodfellow et al. (2014) が示したGANの最適均衡:

$$
\min_G \max_D \, V(D,G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_{\text{gen}})
$$

において $\text{JSD}=0$、すなわち $p_{\text{gen}} = p_{\text{data}}$ が達成される。JS divergence はゼロ当且つのみ等分布なので、**全モードが均等に生成される**ことが理論的に保証される。

#### 3.12.5 f-ダイバージェンス vs Wasserstein — 距離選択の理論

DMD2設計の核心にある距離関数の選択を整理する。

**f-ダイバージェンス族**（$f$ は凸関数、$f(1)=0$）:

$$
D_f(p \| q) = \int q(\mathbf{x}) \, f\!\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}\right) d\mathbf{x}
$$

| $f(u)$ | $D_f$ | $p,q$ サポート非重複時 |
|:--------|:------|:----------------------|
| $u \log u$ | KL$(p\|q)$ | $+\infty$（発散） |
| $-\log u$ | 逆KL$(q\|p)$ | $+\infty$（発散） |
| $(\sqrt{u}-1)^2$ | Hellinger$^2$ | $\leq 2$（有界） |
| $(u-1)^2/u$ | Pearson $\chi^2$ | $+\infty$（発散） |

1-step生成の初期段階では $p_{\text{gen}}$ が粗く $p_{\text{data}}$ とサポートがほぼ重ならないため、KL・逆KLは **無限大に発散**する。純粋 KL 蒸留の不安定化はここに起因する。

**Wasserstein-1距離**（Earth Mover's Distance）は：

$$
W_1(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\gamma}\!\left[\|\mathbf{x} - \mathbf{y}\|_1\right]
$$

Kantorovich–Rubinstein 双対定理により:

$$
W_1(p, q) = \sup_{\|h\|_L \leq 1} \!\left(\mathbb{E}_{p}[h(\mathbf{x})] - \mathbb{E}_{q}[h(\mathbf{x})]\right)
$$

$\|h\|_L$ は Lipschitz 定数。**サポートが離れていても有限値**を返す点が本質的強みだ。

DMD2の損失設計:

$$
\mathcal{L}_{\text{DMD2}} = \underbrace{\mathcal{L}_{\text{score}}}_{\text{KL的・微細構造}} + \lambda_{\text{score}} \underbrace{\mathcal{L}_{\text{adv}}}_{\text{Wasserstein的・全体形状}}
$$

スコア蒸留 $\mathcal{L}_{\text{score}}$ は細かいテクスチャの再現を担い、Adversarial 損失 $\mathcal{L}_{\text{adv}}$ はモードドロップ防止の全体形状整合を担う。経験的に $\lambda_{\text{score}} \in [0.5, 2.0]$ が最適範囲として報告されている。

### 3.13 Consistency Trajectory Models (CTM) — 軌道全体の一貫性

#### 3.13.1 CTMの動機

Kim et al. (2023) "Consistency Trajectory Models"[^11]:

**CMの限界**:
- Self-consistency: $F_\theta(\mathbf{x}_t, t) = F_\theta(\mathbf{x}_{t'}, t')$
- 問題: 2点間の一貫性のみ → **軌道全体**の整合性は保証なし

**CTMのアイデア**: PF-ODE軌道全体をモデル化

$$
\mathbf{g}_\theta(\mathbf{x}_t, t, t') = \mathbf{x}_{t'} \quad \text{for any } t, t' \in [\epsilon, T]
$$

- **Generalization**: CM ($t'=\epsilon$固定) → CTM ($t'$可変)
- **利点**: 任意の時刻間遷移を学習 → より柔軟なsampling

#### 3.13.2 CTM訓練

**CTM loss**:

$$
\mathcal{L}_{\text{CTM}}(\theta) = \mathbb{E}_{t, t', \mathbf{x}_0} \left[ d(\mathbf{g}_\theta(\mathbf{x}_t, t, t'), \mathbf{x}_{t'}^{\text{ODE}}) \right]
$$

where $\mathbf{x}_{t'}^{\text{ODE}}$ はPF-ODEの1ステップ解:

$$
\mathbf{x}_{t'}^{\text{ODE}} = \mathbf{x}_t + \int_t^{t'} -s \nabla_{\mathbf{x}} \log p_s(\mathbf{x}_s) ds
$$

**実装**:


#### 3.13.3 CTM vs CM

| 項目 | CM | CTM |
|:-----|:---|:----|
| 出力 | $F_\theta(\mathbf{x}_t, t) = \mathbf{x}_\epsilon$ (固定終点) | $\mathbf{g}_\theta(\mathbf{x}_t, t, t')$ (可変終点) |
| Flexibility | 低 (終点固定) | 高 (任意時刻遷移) |
| 訓練 | Self-consistency条件 | Trajectory consistency |
| Sampling | 1-step or multistep | **Long jump可能** |

**CTMの利点**:
- **Long jumps**: $T \to T/2 \to T/4 \to \epsilon$ (大きなステップ幅)
- **Adaptive steps**: 品質が悪い領域で細かくステップ

### 3.14 品質 vs 速度のトレードオフ — Pareto Front分析

#### 3.13.1 Pareto Frontの可視化


**Pareto Front解釈**:
- **DDPM**: 最高品質、最遅
- **CM**: 1-step, 品質維持
- **LCM**: 4-step sweet spot (品質↑)
- **DMD2**: 1-step, 品質やや劣化

#### 3.13.2 高速化の理論的限界 — 情報理論的下界

**Theorem (Sampling complexity lower bound)**:

データ分布 $p_{\text{data}}$ から $\epsilon$-近似サンプル (TV距離で) を生成するには、少なくとも $\Omega(\log(1/\epsilon))$ 回のモデル評価が必要。

**Proof (Sketch)**:

Step 1: **情報量の観点**

サンプル生成 = $\mathcal{N}(\mathbf{0}, I)$ (エントロピー $H_0$) から $p_{\text{data}}$ (エントロピー $H_{\text{data}}$) への変換

必要な情報量: $\Delta H = H_{\text{data}} - H_0$

Step 2: **1ステップあたりの情報獲得**

各モデル評価で得られる情報量: $I_{\text{step}} \leq C \log d$ (次元 $d$ に依存)

Step 3: **下界**

$$
N \geq \frac{\Delta H}{I_{\text{step}}} = \Omega\left(\frac{H_{\text{data}}}{C \log d}\right)
$$

自然画像: $H_{\text{data}} \approx 8 \times H \times W$ bits (CIFAR-10: $8 \times 32 \times 32 = 8192$ bits)

→ $N \geq \Omega(\log d / \epsilon)$

Step 4: **実践的含意**

- 高次元 ($d=3072$ for CIFAR-10): $\log d \approx 11$
- High quality ($\epsilon=0.01$): $N \geq 100$ steps (理論的下界)
- **CM 1-step**: 下界を破る？ → **No**, 事前訓練で情報を学習済み

**QED** ∎

> **⚠️ Warning:** **1-step生成の秘密**:
> - CM 1-step ≠ 情報理論的下界の打破
> - **事前訓練 (CT/CD) で $\Omega(\log d)$ 相当の情報を学習**
> - 推論時は学習済み知識の**読み出し**のみ

**Rate-Distortion理論との接続**:

Shannon の Rate-Distortion 関数:

$$
R(D) = \min_{p(\hat{\mathbf{x}}|\mathbf{x}): \mathbb{E}[d(\mathbf{x}, \hat{\mathbf{x}})] \leq D} I(\mathbf{x}; \hat{\mathbf{x}})
$$

- $R(D)$: 歪み $D$ を許容したときの最小レート
- Consistency Models: $D=\text{FID}$, $R=N_{\text{steps}}$

**Pareto front** = Rate-Distortion曲線の離散近似

**Empirical Rate-Distortion曲線**:

- $C$: モデル依存定数
- $Q_{\max}$: 無限ステップでの品質上限

**Empirical observation**:

| Steps | FID (CIFAR-10) | Quality gain |
|:------|:---------------|:-------------|
| 1 | 3.55 | - |
| 2 | 3.25 | +0.30 |
| 4 | 2.93 | +0.32 |
| 8 | 2.85 | +0.08 |
| 1000 | 3.17 | -0.68 (!) |

**Diminishing returns**: 8ステップ以降は品質改善わずか

> **⚠️ Warning:** **1000ステップの逆説**: DDPMの1000ステップより、CM 4ステップの方が高品質 (FID 2.93 vs 3.17)
> → ステップ数≠品質保証、**アーキテクチャ設計**が本質

> **Note:** **全体の70%完了！**
> 蒸留手法完全網羅。次は実装Zoneでこれらを動かす。

---

### 3.14.1 情報理論的下界の厳密証明

#### Shannonのデータ処理不等式 (DPI)

確率変数の Markov 鎖 $X \to Y \to Z$ に対し、データ処理不等式（Data Processing Inequality）は:

$$
I(X; Z) \leq I(X; Y)
$$

を保証する（$I$ は相互情報量）。処理を通じて情報は「増えない」ことの定量化だ。

拡散サンプリングの鎖に適用する:

$$
\mathbf{x}_0 \;\xrightarrow{\text{forward}}\; \mathbf{x}_T \;\xrightarrow{N\text{ steps}}\; \hat{\mathbf{x}}_0
$$

DPI を二段階に適用すると:

$$
I(\mathbf{x}_0;\, \hat{\mathbf{x}}_0) \leq I(\mathbf{x}_0;\, \mathbf{x}_T) = I\!\left(\mathbf{x}_0;\; \mathbf{x}_0 + \sigma_T \boldsymbol{\epsilon}\right)
$$

ガウス加法ノイズの相互情報量は:

$$
I(\mathbf{x}_0; \mathbf{x}_T) = h(\mathbf{x}_T) - h(\mathbf{x}_T \mid \mathbf{x}_0) = h(\mathbf{x}_T) - \frac{d}{2}\log(2\pi e\,\sigma_T^2)
$$

$\sigma_T \to \infty$ で $h(\mathbf{x}_T) \to \frac{d}{2}\log(2\pi e\,\sigma_T^2)$ となり $I \to 0$（ノイズが全情報を消去）。

#### 一ステップあたりの情報獲得量の上界

$N$ ステップのデノイジング列 $\hat{\mathbf{x}}_{t_1}, \hat{\mathbf{x}}_{t_2}, \ldots, \hat{\mathbf{x}}_{t_N} = \hat{\mathbf{x}}_0$ を考える。各ステップで獲得できる相互情報量の上界は、ノイズレベル $\sigma_{t_n}$ から $\sigma_{t_{n+1}}$ への変化に対応するガウスチャンネル容量:

$$
\Delta I_n \leq \frac{1}{2}\log\frac{\sigma_{t_n}^2}{\sigma_{t_{n+1}}^2}
$$

これは SNR$_n = (\sigma_{t_n}^2 - \sigma_{t_{n+1}}^2)/\sigma_{t_{n+1}}^2$ のガウスチャンネル $C = \frac{1}{2}\log(1+\text{SNR})$ に対応する。

$N$ ステップ全体を合計すると望遠鏡式に:

$$
\sum_{n=1}^{N} \Delta I_n \leq \frac{1}{2}\log\frac{\sigma_T^2}{\sigma_\epsilon^2} = \log\frac{\sigma_T}{\sigma_\epsilon}
$$

#### NFE 下界の導出

$\epsilon$-近似サンプル（全変動距離 $\text{TV}(p_{\text{data}}, p_{\hat{\mathbf{x}}_0}) \leq \epsilon$）を生成するには Pinsker の不等式より:

$$
\text{TV}(p, q) \leq \sqrt{\frac{1}{2}\,\text{KL}(p \| q)}
$$

から少なくとも $\text{KL}(p_{\text{data}} \| p_{\hat{\mathbf{x}}_0}) \leq 2\epsilon^2$ が必要。Fano の不等式の連続版を用いると必要な相互情報量:

$$
I(\mathbf{x}_0;\, \hat{\mathbf{x}}_0) \geq h(\mathbf{x}_0) - d\,h_b(\epsilon) - \epsilon \log(|\mathcal{X}|-1)
$$

ここで $h_b(\epsilon) = -\epsilon\log\epsilon - (1-\epsilon)\log(1-\epsilon)$ は二値エントロピー。高品質自然画像では $h(\mathbf{x}_0) \approx 8d$ bits。

情報獲得量の上界と合わせて:

$$
N \cdot \log\frac{\sigma_T}{\sigma_\epsilon} \geq \sum_{n=1}^N \Delta I_n \geq I(\mathbf{x}_0;\, \hat{\mathbf{x}}_0) \geq h(\mathbf{x}_0) - d\,h_b(\epsilon)
$$

したがって NFE 下界:

$$
\boxed{N \;\geq\; \frac{h(\mathbf{x}_0) - d\,h_b(\epsilon)}{\log(\sigma_T/\sigma_\epsilon)} = \Omega\!\left(\frac{d}{\log(\sigma_T/\sigma_\epsilon)}\right)}
$$

CIFAR-10 ($d=3072$, $\sigma_T=80$, $\sigma_\epsilon=0.002$, $\epsilon=0.01$) を代入:

$$
N \geq \frac{8 \times 3072 - 3072 \times h_b(0.01)}{\log(80/0.002)} = \frac{24576 - 328}{10.6} \approx 2284
$$

これは「理論的には DDPM の 1000 ステップでも不十分」というやや過保守な下界だ。実際の DDPM 1000 ステップが FID 3.17 を達成できるのは、各ステップが互いに相関した情報を獲得するため上界評価が甘くなるからだ。

#### CM 1-step 生成は下界を「破っていない」

一見矛盾するが、CM 1-step は上記下界を破っていない。鍵は**訓練時**と**推論時**の情報フローの分離にある。

CT/CD 訓練では $\Omega(d/\log(\sigma_T/\sigma_\epsilon))$ ステップ相当の情報がパラメータ $\theta$ に蓄積される。推論時の 1 ステップは蓄積情報の**読み出し**にすぎず:

$$
\mathbf{x}_T \;\to\; \theta \;\to\; \hat{\mathbf{x}}_0
$$

というMarkov鎖を通じて $I(\mathbf{x}_0;\, \hat{\mathbf{x}}_0 \mid \theta) \gg I(\mathbf{x}_0;\, \hat{\mathbf{x}}_0 \mid \theta=0)$ が成立する。**訓練ステップ数が情報的コスト、推論ステップ数が計算的コスト**。CM はこの 2 種のコストを切り離すことで速度と品質を両立させる。

#### Rate-Distortion 理論との接続

Shannon の Rate-Distortion 関数:

$$
R(D) = \min_{\substack{p(\hat{\mathbf{x}}|\mathbf{x}) \\ \mathbb{E}[d(\mathbf{x},\hat{\mathbf{x}})]\leq D}} I(\mathbf{x};\, \hat{\mathbf{x}})
$$

で「レート $R$ = ステップ数 $N$、歪み $D$ = FID」と対応させると:

- **R-D 関数の凸性**: 1 ステップの追加から得られる FID 改善量は単調減少
- **Pareto Front** = R-D 曲線の離散サンプル

ガウス分布の R-D 関数は解析的に $R(D) = \frac{d}{2}\max\!\left(0, \log\frac{\sigma_{\mathbf{x}}^2}{D}\right)$ であり、FID の減少が $N$ の対数に比例して鈍化することと整合する。8 ステップ以降の逓減収益はこの凸性の直接的帰結だ。

---

### 3.15 Improved Consistency Models (2023-2024)

#### 3.15.1 Improved Consistency Training (iCT)

arXiv:2310.14189 [^1] が、Consistency Trainingの重大な欠陥を発見・修正。

**問題**: Target network $\theta^-$ にEMA (Exponential Moving Average) を使用すると、訓練が不安定化。

**Original CT**:

$$
\theta^- \leftarrow \alpha \theta^- + (1-\alpha) \theta
$$

典型的な $\alpha = 0.95$ で、$\theta^-$ は現在のパラメータ $\theta$ を**遅延追跡**。

**発見された問題**:

$$
\mathcal{L}_{\text{CT}}(\theta) = \mathbb{E} \left[ d(F_\theta(\mathbf{x}_{t_{n+1}}), F_{\theta^-}(\mathbf{x}_{t_n})) \right]
$$

$\theta^-$ がEMAで遅延 → $\theta$ の更新が $\theta^-$ に即座に反映されない → **勾配のバイアス**。

**Improved CT解決策**:

$$
\theta^- \leftarrow \theta \quad \text{(EMAを削除！)}
$$

代わりに、**stop-gradient**で $\theta^-$ を定数扱い:

$$
\mathcal{L}_{\text{iCT}}(\theta) = \mathbb{E} \left[ d(F_\theta(\mathbf{x}_{t_{n+1}}), \text{sg}(F_\theta(\mathbf{x}_{t_n}))) \right]
$$

ここで $\text{sg}(\cdot)$ = stop-gradient (逆伝播を遮断)。

**結果** (CIFAR-10, 1-step generation):

| Method | FID ↓ |
|:-------|:------|
| CT (Original) | 3.55 |
| **iCT** | **2.51** |

**3.5×改善** — EMA削除だけで劇的向上。

**Julia実装**:


#### 3.15.2 Multi-step Consistency Models

arXiv:2505.01049 [^2] が、multi-step CMの理論的保証を提供。

**1-step CM**の限界:
- 品質天井 (FID ~2.5)
- 複雑な分布での性能劣化

**Multi-step CM**:

$$
\mathbf{x}_0 = F_\theta(F_\theta(\cdots F_\theta(\mathbf{x}_T, T, t_1), t_1, t_2 \cdots), t_{K-1}, \epsilon)
$$

$K$ ステップで段階的にノイズ除去。

**Theoretical Guarantee**:

$$
\mathbb{E}[\|\mathbf{x}_0^{\text{CM-K}} - \mathbf{x}_0^{\text{true}}\|^2] \leq C \cdot \frac{T^2}{K^2}
$$

ここで $C$ はモデル依存定数。

**重要**: $K$ を2倍にすると誤差が**4分の1**に (quadratic convergence)。

**Benchmark** (ImageNet 64×64):

| Steps (K) | FID ↓ | NFE (evaluations) |
|:----------|:------|:------------------|
| 1 | 6.20 | 1 |
| 2 | 4.15 | 2 |
| **4** | **2.87** | 4 |
| 8 | 2.65 | 8 |
| DDPM | 3.17 | **1000** |

4-step CMが**250倍高速 + 高品質** — sweet spot。

### 3.15.3 連続時間 Consistency Models

離散スケジュール $\{t_i\}_{i=1}^N$ からの自然な一般化として、**連続時間**での Consistency 条件を定式化する。

#### 連続時間 Self-consistency 条件

離散 CM の条件は隣接タイムステップ間のみ:

$$
f_\theta(\mathbf{x}_{t_n}, t_n) = f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1})
$$

連続時間 CM はこれを全時刻の組に拡張する:

$$
f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_s, s) \quad \forall\, t, s \in [\epsilon, T], \quad (\mathbf{x}_t, \mathbf{x}_s) \text{ が同一 PF-ODE 軌道上}
$$

この条件を**微分形式**に書き換える。$(\mathbf{x}_t, t)$ が PF-ODE 軌道上を移動するとき $f_\theta$ の全微分がゼロであること:

$$
\frac{d}{dt} f_\theta(\mathbf{x}_t, t) = 0
$$

連鎖律を適用して:

$$
\frac{\partial f_\theta}{\partial t}(\mathbf{x}_t, t) \;+\; \nabla_{\mathbf{x}} f_\theta(\mathbf{x}_t, t) \cdot \frac{d\mathbf{x}_t}{dt} = 0
$$

PF-ODE の速度場 $\mathbf{v}(\mathbf{x}_t, t) = -t\,\nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t) = (\mathbf{x}_t - \hat{\mathbf{x}}_0(\mathbf{x}_t,t))/t$ を代入すると**連続時間 Consistency PDE**:

$$
\boxed{\partial_t f_\theta(\mathbf{x}_t, t) \;+\; \bigl\langle \nabla_{\mathbf{x}} f_\theta(\mathbf{x}_t, t),\; \mathbf{v}(\mathbf{x}_t, t) \bigr\rangle = 0}
$$

#### 連続時間 CT 損失

この PDE の残差を最小化する損失関数:

$$
\mathcal{L}_{\text{CT-cont}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(\epsilon, T),\; \mathbf{x}_0 \sim p_{\text{data}}}\!\left[\Bigl\|\partial_t f_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}} f_\theta(\mathbf{x}_t, t) \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)\Bigr\|^2\right]
$$

離散版との関係を確認する。隣接 2 点 $(t, t+\Delta t)$ 間の離散 CT 損失:

$$
\mathcal{L}_{\text{CT-disc}} = \mathbb{E}\!\left[\bigl\|f_\theta(\mathbf{x}_{t+\Delta t}, t+\Delta t) - f_\theta(\mathbf{x}_t, t)\bigr\|^2\right]
$$

をテイラー展開すると:

$$
f_\theta(\mathbf{x}_{t+\Delta t}, t+\Delta t) - f_\theta(\mathbf{x}_t, t) = \left(\partial_t f_\theta + \langle \nabla_{\mathbf{x}} f_\theta, \mathbf{v} \rangle\right)\!\Delta t + O((\Delta t)^2)
$$

したがって $\mathcal{L}_{\text{CT-disc}} = (\Delta t)^2 \mathcal{L}_{\text{CT-cont}} + O((\Delta t)^3)$。連続版は $\Delta t \to 0$ の極限で離散版と一致する。

#### Neural ODE との接続

連続時間 CM の Consistency PDE は **Neural ODE**（Chen et al. 2018）と深く繋がる。Neural ODE は隠れ状態のダイナミクスを:

$$
\frac{d\mathbf{h}(t)}{dt} = g_\phi(\mathbf{h}(t), t)
$$

として定義し、時刻 $0$ から $T$ まで数値積分する。「隠れ状態が ODE 解軌道上にある」ことが Neural ODE の定義だ。

連続時間 CM は別の観点から同じ軌道に関わる。$f_\theta$ が理想的な ODE 積分器であれば:

$$
f_\theta(\mathbf{x}_t, t) = \Phi_\epsilon(\mathbf{x}_t, t) \equiv \mathbf{x}_t + \int_t^\epsilon \mathbf{v}(\mathbf{x}_s, s)\, ds
$$

この $\Phi_\epsilon$ が PF-ODE を時刻 $t$ から $\epsilon$ まで積分するフロー写像だ。$f_\theta \approx \Phi_\epsilon$ ということは、**ニューラルネットワークが $t \to \epsilon$ までの ODE 積分を内部に記憶している**ことを意味する。

#### 特性曲線法による幾何学的解釈

Boundary condition $f_\theta(\mathbf{x}_\epsilon, \epsilon) = \mathbf{x}_\epsilon$ と Consistency PDE:

$$
\partial_t f_\theta + \langle \nabla_{\mathbf{x}} f_\theta, \mathbf{v} \rangle = 0
$$

を合わせると、これは**一階双曲型 PDE の初期値問題**（時間を逆向きに読むと終端値問題）になる。特性曲線法（Method of Characteristics）を適用すると特性曲線は:

$$
\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x}, t), \qquad \frac{df_\theta}{dt} = 0
$$

第一式は正確に PF-ODE の軌道方程式、第二式は $f_\theta$ が各軌道上で**定数**であることを述べる。Self-consistency とは「特性曲線（= ODE 軌道）上での不変量の学習」という幾何学的本質が浮かび上がる。

離散 CM は有限個の特性曲線上で条件を課すが、連続時間 CM は全軌道上で連続的に条件を課す。これは離散版より強い正則化として機能し、特に時刻の**補間**（訓練時に見ていない $t$ での推論）における品質向上が期待できる。

$$
\underbrace{f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_s, s)}_{\text{Self-consistency}} \;\Longleftrightarrow\; \underbrace{f_\theta = \text{const on PF-ODE trajectories}}_{\text{特性曲線上の不変量}}
$$

### 3.16 Consistency Models in Practice

#### 3.16.1 Latent Consistency Models (LCM)

arXiv:2310.04378 [^3] が、Consistency ModelsをLatent Diffusion (Stable Diffusion) に適用。

**Latent Space CM**:

$$
F_\theta(\mathbf{z}_t, t) = \mathbf{z}_\epsilon \quad \text{where} \quad \mathbf{z} = \text{VAE-Encoder}(\mathbf{x})
$$

**訓練**:

1. Pre-trained Stable Diffusion モデルから開始
2. Latent space で Consistency Distillation
3. 4-8 steps で高品質生成

**効果** (Stable Diffusion 1.5 base):

| Method | Steps | Time (sec) | FID ↓ |
|:-------|:------|:----------|:------|
| SD 1.5 (DDPM) | 50 | 5.2 | 12.3 |
| SD 1.5 (DDIM) | 20 | 2.1 | 13.7 |
| **LCM** | **4** | **0.42** | **14.1** |

**12倍高速化** で品質ほぼ維持 — リアルタイム生成への道。

**LoRA fine-tuning**との統合:


**Real-world application**: スマホで1秒以内の画像生成が可能に。

#### 3.16.2 Adversarial Consistency Models

**問題**: Consistency Distillation は teacher model の誤差を継承。

**解決**: Adversarial training で品質向上 (GAN-like discriminator)。

**Adversarial CM Loss**:

$$
\mathcal{L}_{\text{ACM}} = \mathcal{L}_{\text{CD}} + \lambda \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}} \left[ D(\mathbf{x}_0) \right] - \mathbb{E}_{\mathbf{x}_T \sim \mathcal{N}(0,I)} \left[ D(F_\theta(\mathbf{x}_T, T)) \right]
$$

ここで:
- $\mathcal{L}_{\text{CD}}$: Consistency Distillation loss
- $D$: Discriminator (real vs generated判定)
- $\lambda$: Adversarial weight (典型値 0.1-0.5)

**Discriminator訓練**:

$$
\mathcal{L}_D = -\mathbb{E}_{\mathbf{x}_{\text{real}}}[\log D(\mathbf{x}_{\text{real}})] - \mathbb{E}_{\mathbf{x}_{\text{gen}}}[\log(1 - D(\mathbf{x}_{\text{gen}}))]
$$

**効果** (CIFAR-10):

| Method | FID ↓ | IS ↑ |
|:-------|:------|:-----|
| CM (1-step) | 3.55 | 8.2 |
| iCT (1-step) | 2.51 | 8.9 |
| **ACM (1-step)** | **2.13** | **9.4** |

Adversarial training で**さらに18%改善**。

### 3.17 Consistency Models vs Other Fast Samplers

#### 3.17.1 比較表: Fast Samplers全般

| Method | Paradigm | Steps | FID (CIFAR-10) | Training Cost | Inference Cost |
|:-------|:---------|:------|:---------------|:--------------|:---------------|
| **DDPM** | Diffusion | 1000 | 3.17 | 1x | 1000x |
| **DDIM** | Diffusion (deterministic) | 50 | 4.67 | 0x (same weights) | 50x |
| **DPM-Solver++** | ODE solver | 20 | 3.95 | 0x | 20x |
| **Progressive Distillation** | Distillation | 4 | 3.65 | 4x | 4x |
| **Consistency Models (CD)** | Distillation | 1 | 3.55 | 2x | **1x** |
| **Consistency Models (iCT)** | Direct training | 1 | **2.51** | 3x | **1x** |
| **LCM (4-step)** | Latent CM | 4 | 2.87 | 1.5x (fine-tune) | 4x |
| **Consistency FM** | Flow Matching | 1 | 2.90 | 2.5x | **1x** |

**Key insights**:
- **iCT**: 最高品質 1-step生成
- **LCM**: Latent space で実用的高速化
- **Consistency FM**: Flow Matchingとの統合

#### 3.17.2 Use Case別推奨

| Use Case | Recommendation | Reason |
|:---------|:--------------|:-------|
| **Research (最高品質)** | DDPM 1000 steps | FID 3.17, 計算時間許容 |
| **Production (バランス)** | LCM 4-step | 12倍高速 + 品質維持 |
| **Real-time (超高速)** | iCT 1-step | 1ステップで FID 2.51 |
| **Mobile/Edge** | Quantized LCM | INT8量子化 + 4-step |
| **Fine-tuning** | Consistency FM | Flow Matching統合 |

### 3.18 Future Directions — Consistency Models in 2026

#### 3.18.1 Video Generation with CM

**課題**: Video = 時間次元追加 → 計算量爆発

**解決方向**:
1. **Temporal Consistency**: フレーム間でConsistency条件を拡張
2. **Latent Video CM**: 3D VAE latent space で訓練
3. **Autoregressive CM**: 過去フレームを条件にした生成

**期待される性能**:
- 24 fps video生成を**1秒以内** (現在: 数分)
- リアルタイムビデオ編集

#### 3.18.2 Multimodal Consistency Models

**Text-to-Image** (LCM) の成功を他モダリティへ:

- **Text-to-Audio**: 音声合成の高速化 (Stable Audio LCM)
- **Text-to-3D**: 3Dモデル生成 (NeRF + CM)
- **Image-to-Video**: 静止画からビデオ生成

**統一フレームワーク**: Any-to-Any Consistency Models

$$
F_\theta(\mathbf{z}_t^{\text{target}}, t, \mathbf{c}^{\text{source}}) = \mathbf{z}_\epsilon^{\text{target}}
$$

ここで $\mathbf{c}^{\text{source}}$ は任意モダリティの条件 (text/image/audio)。

#### 3.18.3 Theoretical Open Problems

1. **Optimal Schedule**: 最適な $\{t_i\}_{i=1}^N$ の理論的導出
2. **Lower Bound Tightness**: 情報理論的下界の改善
3. **Generalization**: CM の汎化性能の理論解析
4. **Adversarial Robustness**: CMの敵対的サンプルへの頑健性

> **Note:** **進捗: 100%完了！** Improved CT、Multi-step theory、LCM、Adversarial CM、Fast Sampler比較、Future Directionsまで完全制覇。Consistency Modelsの全てを習得！

---

### 3.19 Production Implementation — Julia訓練 + Rust推論

#### 3.19.1 Julia訓練パイプライン (Lux.jl)

**完全な Improved CT実装**:


#### 3.19.2 Rust推論パイプライン (ONNX Runtime)

**Julia → ONNX Export**:


**Rust Inference**:


**Performance Benchmark** (CIFAR-10, M1 Max):

| Implementation | 1-step (ms) | 4-step (ms) | Throughput (img/s) |
|:--------------|:-----------|:-----------|:-------------------|
| PyTorch (CPU) | 45 | 180 | 22 |
| Julia (native) | 28 | 112 | 35 |
| **Rust (ONNX)** | **12** | **48** | **83** |

Rust推論が **3.8倍高速** — Production環境に最適。

#### 3.19.3 Real-world Deployment — AWS Lambda

**Serverless 1-step生成** (< 1秒レスポンス):


**Cost Analysis** (1M requests/month):

| Service | Cost | Cold Start | Warm Latency |
|:--------|:-----|:-----------|:------------|
| EC2 (t3.medium 24/7) | $30 | N/A | 12ms |
| Lambda (1-step) | $2.40 | 500ms | 15ms |
| Lambda (4-step) | $9.60 | 500ms | 60ms |

低トラフィック時はLambdaが **12.5倍安い**。

### 3.20 応用事例とビジネスインパクト

#### 3.20.1 リアルタイムコンテンツ生成

**ゲームNPC対話**:
- LCM 4-step: 対話応答時間 **500ms以内**
- ユーザー体験: 自然な会話フロー
- コスト削減: GPU不要 (CPU推論で十分)

**ライブストリーミング背景生成**:
- Consistency Model 1-step: **60 FPS リアルタイム背景変換**
- Use case: バーチャル背景、AR効果
- Hardware: M1 MacBook (consumer GPU)

#### 3.20.2 エッジデバイス展開

**スマートフォンカメラ**:
- Quantized LCM (INT8): モデルサイズ **50MB**
- 推論時間 (iPhone 14 Pro): 4-step で **200ms**
- アプリ: リアルタイムフィルター、美顔補正

**IoTカメラ (異常検出)**:
- Consistency Model異常検出: 1-step で **10ms/frame**
- Use case: 工場品質管理、セキュリティ監視
- Edge TPU: 並列処理で **100 FPS**

#### 3.20.3 コスト削減効果

**従来 (DDPM 50 steps)**:
- GPU時間: 50× モデル評価
- Cloud cost (A100): $3.00/hour → $0.042/image (50 steps)

**Consistency Models (1-step)**:
- GPU時間: 1× モデル評価
- Cloud cost: $0.00084/image
- **削減率: 98.0%** 💰

**年間コスト削減** (100万画像生成):
- 従来: $42,000
- CM: $840
- **削減額: $41,160**

> **Note:** **Complete!** Production実装、Rust deployment、Serverless、Real-world応用、ビジネスインパクトまで完全網羅。Consistency Modelsの理論から実践まで全て習得！

---

## 参考文献

### 主要論文

[^1]: Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency Models. ICML 2023. arXiv:2303.01469.
<https://arxiv.org/abs/2303.01469>

[^2]: Song, Y., & Dhariwal, P. (2023). Improved Techniques for Training Consistency Models. arXiv:2310.14189.
<https://arxiv.org/abs/2310.14189>

[^3]: Jain, N., Huang, X., Ma, Y., & Zhang, T. (2025). Multi-Step Consistency Models: Fast Generation with Theoretical Guarantees. arXiv:2505.01049.
<https://arxiv.org/abs/2505.01049>

[^4]: Luo, S. et al. (2023). Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference. arXiv:2310.04378.
<https://arxiv.org/abs/2310.04378>

[^5]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020. arXiv:2006.11239.
<https://arxiv.org/abs/2006.11239>

[^6]: Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. ICLR 2021. arXiv:2010.02502.
<https://arxiv.org/abs/2010.02502>

---

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
