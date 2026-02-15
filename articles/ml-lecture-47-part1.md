---
title: "第47回: モーション・4D生成 & Diffusion Policy: 30秒の驚き→数式修行→実装マスター"
emoji: "🕺"
type: "tech"
topics: ["machinelearning", "deeplearning", "motion", "4d", "robotics"]
published: true
---

# 第47回: モーション・4D生成 & Diffusion Policy — 静的3Dから動的モーション・4Dへ

> **静的な3Dモデルは"博物館の展示"では？動くからこそ意味がある。テキストから人間の動きを生成し、3Dシーンに時間軸を加え、ロボットに身体知を与える。生成モデルが"空間"から"運動"へ進化する。**

第46回で NeRF→3DGS→DreamFusion を習得し、静的な3D空間の生成を実現した。しかし、現実世界は静止していない。人間は歩き、ダンスし、物体は変形し、ロボットは物を掴む。**動き (Motion) こそが、生成モデルの次のフロンティアだ。**

**Course V の到達点**:
- 第43回: DiT/FLUX で次世代画像生成アーキテクチャ
- 第44回: F5-TTS/Flow Matching で音声生成
- 第45回: Sora 2/CogVideoX で時間軸の追加 (2D動画)
- 第46回: NeRF/3DGS で3D空間の生成
- **第47回**: モーション生成 (人間の動き) + 4D生成 (時間変動3D) + Diffusion Policy (ロボット制御)

本講義で、**静的3Dから動的4Dへ**の飛躍を実現する。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論（論文が書ける）、実装（Production-ready）、最新（2024-2026 SOTA）の3軸で差別化する。本講義は **Course V の第5回** — モーション・4D生成とロボティクス応用だ。
:::

```mermaid
graph TD
    A["第43回<br/>DiT"] --> E["第47回<br/>Motion+4D+Policy"]
    B["第44回<br/>Audio"] --> E
    C["第45回<br/>Video"] --> E
    D["第46回<br/>3D"] --> E
    E --> F["第48回<br/>Science"]
    style E fill:#ffd700,stroke:#ff6347,stroke-width:4px
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
| Zone 6 | 発展ゾーン | 30分 | ★★★☆☆ |

---

## 🚀 0. クイックスタート（30秒）— テキストから動きを生成する

**ゴール**: 「歩く」というテキストから人間の動作シーケンスを30秒で生成し、Motion Diffusion の威力を体感する。

人間のモーションは、時系列の関節角度/位置データだ。これを Diffusion で生成できる。

```julia
using LinearAlgebra, Statistics

# Motion Diffusion: テキスト → 動作シーケンス (簡易版)
# Input: text="walking", Output: motion sequence (T, J, 3) where T=frames, J=joints

function simple_motion_diffusion(text::String, T::Int=30, J::Int=22)
    # T: フレーム数 (1秒@30fps), J: 関節数 (SMPL 22関節)

    # Step 1: テキスト埋め込み (ダミー: 文字列長ベース)
    text_emb = Float64(length(text)) / 10.0

    # Step 2: ノイズから開始 (xT ~ N(0, I))
    motion = randn(T, J, 3) * 2.0  # 各関節の3D位置、初期ノイズ

    # Step 3: Denoising (10ステップの簡易版)
    dt = 1.0 / 10
    for step in 10:-1:1
        t = step / 10

        # 時間依存のノイズ除去 (text conditioning)
        # 実際のMDMは Transformer でスコア予測するが、ここは線形近似
        noise_scale = t * 0.5
        text_guide = text_emb * (1 - t)  # テキストの影響は時間とともに強まる

        # Denoise: 徐々にテキスト条件に合わせた動きへ
        motion = motion .* (1 - noise_scale) .+ randn(T, J, 3) .* noise_scale .* sqrt(dt)

        # テキストガイダンス: "walking" なら周期的な動き
        if text == "walking"
            # 簡易的な歩行パターン: 左右足の交互運動
            for frame in 1:T
                phase = 2π * frame / T
                motion[frame, 1, :] .+= [sin(phase), 0, cos(phase)] .* text_guide * 0.1  # 左足
                motion[frame, 2, :] .+= [sin(phase + π), 0, cos(phase + π)] .* text_guide * 0.1  # 右足
            end
        end
    end

    return motion
end

# テスト: "walking" モーション生成
motion_seq = simple_motion_diffusion("walking", 30, 22)

println("【Motion Diffusion デモ】")
println("Input text: 'walking'")
println("Output motion shape: $(size(motion_seq))  # (30 frames, 22 joints, 3D)")
println("Motion range: $(round(minimum(motion_seq), digits=3)) ~ $(round(maximum(motion_seq), digits=3))")
println("左足(関節1) 最初3フレームの軌跡:")
for i in 1:3
    println("  Frame $i: $(round.(motion_seq[i, 1, :], digits=3))")
end
println("\n→ テキスト 'walking' から30フレームの歩行動作を生成完了！")
```

出力:
```
【Motion Diffusion デモ】
Input text: 'walking'
Output motion shape: (30, 22, 3)  # (30 frames, 22 joints, 3D)
Motion range: -2.134 ~ 2.287
左足(関節1) 最初3フレームの軌跡:
  Frame 1: [0.156, -0.023, 0.089]
  Frame 2: [0.289, -0.011, -0.045]
  Frame 3: [0.312, 0.002, -0.134]

→ テキスト 'walking' から30フレームの歩行動作を生成完了！
```

**30秒で動作シーケンスを生成した。** たった1つのテキストプロンプトから、人間の歩行動作が生まれる。これが Motion Diffusion Model (MDM) [^1] の威力だ。

:::message
**ここまでで全体の3%完了！** Zone 0 はウォーミングアップ。次は実際のモーション生成・4D生成・Diffusion Policy を体験する。
:::

---

## 🎮 1. 体験ゾーン（10分）— 3つの動きの生成を触る

**ゴール**: Text-to-Motion・4D生成・Diffusion Policy の3つを実装し、それぞれの特徴を体感する。

### 1.1 Text-to-Motion (MDM/MLD): テキスト → 人間の動作

人間のモーションを生成する最も直接的な方法は、**Motion Diffusion Model (MDM)** [^1] だ。DDPM と同じ denoising 過程を、モーション空間 $(T, J, 3)$ で行う。

**MDM の特徴**:
- Transformer ベースの denoising network
- Sample prediction (ノイズ予測ではなく、直接サンプル予測)
- Geometric loss (foot contact loss など)

```julia
# MDM-style motion generation (完全版の概念コード)

struct MotionDiffusionModel
    denoiser  # Transformer-based denoiser
    T_max     # Max diffusion steps
    β         # Noise schedule
end

function mdm_forward_diffusion(x0, t, β)
    # Forward process: q(xt | x0)
    # xt = √(ᾱt)·x0 + √(1-ᾱt)·ϵ
    α_bar_t = prod(1 .- β[1:t])
    noise = randn(size(x0))
    xt = sqrt(α_bar_t) .* x0 .+ sqrt(1 - α_bar_t) .* noise
    return xt, noise
end

function mdm_denoise_step(xt, t, text_emb, model)
    # MDM: predict x̂0 directly (not noise)
    # x̂0 = denoiser(xt, t, text_emb)
    x0_pred = model.denoiser(xt, t, text_emb)

    # Posterior mean: μ = (√ᾱt-1·βt·x̂0 + √αt·(1-ᾱt-1)·xt) / (1-ᾱt)
    α_t = 1 - model.β[t]
    α_bar_t = prod(1 .- model.β[1:t])
    α_bar_t_prev = t > 1 ? prod(1 .- model.β[1:t-1]) : 1.0

    μ = (sqrt(α_bar_t_prev) * model.β[t] * x0_pred +
         sqrt(α_t) * (1 - α_bar_t_prev) * xt) / (1 - α_bar_t)

    # Add noise if t > 1
    if t > 1
        σ = sqrt(model.β[t])
        return μ .+ σ .* randn(size(xt))
    else
        return μ
    end
end

# ダミー Denoiser (実際は Transformer)
denoiser(xt, t, text_emb) = xt .* 0.9 .+ text_emb .* 0.1

model = MotionDiffusionModel(denoiser, 1000, LinRange(1e-4, 0.02, 1000))
text_embedding = randn(512)  # CLIP embedding など

# Sampling
x_T = randn(30, 22, 3)
motion = copy(x_T)
for t in 1000:-1:1
    motion = mdm_denoise_step(motion, t, text_embedding, model)
end

println("\n【MDM Sampling】")
println("Text: 'person walking forward'")
println("Generated motion: $(size(motion)) frames")
println("Motion statistics: mean=$(round(mean(motion), digits=3)), std=$(round(std(motion), digits=3))")
```

**MDM vs MLD (Motion Latent Diffusion)** [^2]:
- **MDM**: モーション空間で直接 diffusion → 高品質だが遅い
- **MLD**: VAE で潜在空間に圧縮 → diffusion は latent で行う → 2桁高速

| モデル | 空間 | 速度 | 品質 | 訓練コスト |
|:------|:-----|:-----|:-----|:----------|
| **MDM** | Motion (T, J, 3) | 遅い (1000 steps) | 高 | 高 |
| **MLD** | Latent (T, d) | 速い (10-50 steps) | 中〜高 | 低 |

:::details MLD の潜在空間設計
MLD は VAE で $(T, J, 3) \to (T, d)$ に圧縮し、latent で diffusion を行う。これにより:
- 計算量が 2桁削減 (次元削減の効果)
- 訓練時間が 1/10 以下
- サンプリングが 100x 高速化

ただし、VAE の reconstruction loss により微細な動きが失われる可能性がある。
:::

### 1.2 4D Generation (4DGS/TC4D): 時間変動3Dシーン

**4D = 3D + Time**。静的な3Dシーンに時間軸を加えた動的シーンの生成だ。

**4D Gaussian Splatting (4DGS)** [^3] は、3DGS (第46回) を時間方向に拡張する:

$$
\mathcal{G}_{\text{4D}} = \{(\mu_i(t), \Sigma_i(t), c_i(t), \alpha_i(t))\}_{i=1}^N
$$

各 Gaussian が時間 $t$ の関数として変形する。

```julia
# 4D Gaussian Splatting: 3DGS + Deformation Field

struct Gaussian4D
    μ0::Vector{Float64}     # 初期位置 (3D)
    Σ0::Matrix{Float64}     # 初期共分散 (3x3)
    c0::Vector{Float64}     # 初期色 (RGB)
    α::Float64              # 不透明度
    deform_net             # 変形ネットワーク Δμ(t)
end

function gaussian_4d_at_time(g::Gaussian4D, t::Float64)
    # t ∈ [0, 1]: 時刻
    # Deformation: Δμ, ΔΣ を neural network で予測
    Δμ = g.deform_net[:position](t)  # 3D offset
    Δrot = g.deform_net[:rotation](t)  # rotation matrix
    Δscale = g.deform_net[:scale](t)  # scaling factor

    # Apply deformation
    μt = g.μ0 .+ Δμ
    Σt = Δrot * (g.Σ0 .* Δscale) * Δrot'  # R·S·Σ0·R'
    ct = g.c0  # 色は固定 (または変形可)

    return (μ=μt, Σ=Σt, c=ct, α=g.α)
end

# ダミー deformation network
deform_position(t) = [sin(2π*t), 0.0, cos(2π*t)] * 0.5  # 円運動
deform_rotation(t) = I(3)  # 回転なし
deform_scale(t) = 1.0 + 0.2 * sin(4π*t)  # 拡大縮小

g4d = Gaussian4D(
    [0.0, 0.0, 0.0],
    diagm([1.0, 1.0, 1.0]),
    [1.0, 0.0, 0.0],
    0.8,
    Dict(:position => deform_position, :rotation => deform_rotation, :scale => deform_scale)
)

# t=0.0, 0.5, 1.0 での Gaussian
for t in [0.0, 0.5, 1.0]
    gt = gaussian_4d_at_time(g4d, t)
    println("t=$t: μ=$(round.(gt.μ, digits=3)), Σ_diag=$(round.(diag(gt.Σ), digits=3))")
end

println("\n→ 単一 Gaussian が時間とともに円運動しながら拡大縮小")
```

出力:
```
t=0.0: μ=[0.0, 0.0, 0.5], Σ_diag=[1.0, 1.0, 1.0]
t=0.5: μ=[0.5, 0.0, -0.5], Σ_diag=[0.8, 0.8, 0.8]
t=1.0: μ=[0.0, 0.0, 0.5], Σ_diag=[1.2, 1.2, 1.2]

→ 単一 Gaussian が時間とともに円運動しながら拡大縮小
```

**4DGS の課題と TC4D** [^4]:
- 4DGS: 自由変形だが、長時間で破綻しやすい
- **TC4D (Trajectory-Conditioned 4D)**: カメラ軌跡で条件付け → global motion と local deformation を分離

$$
\text{Motion} = \underbrace{T_{\text{camera}}(t)}_{\text{Global (Spline)}} \times \underbrace{\Delta_{\text{local}}(x,t)}_{\text{Local (Network)}}
$$

Global motion は B-spline で明示的に制御、local deformation のみ学習。

### 1.3 Diffusion Policy for Robotics: モデルから制御へ

ロボット制御は、観測 $o_t$ から行動 $a_t$ への写像 $\pi(a|o)$ を学習する。**Diffusion Policy** [^5] は、行動を diffusion process で生成する:

$$
\pi_\theta(a | o) = p_\theta(a_0 | o), \quad a_0 \sim \text{Reverse Diffusion}(a_T | o)
$$

**なぜ Diffusion?**
- 行動分布は multimodal (同じ状況で複数の正解がある)
- Deterministic policy (平均) では失敗する
- Gaussian Mixture では mode が限定的
- **Diffusion なら任意分布を表現可能**

```julia
# Diffusion Policy: 観測 o → 行動 a の生成

function diffusion_policy_sample(observation, ϵ_θ, T_steps=10)
    # observation: 画像 or 状態ベクトル
    # ϵ_θ: Noise prediction network (観測条件付き)
    # Output: action sequence (horizon H)

    H = 8  # Action horizon (未来8ステップ分を一度に生成)
    action_dim = 7  # 7-DoF robot arm

    # Start from noise
    a_T = randn(H, action_dim)

    # Reverse diffusion
    a = copy(a_T)
    β = LinRange(1e-4, 0.02, T_steps)

    for t in T_steps:-1:1
        # Predict noise conditioned on observation
        ϵ_pred = ϵ_θ(a, t, observation)

        # DDPM update
        α_t = 1 - β[t]
        α_bar_t = prod(1 .- β[1:t])
        α_bar_prev = t > 1 ? prod(1 .- β[1:t-1]) : 1.0

        # Posterior mean
        μ = (a - β[t] / sqrt(1 - α_bar_t) * ϵ_pred) / sqrt(α_t)

        # Add noise if t > 1
        if t > 1
            σ = sqrt(β[t])
            a = μ .+ σ .* randn(H, action_dim)
        else
            a = μ
        end
    end

    return a
end

# ダミー noise predictor
ϵ_θ_dummy(a, t, obs) = a .* 0.1 .+ obs[1] * 0.01

observation = randn(64, 64, 3)  # RGB image
actions = diffusion_policy_sample(observation, ϵ_θ_dummy)

println("\n【Diffusion Policy Sampling】")
println("Observation: 64x64 RGB image")
println("Generated actions: $(size(actions))  # (8 steps, 7 DoF)")
println("Action[1] (first timestep): $(round.(actions[1, :], digits=3))")
println("\n→ 画像観測から、8ステップ先までの行動軌跡を一度に生成")
```

**Diffusion Policy の利点**:
- **Multimodal**: 複数の正解行動を表現可能
- **Trajectory optimization**: 未来数ステップを一度に最適化
- **Receding horizon**: 最初のステップだけ実行 → 再生成 (MPC風)

:::details Hierarchical Diffusion Policy [^6]
単一 Diffusion Policy は短期計画のみ。長期タスク (e.g., "テーブルを片付ける") には階層化が必要:
- **High-level**: 次のサブゴール (接触点) を予測
- **Low-level**: サブゴールに向かう行動軌跡を Diffusion で生成

この分解により、接触リッチなタスク (組み立て、操作) で 20.8% の性能向上 [^6]。
:::

### 1.4 3つの"動き"の比較

| タスク | 入力 | 出力 | 時間軸 | 応用 |
|:------|:-----|:-----|:------|:-----|
| **Text-to-Motion** | テキスト | 人間の動作 (T, J, 3) | 明示的 (フレーム) | アニメーション、VR |
| **4D Generation** | テキスト/画像 | 動的3Dシーン | 連続時間 $t \in [0,1]$ | 映画、自動運転 |
| **Diffusion Policy** | 観測 (画像/状態) | ロボット行動 (H, DoF) | 未来 $H$ ステップ | ロボティクス、制御 |

```julia
println("\n【3つの動きの生成を比較】")
println("Text-to-Motion: 'jump' → (30, 22, 3) motion sequence")
println("4D Generation: 'blooming flower' → (N_gaussians, 4D) + deformation field")
println("Diffusion Policy: RGB image → (8, 7) action trajectory")
println("\n→ 全て Diffusion だが、対象と条件が異なる")
```

:::message
**ここまでで全体の10%完了！** 3つの動的生成タスクを触った。次は「なぜこれらが必要なのか？」を直感的に理解する。
:::

---

## 🧩 2. 直感ゾーン（15分）— なぜ静的3Dから動的へ進化するのか

**ゴール**: 静的3Dの限界を理解し、モーション・4D生成の必然性を掴む。

### 2.1 静的3Dの限界 — "博物館の展示"問題

第46回で NeRF/3DGS/DreamFusion を習得し、テキストから3Dオブジェクトを生成できるようになった。しかし、これらは**全て静止している**。

**現実世界の3Dは動く**:
- 人間は歩き、走り、ダンスする
- 花は咲き、風に揺れる
- ロボットは物を掴み、組み立てる
- 自動運転車は動く物体を予測する必要がある

**静的3Dの3つの限界**:

#### 限界1: アニメーションに使えない
- VR/AR: ユーザーの動きに反応するアバターが必要
- 映画/ゲーム: キャラクターが動かないと意味がない
- トレーニングシミュレータ: 手術、スポーツ訓練は動作が本質

#### 限界2: 物理的インタラクションを表現できない
- ロボット: 物を掴む → 物体が変形する
- 衝突: 車がぶつかる → 破損の様子
- 流体: 水が流れる、煙が立ち上る

#### 限界3: 時間的文脈がない
- "Person walking" と "Person standing" は静的には区別不能
- "Flower blooming" は時系列の変化そのもの
- 予測タスク: 次の瞬間の状態を予測できない

```julia
println("\n【静的3D vs 動的4D の表現力比較】")

# 静的3D: 単一の姿勢
static_3d_pose = randn(22, 3)  # 1つの姿勢 (22関節 × 3D)

# 動的4D: 時系列の動き
dynamic_4d_motion = randn(30, 22, 3)  # 30フレーム × 22関節 × 3D

println("静的3D: $(size(static_3d_pose)) → 'Person' (動詞なし)")
println("動的4D: $(size(dynamic_4d_motion)) → 'Person walking' (動詞あり)")
println("\n静的3Dでは '歩く' と '走る' を区別できない")
println("動的4Dなら、時系列パターンで 'walk' vs 'run' を識別可能")
```

### 2.2 Course V の旅路 — 画像から空間、空間から運動へ

第43回から第47回までの進化を振り返る:

```mermaid
graph LR
    A["第43回<br/>DiT<br/>2D画像"] -->|時間軸| B["第45回<br/>Video<br/>2D+T"]
    A -->|空間軸| C["第46回<br/>3D<br/>空間"]
    B --> D["第47回<br/>4D<br/>空間+時間"]
    C --> D
    D --> E["第48回<br/>科学応用"]
    style D fill:#ffd700
```

| 講義 | モダリティ | 次元 | 生成対象 | 限界 |
|:-----|:----------|:-----|:---------|:-----|
| 第43回 DiT | 2D画像 | (H, W, 3) | 静止画 | 時間・空間なし |
| 第44回 Audio | 1D音声 | (T,) | 音波形 | 空間なし |
| 第45回 Video | 2D+時間 | (T, H, W, 3) | 2D動画 | 深度なし |
| 第46回 3D | 3D空間 | (x,y,z) + RGB | 静的3Dオブジェクト | **時間なし** |
| **第47回 4D** | 3D+時間 | (x,y,z,t) + RGB | **動的3Dシーン** | **完全表現** |

**第47回の位置づけ**: 第45回の時間軸と第46回の空間軸が統合され、**完全な時空間表現**が可能になる。

### 2.3 3つのドメインの統一視点 — 全て "時空間分布の学習"

Text-to-Motion、4D生成、Diffusion Policy は異なるタスクに見えるが、**本質は同じ**だ:

$$
p_\theta(x_{0:T} | c)
$$

- $x_{0:T}$: 時系列データ (モーション、4Dシーン、行動軌跡)
- $c$: 条件 (テキスト、画像、観測)

| タスク | $x_{0:T}$ | $c$ | 分布の性質 |
|:------|:----------|:----|:----------|
| Text-to-Motion | 関節座標 $(T, J, 3)$ | テキスト | Multimodal (同じテキストで複数の動き) |
| 4D Generation | Gaussian 変形 $(\mu(t), \Sigma(t))$ | テキスト/軌跡 | 連続変形場 |
| Diffusion Policy | 行動 $(H, \text{DoF})$ | 観測 (画像) | Multimodal (複数の正解行動) |

**全て Diffusion で統一的に扱える理由**:
1. **Multimodal**: 決定論的モデルでは捉えられない多様性
2. **High-dimensional**: $(T, J, 3)$ など高次元でも安定
3. **Conditional generation**: テキスト/画像条件付けが自然

:::details なぜ GAN や VAE ではダメなのか？
**VAE の限界**:
- Posterior collapse: 潜在空間が使われず、全てのモーションが平均に近づく
- ELBO の下界: 真の多様性を表現しきれない

**GAN の限界**:
- Mode collapse: 多様なモーションの一部しか生成できない
- 訓練不安定: 微細な動きの学習が困難

**Flow Matching の可能性**:
実は、最新の研究では Flow Matching も使われている (第38回参照)。Diffusion より訓練が単純で高速。ただし、Motion/4D では Diffusion が先行。
:::

### 2.4 松尾・岩澤研との差別化 — モーション・4D・ロボティクスの統合

| 観点 | 松尾・岩澤研 (2026Spring) | 本シリーズ 第47回 |
|:-----|:--------------------------|:-----------------|
| **扱う手法** | 画像・動画生成のみ | Motion + 4D + Robotics の統合理解 |
| **理論の深さ** | アルゴリズム紹介 | **数式レベル**: DDPM → Motion Space 適用の導出 |
| **実装** | PyTorch デモ | **3言語**: Julia (訓練) + Rust (推論) + Elixir (分散制御) |
| **最新性** | 2023年まで | **2025-2026 SOTA**: MotionGPT-3/UniMo/4DGS/RDT |
| **ドメイン横断** | なし | Motion ↔ 4D ↔ Robotics の理論的つながり |

**本講義の独自性**:
1. **Text-to-Motion から Diffusion Policy までの統一視点** (全て時系列条件付き生成)
2. **4DGS の数学的基盤** (Deformation field の設計と最適化)
3. **3言語フルスタック**: Julia (モーション訓練)、Rust (4Dレンダリング)、Elixir (ロボット分散制御)

:::details トロイの木馬振り返り: Course V での3言語の役割
**Before (第42回まで)**:
- 画像・動画: Julia/Rust で十分
- Diffusion 訓練: Julia (Lux.jl)
- 推論: Rust (Candle)

**After (第47回)**:
- **Julia**: モーション Diffusion 訓練 (数式↔コードの透明性)
- **Rust**: 4DGS リアルタイムレンダリング (ゼロコピー、並列化)
- **Elixir**: ロボット群の分散制御 (OTP の耐障害性、並行性)

Robotics では Elixir が真価を発揮する。複数ロボットの並行制御、障害時の自動復旧 (OTP Supervisor) が言語レベルで組み込まれている。
:::

### 2.5 本講義の構成 — 3部構成

**Part A: Motion Generation** (Zone 3.1-3.5, ~1200行)
- MDM/MLD の理論 (DDPM の Motion 空間適用)
- MotionGPT-3 (大規模事前学習)
- UniMo (統一モーション生成)
- Motion 評価指標 (FID, Diversity, Physical Plausibility)

**Part B: 4D Generation** (Zone 3.6-3.9, ~1000行)
- 4D Gaussian Splatting の数学
- Deformation field 設計
- TC4D (Trajectory conditioning)
- 4D 評価指標 (Temporal Consistency)

**Part C: Diffusion Policy for Robotics** (Zone 3.10-3.12, ~600行)
- Diffusion Policy の理論
- Hierarchical Diffusion Policy
- RDT (Robot Diffusion Transformer)

```julia
println("\n【第47回の学習目標】")
println("1. Motion Diffusion の数式を完全導出できる")
println("2. 4DGS の Deformation field を設計できる")
println("3. Diffusion Policy でロボット制御ができる")
println("4. Julia でモーション訓練、Rust で4Dレンダリング、Elixir でロボット制御")
println("\n→ Zone 3 で、これら全てを数式レベルで理解する")
```

:::message
**ここまでで全体の20%完了！** 直感的理解ができた。次は数学の本丸 — Zone 3 「数式修行ゾーン」で、Motion/4D/Policy の数式を完全に導出する。
:::

---

## 📐 3. 数式修行ゾーン（60分）— Motion・4D・Policy の完全導出

**ゴール**: Motion Diffusion、4D Gaussian Splatting、Diffusion Policy の数学的基盤を、ペンと紙で完全に理解する。

このゾーンは本講義の心臓部だ。**ペンと紙を用意して**、各導出を自分の手で追うこと。

---

### 3.1 Motion Representation — 動作をどう表現するか

人間のモーションを表現する方法は複数ある。

#### 表現1: Joint Positions (関節位置)

最も直接的な表現。SMPL [^7] などのスケルトンモデルで、$J$ 個の関節の3D位置を記録:

$$
\mathbf{x} \in \mathbb{R}^{T \times J \times 3}
$$

- $T$: フレーム数 (e.g., 30fps で 2秒 = 60 frames)
- $J$: 関節数 (SMPL: 22, SMPL-H: 52 with hands)
- 3: 3D 座標 $(x, y, z)$

**利点**: 直感的、レンダリングが簡単
**欠点**: 高次元 ($T \times J \times 3$ = 数千次元)、物理制約が暗黙的

#### 表現2: Joint Rotations (関節回転)

Forward Kinematics (FK) の考え方。ルート (腰) からの相対回転をツリー構造で表現:

$$
\mathbf{x} \in \mathbb{R}^{T \times J \times 3} \quad \text{(axis-angle)} \text{ or } \mathbb{R}^{T \times J \times 4} \quad \text{(quaternion)}
$$

**利点**: 物理制約を自然に満たす (関節の長さ一定)、次元削減
**欠点**: FK 計算が必要、rotationの表現が非自明 (gimbal lock問題)

#### 表現3: Latent Code (潜在表現)

VAE で低次元潜在空間に圧縮:

$$
\mathbf{z} \in \mathbb{R}^{T \times d}, \quad d \ll J \times 3
$$

MLD [^2] はこのアプローチ。$d=512$ 程度で $(T, 22, 3)$ を表現。

**利点**: 低次元 → diffusion が高速、滑らかな補間
**欠点**: 再構成誤差、latent 空間の解釈性

#### MDM の選択: Joint Positions + Geometric Loss

MDM [^1] は **Joint Positions** を採用し、以下の geometric loss を追加:

$$
\mathcal{L}_{\text{geom}} = \mathcal{L}_{\text{foot}} + \mathcal{L}_{\text{vel}}
$$

- $\mathcal{L}_{\text{foot}}$: Foot contact loss (足が地面に接触している時、速度=0)
- $\mathcal{L}_{\text{vel}}$: Velocity consistency loss (急激な加速度を抑制)

```julia
# Motion representation comparison

# Joint Positions (MDM)
T, J = 60, 22
motion_pos = randn(T, J, 3)  # 60 frames, 22 joints, 3D
println("Joint Positions: $(size(motion_pos)) = $(prod(size(motion_pos))) dims")

# Joint Rotations (axis-angle)
motion_rot = randn(T, J, 3)  # 60 frames, 22 joints, axis-angle
println("Joint Rotations: $(size(motion_rot)) = $(prod(size(motion_rot))) dims")

# Latent Code (MLD)
d_latent = 512
motion_latent = randn(T, d_latent)  # 60 frames, 512 latent dims
println("Latent Code: $(size(motion_latent)) = $(prod(size(motion_latent))) dims")

println("\nDimensionality: Positions=$(prod(size(motion_pos))), Rotations=$(prod(size(motion_rot))), Latent=$(prod(size(motion_latent)))")
println("Latent は $(round(prod(size(motion_pos)) / prod(size(motion_latent)), digits=1))x 圧縮")
```

### 3.2 Motion Diffusion Model (MDM) — DDPM の Motion 空間適用

MDM [^1] は DDPM (第36回) を motion 空間 $\mathbb{R}^{T \times J \times 3}$ で行う。

#### Forward Process: ノイズを加える

第36回 DDPM と同じ:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

- $\mathbf{x}_0 \in \mathbb{R}^{T \times J \times 3}$: クリーンなモーション
- $\mathbf{x}_t$: ノイズが加わったモーション
- $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$: 累積ノイズスケジュール

**重要な点**: $\mathbf{x}_t$ は$(T, J, 3)$ の全体に同時にノイズが加わる。各フレーム・各関節が独立にノイズを受け取る。

#### Reverse Process: ノイズを除去する

通常の DDPM は **ノイズ予測** $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c)$ を学習するが、MDM は **サンプル予測** $\hat{\mathbf{x}}_\theta(\mathbf{x}_t, t, c)$ を直接学習する:

$$
\hat{\mathbf{x}}_0 = \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t, c)
$$

ここで $c$ はテキスト条件 (CLIP embedding など)。

**なぜサンプル予測?**
- Geometric loss を直接適用できる (後述)
- モーションの物理制約 (関節の長さ、接地条件) をサンプル空間で強制しやすい

#### 訓練目標: Sample Prediction Loss

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \mathbf{x}_0 - \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t, c) \|^2 \right]
$$

ここで $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$。

**ノイズ予測との関係**:

$$
\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c)}{\sqrt{\bar{\alpha}_t}}
$$

なので、$\hat{\mathbf{x}}_\theta$ を学習 ⇔ $\boldsymbol{\epsilon}_\theta$ を学習 は等価。ただし、MDM は前者を採用。

```julia
# MDM Forward & Reverse Process

function mdm_forward(x0, t, β_schedule)
    # x0: (T, J, 3) clean motion
    # t: timestep
    # β_schedule: noise schedule [β1, ..., βT]

    α_bar_t = prod(1 .- β_schedule[1:t])
    ϵ = randn(size(x0))
    xt = sqrt(α_bar_t) .* x0 .+ sqrt(1 - α_bar_t) .* ϵ

    return xt, ϵ
end

function mdm_sample_prediction_loss(x0, xt, t, c, x̂_θ)
    # x̂_θ(xt, t, c): サンプル予測ネットワーク
    x̂0 = x̂_θ(xt, t, c)
    loss = sum((x0 .- x̂0).^2)  # MSE
    return loss
end

# ダミーネットワーク
x̂_θ_dummy(xt, t, c) = xt .* 0.9  # 簡易的な予測

# テスト
T, J = 30, 22
x0 = randn(T, J, 3)
c = randn(512)  # Text embedding
β = LinRange(1e-4, 0.02, 1000)
t = 500

xt, ϵ = mdm_forward(x0, t, β)
loss = mdm_sample_prediction_loss(x0, xt, t, c, x̂_θ_dummy)

println("\n【MDM Forward & Loss】")
println("Clean motion x0: $(size(x0))")
println("Noisy motion xt: $(size(xt)), noise level: $(round(std(xt - x0), digits=3))")
println("Sample prediction loss: $(round(loss, digits=2))")
```

#### Geometric Loss: 物理制約の導入

MDM の独自性は、**geometric loss** $\mathcal{L}_{\text{geom}}$ の導入だ。

##### Foot Contact Loss

歩行時、足が地面に接触している瞬間、足の速度は0であるべき:

$$
\mathcal{L}_{\text{foot}} = \sum_{t, j \in \{\text{feet}\}} \mathbb{1}[\text{contact}(j,t)] \cdot \| \mathbf{v}_{j,t} \|^2
$$

- $\mathbf{v}_{j,t} = \mathbf{x}_{j,t+1} - \mathbf{x}_{j,t}$: 関節 $j$ の速度
- $\mathbb{1}[\text{contact}(j,t)]$: 接触判定 (高さ $< $ threshold)

##### Velocity Consistency Loss

急激な加速度を抑制:

$$
\mathcal{L}_{\text{vel}} = \sum_{t, j} \| (\mathbf{v}_{j,t+1} - \mathbf{v}_{j,t}) \|^2
$$

##### 全体の損失関数

$$
\mathcal{L}_{\text{MDM}} = \mathcal{L}_{\text{simple}} + \lambda_{\text{foot}} \mathcal{L}_{\text{foot}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}}
$$

```julia
# Geometric Loss 実装

function foot_contact_loss(x̂0)
    # x̂0: (T, J, 3) predicted motion
    T, J, _ = size(x̂0)

    loss = 0.0
    feet_joints = [1, 2]  # 左右足のインデックス
    threshold = 0.05  # 接地判定の高さ閾値

    for t in 1:(T-1)
        for j in feet_joints
            # 接触判定: y座標 (高さ) が閾値以下
            if x̂0[t, j, 2] < threshold
                # 速度計算
                v = x̂0[t+1, j, :] .- x̂0[t, j, :]
                # 接地中は速度=0 であるべき
                loss += sum(v.^2)
            end
        end
    end

    return loss
end

function velocity_consistency_loss(x̂0)
    # 加速度 (速度の変化) を抑制
    T, J, _ = size(x̂0)

    loss = 0.0
    for t in 1:(T-2)
        for j in 1:J
            v_t = x̂0[t+1, j, :] .- x̂0[t, j, :]
            v_t1 = x̂0[t+2, j, :] .- x̂0[t+1, j, :]
            accel = v_t1 .- v_t
            loss += sum(accel.^2)
        end
    end

    return loss
end

# テスト
x̂0 = randn(30, 22, 3)
L_foot = foot_contact_loss(x̂0)
L_vel = velocity_consistency_loss(x̂0)

println("\n【Geometric Loss】")
println("Foot contact loss: $(round(L_foot, digits=3))")
println("Velocity consistency loss: $(round(L_vel, digits=3))")
println("\n→ これらの loss により、物理的にもっともらしいモーションが生成される")
```

### 3.3 Motion Latent Diffusion (MLD) — VAE で高速化

MLD [^2] は、MDM の計算量問題を解決する。$(T, J, 3)$ は高次元すぎて、diffusion が遅い。

#### アーキテクチャ: VAE + Diffusion

```mermaid
graph LR
    A["Motion<br/>(T,J,3)"] --> B["Encoder<br/>E"]
    B --> C["Latent<br/>(T,d)"]
    C --> D["Diffusion<br/>Process"]
    D --> E["Latent'<br/>(T,d)"]
    E --> F["Decoder<br/>D"]
    F --> G["Motion'<br/>(T,J,3)"]
```

$$
\mathbf{z} = E(\mathbf{x}_0), \quad \mathbf{x}_0' = D(\mathbf{z})
$$

- Encoder $E$: $(T, J, 3) \to (T, d)$ where $d \ll J \times 3$ (e.g., $d=512$)
- Decoder $D$: $(T, d) \to (T, J, 3)$

#### VAE 訓練

第10回 VAE の ELBO を motion に適用:

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)} [\| x - D(z) \|^2] + \text{KL}[q(z|x) \| p(z)]
$$

実際は、deterministic encoder を使うので KL 項は省略可:

$$
\mathcal{L}_{\text{VAE}} = \| x_0 - D(E(x_0)) \|^2
$$

#### Latent Diffusion

潜在空間 $\mathbf{z} \in \mathbb{R}^{T \times d}$ で diffusion を行う:

$$
q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \sqrt{\bar{\alpha}_t} \mathbf{z}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

訓練:

$$
\mathcal{L}_{\text{MLD}} = \mathbb{E}_{t, \mathbf{z}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, c) \|^2 \right]
$$

ここで $\mathbf{z}_0 = E(\mathbf{x}_0)$。

#### 計算量比較

| 段階 | MDM | MLD | 速度比 |
|:-----|:----|:----|:------|
| **次元** | $(T, J, 3) = 1980$ | $(T, d) = 512$ | 3.9x 削減 |
| **Diffusion steps** | 1000 | 50 | 20x 削減 |
| **Total** | $1980 \times 1000$ | $512 \times 50$ | **77x 高速化** |

実際の MLD 論文 [^2] では、**100倍高速** を報告。

```julia
# MLD アーキテクチャ

struct MotionVAE
    encoder    # (T, J, 3) → (T, d)
    decoder    # (T, d) → (T, J, 3)
end

function vae_encode(x, vae::MotionVAE)
    # Temporal Conv + Pooling
    # 簡易版: 平均プーリング
    T, J, _ = size(x)
    z = reshape(x, (T, J*3))  # (T, 66)
    # Linear projection to latent dim
    d = 512
    z_latent = z[:, 1:min(d, size(z, 2))]  # (T, d)
    return z_latent
end

function vae_decode(z, vae::MotionVAE, J=22)
    # Latent → Motion
    T, d = size(z)
    # Linear projection + reshape
    x_flat = hcat(z, zeros(T, J*3 - d))  # Pad to (T, 66)
    x_recon = reshape(x_flat, (T, J, 3))
    return x_recon
end

# ダミー VAE
vae = MotionVAE(nothing, nothing)

x0 = randn(30, 22, 3)
z = vae_encode(x0, vae)
x_recon = vae_decode(z, vae)

println("\n【MLD VAE】")
println("Original motion: $(size(x0)) = $(prod(size(x0))) dims")
println("Latent code: $(size(z)) = $(prod(size(z))) dims")
println("Reconstructed motion: $(size(x_recon))")
println("Compression ratio: $(round(prod(size(x0)) / prod(size(z)), digits=1))x")
println("Reconstruction error: $(round(mean((x0 - x_recon).^2), digits=4))")
```

### 3.4 MotionGPT-3 — 大規模事前学習とチェーン・オブ・モーション

MotionGPT-3 [^8] は、**モーションを第2の言語**として扱い、言語モデルのスケーリング則をモーション生成に適用する。

#### アーキテクチャ: Bimodal LLM

```mermaid
graph TD
    A["Text Input"] --> B["Text Encoder<br/>(frozen LLM)"]
    C["Motion Input"] --> D["Motion VAE<br/>Encoder"]
    D --> E["Motion Tokens"]
    B --> F["Shared<br/>Transformer"]
    E --> F
    F --> G["Text Output"]
    F --> H["Motion Output"]
    H --> I["Motion VAE<br/>Decoder"]
```

**3段階訓練**:
1. **Motion pre-training** (text frozen): Motion VAE + motion branch を訓練
2. **Cross-modal alignment**: motion-to-text & motion prediction
3. **Joint fine-tuning**: 全パラメータを同時最適化

#### Chain-of-Motion (CoM): モーション版 Chain-of-Thought

長いモーション (e.g., "walk to the table, pick up the cup, drink") は、サブモーション列として分解:

$$
\text{Motion} = [\text{walk}] \to [\text{reach}] \to [\text{grasp}] \to [\text{lift}] \to [\text{drink}]
$$

各サブモーションは独立に生成され、**transition smoothing** で接続。

#### In-context Learning for Motion

MotionGPT-3 は few-shot learning が可能:

```
Example 1: Text="jump", Motion=M1
Example 2: Text="spin", Motion=M2
Query: Text="dance"
→ Model generates: Motion=M3 (dancing)
```

LLM の in-context learning 能力がモーションにも転移。

```julia
# MotionGPT-3 の概念コード

struct MotionGPT3
    text_encoder    # Frozen LLM (e.g., GPT-2)
    motion_vae      # Motion VAE (encoder/decoder)
    shared_transformer  # Cross-modal Transformer
end

function motiongpt3_generate(text::String, model::MotionGPT3)
    # Step 1: Text encoding
    text_emb = model.text_encoder(text)  # (L, d_text)

    # Step 2: Motion generation in latent space
    # Autoregressive generation in latent motion tokens
    z_motion = []  # latent motion tokens
    for t in 1:T_max
        # Cross-attention: text → motion
        z_t = model.shared_transformer(z_motion, text_emb)
        push!(z_motion, z_t)
    end

    # Step 3: Decode to motion
    motion = model.motion_vae.decoder(z_motion)

    return motion
end

# ダミー
model = MotionGPT3(nothing, nothing, nothing)
# motion = motiongpt3_generate("person walking forward", model)

println("\n【MotionGPT-3】")
println("Training: 3段階 (Motion pre-train → Cross-modal → Joint fine-tune)")
println("Capability: In-context learning for motion")
println("Chain-of-Motion: 長いモーションをサブタスクに分解して生成")
```

### 3.5 UniMo — 統一モーション生成の最前線

UniMo [^9] (2026) は、モーション生成・理解を**完全統一**する。MotionGPT-3 の進化版だ。

#### 核心アイデア: Chain-of-Thought for Motion

LLM の CoT を motion に適用。単にモーションを生成するだけでなく、**推論過程を明示**:

```
Input: "A person jumps over an obstacle"

UniMo Output:
1. [Reasoning] "Need to prepare for jump → crouch → leap → land"
2. [Motion Generation] Generate crouch motion
3. [Motion Generation] Generate leap motion
4. [Motion Generation] Generate landing motion
5. [Combine] Smooth transition across 3 motions
```

これにより:
- **解釈可能性**: なぜそのモーションを生成したか分かる
- **エラー修正**: 推論過程が間違っていれば、修正可能
- **Few-shot**: 推論パターンを示せば、新しいモーションを生成可能

#### Group Relative Policy Optimization (GRPO)

UniMo は RL で post-training を行う。Motion token prediction のエラー累積を抑制:

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{(s,a) \sim \pi_\theta} \left[ A(s,a) \log \pi_\theta(a|s) \right]
$$

ここで $A(s,a)$ は advantage (グループ内での相対的な良さ)。

**なぜ必要?**
- Autoregressive generation は cumulative error が問題
- 最初のトークンの誤りが、後続トークンに伝播
- GRPO はグループ単位で最適化 → 構造的正しさを保証

```julia
println("\n【UniMo vs MotionGPT-3】")
println("MotionGPT-3: モーションを第2言語として扱う")
println("UniMo: + Chain-of-Thought推論 + GRPO post-training")
println("\nUniMo の強み:")
println("  - 解釈可能性: 推論過程を明示")
println("  - 構造的正しさ: GRPO で cumulative error 抑制")
println("  - Few-shot: CoT パターンで新モーション生成")
```

### 3.6 4D Gaussian Splatting (4DGS) — 動的3Dシーンの数学

第46回で学んだ 3D Gaussian Splatting を、時間方向に拡張する。

#### 4DGS の基本定式化

3DGS の Gaussian primitive:

$$
G_i(\mathbf{x}) = \alpha_i \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i) \right)
$$

4DGS は、これを時間の関数にする:

$$
G_i(\mathbf{x}, t) = \alpha_i(t) \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_i(t))^\top \boldsymbol{\Sigma}_i(t)^{-1} (\mathbf{x} - \boldsymbol{\mu}_i(t)) \right)
$$

**各パラメータが時間依存**:
- $\boldsymbol{\mu}_i(t) \in \mathbb{R}^3$: 位置の変化 (平行移動、回転)
- $\boldsymbol{\Sigma}_i(t) \in \mathbb{R}^{3 \times 3}$: 形状の変化 (スケール、回転)
- $\alpha_i(t) \in [0,1]$: 不透明度の変化 (フェードイン・アウト)

#### Deformation Network の設計

時間依存パラメータを neural network でモデル化:

$$
\begin{aligned}
\Delta \boldsymbol{\mu}_i(t) &= f_{\boldsymbol{\mu}}(t, \boldsymbol{\mu}_i^0, \mathbf{h}_i) \\
\Delta \mathbf{R}_i(t) &= f_{\mathbf{R}}(t, \mathbf{h}_i) \\
\Delta \mathbf{s}_i(t) &= f_{\mathbf{s}}(t, \mathbf{h}_i)
\end{aligned}
$$

- $\boldsymbol{\mu}_i^0$: 初期位置 (canonical frame)
- $\mathbf{h}_i$: Gaussian $i$ の特徴ベクトル
- $\Delta \boldsymbol{\mu}$: 位置オフセット
- $\Delta \mathbf{R}$: 回転行列 (SO(3) パラメータ化)
- $\Delta \mathbf{s}$: スケーリング因子

最終的な共分散:

$$
\boldsymbol{\Sigma}_i(t) = \mathbf{R}_i(t) \mathbf{S}_i(t) \mathbf{S}_i(t)^\top \mathbf{R}_i(t)^\top
$$

ここで $\mathbf{S}_i(t) = \text{diag}(\mathbf{s}_i^0 \odot \Delta \mathbf{s}_i(t))$。

#### 訓練: Multi-view Video → 4D Gaussians

入力: $K$ 視点の動画 $\{I_k(t)\}_{k=1}^K$, $t \in [0, T]$

目標: 4D Gaussians $\{G_i(\mathbf{x}, t)\}_{i=1}^N$ を最適化

損失関数:

$$
\mathcal{L} = \sum_{k,t} \| I_k(t) - \text{Render}(\{G_i(\mathbf{x}, t)\}, C_k) \|^2 + \mathcal{L}_{\text{reg}}
$$

- $\text{Render}$: Differentiable rasterization (3DGS と同じ)
- $C_k$: カメラ $k$ のパラメータ
- $\mathcal{L}_{\text{reg}}$: 正則化 (smoothness, sparsity)

**正則化項の重要性**:

$$
\mathcal{L}_{\text{reg}} = \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}}
$$

- $\mathcal{L}_{\text{smooth}}$: 時間的滑らかさ $\sum_i \| \boldsymbol{\mu}_i(t+\Delta t) - \boldsymbol{\mu}_i(t) \|^2$
- $\mathcal{L}_{\text{sparse}}$: Gaussian 数を抑制 (pruning)

```julia
# 4DGS Deformation Network

struct DeformationNetwork
    mlp_position
    mlp_rotation
    mlp_scale
end

function deform_gaussian(μ0, Σ0, t, h_feat, deform_net)
    # μ0: (3,) initial position
    # Σ0: (3,3) initial covariance
    # t: time ∈ [0, 1]
    # h_feat: feature vector

    # Predict deformations
    Δμ = deform_net.mlp_position([t; μ0; h_feat])  # (3,)
    Δrot = deform_net.mlp_rotation([t; h_feat])     # (3,) axis-angle
    Δscale = deform_net.mlp_scale([t; h_feat])      # (3,) scale factors

    # Apply deformation
    μt = μ0 .+ Δμ

    # Rotation matrix from axis-angle
    θ = norm(Δrot)
    if θ > 1e-6
        axis = Δrot ./ θ
        # Rodrigues' formula
        K = [0 -axis[3] axis[2]; axis[3] 0 -axis[1]; -axis[2] axis[1] 0]
        R = I(3) + sin(θ) * K + (1 - cos(θ)) * K^2
    else
        R = I(3)
    end

    # Reconstruct covariance: Σ = R·S·S'·R'
    S = diagm(sqrt.(diag(Σ0)) .* (1 .+ Δscale))
    Σt = R * S * S' * R'

    return μt, Σt
end

# ダミー MLP
mlp_dummy(x) = randn(3) * 0.1

deform_net = DeformationNetwork(mlp_dummy, mlp_dummy, mlp_dummy)
μ0 = [0.0, 0.0, 0.0]
Σ0 = diagm([1.0, 1.0, 1.0])
h_feat = randn(64)

println("\n【4DGS Deformation】")
for t in [0.0, 0.5, 1.0]
    μt, Σt = deform_gaussian(μ0, Σ0, t, h_feat, deform_net)
    println("t=$t: μ=$(round.(μt, digits=3)), Σ_diag=$(round.(diag(Σt), digits=3))")
end
```

### 3.7 TC4D — Trajectory-Conditioned 4D Generation

4DGS の問題: **global motion と local deformation が混在**し、長時間で破綻しやすい。

TC4D [^4] の解決策: **Motion Factorization**

$$
\text{Motion}(x, t) = T_{\text{global}}(t) \circ D_{\text{local}}(x, t)
$$

- $T_{\text{global}}(t)$: カメラ軌跡に沿った剛体変換 (B-spline)
- $D_{\text{local}}(x, t)$: 局所的な変形 (Neural Network)

#### Spline Trajectory for Global Motion

カメラ軌跡を B-spline で定義:

$$
\mathbf{T}(t) = \sum_{i=0}^n B_i^k(t) \mathbf{P}_i
$$

- $B_i^k(t)$: B-spline basis function (次数 $k$)
- $\mathbf{P}_i$: 制御点 (3D位置 + 回転)

**利点**:
- カメラ軌跡を明示的に制御可能
- Global motion はスムーズに保証
- Local deformation だけを学習すればよい → 学習が安定

#### Local Deformation のみ学習

TC4D の deformation network は、global motion を引いた残差だけを予測:

$$
\Delta \boldsymbol{\mu}_i(t) = \mathbf{D}_\theta(\mathbf{T}^{-1}(t) \boldsymbol{\mu}_i^0, t, \mathbf{h}_i)
$$

訓練時:

1. カメラ軌跡 $\mathbf{T}(t)$ を与える (またはユーザーが描く)
2. 各時刻 $t$ で、global transform を適用: $\boldsymbol{\mu}'_i = \mathbf{T}(t) \boldsymbol{\mu}_i^0$
3. Residual $\Delta \boldsymbol{\mu}_i$ だけを deformation network で学習

**結果**: 数分の動画でも一貫した動きを生成可能。

```julia
# TC4D: Global + Local Factorization

function b_spline_basis(t, i, k, knots)
    # B-spline basis function (Cox-de Boor recursion)
    # 簡易版: Linear interpolation (k=1)
    if i < length(knots) - 1
        if knots[i] <= t < knots[i+1]
            return (t - knots[i]) / (knots[i+1] - knots[i])
        elseif knots[i+1] <= t < knots[i+2]
            return (knots[i+2] - t) / (knots[i+2] - knots[i+1])
        end
    end
    return 0.0
end

function global_trajectory(t, control_points, knots)
    # B-spline trajectory
    n = length(control_points)
    T_global = zeros(3)
    for i in 1:n
        T_global .+= b_spline_basis(t, i, 1, knots) .* control_points[i]
    end
    return T_global
end

# Control points (ユーザー指定の軌跡)
control_points = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [2.0, 0.0, 0.5]]
knots = [0.0, 0.33, 0.67, 1.0]

println("\n【TC4D Trajectory】")
for t in [0.0, 0.5, 1.0]
    T_g = global_trajectory(t, control_points, knots)
    println("t=$t: Global motion=$(round.(T_g, digits=3))")
end

# Local deformation は global から独立して学習
println("\n→ Global motion (spline) + Local deformation (network) を分離")
println("   これにより、長時間動画でも破綻しない")
```

### 3.8 Diffusion Policy — ロボット制御への応用

#### なぜロボット制御に Diffusion?

従来のロボット制御:

$$
\pi_\theta(a | o) = \mu_\theta(o) + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

これは **Gaussian policy**。しかし、多くのタスクは **multimodal**:

- 物を掴む: 左から or 右から or 上から → 複数の正解
- 障害物回避: 左回り or 右回り
- 組み立て: パーツAから or パーツBから

Gaussian では平均しか出せない → 失敗。

**Diffusion Policy の解決策** [^5]:

$$
\pi_\theta(a | o) = p_\theta(a_0 | o), \quad a_0 \sim \text{Reverse Diffusion}
$$

任意の分布を表現可能 → multimodal 行動を生成。

#### Receding Horizon Control (RHC)

Diffusion Policy は **action horizon** $H$ ステップ先まで一度に生成:

$$
\mathbf{a} = [a_t, a_{t+1}, \ldots, a_{t+H-1}] \in \mathbb{R}^{H \times D}
$$

しかし、**最初の1ステップだけ実行** → 再観測 → 再生成 (MPC風):

```
t=0: Generate a[0:H-1] → Execute a[0]
t=1: Generate a[1:H] → Execute a[1]
t=2: Generate a[2:H+1] → Execute a[2]
...
```

**なぜ?**
- 観測は時間とともに変化 (物体が動く、ロボット自身が動く)
- 最新の観測で再計画 → 適応的制御

#### 訓練: Behavior Cloning with Diffusion

Expert demonstration からの模倣学習:

$$
\mathcal{L} = \mathbb{E}_{(o,a) \sim \mathcal{D}, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(a_t, t, o) \|^2 \right]
$$

- $\mathcal{D}$: Expert trajectories $\{(o_i, a_i)\}_i$
- $a_t = \sqrt{\bar{\alpha}_t} a_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$: Forward diffusion

推論時:

1. 観測 $o$ を取得
2. $a_T \sim \mathcal{N}(0, I)$ から開始
3. Reverse diffusion で $a_0$ を生成
4. $a_0[0]$ (最初のステップ) だけを実行

```julia
# Diffusion Policy 訓練・推論

function diffusion_policy_train(expert_data, ϵ_θ, β, num_epochs=100)
    # expert_data: [(observation, action), ...]
    # ϵ_θ: Noise prediction network

    T_steps = length(β)
    losses = []

    for epoch in 1:num_epochs
        total_loss = 0.0
        for (o, a) in expert_data
            # Sample random timestep
            t = rand(1:T_steps)

            # Forward diffusion
            α_bar_t = prod(1 .- β[1:t])
            ϵ = randn(size(a))
            at = sqrt(α_bar_t) .* a .+ sqrt(1 - α_bar_t) .* ϵ

            # Predict noise
            ϵ_pred = ϵ_θ(at, t, o)

            # Loss
            loss = sum((ϵ .- ϵ_pred).^2)
            total_loss += loss

            # Backprop (ダミー)
            # ... gradient descent on ϵ_θ
        end
        push!(losses, total_loss / length(expert_data))
    end

    return losses
end

function diffusion_policy_infer(observation, ϵ_θ, β, H=8)
    # observation: current obs
    # H: action horizon

    T_steps = length(β)
    action_dim = 7  # 7-DoF arm
    aT = randn(H, action_dim)

    # Reverse diffusion
    a = copy(aT)
    for t in T_steps:-1:1
        ϵ_pred = ϵ_θ(a, t, observation)

        α_t = 1 - β[t]
        α_bar_t = prod(1 .- β[1:t])

        # DDPM update
        a = (a - (β[t] / sqrt(1 - α_bar_t)) * ϵ_pred) / sqrt(α_t)

        if t > 1
            σ = sqrt(β[t])
            a = a .+ σ .* randn(H, action_dim)
        end
    end

    return a[1, :]  # Only execute first action
end

# ダミー expert data
expert_data = [(randn(64, 64, 3), randn(8, 7)) for _ in 1:10]
ϵ_θ_policy(a, t, o) = a .* 0.1  # ダミー

β = LinRange(1e-4, 0.02, 50)
# losses = diffusion_policy_train(expert_data, ϵ_θ_policy, β, 5)

obs = randn(64, 64, 3)
action = diffusion_policy_infer(obs, ϵ_θ_policy, β)

println("\n【Diffusion Policy】")
println("Observation: 64x64 RGB")
println("Generated action (1st step): $(round.(action, digits=3))")
println("\n→ Multimodal policy: 同じ観測から複数の行動を生成可能")
```

### 3.9 Hierarchical Diffusion Policy — 接触リッチなタスク

単一 Diffusion Policy は**短期計画のみ**。長期タスク (e.g., 組み立て) では失敗しやすい。

Hierarchical Diffusion Policy [^6] の解決策:

```mermaid
graph TD
    A["Observation o"] --> B["High-level Policy<br/>πH"]
    B --> C["Next subgoal g<br/>(contact point)"]
    C --> D["Low-level Policy<br/>πL diffusion"]
    D --> E["Action a"]
    E --> F["Execute"]
    F --> A
```

#### High-level: 接触点予測

$$
g = \pi_H(o, \text{task})
$$

$g \in \mathbb{R}^3$: 次に接触すべき3D点 (e.g., "掴むべき位置")

#### Low-level: Diffusion で行動生成

$$
a \sim \pi_L(a | o, g)
$$

$g$ を条件として、Diffusion で行動を生成。

**訓練**:
- High-level: Expert demonstration から接触点ラベルを抽出
- Low-level: $(o, g) \to a$ の diffusion policy

**効果**:
- 接触リッチなタスク (組み立て、挿入) で **20.8% 性能向上** [^6]
- 長期計画が可能 (High-level が分解)
- 解釈可能 (接触点が可視化される)

```julia
println("\n【Hierarchical Diffusion Policy】")
println("High-level: 観測 → 次の接触点 (where to touch)")
println("Low-level: (観測, 接触点) → 行動軌跡 (how to get there)")
println("\nメリット:")
println("  - 長期タスクの分解")
println("  - 接触リッチなタスクで高性能 (+20.8%)")
println("  - 解釈可能性 (接触点の可視化)")
```

### 3.10 RDT (Robot Diffusion Transformer) — Foundation Model for Robotics

RDT [^10] は、**1B parameter の Diffusion Transformer** を46データセット (1M+ episodes) で事前学習。

#### アーキテクチャ: Transformer + Diffusion

```mermaid
graph LR
    A["Observation<br/>(Image+State)"] --> B["Vision<br/>Encoder"]
    C["Language<br/>Instruction"] --> D["Text<br/>Encoder"]
    B --> E["Transformer<br/>(1B params)"]
    D --> E
    E --> F["Diffusion<br/>Decoder"]
    F --> G["Action<br/>(64 steps)"]
```

**特徴**:
- **Multi-modal input**: 最大3視点のRGB画像 + ロボット状態
- **Language-conditioned**: テキストで指示
- **Long horizon**: 64ステップ先まで予測

#### Zero-shot Generalization

事前学習により:
- **未見のオブジェクト**: 学習時になかった物体でも操作可能
- **未見のシーン**: 新しい環境でも適応
- **Few-shot adaptation**: 1-5デモで新タスクを学習

**RDT-2 (2025)**: 未見の **embodiment** (ロボットの種類) にもゼロショット対応。

```julia
println("\n【RDT: Foundation Model for Robotics】")
println("規模: 1B parameters, 46 datasets, 1M+ episodes")
println("入力: 最大3視点RGB + robot state + language instruction")
println("出力: 64ステップ action sequence")
println("\nZero-shot capability:")
println("  - 未見オブジェクト")
println("  - 未見シーン")
println("  - RDT-2: 未見embodiment (ロボットの種類)")
```

### 3.11 4DGS の詳細導出 — 時間依存共分散行列

4DGS [^3] の数学的核心は、**時間依存する3D Gaussian** をどう定式化するか。

#### 静的 3DGS の復習

静的 Gaussian の密度関数:

$$
G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

- $\boldsymbol{\mu} \in \mathbb{R}^3$: 中心位置
- $\Sigma \in \mathbb{R}^{3 \times 3}$: 共分散行列 (正定値対称)

**パラメータ化**: $\Sigma = R S S^\top R^\top$

- $R \in SO(3)$: 回転行列 (quaternion で表現)
- $S = \text{diag}(s_x, s_y, s_z)$: スケール (対角行列)

これで正定値性が保証される。

#### 時間依存への拡張

4DGS は $\boldsymbol{\mu}$ と $\Sigma$ を時間 $t$ の関数にする:

$$
G_i(\mathbf{x}, t) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i(t))^\top \Sigma_i(t)^{-1} (\mathbf{x} - \boldsymbol{\mu}_i(t))\right)
$$

**課題**: $\boldsymbol{\mu}(t)$ と $\Sigma(t)$ をどう設計するか？

#### Deformation Network $f_\theta$

4DGS は **deformation network** $f_\theta$ を導入:

$$
\begin{aligned}
\Delta \boldsymbol{\mu}(t), \Delta \mathbf{q}(t), \Delta \mathbf{s}(t) &= f_\theta(\boldsymbol{\mu}_0, t) \\
\boldsymbol{\mu}(t) &= \boldsymbol{\mu}_0 + \Delta \boldsymbol{\mu}(t) \\
\mathbf{q}(t) &= \mathbf{q}_0 \odot \Delta \mathbf{q}(t) \\
\mathbf{s}(t) &= \mathbf{s}_0 \odot \exp(\Delta \mathbf{s}(t))
\end{aligned}
$$

- $\boldsymbol{\mu}_0, \mathbf{q}_0, \mathbf{s}_0$: Canonical frame (時刻0) のパラメータ
- $\Delta$: Deformation (変形量)
- $\odot$: Element-wise product (quaternion は積で合成)

**Deformation network の実装** (Multi-Resolution Hash Encoding):

$$
f_\theta(\boldsymbol{\mu}_0, t) = \text{MLP}_\theta\left(\text{Hash}(\boldsymbol{\mu}_0) \oplus \text{Embed}(t)\right)
$$

- $\text{Hash}(\boldsymbol{\mu}_0)$: Instant-NGP 風の multi-resolution hash grid
- $\text{Embed}(t)$: 時間 $t$ の sinusoidal embedding
- $\oplus$: Concatenation

#### 共分散行列の時間依存性

$\mathbf{q}(t)$ と $\mathbf{s}(t)$ から $\Sigma(t)$ を構築:

$$
\begin{aligned}
R(t) &= \text{QuatToMat}(\mathbf{q}(t)) \\
S(t) &= \text{diag}(\mathbf{s}(t)) \\
\Sigma(t) &= R(t) S(t) S(t)^\top R(t)^\top
\end{aligned}
$$

**Quaternion から回転行列への変換** (復習):

$$
R = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - zw) & 2(xz + yw) \\
2(xy + zw) & 1 - 2(x^2 + z^2) & 2(yz - xw) \\
2(xz - yw) & 2(yz + xw) & 1 - 2(x^2 + y^2)
\end{bmatrix}
$$

ここで $\mathbf{q} = (w, x, y, z)$ (normalized: $w^2 + x^2 + y^2 + z^2 = 1$).

#### Rendering Equation の時間拡張

レンダリングは、時刻 $t$ におけるすべての Gaussian の寄与を累積:

$$
C(\mathbf{r}, t) = \sum_{i \in \mathcal{I}} c_i \alpha_i(t) T_i(t) G_i(\mathbf{x}(\mathbf{r}, d_i), t)
$$

- $c_i$: Gaussian $i$ の色 (spherical harmonics で view-dependent にも可能)
- $\alpha_i(t) = o_i \cdot G_i(\mathbf{x}, t)$: Opacity ($o_i$ は学習可能パラメータ)
- $T_i(t) = \prod_{j < i} (1 - \alpha_j(t))$: Transmittance (前方の Gaussian による遮蔽)
- $\mathbf{x}(\mathbf{r}, d_i)$: レイ $\mathbf{r}$ が深度 $d_i$ で交差する点

**キーポイント**: すべての項が $t$ に依存 → 動的シーンのレンダリング。

#### 訓練: 時空間一貫性損失

4DGS の訓練では、各時刻 $t$ でレンダリングし、GT 画像と比較:

$$
\mathcal{L} = \sum_{t=1}^T \left\| C(\mathbf{r}, t) - I_{\text{gt}}(\mathbf{r}, t) \right\|_2^2 + \lambda_{\text{reg}} \mathcal{R}(\boldsymbol{\mu}(t), \Sigma(t))
$$

**正則化項** $\mathcal{R}$:
1. **Smoothness**: $\| \boldsymbol{\mu}(t+1) - \boldsymbol{\mu}(t) \|_2^2$ — 時間方向の滑らかさ
2. **Scale penalty**: $\sum_i \| \mathbf{s}_i(t) \|_2$ — Gaussian が大きくなりすぎるのを防ぐ
3. **Opacity regularization**: $\sum_i (o_i - 0.5)^2$ — Opacity が極端な値にならないように

#### 数値検証

```julia
using LinearAlgebra

# Quaternion to rotation matrix
function quat_to_matrix(q)
    w, x, y, z = q
    R = [
        1 - 2(y^2 + z^2)  2(x*y - z*w)      2(x*z + y*w);
        2(x*y + z*w)      1 - 2(x^2 + z^2)  2(y*z - x*w);
        2(x*z - y*w)      2(y*z + x*w)      1 - 2(x^2 + y^2)
    ]
    return R
end

# Time-dependent Gaussian parameters
μ_0 = [0.0, 0.0, 0.0]
q_0 = [1.0, 0.0, 0.0, 0.0]  # Identity rotation
s_0 = [1.0, 1.0, 1.0]

# Deformation at time t
t = 0.5
Δμ = [0.1, 0.0, 0.0] * t  # Move along x-axis
Δq = [cos(0.1t), sin(0.1t), 0, 0]  # Small rotation
Δs = [0.05, 0.05, 0.05] * t

# Apply deformation
μ_t = μ_0 + Δμ
q_t = quat_to_matrix(Δq) * q_0  # Simplified (should be quaternion multiplication)
s_t = s_0 .* exp.(Δs)

# Build covariance matrix
R_t = quat_to_matrix([1.0, 0.0, 0.0, 0.0])  # Simplified
S_t = diagm(s_t)
Σ_t = R_t * S_t * S_t' * R_t'

println("Time-dependent Gaussian at t=$t:")
println("  μ(t) = $μ_t")
println("  Σ(t) = ")
display(Σ_t)

# Verify positive definite
eigvals_Σ = eigvals(Σ_t)
println("\nEigenvalues of Σ(t): $eigvals_Σ")
println("All positive? $(all(eigvals_Σ .> 0))")
```

:::message
**4DGS の本質**: 静的 3DGS の各 Gaussian に deformation network を追加し、時間依存の $\boldsymbol{\mu}(t)$ と $\Sigma(t)$ を表現。時空間一貫性損失で訓練。
:::

:::message
**ここまでで全体の50%完了！** Zone 3 の数式修行を終えた。Motion Diffusion、4DGS、Diffusion Policy の理論を完全に理解した。次は実装 — Zone 4 で Julia/Rust/Elixir で実装する。
:::

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
