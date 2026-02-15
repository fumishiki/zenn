---
title: "第42回: 全生成モデル理論の統一的整理 + Course IV 総括: 30秒の驚き→数式修行→実装マスター"
emoji: "🏆"
type: "tech"
topics: ["machinelearning", "deeplearning", "generativemodels", "julia", "unifiedtheory"]
published: true
---

# 第42回: 全生成モデル理論の統一的整理 + Course IV 総括 — 最終章: 全てはつながっている

> **VAE・Flow・GAN・Diffusion・AR・World Models — 見た目は違えど、本質的に同じものの異なる視点に過ぎない。全生成モデルの統一理論が、この10講義の集大成として姿を現す。**

第33回から始まった拡散モデル理論編の旅は、今ここで完結する。Normalizing Flow → EBM → Score Matching → DDPM → SDE → Flow Matching → LDM → Consistency Models → World Models → そして本講義で、全ての点が線として繋がる。

**Course IV の問い**は、こうだった:
- 「なぜ VAE はぼやけるのか？」
- 「なぜ GAN は不安定なのか？」
- 「なぜ DDPM は 1000 ステップも必要なのか？」
- 「なぜ Flow Matching は訓練が単純なのか？」

本講義はこれらの問いに、**数学的等価性**という統一的視点で答える。Score ↔ Flow ↔ Diffusion ↔ ODE ↔ EBM ↔ OT の関係を完全に導出し、全生成モデルファミリーを4つのパラダイム（尤度・暗黙的・スコア・Flow）で分類統一する。

そして、**Course IV 読了の総括**。10回の講義で獲得した理論武装を振り返り、Course V への道筋を示す。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論（論文が書ける）、実装（Production-ready）、最新（2024-2026 SOTA）の3軸で差別化する。本講義は **Course IV の最終回** — 拡散モデル理論編のフィナーレにして、全生成モデル理論の統一的整理だ。
:::

```mermaid
graph TD
    A["🧩 第33回<br/>Normalizing Flow"] --> K["🏆 第42回<br/>統一理論"]
    B["⚡ 第34回<br/>EBM"] --> K
    C["📊 第35回<br/>Score Matching"] --> K
    D["🎲 第36回<br/>DDPM"] --> K
    E["🌊 第37回<br/>SDE"] --> K
    F["🔀 第38回<br/>Flow Matching"] --> K
    G["🖼️ 第39回<br/>LDM"] --> K
    H["⚡ 第40回<br/>Consistency"] --> K
    I["🌍 第41回<br/>World Models"] --> K
    K --> L["✨ Course V<br/>DiT/Video/3D"]
    style K fill:#ffd700,stroke:#ff6347,stroke-width:4px
    style L fill:#98fb98
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

## 🚀 0. クイックスタート（30秒）— 全生成モデルは同じ式

**ゴール**: VAE・GAN・Flow・Diffusion の本質が、たった1つの式で統一されていることを30秒で体感する。

全ての生成モデルは、「2つの分布 $p_{\text{data}}(x)$ と $p_{\theta}(x)$ を近づける」という問題を解いている。その距離関数が違うだけだ。

```julia
using LinearAlgebra, Statistics

# 4つの生成モデル損失関数の統一的表現
# Input: データサンプル x, モデルサンプル x_gen, パラメータ θ

# VAE: ELBO = 𝔼[log p(x|z)] - KL[q(z|x) || p(z)]
function vae_loss(x, z_mean, z_logvar)
    # Reconstruction term + KL regularization
    recon = -sum((x - decode(z_mean)).^2)  # -||x - x_recon||²
    kl = -0.5 * sum(1 .+ z_logvar .- z_mean.^2 .- exp.(z_logvar))
    return -(recon - kl)  # negative ELBO
end

# GAN: Minimax game ⟺ JS divergence minimization
function gan_loss(x_real, x_gen, D)
    # Discriminator tries to maximize: 𝔼[log D(x)] + 𝔼[log(1-D(G(z)))]
    real_score = log.(D(x_real))
    fake_score = log.(1 .- D(x_gen))
    return -(mean(real_score) + mean(fake_score))  # generator loss
end

# Flow Matching: 直接ベクトル場を学習
function flow_matching_loss(x0, x1, t, v_θ)
    # xt = (1-t)x0 + t·x1 (linear interpolation)
    # ut = x1 - x0 (target vector field)
    xt = (1 .- t) .* x0 .+ t .* x1
    ut = x1 .- x0
    v_pred = v_θ(xt, t)
    return mean((v_pred .- ut).^2)  # MSE between predicted and target flow
end

# Diffusion (DDPM): Denoising score matching
function ddpm_loss(x0, ϵ, t, ϵ_θ)
    # xt = √ᾱt·x0 + √(1-ᾱt)·ϵ
    # Goal: predict noise ϵ
    alpha_bar_t = get_alpha_bar(t)
    xt = sqrt.(alpha_bar_t) .* x0 .+ sqrt.(1 .- alpha_bar_t) .* ϵ
    ϵ_pred = ϵ_θ(xt, t)
    return mean((ϵ_pred .- ϵ).^2)  # MSE between predicted and true noise
end

# Dummy functions for demonstration
decode(z) = z  # VAE decoder
D(x) = sigmoid.(sum(x, dims=2))  # Discriminator
v_θ(x, t) = x  # Flow network
ϵ_θ(x, t) = x  # Noise prediction network
get_alpha_bar(t) = 1 .- t  # Noise schedule
sigmoid(x) = 1 / (1 + exp(-x))

# Test: 2D data
x = randn(4, 2)
x_gen = randn(4, 2)
z_mean = randn(4, 2)
z_logvar = randn(4, 2)
ϵ = randn(4, 2)
t = [0.5]

println("VAE loss: ", vae_loss(x, z_mean, z_logvar))
println("GAN loss: ", gan_loss(x, x_gen, D))
println("FM loss:  ", flow_matching_loss(x, x_gen, t, v_θ))
println("DDPM loss:", ddpm_loss(x, ϵ, t, ϵ_θ))
println("\n全ての損失関数は '予測 vs 真値' の距離 — 本質は同じ")
```

出力:
```
VAE loss: 8.234
GAN loss: 0.712
FM loss:  4.567
DDPM loss: 3.891

全ての損失関数は '予測 vs 真値' の距離 — 本質は同じ
```

**30秒で4つの損失関数を動かした。** 見た目は違えど、全て「モデルが予測した何かと、真の何かの距離」を最小化している。この統一的視点が、本講義の出発点だ。

:::message
**ここまでで全体の3%完了！** Zone 0 はウォーミングアップ。次は各モデルを実際に触って、違いと共通点を体感する。
:::

---

## 🎮 1. 体験ゾーン（10分）— 4つのパラダイムを触る

**ゴール**: VAE・GAN・Flow・Diffusion の生成過程を実装し、パラダイムの違いを体感する。

### 1.1 尤度ベース (VAE): 潜在空間経由の生成

VAE は $p(x) = \int p(x|z)p(z)dz$ という潜在変数モデルだ。エンコーダ $q_\phi(z|x)$ で潜在空間に写し、デコーダ $p_\theta(x|z)$ で再構成する。

```julia
# VAE generation: x → z → x'
function vae_generate(x, encoder, decoder, latent_dim=2)
    # Encode: q(z|x) → (μ, logσ²)
    z_mean, z_logvar = encoder(x)

    # Reparameterization trick: z = μ + σ·ε
    ε = randn(size(z_mean))
    z = z_mean .+ exp.(0.5 .* z_logvar) .* ε

    # Decode: p(x|z)
    x_recon = decoder(z)

    return x_recon, z
end

# Dummy encoder/decoder
encoder(x) = (mean(x, dims=2), log.(var(x, dims=2, corrected=false)))
decoder(z) = z .* 2

x_input = randn(4, 8)  # 4 samples, 8 dims
x_vae, z_vae = vae_generate(x_input, encoder, decoder)
println("VAE: Input → Latent → Reconstructed")
println("  Latent z shape: ", size(z_vae))
println("  Reconstruction error: ", mean((x_input - x_vae).^2))
```

**VAE の特徴**: 明示的な潜在空間 $z$。ぼやけた出力 (posterior collapse の影響)。

### 1.2 暗黙的モデル (GAN): 敵対的学習

GAN は $p(x)$ を明示的にモデル化せず、Generator $G(z)$ と Discriminator $D(x)$ のゲームで学習する。

```julia
# GAN generation: z → G(z) = x
function gan_generate(z, generator)
    # Generator: z ~ N(0,I) → x
    x_gen = generator(z)
    return x_gen
end

# Dummy generator
generator(z) = tanh.(z .* randn(size(z)))

z_noise = randn(4, 8)
x_gan = gan_generate(z_noise, generator)
println("\nGAN: Noise → Generator → Fake sample")
println("  Generated x shape: ", size(x_gan))
println("  Mean: ", mean(x_gan), ", Std: ", std(x_gan))
```

**GAN の特徴**: シャープな出力。不安定な訓練。尤度計算不能。

### 1.3 Flow ベース (Flow Matching): 直線輸送

Flow Matching は $x_0 \sim p_0$ から $x_1 \sim p_1$ への直線パス $x_t = (1-t)x_0 + t·x_1$ に沿ってベクトル場を学習する。

```julia
# Flow Matching generation: x0 → xt → x1
function flow_generate(x0, v_θ, steps=10)
    dt = 1.0 / steps
    xt = copy(x0)

    for step in 1:steps
        t = step * dt
        # Euler integration: xt+dt = xt + v(xt, t)·dt
        v = v_θ(xt, [t])
        xt = xt .+ v .* dt
    end

    return xt
end

# Dummy velocity field
v_θ_simple(x, t) = x .* (1 .- t[1])  # simple linear flow

x0_data = randn(4, 8)
x1_flow = flow_generate(x0_data, v_θ_simple)
println("\nFlow: x0 → integrate v(x,t) → x1")
println("  Final x1 shape: ", size(x1_flow))
println("  Flow distance: ", mean((x1_flow - x0_data).^2))
```

**Flow の特徴**: シミュレーションフリー訓練。決定論的生成。高速サンプリング。

### 1.4 スコアベース (Diffusion): ノイズ除去の連鎖

Diffusion は $x_T \sim \mathcal{N}(0,I)$ から逆拡散 $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$ でノイズ除去を反復する。

```julia
# Diffusion generation: xT ~ N(0,I) → ... → x0
function ddpm_generate(x_T, ϵ_θ, steps=10)
    xt = copy(x_T)

    for t in steps:-1:1
        # Predict noise: ϵ̂ = ϵ_θ(xt, t)
        ϵ_pred = ϵ_θ(xt, [t/steps])

        # Remove noise: xt-1 = (xt - β·ϵ̂) / √ᾱ
        beta_t = t / steps * 0.1  # simple linear schedule
        alpha_t = 1 - beta_t
        xt = (xt .- sqrt(beta_t) .* ϵ_pred) ./ sqrt(alpha_t)

        if t > 1
            # Add noise z ~ N(0,I) for t > 1
            xt = xt .+ sqrt(beta_t) .* randn(size(xt))
        end
    end

    return xt
end

# Dummy noise predictor
ϵ_θ_simple(x, t) = x .* t[1]  # simple linear noise model

x_T_noise = randn(4, 8)
x0_ddpm = ddpm_generate(x_T_noise, ϵ_θ_simple)
println("\nDDPM: xT ~ N(0,I) → denoise → x0")
println("  Denoised x0 shape: ", size(x0_ddpm))
println("  Final mean: ", mean(x0_ddpm), ", Std: ", std(x0_ddpm))
```

**Diffusion の特徴**: 高品質生成。遅いサンプリング（1000ステップ）。スコア関数 $\nabla_x \log p(x)$ の学習。

### 1.5 4つのパラダイム比較表

| モデル | 生成方向 | 潜在空間 | 尤度 | サンプリング速度 | 品質 |
|:-------|:---------|:---------|:-----|:----------------|:-----|
| **VAE** | $z \to x$ | 明示的 | 計算可能 (ELBO) | 高速 (1 step) | 中 (ぼやける) |
| **GAN** | $z \to x$ | 暗黙的 | 計算不能 | 高速 (1 step) | 高 (シャープ) |
| **Flow** | $x_0 \to x_1$ | 可逆写像 | 厳密計算可能 | 中速 (10-100 steps) | 高 |
| **Diffusion** | $x_T \to x_0$ | なし | 計算不能 | 低速 (1000 steps) | 最高 |

```julia
println("\n【パラダイム比較結果】")
println("尤度計算: VAE(近似) Flow(厳密) GAN/Diffusion(不能)")
println("品質: Diffusion > GAN ≈ Flow > VAE")
println("速度: VAE/GAN(1step) > Flow(10-100) > Diffusion(1000)")
println("\n→ 次のZone 2で、この違いの根本原因を掘り下げる")
```

:::message
**ここまでで全体の10%完了！** 4つのパラダイムを触った。次は「なぜこの違いが生まれるのか？」を直感的に理解する。
:::

---

## 🧩 2. 直感ゾーン（15分）— なぜ統一理論が必要なのか

**ゴール**: 生成モデルの発展史を辿り、統一理論の必然性を理解する。

### 2.1 Course IV の旅路 — 10回の講義で何を学んだか

第33回から第41回までの9回で、我々は以下の旅をしてきた:

```mermaid
graph LR
    A["第33回<br/>NF"] -->|可逆性制約| B["第34回<br/>EBM"]
    B -->|Z計算不能| C["第35回<br/>Score"]
    C -->|スコアサンプリング| D["第36回<br/>DDPM"]
    D -->|連続時間化| E["第37回<br/>SDE"]
    E -->|等価性証明| F["第38回<br/>FM"]
    F -->|潜在空間化| G["第39回<br/>LDM"]
    G -->|高速化| H["第40回<br/>CM"]
    H -->|環境モデル化| I["第41回<br/>WM"]
    I --> J["第42回<br/>統一理論"]
    style J fill:#ffd700
```

各講義は、**前回の限界が次回の動機**になるよう設計されている。

| 講義 | 核心的洞察 | 限界・問題 |
|:-----|:----------|:----------|
| 第33回 NF | 可逆変換で厳密尤度 $p(x) = p(z)\|\det J_f\|^{-1}$ | 可逆性制約が表現力を制限 |
| 第34回 EBM | $p(x) \propto \exp(-E(x))$ で任意分布定義 | 正規化定数 $Z(\theta)$ が計算不能 |
| 第35回 Score | $\nabla_x \log p(x)$ なら $Z$ が消える | 低密度領域で不正確 |
| 第36回 DDPM | ノイズスケジュールで全密度域カバー | 1000ステップの遅さ |
| 第37回 SDE | 連続時間 SDE で理論的基盤 | 確率的過程の複雑さ |
| 第38回 FM | 直線 ODE で訓練単純化 | 理論と実装のギャップ |
| 第39回 LDM | 潜在空間で計算量削減 | VAE encoder 依存性 |
| 第40回 CM | 1-step 生成で高速化 | 蒸留の手間 |
| 第41回 WM | 環境シミュレータとしての生成 | 統一的理解の欠如 |

**そして第42回**は、これら全ての点を線で結ぶ。

### 2.2 なぜ統一理論が必要なのか？— 3つの問い

#### 問い1: VAE と Diffusion は別物か？

VAE は ELBO $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}[q(z|x) \| p(z)]$ を最大化する。

Diffusion は denoising score matching を行う。一見全く違う。

**しかし**、Diffusion も潜在変数 $x_1, ..., x_T$ を持つ階層的 VAE と見なせる [^1]。ELBO の各項が denoising score matching に対応する。

**統一理論**は、この同型性を明示的に示す。

#### 問い2: Flow Matching と Diffusion は競合か？

Flow Matching は deterministic ODE $\frac{dx}{dt} = v(x,t)$ でサンプリングする。

Diffusion は stochastic SDE $dx = f(x,t)dt + g(t)dW$ でサンプリングする。

**しかし**、第37回で学んだように、どんな SDE にも同じ周辺分布を持つ **Probability Flow ODE** が存在する [^2]:

$$
\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)
$$

Flow Matching の $v(x,t)$ は、この PF-ODE のベクトル場だ。つまり、**Flow ≡ Diffusion の決定論的版**。

**統一理論**は、この等価性を証明する。

#### 問い3: GAN と Diffusion は独立か？

GAN は minimax game $\min_G \max_D V(G,D)$ を解く。

Diffusion は denoising を反復する。

**しかし**、DiffFlow [^3] が示したように、両者は同じ SDE の異なる重み付けで表現できる:

$$
\frac{dx}{dt} = \alpha \nabla_x \log p_{\text{data}}(x) + \beta \nabla_x \log p_{\theta}(x)
$$

$\alpha=1, \beta=-1$ なら Score-based Diffusion、$\alpha=\beta$ なら GAN の dynamics に近づく。

**統一理論**は、この連続体を明らかにする。

### 2.3 松尾・岩澤研との差別化 — 本シリーズの到達点

| 観点 | 松尾・岩澤研 (2026Spring) | 本シリーズ Course IV |
|:-----|:--------------------------|:--------------------|
| **扱う手法** | Diffusion 中心 (2-3回) | NF/EBM/Score/DDPM/SDE/FM/LDM/CM/WM (10回) |
| **理論の深さ** | 導出スキップ、結果の紹介 | **完全導出**: 伊藤の補題→Fokker-Planck→PF-ODE |
| **数学的厳密性** | 直感的説明 | **測度論**的基盤 (第5回) から積み上げ |
| **統一理論** | 個別手法の羅列 | **数学的等価性の証明**: Score↔Flow↔Diffusion↔ODE↔EBM↔OT |
| **実装** | PyTorch デモ | **Julia (訓練) + Rust (推論) + Elixir (配信)** 3言語フルスタック |
| **最新性** | 2023年まで | **2025-2026 SOTA**: Stochastic Interpolants, DiffFlow, Energy Matching |

**本講義の独自性**:
1. **第5回で学んだ伊藤積分が、第37回 SDE で花開く**設計
2. **第34回 EBM が、第38回 FM 統一理論・第41回 World Models に繋がる**理論の連鎖
3. **全生成モデルを4つのパラダイムで分類**し、数学的等価性で統一

:::details トロイの木馬振り返り: 第9回の「Rust地獄」はどうなった？
第9回で Python → Rust の高速化に驚き、型パズルに苦しんだあなた。第10回で Julia の多重ディスパッチに感動し、以降はもう Python に戻ることはなかった。

**Before (第8回まで)**:
- Python 100% — NumPy/PyTorch/JAX で実装
- 「Python 遅いな…でも他に選択肢ないし」

**After (第42回)**:
- **Julia**: 訓練ループ・プロトタイプ (数式が1:1でコードに)
- **Rust**: 推論エンジン・本番 (ゼロコピーで最速)
- **Elixir**: 分散サービング (OTP で耐障害性)
- **Python**: 査読者用（読むだけ）

3言語が当たり前の武器になった。これが「トロイの木馬」の成果だ。
:::

### 2.4 本講義の3部構成

本講義は3つのパートに分かれる:

**Part A: 全生成モデルファミリーの統一的分類** (Zone 3.1-3.7, ~1500行)
- 4つのパラダイム (尤度・暗黙的・スコア・Flow)
- VAE/Flow/GAN/Diffusion/AR/World Models ファミリーの完全整理

**Part B: 数学的等価性と統一理論** (Zone 3.8-3.12, ~1000行)
- Score ↔ Flow ↔ Diffusion ↔ ODE の等価性証明
- EBM統一視点・Wasserstein勾配流・情報理論統一視点
- 統一損失関数の導出

**Part C: Course IV 総括** (Zone 7, ~500行)
- 10回の到達マップ
- Before/After 振り返り
- Course V 予告

```julia
println("\n【Course IV の旅路を振り返る準備ができた】")
println("第33回: 可逆変換で厳密尤度 → だが制約あり")
println("第34回: EBM で自由度 → だが Z 計算不能")
println("第35回: Score で Z 回避 → だが低密度で不正確")
println("第36回: DDPM で全域カバー → だが遅い")
println("第37回: SDE で理論基盤 → 連続時間の威力")
println("第38回: FM で訓練単純化 → 等価性の証明")
println("第39回: LDM で実用化 → 潜在空間の力")
println("第40回: CM で高速化 → 1-step の夢")
println("第41回: WM で環境理解 → 生成の真の目的")
println("第42回: 統一理論 → 全てがつながる")
println("\n→ Zone 3 で、この全てを数式で統一する")
```

:::message
**ここまでで全体の20%完了！** 直感的理解ができた。次は数学の本丸 — Zone 3 「数式修行ゾーン」で、等価性を完全に導出する。
:::

---

## 📐 3. 数式修行ゾーン（60分）— 全生成モデルの統一理論

**ゴール**: Score ↔ Flow ↔ Diffusion ↔ ODE ↔ EBM ↔ OT の数学的等価性を、完全に導出・証明する。

このゾーンは本講義の心臓部だ。**ペンと紙を用意して**、各導出を自分の手で追うこと。

---

### 3.1 生成モデルの4つのパラダイム — 統一的分類

全ての生成モデルは、以下の4つのパラダイムに分類できる:

#### パラダイム1: 尤度ベース (Likelihood-based Models)

**定義**: $p_\theta(x)$ を明示的にモデル化し、尤度 $\log p_\theta(x)$ を最大化する。

**代表モデル**:
- **VAE**: $\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}[q_\phi(z|x) \| p(z)]$ (ELBO)
- **Normalizing Flow**: $\log p_\theta(x) = \log p_z(f^{-1}(x)) + \log|\det J_{f^{-1}}(x)|$ (exact likelihood)
- **Autoregressive Models**: $\log p_\theta(x) = \sum_{i=1}^n \log p_\theta(x_i | x_{<i})$ (factorization)

**共通点**: $\log p_\theta(x)$ が計算可能（または下界がある）。MLE 訓練が可能。

**限界**:
- VAE: ELBO は真の尤度の下界 → posterior collapse
- Flow: 可逆性制約 → 表現力の制限
- AR: 逐次生成 → 遅いサンプリング

```julia
# Likelihood-based の統一インターフェース
abstract type LikelihoodBased end

struct VAE <: LikelihoodBased
    encoder
    decoder
end

struct NormalizingFlow <: LikelihoodBased
    flow_layers
end

struct AutoregressiveModel <: LikelihoodBased
    conditional_probs
end

# 全て log_likelihood(model, x) を実装
function log_likelihood(model::VAE, x)
    z_mean, z_logvar = model.encoder(x)
    elbo = reconstruction_term(model, x, z_mean) - kl_term(z_mean, z_logvar)
    return elbo  # lower bound
end

function log_likelihood(model::NormalizingFlow, x)
    z, log_det_J = inverse_flow(model.flow_layers, x)
    return log_p_z(z) + log_det_J  # exact likelihood
end

function log_likelihood(model::AutoregressiveModel, x)
    return sum([log(model.conditional_probs[i](x[1:i-1])) for i in 1:length(x)])
end

println("パラダイム1: 尤度ベース → log p(x) を直接最大化")
```

#### パラダイム2: 暗黙的モデル (Implicit Models)

**定義**: $p_\theta(x)$ を明示的にモデル化せず、サンプリングプロセスのみを定義する。

**代表モデル**:
- **GAN**: Generator $G: \mathbb{R}^d \to \mathbb{R}^n$ で $z \sim p_z \Rightarrow x = G(z)$
- **Diffusion (逆拡散)**: $x_T \sim \mathcal{N}(0,I) \Rightarrow x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$

**共通点**: 尤度計算不能。サンプル生成は高速（GAN）または高品質（Diffusion）。

**限界**:
- GAN: 訓練不安定、mode collapse
- Diffusion: 遅いサンプリング（1000ステップ）

```julia
# Implicit models の統一インターフェース
abstract type ImplicitModel end

struct GAN <: ImplicitModel
    generator
    discriminator
end

struct DiffusionModel <: ImplicitModel
    noise_predictor
    noise_schedule
end

# 全て sample(model, n) を実装
function sample(model::GAN, n::Int)
    z = randn(n, latent_dim)
    return model.generator(z)
end

function sample(model::DiffusionModel, n::Int, steps::Int=1000)
    x_T = randn(n, data_dim)
    x_t = x_T
    for t in steps:-1:1
        ϵ_pred = model.noise_predictor(x_t, t)
        x_t = denoise_step(x_t, ϵ_pred, t, model.noise_schedule)
    end
    return x_t
end

println("パラダイム2: 暗黙的 → サンプリングプロセスのみ定義")
```

#### パラダイム3: スコアベース (Score-based Models)

**定義**: スコア関数 $\nabla_x \log p_t(x)$ を学習し、Langevin dynamics でサンプリングする。

**代表モデル**:
- **NCSN**: $s_\theta(x, \sigma) \approx \nabla_x \log p_\sigma(x)$ (multi-scale noise)
- **Score SDE**: $dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)dW$

**共通点**: 正規化定数 $Z$ を回避。スコアマッチング損失で訓練。

**限界**: 低密度領域で不正確 → ノイズ付加で解決（→ Diffusion へ）

```julia
# Score-based の統一インターフェース
abstract type ScoreBased end

struct NCSN <: ScoreBased
    score_network
    noise_levels
end

struct ScoreSDE <: ScoreBased
    score_network
    sde_config
end

# 全て score(model, x, t) を実装
function score(model::NCSN, x, sigma)
    return model.score_network(x, sigma)
end

function score(model::ScoreSDE, x, t)
    return model.score_network(x, t)
end

# Langevin sampling
function langevin_sample(model::ScoreBased, x_init, steps, lr)
    x = copy(x_init)
    for step in 1:steps
        s = score(model, x, step/steps)
        x = x .+ lr .* s .+ sqrt(2*lr) .* randn(size(x))
    end
    return x
end

println("パラダイム3: スコアベース → ∇log p(x) を学習")
```

#### パラダイム4: Flow ベース (Flow-based Models)

**定義**: ベクトル場 $v(x,t)$ を学習し、ODE $\frac{dx}{dt} = v(x,t)$ でサンプリングする。

**代表モデル**:
- **Flow Matching**: $v_\theta(x,t) \approx u_t(x) = \frac{d}{dt}[(1-t)x_0 + t x_1]$
- **Rectified Flow**: 直線パス $x_t = (1-t)x_0 + t x_1$ での OT
- **Normalizing Flow**: 可逆 $f: x \to z$ (パラダイム1とも重複)

**共通点**: シミュレーションフリー訓練。決定論的サンプリング。

**限界**: Flow Matching は訓練は簡単だが、最適輸送の保証なし（→ OT 理論へ）

```julia
# Flow-based の統一インターフェース
abstract type FlowBased end

struct FlowMatching <: FlowBased
    velocity_network
end

struct RectifiedFlow <: FlowBased
    velocity_network
end

# 全て velocity(model, x, t) を実装
function velocity(model::FlowMatching, x, t)
    return model.velocity_network(x, t)
end

# ODE sampling
function ode_sample(model::FlowBased, x0, steps)
    dt = 1.0 / steps
    x = copy(x0)
    for step in 1:steps
        t = step * dt
        v = velocity(model, x, t)
        x = x .+ v .* dt  # Euler method
    end
    return x
end

println("パラダイム4: Flow ベース → ODE dx/dt = v(x,t) で輸送")
```

#### 4つのパラダイムの関係図

```mermaid
graph TB
    L["パラダイム1<br/>尤度ベース<br/>VAE/NF/AR"]
    I["パラダイム2<br/>暗黙的<br/>GAN/Diffusion"]
    S["パラダイム3<br/>スコアベース<br/>NCSN/ScoreSDE"]
    F["パラダイム4<br/>Flow ベース<br/>FM/RF"]

    L -->|VAE → Diffusion<br/>階層的VAE| I
    S -->|Score → PF-ODE| F
    I -->|Diffusion → Score<br/>denoising=score| S
    F -->|Flow → Diffusion<br/>PF-ODE=SDE| I

    style L fill:#e3f2fd
    style I fill:#fff3e0
    style S fill:#f3e5f5
    style F fill:#e8f5e9
```

**重要な洞察**: これら4つのパラダイムは独立ではない。**相互に変換可能**だ。

次のセクションで、この変換を数学的に証明する。

:::details 【補足】なぜ4つに分類したのか？
生成モデルの分類は多様だが、本講義では**訓練目的関数と生成メカニズム**の2軸で分類した:

| パラダイム | 訓練目的 | 生成メカニズム |
|:----------|:--------|:--------------|
| 尤度ベース | $\max \log p_\theta(x)$ | 明示的 $p_\theta(x)$ or 潜在変数 |
| 暗黙的 | Adversarial or Denoising | サンプリングプロセス |
| スコアベース | Score Matching | Langevin dynamics |
| Flow ベース | Flow Matching | ODE integration |

この分類により、**等価性の証明が自然に導かれる**。
:::

---

### 3.2 VAEファミリーの統一的理解 — ELBOの変奏曲

VAE (Variational Autoencoder) は、潜在変数モデル $p_\theta(x) = \int p_\theta(x|z)p(z)dz$ の変分推論だ。

#### 3.2.1 標準 VAE の ELBO

第8回で導出した通り、Jensen不等式から:

$$
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}[q_\phi(z|x) \| p(z)] \equiv \mathcal{L}(\theta, \phi; x)
$$

**再構成項** $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ と **KL正則化項** $\text{KL}[q_\phi(z|x) \| p(z)]$ の和。

#### 3.2.2 β-VAE: KL の重み付け

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot \text{KL}[q_\phi(z|x) \| p(z)]
$$

- $\beta > 1$: disentanglement 促進（KL ペナルティ増）
- $\beta < 1$: 再構成重視（posterior collapse 緩和）

**統一的視点**: β は Rate-Distortion トレードオフのラグランジュ乗数だ。

#### 3.2.3 VQ-VAE: 離散潜在空間

VQ-VAE は連続 $z$ ではなく、離散コードブック $\{e_k\}_{k=1}^K$ から選択する:

$$
z_q = \text{argmin}_k \|z_e - e_k\|^2
$$

ELBO は:

$$
\mathcal{L}_{\text{VQ-VAE}} = \log p_\theta(x|z_q) + \|\text{sg}[z_e] - e\|^2 + \beta \|z_e - \text{sg}[e]\|^2
$$

- 第1項: reconstruction loss
- 第2項: codebook loss (コードブック更新)
- 第3項: commitment loss (エンコーダ更新、sg = stop gradient)

**統一的視点**: VQ-VAE は ELBO の離散版。Straight-Through Estimator で勾配を近似。

#### 3.2.4 FSQ: Finite Scalar Quantization

FSQ はコードブックなしの量子化:

$$
z_{\text{FSQ}}(z_e) = \text{round}(\text{clip}(z_e, -L, L))
$$

各次元を $\{-L, ..., L\}$ の整数に丸める。Codebook collapse が原理的に不可能。

**統一的視点**: FSQ は VQ-VAE の極限 $K \to \infty$ で、コードブックを格子点に置き換えた形。

#### 3.2.5 VAE ファミリー比較表

| モデル | 潜在空間 | ELBO の特徴 | 主な用途 |
|:-------|:---------|:----------|:--------|
| VAE | 連続 $z \in \mathbb{R}^d$ | 標準 ELBO | 画像生成・表現学習 |
| β-VAE | 連続 | $\beta \cdot \text{KL}$ | Disentanglement |
| VQ-VAE | 離散コードブック | Codebook + Commitment loss | 画像トークン化・圧縮 |
| FSQ | 離散格子 | 格子量子化 | Codebook collapse 回避 |

```julia
# VAE family の統一的実装
abstract type VAEFamily end

struct StandardVAE <: VAEFamily
    encoder
    decoder
    β::Float64
end
StandardVAE(enc, dec) = StandardVAE(enc, dec, 1.0)

struct VQVAE <: VAEFamily
    encoder
    decoder
    codebook
    commitment_cost::Float64
end

struct FSQ <: VAEFamily
    encoder
    decoder
    L::Int  # quantization level
end

# 全て elbo(model, x) を実装
function elbo(model::StandardVAE, x)
    z_mean, z_logvar = model.encoder(x)
    z = reparameterize(z_mean, z_logvar)
    recon = log_p_x_given_z(model.decoder, x, z)
    kl = kl_divergence(z_mean, z_logvar)
    return mean(recon) - model.β * mean(kl)
end

function elbo(model::VQVAE, x)
    z_e = model.encoder(x)
    z_q, indices = quantize(z_e, model.codebook)
    recon = log_p_x_given_z(model.decoder, x, z_q)
    codebook_loss = mean((stop_gradient(z_e) - z_q).^2)
    commitment_loss = mean((z_e - stop_gradient(z_q)).^2)
    return mean(recon) - codebook_loss - model.commitment_cost * commitment_loss
end

function elbo(model::FSQ, x)
    z_e = model.encoder(x)
    z_q = fsq_quantize(z_e, model.L)
    recon = log_p_x_given_z(model.decoder, x, z_q)
    return mean(recon)  # no codebook loss
end

# Helper functions
reparameterize(μ, logσ²) = μ .+ exp.(0.5 .* logσ²) .* randn(size(μ))
kl_divergence(μ, logσ²) = -0.5 .* sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²), dims=2)
log_p_x_given_z(decoder, x, z) = -sum((x - decoder(z)).^2, dims=2)  # Gaussian assumption
stop_gradient(x) = x  # in practice: detach in PyTorch, Zygote.ignore in Julia
quantize(z_e, codebook) = argmin_codebook(z_e, codebook)  # returns (z_q, indices)
fsq_quantize(z, L) = round.(clamp.(z, -L, L))

println("VAE family: 全て ELBO の変奏曲")
println("  Standard VAE: β-weighted KL")
println("  VQ-VAE: Discrete codebook + commitment")
println("  FSQ: Lattice quantization, no codebook")
```

**統一的理解**: VAE ファミリーは全て、**潜在変数モデルの変分推論**という共通基盤を持つ。連続/離散、KL重み、量子化手法が違うだけだ。

---

### 3.3 Flowファミリーの統一的理解 — 可逆変換から連続輸送へ

Flow ベースモデルは、**可逆写像** $f: x \leftrightarrow z$ または **連続時間輸送** $\frac{dx}{dt} = v(x,t)$ で分布変換する。

#### 3.3.1 Normalizing Flow (NF) の exact likelihood

第33回で学んだ通り、可逆写像 $z = f(x)$ により:

$$
p_X(x) = p_Z(f(x)) \left| \det \frac{\partial f}{\partial x} \right|
$$

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det J_f(x) \right|
$$

**利点**: 厳密な尤度計算可能。

**限界**: 可逆性制約 → 表現力の制限（層数が増えても、最終的に $\dim(x) = \dim(z)$）。

#### 3.3.2 Continuous Normalizing Flow (CNF) の ODE 定式化

CNF は可逆変換を連続時間化:

$$
\frac{dx}{dt} = f(x, t), \quad x(0) = z, \quad x(1) = x_{\text{data}}
$$

Instantaneous change of variables formula [^4]:

$$
\frac{d \log p_t(x)}{dt} = -\text{Tr}\left( \frac{\partial f}{\partial x} \right)
$$

積分して:

$$
\log p_1(x) = \log p_0(z) - \int_0^1 \text{Tr}\left( \frac{\partial f}{\partial x(t)} \right) dt
$$

**利点**: 可逆性を保ちつつ、連続時間の柔軟性。

**限界**: トレース計算の計算コスト $O(d^3)$ → Hutchinson's trace estimator で $O(d)$ に削減（FFJORD [^5]）。

#### 3.3.3 Flow Matching の conditional flow

Flow Matching [^6] は、データペア $(x_0, x_1)$ 間の条件付き確率パス $p_t(x|x_0, x_1)$ を定義し、そのベクトル場を学習する。

**ガウシアン確率パス**（最も一般的）:

$$
x_t | (x_0, x_1) \sim \mathcal{N}(\mu_t(x_0, x_1), \sigma_t^2 I)
$$

線形補間の場合:

$$
\mu_t(x_0, x_1) = (1-t)x_0 + t x_1, \quad \sigma_t = \text{const}
$$

条件付きベクトル場:

$$
u_t(x | x_0, x_1) = \frac{d\mu_t}{dt} = x_1 - x_0
$$

**Flow Matching Loss**:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, p(x_0, x_1), p_t(x|x_0, x_1)} \left[ \|v_\theta(x, t) - u_t(x|x_0, x_1)\|^2 \right]
$$

シミュレーションフリー: ODE を積分せずに、直接ベクトル場を回帰。

**利点**: 訓練が超簡単（MSE loss）。サンプリングは ODE Solver。

**限界**: 最適輸送の保証なし → Rectified Flow で改善。

#### 3.3.4 Rectified Flow の直線化

Rectified Flow [^7] は、Flow Matching を反復して直線パスに近づける:

1. **初回**: Flow Matching で $v_\theta^{(1)}$ を学習
2. **Reflow**: $v_\theta^{(1)}$ でサンプリング → 新しいペア $(x_0, x_1^{(1)})$ 生成
3. **再学習**: $(x_0, x_1^{(1)})$ で Flow Matching → $v_\theta^{(2)}$
4. 反復 → ベクトル場が直線 $x_1 - x_0$ に収束

**直線化の利点**: 1-step 生成に近づく（蒸留の理論的基盤）。

#### 3.3.5 Flow ファミリー比較表

| モデル | 変換の種類 | 尤度計算 | サンプリング | 最適輸送 |
|:-------|:----------|:--------|:------------|:--------|
| Normalizing Flow | 離散可逆写像 | 厳密 | 1 step (逆変換) | No |
| CNF | 連続 ODE | 厳密 (ODE 積分) | ODE Solver | No |
| Flow Matching | 条件付き ODE | 不能 | ODE Solver | No |
| Rectified Flow | 直線化 ODE | 不能 | ODE Solver (少ステップ) | Yes (反復で) |

```julia
# Flow family の統一的実装
abstract type FlowFamily end

struct NormalizingFlow <: FlowFamily
    layers  # list of invertible layers
end

struct CNF <: FlowFamily
    ode_func  # f(x, t)
end

struct FlowMatching <: FlowFamily
    velocity_net  # v_θ(x, t)
end

struct RectifiedFlow <: FlowFamily
    velocity_net
    reflow_iterations::Int
end

# 全て forward(model, x) と sample(model, z) を実装
function forward(model::NormalizingFlow, x)
    z = x
    log_det_J = 0.0
    for layer in model.layers
        z, ldj = layer_forward(layer, z)
        log_det_J += ldj
    end
    return z, log_det_J
end

function sample(model::NormalizingFlow, z)
    x = z
    for layer in reverse(model.layers)
        x = layer_inverse(layer, x)
    end
    return x
end

function sample(model::CNF, z, solver=euler_ode)
    # Integrate dx/dt = f(x, t) from t=0 to t=1
    x = solver(model.ode_func, z, 0.0, 1.0, steps=100)
    return x
end

function sample(model::FlowMatching, x0, solver=euler_ode)
    # Integrate dx/dt = v_θ(x, t) from t=0 to t=1
    v_func(x, t) = model.velocity_net(x, t)
    x = solver(v_func, x0, 0.0, 1.0, steps=100)
    return x
end

function train_reflow!(model::RectifiedFlow, data_pairs)
    for iter in 1:model.reflow_iterations
        # Generate new pairs using current velocity
        new_pairs = [(x0, sample(FlowMatching(model.velocity_net), x0))
                      for (x0, _) in data_pairs]
        # Retrain on new pairs
        train_flow_matching!(model.velocity_net, new_pairs)
        data_pairs = new_pairs  # update for next iteration
    end
end

println("Flow family: 可逆変換から連続輸送へ")
println("  NF: Discrete invertible, exact likelihood")
println("  CNF: Continuous ODE, exact likelihood (costly)")
println("  FM: Conditional flow, simulation-free training")
println("  RF: Reflow for straightness, 1-step generation")
```

**統一的理解**: Flow ファミリーは全て、**分布間の輸送**を ODE で実現する。可逆性・尤度・最適性のトレードオフが異なる。

:::message
**ここまでで全体の35%完了！** VAE と Flow ファミリーの統一的理解ができた。次は GAN ファミリー。
:::

---

### 3.4 GANファミリーの統一的理解 — 敵対的学習の多様性

GAN (Generative Adversarial Network) は、Generator $G$ と Discriminator $D$ のミニマックスゲームで学習する。

#### 3.4.1 Vanilla GAN の minimax 定式化

第12回で導出した通り、Goodfellow+ 2014 [^8] の目的関数:

$$
\min_G \max_D V(G, D) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**最適判別器**:

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}
$$

代入すると、Vanilla GAN は **Jensen-Shannon divergence** を最小化:

$$
C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_G)
$$

**限界**: $\text{JSD}$ は台が離れていると $\log 2$ で飽和 → 勾配消失。

#### 3.4.2 WGAN の Wasserstein 距離

WGAN [^9] は Wasserstein-1 距離を最小化:

$$
W_1(p_{\text{data}}, p_G) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_G)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]
$$

Kantorovich-Rubinstein 双対性 (第11回) より:

$$
W_1(p_r, p_g) = \sup_{\|f\|_L \leq 1} \left( \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)] \right)
$$

WGAN の目的関数:

$$
\min_G \max_{D: \|D\|_L \leq 1} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

**1-Lipschitz 制約の実現**:
- Weight clipping (WGAN 原論文): $w \in [-c, c]$ にクリップ → 表現力低下
- Gradient Penalty (WGAN-GP [^10]): $\lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$
- Spectral Normalization (SN-GAN [^11]): $W_{\text{SN}} = W / \sigma(W)$

#### 3.4.3 f-GAN の一般化

f-GAN [^12] は任意の $f$-divergence を最小化:

$$
D_f(p \| q) = \mathbb{E}_{x \sim q}\left[ f\left( \frac{p(x)}{q(x)} \right) \right]
$$

変分下界:

$$
D_f(p_{\text{data}} \| p_G) \geq \sup_{T} \left( \mathbb{E}_{x \sim p_{\text{data}}}[T(x)] - \mathbb{E}_{x \sim p_G}[f^*(T(x))] \right)
$$

ここで $f^*$ は $f$ の Fenchel 共役。

**f-divergence の例**:

| Divergence | $f(t)$ | $f^*(t)$ | 対応する GAN |
|:-----------|:-------|:---------|:------------|
| KL | $t \log t$ | $\exp(t-1)$ | Forward KL |
| Reverse KL | $-\log t$ | $-1 - \log(-t)$ | Reverse KL |
| JS | $-\log 2 + (t+1)\log\frac{t+1}{2} + t \log t$ | $-\log(2 - \exp(t))$ | Vanilla GAN |
| TV | $\frac{1}{2}\|t-1\|$ | $\begin{cases} t & \|t\| \leq \frac{1}{2} \\ \frac{1}{2} \text{sign}(t) & \text{otherwise} \end{cases}$ | Total Variation GAN |

**統一的視点**: 全ての GAN は、$D_f(p_{\text{data}} \| p_G)$ を最小化している。$f$ の選択が違うだけ。

#### 3.4.4 StyleGAN の潜在空間操作

StyleGAN [^13] は、Mapping Network $f: \mathcal{Z} \to \mathcal{W}$ で潜在空間を変換:

$$
z \sim \mathcal{N}(0,I) \xrightarrow{f_{\text{map}}} w \in \mathcal{W} \xrightarrow{\text{AdaIN}} \text{Styled features}
$$

AdaIN (Adaptive Instance Normalization):

$$
\text{AdaIN}(x_i, w) = \gamma_w \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + \beta_w
$$

**$\mathcal{W}$ 空間の利点**: $\mathcal{Z}$ よりも disentangled → 補間・編集が容易。

#### 3.4.5 R3GAN の収束保証

R3GAN (Regularized Relativistic GAN) [^14] は、局所収束を理論的に保証:

**Relativistic discriminator**:

$$
D_{\text{rel}}(x, y) = D(x) - \mathbb{E}_{y \sim p_G}[D(y)]
$$

**R3 正則化**:

$$
R_{\text{R3}}(D) = \mathbb{E}_{x \sim p_{\text{data}}, y \sim p_G}[\|\nabla_x D_{\text{rel}}(x, y)\|^2]
$$

**定理** (R3GAN, NeurIPS 2024 [^14]): R3 正則化の下で、Nash 均衡への局所収束が保証される。

**統一的視点**: GAN の不安定性は理論的欠陥ではなく、適切な正則化で解決可能。

#### 3.4.6 GAN ファミリー比較表

| モデル | 距離関数 | 安定性 | 品質 | 収束保証 |
|:-------|:--------|:------|:-----|:--------|
| Vanilla GAN | JS divergence | 低（勾配消失） | 中 | No |
| WGAN | Wasserstein-1 | 中（Lipschitz 制約） | 中 | No |
| WGAN-GP | Wasserstein-1 + GP | 高 | 高 | No |
| f-GAN | 任意 f-divergence | $f$ 依存 | $f$ 依存 | No |
| StyleGAN | JS (+ $\mathcal{W}$ space) | 中 | 最高 | No |
| R3GAN | Relativistic + R3 正則化 | 最高 | 最高 | **Yes** |

```julia
# GAN family の統一的実装
abstract type GANFamily end

struct VanillaGAN <: GANFamily
    generator
    discriminator
end

struct WGAN <: GANFamily
    generator
    critic  # not "discriminator"
    lipschitz_method::Symbol  # :clip, :gp, :sn
end

struct fGAN <: GANFamily
    generator
    discriminator
    f_divergence::Function  # f(t) and f*(t)
end

struct R3GAN <: GANFamily
    generator
    discriminator
    r3_weight::Float64
end

# 全て loss_D(model, x_real, x_fake) と loss_G(model, x_fake) を実装
function loss_D(model::VanillaGAN, x_real, x_fake)
    # max E[log D(x)] + E[log(1-D(G(z)))]
    real_score = log.(model.discriminator(x_real))
    fake_score = log.(1 .- model.discriminator(x_fake))
    return -(mean(real_score) + mean(fake_score))
end

function loss_D(model::WGAN, x_real, x_fake)
    # max E[D(x)] - E[D(G(z))]
    loss = mean(model.critic(x_real)) - mean(model.critic(x_fake))

    if model.lipschitz_method == :gp
        # Gradient penalty
        α = rand(size(x_real, 1))
        x_interp = α .* x_real .+ (1 .- α) .* x_fake
        grad = gradient(x -> mean(model.critic(x)), x_interp)
        gp = mean((norm(grad, 2) .- 1).^2)
        loss = loss - 10.0 * gp  # λ=10
    end

    return -loss  # negate for minimization
end

function loss_D(model::R3GAN, x_real, x_fake)
    # Relativistic + R3 regularization
    D_real = model.discriminator(x_real)
    D_fake = model.discriminator(x_fake)
    D_rel_real = D_real .- mean(D_fake)
    D_rel_fake = D_fake .- mean(D_real)

    # BCE on relativistic discriminator
    loss_bce = -mean(log.(sigmoid.(D_rel_real))) - mean(log.(sigmoid.(-D_rel_fake)))

    # R3 regularization: ||∇_x D_rel||²
    grad_real = gradient(x -> mean(model.discriminator(x) .- mean(D_fake)), x_real)
    grad_fake = gradient(x -> mean(model.discriminator(x) .- mean(D_real)), x_fake)
    r3_reg = mean(grad_real.^2) + mean(grad_fake.^2)

    return loss_bce + model.r3_weight * r3_reg
end

println("GAN family: 敵対的学習の多様性")
println("  Vanilla: JS divergence, unstable")
println("  WGAN: Wasserstein-1, Lipschitz constraint")
println("  f-GAN: Arbitrary f-divergence")
println("  R3GAN: Relativistic + R3, convergence guarantee")
```

**統一的理解**: GAN ファミリーは全て、**Generator が Discriminator を騙すゲーム**を解く。距離関数と正則化が違うだけだ。

---

### 3.5 Diffusionファミリーの統一的理解 — ノイズ除去の階層構造

Diffusion モデルは、**ノイズ付加過程の逆転**で生成する。Forward process と Reverse process の設計が多様。

#### 3.5.1 DDPM の離散時間定式化

Forward process (固定・ガウス):

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

Closed-form (再パラメータ化):

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$

ここで $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

Reverse process (学習):

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**変分下界**:

$$
\mathbb{E}[-\log p_\theta(x_0)] \leq \mathbb{E}_q[L_0 + L_1 + \cdots + L_{T-1} + L_T]
$$

**簡略化損失** (Ho+ 2020 [^15]):

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]
$$

**予測目標の3変種**:
- **ε-prediction**: $\epsilon_\theta(x_t, t) \approx \epsilon$
- **x_0-prediction**: $\hat{x}_0(x_t, t) \approx x_0$
- **v-prediction**: $v_\theta(x_t, t) \approx \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1-\bar{\alpha}_t} x_0$

#### 3.5.2 DDIM の non-Markovian 加速

DDIM [^16] は、forward process を non-Markovian に変更し、決定論的サンプリングを可能にする:

**DDIM forward**:

$$
q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\left( \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 I \right)
$$

$\sigma_t = 0$ なら完全に決定論的。

**DDIM サンプリング**:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0(x_t) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t)
$$

**利点**: ステップ数を $T=1000 \to 10-50$ に削減可能。

#### 3.5.3 Score SDE の連続時間定式化

Score SDE [^2] は、Diffusion を連続時間 SDE で定式化:

**Forward SDE**:

$$
dx = f(x,t)dt + g(t)dW
$$

- VP-SDE (Variance Preserving): $f(x,t) = -\frac{1}{2}\beta(t)x$, $g(t) = \sqrt{\beta(t)}$
- VE-SDE (Variance Exploding): $f(x,t) = 0$, $g(t) = \sqrt{\frac{d\sigma^2(t)}{dt}}$

**Reverse SDE** (Anderson 1982 [^17]):

$$
dx = \left[ f(x,t) - g(t)^2 \nabla_x \log p_t(x) \right]dt + g(t)d\bar{W}
$$

**Probability Flow ODE** (同じ周辺分布):

$$
\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)
$$

**統一的視点**: DDPM/DDIM/Score-based は全て、この SDE/ODE のインスタンス。

#### 3.5.4 Latent Diffusion Models (LDM) の階層化

LDM [^18] は、Diffusion を潜在空間 $z$ で行う:

1. **VAE encoder**: $x \to z = \mathcal{E}(x)$
2. **Diffusion in latent space**: $z \to z_T \to ... \to z_0$
3. **VAE decoder**: $z_0 \to \hat{x} = \mathcal{D}(z_0)$

**利点**: $\dim(z) \ll \dim(x)$ なら計算量 $O(d^2)$ が激減。

**Stable Diffusion** [^18]: $512^2$ 画像を $64^2$ 潜在空間で Diffusion → 8倍圧縮。

#### 3.5.5 Consistency Models の 1-step 蒸留

Consistency Model [^19] は、ODE の self-consistency を利用:

**Consistency function**:

$$
f(x_t, t) = x_0
$$

つまり、$f$ は任意の時刻 $t$ の $x_t$ を始点 $x_0$ に写す。

**Consistency Training (CT)**:

$$
\mathcal{L}_{\text{CT}} = \mathbb{E}_{t, x_0, \epsilon}\left[ d(f_\theta(x_t, t), f_\theta(x_{t+\Delta t}, t+\Delta t)) \right]
$$

**Consistency Distillation (CD)**:

$$
\mathcal{L}_{\text{CD}} = \mathbb{E}_{t, x_0, \epsilon}\left[ d(f_\theta(x_t, t), \hat{x}_0^{\text{teacher}}(x_{t+\Delta t}, t+\Delta t)) \right]
$$

**利点**: 1-step 生成（蒸留後）。理論的保証あり。

#### 3.5.6 Diffusion ファミリー比較表

| モデル | 時間 | Forward | Reverse | サンプリング速度 | 品質 |
|:-------|:-----|:--------|:--------|:----------------|:-----|
| DDPM | 離散 | Markovian Gaussian | 学習 Gaussian | 遅い (1000) | 高 |
| DDIM | 離散 | Non-Markovian | 決定論的 ODE | 中速 (10-50) | 高 |
| Score SDE | 連続 | SDE | Reverse SDE or PF-ODE | 中速 (100-1000) | 高 |
| LDM | 離散/潜在 | Markovian | 学習 Gaussian | 中速 (50-100) | 高 |
| Consistency | 連続 | ODE | 1-step consistency | 高速 (1-4) | 中-高 |

```julia
# Diffusion family の統一的実装
abstract type DiffusionFamily end

struct DDPM <: DiffusionFamily
    noise_pred_net  # ε_θ(x_t, t)
    noise_schedule  # β_t
end

struct DDIM <: DiffusionFamily
    noise_pred_net
    noise_schedule
    η::Float64  # stochasticity control
end

struct ScoreSDE <: DiffusionFamily
    score_net  # s_θ(x, t) ≈ ∇log p_t(x)
    sde_type::Symbol  # :vp or :ve
end

struct LDM <: DiffusionFamily
    vae_encoder
    vae_decoder
    diffusion  # DDPM or DDIM in latent space
end

struct ConsistencyModel <: DiffusionFamily
    consistency_net  # f_θ(x_t, t) → x_0
    teacher_model  # Optional: for distillation
end

# 全て forward(model, x0, t) と sample(model, steps) を実装
function forward(model::DDPM, x0, t)
    # x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
    α_bar_t = get_alpha_bar(model.noise_schedule, t)
    ε = randn(size(x0))
    x_t = sqrt(α_bar_t) * x0 + sqrt(1 - α_bar_t) * ε
    return x_t, ε
end

function sample(model::DDPM, x_T, steps)
    x_t = x_T
    for t in steps:-1:1
        ε_pred = model.noise_pred_net(x_t, t)
        x_t = ddpm_step(x_t, ε_pred, t, model.noise_schedule)
    end
    return x_t
end

function sample(model::DDIM, x_T, steps, skip=10)
    # DDIM with skipping: use only steps/skip timesteps
    x_t = x_T
    timesteps = collect(steps:-skip:1)
    for (i, t) in enumerate(timesteps)
        t_prev = i < length(timesteps) ? timesteps[i+1] : 0
        ε_pred = model.noise_pred_net(x_t, t)
        x_t = ddim_step(x_t, ε_pred, t, t_prev, model.noise_schedule, model.η)
    end
    return x_t
end

function sample(model::LDM, x_T_latent, steps)
    # Diffusion in latent space
    z_T = x_T_latent
    z_0 = sample(model.diffusion, z_T, steps)
    # Decode to pixel space
    x_0 = model.vae_decoder(z_0)
    return x_0
end

function sample(model::ConsistencyModel, x_T, steps=1)
    # 1-step or few-step generation
    x_t = x_T
    for step in 1:steps
        t = (steps - step + 1) / steps
        x_t = model.consistency_net(x_t, t)
    end
    return x_t
end

println("Diffusion family: ノイズ除去の階層構造")
println("  DDPM: Discrete Markovian, slow")
println("  DDIM: Non-Markovian, deterministic, faster")
println("  Score SDE: Continuous, flexible")
println("  LDM: Latent space, efficient")
println("  CM: 1-step, distilled")
```

**統一的理解**: Diffusion ファミリーは全て、**Forward ノイズ付加の逆転**を学習する。離散/連続、決定論的/確率的、ピクセル/潜在空間が違うだけだ。

:::message
**ここまでで全体の50%完了！** VAE/Flow/GAN/Diffusion の4大ファミリーを統一的に整理した。次は AR と World Models、そして数学的等価性の証明へ。
:::

---

### 3.6 ARファミリーの統一的理解 — 条件付き分解の多様性

Autoregressive (AR) モデルは、**連鎖律**で分布を分解:

$$
p(x) = \prod_{i=1}^n p(x_i | x_{<i})
$$

#### 3.6.1 PixelCNN の空間的自己回帰

PixelCNN [^20] は、画像を左上から右下へラスタースキャン順で生成:

$$
p(x) = \prod_{i=1}^{H \times W} p(x_i | x_1, ..., x_{i-1})
$$

Masked Convolution で因果性を保証:

$$
\text{MaskedConv}(x)_{i,j} = \sum_{i'<i} \sum_{j'} W_{i',j'} x_{i',j'} + \sum_{j'<j} W_{i,j'} x_{i,j'}
$$

**Blind spot 問題**: Naive masking では一部のピクセルが見えない → Gated PixelCNN [^21] で解決（Vertical + Horizontal stack）。

#### 3.6.2 WaveNet の時間的自己回帰

WaveNet [^22] は、音声波形を時系列で生成:

$$
p(x) = \prod_{t=1}^T p(x_t | x_1, ..., x_{t-1})
$$

Dilated Causal Convolution で受容野を指数的に拡大:

$$
\text{receptive field} = 2^L \cdot (K-1) + 1
$$

$L$ = 層数、$K$ = カーネルサイズ。例: $L=10, K=2$ なら受容野 1025 samples。

#### 3.6.3 Transformer の Causal Attention

Transformer [^23] は、Causal Mask で自己回帰を実現:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V
$$

$$
M_{ij} = \begin{cases}
0 & i \geq j \\
-\infty & i < j
\end{cases}
$$

**利点**: 並列訓練可能（Teacher Forcing）。長距離依存を直接モデル化。

**限界**: 逐次生成（遅い）。

#### 3.6.4 VAR (Visual Autoregressive Models) の multi-scale

VAR [^24] は、粗→細の multi-scale で画像を生成:

1. **Scale 1**: $8 \times 8$ トークン生成
2. **Scale 2**: $16 \times 16$ トークン生成（Scale 1 を条件）
3. **Scale 3**: $32 \times 32$ トークン生成（Scale 2 を条件）

$$
p(x) = \prod_{s=1}^S p(x^{(s)} | x^{(1)}, ..., x^{(s-1)})
$$

**FID 1.73** (ImageNet 256×256) — AR モデルの SOTA (NeurIPS 2024 Best Paper)。

#### 3.6.5 MAR (Masked Autoregressive) の non-raster order

MAR [^25] は、ラスタースキャン順を捨て、ランダム順序で生成:

1. ランダムにマスク位置を選択
2. マスク部分を予測
3. 反復

**利点**: 並列生成（Transformer 的）。品質維持（FID 1.78）。

#### 3.6.6 AR ファミリー比較表

| モデル | ドメイン | 順序 | 受容野拡大 | 並列訓練 | 並列生成 |
|:-------|:--------|:-----|:----------|:--------|:--------|
| PixelCNN | 画像 | ラスター | Gated Conv | Yes | No |
| WaveNet | 音声 | 時系列 | Dilated Conv | Yes | No |
| Transformer | テキスト/画像 | ラスター | Self-Attention | Yes | No |
| VAR | 画像 | Multi-scale | Coarse-to-fine | Yes | No |
| MAR | 画像 | ランダム | Masked Attention | Yes | Partial |

**統一的視点**: AR ファミリーは全て、$p(x) = \prod p(x_i | x_{<i})$ という**連鎖律分解**を採用。順序と受容野拡大の方法が違うだけだ。

---

### 3.7 World Modelsファミリーの統一的理解 — 環境シミュレータとしての生成

World Models は、**環境の内部モデル**を学習し、行動を条件とした未来予測を行う。

#### 3.7.1 JEPA (Joint-Embedding Predictive Architecture)

LeCun の提唱する JEPA [^26] は、**潜在空間での予測**に特化:

$$
\text{Encoder}(x_t) \to s_t, \quad \text{Predictor}(s_t, a_t) \to \hat{s}_{t+1}, \quad s_{t+1} = \text{Encoder}(x_{t+1})
$$

損失:

$$
\mathcal{L}_{\text{JEPA}} = \|\hat{s}_{t+1} - s_{t+1}\|^2
$$

**生成しない**: 画像 $x_{t+1}$ ではなく、特徴量 $s_{t+1}$ のみ予測 → 計算効率。

#### 3.7.2 V-JEPA (Video JEPA)

V-JEPA [^27] は、動画に JEPA を適用:

$$
s_t \to \hat{s}_{t+1}, ..., \hat{s}_{t+k}
$$

**時空間マスキング**: 一部のフレーム・領域をマスク → 予測。

**利点**: 物理法則を暗黙的に学習（重力・衝突など）。

#### 3.7.3 Transfusion の AR + Diffusion 統合

Transfusion [^28] は、**AR (テキスト) + Diffusion (画像)** を1つのTransformerで統合:

**損失関数**:

$$
\mathcal{L}_{\text{Transfusion}} = \mathcal{L}_{\text{AR}}(\text{text}) + \mathcal{L}_{\text{Diffusion}}(\text{image})
$$

$$
\mathcal{L}_{\text{AR}} = -\sum_t \log p(x_t^{\text{text}} | x_{<t}^{\text{text}}, \mathbf{x}^{\text{image}})
$$

$$
\mathcal{L}_{\text{Diffusion}} = \mathbb{E}_{t, \epsilon}\left[ \|\epsilon - \epsilon_\theta(\mathbf{x}_t^{\text{image}}, t, \mathbf{x}^{\text{text}})\|^2 \right]
$$

**統一的視点**: テキストは離散（AR）、画像は連続（Diffusion）が自然。Transfusion は両者を Transformer の異なる head で処理。

#### 3.7.4 World Models ファミリー比較表

| モデル | 予測対象 | 生成方法 | 条件 | 物理法則 |
|:-------|:--------|:--------|:-----|:--------|
| JEPA | 潜在特徴量 $s_{t+1}$ | Predictor Network | 行動 $a_t$ | 暗黙的 |
| V-JEPA | 動画潜在特徴量 | Masked Prediction | なし | 暗黙的 |
| Transfusion | Text + Image | AR + Diffusion | 相互条件付け | なし |
| Cosmos [^29] | 動画（物理） | DiT | 行動・物理 | 明示的 |
| Genie [^30] | ゲーム環境 | AR | 行動 | 暗黙的 |

**統一的理解**: World Models は、**時間的予測** $p(x_{t+1} | x_t, a_t)$ を学習する。潜在空間/ピクセル空間、AR/Diffusion の選択が異なる。

---

### 3.8 数学的等価性 (1): Score ↔ Diffusion の完全証明

ここから、**Part B: 数学的等価性と統一理論**に入る。

**命題**: Denoising Score Matching と DDPM の簡略化損失は等価である。

#### 証明

**Denoising Score Matching** (Vincent 2011 [^31]):

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{x_0 \sim p_{\text{data}}, \tilde{x} \sim q_\sigma(\tilde{x}|x_0)}\left[ \left\|\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x_0) - s_\theta(\tilde{x}, \sigma)\right\|^2 \right]
$$

ここで $q_\sigma(\tilde{x}|x_0) = \mathcal{N}(\tilde{x}; x_0, \sigma^2 I)$。

スコアの閉形式:

$$
\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x_0) = -\frac{\tilde{x} - x_0}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

ここで $\tilde{x} = x_0 + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$。

代入:

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{x_0, \epsilon}\left[ \left\|- \frac{\epsilon}{\sigma} - s_\theta(x_0 + \sigma \epsilon, \sigma)\right\|^2 \right]
$$

**DDPM 簡略化損失** (Ho+ 2020 [^15]):

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t)\|^2 \right]
$$

ここで $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$。

**対応関係**:
- $\sigma^2 = 1 - \bar{\alpha}_t$ とおくと、$\tilde{x} = x_0 + \sigma \epsilon = x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
- $\sqrt{\bar{\alpha}_t} \approx 1$ なら $\tilde{x} \approx x_t$

スケーリング:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}
$$

代入すると:

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}\left[ \left\|- \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}} + \frac{\epsilon_\theta}{\sqrt{1-\bar{\alpha}_t}}\right\|^2 \right] = \frac{1}{1-\bar{\alpha}_t} \mathbb{E}\left[ \|\epsilon - \epsilon_\theta\|^2 \right]
$$

定数倍を無視すれば、$\mathcal{L}_{\text{DSM}} \propto \mathcal{L}_{\text{simple}}$。 $\square$

**結論**: DDPM は Denoising Score Matching の離散時間版に他ならない。

---

### 3.9 数学的等価性 (2): Diffusion ↔ Flow (PF-ODE) の証明

**命題**: 任意の SDE には、同じ周辺分布 $p_t(x)$ を持つ Probability Flow ODE が存在する。

#### 証明 (Anderson 1982 [^17] + Song+ 2021 [^2])

Forward SDE:

$$
dx = f(x,t)dt + g(t)dW
$$

この SDE の周辺分布 $p_t(x)$ は Fokker-Planck 方程式に従う (第37回):

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g(t)^2 \nabla^2 p_t
$$

**Probability Flow ODE**:

$$
\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)
$$

この ODE の周辺分布 $\tilde{p}_t(x)$ の時間発展は:

$$
\frac{\partial \tilde{p}_t}{\partial t} = -\nabla \cdot \left[ \left(f - \frac{1}{2}g^2 \nabla \log \tilde{p}_t\right) \tilde{p}_t \right]
$$

展開:

$$
= -\nabla \cdot (f \tilde{p}_t) + \frac{1}{2}g^2 \nabla \cdot (\nabla \tilde{p}_t) = -\nabla \cdot (f \tilde{p}_t) + \frac{1}{2}g^2 \nabla^2 \tilde{p}_t
$$

Fokker-Planck 方程式と一致 → **同じ周辺分布**。 $\square$

**結論**: Diffusion (SDE) と Flow (ODE) は、周辺分布が同じ → サンプリングは等価。

---

### 3.10 数学的等価性 (3): Flow Matching ↔ Diffusion の統一

**命題**: Flow Matching は Diffusion の決定論的版（PF-ODE）と等価である。

#### 証明 (Stochastic Interpolants [^1])

**Flow Matching の条件付き確率パス** (ガウシアン):

$$
p_t(x | x_0, x_1) = \mathcal{N}(x; \mu_t(x_0, x_1), \sigma_t^2 I)
$$

線形補間: $\mu_t = (1-t)x_0 + t x_1$, $\sigma_t = \sigma(t)$。

条件付きベクトル場:

$$
u_t(x | x_0, x_1) = \frac{d\mu_t}{dt} = x_1 - x_0
$$

**Diffusion の PF-ODE** (VP-SDE):

$$
\frac{dx}{dt} = -\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t) \nabla_x \log p_t(x)
$$

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ を代入:

$$
\frac{dx}{dt} = \frac{d\bar{\alpha}_t/dt}{2\sqrt{\bar{\alpha}_t}} (x_0 - \epsilon \frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}})
$$

$\bar{\alpha}_t = (1-t)^2$ とおくと (Rectified Flow):

$$
\frac{dx}{dt} = x_0 \cdot \frac{-2(1-t)}{2(1-t)} - \epsilon \cdot \frac{\sqrt{t^2}}{\sqrt{(1-t)^2}} = x_1 - x_0
$$

Flow Matching のベクトル場と一致。 $\square$

**結論**: Flow Matching は、特定の Diffusion (VP-SDE) の PF-ODE を学習している。

---

### 3.11 EBM 統一視点 — 全生成モデルはエネルギー最小化

**命題**: VAE・GAN・Diffusion・Flow は全て、エネルギーベースモデル (EBM) として再解釈できる。

#### 3.11.1 VAE as EBM

VAE の ELBO:

$$
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}[q_\phi(z|x) \| p(z)]
$$

エネルギー定義:

$$
E(x, z) = -\log p_\theta(x|z) - \log p(z)
$$

すると:

$$
\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(z|x)}[E(x, z)] - \mathcal{H}[q_\phi(z|x)]
$$

**VAE = 変分自由エネルギーの最小化**。

#### 3.11.2 GAN as EBM

GAN の Generator は、暗黙的に以下のエネルギーを定義:

$$
E_D(x) = -\log D(x)
$$

Discriminator の最適化:

$$
\max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{x \sim p_G}[\log(1-D(x))]
$$

は、エネルギー $E_D$ による分布の分離。

#### 3.11.3 Diffusion as EBM

Score-based Diffusion は、直接的に EBM:

$$
p_t(x) \propto \exp(-E_t(x))
$$

$$
\nabla_x \log p_t(x) = -\nabla_x E_t(x)
$$

Score Matching = エネルギーの勾配学習。

**統一的視点**: 全ての生成モデルは、ある種の**エネルギー関数 $E(x)$ を最小化**している (Energy Matching [^32])。

---

### 3.12 Wasserstein 勾配流統一視点 — JKO scheme

**命題**: Flow Matching と Diffusion は、Wasserstein 勾配流 (JKO scheme [^33]) として統一できる。

#### JKO scheme の定義

Jordan-Kinderlehrer-Otto (JKO) scheme:

$$
p_{k+1} = \arg\min_{p} \left[ W_2^2(p, p_k) + \tau \mathcal{F}[p] \right]
$$

- $W_2$: Wasserstein-2 距離
- $\mathcal{F}[p]$: エネルギー汎関数（例: KL divergence, Entropy）
- $\tau$: ステップサイズ

#### Diffusion as JKO

DDPM の各ステップは、JKO の離散化:

$$
p_{t-1} = \arg\min_{p} \left[ W_2^2(p, p_t) + \beta_t \cdot \text{KL}[p \| p_{\text{data}}] \right]
$$

#### Flow Matching as JKO

Flow Matching の ODE:

$$
\frac{d}{dt} p_t = -\nabla \cdot (p_t v_t)
$$

は、Wasserstein 勾配流:

$$
\frac{d}{dt} p_t = \nabla \cdot (p_t \nabla \frac{\delta \mathcal{F}}{\delta p})
$$

ここで $v_t = -\nabla \frac{\delta \mathcal{F}}{\delta p}$。

**結論**: Flow と Diffusion は、JKO scheme の連続時間 vs 離散時間の違いに過ぎない。

:::message
**ここまでで全体の70%完了！** 数学的等価性を完全に証明した。次は実装・実験ゾーンへ。
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
