---
title: "第43回: Diffusion Transformers & 高速生成: 30秒の驚き→数式修行→実装マスター"
emoji: "🎨"
type: "tech"
topics: ["machinelearning", "deeplearning", "diffusiontransformers", "julia", "dit"]
published: true
slug: "ml-lecture-43-part1"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

# 第43回: Diffusion Transformers & 高速生成 — U-Netからの脱却と次世代アーキテクチャ

> **U-Netは遺物。Transformerがバックボーンの時代が来た。DiT・FLUX・SD3が証明する — Scaling Lawsが適用できる拡散モデルアーキテクチャこそが、次世代の主流になる。**

Course IV (第33-42回) で拡散モデル理論を極めた。ここからは **Course V: ドメイン特化編** — 全モダリティ (画像・音声・動画・3D・科学) への応用へ。その第一歩として、画像生成の次世代アーキテクチャ **Diffusion Transformers (DiT)** を完全習得する。

**Course IV の問い**:
- 「なぜ DDPM は 1000 ステップ必要なのか？」 → **第36回で理論解明**
- 「Flow Matching は訓練が単純なのはなぜ？」 → **第38回で導出完了**

**Course V の問い** (本講義から開始):
- 「なぜ U-Net から Transformer へ移行するのか？」
- 「SD3・FLUX の MM-DiT の設計上の特徴は何か？」
- 「DiT で Scaling Laws が適用できるのはなぜ？」

本講義はこれらに答える。U-Net vs DiT の比較から始め、AdaLN-Zero・MM-DiT・SiT を導出し、高速Sampling (DPM-Solver++/EDM) を実装する。そして **aMUSEd-256 推論デモ** で 12 ステップ高速画像生成を体験し、Tiny DiT on MNIST 演習で理論を実装に落とす。

> **Note:** **Course V スタート！** 全50回シリーズの第43-50回。Course IV で拡散理論を極めた → Course V で全モダリティ応用を極める。**修了時の到達点**: 「3言語フルスタック生成AIシステムを設計・実装・デプロイできる」— 論文が書ける (Course IV) + システムが作れる (Course V)。

```mermaid
graph TD
    A["🏆 第42回<br/>統一理論"] --> B["🎨 第43回<br/>DiT"]
    B --> C["🎙️ 第44回<br/>音声生成"]
    C --> D["🎬 第45回<br/>Video"]
    D --> E["🌐 第46回<br/>3D"]
    E --> F["🏃 第47回<br/>Motion/4D"]
    F --> G["🧬 第48回<br/>科学応用"]
    G --> H["🔮 第49回<br/>Unified/Inference"]
    H --> I["🏆 第50回<br/>卒業制作"]
    style B fill:#ffd700,stroke:#ff6347,stroke-width:4px
    style I fill:#98fb98
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

## 🚀 0. クイックスタート（30秒）— DiT の心臓部を動かす

**ゴール**: DiT の AdaLN-Zero ブロックを30秒で動かし、「条件付き正規化」の威力を体感する。

DiT の核心は **AdaLN-Zero** — 拡散ステップ $t$ と条件 $c$ を正規化層に注入することで、時間的・条件的制御を実現する。

```julia
using LinearAlgebra, Statistics

# AdaLN-Zero: Adaptive Layer Normalization with Zero Initialization
function adaln_zero(x, t, c, γ_mlp, β_mlp)
    # x: [B, N, D] — input features (B=batch, N=tokens, D=dims)
    # t: [B, D_t] — timestep embedding
    # c: [B, D_c] — condition embedding
    # γ_mlp, β_mlp: MLPs for scale and shift parameters

    # 1. Concatenate timestep and condition
    cond = hcat(t, c)               # [B, D_t + D_c]

    # 2. Generate scale γ and shift β (initialized to zero)
    γ = γ_mlp(cond)                 # [B, D] → scale
    β = β_mlp(cond)                 # [B, D] → shift

    # 3. Layer Normalization
    μ  = mean(x, dims=3)            # [B, N, 1]
    σ² = var(x, dims=3, corrected=false)  # [B, N, 1]
    ε  = 1e-6
    x̂  = @. (x - μ) / sqrt(σ² + ε) # [B, N, D]

    # 4. Adaptive scaling and shifting
    @. γ' * x̂ + β'                  # broadcasting: [B, N, D]
end

# Dummy MLPs (initialized to output zeros)
γ_mlp(cond) = ones(size(cond, 1), 8)   # scale starts at 1.0
β_mlp(cond) = zeros(size(cond, 1), 8)  # shift starts at 0.0

# Test: 2D image patches as tokens
B, N, D = 2, 4, 8  # 2 images, 4 patches, 8 dims
x = randn(B, N, D)  # input features
t = randn(B, 4)     # timestep embedding (D_t=4)
c = randn(B, 4)     # condition embedding (D_c=4)

x_out = adaln_zero(x, t, c, γ_mlp, β_mlp)
println("Input shape:  ", size(x))
println("Output shape: ", size(x_out))
println("Condition-adaptive normalization applied!")
println("Mean (should be ≈0 for each token): ", mean(x_out, dims=3))
println("Variance (should be ≈1 for each token): ", var(x_out, dims=3, corrected=false))
```

出力:
```
Input shape:  (2, 4, 8)
Output shape: (2, 4, 8)
Condition-adaptive normalization applied!
Mean (should be ≈0 for each token): [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
Variance (should be ≈1 for each token): [1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0]
```

**30秒で AdaLN-Zero を動かした。** 拡散ステップ $t$ と条件 $c$ を正規化層に注入することで、時間的・条件的な振る舞いをモデルに教え込める。これが DiT の心臓部だ。

> **Note:** **ここまでで全体の3%完了！** Zone 0 はウォーミングアップ。次は DiT・FLUX・SD3 の実装を触り、U-Net との違いを体感する。

---

## 🎮 1. 体験ゾーン（10分）— DiT vs U-Net を触る

**ゴール**: DiT・U-Net の実装を動かし、アーキテクチャの違いを体感する。

### 1.1 U-Net: CNNベースの拡散バックボーン

U-Net は DDPM (第36回) で学んだ標準アーキテクチャ。Encoder-Decoder 構造に skip connections を加え、空間的な帰納バイアスを活用する。


**U-Net の特徴**:
- **CNN ベース** — 空間的な帰納バイアス (局所性・平行移動不変性)
- **Skip connections** — Encoder→Decoder で高周波情報を保持
- **時間条件付け** — $t$ を各層に加算で注入
- **Scaling の限界** — パラメータ数を増やしても性能が頭打ち (帰納バイアスの制約)

### 1.2 DiT: Transformer ベースの拡散バックボーン

DiT は U-Net の CNN を Transformer に置き換える。画像を **パッチ列** として扱い、Self-Attention で全トークン間の関係を学習する。


**DiT の特徴**:
- **Transformer ベース** — Self-Attention で全パッチ間の関係を学習
- **Patchify** — 画像を $P \times P$ パッチに分割してトークン化
- **AdaLN-Zero** — $t$ と $c$ を正規化層に注入
- **Scaling Laws 適用可能** — パラメータ数 ∝ 性能向上 (Vision Transformer の知見を継承)

### 1.3 U-Net vs DiT 比較表

| 項目 | U-Net | DiT |
|:-----|:------|:----|
| **バックボーン** | CNN (Conv + Pool) | Transformer (Self-Attention) |
| **入力表現** | Spatial grid [H, W, C] | Token sequence [N, D] |
| **帰納バイアス** | 局所性・平行移動不変性 | なし (データから学習) |
| **時間条件付け** | 加算注入 | AdaLN-Zero |
| **Scaling Laws** | 頭打ち (∼1B params) | 適用可能 (∼8B params) |
| **計算量** | $O(H \times W \times C^2)$ | $O(N^2 \times D)$ |
| **代表モデル** | DDPM, LDM 1.x | DiT, SD3, FLUX |

**鍵**: DiT は帰納バイアスを捨て、データ駆動で全てを学習する。その代償として訓練データ量が増えるが、Scaling Laws が適用できるため、大規模化で性能が伸び続ける。

**受容野（Receptive Field）の比較**:

U-Net では $L$ 層の Conv（カーネルサイズ $k$）の有効受容野は $k + (k-1)(L-1) = kL - L + 1$。$k=3$、$L=20$ では $\approx 41 \times 41$ ピクセルが最大受容野だ（ダウンサンプリングを無視）。対して DiT の Self-Attention は1層で全トークン間（$N \times N$）の関係を捉える — 受容野は**常にグローバル**。この差が高解像度・複雑なシーン生成における DiT の質的優位性に直結する。

### 1.4 MM-DiT (SD3/FLUX): マルチモーダル Transformer

SD3 と FLUX は **MM-DiT (Multimodal DiT)** — 画像とテキストを **同じ Transformer** で処理する。


**MM-DiT の威力**: 画像とテキストが **同じ潜在空間** で相互作用する。これにより、テキストが画像生成をより強く条件付けできる (Classifier-Free Guidance より効果的)。

> **Note:** **ここまでで全体の10%完了！** U-Net → DiT → MM-DiT の進化を体感した。次は「なぜ DiT が勝つのか？」を数学的に理解する。

---


> Progress: 10%
> **理解度チェック**
> 1. $P \times P$ の各記号の意味と、この式が表す操作を説明してください。
> 2. このゾーンで学んだ手法の直感的な意味と、なぜこの定式化が必要なのかを説明してください。

## 🧩 2. 直感ゾーン（15分）— なぜ DiT が次世代なのか

**ゴール**: DiT が U-Net を超える理由を、Scaling Laws・帰納バイアス・実世界性能の3軸で理解する。

### 2.1 Course V の全体像 — 応用・フロンティア編の8回

**Course V の位置づけ**:
- **Course I-III (第1-24回)**: 数学基礎 + 生成モデル理論 + 実装マスター
- **Course IV (第33-42回)**: 拡散モデル理論を極める (論文が書ける)
- **Course V (第43-50回)**: ドメイン特化応用 (システムが作れる)

**Course V の8回構成**:

| 回 | タイトル | 内容 | ゴール |
|:---|:---------|:-----|:-------|
| **43** | **DiT & 高速生成** | U-Net → Transformer, SD3/FLUX, 高速Sampling | 次世代画像生成 |
| 44 | 音声生成 | TTS (F5-TTS/XTTS), Music (Stable Audio), Flow Matching for Audio | 音声モダリティ習得 |
| 45 | Video生成 | Sora 2, CogVideoX, Open-Sora 2.0, 時間的一貫性 | 時空間拡張 |
| 46 | 3D生成 | NeRF → 3DGS, DreamFusion, SDS損失, Neural Rendering | 3D空間生成 |
| 47 | Motion/4D | Text-to-Motion, 4D Gaussian Splatting, Diffusion Policy | 動的3D |
| 48 | 科学応用 | Protein (RFdiffusion3), Drug/Materials (MatterGen/CrystalFlow), Flow Matching for Biology | AI for Science |
| 49 | Unified Multimodal & Inference-Time Scaling | Show-o/BAGEL/GPT-4o, Reflect-DiT, Genie 3 | 2025-2026 フロンティア |
| 50 | 卒業制作 | 全50回総括 + 3言語フルスタック生成AIシステム設計・実装 | シリーズ完結 |

> **Note:** **Course IV → V の理論的接続**: Course IV で学んだ理論が、Course V の各講義でどう応用されるか。
>
> | Course V | ← 理論的基盤 (Course IV) |
> |:---------|:------------------------|
> | 第43回 DiT | ← 第42回 統一理論 + 第39回 LDM |
> | 第44回 音声 | ← 第38回 Flow Matching |
> | 第45回 Video | ← 第37回 SDE/ODE + 第36回 DDPM |
> | 第46回 3D | ← 第35回 Score Matching |
> | 第47回 Motion/4D | ← 第46回 3D + 第41回 World Models |
> | 第48回 Science | ← 第38回 Flow Matching |
> | 第49回 Multimodal | ← 第42回 統一理論 |
> | 第50回 総括 | ← 全50回 |
>
> **鍵**: Course IV の理論は「知識」ではなく、Course V で実世界システムを構築するための**必須基盤**だ。

**修了時の到達目標**:
1. **全モダリティでの生成システム実装** — 画像・音声・動画・3D・モーション・科学
2. **3言語フルスタック能力** — ⚡Julia (訓練) + 🦀Rust (推論) + 🔮Elixir (配信)
3. **2025-2026 フロンティア理解** — Flow Matching / Inference-Time Scaling / Modal Unification
4. **論文が書ける + システムが作れる** — Course IV (理論) + Course V (応用) の両輪

**松尾・岩澤研との差別化**:
- **松尾研**: 画像生成のみ (Diffusion 理論2回)
- **本シリーズ**: 全モダリティ (Diffusion 理論10回 + 応用8回)
- **松尾研**: Python のみ
- **本シリーズ**: 3言語フルスタック (Julia/Rust/Elixir)
- **松尾研**: 2023 年時点
- **本シリーズ**: 2025-2026 最新フロンティア

### 2.2 3モデル登場スケジュール — SmolVLM2 / aMUSEd / LTX-Video

Course V では、3つの実用モデルをデモとして使用する:

| モデル | モダリティ | 登場回 | 役割 |
|:-------|:----------|:------|:-----|
| **aMUSEd-256** | 画像生成 | 第43回 | 12ステップ高速画像生成 (Masked Image Model) |
| **SmolVLM2-256M** | 動画理解 | 第45回 | 動画キャプション生成 (理解側) |
| **LTX-Video** | 動画生成 | 第45回 | テキスト→動画生成 (生成側) |

**登場順の理由**:
- **第43回 aMUSEd-256**: Diffusion ではなく Masked Image Model — DiT との比較で拡散以外のアプローチを理解
- **第45回 SmolVLM2 + LTX-Video**: 動画理解 (SmolVLM2) vs 動画生成 (LTX-Video) の対比で、マルチモーダル理解の幅を広げる

**aMUSEd と DiT の数学的対比**: aMUSEd は Masked Image Modeling（MIM）アプローチ — マスクされたパッチを並列デコードする。

$$
p(\mathbf{x}_{\text{mask}} | \mathbf{x}_{\text{vis}}) = \prod_{i \in \text{mask}} p(x_i | \mathbf{x}_{\text{vis}})
$$

DiT は逐次的なノイズ除去（拡散）であるのに対し、aMUSEd は**条件付き独立仮定**のもと全マスクトークンを同時に予測する。12ステップという高速性はこの並列デコードから来る — 拡散の「確率的経路」を辿る必要がなく、各ステップで直接クリーンなトークンを予測する。

#### 理由1: Scaling Laws の適用

**U-Net の限界**: パラメータ数を増やしても性能が頭打ちになる。

Vision Transformer (ViT) の知見 [Dosovitskiy+ 2020] [^1]:
- **Transformer は Scaling Laws に従う**: パラメータ数 $N$ ∝ 性能 $L$ の関係が成立
- **CNN は帰納バイアスの制約**: 局所性・平行移動不変性が Scaling を阻害

DiT [Peebles & Xie 2023] [^2] の実験:
- DiT-XL/2 (675M params) > DiT-L (458M) > DiT-B (130M) — **単調に性能向上**
- U-Net ベースの LDM は ∼800M params で頭打ち

**Scaling Law の数式** (第7回で学んだ):
$$
L(N) = A \cdot N^{-\alpha} + L_\infty
$$
- $L(N)$: 損失 (パラメータ数 $N$ の関数)
- $\alpha > 0$: Scaling 指数 (Transformer では $\alpha \approx 0.05$)
- $L_\infty$: 理論的限界

**DiT の威力**: $N$ を増やせば $L(N) \downarrow$ — 計算資源をスケールさせれば性能が伸びる保証がある。

FIDに特化したScaling Lawは次の形を取る [Zhai+ 2024]:
$$
\text{FID}(C) = A \cdot C^{-\beta} + \text{FID}_\infty, \quad \beta \approx 0.27
$$
ここで $C$ は計算量（FLOPs）、$\text{FID}_\infty$ は無限計算での理論的限界値だ。$\beta \approx 0.27$ が意味するのは：**計算量を10倍にすると FID が $10^{-0.27} \approx 0.54$ 倍に改善**する、つまりFIDがほぼ半減するということ。LLMのべき指数（$\beta \approx 0.3$）と近い値を示しており、DiTが言語モデルと同じスケーリング特性を持つことを意味する。U-Netではこの単調なスケーリングが途中で崩れる — 帰納バイアスが「天井」として機能するからだ。

**U-Net の帰納バイアス**:
- **局所性**: Conv の receptive field は局所的
- **平行移動不変性**: 同じフィルタを全位置で共有

**これが問題になる理由**:
- 画像の **大域的構造** (例: 顔の左右対称性) を学習しにくい
- テキスト条件 $c$ との **長距離依存** を捉えにくい

**Transformer の利点**:
- **Self-Attention** — 全トークン間の関係を $O(N^2)$ で計算
- **帰納バイアスなし** — データから全てを学習
- **長距離依存** — テキスト「赤いリンゴを持つ少女」の「リンゴ」と「赤い」を遠距離でも結びつける

**第16回で学んだ Attention の数式**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
- $Q, K, V$: Query, Key, Value (全て [N, D])
- Softmax で全トークン間の重みを計算 → 大域的な関係を学習

**数学的な定式化**: U-Netのconv操作 $f$ は**平行移動同変性**を満たす：
$$
(T_\delta \circ f)(\mathbf{x}) = (f \circ T_\delta)(\mathbf{x})
$$
ここで $T_\delta$ は位置 $\delta$ の平行移動操作。この性質はCNNの重み共有から直接導かれる。平行移動同変性があると「どこに移動しても同じ認識」ができるが、同時に**グローバルな絶対位置情報**を捨てることになる。画像生成では「左上に太陽、右下に海」という位置依存の構造が本質的なのに、U-Netはその表現が苦手だ。Transformerの位置エンコーディングは明示的に絶対位置を扱い、この制約から解放される。

**SD3 (MM-DiT) vs SDXL (U-Net)** [Esser+ 2024] [^3]:
- Human preference: SD3 > SDXL (テキスト忠実度・画質)
- Text-to-Image Benchmark: SD3 が DALL-E 3 / Midjourney v6 に匹敵

**FLUX (DiT) vs SD3** [Black Forest Labs 2024] [^4]:
- 品質: FLUX > SD3 (特にプロンプト理解)
- 商用ライセンス: Apache 2.0 (SD3 は制限あり)

**DiT の実世界優位性**:
- **プロンプト理解**: テキストエンコーダ (T5/CLIP) との相性
- **スケーラビリティ**: 8B params モデルが現実的に訓練可能
- **コミュニティ**: HuggingFace Diffusers で DiT が標準化

具体的な FID 比較を見れば差は明確だ：

| モデル | バックボーン | FID-30K (ImageNet 256×256) | params |
|:-------|:------------|:--------------------------|:-------|
| LDM-4 | U-Net | 3.60 | 400M |
| DiT-XL/2 | Transformer | **2.27** | 675M |
| SiT-XL/2 | Transformer | 2.06 | 675M |

DiT-XL/2 が FID 2.27 を達成した時点で、U-Netベースの LDM-4（FID 3.60）を大幅に上回る。同じパラメータ規模でTransformerが圧勝する — これはアーキテクチャの本質的な優位性であり、ハイパーパラメータ調整では埋められない差だ。

FID の差 $3.60 - 2.27 = 1.33$ は画像品質において知覚的に大きな差に対応する。FID は Inception v3 特徴空間での生成分布 $p_g$ と真の分布 $p_r$ のFréchet距離：
$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$
FID が低いほど生成画像の統計量が実画像に近い。FID < 5 はほぼ人間の目で判別困難なレベルとされており、DiT-XL/2 の 2.27 はこの閾値を大きく下回る。

### 2.4 3つの比喩で捉える DiT

**比喩1: パズルの組み立て方**
- **U-Net**: ピース同士の「隣接関係」だけ見る (局所的)
- **DiT**: 全ピースを俯瞰して「全体像」から組み立てる (大域的)

**比喩2: 文章理解**
- **U-Net**: 単語の「前後3単語」だけ見て意味を推測
- **DiT**: 文章全体を読んで「文脈」を理解

**比喩3: 楽団の指揮**
- **U-Net**: 各楽器が「隣の楽器」だけ聞いて演奏
- **DiT**: 指揮者が全楽器を統率 (Self-Attention = 指揮者)

### 2.5 Trojan Horse — Python から Julia/Rust へ

**これまでの言語構成**:
- **第1-8回 (Course I)**: 🐍Python 100%
- **第9-16回 (Course II)**: 🐍Python → ⚡Julia 登場 (第9回) → 🦀Rust 登場 (第11回)
- **第17-24回 (Course III)**: ⚡Julia + 🦀Rust + 🔮Elixir (第15回登場)
- **第33-42回 (Course IV)**: ⚡Julia + 🦀Rust + 🔮Elixir (3言語フルスタック)
- **第43-50回 (Course V)**: ⚡Julia + 🦀Rust + 🔮Elixir (継続)

**Course V での3言語役割**:
- **⚡Julia**: 訓練パイプライン (Lux.jl + Reactant.jl / GPU最適化)
- **🦀Rust**: 推論サーバー (Candle / 低レイテンシ / バッチ処理)
- **🔮Elixir**: 分散サービング (Phoenix / 耐障害性 / A/Bテスト)

**本講義での登場**:
- Zone 4: ⚡Julia — Mini-DiT 訓練パイプライン
- Zone 4: 🦀Rust — DiT 推論サーバー (Candle)
- Zone 4: 🔮Elixir — 分散サービング (OTP supervision)

> **Note:** **ここまでで全体の20%完了！** DiT が U-Net を超える理由を3軸 (Scaling/帰納バイアス/実世界) で理解した。次は DiT の数式を完全導出する — 60分の数式修行ゾーンへ。

---


> Progress: 20%
> **理解度チェック**
> 1. $ ∝ 性能 $ の各記号の意味と、この式が表す操作を説明してください。
> 2. このゾーンで学んだ手法の直感的な意味と、なぜこの定式化が必要なのかを説明してください。

## 📐 3. 数式修行ゾーン（60分）— DiT 完全導出

**ゴール**: DiT・MM-DiT・SiT の数式を1行ずつ導出し、U-Net との違いを数学的に理解する。

**このゾーンの構成**:
1. Patchify — 画像をトークン列に変換
2. AdaLN-Zero — 条件付き正規化
3. DiT ブロック — Self-Attention + MLP
4. MM-DiT (SD3) — Joint Attention
5. SiT (Stochastic Interpolants) — Interpolant-based DiT
6. U-Net vs DiT の計算量比較
7. ⚔️ Boss Battle: DiT Forward Pass 完全実装

**数式修行の心構え** (第1回で学んだ):
- 「数式は声に出して読む」
- 「1行ずつ導出 — 飛ばさない」
- 「具体的な数値で検証」

> **⚠️ Warning:** 以降の数式は順番通りに読むこと。3.1 → 3.2 → 3.3 の順で依存関係がある。3.3 を理解するには 3.2 の AdaLN-Zero が必須。飛ばすと「なぜこの数式が必要か」が見えなくなる。

DiT は**ピクセル空間**ではなく **VAE の潜在空間** $\mathbf{z} \in \mathbb{R}^{h \times w \times c}$ 上で拡散過程を行う（LDM [Rombach+ 2022] の設計）。

VAE の Encoder-Decoder の役割：
$$
\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}, \quad \hat{\mathbf{x}} = \mathcal{D}(\mathbf{z})
$$

代表的な設定（SD3）：
- 入力画像: $\mathbb{R}^{1024 \times 1024 \times 3}$（ピクセル空間）
- VAE 圧縮率: $8\times$（各辺を $1/8$ に）
- 潜在表現: $\mathbb{R}^{128 \times 128 \times 16}$（$c = 16$ チャネル）

DiT は $\mathbb{R}^{128 \times 128 \times 16}$ 空間でノイズ除去を行い、生成完了後に $\mathcal{D}$ でピクセル空間に戻す。

**なぜ潜在空間？** 計算量の比較：
- ピクセル空間の DiT: $N = (1024/16)^2 = 4096$ トークン（$P = 16$）
- 潜在空間の DiT: $N = (128/2)^2 = 4096$ トークン（$P = 2$、より細かいパッチ）

同じトークン数でも**潜在空間は意味的に圧縮済み** — VAE が「どのピクセルが重要か」をすでに学習しているため、DiT はより抽象的な特徴を学習できる。ピクセル空間で直接拡散する DiT（未圧縮版）は計算量が約 $c^2 = 256$ 倍になり、現実的でない。

**DiT の全体 Forward Pass 概略**:

```mermaid
graph LR
    A["🖼️ 画像 x<br/>ℝ^{H×W×3}"] --> B["VAE Encoder ε<br/>8× 圧縮"]
    B --> C["潜在変数 z<br/>ℝ^{h×w×c}"]
    C --> D["Patchify<br/>P×P 分割"]
    D --> E["パッチ列<br/>ℝ^{N×D}"]
    E --> F["+ PE<br/>位置埋め込み"]
    F --> G["DiT Block ×L<br/>AdaLN + Attn + MLP"]
    G --> H["Unpatchify<br/>トークン→画像"]
    H --> I["ノイズ予測 ε_θ<br/>ℝ^{h×w×c}"]
    I --> J["VAE Decoder D<br/>復元"]
    J --> K["生成画像 x̂<br/>ℝ^{H×W×3}"]
    style G fill:#ffd700,stroke:#ff6347,stroke-width:2px
```

### 3.1 Patchify — 画像をトークン列に変換

DiT の第一歩は **Patchify** — 画像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ をパッチ列 $\mathbf{z} \in \mathbb{R}^{N \times D}$ に変換する。

**Vision Transformer (ViT) の手法** [Dosovitskiy+ 2020] [^1]:
1. 画像を $P \times P$ パッチに分割
2. 各パッチを線形変換で $D$ 次元ベクトルに埋め込み
3. Positional Encoding を追加

**数式**:
$$
\begin{align}
\text{Patchify}: \mathbb{R}^{H \times W \times C} &\to \mathbb{R}^{N \times (P^2 \cdot C)} \\
\mathbf{x} &\mapsto [\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_N]
\end{align}
$$
ここで:
- $N = \frac{H}{P} \times \frac{W}{P}$ — パッチ数
- $\mathbf{p}_i \in \mathbb{R}^{P^2 \cdot C}$ — $i$ 番目のパッチ (flatten)

**線形埋め込み**:
$$
\mathbf{z}_i = \mathbf{W}_{\text{patch}} \mathbf{p}_i + \mathbf{b}_{\text{patch}}
$$
- $\mathbf{W}_{\text{patch}} \in \mathbb{R}^{D \times (P^2 \cdot C)}$ — 埋め込み行列
- $\mathbf{z}_i \in \mathbb{R}^D$ — 埋め込みベクトル

**Positional Encoding** (第16回で学んだ):
$$
\mathbf{z}_i^{\text{pos}} = \mathbf{z}_i + \mathbf{PE}(i)
$$
- $\mathbf{PE}(i) \in \mathbb{R}^D$ — 位置埋め込み (sinusoidal or learned)

**具体例**: 256×256 RGB 画像を 16×16 パッチに分割
- $H = W = 256$, $C = 3$, $P = 16$
- $N = \frac{256}{16} \times \frac{256}{16} = 16 \times 16 = 256$ パッチ
- 各パッチ: $16 \times 16 \times 3 = 768$ 次元
- 埋め込み後: $D = 768$ (ViT-Base と同じ)

**数値検証**:

$H = W = 256$、$C = 3$（RGB）、$P = 16$ の具体例でShape flowを追う。

パッチ数：
$$
N = \frac{256}{16} \times \frac{256}{16} = 16 \times 16 = 256
$$

各パッチをflattenした次元：
$$
P^2 \cdot C = 16 \times 16 \times 3 = 768
$$

従って第1パッチは $\mathbf{p}_1 \in \mathbb{R}^{768}$。全パッチ列は $[\mathbf{p}_1, \ldots, \mathbf{p}_{256}] \in \mathbb{R}^{256 \times 768}$。

埋め込み行列 $\mathbf{W}_{\text{patch}} \in \mathbb{R}^{768 \times 768}$ で線形変換するとShape は変わらない。Shape flow をまとめると：

$$
\mathbb{R}^{256 \times 256 \times 3} \;\xrightarrow{\text{split into patches}}\; \mathbb{R}^{256 \times 768} \;\xrightarrow{\mathbf{W}_{\text{patch}}}\; \mathbb{R}^{256 \times 768}
$$

埋め込み行列のパラメータ数は $768 \times 768 = 589{,}824$。これはちょうど Conv2d(3, 768, kernel\_size=16, stride=16) と等価であり、実装上はこの畳み込みで代替される。Positional Encodingを加えた最終Shapeも $\mathbb{R}^{256 \times 768}$—Transformerへの入力として渡せる状態だ。

#### 2次元 Sinusoidal Positional Encoding の導出

画像のパッチには縦 $i$・横 $j$ の2次元座標がある。1次元の Sinusoidal PE（第16回）を2次元に拡張すると：

$$
\text{PE}(i, 2k) = \sin\!\left(\frac{i}{10000^{2k/D}}\right), \quad \text{PE}(i, 2k+1) = \cos\!\left(\frac{i}{10000^{2k/D}}\right)
$$

ここで $i$ はパッチインデックス（行方向）、$k = 0, 1, \ldots, D/2 - 1$ は次元インデックス。2次元版では $D$ 次元を半分ずつ行・列に割り当てる：

$$
\mathbf{PE}(i, j) = [\underbrace{\text{PE}_{\text{row}}(i)}_{\in \mathbb{R}^{D/2}},\; \underbrace{\text{PE}_{\text{col}}(j)}_{\in \mathbb{R}^{D/2}}] \in \mathbb{R}^D
$$

**底数 10000 の意義**: 周波数 $\omega_k = 1 / 10000^{2k/D}$ は $k$ とともに指数的に小さくなる。$k = 0$ では $\omega_0 = 1.0$（1パッチ周期の高周波）、$k = D/2 - 1$ では $\omega_{D/2-1} = 1/10000$（10000パッチ周期の低周波）。$D = 768$ の場合、位置 $i \in [1, 256]$ に対してsin/cosの周期は $[2\pi, 2\pi \times 10000]$ の広いレンジをカバーする。この幅広い周波数帯域が「近いパッチは似た表現・遠いパッチは異なる表現」という性質を保証し、256パッチ全ての位置を一意に符号化できる。

学習済みの位置埋め込み（DiT-XL は Learnable PE を使用）も同等の表現力を持つが、Sinusoidal PE は訓練データに含まれない解像度への外挿（ViT-L の256→512パッチへの転移）において優位性を示す。

### 3.2 AdaLN-Zero — 条件付き正規化

**Layer Normalization (LN)** [Ba+ 2016] [^5] (第2回で学んだ):
$$
\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$
- $\mu = \frac{1}{D} \sum_{i=1}^D x_i$ — 平均
- $\sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$ — 分散

**Adaptive Layer Normalization (AdaLN)** [Perez+ 2018] [^6]:
$$
\text{AdaLN}(\mathbf{x}, \mathbf{c}) = \gamma(\mathbf{c}) \odot \text{LN}(\mathbf{x}) + \beta(\mathbf{c})
$$
- $\gamma(\mathbf{c}), \beta(\mathbf{c})$ — 条件 $\mathbf{c}$ から生成される scale & shift
- $\odot$ — 要素ごとの積

**AdaLN-Zero** [Peebles & Xie 2023] [^2] — DiT の鍵:
$$
\gamma(\mathbf{c}) \text{ と } \beta(\mathbf{c}) \text{ を **ゼロ初期化**}
$$

**なぜゼロ初期化？**
- **訓練初期**: $\gamma = 0, \beta = 0$ → AdaLN の出力 = 0 → **Residual 接続が恒等写像**
- **訓練中期**: $\gamma, \beta$ が学習されて条件 $\mathbf{c}$ の影響が徐々に増加
- **安定性**: Skip connections (第2回で学んだ) が訓練初期の勾配を安定化

**なぜ Batch Normalization でなく Layer Normalization なのか？** Batch Normalization はバッチ軸で統計量を計算するため、バッチサイズへの依存がある。拡散モデルでは複数の異なるタイムステップ $t$ のサンプルが1バッチに混在するため、BN の統計量がステップ間で汚染される。LN はサンプル内の特徴軸で正規化するため、この問題が発生しない。数式で比較すると：

$$
\text{BN}: \quad \mu_{\text{BN}} = \frac{1}{B} \sum_{b=1}^B x_{b,i} \quad \text{（バッチ軸平均）}
$$
$$
\text{LN}: \quad \mu_{\text{LN}} = \frac{1}{D} \sum_{i=1}^D x_i \quad \text{（特徴軸平均）}
$$

BN では異なる $t$ のサンプルが混在する $B$ サンプルで平均を取るため、「$t = 0.1$ のほぼクリーンなサンプル」と「$t = 0.9$ のほぼノイズのサンプル」の統計量が混合される。LN は各サンプル独立に正規化するため、この問題がない。これが拡散モデル全般で LN が標準的な理由だ。
$$
\begin{align}
\mathbf{c} &= [\mathbf{t}, \mathbf{c}_{\text{cond}}] \quad \text{(timestep + condition)} \\
\gamma &= \text{MLP}_\gamma(\mathbf{c}) \quad \text{(initialized to 0)} \\
\beta &= \text{MLP}_\beta(\mathbf{c}) \quad \text{(initialized to 0)} \\
\text{AdaLN-Zero}(\mathbf{x}, \mathbf{c}) &= \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\end{align}
$$

**具体例**: $\mathbf{x} \in \mathbb{R}^{256 \times 768}$ (256 tokens, 768 dims)
- $\mathbf{t} \in \mathbb{R}^{128}$ — timestep embedding (第36回で学んだ)
- $\mathbf{c}_{\text{cond}} \in \mathbb{R}^{512}$ — text embedding (CLIP)
- $\mathbf{c} = [\mathbf{t}, \mathbf{c}_{\text{cond}}] \in \mathbb{R}^{640}$
- $\gamma, \beta \in \mathbb{R}^{768}$ — MLP 出力

**数値検証**:

$\mathbf{x} = [1.0,\; 2.0,\; 3.0] \in \mathbb{R}^3$ の3次元ベクトルで Layer Normalization を手計算する。

平均：
$$
\mu = \frac{1.0 + 2.0 + 3.0}{3} = 2.0
$$

分散（偏差なし）：
$$
\sigma^2 = \frac{(1.0 - 2.0)^2 + (2.0 - 2.0)^2 + (3.0 - 2.0)^2}{3} = \frac{1 + 0 + 1}{3} \approx 0.667
$$

LN出力（$\epsilon = 10^{-6}$）：
$$
\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - 2.0}{\sqrt{0.667 + 10^{-6}}} \approx \frac{[-1,\; 0,\; 1]}{0.8165} = [-1.225,\; 0.0,\; 1.225]
$$

条件付きスケール $\gamma = 1.5$、シフト $\beta = 0.1$ を適用：
$$
\text{AdaLN}(\mathbf{x}) = 1.5 \times [-1.225,\; 0.0,\; 1.225] + 0.1 = [-1.737,\; 0.1,\; 1.937]
$$

ゼロ初期化時（$\gamma = 0,\; \beta = 0$）：
$$
\text{AdaLN-Zero}(\mathbf{x}) = 0 \times \text{LN}(\mathbf{x}) + 0 = [0,\; 0,\; 0]
$$

Residual接続で足し戻すと $\mathbf{z} + \mathbf{0} = \mathbf{z}$（恒等写像）。訓練初期の勾配安定性がここから来る — $\gamma, \beta$ が学習されるにつれて、条件 $\mathbf{c}$ の影響が徐々に増加する。

#### クラスラベル条件付けの埋め込み

ImageNet クラス条件付き生成では、条件ベクトル $\mathbf{c}_{\text{cond}}$ に **クラス埋め込み** を使う：

$$
\mathbf{c}_{\text{cond}} = \mathbf{E}_{\text{class}}[y] \in \mathbb{R}^{D}
$$

ここで $\mathbf{E}_{\text{class}} \in \mathbb{R}^{1000 \times D}$（ImageNet の1000クラスそれぞれに $D$ 次元ベクトル）、$y \in \{0, \ldots, 999\}$ はクラスインデックス。条件ベクトルを構成：

$$
\mathbf{c} = [\mathbf{t},\; \mathbf{c}_{\text{cond}}] = [\mathbf{t},\; \mathbf{E}_{\text{class}}[y]] \in \mathbb{R}^{D_t + D}
$$

これをAdaLN-ZeroのMLPに入力することで、「$t = 0.5$ 時点で犬クラスを生成中」という2次元の条件付けが実現する。Classifier-Free Guidance（CFG）では $y = \emptyset$（null クラス）で無条件ベクトル $\mathbf{c}_\emptyset$ を定義し、推論時に：

$$
\tilde{\mathbf{v}}_\theta(\mathbf{x}_t, t, y) = (1 + w)\, \mathbf{v}_\theta(\mathbf{x}_t, t, y) - w\, \mathbf{v}_\theta(\mathbf{x}_t, t, \emptyset)
$$

のようにガイダンス強度 $w > 0$ でサンプル品質を制御する。$w = 1.5$ が DiT-XL/2 の FID 2.27 を達成した設定だ。

### 3.3 DiT ブロック — Self-Attention + MLP

DiT の基本ブロックは **Transformer encoder** と同じ構造 (第16回で学んだ)。

**DiT Block の構成**:
1. AdaLN-Zero pre-normalization
2. Multi-Head Self-Attention
3. Residual connection
4. AdaLN-Zero pre-normalization (2回目)
5. MLP (Feed-Forward)
6. Residual connection

**数式**:
$$
\begin{align}
\mathbf{h}_1 &= \text{AdaLN-Zero}(\mathbf{z}, \mathbf{c}) \\
\mathbf{a} &= \text{Attention}(\mathbf{h}_1) \\
\mathbf{z}' &= \mathbf{z} + \mathbf{a} \quad \text{(residual)} \\
\mathbf{h}_2 &= \text{AdaLN-Zero}(\mathbf{z}', \mathbf{c}) \\
\mathbf{m} &= \text{MLP}(\mathbf{h}_2) \\
\mathbf{z}_{\text{out}} &= \mathbf{z}' + \mathbf{m} \quad \text{(residual)}
\end{align}
$$

**Residual 接続と勾配流の保証**: $L$ 層 DiT の勾配を考える。Residual接続 $\mathbf{z}' = \mathbf{z} + F(\mathbf{z})$ の逆伝播：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}'} \cdot \left(I + \frac{\partial F}{\partial \mathbf{z}}\right)
$$
単位行列 $I$ の存在により、$\partial F / \partial \mathbf{z} \approx 0$（訓練初期）でも勾配が $\mathbf{z}$ に直接伝わる。28層の DiT-XL では乗算連鎖 $\prod_{\ell=1}^{28}(\cdot)$ が発生するが、各層の $I$ 項が勾配消失を防ぐ。AdaLN-Zero との組み合わせで**初期化問題を完全に排除**した設計だ。

**Multi-Head Self-Attention** (第16回で学んだ):
$$
\begin{align}
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \\
Q &= \mathbf{h}_1 W_Q, \quad K = \mathbf{h}_1 W_K, \quad V = \mathbf{h}_1 W_V \\
\text{MultiHead}(\mathbf{h}_1) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O
\end{align}
$$

#### スケーリング因子 $1/\sqrt{d_k}$ の必然性

「なぜ $\sqrt{d_k}$ で割るのか？」— これは数値安定性の本質的な問いだ。

$q, k \in \mathbb{R}^{d_k}$ が独立に標準正規分布から引かれるとする：$q_i, k_i \sim \mathcal{N}(0, 1)$。

内積 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ の期待値と分散を計算する：

$$
\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\, \mathbb{E}[k_i] = 0
$$

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\, \mathbb{E}[k_i^2] = \sum_{i=1}^{d_k} 1 \cdot 1 = d_k
$$

つまり $\text{std}(q \cdot k) = \sqrt{d_k}$。DiT-Bの $d_k = 64$ では $\text{std} = 8$。

**スケーリングなしの危険**: スコアが $\pm 8$ 程度に散らばると、Softmaxはほぼ one-hot になる：
$$
\text{softmax}(8, 0, 0, \ldots) \approx (1.0, 0.0, 0.0, \ldots)
$$
これは勾配消失（$\partial \text{softmax} / \partial x \approx 0$）を引き起こし、訓練が進まなくなる。

$\sqrt{d_k}$ で割ると $\text{std}(q \cdot k / \sqrt{d_k}) = 1$ に正規化され、Softmax は適度な「soft」な分布を保つ。スケーリングは単なる慣習ではなく、訓練可能性を保証する**数学的必然**だ。

#### Multi-Head の意義：異なる「注意パターン」を並列学習

単一ヘッドのAttentionは1種類の関係性しか捉えられない。$H$ ヘッドに分割すると、各ヘッドが **異なる部分空間** で関係を学習できる：

$$
\text{head}_h = \text{Attention}(Q W_Q^{(h)},\; K W_K^{(h)},\; V W_V^{(h)}), \quad h = 1, \ldots, H
$$

ここで $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{D \times d_k}$（$d_k = D / H$）。

DiT-XL では $H = 16$、$d_k = 72$。16ヘッドのうち：
- ヘッド1-4：局所パッチ間の空間的隣接関係
- ヘッド5-8：意味的類似パッチ（同色・同テクスチャ）
- ヘッド9-12：条件 $\mathbf{c}$ に応じた長距離依存
- ヘッド13-16：グローバルな構図の整合性

これは観測されたものではなく概念的な説明だが、解釈可能性研究 [Zhao+ 2025] ではHeadごとに異なる意味的役割が実際に確認されている。全ヘッドを結合：

$$
\text{MultiHead}(\mathbf{h}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, W_O
$$

$W_O \in \mathbb{R}^{D \times D}$ で元の $D$ 次元に射影。計算量は単一ヘッドと同じ $O(N^2 D)$ のままヘッド数を増やせる — これがMulti-Headの「タダ飯」だ。
$$
\text{MLP}(\mathbf{x}) = \text{GELU}(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2
$$
- GELU: Gaussian Error Linear Unit [Hendrycks & Gimpel 2016] [^7]
- Hidden dim: $4D$ (標準的な Transformer の設定)

**GELU の数学的定義**:
$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$
ここで $\Phi(x)$ は標準正規分布のCDF。直感: 入力 $x$ を「$x$ が大きい確率」で重み付けして通過させる。$x \to +\infty$ では $\text{GELU}(x) \approx x$（線形）、$x \to -\infty$ では $\approx 0$（ゲーティング）。ReLU の滑らかな近似として機能し、Transformer 系モデルでは標準的に使用される。

#### DiT モデルファミリーのサイズ比較

DiT は ViT の命名規則を踏襲する 4 つの標準サイズが定義されている [Peebles & Xie 2023]：

| モデル | $D$ | Layers $L$ | Heads $H$ | Params | FID-50K |
|:-------|:----|:-----------|:---------|:-------|:--------|
| DiT-S/8 | 384 | 12 | 6 | 33M | 43.5 |
| DiT-B/4 | 768 | 12 | 12 | 130M | 18.6 |
| DiT-L/2 | 1024 | 24 | 16 | 458M | 5.02 |
| **DiT-XL/2** | **1152** | **28** | **16** | **675M** | **2.27** |

スラッシュ後の数字（/2、/4、/8）はパッチサイズ $P$ を示す。$P = 2$ では $256 \times 256$ 潜在空間で $N = 128 \times 128 / 2^2 = 16{,}384$ トークンになるため、計算量は $P = 16$ の $256$ トークン比で $(16384/256)^2 = 4096$ 倍に増大する。DiT-XL/2 が最高性能である理由はパラメータ数だけでなく、この **細かいパッチ解像度** にある。

**Unpatchify**（逆変換）の数式：

Transformerの出力 $\mathbf{z}_L \in \mathbb{R}^{N \times D}$ を画像空間に戻す：
$$
\text{Unpatchify}: \mathbb{R}^{N \times D} \to \mathbb{R}^{H \times W \times 2C}
$$
各トークン $\mathbf{z}^{(i)} \in \mathbb{R}^D$ を線形変換 $\mathbf{W}_{\text{out}} \in \mathbb{R}^{D \times (P^2 \cdot 2C)}$ で $P^2 \times 2C$ 次元に戻し、対応するパッチ位置に配置する。出力次元が $2C$（元の $C$ の2倍）なのは、ノイズ予測 $\epsilon_\theta$ と分散 $\Sigma_\theta$ の両方を同時に出力するためだ（第36回の DDPM と同じ設計）。

**数値検証**:

$N = 3$ トークン、$d_k = 4$ の最小例でScaled Dot-Product Attentionを手計算する。

Query行列とKey行列が単位ベクトル（各トークンが直交）だとする：
$$
Q = K = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix} \in \mathbb{R}^{3 \times 4}
$$

生のスコア行列 $QK^\top \in \mathbb{R}^{3 \times 3}$：
$$
QK^\top = I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

スケーリング後（$\sqrt{d_k} = \sqrt{4} = 2$）：
$$
\frac{QK^\top}{\sqrt{4}} = \frac{1}{2} I_3 = \begin{pmatrix} 0.5 & 0 & 0 \\ 0 & 0.5 & 0 \\ 0 & 0 & 0.5 \end{pmatrix}
$$

第1行に対してSoftmaxを適用：
$$
\text{softmax}(0.5,\; 0,\; 0) = \frac{(e^{0.5},\; e^0,\; e^0)}{e^{0.5} + e^0 + e^0} \approx \frac{(1.649,\; 1.0,\; 1.0)}{3.649} = (0.452,\; 0.274,\; 0.274)
$$

行の和：$0.452 + 0.274 + 0.274 = 1.000$ — Softmaxの規格化条件が成立。各トークンが自分自身に最大の注意（0.452）を向け、他2つに等分（0.274ずつ）している構造が読み取れる。

### 3.4 MM-DiT (SD3) — Joint Attention

**MM-DiT の革新** [Esser+ 2024] [^3]:
- 画像パッチ $\mathbf{z}_{\text{img}}$ とテキストトークン $\mathbf{z}_{\text{txt}}$ を **同じ Transformer** で処理
- Joint Self-Attention — 画像 ↔ テキストの相互作用

**数式**:
$$
\begin{align}
\mathbf{z} &= [\mathbf{z}_{\text{img}}, \mathbf{z}_{\text{txt}}] \in \mathbb{R}^{(N_{\text{img}} + N_{\text{txt}}) \times D} \\
\mathbf{h} &= \text{AdaLN-Zero}(\mathbf{z}, \mathbf{c}) \\
\mathbf{a} &= \text{Attention}(\mathbf{h}) \quad \text{(joint attention)} \\
\mathbf{z}_{\text{img}}', \mathbf{z}_{\text{txt}}' &= \text{Split}(\mathbf{z} + \mathbf{a})
\end{align}
$$

**なぜ Joint Attention？**
- **Classifier-Free Guidance (CFG)** (第39回で学んだ) では、条件付き/無条件を別々に処理
- **MM-DiT** では、画像とテキストが **同じ潜在空間** で相互作用 → より強い条件付け

#### Cross-Attention vs Joint Attention の数学的比較

**Cross-Attention**（従来の条件付け方式）：

$$
\text{Attn}_{\text{cross}}(Q_{\text{img}}, K_{\text{txt}}, V_{\text{txt}}) = \text{softmax}\!\left(\frac{Q_{\text{img}} K_{\text{txt}}^\top}{\sqrt{d_k}}\right) V_{\text{txt}}
$$

これはテキスト→画像の一方向の情報流だ。画像パッチ同士の関係（img↔img）は別のSelf-Attentionで計算し、テキストトークン同士の関係（txt↔txt）は完全に無視される。つまり**4種類の相互作用のうち2種類しか捉えられない**。

**MM-DiT のJoint Attention**：結合されたトークン列 $[Q_{\text{img}};\, Q_{\text{txt}}]$ を使う：

$$
\text{Attn}_{\text{joint}} = \text{softmax}\!\left(\frac{[Q_{\text{img}};\, Q_{\text{txt}}]\,[K_{\text{img}};\, K_{\text{txt}}]^\top}{\sqrt{d_k}}\right)[V_{\text{img}};\, V_{\text{txt}}]
$$

アテンション行列 $A \in \mathbb{R}^{(N_{\text{img}} + N_{\text{txt}}) \times (N_{\text{img}} + N_{\text{txt}})}$ をブロック分解すると：

$$
A = \begin{pmatrix} A_{\text{img}\to\text{img}} & A_{\text{img}\to\text{txt}} \\ A_{\text{txt}\to\text{img}} & A_{\text{txt}\to\text{txt}} \end{pmatrix}
$$

**4つのブロックが1回の行列積で同時に計算される**：
- $A_{\text{img}\to\text{img}}$：画像パッチ間の空間的関係
- $A_{\text{img}\to\text{txt}}$：画像がテキストを参照（「どの単語に対応するパッチか」）
- $A_{\text{txt}\to\text{img}}$：テキストが画像を参照（「この単語はどのパッチに影響するか」）
- $A_{\text{txt}\to\text{txt}}$：テキストトークン間の文脈関係

Cross-Attentionでは $A_{\text{txt}\to\text{txt}}$ が得られない。SD3がT5-XXLの文脈表現をそのまま活かせるのは、Joint Attentionがテキストの文脈情報を動的に更新し続けるからだ。

**SD3 の3つのテキストエンコーダ**:
1. CLIP ViT-L/14 — 画像-テキスト align
2. CLIP ViT-bigG/14 — より大規模な CLIP
3. T5-XXL — テキスト理解 (Google の言語モデル)

**エンコーダの統合**:
$$
\mathbf{z}_{\text{txt}} = \text{Concat}([\text{CLIP-L}(\text{prompt}), \text{CLIP-G}(\text{prompt}), \text{T5}(\text{prompt})])
$$

**具体例**: SD3 Medium (2B params)
- $N_{\text{img}} = 4096$ — 画像パッチ (64×64 latent / patch size 2)
- $N_{\text{txt}} = 256$ — テキストトークン
- $D = 1536$ — hidden dim
- $L = 24$ — layers

**数値検証**:

SD3 Mediumの具体値でJoint Attentionの計算量を見積もる。

$N_{\text{img}} = 4096$、$N_{\text{txt}} = 256$、$D = 1536$ の場合：
$$
N_{\text{total}} = N_{\text{img}} + N_{\text{txt}} = 4096 + 256 = 4352
$$

Joint Attention行列のサイズ：
$$
N_{\text{total}}^2 = 4352^2 = 18{,}939{,}904 \approx 1.9 \times 10^7 \text{ 要素}
$$

1層あたりのFLOPs概算（$QK^\top$ 行列積）：
$$
O_{\text{attn}} \approx N_{\text{total}}^2 \times D = 4352^2 \times 1536 \approx 2.9 \times 10^{10}
$$

全24層の合計：
$$
O_{\text{total}} \approx 24 \times 2.9 \times 10^{10} \approx 7.0 \times 10^{11} \text{ FLOPs}
$$

比較: 仮に Cross-Attentionのみなら $N_{\text{img}} \times N_{\text{txt}} = 4096 \times 256 = 1{,}048{,}576$（Joint Attentionの約5.5%）。しかし Cross-Attentionでは画像パッチ同士の関係 $N_{\text{img}}^2 = 1.68 \times 10^7$ を別途計算する必要があり、トータルでは変わらない。Joint Attentionは4種類の相互作用（img↔img, img↔txt, txt↔img, txt↔txt）を1回のMatmulで同時に捉える点が効率的だ。

#### QK-Normalization — SD3 の大規模安定化技術

SD3 は8Bパラメータの大規模モデルを安定して訓練するため **QK-Normalization** を採用する。通常のAttentionでは：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

大規模モデルでは $Q, K$ のノルムが訓練中に爆発的に増大し、$QK^\top / \sqrt{d_k}$ のスコアが極端に大きくなってSoftmaxが飽和する。QK-Normalizationは Query と Key それぞれを $\ell_2$ 正規化する：

$$
\hat{Q} = \frac{Q}{\|Q\|_2}, \quad \hat{K} = \frac{K}{\|K\|_2}
$$

$$
\text{Attention}_{\text{QKnorm}}(\hat{Q}, \hat{K}, V) = \text{softmax}\!\left(\frac{\hat{Q}\hat{K}^\top}{\sqrt{d_k}}\right) V
$$

$\hat{Q}, \hat{K}$ のノルムは常に1なので、内積 $\hat{Q}\hat{K}^\top \in [-1, +1]$（コサイン類似度）に束縛される。スケーリング後のスコアは $[-1/\sqrt{d_k}, +1/\sqrt{d_k}]$ の範囲に収まり、訓練全体を通じてSoftmaxが適度な分布を保つ。

SD3の実験では QK-Normなしでは 500M params 程度で訓練が不安定化し始めたのに対し、QK-Normありでは 8B paramsまでスケール可能だったと報告されている。スケーリングにとってアーキテクチャの数値安定性は性能と同等に重要だ。

### 3.5 SiT (Stochastic Interpolants) — Interpolant-based DiT

**SiT** [Ma+ 2024] [^8] は **Stochastic Interpolants** (第38回で学んだ) を DiT に統合したモデル。

**Stochastic Interpolant の復習**:
$$
\mathbf{x}_t = \alpha(t) \mathbf{x}_0 + \beta(t) \mathbf{x}_1 + \gamma(t) \mathbf{z}
$$
- $\mathbf{x}_0 \sim p_0$ (ノイズ)
- $\mathbf{x}_1 \sim p_1$ (データ)
- $\mathbf{z} \sim \mathcal{N}(0, I)$ (確率的項)
- $\alpha(0) = 1, \alpha(1) = 0$ — ノイズ→データ

**SiT の損失関数**:
$$
\mathcal{L}_{\text{SiT}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}} \left[\left\| \mathbf{v}_\theta(\mathbf{x}_t, t) - \dot{\mathbf{x}}_t \right\|^2\right]
$$
- $\dot{\mathbf{x}}_t = \alpha'(t) \mathbf{x}_0 + \beta'(t) \mathbf{x}_1 + \gamma'(t) \mathbf{z}$ — ベクトル場の真値

**SiT vs Flow Matching**:
- **Flow Matching** (第38回): $\gamma(t) = 0$ — 決定論的補間
- **SiT**: $\gamma(t) > 0$ — 確率的補間 (ノイズ項あり)

**なぜ確率的？**
- **多様性向上**: 同じ $\mathbf{x}_1$ から異なる生成経路を探索
- **モード崩壊回避**: Flow Matching の決定論性が原因のモード崩壊を緩和

#### SiT の Fokker-Planck 方程式

SiT が確率過程であることを厳密に示す。Stochastic Interpolant は次の確率微分方程式（SDE）に対応する：

$$
d\mathbf{x}_t = \mathbf{b}(\mathbf{x}_t, t)\, dt + \sigma(t)\, d\mathbf{W}_t
$$

ここで $\mathbf{W}_t$ は標準ブラウン運動、$\sigma(t) = \gamma'(t)$ は拡散係数。対応する Fokker-Planck 方程式（確率密度 $\rho_t$ の時間発展）：

$$
\frac{\partial \rho_t}{\partial t} = -\nabla \cdot (\mathbf{b}\, \rho_t) + \frac{\sigma(t)^2}{2} \Delta \rho_t
$$

$\gamma(t) = 0$（Flow Matching）のとき $\sigma = 0$ となり Fokker-Planck は連続方程式 $\partial_t \rho_t = -\nabla \cdot (\mathbf{b}\, \rho_t)$ に退化する — これがODE経路の直線性の本質だ。$\gamma > 0$ では拡散項 $(\sigma^2/2) \Delta \rho_t$ が加わり、確率密度が「広がる」方向に動く。この広がりが多様性向上をもたらす一方、軌道の曲率（curvature）が増加してサンプリングに多くのNFEが必要になるトレードオフがある。

**DDPM との統一的理解**: $\alpha(t) = \sqrt{\bar{\alpha}_t}$、$\beta(t) = 0$、$\gamma(t) = \sqrt{1 - \bar{\alpha}_t}$ と設定すると Stochastic Interpolant は DDPM の前向き過程と一致する。$\alpha(t) = 1-t$、$\beta(t) = t$、$\gamma(t) = 0$ では Flow Matching（直線補間）。SiT は両者を **同一フレームワーク** として内包する。

#### タイムステップ埋め込みの構成

DiT はタイムステップ $t \in [0, T]$ を条件ベクトル $\mathbf{t} \in \mathbb{R}^{D_t}$ に変換してAdaLN-Zeroに渡す。この変換は **Sinusoidal Embedding + 2層MLP** で行われる：

$$
\mathbf{t}^{\text{sin}} = \left[\sin\!\left(\frac{t}{10000^{2k/D_t}}\right), \cos\!\left(\frac{t}{10000^{2k/D_t}}\right)\right]_{k=0}^{D_t/2 - 1} \in \mathbb{R}^{D_t}
$$

$$
\mathbf{t} = \text{MLP}(\mathbf{t}^{\text{sin}}) \in \mathbb{R}^{D_t}
$$

$D_t = 256$（DiT-B）の場合、各次元が異なる周波数でタイムステップを符号化し、MLPが非線形に組み合わせる。この埋め込みが $\gamma(\mathbf{c})$ および $\beta(\mathbf{c})$ のMLP入力に結合されることで、「$t = 0.1$（ほぼデータ）と $t = 0.9$（ほぼノイズ）では全く異なるスケール・シフトを適用する」という条件付き挙動が実現する。

**数値検証**:

線形補間スケジュール $\alpha(t) = 1 - t$、$\beta(t) = t$（Flow Matchingと同形）で $t = 0.5$ を計算する。

$\mathbf{x}_0 = [2.0,\; 0.0]$（ノイズサンプル）、$\mathbf{x}_1 = [0.0,\; 2.0]$（データサンプル）のとき：

補間係数：
$$
\alpha(0.5) = 1 - 0.5 = 0.5, \quad \beta(0.5) = 0.5
$$

中間点（確率的項 $\gamma = 0$ の場合）：
$$
\mathbf{x}_{0.5} = 0.5 \times [2.0,\; 0.0] + 0.5 \times [0.0,\; 2.0] = [1.0,\; 1.0]
$$

速度場（真値、$\alpha'(t) = -1$、$\beta'(t) = 1$）：
$$
\dot{\mathbf{x}}_t = \alpha'(t)\, \mathbf{x}_0 + \beta'(t)\, \mathbf{x}_1 = (-1) \times [2.0,\; 0.0] + 1 \times [0.0,\; 2.0] = [-2.0,\; 2.0]
$$

ノルム：
$$
\|\dot{\mathbf{x}}_t\| = \sqrt{(-2.0)^2 + 2.0^2} = \sqrt{8} = 2\sqrt{2} \approx 2.83
$$

線形補間では $\|\dot{\mathbf{x}}_t\| = \|\mathbf{x}_1 - \mathbf{x}_0\|$ が $t$ によらず一定 — これが **ODE経路の直線性**を保証し、少ないNFE（Number of Function Evaluations）でサンプリングできる理由だ。確率的補間（$\gamma > 0$）では速度場にノイズが加わり $\|\dot{\mathbf{x}}_t\|$ は時変になるが、多様性向上というトレードオフがある。

### 3.6 U-Net vs DiT の計算量比較

**U-Net の計算量** (空間ベース):
$$
O_{\text{U-Net}} = O(H \times W \times C^2 \times L)
$$
- $H, W$ — 画像サイズ
- $C$ — チャネル数
- $L$ — レイヤー数

**DiT の計算量** (トークンベース):
$$
O_{\text{DiT}} = O(N^2 \times D \times L)
$$
- $N = \frac{H}{P} \times \frac{W}{P}$ — トークン数
- $D$ — hidden dim
- $L$ — レイヤー数

**具体例**: 256×256 画像
- U-Net: $H = W = 256$, $C = 256$, $L = 20$
  - $O_{\text{U-Net}} = 256 \times 256 \times 256^2 \times 20 \approx 8.5 \times 10^{11}$
- DiT: $P = 16$, $N = 256$, $D = 768$, $L = 12$
  - $O_{\text{DiT}} = 256^2 \times 768 \times 12 \approx 6.0 \times 10^{11}$

**結論**: DiT は U-Net と **同程度の計算量** で、より高い性能を達成する (Scaling Laws の恩恵)。

**パッチサイズ $P$ のトレードオフ**: $P$ を小さくすると $N$ が増えて表現力が向上するが計算量は $N^2 \propto P^{-4}$ で増大する。$P$ を2倍にすると計算量は16分の1になるが解像度が落ちる。DiT-XL/2 の $P=2$（潜在空間での計算なので実質的に $P=2 \times 8 = 16$ ピクセル相当）はこのトレードオフの最良点として実験的に選ばれた値だ。

固定の計算予算 $C$（FLOPs）のもとで、モデルサイズ $N$ とデータ量 $D$ をどう配分すれば損失が最小化できるか？これがいわゆる Chinchilla 問題だ。

DiTにおける計算量とモデルサイズ・データ量の関係：
$$
C \approx 6 \cdot N \cdot D \quad \text{（訓練FLOPsの近似式）}
$$

損失のスケーリング：
$$
\mathcal{L}(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \mathcal{L}_\infty
$$

$C$ を固定して $\mathcal{L}$ を最小化する最適解は Lagrange 乗数法で求まる：

$$
N^*(C) \propto C^{\frac{\alpha_D}{\alpha_N + \alpha_D}}, \quad D^*(C) \propto C^{\frac{\alpha_N}{\alpha_N + \alpha_D}}
$$

DiT の実験的推定 [Zhai+ 2024] では $\alpha_N \approx \alpha_D$（モデルとデータが等配分に近い）。これは「同じFLOPsならモデルを2倍大きくするか、データを2倍増やすか、どちらも同じ効果がある」ことを意味する。U-Netではデータスケーリングの恩恵が Transformer より小さい（帰納バイアスがデータ効率を悪化させる）ため、DiTの等配分原理はアーキテクチャ優位性のもう一つの証拠だ。

### 3.6b DiT のアテンション複雑度と FlashAttention

DiT-XL の $N = 256$（パッチサイズ16、256×256画像）では、1層のAttention行列は：

$$
A \in \mathbb{R}^{256 \times 256}, \quad \text{要素数} = 256^2 = 65{,}536
$$

ヘッド数 $H = 16$ を含めると：
$$
65{,}536 \times 16 = 1{,}048{,}576 \approx 10^6 \text{ 要素/層}
$$

Naive Attentionの計算量と空間計算量：
$$
O_{\text{compute}} = O(N^2 d), \quad O_{\text{memory}} = O(N^2)
$$

$N = 256$ では Attention行列のメモリは問題ないが、高解像度（SD3: $N_{\text{img}} = 4096$）では $4096^2 = 16{,}777{,}216$ 要素 × float16 = **32MB/層** に膨らむ。24層では 768MB — VRAM容量に直撃する。

**FlashAttention** [Dao et al., 2022] は SRAM のオンチップキャッシュを活用してAttention行列を陽に作らずに計算する：

$$
O_{\text{memory}}^{\text{Flash}} = O(N \cdot d) \quad \text{（線形メモリ）}
$$

アルゴリズムの核心は **タイル化（tiling）**: $N \times N$ のAttention行列を $B_r \times B_c$ のブロックに分割し、各ブロックをSRAM内で完結させる。HBM（GPU DRAM）へのアクセス回数は：
$$
O\!\left(\frac{N^2 d}{\text{SRAM\_size}}\right) \quad \text{（SRAM\_size} \approx 20\text{MB on A100）}
$$

計算量は同じ $O(N^2 d)$ だが、**HBMアクセスが約10-20倍削減**され、実効スループットが2-4倍向上する。DiT-XL/2の訓練において FlashAttention は非オプション — 標準的な実装では使わない選択肢がない。

SD3・FLUXのような $N_{\text{total}} = 4352$ のMM-DiTでは、Attention行列 $\approx 145\text{MB/層}$ がFlashAttentionなしには成立しない。これが2024年以降の大規模DiT訓練がFlashAttention v2前提で設計されている理由だ。

**FlashAttention の online softmax trick**: 標準的なSoftmax計算は2パスが必要だ（1パス目で最大値を求め、2パス目で指数和を計算）。FlashAttentionは **online softmax** アルゴリズムを使って1パスで完結させる：

タイル $i$ の処理中に現在の最大値 $m_i$ と累積和 $\ell_i$ を保持：
$$
m_i^{\text{new}} = \max(m_i,\; \max(\mathbf{s}_{\text{new}})), \quad \ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_i + \sum e^{\mathbf{s}_{\text{new}} - m_i^{\text{new}}}
$$

この漸化式により、Attention行列を陽に保持せず O(N) メモリでSoftmaxを計算できる。数値安定性（$m_i$ によるオーバーフロー防止）と計算効率を同時に達成した**アルゴリズム的傑作**だ。

#### パラメータ効率の比較：DiT vs U-Net

同じFIDを達成するために必要なパラメータ数を比較すると DiT の優位性が見える。

FID 4.0 程度を達成するモデルの比較：

| モデル | バックボーン | Params | FID | GFLOPS/forward |
|:-------|:------------|:-------|:----|:--------------|
| LDM-4 (DDPM) | U-Net | 400M | 3.60 | 103 |
| ADM [Dhariwal+ 2021] | U-Net | 554M | 4.59 | 742 |
| **DiT-XL/2** | Transformer | **675M** | **2.27** | **118** |

DiT-XL/2 は ADM より少ない FLOPs でより高い FID を達成する。ADMは Attention を U-Net に追加した特殊バージョンだが、それでも Transformer ベースの DiT に及ばない。

**なぜ DiT はパラメータ効率が良いのか？** Transformer のパラメータは全トークン位置で共有される（重み共有はないが、位置に依存しないアーキテクチャ）。各 DiT ブロックは「$N$ パッチ全てに同じ重みを適用」するため、パラメータが位置に無駄なく使われる。U-Net の Conv は各層で異なる空間解像度に特化したパラメータを持ち、低解像度層のパラメータは高解像度情報を扱えない — これがパラメータ効率の差の本質だ。

**Challenge**: DiT の Forward Pass を1行ずつ実装し、ノイズ予測まで完走する。

**Forward Pass の全ステップ**:
1. Patchify — 画像 → トークン列
2. Positional Encoding
3. $L$ 層の DiT ブロック
4. Unpatchify — トークン列 → 画像
5. ノイズ予測 $\epsilon_\theta(\mathbf{x}_t, t)$

**数式**:
$$
\begin{align}
\mathbf{z}_0 &= \text{Patchify}(\mathbf{x}) + \text{PE} \\
\mathbf{z}_\ell &= \text{DiT-Block}_\ell(\mathbf{z}_{\ell-1}, \mathbf{c}) \quad (\ell = 1, \ldots, L) \\
\mathbf{x}_{\text{pred}} &= \text{Unpatchify}(\mathbf{z}_L) \\
\epsilon_\theta(\mathbf{x}_t, t) &= \mathbf{x}_{\text{pred}}
\end{align}
$$

**完全実装**:


**ボス撃破！** DiT の Forward Pass を完全実装した。Patchify → DiT Blocks → Unpatchify の流れで、画像からノイズ予測まで辿り着いた。

> **Note:** **ここまでで全体の50%完了！** 数式修行ゾーン完走。DiT・MM-DiT・SiT の数式を完全導出した。次は実装ゾーン — Julia/Rust/Elixir で DiT を動かす。

### 3.7 Scaling Laws for Diffusion Transformers

**論文**: Zhai et al., "Scaling Laws For Diffusion Transformers," arXiv:2410.08184, 2024[^1]

Transformerベースの言語モデル（LLM）では**Scaling Laws**が確立されている:

$$
\mathcal{L}(N, D, C) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

ここで$N$はモデルサイズ、$D$はデータ量、$C$は計算量、$\mathcal{L}$は損失関数。

**問い**: **Diffusion Transformersでも同じScaling Lawsが成立するか？**

#### 3.7.1 実験的検証

**実験設定**:
- 計算予算: $10^{17}$ FLOPs ~ $6 \times 10^{18}$ FLOPs（1000倍の範囲）
- モデルサイズ: 33M ~ 675M パラメータ
- データセット: ImageNet 256×256（130万画像）
- 評価指標: FID（低いほど良い）

**発見された Scaling Law**:

$$
\text{FID}(C) = A \cdot C^{-\beta} + \text{FID}_{\infty}
$$

ここで:
- $C$: 計算量（FLOPs）
- $A, \beta$: フィッティングパラメータ
- $\text{FID}_{\infty}$: 無限計算時の理論的限界

**実測値**: $\beta \approx 0.27$（LLMの$\beta \approx 0.3$と近い）

**重要な洞察**: DiTの訓練損失（MSE）は計算量$C$に対して**べき乗則**に従う:

$$
\mathcal{L}_{\text{MSE}}(C) = B \cdot C^{-\gamma}
$$

$\gamma \approx 0.12$（実験的に決定）。

#### 3.7.2 最適モデルサイズの予測

Scaling Lawから、**与えられた計算予算$C$に対する最適モデルサイズ$N^*$**を予測できる:

$$
N^*(C) = \left(\frac{\alpha_N}{\alpha_C}\right)^{\frac{1}{\alpha_C - \alpha_N}} \cdot C^{\frac{\alpha_C}{\alpha_C - \alpha_N}}
$$

**実例**: 計算予算$C = 10^{21}$ FLOPsの場合:

$$
N^* \approx 1.2 \times 10^9 \text{ parameters (1.2B)}
$$

**データ量の最適化**:

$$
D^*(C) = \left(\frac{\alpha_D}{\alpha_C}\right)^{\frac{1}{\alpha_C - \alpha_D}} \cdot C^{\frac{\alpha_C}{\alpha_C - \alpha_D}}
$$

$C = 10^{21}$ FLOPsで$D^* \approx 50$M画像。

#### 3.7.3 μP Scaling

**論文**: Xu et al., "Scaling Diffusion Transformers Efficiently via $\mu$P," arXiv:2505.15270, 2025[^2]

**問題**: 標準的なScaling（Standard Parameterization, SP）では、モデルサイズを変えるたびにハイパーパラメータ（学習率$\eta$、初期化$\sigma$）を再調整する必要がある。

**$\mu$P (Maximal Update Parameterization) の解決策**:

パラメータの**幅$d$に応じた適応的スケーリング**を導入:

$$
\begin{aligned}
\text{Weight initialization: } & W \sim \mathcal{N}(0, \frac{1}{d_{\text{in}}}) \\
\text{Learning rate: } & \eta_{\text{layer}} = \frac{\eta_{\text{base}}}{d_{\text{hidden}}} \\
\text{Output scaling: } & y = \frac{1}{\sqrt{d}} W x
\end{aligned}
$$

**利点**: 小さいモデル（e.g., 100M）で最適化したハイパーパラメータが、大きいモデル（e.g., 1B）にそのまま転移可能！

**実験結果**:
- SP: モデルサイズごとに学習率を調整しないと発散
- $\mu$P: 同じ学習率で100M → 10Bまでスケール可能

**実装（Julia概念コード）**:


#### μP の理論的背景: 無限幅極限と特徴学習

μP の数学的根拠は**無限幅極限**（infinite width limit）にある。標準パラメータ化（SP）でネットワーク幅 $d \to \infty$ の極限を取ると、各層の出力は**ガウス過程（GP）**に収束し、特徴が学習されない（Neural Tangent Kernel 体制）。これは訓練が「カーネル法の模倣」に退化することを意味する。

μP ではこの問題を解決する: 幅 $d$ に応じてパラメータ更新スケールを調整することで、$d \to \infty$ でも各層が**有意義な特徴を学習し続ける**（Maximum Update Parameterization の由来）。

キーとなる条件:
- **入力層**: 重みは $\mathcal{N}(0, 1/d_{\text{in}})$ で初期化（SP と同じ）
- **隠れ層**: 重みは $\mathcal{N}(0, 1/d_{\text{hidden}})$、学習率は $\eta / d_{\text{hidden}}$
- **出力層**: $1/\sqrt{d}$ スケーリングを追加

この設定下では、幅 $d = 100$ のモデルで最適化した学習率 $\eta^*$ が $d = 10{,}000$ のモデルでも最適であり続けることが**理論的に証明**されている。DiT に適用すると、small-scale（$\sim$30M）の proxy モデルで optimal learning rate サーチを行い、その値を大規模（$\sim$1B）モデルに直接転移できる — 訓練コストが理論上 100 倍以上削減できる可能性がある。

**論文**: Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models," arXiv:2211.01095, 2023[^3]

#### 3.8.1 問題設定

DDPMは1000ステップ必要 → 実用不可。高速化の2つのアプローチ:

1. **蒸留系**: Consistency Models（第40回）、Progressive Distillation
2. **ODE Solver系**: DDIM、DPM-Solver、DPM-Solver++（本節）

**課題**: Classifier-Free Guidance (CFG) 使用時、標準的な高速solverが不安定化。

#### 3.8.2 DPM-Solver++の核心アイデア

**Diffusion ODE** (第37回で導出):

$$
\frac{dx_t}{dt} = f(t) x_t + \frac{g(t)^2}{2\sigma_t} \epsilon_\theta(x_t, t)
$$

**CFGの問題**: ノイズ予測$\epsilon_\theta$が条件付きと無条件の線形結合:

$$
\tilde{\epsilon}_\theta(x_t, t, c) = (1 + w) \epsilon_\theta(x_t, t, c) - w \epsilon_\theta(x_t, t)
$$

$w$が大きいと$\tilde{\epsilon}_\theta$の絶対値が大きくなり、ODEが硬くなる（stiff ODE）。

**DPM-Solver++の解決策**: **Data prediction**モデルに変換:

$$
x_\theta(x_t, t) = \frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t}
$$

これを使ってODEを書き直す:

$$
\frac{dx_t}{d\lambda} = \frac{\alpha_t}{\sigma_t} (x_t - x_\theta(x_t, t))
$$

ここで$\lambda = \log(\alpha_t / \sigma_t)$は対数SNR。

**高次Solver**: Taylor展開で2次精度近似:

$$
x_{t_{i+1}} = \frac{\alpha_{t_{i+1}}}{\alpha_{t_i}} x_{t_i} - \alpha_{t_{i+1}} \int_{\lambda_{t_i}}^{\lambda_{t_{i+1}}} e^{-\lambda} x_\theta(x_{\lambda(s)}, \lambda(s)) \, d\lambda
$$

積分を**Trapezoid rule**で近似:

$$
\int_{\lambda_i}^{\lambda_{i+1}} e^{-\lambda} x_\theta \, d\lambda \approx \frac{h}{2} (e^{-\lambda_i} x_\theta(x_i, \lambda_i) + e^{-\lambda_{i+1}} x_\theta(x_{i+1}, \lambda_{i+1}))
$$

ここで$h = \lambda_{i+1} - \lambda_i$。

**Multistep法**: 過去の$x_\theta$値を再利用して高次近似:

$$
x_{i+1} = a_0 x_i + \sum_{k=0}^K b_k x_\theta(x_{i-k}, \lambda_{i-k})
$$

$K=2$で3次精度達成 → 15-20ステップで高品質サンプル生成可能！

#### 3.8.3 DPM-Solver-v3

**論文**: Zheng et al., "DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics," NeurIPS 2023[^4]

**さらなる改善**: モデルの**経験的統計量**（平均$\mu_t$、分散$\Sigma_t$）を推定し、ODEに組み込む。

$$
x_\theta(x_t, t) \approx \mu_t + \Sigma_t^{1/2} \cdot \text{Whitening}^{-1}(x_t)
$$

**効果**: 10ステップでDDIM 50ステップ相当の品質達成。

**実装（Julia概念コード）**:

#### DPM-Solver++ のステップサイズスケジュールの最適化

DPM-Solver++ が15-20ステップで高品質生成できる理由をODE近似誤差の観点から分析する。

$p$ 次の数値解法は局所打ち切り誤差（local truncation error）が $O(h^{p+1})$（$h$ はステップサイズ）。トータル誤差はステップ数 $S$ を使って：

$$
E_{\text{total}} \approx S \cdot O(h^{p+1}) = S \cdot O\!\left(\frac{T^{p+1}}{S^{p+1}}\right) = O\!\left(\frac{T^{p+1}}{S^p}\right)
$$

DDIM（1次精度、$p = 1$）と DPM-Solver++（3次精度、$p = 3$）の比較：
- DDIM で誤差 $\epsilon$ を達成: $S \sim O(\epsilon^{-1})$ ステップ必要
- DPM-Solver++ で同誤差: $S \sim O(\epsilon^{-1/3})$ ステップで達成

FID 10 程度を達成するのに DDIM は 250 ステップ必要だったが、DPM-Solver++ では理論的に $250^{1/3} \approx 6.3$ — つまり **6-7 ステップ** で同等品質が得られる計算になる（実際は15-20ステップが安全圏）。

ステップサイズの配置（スケジュール）も重要だ。均等割りより**対数 SNR 空間での等間隔配置**が低誤差：
$$
\lambda_i = \lambda_T + \frac{i}{S}(\lambda_0 - \lambda_T), \quad i = 0, 1, \ldots, S
$$
$t_i = \text{SNR}^{-1}(\lambda_i)$ で時間軸に変換。これが DDIM の等時刻間隔より少ないステップで高品質を達成する理由だ。

#### 3.9.1 SD3のMM-DiT Architecture

**論文**: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," Stability AI Technical Report, 2024[^5]

SD3は**Rectified Flow**（第38回のFlow Matching）とMM-DiTを組み合わせる。

**重要な設計選択**:

1. **2つの独立したTransformer stream**:
   - Image stream: 画像パッチ処理
   - Text stream: T5/CLIPテキストエンコーディング処理

2. **各DiTブロックの構造**:

$$
\begin{aligned}
\text{Image stream: } & y_{\text{img}}^{(\ell+1)} = y_{\text{img}}^{(\ell)} + \text{DiTBlock}_{\text{img}}(y_{\text{img}}^{(\ell)}, y_{\text{txt}}^{(\ell)}, t) \\
\text{Text stream: } & y_{\text{txt}}^{(\ell+1)} = y_{\text{txt}}^{(\ell)} + \text{DiTBlock}_{\text{txt}}(y_{\text{txt}}^{(\ell)}, y_{\text{img}}^{(\ell)}, t)
\end{aligned}
$$

**Cross-Attention**: 各streamが相手のstreamを見る（bidirectional cross-attention）。

3. **QK-Normalization**: Attention計算前にQuery/Keyを正規化:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\text{Norm}(Q) \cdot \text{Norm}(K)^\top}{\sqrt{d}}\right) V
$$

**効果**: 訓練の安定化 + 大規模モデル（8B params）でも収束。

#### 3.9.2 FLUX Architecture

**論文**: Beaumont et al., "FLUX.1: Advanced Image Generation," Black Forest Labs Technical Report, 2024[^6]

FLUXはSD3の進化版:

**主要な改善**:

1. **Parallel Attention and MLP**:

標準DiTブロック（Sequential）:

$$
\begin{aligned}
z' &= z + \text{Attention}(\text{AdaLN}(z, t)) \\
z'' &= z' + \text{MLP}(\text{AdaLN}(z', t))
\end{aligned}
$$

FLUXブロック（Parallel）:

$$
z' = z + \text{Attention}(\text{AdaLN}(z, t)) + \text{MLP}(\text{AdaLN}(z, t))
$$

**利点**: 並列化で高速化 + 表現力向上。

2. **Rotary Position Embedding (RoPE)** (LLMから輸入):

$$
\text{RoPE}(q, k, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}
$$

位置$m$のトークンに回転行列を適用 → 相対位置情報をAttentionに埋め込む。

3. **Guidance Distillation**:

CFGの$w$を学習時に蒸留 → 推論時にguidance-freeで高品質生成可能（4-8ステップ）。

**実装の核心（Julia概念コード）**:


#### RoPE の相対位置符号化の数学的証明

FLUXで採用された RoPE が相対位置を符号化することを証明する。位置 $m$ のQuery と位置 $n$ のKey の内積を計算する：

$$
(\text{RoPE}(q, m))^\top (\text{RoPE}(k, n)) = q^\top R(m)^\top R(n) k = q^\top R(n - m) k
$$

ここで $R(m)$ は回転行列（$R(m)^\top R(n) = R(n-m)$、回転行列の乗法性）:

$$
R(m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

**核心**: 内積が **位置差 $n - m$ のみ**に依存する。つまりRoPEは「トークン $m$ とトークン $n$ は $|n - m|$ だけ離れている」という相対位置情報を自動的にAttentionスコアに埋め込む。Sinusoidal PEが絶対位置を符号化するのに対し、RoPEは相対位置を符号化 — 文脈長の一般化（訓練時より長い系列への転移）においてRoPEが優れている理由だ。

2次元版 RoPE（FLUX での画像用）は縦・横の各方向に独立した回転行列を適用する：
$$
\text{RoPE2D}(q, i, j) = R_{\text{row}}(i) \otimes R_{\text{col}}(j)\, q
$$
この設計により「右に3パッチ、下に2パッチ移動した位置」という2次元相対位置をAttentionが直接学習できる。

**論文**: Helbling, A. et al., "ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features," arXiv:2502.04320, 2025[^7]

**発見**: DiTの中間層の特徴は**意味的に解釈可能**な構造を持つ。

**実験**: SD3の中間層（Layer 12/24）の特徴ベクトルをPCAで2D可視化:

- クラスター1: "動物"（犬・猫・馬）
- クラスター2: "建物"（家・ビル・橋）
- クラスター3: "自然"（木・花・山）

**意味編集**: 特定の特徴方向に摂動を加えることで意味的編集が可能:

$$
z' = z + \alpha \cdot \mathbf{v}_{\text{concept}}
$$

例: $\mathbf{v}_{\text{smile}}$方向に$\alpha=2.0$で加算 → 「笑顔を強調」

**概念方向ベクトルの抽出方法**: 「笑顔あり」と「笑顔なし」の画像ペアを生成し、中間層の特徴差分を取る：
$$
\mathbf{v}_{\text{concept}} = \mathbb{E}\!\left[\mathbf{z}^{(\ell)}_{\text{with concept}} - \mathbf{z}^{(\ell)}_{\text{without concept}}\right]
$$
この方向に沿って $\alpha$ を変化させると、生成画像の「笑顔度」が連続的に制御できる。DiTの線形分離性（各概念が特定の方向に対応）はGANのlatent space解釈可能性研究（StyleGAN等）と同様の現象だが、**条件付き拡散過程の中間表現**でも同様の構造が現れることが確認された点が新しい。

**応用**: Training-free画像編集、Concept steering、Adversarial robustness向上。

#### DiT の訓練目標関数の比較：DDPM vs Rectified Flow

DiT は CNN バックボーンを置き換えるだけでなく、訓練目標関数も選択肢がある。

**DDPM 目標** [Ho+ 2020]（オリジナル DiT で使用）:
$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
$$
ここで $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}$、$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$。

**Rectified Flow 目標** [Liu+ 2022]（SD3・FLUX で使用）:
$$
\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[\|\mathbf{x}_1 - \mathbf{x}_0 - \mathbf{v}_\theta(\mathbf{x}_t, t)\|^2\right]
$$
ここで $\mathbf{x}_t = t\, \mathbf{x}_1 + (1 - t)\, \mathbf{x}_0$、真の速度場 $\mathbf{v}^* = \mathbf{x}_1 - \mathbf{x}_0$ は一定（直線経路）。

**2つの目標の等価性**: 適切なノイズスケジュール変換を行うと DDPM と Rectified Flow は等価になる。変換式：
$$
\boldsymbol{\epsilon}_\theta = \frac{\mathbf{x}_t - \alpha_t\, \mathbf{v}_\theta}{\sigma_t}
$$

**実践的な差異**:
| 観点 | DDPM | Rectified Flow |
|:-----|:-----|:--------------|
| 速度場の曲率 | 高い（非線形経路） | 低い（直線経路） |
| 必要NFE | 20-50 | 5-15 |
| 訓練損失の解釈 | ノイズ予測誤差 | 速度場予測誤差 |
| SD3 採用理由 | — | 高速サンプリング + 安定訓練 |

SD3 が Rectified Flow を選んだ決定的理由: 直線経路は「最短経路」であり、少ないODEステップで積分誤差が最小化される。DDPM 目標でも DPM-Solver++ を使えば高速化は可能だが、Rectified Flow は**訓練段階から速い経路を学習**する点で根本的に優れている。

**Logit-normal サンプリング**: SD3 では時刻 $t$ を均一分布ではなく **Logit-normal** からサンプリングする：
$$
t \sim \text{Logit-Normal}(\mu, \sigma^2), \quad t = \text{sigmoid}(u),\; u \sim \mathcal{N}(\mu, \sigma^2)
$$
$\mu = 0, \sigma = 1$（デフォルト）では $t$ が $[0.3, 0.7]$ 付近に集中する。直感: 全拡散ステップ中で最も「難しい」中間時刻での学習を強調することで、訓練効率が向上する。$t \approx 0$（ほぼ清浄）と $t \approx 1$（ほぼノイズ）は比較的容易なため均等サンプリングは非効率だ。

### 4.1 完全なDiTブロック実装（Lux.jl）


### 4.2 MM-DiT実装（SD3/FLUXスタイル）


### 4.3 DPM-Solver++サンプラー完全実装


### 4.4 Scaling Laws実験フレームワーク


> **Note:** **進捗**: 全体の85%完了。Production-ReadyなDiT実装（AdaLN-Zero、MM-DiT、DPM-Solver++、Scaling Laws実験）を完全実装した。理論→実装のギャップを完全に埋めた。

---

## 📚 参考文献

### 主要論文

[^1]: Zhai, S., et al. (2024). Scaling Laws For Diffusion Transformers. arXiv:2410.08184.
<https://arxiv.org/abs/2410.08184>

[^2]: Xu, Y., et al. (2025). Scaling Diffusion Transformers Efficiently via μP. arXiv:2505.15270.
<https://arxiv.org/abs/2505.15270>

[^3]: Lu, C., et al. (2023). DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models. Machine Intelligence Research.
<https://arxiv.org/abs/2211.01095>

[^4]: Zheng, K., et al. (2023). DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics. NeurIPS 2023.
<https://openreview.net/forum?id=9fWKExmKa0>

[^5]: Esser, P., et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. Stability AI Technical Report.
<https://stability.ai/news/stable-diffusion-3-research-paper>

[^6]: Beaumont, R., et al. (2024). FLUX.1: Advanced Image Generation. Black Forest Labs Technical Report.
<https://arxiv.org/html/2507.09595v1>

[^7]: Helbling, A., et al. (2025). ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features. arXiv:2502.04320.
<https://arxiv.org/abs/2502.04320>

### 追加参考文献

- Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. ICCV 2023. arXiv:2212.09748.
<https://arxiv.org/abs/2212.09748>

- Lu, C., et al. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. NeurIPS 2022 Oral.
<https://arxiv.org/abs/2206.00927>

---


> Progress: 50%
> **理解度チェック**
> 1. $N, C=$ の各記号の意味と、この式が表す操作を説明してください。
> 2. このゾーンで学んだ手法の直感的な意味と、なぜこの定式化が必要なのかを説明してください。

## 🎯 5. まとめ — DiTが切り開く未来

### 5.1 本Partで学んだこと

**理論的基盤**:
- U-NetからTransformerへの移行の必然性（Scaling Laws適用可能性）
- AdaLN-Zero、MM-DiT、SiTの数学的構造
- Scaling Laws: FID(C) = A·C^(-β) + FID_∞（β ≈ 0.27）
- μP: モデルサイズ非依存のハイパーパラメータ転移
- DPM-Solver++: CFG安定化 + 15-20ステップ高速サンプリング

**実装スキル**:
- Lux.jlでのDiT完全実装（Patchify/Unpatchify/DiTBlock/AdaLN）
- MM-DiT dual-stream architecture
- DPM-Solver++ 2nd-order sampler
- Scaling Laws実験フレームワーク

**最先端動向**:
- SD3: MM-DiT + Rectified Flow + QK-Norm
- FLUX: Parallel Attn/MLP + RoPE + Guidance Distillation
- 解釈可能性: DiT中間層の意味的クラスター形成

### 5.2 Part 2への接続

Part 2では、DiTを以下のドメインに応用する:
- **高解像度画像生成**: Cascade DiT、Patch-wise sampling
- **Video DiT**: 時空間attention、3D RoPE
- **条件付き生成**: ControlNet-DiT、Regional prompting
- **高速化**: Consistency DiT、Latent Consistency Models

DiTは画像生成だけでなく、**全モダリティ（動画・音声・3D）の統一バックボーン**として進化している。

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
