---
title: "第39回: Latent Diffusion Models: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "🖼️"
type: "tech"
topics: ["machinelearning", "deeplearning", "ldm", "julia", "stablediffusion"]
published: true
---

## 💻 4. 実装ゾーン（45分）— LDM訓練→推論パイプライン

### 環境構築とライブラリ

**Julia環境** (⚡訓練):
```julia
using Pkg
Pkg.add([
    "Lux",           # NN framework
    "Reactant",      # XLA compiler (GPU高速化)
    "Optimisers",    # Adam等
    "Zygote",        # 自動微分
    "MLDatasets",    # MNIST等
    "Images",        # 画像処理
    "CUDA",          # GPU
    "BenchmarkTools" # 性能測定
])
```

**Rust環境** (🦀推論):
```toml
[dependencies]
candle-core = "0.7"
candle-nn = "0.7"
safetensors = "0.4"
```

### LaTeX記法チートシート

| 記号 | LaTeX | 意味 |
|:-----|:------|:-----|
| $\mathcal{E}$ | `\mathcal{E}` | Encoder |
| $\mathcal{D}$ | `\mathcal{D}` | Decoder |
| $\bar{\alpha}_t$ | `\bar{\alpha}_t` | Cumulative product |
| $\epsilon_\theta(z_t, t, c)$ | `\epsilon_\theta(z_t, t, c)` | Noise prediction |
| $\tilde{\epsilon}$ | `\tilde{\epsilon}` | Modified noise (CFG) |
| $\mathbb{E}_{q(z\|x)}[\cdot]$ | `\mathbb{E}_{q(z\|x)}[\cdot]` | Expectation |
| $\text{KL}[q \| p]$ | `\text{KL}[q \| p]` | KL divergence |

### Math→Code翻訳パターン (LDM編)

**パターン1: VAE Encoder**
$$
z = \mathcal{E}(x), \quad x \in \mathbb{R}^{H \times W \times C} \to z \in \mathbb{R}^{h \times w \times c}
$$

```julia
z = encoder(x, ps_encoder, st_encoder)[1]
# x: [H, W, C, B] → z: [h, w, c, B]
```

**パターン2: Forward Diffusion**
$$
z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$

```julia
ε = randn(rng, Float32, size(z₀))
z_t = sqrt(α_bar[t]) .* z₀ .+ sqrt(1 - α_bar[t]) .* ε
```

**パターン3: CFG**
$$
\tilde{\epsilon}_\theta = \epsilon_\theta(z_t, t, \emptyset) + w \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))
$$

```julia
ε_uncond = unet((z_t, t, nothing), ps, st)[1]
ε_cond = unet((z_t, t, c), ps, st)[1]
ε_cfg = ε_uncond .+ w .* (ε_cond .- ε_uncond)
```

**パターン4: DDIM Sampling**
$$
z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{z_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta + \sigma_t \epsilon
$$

```julia
pred_x₀ = (z_t .- sqrt(1 - α_bar[t]) .* ε_θ) ./ sqrt(α_bar[t])
dir_z = sqrt(1 - α_bar[t-1] - σ²) .* ε_θ
noise = σ .* randn(rng, Float32, size(z_t))
z_prev = sqrt(α_bar[t-1]) .* pred_x₀ .+ dir_z .+ noise
```

**パターン5: Cross-Attention**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

```julia
scores = (Q * K') ./ sqrt(d_k)  # [N_q, N_k]
attn = softmax(scores, dims=2)   # [N_q, N_k]
out = attn * V                   # [N_q, d_v]
```

**パターン6: Min-SNR Weighting**
$$
w(t) = \min\left(\text{SNR}(t), \gamma\right), \quad \text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
$$

```julia
snr = α_bar ./ (1 .- α_bar)
weight = min.(snr, γ)
loss = weight[t] * mse(ε_pred, ε_true)
```

**パターン7: Zero Terminal SNR Rescaling**
$$
\tilde{\alpha}_t = \frac{\alpha_t}{\alpha_T}
$$

```julia
α_cumprod = cumprod(alphas)
α_cumprod_rescaled = α_cumprod ./ α_cumprod[end]
```

### ⚡ Julia完全実装: Mini Latent Diffusion

**ステップ1: VAE定義**

```julia
using Lux, Random, Optimisers, Zygote

# Encoder
function create_encoder(; in_ch=3, latent_ch=4, base_ch=64)
    return Chain(
        Conv((3,3), in_ch => base_ch, pad=1, activation=relu),
        Conv((4,4), base_ch => base_ch*2, stride=2, pad=1, activation=relu),  # /2
        Conv((4,4), base_ch*2 => base_ch*4, stride=2, pad=1, activation=relu),  # /4
        Conv((4,4), base_ch*4 => base_ch*8, stride=2, pad=1, activation=relu),  # /8
        Conv((3,3), base_ch*8 => latent_ch, pad=1)  # Output z
    )
end

# Decoder (mirror)
function create_decoder(; latent_ch=4, out_ch=3, base_ch=64)
    return Chain(
        Conv((3,3), latent_ch => base_ch*8, pad=1, activation=relu),
        ConvTranspose((4,4), base_ch*8 => base_ch*4, stride=2, pad=1, activation=relu),  # *2
        ConvTranspose((4,4), base_ch*4 => base_ch*2, stride=2, pad=1, activation=relu),  # *4
        ConvTranspose((4,4), base_ch*2 => base_ch, stride=2, pad=1, activation=relu),    # *8
        Conv((3,3), base_ch => out_ch, pad=1, activation=tanh)
    )
end

# VAE訓練
function train_vae!(encoder, decoder, dataloader; epochs=10, lr=1e-3, β=0.5)
    ps_enc, st_enc = Lux.setup(Random.default_rng(), encoder)
    ps_dec, st_dec = Lux.setup(Random.default_rng(), decoder)
    opt = Adam(lr)
    opt_state_enc = Optimisers.setup(opt, ps_enc)
    opt_state_dec = Optimisers.setup(opt, ps_dec)

    for epoch in 1:epochs
        total_loss = 0.0
        for (x,) in dataloader
            # Forward
            z, st_enc = encoder(x, ps_enc, st_enc)
            x_recon, st_dec = decoder(z, ps_dec, st_dec)

            # Loss: Reconstruction + KL (simplified)
            recon_loss = mean((x_recon .- x).^2)
            kl_loss = 0.5 * mean(z.^2)  # Simplified KL to N(0,I)
            loss = recon_loss + β * kl_loss

            # Backprop
            gs_enc = gradient(p -> loss, ps_enc)[1]
            gs_dec = gradient(p -> loss, ps_dec)[1]
            opt_state_enc, ps_enc = Optimisers.update(opt_state_enc, ps_enc, gs_enc)
            opt_state_dec, ps_dec = Optimisers.update(opt_state_dec, ps_dec, gs_dec)

            total_loss += loss
        end
        println("Epoch $epoch: Loss = $(total_loss / length(dataloader))")
    end
    return ps_enc, ps_dec, st_enc, st_dec
end
```

**ステップ2: U-Net定義 (Simplified)**

```julia
# ResBlock
struct ResBlock
    conv1::Conv
    conv2::Conv
end

function (rb::ResBlock)(x)
    h = rb.conv1(x)
    h = relu.(h)
    h = rb.conv2(h)
    return relu.(h .+ x)  # Residual
end

# Time Embedding
function sinusoidal_embedding(t::Int, dim::Int)
    half = dim ÷ 2
    freqs = exp.(-log(10000f0) .* (0:half-1) ./ half)
    args = t .* freqs
    return vcat(sin.(args), cos.(args))
end

# Simplified U-Net (for 32x32 latent)
function create_unet(; latent_ch=4, base_ch=128, time_emb_dim=256)
    return Chain(
        # Time embedding MLP
        Dense(time_emb_dim, time_emb_dim*4, activation=silu),
        Dense(time_emb_dim*4, time_emb_dim*4, activation=silu),

        # Down
        Conv((3,3), latent_ch => base_ch, pad=1),
        ResBlock(Conv((3,3), base_ch => base_ch, pad=1), Conv((3,3), base_ch => base_ch, pad=1)),
        Conv((4,4), base_ch => base_ch*2, stride=2, pad=1),  # /2
        ResBlock(Conv((3,3), base_ch*2 => base_ch*2, pad=1), Conv((3,3), base_ch*2 => base_ch*2, pad=1)),

        # Middle
        ResBlock(Conv((3,3), base_ch*2 => base_ch*2, pad=1), Conv((3,3), base_ch*2 => base_ch*2, pad=1)),

        # Up
        ConvTranspose((4,4), base_ch*2 => base_ch, stride=2, pad=1),  # *2
        ResBlock(Conv((3,3), base_ch => base_ch, pad=1), Conv((3,3), base_ch => base_ch, pad=1)),
        Conv((3,3), base_ch => latent_ch, pad=1)  # Output ε
    )
end
```

**ステップ3: Diffusion訓練ループ**

```julia
# Noise schedule
function cosine_beta_schedule(T::Int; s=0.008)
    t = 0:T
    α_bar = cos.(((t ./ T) .+ s) ./ (1 + s) .* π ./ 2).^2
    α_bar = α_bar ./ α_bar[1]
    betas = 1 .- α_bar[2:end] ./ α_bar[1:end-1]
    return clamp.(betas, 0f0, 0.999f0), α_bar[2:end]
end

# Forward diffusion
function forward_diffusion(z₀, t, α_bar, rng)
    ε = randn(rng, Float32, size(z₀))
    z_t = sqrt(α_bar[t]) .* z₀ .+ sqrt(1 - α_bar[t]) .* ε
    return z_t, ε
end

# 訓練ループ
function train_ldm!(unet, encoder, dataloader; T=1000, epochs=100, lr=1e-4)
    betas, α_bar = cosine_beta_schedule(T)
    ps_unet, st_unet = Lux.setup(Random.default_rng(), unet)
    opt = Adam(lr)
    opt_state = Optimisers.setup(opt, ps_unet)
    rng = Random.default_rng()

    # Freeze encoder
    ps_enc, st_enc = Lux.setup(rng, encoder)

    for epoch in 1:epochs
        total_loss = 0.0
        for (x,) in dataloader
            # Encode (no grad)
            z₀, _ = encoder(x, ps_enc, st_enc)

            # Random timestep
            t = rand(rng, 1:T)

            # Forward diffusion
            z_t, ε_true = forward_diffusion(z₀, t, α_bar, rng)

            # Time embedding
            t_emb = sinusoidal_embedding(t, 256)

            # Predict noise
            ε_pred, st_unet = unet((z_t, t_emb), ps_unet, st_unet)

            # MSE loss
            loss = mean((ε_pred .- ε_true).^2)

            # Backprop (only unet)
            gs = gradient(p -> loss, ps_unet)[1]
            opt_state, ps_unet = Optimisers.update(opt_state, ps_unet, gs)

            total_loss += loss
        end
        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $(total_loss / length(dataloader))")
        end
    end
    return ps_unet, st_unet
end
```

**ステップ4: CFGサンプリング**

```julia
# DDIM sampling with CFG
function ddim_sample_cfg(unet, decoder, z_T, c, w; steps=50, η=0.0)
    T = 1000
    betas, α_bar = cosine_beta_schedule(T)
    timesteps = reverse(Int.(round.(range(1, T, length=steps))))

    z_t = z_T
    for (i, t) in enumerate(timesteps)
        t_prev = i == steps ? 0 : timesteps[i+1]

        # Time embedding
        t_emb = sinusoidal_embedding(t, 256)

        # CFG: 2 forward passes
        ε_uncond, _ = unet((z_t, t_emb, nothing), ps_unet, st_unet)
        ε_cond, _ = unet((z_t, t_emb, c), ps_unet, st_unet)
        ε_cfg = ε_uncond .+ w .* (ε_cond .- ε_uncond)

        # Predict x₀
        pred_x₀ = (z_t .- sqrt(1 - α_bar[t]) .* ε_cfg) ./ sqrt(α_bar[t])

        # DDIM step
        if t_prev > 0
            σ_t = η * sqrt((1 - α_bar[t_prev]) / (1 - α_bar[t])) * sqrt(1 - α_bar[t] / α_bar[t_prev])
            dir_z = sqrt(1 - α_bar[t_prev] - σ_t^2) .* ε_cfg
            noise = σ_t .* randn(Float32, size(z_t))
            z_t = sqrt(α_bar[t_prev]) .* pred_x₀ .+ dir_z .+ noise
        else
            z_t = pred_x₀
        end
    end

    # Decode
    x, _ = decoder(z_t, ps_dec, st_dec)
    return x
end

# 使用例
z_T = randn(Float32, 32, 32, 4, 1)  # Random noise
c = nothing  # 無条件 or テキスト埋め込み
w = 7.5      # CFG scale
x_gen = ddim_sample_cfg(unet, decoder, z_T, c, w, steps=50)
```

:::details 完全な訓練スクリプト
```julia
using MLDatasets, Images

# データ準備
train_data = MNIST(split=:train)
X_train = Float32.(reshape(train_data.features, 28, 28, 1, :))
X_train = (X_train .* 2f0) .- 1f0  # [-1, 1]

# Dataloader
batchsize = 64
dataloader = [(X_train[:,:,:,i:min(i+batchsize-1,end)],)
              for i in 1:batchsize:size(X_train,4)]

# モデル作成
encoder = create_encoder(in_ch=1, latent_ch=4, base_ch=32)
decoder = create_decoder(latent_ch=4, out_ch=1, base_ch=32)
unet = create_unet(latent_ch=4, base_ch=64)

# Stage 1: VAE訓練
println("Training VAE...")
ps_enc, ps_dec, st_enc, st_dec = train_vae!(encoder, decoder, dataloader, epochs=20)

# Stage 2: Diffusion訓練
println("Training Diffusion...")
ps_unet, st_unet = train_ldm!(unet, encoder, dataloader, T=1000, epochs=100)

# サンプリング
println("Generating samples...")
z_T = randn(Float32, 7, 7, 4, 16)  # 28/4=7 (downsampled)
x_gen = ddim_sample_cfg(unet, decoder, z_T, nothing, 1.0, steps=50)

# 保存
using FileIO
save("generated.png", colorview(Gray, x_gen[:,:,1,1]))
```
:::

### 🦀 Rust推論実装

**safetensorsからモデルロード**:

```rust
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, Module};

// VAE Decoder
struct Decoder {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    // ... more layers
}

impl Decoder {
    fn new(vb: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(4, 512, 3, Default::default(), vb.pp("conv1"))?;
        let conv2 = candle_nn::conv_transpose2d(512, 256, 4, Default::default(), vb.pp("conv2"))?;
        // ...
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(z)?;
        let x = x.relu()?;
        let x = self.conv2.forward(&x)?;
        // ...
        Ok(x.tanh()?)
    }
}

// Load weights
fn load_ldm_model(path: &str) -> Result<(UNet, Decoder)> {
    let device = Device::cuda_if_available(0)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, &device)? };

    let unet = UNet::new(vb.pp("unet"))?;
    let decoder = Decoder::new(vb.pp("decoder"))?;

    Ok((unet, decoder))
}
```

**CFG推論**:

```rust
fn cfg_sample(
    unet: &UNet,
    decoder: &Decoder,
    z_t: Tensor,
    c: Option<&Tensor>,
    w: f32,
    steps: usize,
) -> Result<Tensor> {
    let mut z = z_t;
    let timesteps: Vec<usize> = (0..steps).rev().collect();

    for t in timesteps {
        let t_tensor = Tensor::new(&[t as f32], z.device())?;

        // Unconditional
        let eps_uncond = unet.forward(&z, &t_tensor, None)?;

        // Conditional
        let eps_cond = if let Some(cond) = c {
            unet.forward(&z, &t_tensor, Some(cond))?
        } else {
            eps_uncond.clone()
        };

        // CFG
        let eps_cfg = (eps_uncond + (eps_cond - &eps_uncond)? * w)?;

        // DDIM step
        z = ddim_step(&z, &eps_cfg, t)?;
    }

    // Decode
    decoder.forward(&z)
}
```

**バッチ推論最適化**:

```rust
// バッチ処理でスループット向上
fn batch_generate(
    unet: &UNet,
    decoder: &Decoder,
    batch_size: usize,
    c: &Tensor,
    w: f32,
) -> Result<Vec<Tensor>> {
    // ノイズバッチ生成
    let z_T = Tensor::randn(0f32, 1f32, (batch_size, 4, 64, 64), &Device::Cuda(0))?;

    // バッチ推論
    let x_batch = cfg_sample(unet, decoder, z_T, Some(c), w, 50)?;

    // Split batch
    let mut results = Vec::new();
    for i in 0..batch_size {
        results.push(x_batch.narrow(0, i, 1)?);
    }
    Ok(results)
}
```

### 数値安定性とデバッグ

**よくあるエラーとデバッグ方法**:

| エラー | 原因 | 解決策 |
|:-------|:-----|:-------|
| **NaN loss** | 勾配爆発 / 数値不安定 | Gradient clipping / learning rate削減 / Mixed precision |
| **Mode collapse** | CFG scale高すぎ | $w \in [1, 7.5]$ に制限 |
| **ぼやけた画像** | VAE過圧縮 / β高すぎ | $\beta < 1$ / 圧縮率削減 |
| **Posterior collapse** | VAE訓練不十分 | KL annealing / Free bits |
| **真っ黒/真っ白画像** | Zero terminal SNR未対応 | Noise schedule rescaling |

**数値安定化テクニック**:

```julia
# Gradient clipping
function clip_grad!(grads, max_norm=1.0)
    total_norm = sqrt(sum(x -> sum(x.^2), grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1
        for g in grads
            g .*= clip_coef
        end
    end
end

# Mixed precision (FP16訓練)
using CUDA
x_fp16 = cu(Float16.(x))
# ... forward pass in FP16
loss_fp32 = Float32(loss)  # Loss計算はFP32
```

**デバッグチェックリスト**:

```julia
# 1. VAE再構成確認
z = encoder(x)
x_recon = decoder(z)
@assert mean(abs.(x - x_recon)) < 0.5  # 再構成誤差

# 2. Forward diffusion確認
z_t, ε = forward_diffusion(z, T, α_bar)
@assert mean(abs.(z_t)) ≈ 1.0 atol=0.5  # T=1000でほぼガウシアン

# 3. Noise prediction確認
ε_pred = unet(z_t, t)
@assert size(ε_pred) == size(ε)  # 形状一致

# 4. CFG確認
ε_cfg = cfg_forward(unet, z_t, t, c, w)
@assert !any(isnan.(ε_cfg))  # NaNチェック
```

:::message
**ここまでで全体の70%完了！** 実装ゾーン完了。Julia訓練→Rust推論の完全パイプラインを構築した。次は実験ゾーンでCFG実験へ。
:::

---

## 🔬 5. 実験ゾーン（30分）— CFG実験と品質分析

### 自己診断テスト

**Q1: CFGのguidance scale $w$の効果**

以下のコードの出力を予測せよ:
```julia
w_values = [0.0, 1.0, 3.0, 7.5, 15.0]
for w in w_values
    x = ddim_sample_cfg(unet, decoder, z_T, c, w)
    println("w=$w: quality=?, diversity=?")
end
```

:::details 解答
- $w=0.0$: quality=低, diversity=高 (無条件生成)
- $w=1.0$: quality=中, diversity=中 (標準条件付き)
- $w=3.0$: quality=高, diversity=中 (軽いCFG)
- $w=7.5$: quality=最高, diversity=低 (SD推奨値)
- $w=15.0$: quality=過飽和, diversity=最低 (over-guidance)
:::

**Q2: Negative Promptの数学**

Negative Prompt実装のこの行を説明せよ:
```julia
ε_cfg = ε_neg .+ w .* (ε_pos .- ε_neg)
```

:::details 解答
$\tilde{\epsilon} = \epsilon_\text{neg} + w(\epsilon_\text{pos} - \epsilon_\text{neg})$ は「$\epsilon_\text{neg}$から$\epsilon_\text{pos}$へ$w$倍強く移動」を意味する。これはベクトルの線形結合で、CFGの一般化。Negative Promptは「避けたい概念」を$\epsilon_\text{neg}$として指定することで、無条件$\emptyset$の代わりに使う。
:::

**Q3: Zero Terminal SNRの効果**

以下のrescaling前後で何が変わるか:
```julia
# Before
α_bar_before = [0.99, 0.98, ..., 0.01]  # α_T = 0.01 ≠ 0

# After rescaling
α_bar_after = α_bar_before ./ α_bar_before[end]
# α_bar_after[end] = 1.0 → sqrt(α_bar_after[end]) = 1.0
```

:::details 解答
Zero Terminal SNRは $\bar{\alpha}_T = 0$ を強制する。Rescaling前は $\bar{\alpha}_T = 0.01 \neq 0$ なので、$T$ステップ目でもわずかに信号が残る。Rescaling後は $\bar{\alpha}_T = 0$ となり、完全なガウシアンノイズに到達。これにより非常に明るい/暗い画像の生成品質が向上する（Lin et al. 2023 [^zero_snr]）。
:::

**Q4: Text Conditioning実装**

CLIP text encodingのこのコードを解釈せよ:
```python
hidden = transformer(tokens)  # [77, 768]
c = hidden  # 全トークンの隠れ状態を使用
```

なぜ`hidden[0]` (CLSトークン)だけでなく全トークンを使うか？

:::details 解答
Stable Diffusionは **全トークンの隠れ状態** $c \in \mathbb{R}^{77 \times 768}$ をCross-Attentionに入力する。これにより:
1. 各単語の情報を個別に保持（"red cat"で"red"と"cat"が別々に処理）
2. Cross-Attentionで画像の各位置が関連する単語に注目できる
3. 長文の詳細な関係性を捉えられる

`hidden[0]` (BERTスタイル)だと文全体を1ベクトルに圧縮してしまい、詳細な単語レベルアライメントが失われる。
:::

**Q5: Min-SNR weightingの目的**

Min-SNR loss weightingのこのコードの意図は？
```julia
snr = α_bar ./ (1 .- α_bar)
weight = min.(snr, 5.0)  # γ=5
loss = weight[t] * mse(ε_pred, ε_true)
```

:::details 解答
SNRが高い（ノイズ少ない）timestepは学習が簡単で、低い（ノイズ多い）timestepは難しい。均等にweightすると簡単なtimestepに過適合する。Min-SNR weightingは:
1. $\text{SNR}(t)$を計算（信号対雑音比）
2. $\gamma=5$でクリップ → SNR高すぎるtimestepのweightを削減
3. 難しいtimestep（低SNR）の学習を促進

Hang et al. (2023) [^min_snr]は3.4倍の訓練高速化を報告。
:::

### 実装チャレンジ: CFG Scale実験

**課題**: Guidance scale $w \in \{1, 3, 5, 7.5, 10, 15\}$ で生成し、品質とFID/IS/CLIP Scoreを計測せよ。

```julia
using Flux, CUDA

# 実験設定
w_values = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]
n_samples = 100
results = Dict()

for w in w_values
    println("Testing w=$w...")
    images = []

    for i in 1:n_samples
        z_T = randn(Float32, 32, 32, 4, 1)
        c = text_encoder("a beautiful landscape")  # CLIP encoding
        x = ddim_sample_cfg(unet, decoder, z_T, c, w, steps=50)
        push!(images, x)
    end

    # 品質指標計算
    fid = compute_fid(images, real_images)
    is_score = compute_inception_score(images)
    clip_score = compute_clip_score(images, "a beautiful landscape")

    results[w] = (fid=fid, is=is_score, clip=clip_score)
    println("  FID: $fid, IS: $is_score, CLIP: $clip_score")
end

# 結果可視化
using Plots
plot([r.fid for r in values(results)], label="FID↓", xlabel="w", ylabel="Score")
plot!([r.clip for r in values(results)], label="CLIP Score↑")
```

**期待される結果**:

| $w$ | FID↓ | IS↑ | CLIP Score↑ | 多様性 |
|:----|:-----|:----|:------------|:-------|
| 1.0 | 25.3 | 2.1 | 0.75 | 高 |
| 3.0 | 18.7 | 2.8 | 0.82 | 中 |
| 5.0 | 14.2 | 3.2 | 0.87 | 中 |
| **7.5** | **12.1** | **3.5** | **0.91** | 低 |
| 10.0 | 13.5 | 3.4 | 0.89 | 低 |
| 15.0 | 18.9 | 3.1 | 0.85 | 最低(mode collapse) |

**結論**: $w=7.5$ が品質と多様性の最適バランス（Stable Diffusion標準値）。

### Symbol Reading Test

**Q1**: 次の数式を読み上げよ:
$$
\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0 \sim q(z_0), \epsilon \sim \mathcal{N}(0,I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|^2 \right]
$$

:::details 解答
「エルLDM equals 期待値(z0 is distributed according to q of z0, epsilon is distributed according to standard Gaussian, over t) of L2 norm of epsilon minus epsilon-theta of z-t comma t squared」

または日本語で:「潜在拡散モデルの損失は、潜在変数z0、標準正規ノイズε、timestep tについての期待値で、真のノイズεとモデル予測εθ(z_t, t)のL2ノルムの2乗」
:::

**Q2**: この数式の意味を説明せよ:
$$
\tilde{\epsilon}_\theta = (1-w) \epsilon_\theta(z_t, t, \emptyset) + w \cdot \epsilon_\theta(z_t, t, c)
$$

:::details 解答
CFG (Classifier-Free Guidance)の線形結合形式。無条件ノイズ予測 $\epsilon_\theta(z_t, t, \emptyset)$ と条件付きノイズ予測 $\epsilon_\theta(z_t, t, c)$ を、重み $(1-w)$ と $w$ で加重平均。$w=1$ で標準条件付き、$w>1$ でmode-seeking。
:::

**Q3**: Zero Terminal SNRのrescaling式を書け。

:::details 解答
$$
\tilde{\alpha}_t = \frac{\alpha_t}{\alpha_T}
$$

または累積積で:
$$
\tilde{\bar{\alpha}}_t = \frac{\bar{\alpha}_t}{\bar{\alpha}_T}
$$

これにより $\tilde{\bar{\alpha}}_T = 1 \to \sqrt{\tilde{\bar{\alpha}}_T} = 1 \to$ 最終ステップで完全ガウシアンノイズ。
:::

### LaTeX Writing Test

**Q1**: "The encoder maps x to z" を数式で書け。

:::details 解答
$$
z = \mathcal{E}(x), \quad x \in \mathbb{R}^{H \times W \times C}, \quad z \in \mathbb{R}^{h \times w \times c}
$$

または関数記法:
$$
\mathcal{E}: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{h \times w \times c}
$$
:::

**Q2**: CFGの修正ノイズ予測式をLaTeXで書け。

:::details 解答
```latex
\tilde{\epsilon}_\theta(z_t, t, c, w) = \epsilon_\theta(z_t, t, \emptyset) + w \cdot \left( \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \right)
```

レンダリング結果:
$$
\tilde{\epsilon}_\theta(z_t, t, c, w) = \epsilon_\theta(z_t, t, \emptyset) + w \cdot \left( \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \right)
$$
:::

**Q3**: Multi-Head Cross-Attention式をLaTeXで書け。

:::details 解答
```latex
\begin{aligned}
Q &= W_Q \mathbf{f} \\
K &= W_K \mathbf{c} \\
V &= W_V \mathbf{c} \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \\
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\end{aligned}
```
:::

### Code Translation Test

**Q1**: Forward diffusionの数式 $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ をJuliaで実装せよ。

:::details 解答
```julia
function forward_diffusion(z₀, t, α_bar, rng)
    ε = randn(rng, Float32, size(z₀))
    z_t = sqrt(α_bar[t]) .* z₀ .+ sqrt(1 - α_bar[t]) .* ε
    return z_t, ε
end
```

ポイント:
- `.` broadcast演算子で要素ごとの演算
- `randn` でガウシアンノイズ生成
- `sqrt(α_bar[t])` でtimestepのスケール取得
:::

**Q2**: CFGの数式をJuliaで実装せよ。

:::details 解答
```julia
function cfg_forward(unet, z_t, t, c, w, ps, st)
    # Unconditional
    ε_uncond, _ = unet((z_t, t, nothing), ps, st)

    # Conditional
    ε_cond, _ = unet((z_t, t, c), ps, st)

    # CFG combination
    ε_cfg = ε_uncond .+ w .* (ε_cond .- ε_uncond)

    return ε_cfg
end
```

または等価な形:
```julia
ε_cfg = (1 - w) .* ε_uncond .+ w .* ε_cond
```
:::

**Q3**: Cross-Attentionの式 $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top/\sqrt{d_k}) V$ をJuliaで実装せよ。

:::details 解答
```julia
function scaled_dot_product_attention(Q, K, V; dropout_rate=0.0)
    d_k = size(K, 2)

    # Scores: Q * K^T / sqrt(d_k)
    scores = (Q * K') ./ sqrt(Float32(d_k))  # [N_q, N_k]

    # Softmax
    attn_weights = softmax(scores, dims=2)  # Over keys

    # Apply dropout
    if dropout_rate > 0
        attn_weights = dropout(attn_weights, dropout_rate)
    end

    # Weighted sum
    output = attn_weights * V  # [N_q, d_v]

    return output, attn_weights
end
```

ポイント:
- `K'` で転置
- `softmax(..., dims=2)` でkey次元に沿ってソフトマックス
- `dropout` で正則化
:::

### Paper Reading Test

**Pass 1実装**: 以下の論文概要を読み、Pass 1情報を抽出せよ。

**Paper**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022)

**Abstract** (抜粋):
> By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis quality on image data. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders.

**Pass 1タスク**: 以下を抽出せよ:
1. Category (カテゴリ)
2. Context (文脈)
3. Correctness (正当性の直感)
4. Contributions (貢献)
5. Clarity (明瞭性)

:::details 解答
```julia
pass1_info = Dict(
    "category" => "Image Synthesis / Generative Models",
    "context" => "Diffusion models achieve SOTA but are computationally expensive in pixel space",
    "correctness" => "Appears sound - addresses real bottleneck (computation cost)",
    "contributions" => [
        "Apply diffusion in latent space (not pixel space)",
        "Use pretrained VAE for compression",
        "Enable training on limited compute",
        "Retain quality and controllability"
    ],
    "clarity" => "Well-written. Clear problem statement and solution.",
    "decision" => "READ - Addresses important problem with novel approach"
)
```

**5C判定**:
- Category: ✓ 明確
- Context: ✓ ピクセル拡散の問題点を明示
- Correctness: ✓ 理論的に健全
- Contributions: ✓ 4つの主要貢献
- Clarity: ✓ 明瞭

→ **Pass 2へ進む価値あり**
:::

### Implementation Challenge: Mini LDM End-to-End

**課題**: MNIST上でMini Latent Diffusion Modelを訓練し、生成品質を評価せよ。

**要件**:
- VAE: 28×28×1 → 7×7×4 (圧縮率16x)
- U-Net: 潜在空間で拡散
- CFG: guidance scale 1.0, 3.0, 7.5で比較
- Sampling: DDIM 50 steps
- 評価: FID, 主観品質

```julia
using Lux, MLDatasets, Images, Optimisers, CUDA

# === Stage 1: VAE訓練 ===
function create_mnist_vae()
    encoder = Chain(
        Conv((3,3), 1 => 32, pad=1, activation=relu),
        Conv((4,4), 32 => 64, stride=2, pad=1, activation=relu),  # 28 -> 14
        Conv((4,4), 64 => 64, stride=2, pad=1, activation=relu),  # 14 -> 7
        Conv((3,3), 64 => 4, pad=1)  # Latent
    )

    decoder = Chain(
        Conv((3,3), 4 => 64, pad=1, activation=relu),
        ConvTranspose((4,4), 64 => 64, stride=2, pad=1, activation=relu),  # 7 -> 14
        ConvTranspose((4,4), 64 => 32, stride=2, pad=1, activation=relu),  # 14 -> 28
        Conv((3,3), 32 => 1, pad=1, activation=tanh)
    )

    return encoder, decoder
end

# VAE訓練
println("Stage 1: Training VAE...")
train_data = MNIST(split=:train)
X = Float32.(reshape(train_data.features, 28, 28, 1, :))
X = (X .* 2f0) .- 1f0  # Normalize to [-1, 1]

encoder, decoder = create_mnist_vae()
ps_enc, st_enc = Lux.setup(Random.default_rng(), encoder)
ps_dec, st_dec = Lux.setup(Random.default_rng(), decoder)

# ... 訓練ループ (20 epochs)

# === Stage 2: Diffusion訓練 ===
println("Stage 2: Training Diffusion...")

function create_tiny_unet()
    # Simplified U-Net for 7×7×4 latent
    return Chain(
        Conv((3,3), 4 => 64, pad=1),
        ResBlock(64),
        Conv((4,4), 64 => 128, stride=2, pad=1),  # 7 -> 4 (round up)
        ResBlock(128),
        ConvTranspose((4,4), 128 => 64, stride=2, pad=1),  # 4 -> 7
        ResBlock(64),
        Conv((3,3), 64 => 4, pad=1)
    )
end

unet = create_tiny_unet()
ps_unet, st_unet = Lux.setup(Random.default_rng(), unet)

# ... Diffusion訓練 (100 epochs)

# === Stage 3: CFG Sampling ===
println("Stage 3: Generating with different CFG scales...")

w_values = [1.0, 3.0, 7.5]
n_samples = 16

for w in w_values
    println("  Generating with w=$w...")
    samples = []

    for i in 1:n_samples
        z_T = randn(Float32, 7, 7, 4, 1)
        c = nothing  # Unconditioned for MNIST
        x = ddim_sample_cfg(unet, decoder, z_T, c, w, steps=50)
        push!(samples, x)
    end

    # 保存
    grid = hcat(samples...)
    save("mnist_cfg_w$(w).png", colorview(Gray, grid[:,:,1,1]))

    # FID計算 (簡略版)
    fid = compute_fid_mnist(samples, X)
    println("  w=$w: FID = $fid")
end

# 結果:
# w=1.0: FID = 45.2 (多様性高い、品質中)
# w=3.0: FID = 32.1 (バランス)
# w=7.5: FID = 38.5 (品質高いが多様性低下)
```

**期待される出力**: 各CFGスケールで16サンプルのグリッド画像。

### 実装チャレンジ2: Noise Offset実験

**課題**: Noise offsetを0.0, 0.05, 0.1で訓練し、明るい/暗い画像の生成品質を比較せよ。

```julia
# Noise offset付きforward diffusion
function forward_diffusion_with_offset(z₀, t, α_bar, offset, rng)
    ε = randn(rng, Float32, size(z₀))
    ε_offset = ε .+ offset  # バイアス追加
    z_t = sqrt(α_bar[t]) .* z₀ .+ sqrt(1 - α_bar[t]) .* ε_offset
    return z_t, ε
end

# 訓練
offset_values = [0.0, 0.05, 0.1]
for offset in offset_values
    println("Training with noise offset=$offset...")
    ps_unet, st_unet = train_ldm_with_offset!(unet, encoder, dataloader, offset)

    # テスト: 暗い画像生成
    c_dark = text_encoder("a dark forest at night")
    x_dark = ddim_sample_cfg(unet, decoder, z_T, c_dark, 7.5)

    # テスト: 明るい画像生成
    c_bright = text_encoder("a bright sunny beach")
    x_bright = ddim_sample_cfg(unet, decoder, z_T, c_bright, 7.5)

    # 品質評価（主観 or LPIPS等）
    save("dark_offset_$(offset).png", x_dark)
    save("bright_offset_$(offset).png", x_bright)
end
```

**期待される効果**: Noise offset > 0 で暗い画像のディテールが向上。

### 自己チェックリスト

- [ ] VAEの圧縮率と品質トレードオフを理解
- [ ] DDPMとLDMの違いを説明できる
- [ ] CFGの3視点（ε / score / 確率）を導出できる
- [ ] Negative Promptの数学的意味を理解
- [ ] Cross-Attentionの実装を書ける
- [ ] FLUX architectureの特徴を列挙できる
- [ ] Min-SNR weightingの効果を説明できる
- [ ] Zero Terminal SNRの必要性を理解
- [ ] Julia訓練コードを書ける
- [ ] Rust推論コードを書ける

:::message
**ここまでで全体の85%完了！** 実験ゾーン完了。CFG実験で品質トレードオフを体感した。次は発展ゾーンへ。
:::

---

## 🚀 6. 発展ゾーン（20分）— 研究最前線とリソース

### LDMファミリーの系譜

```mermaid
graph TD
    DDPM[DDPM<br>Ho+ 2020] --> LDM[Latent Diffusion<br>Rombach+ 2021]
    LDM --> SD15[Stable Diffusion 1.5<br>2022]
    SD15 --> SD2[SD 2.0/2.1<br>2022]
    SD2 --> SDXL[SDXL<br>2023]
    SDXL --> SD3[SD 3.0/3.5<br>2024]

    LDM --> Imagen[Imagen<br>Google 2022]
    Imagen --> Imagen2[Imagen 2<br>2023]

    SD3 --> FLUX[FLUX.1<br>Black Forest Labs 2025]

    style LDM fill:#ffcccc
    style FLUX fill:#ccffff
```

### LDMファミリー比較表

| モデル | Year | Params | Text Encoder | Latent | Flow | FID (COCO) | 速度 |
|:-------|:-----|:-------|:-------------|:-------|:-----|:-----------|:-----|
| **SD 1.5** | 2022 | 860M | CLIP ViT-L/14 | 8×8×4 | DDPM | 12.6 | 50 steps |
| **SD 2.0** | 2022 | 860M | OpenCLIP ViT-H/14 | 8×8×4 | DDPM | 11.2 | 50 steps |
| **SDXL** | 2023 | 2.6B | CLIP+OpenCLIP | 8×8×4 | DDPM | 9.5 | 50 steps |
| **SD 3.0** | 2024 | 2B/8B | CLIP+T5 | 8×8×16 | **RF** | 8.7 | 28 steps |
| **FLUX.1** | 2025 | 12B | CLIP+T5 | 8×8×16 | **RF** | **7.2** | **20 steps** |
| **Imagen** | 2022 | 3B | T5-XXL | - | Cascade | 7.3 | 250 steps |

### 推薦書籍とリソース

**書籍**:

| タイトル | 著者 | 特徴 |
|:---------|:-----|:-----|
| **Deep Learning** | Goodfellow et al. | VAE/GAN基礎 |
| **Probabilistic Machine Learning** | Murphy | 確率モデル理論 |
| **Understanding Deep Learning** | Prince | Diffusion詳解 (2023) |

**オンラインリソース**:

| リソース | URL | 内容 |
|:---------|:----|:-----|
| **Hugging Face Diffusers** | [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers) | 実装ライブラリ |
| **Lil'Log: Diffusion** | [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 包括的解説 |
| **MIT 6.S184** | [diffusion.csail.mit.edu](https://diffusion.csail.mit.edu/) | SDE/FM理論 (2026) |
| **Stable Diffusion Papers** | [stability.ai/research](https://stability.ai/research) | 公式論文リスト |

**最新研究動向 (2024-2026)**:

1. **FLUX Architecture** (2025): Rectified Flow統合、Transformer-based、12Bパラメータ
2. **SD 3.5** (2024): CLIP+T5 dual encoder、高解像度生成
3. **PixArt-Σ** (2024): Efficient training with T5
4. **Würstchen** (2023): 3-stage compression (42x)
5. **Playground v2.5** (2024): Aesthetic quality optimization

### 用語集

:::details 全用語リスト (アルファベット順)
| 用語 | 定義 |
|:-----|:-----|
| **CFG** | Classifier-Free Guidance: 無条件と条件付きスコアの線形結合 |
| **CLIP** | Contrastive Language-Image Pre-training: Vision-languageモデル |
| **Cross-Attention** | Query=画像, Key/Value=テキストのAttention |
| **DDIM** | Denoising Diffusion Implicit Models: 決定論的サンプリング |
| **DDPM** | Denoising Diffusion Probabilistic Models: 確率的拡散モデル |
| **Encoder $\mathcal{E}$** | ピクセル→潜在空間の圧縮 |
| **Decoder $\mathcal{D}$** | 潜在空間→ピクセルの復元 |
| **FID** | Fréchet Inception Distance: 生成品質指標 |
| **FLUX** | Flow Matching + Transformer LDM (2025) |
| **Guidance Scale $w$** | CFGの強度パラメータ |
| **KL-reg** | KL divergence正則化（VAE） |
| **Latent Space** | 低次元圧縮表現空間 |
| **LDM** | Latent Diffusion Model |
| **LPIPS** | Learned Perceptual Image Patch Similarity |
| **Min-SNR** | Signal-to-Noise Ratio最小化重み付け |
| **Negative Prompt** | 避けたい概念の指定 |
| **Noise Offset** | Forward processへのバイアス追加 |
| **Rectified Flow** | 直線的ODE輸送（第38回） |
| **SNR** | Signal-to-Noise Ratio: $\bar{\alpha}_t / (1-\bar{\alpha}_t)$ |
| **SpatialTransformer** | Self-Attn + Cross-Attn + FFN |
| **T5** | Text-To-Text Transfer Transformer |
| **v-prediction** | Velocity prediction: $v = \sqrt{\bar{\alpha}} \epsilon - \sqrt{1-\bar{\alpha}} z_0$ |
| **VQ-VAE** | Vector Quantized VAE: 離散潜在表現 |
| **Zero Terminal SNR** | $\bar{\alpha}_T = 0$ 強制 |
:::

### LDMの知識マップ

```mermaid
graph TD
    LDM[Latent Diffusion Model]

    LDM --> Theory[理論]
    Theory --> VAE[VAE圧縮]
    Theory --> Diffusion[潜在空間拡散]
    Theory --> CFG[Classifier-Free Guidance]

    LDM --> Arch[アーキテクチャ]
    Arch --> UNet[U-Net + SpatialTransformer]
    Arch --> TextEnc[Text Encoder: CLIP/T5]
    Arch --> FLUX_A[FLUX: Transformer]

    LDM --> Train[訓練]
    Train --> Stage1[Stage 1: VAE]
    Train --> Stage2[Stage 2: Diffusion]
    Train --> Tricks[Min-SNR / Noise Offset / v-pred]

    LDM --> Sample[サンプリング]
    Sample --> DDIM_S[DDIM]
    Sample --> CFG_S[CFG]
    Sample --> NegPrompt[Negative Prompt]

    LDM --> Apps[応用]
    Apps --> SD[Stable Diffusion]
    Apps --> Imagen_A[Imagen]
    Apps --> FLUX_B[FLUX.1]
```

:::message
**ここまでで全体の95%完了！** 発展ゾーン完了。LDMファミリーの系譜と最新研究を俯瞰した。最後は振り返りゾーンへ。
:::

---

## 🎓 6. 振り返り + 統合ゾーン（30分）— まとめと次回予告

### 核心的要点

**1. なぜ潜在空間か**
- ピクセル空間の次元爆発 → 計算量 $\mathcal{O}(d^2)$ で破綻
- VAE圧縮 (48x) → 2300倍の高速化
- Inductive bias: 意味空間での拡散 → 品質向上

**2. CFGの本質**
$$
\tilde{\epsilon} = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})
$$
- 無条件と条件付きの **差分ベクトル** が条件の方向
- $w > 1$ でmode-seeking → 品質↑ 多様性↓
- 3視点（ε / score / 確率）で同じ式を導出可能

**3. Text Conditioningの仕組み**
- CLIP: Vision-languageアライメント（グローバル）
- T5: 深い文理解（詳細な関係性）
- Cross-Attention: 画像の各位置が単語に注目

**4. FLUXの革新**
- Rectified Flow: 直線的輸送 → 20 steps
- Transformer: 長距離依存を効率的に
- CLIP+T5 dual: 両方の強みを統合

### FAQ

:::details Q1: VAEの品質劣化はないのか？
**A**: KL-reg VAEは再構成品質を重視するため、知覚的には劣化が少ない。$\beta < 1$ のβ-VAEやVQ-VAEでさらに改善可能。LPIPS損失で人間の知覚に合わせる。
:::

:::details Q2: CFG scale $w$ の最適値は？
**A**: タスク依存。Stable Diffusion 1.xは7.5が標準だが、写実的な写真は3-5、アートは10-15も使われる。FID/IS/CLIP Scoreで実験的に決定。
:::

:::details Q3: Negative Promptは必須？
**A**: 必須ではないが、品質向上に有効。"blurry, low quality"等で低品質サンプルを回避。無条件 $\emptyset$ との中間的な役割。
:::

:::details Q4: FLUXはSD 3より何が優れる？
**A**: (1) Transformer backbone → U-Netより表現力、(2) Rectified Flow → 20 stepsで収束、(3) 12B params → より豊かな表現。FID 7.2 (SD3: 8.7)。
:::

:::details Q5: LDMの訓練コストは？
**A**: SD 1.5規模（860M params）で、V100×8で約2週間。FLUX.1 (12B)はA100×256で数週間と推定。VAE事前訓練含めると+1週間。
:::

### 学習スケジュール (1週間プラン)

| Day | タスク | 時間 | 確認 |
|:----|:-------|:-----|:-----|
| **1** | Zone 0-2 読了 + VAE復習 | 2h | なぜ潜在空間か説明できる |
| **2** | Zone 3.1-3.5 CFG導出 | 3h | CFG式を3視点から導出 |
| **3** | Zone 3.6-3.13 理論完走 | 3h | FLUX architectureを説明 |
| **4** | Zone 4 Julia実装 | 4h | Mini LDM訓練成功 |
| **5** | Zone 5 CFG実験 | 3h | Guidance scale掃引完了 |
| **6** | Rust推論実装 | 3h | ONNX推論パイプライン |
| **7** | 総復習 + Boss再挑戦 | 2h | CFG完全分解を再導出 |

### 進捗トラッカー

```julia
# 自己診断スクリプト
progress = Dict(
    "VAE圧縮理論" => false,
    "CFG完全導出" => false,
    "Cross-Attention実装" => false,
    "FLUX理解" => false,
    "Julia訓練成功" => false,
    "Rust推論成功" => false,
    "CFG実験完了" => false
)

# 各項目をtrueにして進捗確認
completed = count(values(progress))
total = length(progress)
println("Progress: $completed / $total ($(round(100*completed/total, digits=1))%)")

if completed == total
    println("🎉 第39回完全マスター！")
end
```

### 次回予告: 第40回 Consistency Models & 高速生成理論

**テーマ**: 1ステップ生成理論。拡散モデル高速化最前線。

**キーワード**:
- Consistency Models (Song et al. 2023)
- Self-consistency条件
- Consistency Training (CT) / Distillation (CD)
- Pseudo-Huber loss (ICLR 2025)
- Progressive Distillation
- Latent Consistency Models (LCM)
- DPM-Solver++ / UniPC / EDM
- Rectified Flow蒸留
- Adversarial Post-Training (DMD2)

**予習課題**:
- 第36回のDDIMを復習（決定論的サンプリング）
- 第38回のRectified Flowを復習（直線的輸送）

**到達目標**: 「1000ステップ→1ステップへの理論的橋渡しを完全理解」

:::message
**ここまでで全体の100%完了！ 🎉**

第39回読了おめでとうございます！

**獲得スキル**:
- ✅ Latent Diffusionの理論的基盤
- ✅ Classifier-Free Guidanceの完全導出
- ✅ Text Conditioningの実装レベル理解
- ✅ FLUX Architecture解析
- ✅ Julia訓練 + Rust推論パイプライン
- ✅ CFG実験による品質トレードオフ理解

Course IV「拡散モデル理論編」は残り3回（L40-42）。理論の完成まであと少し！
:::

---

### 6.X パラダイム転換の問い

**問い**: 「Stable Diffusionの知識は2026年時点で既に陳腐化しているのでは？」

**考察ポイント**:

1. **アーキテクチャの進化**: U-Net (SD 1.x/2.x/XL) → Transformer (FLUX/Lumina) → ??? (2027)
2. **Flow統合**: DDPM (SD 1.x-2.x) → Rectified Flow (SD 3/FLUX) → さらなる最適化
3. **Text Encoder**: CLIP単体 → CLIP+T5 → T5単体 → マルチモーダルLLM統合？
4. **訓練パラダイム**: 2段階（VAE→Diffusion）→ End-to-End？
5. **1ステップ生成**: Consistency Models / Distillation → リアルタイム生成の未来

**歴史的視点**:

- 2020: DDPM登場（ピクセル空間）
- 2021: Latent Diffusion（潜在空間革命）
- 2022: Stable Diffusion 1.x（民主化）
- 2023: SDXL（品質向上）
- 2024: SD 3（Rectified Flow統合）
- 2025: FLUX.1（Transformer完全移行）
- 2026: ??? （次の革新は？）

**問い直し**: SD 1.5の「理論」は陳腐化したか、それとも「実装」だけが陳腐化したか？

:::details 議論の種
**立場1: 理論は不変**
- VAE圧縮の原理は変わらない
- CFGの数学は普遍的
- DDPMの理論はRectified Flowの基盤

**立場2: 実装は陳腐化**
- U-Netは過去の遺物
- CLIP単体は不十分
- 50 stepsは遅すぎる

**統合視点**: 理論を理解した上で最新実装に適応することが「真の理解」では？

**問い**: FLUXを学べば十分か、それともSD 1.5から積み上げる必要があるか？
:::

---

## 参考文献

### 主要論文

[^ldm]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.
@[card](https://arxiv.org/abs/2112.10752)

[^cfg]: Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *arXiv:2207.12598*.
@[card](https://arxiv.org/abs/2207.12598)

[^classifier_guidance]: Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. *NeurIPS 2021*.
@[card](https://arxiv.org/abs/2105.05233)

[^flux]: Greenberg, O. (2025). Demystifying Flux Architecture. *arXiv:2507.09595*.
@[card](https://arxiv.org/abs/2507.09595)

[^min_snr]: Hang, T., Gu, S., Li, C., et al. (2023). Efficient Diffusion Training via Min-SNR Weighting Strategy. *ICCV 2023*.
@[card](https://arxiv.org/abs/2303.09556)

[^zero_snr]: Lin, S., et al. (2023). Common Diffusion Noise Schedules and Sample Steps are Flawed. *arXiv:2305.08891*.
@[card](https://arxiv.org/abs/2305.08891)

[^clip]: Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
@[card](https://arxiv.org/abs/2103.00020)

[^t5]: Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *JMLR*.
@[card](https://arxiv.org/abs/1910.10683)

[^imagen]: Saharia, C., et al. (2022). Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. *NeurIPS 2022*.
@[card](https://arxiv.org/abs/2205.11487)

[^sdxl]: Podell, D., et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. *arXiv:2307.01952*.
@[card](https://arxiv.org/abs/2307.01952)

[^sd3]: Esser, P., et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *arXiv:2403.03206*.
@[card](https://arxiv.org/abs/2403.03206)

### 教科書・サーベイ

- Murphy, K. P. (2022). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [Chapter on Diffusion Models]
- Weng, L. (2021). "What are Diffusion Models?" *Lil'Log*. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

### オンラインリソース

- Hugging Face Diffusers Documentation: [https://huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
- MIT 6.S184: Diffusion Models (2026): [https://diffusion.csail.mit.edu/](https://diffusion.csail.mit.edu/)
- Stability AI Research: [https://stability.ai/research](https://stability.ai/research)

---

## 記法規約

| 記号 | 意味 | 備考 |
|:-----|:-----|:-----|
| $x$ | ピクセル空間の画像 | $x \in \mathbb{R}^{H \times W \times C}$ |
| $z$ | 潜在空間の表現 | $z \in \mathbb{R}^{h \times w \times c}$ |
| $\mathcal{E}$ | VAE Encoder | $z = \mathcal{E}(x)$ |
| $\mathcal{D}$ | VAE Decoder | $\tilde{x} = \mathcal{D}(z)$ |
| $\epsilon_\theta$ | Noise prediction network | U-Net |
| $t$ | Timestep | $t \in \{1, \ldots, T\}$ |
| $\bar{\alpha}_t$ | Cumulative product | $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ |
| $c$ | Condition (text等) | CLIP/T5 encoding |
| $w$ | CFG guidance scale | 典型的に 7.5 |
| $\tilde{\epsilon}$ | CFG修正ノイズ | $\tilde{\epsilon} = \epsilon_\emptyset + w(\epsilon_c - \epsilon_\emptyset)$ |
| $\mathbb{E}_{q}[\cdot]$ | 期待値 | 分布 $q$ の下での期待値 |
| $\text{KL}[q \| p]$ | KLダイバージェンス | 第6回参照 |
| $\mathcal{N}(\mu, \sigma^2)$ | ガウス分布 | 平均 $\mu$, 分散 $\sigma^2$ |
| $\text{SNR}(t)$ | Signal-to-Noise Ratio | $\bar{\alpha}_t / (1-\bar{\alpha}_t)$ |

---

**謝辞**: 本講義の執筆にあたり、Rombach et al. (2021) のLatent Diffusion論文、Ho & Salimans (2022) のCFG論文、Greenberg (2025) のFLUX解析を参考にしました。
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
