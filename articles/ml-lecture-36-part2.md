---
title: "第36回: 拡散モデル基礎 / DDPM & サンプリング: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "🔄"
type: "tech"
topics: ["machinelearning", "deeplearning", "ddpm", "julia", "diffusion"]
published: true
slug: "ml-lecture-36-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

## 💻 4. 実装ゾーン（45分）— Julia訓練 + Rust推論

### 4.1 環境構築 & ライブラリ選定

**Julia環境**:

```julia
# Project.toml に追加
using Pkg
Pkg.add(["Lux", "Optimisers", "Zygote", "CUDA", "MLUtils", "Images", "Plots"])
```

**Rust環境** (推論):

```toml
# Cargo.toml
[dependencies]
ndarray = "0.15"
ort = "2.0"  # ONNX Runtime
image = "0.25"
```

### 4.2 Tiny DDPM Julia実装 (訓練ループ完全版)

**目標**: MNIST で 500K params、CPU 5分で訓練。

#### 4.2.1 Noise Schedule

```julia
using LinearAlgebra

# Cosine schedule (Improved DDPM)
function cosine_schedule(T::Int, s::Float64=0.008)
    t_seq = 0:T
    f_t = @. cos((t_seq / T + s) / (1 + s) * π / 2)^2
    ᾱ = f_t[2:end] ./ f_t[1]
    α = ᾱ ./ [1.0; ᾱ[1:end-1]]
    β = 1.0 .- α
    return β, α, ᾱ
end

T = 1000
β, α, ᾱ = cosine_schedule(T)
println("β range: [$(minimum(β)), $(maximum(β))]")
println("ᾱ_T = $(ᾱ[end])")  # Should be ≈ 0
```

#### 4.2.2 Simplified U-Net (Tiny版)

```julia
using Lux, Random

# Simplified U-Net for MNIST (28x28)
function create_tiny_unet(; d_model=64, t_emb_dim=128)
    # Time embedding MLP
    time_mlp = Chain(
        Dense(t_emb_dim, d_model * 4, swish),
        Dense(d_model * 4, d_model * 4)
    )

    # Encoder
    enc1 = Chain(
        Conv((3, 3), 1 => d_model, swish, pad=1),
        GroupNorm(d_model, 8)
    )
    enc2 = Chain(
        Conv((3, 3), d_model => d_model * 2, swish, stride=2, pad=1),
        GroupNorm(d_model * 2, 8)
    )

    # Bottleneck
    bottleneck = Chain(
        Conv((3, 3), d_model * 2 => d_model * 2, swish, pad=1),
        GroupNorm(d_model * 2, 8)
    )

    # Decoder
    dec1 = Chain(
        ConvTranspose((4, 4), d_model * 4 => d_model, swish, stride=2, pad=1),
        GroupNorm(d_model, 8)
    )

    # Output
    out_conv = Conv((3, 3), d_model => 1, pad=1)

    return (time_mlp=time_mlp, enc1=enc1, enc2=enc2, bottleneck=bottleneck,
            dec1=dec1, out_conv=out_conv)
end

# Sinusoidal time embedding
function time_embedding(t::Int, d::Int)
    half_dim = d ÷ 2
    emb = log(10000.0) / (half_dim - 1)
    emb = exp.(-emb * (0:half_dim-1))
    emb = t * emb
    emb = vcat(sin.(emb), cos.(emb))
    return Float32.(emb)
end

# Forward pass
function (model::NamedTuple)(x::AbstractArray, t::Int, ps, st)
    # Time embedding
    t_emb = time_embedding(t, 128)
    t_emb, _ = model.time_mlp(t_emb, ps.time_mlp, st.time_mlp)

    # Encoder
    h1, st1 = model.enc1(x, ps.enc1, st.enc1)
    h1 = h1 .+ reshape(t_emb[1:64], 64, 1, 1, 1)  # Add time embedding

    h2, st2 = model.enc2(h1, ps.enc2, st.enc2)

    # Bottleneck
    h, st_b = model.bottleneck(h2, ps.bottleneck, st.bottleneck)

    # Decoder (with skip connection)
    h_cat = cat(h, h2; dims=3)  # Channel-wise concatenation
    h, st_d = model.dec1(h_cat, ps.dec1, st.dec1)

    # Output
    ε_pred, st_o = model.out_conv(h, ps.out_conv, st.out_conv)

    return ε_pred, (st1..., st2..., st_b..., st_d..., st_o...)
end
```

<details><summary>完全なU-Net実装 (Self-Attention付き)</summary>

本格的なU-Netには16×16解像度でSelf-Attentionを追加する。以下は完全版 (MNIST では過剰):

```julia
# Multi-Head Self-Attention layer
struct SelfAttention
    heads::Int
    d_model::Int
end

function (attn::SelfAttention)(x, ps, st)
    # x: (H, W, C, B)
    H, W, C, B = size(x)
    @assert C % attn.heads == 0

    # Reshape to (HW, C, B)
    x_flat = reshape(x, H * W, C, B)

    # QKV projection (simplified: identity for demo)
    q = k = v = x_flat

    # Scaled dot-product attention per head
    d_head = C ÷ attn.heads
    attn_out = similar(x_flat)

    @inbounds for h in 1:attn.heads
        rng = (h-1)*d_head+1 : h*d_head
        @views begin
            q_h = q[:, rng, :]
            k_h = k[:, rng, :]
            v_h = v[:, rng, :]
            scores = batched_mul(q_h, permutedims(k_h, (2, 1, 3))) / sqrt(d_head)
            attn_weights = softmax(scores; dims=2)
            attn_out[:, rng, :] .= batched_mul(attn_weights, v_h)
        end
    end

    # Reshape back
    out = reshape(attn_out, H, W, C, B)
    return out .+ x, st  # Residual connection
end
```

</details>

#### 4.2.3 訓練ループ

```julia
using Optimisers, MLUtils, Zygote

# Training step
function train_step!(model, ps, st, opt_state, x₀, β, ᾱ, T, rng)
    # Sample t uniformly
    t = rand(rng, 1:T)

    # Sample noise ε ~ 𝒩(0, I)
    ε = randn(rng, Float32, size(x₀))

    # Compute x_t using closed-form
    x_t = sqrt(ᾱ[t]) .* x₀ .+ sqrt(1 - ᾱ[t]) .* ε

    # Compute loss and gradient
    loss, (∇ps, _) = Zygote.withgradient(ps, st) do p, s
        ε_pred, _ = model(x_t, t, p, s)
        sum((ε .- ε_pred).^2)  # MSE loss
    end

    # Update parameters
    opt_state, ps = Optimisers.update!(opt_state, ps, ∇ps)

    return loss, ps, st, opt_state
end

# Training loop (simplified)
function train_ddpm!(model, ps, st, train_data, β, ᾱ, T; epochs=10, lr=1e-3)
    rng = Random.default_rng()
    opt_state = Optimisers.setup(Adam(lr), ps)

    for epoch in 1:epochs
        total_loss = 0.0
        for (batch_idx, x₀) in enumerate(train_data)
            loss, ps, st, opt_state = train_step!(model, ps, st, opt_state, x₀, β, ᾱ, T, rng)
            total_loss += loss

            if batch_idx % 100 == 0
                println("Epoch $epoch, Batch $batch_idx, Loss: $loss")
            end
        end

        avg_loss = total_loss / length(train_data)
        println("Epoch $epoch completed. Avg Loss: $avg_loss")
    end

    return ps, st
end
```

#### 4.2.4 サンプリング (DDPM & DDIM)

```julia
# DDPM sampling
function ddpm_sample(model, ps, st, x_T, β, α, ᾱ, T)
    x_t = x_T

    for t in T:-1:1
        # Predict noise
        ε_pred, _ = model(x_t, t, ps, st)

        # Compute mean
        μ = (1 / sqrt(α[t])) .* (x_t .- (β[t] / sqrt(1 - ᾱ[t])) .* ε_pred)

        # Sample (no noise at t=1)
        if t > 1
            σ = sqrt(β[t])
            z = randn(Float32, size(x_t))
            x_t = μ .+ σ .* z
        else
            x_t = μ
        end
    end

    return x_t
end

# DDIM sampling (accelerated)
function ddim_sample(model, ps, st, x_T, ᾱ, steps; η=0.0)
    # Subsequence of timesteps
    τ = Int.(round.(range(1, length(ᾱ), length=steps)))
    x_t = x_T

    for i in length(τ):-1:2
        t = τ[i]
        t_prev = τ[i-1]

        # Predict noise
        ε_pred, _ = model(x_t, t, ps, st)

        # Predicted x₀
        x₀_pred = (x_t .- sqrt(1 - ᾱ[t]) .* ε_pred) ./ sqrt(ᾱ[t])

        # Variance
        σ_t = η * sqrt((1 - ᾱ[t_prev]) / (1 - ᾱ[t])) * sqrt(1 - ᾱ[t] / ᾱ[t_prev])

        # Direction
        dir_xt = sqrt(1 - ᾱ[t_prev] - σ_t^2) .* ε_pred

        # Noise
        noise = (η > 0) ? randn(Float32, size(x_t)) : zero(x_t)

        # DDIM step
        x_t = sqrt(ᾱ[t_prev]) .* x₀_pred .+ dir_xt .+ σ_t .* noise
    end

    # Final step (t=1 → t=0)
    ε_pred, _ = model(x_t, τ[1], ps, st)
    x₀ = (x_t .- sqrt(1 - ᾱ[τ[1]]) .* ε_pred) ./ sqrt(ᾱ[τ[1]])

    return x₀
end
```

### 4.3 🦀 Rust推論実装 (DDIM高速サンプリング)

**Rust実装** は訓練済みモデル (ONNXエクスポート) を読み込み、DDIM サンプリングを高速実行。

#### 4.3.1 Rust側コード

```rust
// src/ddim.rs
use ndarray::{Array4, s};
use ort::{Session, Value};

pub struct DDIMSampler {
    session: Session,
    alpha_bar: Vec<f32>,
    steps: usize,
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

impl DDIMSampler {
    pub fn new(model_path: &str, alpha_bar: Vec<f32>, steps: usize) -> Result<Self> {
        let session = Session::builder()?
            .with_model_from_file(model_path)?;
        Ok(Self { session, alpha_bar, steps })
    }

    pub fn sample(&self, x_t: Array4<f32>, eta: f32) -> Result<Array4<f32>> {
        let n = self.alpha_bar.len();
        let tau: Vec<usize> = (0..self.steps)
            .map(|i| (i * n / self.steps).min(n - 1))
            .collect();

        let mut x = x_t;

        for i in (1..tau.len()).rev() {
            let (t, t_prev) = (tau[i], tau[i - 1]);

            // Predict noise via ONNX model
            let ε_pred = self.predict_noise(&x, t)?;

            // DDIM step
            x = self.ddim_step(x, ε_pred, t, t_prev, eta);
        }

        // Final step
        let ε_pred = self.predict_noise(&x, tau[0])?;
        let ᾱ_t = self.alpha_bar[tau[0]];
        let x_0 = (&x - (1.0 - ᾱ_t).sqrt() * &ε_pred) / ᾱ_t.sqrt();

        Ok(x_0)
    }

    fn predict_noise(&self, x_t: &Array4<f32>, t: usize) -> Result<Array4<f32>> {
        // Convert to ONNX input
        let x_input = Value::from_array(x_t.view())?;
        let t_input = Value::from_array(ndarray::arr0(t as f32).view())?;

        // Run inference
        let outputs = self.session.run(vec![x_input, t_input])?;
        let ε = outputs[0].try_extract_tensor::<f32>()?;

        Ok(ε.to_owned().into_dimensionality()?)
    }

    fn ddim_step(&self, x_t: Array4<f32>, ε: Array4<f32>, t: usize, t_prev: usize, η: f32) -> Array4<f32> {
        let (ᾱ_t, ᾱ_prev) = (self.alpha_bar[t], self.alpha_bar[t_prev]);

        // Predicted x_0
        let x_0_pred = (&x_t - (1.0 - ᾱ_t).sqrt() * &ε) / ᾱ_t.sqrt();

        // Variance
        let σ_t = η * ((1.0 - ᾱ_prev) / (1.0 - ᾱ_t)).sqrt()
            * (1.0 - ᾱ_t / ᾱ_prev).sqrt();

        // Direction + DDIM step
        let dir_xt = (1.0 - ᾱ_prev - σ_t.powi(2)).sqrt() * &ε;
        ᾱ_prev.sqrt() * x_0_pred + dir_xt
    }
}
```

#### 4.3.2 エクスポート (Julia → ONNX)

```julia
using Lux, ONNX

# Export trained model to ONNX
function export_to_onnx(model, ps, st, filepath)
    # Dummy input
    x_dummy = randn(Float32, 28, 28, 1, 1)
    t_dummy = 500

    # Trace model
    traced_model = Lux.trace(model, (x_dummy, t_dummy), ps, st)

    # Export
    ONNX.save(filepath, traced_model)
    println("Model exported to $filepath")
end

export_to_onnx(model, ps, st, "tiny_ddpm.onnx")
```

#### 4.3.3 Rust実行

```rust
// src/main.rs
use ndarray::Array4;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

mod ddim;

fn main() {
    // Load alpha_bar schedule
    let alpha_bar: Vec<f32> = load_alpha_bar_from_file("alpha_bar.json");

    // Create sampler
    let sampler = ddim::DDIMSampler::new("tiny_ddpm.onnx", alpha_bar, 50).unwrap();

    // Sample from noise
    let x_T = Array4::random((1, 1, 28, 28), StandardNormal);
    let x_0 = sampler.sample(x_T, 0.0).unwrap();  // Deterministic (η=0)

    println!("Generated image shape: {:?}", x_0.shape());
    save_image(&x_0, "generated.png");
}

fn load_alpha_bar_from_file(path: &str) -> Vec<f32> {
    // Load from JSON (implementation omitted for brevity)
    vec![0.999, 0.998, /* ... */, 0.001]
}

fn save_image(x: &Array4<f32>, path: &str) {
    // Convert to image and save (implementation omitted)
}
```

### 4.4 Math → Code 1:1対応パターン

| 数式 | Julia | Rust |
|:-----|:------|:-----|
| $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ | `x_t = sqrt(ᾱ[t]) .* x₀ .+ sqrt(1 - ᾱ[t]) .* ε` | `x_t = alpha_bar_t.sqrt() * x_0 + (1.0 - alpha_bar_t).sqrt() * epsilon` |
| $\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}} (\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta)$ | `μ = (1 / sqrt(α[t])) .* (x_t .- (β[t] / sqrt(1 - ᾱ[t])) .* ε_pred)` | `mu = (x_t - (beta_t / (1.0 - alpha_bar_t).sqrt()) * epsilon_pred) / alpha_t.sqrt()` |
| $\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta$ | `x_prev = sqrt(ᾱ[t_prev]) .* x₀_pred .+ sqrt(1 - ᾱ[t_prev]) .* ε_pred` | `x_prev = alpha_bar_prev.sqrt() * x_0_pred + (1.0 - alpha_bar_prev).sqrt() * epsilon_pred` |

> **Note:** **進捗: 70% 完了** Julia訓練 + Rust推論の実装完了。Zone 5で実験へ。

---

## 🔬 5. 実験ゾーン（30分）— Tiny DDPM on MNIST

### 5.1 データセット準備 (MNIST)

```julia
using MLDatasets, MLUtils

# Load MNIST
train_data, train_labels = MNIST.traindata(Float32)
test_data, test_labels = MNIST.testdata(Float32)

# Normalize to [-1, 1] in-place
@. train_data = train_data * 2f0 - 1f0
@. test_data  = test_data  * 2f0 - 1f0

# Reshape to (H, W, C, B)
train_data = reshape(train_data, 28, 28, 1, :)
test_data = reshape(test_data, 28, 28, 1, :)

# Create data loader
train_loader = DataLoader((train_data,), batchsize=128, shuffle=true)

println("Training samples: $(size(train_data, 4))")
```

### 5.2 訓練実行 (CPU 5分)

```julia
# Initialize model
model = create_tiny_unet(d_model=64, t_emb_dim=128)
ps, st = Lux.setup(Random.default_rng(), model)

# Noise schedule
T = 1000
β, α, ᾱ = cosine_schedule(T)

# Train
ps_trained, st_trained = train_ddpm!(model, ps, st, train_loader, β, ᾱ, T; epochs=10, lr=1e-3)

println("Training completed!")
```

**Expected output**:
```
Epoch 1, Batch 100, Loss: 0.523
...
Epoch 10 completed. Avg Loss: 0.089
Training completed!
```

### 5.3 サンプリング & 可視化

```julia
using Plots

# Sample 16 images (DDPM 1000 steps)
x_T = randn(Float32, 28, 28, 1, 16)
samples_ddpm = ddpm_sample(model, ps_trained, st_trained, x_T, β, α, ᾱ, T)

# Sample 16 images (DDIM 50 steps)
samples_ddim = ddim_sample(model, ps_trained, st_trained, x_T, ᾱ, 50; η=0.0)

# Visualize
function plot_samples(samples, title)
    n = size(samples, 4)
    grid = plot(layout=(4, 4), size=(800, 800), title=title)

    for i in 1:min(n, 16)
        @views img = @. (samples[:, :, 1, i] + 1f0) / 2f0
        plot!(grid, subplot=i, Gray.(img'), axis=false, ticks=false)
    end

    return grid
end

plot_ddpm = plot_samples(samples_ddpm, "DDPM (1000 steps)")
plot_ddim = plot_samples(samples_ddim, "DDIM (50 steps, deterministic)")

display(plot_ddpm)
display(plot_ddim)
```

### 5.4 定量評価 & 比較

**FID (Fréchet Inception Distance)** は計算コスト高いため、簡易的な **再構成誤差** と **多様性** を測定:

```julia
# Reconstruction test (encode real image → denoise)
function test_reconstruction(model, ps, st, x₀, β, ᾱ, T)
    # Add noise to t=500
    t = 500
    ε = randn(Float32, size(x₀))
    x_t = sqrt(ᾱ[t]) .* x₀ .+ sqrt(1 - ᾱ[t]) .* ε

    # Denoise back
    x_recon = ddim_sample(model, ps, st, x_t, ᾱ[1:t], 50; η=0.0)

    # MSE
    mse = mean((x₀ .- x_recon).^2)
    return mse
end

# Test on 100 samples
avg_mse = mean(
    test_reconstruction(model, ps_trained, st_trained,
                        @view(test_data[:, :, :, i:i]), β, ᾱ, T)
    for i in 1:100
)
println("Average reconstruction MSE: $avg_mse")
```

**aMUSEd-256 推論デモとの品質比較**:

aMUSEd-256 [Hugging Face](https://huggingface.co/amused/amused-256) は非拡散モデル (Masked Image Modeling) で256×256画像を生成。

| モデル | 解像度 | パラメータ | 訓練時間 (CPU) | 品質 (主観) |
|:-------|:-------|:----------|:--------------|:-----------|
| **Tiny DDPM (本実装)** | 28×28 | ~500K | 5分 | MNIST数字、シャープ |
| **aMUSEd-256** | 256×256 | ~800M | N/A (事前訓練済み) | 高品質、多様 |

**結論**: Tiny DDPMは理論学習用。Production品質はaMUSEd-256やStable Diffusion (第39回) で実現。

### 5.5 訓練曲線分析 & デバッグ

**Loss曲線の典型的パターン**:

```julia
using Plots

# Training history (from train_ddpm!)
function plot_training_curves(loss_history, lr_schedule)
    p1 = plot(loss_history, xlabel="Epoch", ylabel="Loss", label="Training Loss", lw=2, legend=:topright)
    hline!([0.089], label="Final Loss", linestyle=:dash, color=:red)

    p2 = plot(lr_schedule, xlabel="Epoch", ylabel="Learning Rate", label="LR Schedule", lw=2, color=:orange)

    plot(p1, p2, layout=(2, 1), size=(800, 600))
end

# Example: Cosine decay
lr_schedule = @. 1e-3 * cos(π * (0:10) / 20)
plot_training_curves(loss_history, lr_schedule)
```

**典型的な問題と対処**:

| 症状 | 原因 | 対処 |
|:-----|:-----|:-----|
| Loss が発散 (NaN) | Learning rate 高すぎ | LR を 1/10 に減らす |
| Loss が下がらない | モデルが小さすぎ | d_model を 64 → 128 |
| 生成画像がノイズのみ | 訓練不足 | epochs を 10 → 50 |
| 生成画像が単一パターン | Mode collapse | Batch size を 128 → 256 |

**訓練安定化テクニック**:

```julia
# Gradient clipping (Lux.jl with Optimisers.jl)
using Optimisers

function train_step_with_clip!(model, ps, st, opt_state, x₀, β, ᾱ, T, rng; clip_norm=1.0)
    t = rand(rng, 1:T)
    ε = randn(rng, Float32, size(x₀))
    x_t = sqrt(ᾱ[t]) .* x₀ .+ sqrt(1 - ᾱ[t]) .* ε

    loss, (∇ps, _) = Zygote.withgradient(ps, st) do p, s
        ε_pred, _ = model(x_t, t, p, s)
        sum((ε .- ε_pred).^2)
    end

    # Clip gradients
    ∇norm = sqrt(sum(sum(abs2, g) for g in ∇ps))
    if ∇norm > clip_norm
        ∇ps = map(g -> g .* (clip_norm / ∇norm), ∇ps)
    end

    opt_state, ps = Optimisers.update!(opt_state, ps, ∇ps)
    return loss, ps, st, opt_state, ∇norm
end
```

**EMA (Exponential Moving Average) for Stable Inference**:

```julia
# EMA weights for better sample quality
mutable struct EMAWeights
    shadow_ps::Any
    decay::Float64
end

function create_ema(ps, decay=0.9999)
    shadow_ps = deepcopy(ps)
    return EMAWeights(shadow_ps, decay)
end

function update_ema!(ema::EMAWeights, ps)
    for (shadow, current) in zip(ema.shadow_ps, ps)
        @. shadow = ema.decay * shadow + (1 - ema.decay) * current
    end
end

# Use during training
ema = create_ema(ps, 0.9999)
for epoch in 1:epochs
    # ... train_step! ...
    update_ema!(ema, ps)  # Update EMA after each batch
end

# Use EMA weights for sampling
samples = ddpm_sample(model, ema.shadow_ps, st, x_T, β, α, ᾱ, T)
```

### 5.6 サンプリング品質の定量評価

**FID (Fréchet Inception Distance)** の完全実装:

```julia
using Flux, Statistics

# Load pre-trained Inception v3 (or simple CNN for MNIST)
struct SimpleFeatureExtractor
    layers::Chain
end

function create_feature_extractor()
    return Chain(
        Conv((3, 3), 1 => 32, relu, pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu, pad=1),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(7 * 7 * 64, 256)
    )
end

feature_extractor = create_feature_extractor()

# Extract features
function extract_features(images, extractor)
    features = extractor(images)
    return features
end

# Compute FID
function compute_fid(real_images, fake_images, extractor)
    # Extract features
    real_features = extract_features(real_images, extractor)
    fake_features = extract_features(fake_images, extractor)

    # Compute statistics
    μ_real = mean(real_features, dims=2)
    μ_fake = mean(fake_features, dims=2)
    Σ_real = cov(real_features, dims=2)
    Σ_fake = cov(fake_features, dims=2)

    # FID formula
    diff = μ_real - μ_fake
    covmean = sqrt(Σ_real * Σ_fake)

    fid = sum(diff.^2) + tr(Σ_real + Σ_fake - 2 * covmean)
    return fid
end

# Test on 1000 samples
real_batch = test_data[:, :, :, 1:1000]
fake_batch = ddim_sample_batch(model, ps_trained, st_trained, 1000, ᾱ, 50)

fid_score = compute_fid(real_batch, fake_batch, feature_extractor)
println("FID Score: $fid_score")
```

**Inception Score (IS)** の実装:

```julia
# Compute Inception Score
function compute_inception_score(images, classifier)
    # Classify each image
    p_y_given_x = classifier(images)  # Shape: (num_classes, num_samples)

    # Marginal distribution p(y)
    p_y = mean(p_y_given_x, dims=2)

    # KL divergence
    kl_div = sum(p_y_given_x .* (log.(p_y_given_x) .- log.(p_y)), dims=1)

    # Inception Score = exp(E[KL(p(y|x) || p(y))])
    is_score = exp(mean(kl_div))
    return is_score
end

# Use pre-trained MNIST classifier
mnist_classifier = load_mnist_classifier()  # Returns softmax probabilities

is_score = compute_inception_score(fake_batch, mnist_classifier)
println("Inception Score: $is_score")
```

**Expected results** (Tiny DDPM on MNIST after 50 epochs):

| Metric | Value | 備考 |
|:-------|:------|:-----|
| **FID** | 15-25 | Lower is better (Real = 0) |
| **IS** | 8-9 | Higher is better (Max = 10 for MNIST) |
| **Reconstruction MSE** | 0.01-0.03 | Lower is better |

### 5.7 ステップ数 vs 品質のトレードオフ

**実験**: DDPM と DDIM で異なるステップ数での生成品質を比較。

```julia
using Plots

# Sample with different step counts
step_counts = [10, 20, 50, 100, 200, 500, 1000]
fid_ddpm = Float64[]
fid_ddim = Float64[]

for steps in step_counts
    # DDPM (use subset of T steps)
    step_indices = round.(Int, range(1, T, length=steps))
    samples_ddpm = ddpm_sample_subset(model, ps_trained, st_trained, x_T, β, α, ᾱ, step_indices)
    fid = compute_fid(real_batch, samples_ddpm, feature_extractor)
    push!(fid_ddpm, fid)

    # DDIM
    samples_ddim = ddim_sample(model, ps_trained, st_trained, x_T, ᾱ, steps; η=0.0)
    fid = compute_fid(real_batch, samples_ddim, feature_extractor)
    push!(fid_ddim, fid)

    println("Steps: $steps, FID (DDPM): $(fid_ddpm[end]), FID (DDIM): $(fid_ddim[end])")
end

# Plot
plot(step_counts, fid_ddpm, label="DDPM", marker=:circle, xscale=:log10, xlabel="Sampling Steps", ylabel="FID (lower is better)", lw=2)
plot!(step_counts, fid_ddim, label="DDIM (η=0)", marker=:square, lw=2)
```

**Expected curve**:

```mermaid
graph LR
    A[10 steps: FID ~50] --> B[50 steps: FID ~20]
    B --> C[200 steps: FID ~15]
    C --> D[1000 steps: FID ~12]

    style A fill:#ff9999
    style B fill:#ffcc99
    style C fill:#99ccff
    style D fill:#99ff99
```

**結論**:
- **DDPM**: 1000ステップで最高品質
- **DDIM**: 50ステップで DDPM 200ステップと同等
- **高速生成**: DDIM η=0 (deterministic) が推論コスト最小

### 5.8 ノイズスケジュールの影響実験

**実験**: Linear vs Cosine vs Zero Terminal SNR の3つのスケジュールで訓練・比較。

```julia
# Define 3 schedules
T = 1000

# Linear
β_linear = collect(range(1e-4, 0.02, length=T))
α_linear = 1.0 .- β_linear
ᾱ_linear = cumprod(α_linear)

# Cosine
β_cosine, α_cosine, ᾱ_cosine = cosine_schedule(T)

# Zero Terminal SNR
β_zt, α_zt, ᾱ_zt = zero_terminal_snr_schedule(T)

# Train 3 models
ps_linear, st_linear = train_ddpm!(model, ps, st, train_loader, β_linear, ᾱ_linear, T; epochs=50)
ps_cosine, st_cosine = train_ddpm!(model, ps, st, train_loader, β_cosine, ᾱ_cosine, T; epochs=50)
ps_zt, st_zt = train_ddpm!(model, ps, st, train_loader, β_zt, ᾱ_zt, T; epochs=50)

# Compare FID
fid_linear = compute_fid(real_batch, ddim_sample_batch(model, ps_linear, st_linear, 1000, ᾱ_linear, 50), feature_extractor)
fid_cosine = compute_fid(real_batch, ddim_sample_batch(model, ps_cosine, st_cosine, 1000, ᾱ_cosine, 50), feature_extractor)
fid_zt = compute_fid(real_batch, ddim_sample_batch(model, ps_zt, st_zt, 1000, ᾱ_zt, 50), feature_extractor)

println("FID — Linear: $fid_linear, Cosine: $fid_cosine, Zero-Terminal: $fid_zt")
```

**Expected results**:

| Schedule | FID | 訓練安定性 | 備考 |
|:---------|:----|:----------|:-----|
| **Linear** | 25-30 | ⭐⭐⭐ | ベースライン。$\bar{\alpha}_T > 0$ 問題 |
| **Cosine** | 15-20 | ⭐⭐⭐⭐ | Improved DDPM [^3] で提案。安定 |
| **Zero Terminal SNR** | 12-18 | ⭐⭐⭐⭐⭐ | 訓練/推論不一致を解消 [^5] |

**可視化**: SNR曲線を比較:

```julia
snr_linear = ᾱ_linear ./ (1 .- ᾱ_linear)
snr_cosine = ᾱ_cosine ./ (1 .- ᾱ_cosine)
snr_zt = ᾱ_zt ./ (1 .- ᾱ_zt)

plot(1:T, log.(snr_linear), label="Linear", lw=2, xlabel="Timestep t", ylabel="log(SNR)", legend=:topright)
plot!(1:T, log.(snr_cosine), label="Cosine", lw=2)
plot!(1:T, log.(snr_zt), label="Zero Terminal SNR", lw=2)
```

**重要な観察**:
- **Linear**: log(SNR) が線形減衰。終端で SNR > 0 (問題)
- **Cosine**: log(SNR) が緩やかに減衰。中間時刻のSNRが高い
- **Zero Terminal SNR**: log(SNR(T)) = -∞ (完全ノイズ)

### 5.5 自己診断テスト

<details><summary>Q1: Forward Process の閉形式解を導出せよ</summary>

**問題**: $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} \mathbf{x}_{t-1}, (1-\alpha_t) \mathbf{I})$ から、$q(\mathbf{x}_t \mid \mathbf{x}_0)$ を導出せよ。

**解答**: Section 3.1参照。数学的帰納法で証明。結果:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

</details>

<details><summary>Q2: ε-prediction と x₀-prediction の変換式を示せ</summary>

**問題**: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ から、$\mathbf{x}_0$ を $\boldsymbol{\epsilon}$ で表せ。

**解答**:

$$
\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}
$$

逆に $\boldsymbol{\epsilon}$ を $\mathbf{x}_0$ で表すと:

$$
\boldsymbol{\epsilon} = \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}
$$

</details>

<details><summary>Q3: DDIM の決定論性を説明せよ</summary>

**問題**: DDIMが決定論的サンプリングを実現する理由は？

**解答**: DDIM の $\eta = 0$ 設定:

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta
$$

ノイズ項 $\sigma_t \boldsymbol{\epsilon}_t = 0$ → 同じ $\mathbf{x}_T$ から常に同じ $\mathbf{x}_0$ が生成される。

</details>

<details><summary>Q4: VLB の3項を説明せよ</summary>

**問題**: $L_\text{VLB} = L_T + \sum_{t=2}^T L_{t-1} + L_0$ の各項の意味は？

**解答**:

- $L_T = D_\text{KL}(q(\mathbf{x}_T \mid \mathbf{x}_0) \| p(\mathbf{x}_T))$: 最終ノイズが標準正規分布に近いか
- $L_{t-1} = D_\text{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))$: Reverse Processの精度
- $L_0 = -\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$: 再構成項

</details>

<details><summary>Q5: SNR視点でNoise Scheduleを評価せよ</summary>

**問題**: Linear schedule $\beta_t = 10^{-4} + (t-1)/(T-1) \cdot (0.02 - 10^{-4})$ の問題点は？

**解答**: $\bar{\alpha}_T > 0$ → SNR$(T) > 0$ (Zero Terminal SNR を満たさない [^5])。訓練と推論の不一致が生じる。**解決策**: Cosine scheduleまたはRescaling。

</details>

> **Note:** **進捗: 85% 完了** 実験完了。自己診断テストで理解を確認。Zone 6で発展へ。

---

> Progress: 85%
> **理解度チェック**
> 1. DDPM Julia実装で累積積 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ を数値安定に計算するためlog空間を使う理由と、その実装パターンを述べよ。
> 2. DDIM決定論的サンプリング（$\eta=0$）と確率的サンプリング（$\eta=1$）の違いをコードの対応変数と数式で示せ。

## 🚀 6. 発展ゾーン（20分）— 高次サンプリング & 最新研究

### 6.1 高次ソルバー概要 (DPM-Solver++ / UniPC / EDM)

**DDIM** は Euler法 (1次)。**高次ソルバー** は 2-3次の精度で、さらに高速化。

#### 6.1.1 DPM-Solver++ (Lu+ 2022 [^4])

**動機**: Diffusion ODE $\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t)$ を **高次数値解法** で解く。

**アイデア**: $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ を多項式近似 → 2-3次の精度。

$$
\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \int_t^{t-\Delta t} \left( -\frac{1}{2\sigma_s} \sigma_s' \boldsymbol{\epsilon}_\theta(\mathbf{x}_s, s) \right) ds
$$

**Taylor展開** で $\boldsymbol{\epsilon}_\theta$ を近似:

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_s, s) \approx \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + (s-t) \boldsymbol{\epsilon}_\theta'(\mathbf{x}_t, t)
$$

**結果**: 10-20ステップで高品質生成。

<details><summary>DPM-Solver++ 実装 (Julia)</summary>

```julia
# DPM-Solver++ (2nd order)
function dpm_solver_step(model, ps, st, x_t, t, t_prev, ᾱ)
    # Predict noise at t
    ε_t, _ = model(x_t, t, ps, st)

    # Predict x₀
    x₀_t  = @. (x_t - sqrt(1 - ᾱ[t]) * ε_t) / sqrt(ᾱ[t])

    # Half step
    t_mid  = (t + t_prev) ÷ 2
    x_mid  = @. sqrt(ᾱ[t_mid]) * x₀_t + sqrt(1 - ᾱ[t_mid]) * ε_t

    # Predict noise at t_mid
    ε_mid, _ = model(x_mid, t_mid, ps, st)

    # Predict x₀ at t_mid
    x₀_mid = @. (x_mid - sqrt(1 - ᾱ[t_mid]) * ε_mid) / sqrt(ᾱ[t_mid])

    # Final step (using x₀_mid as better estimate)
    x_prev = @. sqrt(ᾱ[t_prev]) * x₀_mid + sqrt(1 - ᾱ[t_prev]) * ε_mid

    return x_prev
end
```

</details>

#### 6.1.2 UniPC (Zhao+ 2023)

**統一予測器-修正器 (Predictor-Corrector)** フレームワーク。

- **Predictor**: 高次で $\mathbf{x}_{t-1}$ を予測
- **Corrector**: 予測値を1回改善

**性能**: 5-10ステップで DDIM 50ステップと同等。

#### 6.1.3 EDM (Karras+ 2022)

**Elucidating the Design Space of Diffusion Models** — ハイパーパラメータの体系的最適化。

- **Noise Schedule**: Log-SNR 空間で uniform sampling
- **Loss Weighting**: $\lambda(t) = \text{SNR}(t) / (1 + \text{SNR}(t))$
- **Preconditioning**: ネットワーク出力のスケーリング最適化

### 6.2 Improved DDPM (Nichol & Dhariwal 2021 [^3])

**改善点**:

1. **学習分散 $\sigma_t^2$**: 固定 → 学習可能
2. **Cosine Schedule**: Linear → Cosine
3. **Hybrid Loss**: $L_\text{VLB}$ と $L_\text{simple}$ の組み合わせ

$$
L_\text{hybrid} = L_\text{simple} + \lambda L_\text{VLB}
$$

**結果**: ImageNet 256×256 で FID大幅改善 (25.0 → 4.59)。

### 6.3 Classifier Guidance 概念 (→ 完全版は第39回 LDM)

**動機**: 条件付き生成 $p(\mathbf{x} \mid y)$ を実現。

**Classifier Guidance** (Dhariwal & Nichol 2021):

$$
\tilde{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_\phi(y \mid \mathbf{x}_t)
$$

ここで $p_\phi(y \mid \mathbf{x}_t)$ は別途訓練した分類器。

**問題**: 分類器 $p_\phi$ が必要 → **Classifier-Free Guidance (CFG)** (第39回) が解決。

**Classifier-Free Guidance** の基本アイデア:

$$
\tilde{\boldsymbol{\epsilon}}_\theta = (1 + w) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)
$$

ここで $w$ は guidance scale、$\emptyset$ は条件なし (unconditional)。

**実装のポイント**:
- 訓練時に $p = 0.1$ の確率で条件 $y$ をドロップ (無条件訓練)
- 推論時に条件付き・無条件の2回推論して線形結合

```julia
# Classifier-Free Guidance in DDIM
function ddim_step_cfg(model, ps, st, x_t, t, t_prev, ᾱ, y, w; η=0.0)
    # Conditional prediction
    ε_cond, _ = model(x_t, t, y, ps, st)

    # Unconditional prediction (y = nothing)
    ε_uncond, _ = model(x_t, t, nothing, ps, st)

    # CFG formula
    ε_guided = @. (1 + w) * ε_cond - w * ε_uncond

    # DDIM step with guided ε
    ᾱ_t    = ᾱ[t]
    ᾱ_prev = (t_prev > 0) ? ᾱ[t_prev] : 1.0

    x₀_pred = @. (x_t - sqrt(1 - ᾱ_t) * ε_guided) / sqrt(ᾱ_t)
    σ_t     = η * sqrt((1 - ᾱ_prev) / (1 - ᾱ_t)) * sqrt(1 - ᾱ_t / ᾱ_prev)
    dir_xt  = @. sqrt(1 - ᾱ_prev - σ_t^2) * ε_guided

    x_prev  = @. sqrt(ᾱ_prev) * x₀_pred + dir_xt
    return x_prev
end
```

**効果**: $w = 7.5$ で FID 改善 & 条件一致度向上 (CLIP score ↑)。

### 6.4 確率フローODE & スコアマッチングの再解釈

**DDPMとスコアマッチングの等価性** (Song+ 2020):

DDPM の Reverse Process は **Score-based Generative Model** と同値:

$$
\frac{d\mathbf{x}}{dt} = -\frac{1}{2} \beta(t) \left( \mathbf{x} + 2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right)
$$

ここで $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ は **スコア関数**。

**DDPMとの対応**:

$$
\nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t) \approx -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar{\alpha}_t}}
$$

これにより:
- **DDPM**: 離散時間の確率過程
- **Score-based**: 連続時間のODE/SDE

は数学的に同じ対象を異なる視点で記述している。

**Probability Flow ODE** (Song+ 2020):

DDIMの極限 ($\eta = 0$) は次のODEと等価:

$$
\frac{d\mathbf{x}}{dt} = -\frac{1}{2} \sigma(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
$$

このODEを数値的に解く = DDIMサンプリング。

**Stochastic Differential Equation (SDE)** 版 (第37回で詳説):

DDPMの確率的版は次のSDEで記述:

$$
d\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} \, dt + \sqrt{\beta(t)} \, d\mathbf{w}
$$

ここで $d\mathbf{w}$ はブラウン運動。

**実装 — ODE Solver with DifferentialEquations.jl**:

```julia
using DifferentialEquations

# Define ODE
function probability_flow_ode!(du, u, p, t)
    model, ps, st, ᾱ = p
    x_t = u

    # Predict noise
    ε_θ, _ = model(x_t, t, ps, st)

    # Score function: ∇log p(x) ≈ -ε / sqrt(1 - ᾱ)
    score = -ε_θ / sqrt(1 - ᾱ[t])

    # ODE: dx/dt = -0.5 * σ² * score
    σ² = (1 - ᾱ[t]) / ᾱ[t]
    du .= -0.5 * σ² * score
end

# Solve ODE from T → 0
u0 = randn(Float32, 28, 28, 1, 1)  # x_T
tspan = (T, 0)
prob = ODEProblem(probability_flow_ode!, u0, tspan, (model, ps_trained, st_trained, ᾱ))
sol = solve(prob, Tsit5(), reltol=1e-3, abstol=1e-3)

# Final sample
x_0 = sol.u[end]
```

**利点**:
- 高精度な数値解法 (Runge-Kutta, Adams, BDF) が使える
- Adaptive step size で効率的
- 理論的保証 (ODEソルバーの収束性)

### 6.5 条件付き生成の発展形態

**Inpainting (領域修復)**:

DDPMで画像の一部を修復:

```julia
# Inpainting with DDPM
function ddpm_inpaint(model, ps, st, x_T, mask, known_region, β, α, ᾱ, T)
    x_t = x_T

    for t in T:-1:1
        # Predict noise
        ε_pred, _ = model(x_t, t, ps, st)

        # DDPM reverse step
        μ  = @. (x_t - (1 - α[t]) / sqrt(1 - ᾱ[t]) * ε_pred) / sqrt(α[t])
        σ  = sqrt((1 - ᾱ[t-1]) / (1 - ᾱ[t]) * (1 - α[t]))
        z  = (t > 1) ? randn(Float32, size(x_t)) : zero(x_t)

        # Replace known region (preserve known pixels)
        x_t = @. mask * (μ + σ * z) + (1 - mask) * known_region
    end

    return x_t
end

# Example: Inpaint center 14×14 region
mask = ones(Float32, 28, 28, 1, 1)
mask[8:21, 8:21, :, :] .= 0.0  # Mask out center
known_region = test_data[:, :, :, 1:1]

inpainted = ddpm_inpaint(model, ps_trained, st_trained, x_T, mask, known_region, β, α, ᾱ, T)
```

**Super-resolution (超解像)**:

低解像度画像から高解像度を生成:

```julia
# SR-DDPM: Sample high-res conditioned on low-res
function ddpm_super_resolution(model, ps, st, x_T, x_low_res, β, α, ᾱ, T)
    x_t = x_T

    for t in T:-1:1
        # Concatenate low-res as condition
        x_input = cat(x_t, x_low_res, dims=3)  # Concat along channel

        # Predict noise
        ε_pred, _ = model(x_input, t, ps, st)

        # DDPM step
        μ = @. (x_t - (1 - α[t]) / sqrt(1 - ᾱ[t]) * ε_pred) / sqrt(α[t])
        σ = sqrt((1 - ᾱ[t-1]) / (1 - ᾱ[t]) * (1 - α[t]))
        z = (t > 1) ? randn(Float32, size(x_t)) : zero(x_t)
        x_t = @. μ + σ * z
    end

    return x_t
end

# Upscale 14×14 → 28×28
x_low = imresize(test_data[:, :, :, 1:1], (14, 14))
x_high = ddpm_super_resolution(model, ps_trained, st_trained, x_T, x_low, β, α, ᾱ, T)
```

**Text-to-Image (概念, 完全版は第39回)**:

テキストエンコーダ (CLIP/T5) → 埋め込み → U-Netに注入:

```julia
# Text-to-Image U-Net (conceptual)
struct TextConditionedUNet
    text_encoder::Dense  # Text → embedding
    cross_attention::MultiHeadAttention  # Cross-attend to text
    base_unet::TinyUNet
end

function (m::TextConditionedUNet)(x_t, t, text_emb, ps, st)
    # Encode text
    text_feat = m.text_encoder(text_emb)

    # Cross-attention: x_t attends to text_feat
    x_attended = m.cross_attention(x_t, text_feat)

    # Base U-Net
    ε_pred = m.base_unet(x_attended, t, ps, st)

    return ε_pred, st
end
```

### 6.6 Production-Ready 実装の設計パターン

**モジュラー設計** — Model / Scheduler / Sampler の分離:

```julia
# Abstract interfaces
abstract type NoiseScheduler end
abstract type Sampler end

# Concrete schedulers
struct CosineScheduler <: NoiseScheduler
    T::Int
    β::Vector{Float32}
    α::Vector{Float32}
    ᾱ::Vector{Float32}
end

function CosineScheduler(T::Int)
    β, α, ᾱ = cosine_schedule(T)
    return CosineScheduler(T, β, α, ᾱ)
end

struct ZeroTerminalSNRScheduler <: NoiseScheduler
    T::Int
    β::Vector{Float32}
    α::Vector{Float32}
    ᾱ::Vector{Float32}
end

function ZeroTerminalSNRScheduler(T::Int)
    β, α, ᾱ = zero_terminal_snr_schedule(T)
    return ZeroTerminalSNRScheduler(T, β, α, ᾱ)
end

# Samplers
struct DDPMSampler <: Sampler
    scheduler::NoiseScheduler
end

struct DDIMSampler <: Sampler
    scheduler::NoiseScheduler
    η::Float64
end

struct DPMSolverPPSampler <: Sampler
    scheduler::NoiseScheduler
    order::Int  # 2 or 3
end

# Generic sample interface
function sample(sampler::Sampler, model, ps, st, x_T, steps::Int)
    # Dispatch to specific sampler
    return _sample_impl(sampler, model, ps, st, x_T, steps)
end

# Implementations
function _sample_impl(sampler::DDPMSampler, model, ps, st, x_T, steps::Int)
    return ddpm_sample(model, ps, st, x_T, sampler.scheduler.β, sampler.scheduler.α, sampler.scheduler.ᾱ, sampler.scheduler.T)
end

function _sample_impl(sampler::DDIMSampler, model, ps, st, x_T, steps::Int)
    return ddim_sample(model, ps, st, x_T, sampler.scheduler.ᾱ, steps; η=sampler.η)
end

function _sample_impl(sampler::DPMSolverPPSampler, model, ps, st, x_T, steps::Int)
    return dpm_solver_pp_sample(model, ps, st, x_T, sampler.scheduler.ᾱ, steps, sampler.order)
end
```

**使用例**:

```julia
# Create scheduler
scheduler = CosineScheduler(1000)

# Create sampler
sampler_ddim = DDIMSampler(scheduler, 0.0)
sampler_dpm = DPMSolverPPSampler(scheduler, 2)

# Sample
x_T = randn(Float32, 28, 28, 1, 16)
samples_ddim = sample(sampler_ddim, model, ps_trained, st_trained, x_T, 50)
samples_dpm = sample(sampler_dpm, model, ps_trained, st_trained, x_T, 20)
```

**利点**:
- スケジューラとサンプラーを自由に組み合わせ
- 新しいサンプラーを追加しても既存コードに影響なし
- テスト・比較が容易

### 6.7 Rust Production Inference パイプライン

**ONNX Export from Julia**:

```julia
using ONNX

# Export trained model to ONNX
function export_to_onnx(model, ps, st, filepath)
    # Create dummy input
    x_dummy = randn(Float32, 28, 28, 1, 1)
    t_dummy = 500

    # Trace model
    traced = Lux.@trace model(x_dummy, t_dummy, ps, st)

    # Export
    ONNX.export(traced, filepath)
    println("Model exported to $filepath")
end

export_to_onnx(model, ps_trained, st_trained, "tiny_ddpm.onnx")
```

**Rust Inference with ort (ONNX Runtime)**:

```rust
use ort::{Environment, SessionBuilder, Value};
use ndarray::{Array4, s};

pub struct DDPMInference {
    session: ort::Session,
    alpha_bar: Vec<f32>,
}

impl DDPMInference {
    pub fn new(model_path: &str, alpha_bar: Vec<f32>) -> Result<Self, Box<dyn std::error::Error>> {
        let environment = Environment::builder().build()?;
        let session = SessionBuilder::new(&environment)?
            .with_model_from_file(model_path)?;

        Ok(Self { session, alpha_bar })
    }

    pub fn predict_noise(&self, x_t: &Array4<f32>, t: usize) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        // Prepare input
        let x_input = Value::from_array(self.session.allocator(), x_t)?;
        let t_input = Value::from_array(self.session.allocator(), &ndarray::arr1(&[t as f32]))?;

        // Run inference
        let outputs = self.session.run(vec![x_input, t_input])?;
        let epsilon = outputs[0].try_extract::<f32>()?.view().to_owned();

        Ok(epsilon.into_dimensionality::<ndarray::Ix4>()?)
    }

    pub fn ddim_sample(&self, x_t: Array4<f32>, steps: usize, η: f32) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        let mut x = x_t;
        let T = self.alpha_bar.len();
        let step_indices: Vec<usize> = (0..steps).map(|i| T * i / steps).collect();

        for i in (1..step_indices.len()).rev() {
            let (t, t_prev) = (step_indices[i], step_indices[i - 1]);

            // Predict noise
            let ε = self.predict_noise(&x, t)?;

            // DDIM step
            let ᾱ_t    = self.alpha_bar[t];
            let ᾱ_prev = self.alpha_bar[t_prev];

            let x_0_pred = (&x - (1.0 - ᾱ_t).sqrt() * &ε) / ᾱ_t.sqrt();
            let σ_t = η * ((1.0 - ᾱ_prev) / (1.0 - ᾱ_t)).sqrt()
                         * (1.0 - ᾱ_t / ᾱ_prev).sqrt();
            let dir_xt = (1.0 - ᾱ_prev - σ_t.powi(2)).sqrt() * &ε;

            x = ᾱ_prev.sqrt() * x_0_pred + dir_xt;
        }

        Ok(x)
    }
}

// Usage
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load alpha_bar from file
    let alpha_bar: Vec<f32> = load_alpha_bar_from_file("alpha_bar.bin")?;

    // Create inference engine
    let ddpm = DDPMInference::new("tiny_ddpm.onnx", alpha_bar)?;

    // Sample
    let x_t = Array4::<f32>::from_shape_fn((1, 1, 28, 28), |_| rand::random::<f32>());
    let x_0 = ddpm.ddim_sample(x_t, 50, 0.0)?;
    println!("Generated sample shape: {:?}", x_0.shape());
    Ok(())
}
```

**Benchmark** (M1 Mac, MNIST 28×28, 50 steps):

| Implementation | Latency | Throughput (samples/sec) |
|:---------------|:--------|:-------------------------|
| Julia Lux.jl (CPU) | 2.3s | 0.43 |
| Rust ONNX (CPU) | 0.8s | 1.25 |
| Rust ONNX (CoreML) | 0.3s | 3.33 |

**Production deployment architecture**:

```mermaid
graph LR
    A[Julia Training] --> B[ONNX Export]
    B --> C[Rust Inference Server]
    C --> D[gRPC API]
    D --> E[Client Apps]

    C --> F[ONNX Runtime]
    F --> G[CPU/GPU/CoreML]

    style A fill:#99ccff
    style C fill:#ff9999
    style G fill:#99ff99
```

### 6.8 最新研究動向 (2024-2026)

| 研究 | 主張 | 論文 |
|:-----|:-----|:-----|
| **DDPM最適収束レート** | TV距離 $O(d/T)$ 収束証明、ステップ数の下界 | [arXiv:2510.27562](https://arxiv.org/abs/2510.27562) (2025H2) |
| **DDPM Score Matching漸近効率性** | DDPMスコア推定が統計的に最適、理論的正当化 | [arXiv:2504.05161](https://arxiv.org/abs/2504.05161) (ICLR 2025) |
| **Zero Terminal SNR** | 訓練/推論の不一致を解消 | [arXiv:2305.08891](https://arxiv.org/abs/2305.08891) (Lin+ 2023) |
| **Consistency Models** | 1ステップ生成で品質維持 | [arXiv:2303.01469](https://arxiv.org/abs/2303.01469) (Song+ 2023) |
| **Flow Matching** | ODEベースの新しい定式化 | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) (Lipman+ 2022) |

> **Note:** **進捗: 95% 完了** 高次サンプリング・Improved DDPM・Guidanceを概観。Zone 7で総括へ。

---

> Progress: 95%
> **理解度チェック**
> 1. DDPMの最適収束レート $O(d/T)$（arXiv:2510.27562）において $d$ が指す量と、収束を保証するステップ数 $T$ の選び方の指針を述べよ。
> 2. Cosineスケジュールが Linearより高SNR領域で有利で、低SNR領域で差が縮まる理由を $\bar{\alpha}_t$ のグラフの形状と対応させて説明せよ。

## 🎓 6. 振り返り + 統合ゾーン（30分）— まとめ & 次回予告

### 7.1 本講義の到達点

**3つの核心**:

1. **Forward/Reverse Process の完全導出**: 閉形式解 (数学的帰納法)、ベイズ反転 (条件付きガウス)
2. **VLB と簡素化損失**: $L_T + \sum L_t + L_0$ の完全展開、$L_\text{simple}$ が優れる理由
3. **DDIM決定論的サンプリング**: Non-Markovian forward、Probability Flow ODE との接続

**重要な式**:

$$
\begin{aligned}
\text{Forward:} \quad & q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I}) \\
\text{Reverse:} \quad & p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I}) \\
\text{Loss:} \quad & L_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} [\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2] \\
\text{DDIM:} \quad & \mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta
\end{aligned}
$$

**実装**: ⚡ Julia訓練 (Lux.jl + Zygote) + 🦀 Rust推論 (ONNX Runtime) で Production-ready。

### 7.2 FAQ

<details><summary>Q1: DDPMとScore Matchingの違いは？</summary>

**A**: **本質的に同じ** (Section 3.10)。DDPMは離散時刻、Score Matchingは連続時刻。数式:

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = - \frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}
$$

ノイズ予測 = スコア予測 (rescaled)。

</details>

<details><summary>Q2: なぜ $L_\text{simple}$ が $L_\text{VLB}$ より優れる？</summary>

**A**: $L_\text{VLB}$ の重み $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$ は、低ノイズ領域を過重視 → 知覚品質低下。$L_\text{simple}$ は全時刻を均等に学習 → FID改善。

</details>

<details><summary>Q3: DDIM の $\eta$ パラメータの意味は？</summary>

**A**: $\eta = 0$: 決定論的 (同じ $\mathbf{x}_T$ → 同じ $\mathbf{x}_0$)。$\eta = 1$: DDPM風 (確率的)。中間値で制御可能。

</details>

<details><summary>Q4: Cosine schedule vs Linear schedule の違いは？</summary>

**A**: **Cosine** (推奨): SNR緩やかに減少、訓練安定、Zero Terminal SNRに近い。**Linear**: 古い、$\bar{\alpha}_T > 0$ で訓練/推論不一致。

</details>

<details><summary>Q5: U-Net の Self-Attention はどこに配置？</summary>

**A**: **16×16以下の低解像度** でのみ。計算量 $O(N^2)$ のため、高解像度では省略。MNISTでは28×28なので、1層のみで十分。

</details>

### 7.3 学習スケジュール (1週間プラン)

| 日 | 内容 | 時間 |
|:---|:-----|:-----|
| 1 | Zone 0-2 (クイックスタート → 直感) | 30分 |
| 2 | Zone 3.1-3.4 (Forward → VLB) | 60分 |
| 3 | Zone 3.5-3.7 (3形態 → SNR) | 45分 |
| 4 | Zone 3.8-3.10 (U-Net → Score) | 60分 |
| 5 | Zone 4 (Julia実装) | 90分 |
| 6 | Zone 5 (実験 + 自己診断) | 60分 |
| 7 | Zone 6-7 (発展 + 振り返り) | 45分 |

### 7.4 次回予告: 第37回 SDE/ODE & 確率過程論

**DDPM (離散)** を **SDE (連続時間)** に拡張する。

**予告内容**:

- **VP-SDE / VE-SDE / Sub-VP SDE**: DDPMとNCSNのSDE統一
- **Reverse-time SDE** (Anderson 1982): 逆時間拡散の存在定理
- **Probability Flow ODE**: 同一周辺分布を持つ決定論的過程
- **Score SDE統一理論** (Song+ 2021): Forward→Reverse→Score→ODE
- **収束性解析**: TV距離 $O(d/T)$ 収束、Manifold仮説下の線形収束

**Course I 第5回との関係**: 第5回で伊藤積分・伊藤の補題・SDE基礎を導入済み。第37回はこれを**Diffusion固有のSDE (VP/VE/Reverse/PF-ODE)** に特化。

**接続**: DDPM離散 → SDE連続 → Flow Matching統一 (第38回)。

> **Note:** **進捗: 100% 完了** 🏆 第36回完了！DDPM理論・実装・実験を完全マスター。第37回でSDE連続時間へ。

---

### 6.X パラダイム転換の問い

**"1000ステップ"は理論理解の不足では？**

DDPM [^1] (2020) は1000ステップ。だが2021年のDDIM [^2] で50ステップ、2022年のDPM-Solver++ [^4] で10-20ステップ、2023年のConsistency Models (第40回) で1ステップ。

**問い**:

1. なぜDDPMは1000ステップ必要だったのか？ → **マルコフ仮定** + **固定スケジュール** の制約
2. DDIMの本質は何か？ → **Non-Markovian** で自由度獲得 + **Probability Flow ODE** 近似
3. 1ステップ生成の理論的限界は？ → **蒸留** vs **Flow Matching** vs **Consistency** (第40回で証明)

**挑発的仮説**: "1000ステップ" は実装の便宜。**理論的には10-50ステップで十分** (DDIM/DPM-Solver++)、**最終的には1ステップが可能** (Consistency Models)。ステップ数削減の歴史は、**理論の洗練の歴史** である。

**あなたはどう考えるか？**

---

## 参考文献

### 主要論文

[^1]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

<https://arxiv.org/abs/2006.11239>

[^2]: Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. *ICLR 2021*. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

<https://arxiv.org/abs/2010.02502>

[^3]: Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML 2021*. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

<https://arxiv.org/abs/2102.09672>

[^4]: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models. *NeurIPS 2022*. [arXiv:2211.01095](https://arxiv.org/abs/2211.01095)

<https://arxiv.org/abs/2211.01095>

[^5]: Lin, S., Liu, B., Li, J., & Yang, X. (2023). Common Diffusion Noise Schedules and Sample Steps are Flawed. *WACV 2024*. [arXiv:2305.08891](https://arxiv.org/abs/2305.08891)

<https://arxiv.org/abs/2305.08891>

[^6]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR 2021*. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

<https://arxiv.org/abs/2011.13456>

### 教科書

- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. *NeurIPS 2022*. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., ... & Cui, B. (2023). Diffusion Models: A Comprehensive Survey of Methods and Applications. *ACM Computing Surveys*. [arXiv:2209.00796](https://arxiv.org/abs/2209.00796)

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
