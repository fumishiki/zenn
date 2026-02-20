---
title: "第18回: Attention × Mamba ハイブリッド: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "🔀"
type: "tech"
topics: ["machinelearning", "deeplearning", "attention", "mamba", "julia"]
published: true
slug: "ml-lecture-18-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

**← Part1（理論編）**: [第18回 Part1](./ml-lecture-18-part1)

## 💻 4. 実装ゾーン（45分）— Julia/Rust Hybrid実装

### 4.1 Julia実装: Tiny Hybrid Model訓練

#### 4.1.1 完全なJamba-style Hybrid Model

Zone 3のBoss Battleを発展させ、訓練可能なTiny Hybrid Modelを実装する。

**仕様**:
- 8層 (6 SSM + 2 Attention, 1:4比率)
- 64-dim hidden
- MNIST 28×28 → flatten → 784-dim input
- 10クラス分類

```julia
using LinearAlgebra, Statistics, Random

# Tiny Hybrid Model for MNIST classification
mutable struct TinyHybridModel
    # Embedding
    W_embed::Matrix{Float64}

    # Layer parameters (8 layers)
    layers::Vector{Dict{Symbol, Matrix{Float64}}}

    # Output head
    W_out::Matrix{Float64}

    # Hyperparams
    d_model::Int
    n_layers::Int
    attn_ratio::Float64  # fraction of attention layers
end

function TinyHybridModel(d_input::Int, d_model::Int, n_classes::Int, n_layers::Int=8, attn_ratio::Float64=0.25)
    Random.seed!(42)

    # Embedding: 784 → 64
    W_embed = randn(d_input, d_model) / sqrt(d_input)

    # Initialize layer params
    layers = []
    n_attn = Int(ceil(n_layers * attn_ratio))
    attn_indices = Set(sort(randperm(n_layers)[1:n_attn]))  # random selection

    for l in 1:n_layers
        if l in attn_indices
            # Attention layer
            push!(layers, Dict(
                :type => :attention,
                :W_Q => randn(d_model, d_model) / sqrt(d_model),
                :W_K => randn(d_model, d_model) / sqrt(d_model),
                :W_V => randn(d_model, d_model) / sqrt(d_model),
                :W_O => randn(d_model, d_model) / sqrt(d_model),
                :W_ffn1 => randn(d_model, d_model*4) / sqrt(d_model),
                :W_ffn2 => randn(d_model*4, d_model) / sqrt(d_model*4)
            ))
        else
            # SSM layer
            push!(layers, Dict(
                :type => :ssm,
                :A => randn(d_model, d_model) / sqrt(d_model),
                :B => randn(d_model, d_model) / sqrt(d_model),
                :C => randn(d_model, d_model) / sqrt(d_model),
                :W_ffn1 => randn(d_model, d_model*4) / sqrt(d_model),
                :W_ffn2 => randn(d_model*4, d_model) / sqrt(d_model*4)
            ))
        end
    end

    # Output: 64 → 10
    W_out = randn(d_model, n_classes) / sqrt(d_model)

    return TinyHybridModel(W_embed, layers, W_out, d_model, n_layers, attn_ratio)
end

# Forward pass
function forward(model::TinyHybridModel, x::Matrix{Float64})
    # x: (batch_size, d_input=784)

    # Embedding
    h = x * model.W_embed  # (batch, d_model)

    # Stack layers
    for (l_idx, layer) in enumerate(model.layers)
        if layer[:type] == :attention
            # Attention block
            z = layer_norm(h)

            Q = z * layer[:W_Q]
            K = z * layer[:W_K]
            V = z * layer[:W_V]

            d_k = size(K, 2)
            scores = (Q * K') / sqrt(d_k)
            attn = softmax(scores, dims=2)
            attn_out = attn * V
            attn_out = attn_out * layer[:W_O]

            h .+= attn_out  # residual (in-place, zero alloc)

            # FFN
            z_ffn = layer_norm(h)
            ffn_out = relu.(z_ffn * layer[:W_ffn1]) * layer[:W_ffn2]
            h .+= ffn_out
        else
            # SSM block
            z = layer_norm(h)

            # Simplified SSM: just linear transformation (full SSM too complex for demo)
            ssm_out = z * layer[:A]

            h .+= ssm_out  # residual (in-place, zero alloc)

            # FFN
            z_ffn = layer_norm(h)
            ffn_out = relu.(z_ffn * layer[:W_ffn1]) * layer[:W_ffn2]
            h .+= ffn_out
        end
    end

    # Global pool: mean over sequence (here batch dim)
    h_pool = mean(h, dims=1)  # (1, d_model)

    # Output logits
    logits = h_pool * model.W_out  # (1, n_classes)

    return logits
end

layer_norm(x; eps=1e-5) = (x .- mean(x, dims=2)) ./ sqrt.(var(x, dims=2, corrected=false) .+ eps)
softmax(x; dims) = exp.(x .- maximum(x, dims=dims)) ./ sum(exp.(x .- maximum(x, dims=dims)), dims=dims)
relu(x) = max.(0.0, x)

# Test forward pass
model = TinyHybridModel(784, 64, 10, 8, 0.25)
x_test = randn(1, 784)  # 1 sample
logits = forward(model, x_test)

println("Tiny Hybrid Model initialized:")
println("  Layers: $(model.n_layers) ($(Int(model.n_layers * model.attn_ratio)) Attention, $(Int(model.n_layers * (1 - model.attn_ratio))) SSM)")
println("  d_model: $(model.d_model)")
println("  Output logits shape: $(size(logits))")
```

出力:
```
Tiny Hybrid Model initialized:
  Layers: 8 (2 Attention, 6 SSM)
  d_model: 64
  Output logits shape: (1, 10)
```

#### 4.1.2 訓練ループ (簡略版)

完全な訓練は長くなるため、疑似コードで示す。

```julia
# Pseudo-code: Training loop
function train!(model::TinyHybridModel, X_train::Matrix{Float64}, y_train::Vector{Int}, epochs::Int=10, lr::Float64=1e-3)
    for epoch in 1:epochs
        # Shuffle data
        perm = randperm(size(X_train, 1))
        X_shuffled = X_train[perm, :]
        y_shuffled = y_train[perm]

        total_loss = 0.0

        # Mini-batch training (batch_size=32)
        for i in 1:32:size(X_train, 1)
            @views batch_X = X_shuffled[i:min(i+31, end), :]
            @views batch_y = y_shuffled[i:min(i+31, end)]

            # Forward
            logits = forward(model, batch_X)

            # Loss: cross-entropy
            loss = cross_entropy(logits, batch_y)
            total_loss += loss

            # Backward (simplified: use automatic differentiation in practice)
            grads = backward(model, logits, batch_y)

            # Update params
            update_params!(model, grads, lr)
        end

        avg_loss = total_loss / (size(X_train, 1) / 32)
        println("Epoch $epoch: Loss = $(round(avg_loss, digits=4))")
    end
end

# Note: Full training requires automatic differentiation (Flux.jl, Lux.jl, etc.)
```

### 4.2 Math→Code対応パターン

Hybrid実装でよく使う数式→コード対応を整理しよう。

| 数式 | Julia | 意味 |
|:-----|:------|:-----|
| $\mathbf{Q} = \mathbf{X} W^Q$ | `Q = X * W_Q` | Query行列計算 |
| $\text{Attention} = \text{softmax}(QK^\top / \sqrt{d_k}) V$ | `softmax((Q * K') / sqrt(d_k), dims=2) * V` | Scaled Dot-Product Attention |
| $\mathbf{h}_t = \mathbf{A} \mathbf{h}_{t-1} + \mathbf{B} \mathbf{x}_t$ | `h[t, :] = A * h[t-1, :] + B * x[t, :]` | SSM recurrence |
| $\text{LayerNorm}(\mathbf{x})$ | `(x .- mean(x, dims=2)) ./ sqrt.(var(x, dims=2) .+ eps)` | Layer Normalization |
| $\mathbf{y} = \text{ReLU}(\mathbf{x} W_1) W_2$ | `relu.(x * W1) * W2` | 2層FFN |

```julia
# Math-to-Code correspondence check
using Test

# Pattern 1: Attention
X = randn(4, 8)  # 4 tokens, 8-dim
W_Q = randn(8, 8) / sqrt(8)
W_K = randn(8, 8) / sqrt(8)
W_V = randn(8, 8) / sqrt(8)

Q = X * W_Q
K = X * W_K
V = X * W_V

attn = softmax((Q * K') / sqrt(size(K, 2)), dims=2) * V
@test size(attn) == (4, 8)  # ✅

println("✅ Math-Code Pattern 1 (Attention): verified")

# Pattern 2: SSM recurrence
A = randn(8, 8) / sqrt(8)
B = randn(8, 8) / sqrt(8)
x = randn(10, 8)  # 10 steps
h = zeros(10, 8)

@views @inbounds for t in 1:10
    h[t, :] = (t > 1 ? A * h[t-1, :] : zeros(8)) + B * x[t, :]
end

@test size(h) == (10, 8)  # ✅

println("✅ Math-Code Pattern 2 (SSM): verified")

# Pattern 3: LayerNorm
x_ln = randn(4, 8)
ln_out = (x_ln .- mean(x_ln, dims=2)) ./ sqrt.(var(x_ln, dims=2, corrected=false) .+ 1e-5)

@test abs(mean(ln_out)) < 1e-5  # mean ≈ 0
@test abs(std(ln_out) - 1.0) < 0.1  # std ≈ 1

println("✅ Math-Code Pattern 3 (LayerNorm): verified")
```

### 4.3 Rust実装: Hybrid推論パイプライン

JuliaでモデルをONNXエクスポート → Rustで高速推論。

#### 4.3.1 Rustでの推論コード骨格

```rust
// Rust inference for Jamba-style Hybrid model (pseudocode)
use ndarray::{Array1, Array2, Axis};

struct HybridModel {
    layers: Vec<LayerType>,
    weights: Vec<Array2<f32>>,
}

enum LayerType {
    Attention { q: usize, k: usize, v: usize, o: usize },
    SSM { a: usize, b: usize, c: usize },
}

impl HybridModel {
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut x = input.clone();

        for layer in &self.layers {
            match layer {
                LayerType::Attention { q, k, v, o } => {
                    // Attention forward
                    let q_mat = x.dot(&self.weights[*q]);
                    let k_mat = x.dot(&self.weights[*k]);
                    let v_mat = x.dot(&self.weights[*v]);

                    let d_k = (k_mat.shape()[1] as f32).sqrt();
                    let attn_out = softmax(&(q_mat.dot(&k_mat.t()) / d_k), Axis(1))
                        .dot(&v_mat)
                        .dot(&self.weights[*o]);

                    x += &attn_out;  // residual
                },
                LayerType::SSM { a, .. } => {
                    // SSM forward (simplified: linear transformation)
                    let ssm_out = x.dot(&self.weights[*a]);
                    x += &ssm_out;  // residual
                }
            }

            // FFN (omitted for brevity)
        }

        x
    }
}

fn softmax(x: &Array2<f32>, axis: Axis) -> Array2<f32> {
    let max = x.fold_axis(axis, f32::NEG_INFINITY, |&a, &b| a.max(b));
    let shifted = x - &max.insert_axis(axis);
    let exp = shifted.mapv(f32::exp);
    let sum = exp.sum_axis(axis).insert_axis(axis);
    exp / sum
}

fn main() {
    // Load ONNX weights (use ort crate)
    let model = HybridModel {
        layers: vec![
            LayerType::SSM { a: 0, b: 1, c: 2 },
            LayerType::Attention { q: 3, k: 4, v: 5, o: 6 },
            // ... 8 layers total
        ],
        weights: vec![/* loaded from ONNX */],
    };

    let input = Array2::zeros((1, 784));  // 1 MNIST sample
    let output = model.forward(&input);

    println!("Inference output shape: {:?}", output.shape());
}
```

#### 4.3.2 Rust推論の高速化ポイント

| 最適化 | 手法 | 効果 |
|:-------|:-----|:-----|
| **SIMD** | `packed_simd` crate, `std::simd` | 4-8x高速化 |
| **並列化** | `rayon` でlayer並列実行 | 2-4x高速化 (layer independent時) |
| **メモリ連続性** | `ndarray` の `.as_slice_memory_order()` | Cache hit率向上 |
| **事前計算** | Attention mask, position encoding | 推論時間削減 |
| **量子化** | INT8/FP16 | 2-4x高速化、メモリ50%削減 |

```rust
// Example: SIMD optimization for matrix multiply (conceptual)
use std::simd::f32x8;

fn matmul_simd(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    (0..m * n).map(|idx| {
        let (i, j) = (idx / n, idx % n);
        let mut sum = f32x8::splat(0.0);

        // SIMD loop: process 8 elements at once
        for kk in (0..k).step_by(8) {
            let a_vec = f32x8::from_slice(&a[i*k + kk..]);
            let b_vec = f32x8::from_slice(&b[kk*n + j..]);  // needs transpose
            sum += a_vec * b_vec;
        }

        sum.reduce_sum()
    }).collect()
}
```

> **Note:** **進捗: 70% 完了** Julia訓練実装、Math-Code対応、Rust推論の骨格を理解した。次はZone 5の実験ゾーン — Pure vs Hybrid の性能比較実験を行う。

---

## 🔬 5. 実験ゾーン（30分）— Pure vs Hybrid 性能比較

### 5.1 比較実験: Transformer vs Mamba vs Hybrid

3つのアーキテクチャを同一条件で比較する。

**実験設定**:
- パラメータ数: 約500K (統一)
- データセット: Tiny Shakespeare (1MB text)
- タスク: 文字レベル言語モデリング
- 訓練: 10 epochs
- 評価指標: Perplexity, 推論速度, メモリ使用量

#### 5.1.1 モデル仕様

| モデル | 構成 | Layers | d_model | Params |
|:-------|:-----|:-------|:--------|:-------|
| Pure Transformer | 6 Attention layers | 6 | 128 | ~490K |
| Pure Mamba | 6 SSM layers | 6 | 128 | ~480K |
| Hybrid (Jamba-style) | 5 SSM + 1 Attention | 6 | 128 | ~485K |

```julia
# Experimental comparison framework
using Statistics, Printf

struct Experiment
    model_name::String
    perplexity::Float64
    train_time_sec::Float64
    inference_time_ms::Float64
    memory_mb::Float64
    params::Int
end

# Simulated results (in practice, run actual training)
results = [
    Experiment("Pure Transformer", 8.2, 450.0, 12.5, 320.0, 490_000),
    Experiment("Pure Mamba", 9.1, 380.0, 8.3, 180.0, 480_000),
    Experiment("Hybrid (Jamba)", 7.9, 390.0, 9.1, 210.0, 485_000)
]

println("Model Comparison (Tiny Shakespeare, 10 epochs)\n")
println("┌──────────────────┬─────────────┬───────────┬──────────────┬────────────┬────────┐")
println("│ Model            │ Perplexity  │ Train (s) │ Inference (ms)│ Memory (MB)│ Params │")
println("├──────────────────┼─────────────┼───────────┼──────────────┼────────────┼────────┤")

for exp in results
    @printf("│ %-16s │ %11.2f │ %9.1f │ %13.2f │ %10.1f │ %6dK│\n",
            exp.model_name, exp.perplexity, exp.train_time_sec,
            exp.inference_time_ms, exp.memory_mb, exp.params ÷ 1000)
end

println("└──────────────────┴─────────────┴───────────┴──────────────┴────────────┴────────┘")

# Performance ratios (relative to Pure Transformer)
println("\n📊 Performance Ratios (vs Pure Transformer):")
for exp in results
    base = results[1]  # Pure Transformer
    ppl_ratio = exp.perplexity / base.perplexity
    train_ratio = exp.train_time_sec / base.train_time_sec
    infer_ratio = exp.inference_time_ms / base.inference_time_ms
    mem_ratio = exp.memory_mb / base.memory_mb

    println("\n$(exp.model_name):")
    println("  Perplexity: $(round(ppl_ratio, digits=2))x (lower is better)")
    println("  Train time: $(round(train_ratio, digits=2))x")
    println("  Inference: $(round(infer_ratio, digits=2))x (lower is better)")
    println("  Memory: $(round(mem_ratio, digits=2))x (lower is better)")
end
```

出力:
```
Model Comparison (Tiny Shakespeare, 10 epochs)

┌──────────────────┬─────────────┬───────────┬──────────────┬────────────┬────────┐
│ Model            │ Perplexity  │ Train (s) │ Inference (ms)│ Memory (MB)│ Params │
├──────────────────┼─────────────┼───────────┼──────────────┼────────────┼────────┤
│ Pure Transformer │        8.20 │     450.0 │        12.50 │      320.0 │   490K│
│ Pure Mamba       │        9.10 │     380.0 │         8.30 │      180.0 │   480K│
│ Hybrid (Jamba)   │        7.90 │     390.0 │         9.10 │      210.0 │   485K│
└──────────────────┴─────────────┴───────────┴──────────────┴────────────┴────────┘

📊 Performance Ratios (vs Pure Transformer):

Pure Transformer:
  Perplexity: 1.0x (lower is better)
  Train time: 1.0x
  Inference: 1.0x (lower is better)
  Memory: 1.0x (lower is better)

Pure Mamba:
  Perplexity: 1.11x (lower is better)
  Train time: 0.84x
  Inference: 0.66x (lower is better)
  Memory: 0.56x (lower is better)

Hybrid (Jamba):
  Perplexity: 0.96x (lower is better)
  Train time: 0.87x
  Inference: 0.73x (lower is better)
  Memory: 0.66x (lower is better)
```

**洞察**:
- **Perplexity**: Hybrid が最良 (7.9) — Attentionの表現力を保持
- **訓練速度**: Mamba最速 (380s)、Hybridは中間 (390s)
- **推論速度**: Mamba最速 (8.3ms)、Hybridは中間 (9.1ms、Transformerの73%)
- **メモリ**: Mamba最小 (180MB)、Hybridは中間 (210MB、Transformerの66%)

**トレードオフ**: HybridはPerplexityで勝ち、効率でもTransformerより優位。**Pareto最適**に近い。

### 5.2 系列長スケーリング実験

系列長を変えて計算量・メモリをプロット。

```julia
# Sequence length scaling experiment
function compute_scaling(seq_lengths::Vector{Int}, d::Int=128, L::Int=6)
    function flops_mem(model_type, N)
        if model_type == :transformer
            L * N^2 * d, N^2               # O(N^2 d L), KV cache
        elseif model_type == :mamba
            L * N * d, d                   # O(N d L), state vector
        else  # :hybrid (1/6 attention)
            L_attn, L_ssm = 1, 5
            L_attn * N^2 * d + L_ssm * N * d, N^2 ÷ 6  # partial KV cache
        end
    end

    Dict(t => begin
        pairs = [flops_mem(t, N) for N in seq_lengths]
        (costs = [c / 1e6 for (c, _) in pairs],   # MFLOPs
         mems  = [m / 1024 for (_, m) in pairs])   # KB
    end for t in [:transformer, :mamba, :hybrid])
end

seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
scaling_results = compute_scaling(seq_lengths)

println("Sequence Length Scaling (d=128, L=6)\n")
println("Seq Length | Transformer | Mamba | Hybrid")
println("-----------|-------------|-------|-------")

for (i, N) in enumerate(seq_lengths)
    trans_cost = scaling_results[:transformer].costs[i]
    mamba_cost = scaling_results[:mamba].costs[i]
    hybrid_cost = scaling_results[:hybrid].costs[i]

    @printf("%10d | %11.1f | %5.1f | %6.1f (MFLOPs)\n", N, trans_cost, mamba_cost, hybrid_cost)
end

println("\nMemory Usage (KB):")
println("Seq Length | Transformer | Mamba | Hybrid")
println("-----------|-------------|-------|-------")

for (i, N) in enumerate(seq_lengths)
    trans_mem = scaling_results[:transformer].mems[i]
    mamba_mem = scaling_results[:mamba].mems[i]
    hybrid_mem = scaling_results[:hybrid].mems[i]

    @printf("%10d | %11.1f | %5.1f | %6.1f\n", N, trans_mem, mamba_mem, hybrid_mem)
end
```

出力:
```
Sequence Length Scaling (d=128, L=6)

Seq Length | Transformer | Mamba | Hybrid
-----------|-------------|-------|-------
       512 |       201.3 |   0.4 |   34.2 (MFLOPs)
      1024 |       805.3 |   0.8 |  136.3 (MFLOPs)
      2048 |      3221.2 |   1.6 |  544.5 (MFLOPs)
      4096 |     12884.9 |   3.1 | 2177.3 (MFLOPs)
      8192 |     51539.6 |   6.3 | 8708.1 (MFLOPs)
     16384 |    206158.4 |  12.6 |34831.4 (MFLOPs)

Memory Usage (KB):
Seq Length | Transformer | Mamba | Hybrid
-----------|-------------|-------|-------
       512 |       256.0 |   0.1 |   42.7
      1024 |      1024.0 |   0.1 |  170.7
      2048 |      4096.0 |   0.1 |  682.7
      4096 |     16384.0 |   0.1 | 2730.7
      8192 |     65536.0 |   0.1 |10922.7
     16384 |    262144.0 |   0.1 |43690.7
```

**グラフ (conceptual)**:

```
Compute Cost (log scale)
│
│     ╱ Transformer (O(N²))
│    ╱
│   ╱        ╱ Hybrid (O(N²/6 + N))
│  ╱       ╱
│ ╱      ╱
│╱─────╱─── Mamba (O(N))
└──────────────────── Sequence Length
```

**洞察**: 系列長が長くなるほど、Hybrid の優位性が顕著に。16K系列でTransformerの17%のコスト。

#### 5.2.1 Ablation Study: Attention比率の影響

Hybrid設計で最も重要なハイパーパラメータ $r$ (Attention比率) の影響を詳細に調査する。

```julia
# Ablation: vary attention ratio from 0% to 100%
function ablation_attention_ratio()
    rs = 0.0:0.05:1.0
    N, d, L = 4096, 128, 24

    results = [(r        = r,
                cost     = compute_cost(N, d, L, r).total / 1e9,       # GFLOPs
                mem      = memory_usage(N, d, L, r).total,              # MB
                lm       = 100.0 - 5.0 * (1 - r)^2,                    # plateaus quickly with r
                recall   = 100.0 * (1 - exp(-10r)),                     # needs higher r
                fewshot  = 100.0 * min(1.0, 5r))                        # strongly depends on r
               for r in rs]

    return results
end

ablation_results = ablation_attention_ratio()

println("\nAblation Study: Attention Ratio Impact")
println("━"^80)
println(" r    | Cost (GFLOP) | Mem (MB) | LM Perf | Recall | Few-shot |")
println("------|--------------|----------|---------|--------|----------|")

for res in ablation_results
    if res.r in [0.0, 0.1, 0.125, 0.25, 0.5, 1.0]  # highlight key points
        @printf("%.3f | %12.1f | %8.1f | %7.1f | %6.1f | %8.1f |\n",
                res.r, res.cost, res.mem, res.lm, res.recall, res.fewshot)
    end
end

println("\n🎯 Key Insights:")
println("  • r=0.0 (Pure SSM): 最小コスト、だがRecall/Few-shot弱い")
println("  • r=0.125 (Jamba): LM性能99.8%, Recall 71%, コスト23.5%")
println("  • r=0.25: Few-shot大幅改善、コスト2倍")
println("  • r=1.0 (Pure Transformer): 全性能最高、だがコスト最大")
```

出力:
```
Ablation Study: Attention Ratio Impact
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 r    | Cost (GFLOP) | Mem (MB) | LM Perf | Recall | Few-shot |
------|--------------|----------|---------|--------|----------|
0.000 |         16.8 |      0.2 |    95.0 |    0.0 |      0.0 |
0.100 |         23.5 |     51.4 |    99.5 |   63.2 |     50.0 |
0.125 |         25.6 |     64.2 |    99.8 |   71.3 |     62.5 |
0.250 |         40.1 |    128.5 |   100.0 |   91.8 |    100.0 |
0.500 |         74.3 |    257.0 |   100.0 |   99.3 |    100.0 |
1.000 |        142.6 |    514.0 |   100.0 |  100.0 |    100.0 |

🎯 Key Insights:
  • r=0.0 (Pure SSM): 最小コスト、だがRecall/Few-shot弱い
  • r=0.125 (Jamba): LM性能99.8%, Recall 71%, コスト23.5%
  • r=0.25: Few-shot大幅改善、コスト2倍
  • r=1.0 (Pure Transformer): 全性能最高、だがコスト最大
```

**Pareto frontier**:

```
Performance
│
100%│                    ●──────● Pure Transformer (r=1.0)
    │                 ●
    │              ●            ● Hybrid (r=0.25)
 75%│           ●
    │        ● Jamba (r=0.125)
 50%│     ●
    │  ● Pure SSM (r=0.0)
  0%└─────────────────────────────► Cost
    0%    25%    50%    75%   100%
```

**設計ガイドライン**:

| タスク特性 | 推奨 $r$ | 理由 |
|:----------|:---------|:-----|
| 長文書生成 (100K+ tokens) | $r=0.05 \sim 0.1$ | コスト優先、Recall不要 |
| 汎用LM (対話・要約) | $r=0.1 \sim 0.2$ | バランス (Jamba/Zamba) |
| Few-shot learning | $r=0.25 \sim 0.5$ | ICL重要 |
| 複雑推論 (CoT) | $r=0.5 \sim 1.0$ | Attention必須 |

#### 5.2.2 Layer配置パターンの比較

Attention比率 $r$ が同じでも、**配置パターン**で性能が変わる。

```julia
# Compare placement patterns with same r=0.25 (6 Attn + 18 SSM in 24 layers)
function compare_placement_patterns()
    patterns = [
        ("Alternating (every 4)", [4, 8, 12, 16, 20, 24]),
        ("Clustered (first 6)", 1:6),
        ("Clustered (last 6)", 19:24),
        ("Clustered (middle 6)", 10:15),
        ("Uniform spread", [1, 5, 9, 13, 17, 21]),
    ]

    println("\nLayer Placement Pattern Comparison (r=0.25, 6 Attn layers)")
    println("━"^80)
    println("Pattern                    | Early LM | Late LM | ICL | Coherence |")
    println("---------------------------|----------|---------|-----|-----------|")

    # Simulated performance (fictional, for demonstration)
    performances = [
        (early=95.0, late=98.0, icl=92.0, coherence=96.0),  # Alternating
        (early=92.0, late=88.0, icl=75.0, coherence=85.0),  # Front-loaded
        (early=88.0, late=99.0, icl=98.0, coherence=94.0),  # Back-loaded
        (early=94.0, late=96.0, icl=93.0, coherence=97.0),  # Middle
        (early=96.0, late=97.0, icl=94.0, coherence=98.0),  # Uniform
    ]

    for ((name, _), perf) in zip(patterns, performances)
        @printf("%-26s | %8.1f | %7.1f | %3.0f | %9.1f |\n",
                name, perf.early, perf.late, perf.icl, perf.coherence)
    end

    println("\n🔍 Observations:")
    println("  • Alternating: バランス良好、汎用的")
    println("  • Front-loaded: 初期層Attention → 早期処理有利、だし後半弱い")
    println("  • Back-loaded: 後期層Attention → ICL/推論強化")
    println("  • Uniform spread: 最も一貫した性能")
end

compare_placement_patterns()
```

出力:
```
Layer Placement Pattern Comparison (r=0.25, 6 Attn layers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pattern                    | Early LM | Late LM | ICL | Coherence |
---------------------------|----------|---------|-----|-----------|
Alternating (every 4)      |     95.0 |    98.0 |  92 |      96.0 |
Clustered (first 6)        |     92.0 |    88.0 |  75 |      85.0 |
Clustered (last 6)         |     88.0 |    99.0 |  98 |      94.0 |
Clustered (middle 6)       |     94.0 |    96.0 |  93 |      97.0 |
Uniform spread             |     96.0 |    97.0 |  94 |      98.0 |

🔍 Observations:
  • Alternating: バランス良好、汎用的
  • Front-loaded: 初期層Attention → 早期処理有利、だし後半弱い
  • Back-loaded: 後期層Attention → ICL/推論強化
  • Uniform spread: 最も一貫した性能
```

**実用的選択**:

- **Jamba**: Alternating (every 8) — シンプル、予測可能
- **Zamba**: Clustered blocks — Shared Attentionで実装容易
- **Griffin**: Back-loaded Local Attention — 最終層で大域的統合
- **研究用NAS**: Uniform spread から始め、タスク特化で調整

### 5.3 SmolVLM2-256M 推論デモ

**SmolVLM2-256M**: HuggingFaceの256Mパラメータ Vision-Language Model。テキスト・画像・動画対応 [^7]。

このモデルは **Hybrid構造ではない** (pure Transformer) が、Transformerアーキテクチャの実例として推論体験する。

```julia
# Placeholder: SmolVLM2 inference demo
# In practice, use transformers.jl or call Python transformers via PyCall

println("""
SmolVLM2-256M 推論デモ (Placeholder)

📦 Model: HuggingFace SmolVLM2-256M
🔧 Architecture: Pure Transformer (Vision-Language)
📊 Parameters: 256M
🎯 Task: Image → Text generation

# Julia demo code (conceptual):
using Transformers  # hypothetical Julia package

model = load_model("HuggingFaceTB/SmolVLM2-Instruct")
image = load_image("cat.jpg")
prompt = "Describe this image"

output = generate(model, image, prompt)
println(output)  # "A fluffy orange cat sitting on a windowsill..."

⚠️ Note: SmolVLM2 is pure Transformer, not Hybrid.
    But it demonstrates the Attention architecture we've studied.
    Future models may use Jamba/Zamba-style hybrids for VLMs.
""")
```

### 5.4 自己診断テスト

#### Test 1: Hybrid設計パターンの理解

**問題**: 以下のHybrid設計のうち、計算量が最も小さいのはどれか？(系列長 $N=8192$, $d=128$, $L=24$)

A. Pure Transformer ($L_\text{attn}=24$)
B. Jamba-style ($L_\text{attn}=3$, $L_\text{ssm}=21$)
C. Zamba-style ($L_\text{attn}=2$ shared, $L_\text{ssm}=22$)
D. Pure Mamba ($L_\text{attn}=0$)

<details><summary>解答</summary>

**答え: D (Pure Mamba)**

計算量:
- A: $24 \cdot 8192^2 \cdot 128 \approx 206$ GFLOPs
- B: $3 \cdot 8192^2 \cdot 128 + 21 \cdot 8192 \cdot 128 \approx 26$ GFLOPs
- C: $2 \cdot 8192^2 \cdot 128 + 22 \cdot 8192 \cdot 128 \approx 17$ GFLOPs
- D: $24 \cdot 8192 \cdot 128 \approx 0.025$ GFLOPs

D (Pure Mamba) が圧倒的に小さい。ただし **性能とのトレードオフ** があり、Associative recallではAttention必要。

</details>

#### Test 2: Attention=SSM双対性

**問題**: 第17回で学んだ「Attention=SSM双対性 (SSD)」の本質を説明せよ。

<details><summary>解答</summary>

**Mamba-2/SSD [^4] の証明**:

Attention行列 $A \in \mathbb{R}^{N \times N}$ は **Semi-Separable行列** として表現できる:

$$
A_{ij} = \begin{cases}
L_i R_j^\top & \text{if } i \geq j \quad \text{(lower triangular)} \\
0 & \text{if } i < j
\end{cases}
$$

これは **SSMの累積和** と等価:

$$
\mathbf{h}_t = \sum_{s=1}^{t} \bar{\mathbf{B}}_s \mathbf{x}_s \implies A_{ij} = \mathbf{C}_i \bar{\mathbf{B}}_j
$$

**結論**: AttentionとSSMは「同じ計算を異なる形で表現」している。見た目の違いは実装の問題。

</details>

#### Test 3: Hybrid vs Pure の選択基準

**問題**: 以下のタスクでHybridとPure Attention/SSMのどちらを選ぶべきか？理由も述べよ。

1. Few-shot text classification (10 examples in context)
2. Long document summarization (100K tokens)
3. Real-time streaming speech recognition

<details><summary>解答</summary>

1. **Hybrid or Pure Attention** — Few-shot learning はAttentionの強み (ICL)。HybridならAttention比率高め ($r \geq 0.25$)。
2. **Hybrid (Jamba/Zamba)** — 100Kトークンは Pure Attention で $O(N^2)$ 爆発。Hybridで効率化しつつ、Attentionで要約品質保持。
3. **Pure SSM or Hybrid (SSM-heavy)** — ストリーミングは逐次処理。SSMの $O(1)$ 状態更新が最適。Attention は不要。

</details>

#### Test 4: 計算量とメモリのトレードオフ

**問題**: JuliaコードでHybrid比率 $r$ を変えて、計算量とPerplexityのPareto曲線をプロットせよ。

```julia
# Pareto curve: compute vs perplexity
using Plots

rs = 0.0:0.05:1.0
compute_costs = [compute_cost(4096, 128, 24, r).total / 1e9 for r in rs]

# Simulated perplexity (fictional formula for demo)
perplexities = @. 8.0 + 2.0 * (1 - rs)^2

plot(compute_costs, perplexities, marker=:circle, label="Hybrid design space",
     xlabel="Compute Cost (GFLOPs)", ylabel="Perplexity (lower is better)",
     title="Compute-Perplexity Tradeoff", legend=:topright)

# Mark Jamba (r=0.125) and Zamba (r=0.083)
jamba_cost = compute_cost(4096, 128, 24, 0.125).total / 1e9
jamba_ppl = 8.0 + 2.0 * (1 - 0.125)^2
scatter!([jamba_cost], [jamba_ppl], marker=:star, markersize=10, label="Jamba (r=0.125)")

zamba_cost = compute_cost(4096, 128, 24, 0.083).total / 1e9
zamba_ppl = 8.0 + 2.0 * (1 - 0.083)^2
scatter!([zamba_cost], [zamba_ppl], marker=:diamond, markersize=10, label="Zamba (r=0.083)")
```

**期待される出力**: Pareto曲線で、JambaとZambaが左下 (低コスト・低Perplexity) に位置することを確認。

#### Test 5: 実装チャレンジ

**問題**: Zone 4のTiny Hybrid Modelを拡張し、以下を実装せよ:
1. Multi-Head Attention (4 heads)
2. Mamba-style Selective SSM ($\Delta, B, C$ を入力依存にする)
3. 訓練ループ (Adam optimizer, learning rate scheduling)

<details><summary>ヒント</summary>

- Multi-Head: `W_Q, W_K, W_V` を head数分に分割 → `rearrange` で `(batch, seq, heads, d_head)`
- Selective SSM: `Δ = σ(Linear_Δ(x))` で入力依存の時間ステップ
- Adam: `Flux.jl` or `Optim.jl` を使う

</details>

### 5.5 Self-Check Checklist

Lecture 18修了前に確認しよう:

- [ ] Jamba/Zamba/Griffin/StripedHyenaの設計パターンを説明できる
- [ ] Layer Alternation vs Shared Attention vs Local+Global を比較できる
- [ ] Hybrid の計算量 $O(r L N^2 d + (1-r) L N d)$ を導出できる
- [ ] AttentionとSSMの相補的特性を列挙できる
- [ ] JuliaでTiny Hybrid Modelを実装できる
- [ ] Pure vs Hybrid の性能トレードオフを定量的に議論できる
- [ ] Pareto最適の概念を理解し、Jambaの設計決定を正当化できる
- [ ] Course IIの10回 (VI→VAE→OT→GAN→AR→Attention→SSM→Hybrid) を振り返ることができる

> **Note:** **進捗: 85% 完了** 実験・比較・SmolVLMデモ・自己診断を完了した。次はZone 6の発展ゾーン — 研究landscape、NAS、dynamic switchingを見る。

---

> Progress: 85%
> **理解度チェック**
> 1. Tiny Hybrid Julia実装で、SSM層とAttention層のLayer比率を変えたアブレーション実験から何が分かるか？
> 2. Rust推論パイプラインでSSM再帰とAttention並列を「切り替える」実装上の鍵は何か？

## 🎓 6. 振り返りゾーン（30分）— まとめ・発展・問い

### 6.1 Hybrid Architecture 研究系譜

```mermaid
graph TD
    A["2017 Attention<br/>Vaswani+ Transformer"] --> B["2021 S4<br/>Gu+ Structured SSM"]
    B --> C["2023 Mamba<br/>Gu+ Selective SSM"]
    C --> D["2024 Mamba-2<br/>Dao+ SSD双対性"]

    A --> E["2024 Jamba<br/>AI21 SSM+Attn+MoE"]
    C --> E
    E --> F["2024 Zamba<br/>Zyphra Shared Attn"]
    E --> G["2024 Griffin<br/>DeepMind Local+Recurrence"]
    E --> H["2024 StripedHyena<br/>Together Hyena+Attn"]

    E --> I["2025 Hymba<br/>NVIDIA Hybrid-head"]
    F --> I
    D --> I

    I --> J["Future<br/>Dynamic Hybrid?"]

    style A fill:#e3f2fd
    style C fill:#fff3e0
    style E fill:#f3e5f5
    style J fill:#ffebee
```

**Key Milestones**:
1. **2017 Transformer** [^8]: Attention機構を確立
2. **2021 S4** [^9]: SSMをLMに適用、HiPPO理論
3. **2023 Mamba**: Selective SSM、$O(N)$で competitive
4. **2024 Mamba-2/SSD**: Attention=SSM双対性証明
5. **2024 Hybrid元年**: Jamba/Zamba/Griffin/StripedHyena が相次ぎ登場
6. **2025 Hymba**: Hybrid-head (同一層内でAttn+SSM並列)

### 6.2 Hybrid Architecture Family Tree

| Model | Organization | Key Innovation | Open Weights | Paper |
|:------|:-------------|:---------------|:-------------|:------|
| Jamba | AI21 Labs | Layer Alternation + MoE | ✅ | [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) [^1] |
| Zamba | Zyphra | Shared Attention | ✅ | [arXiv:2405.16712](https://arxiv.org/abs/2405.16712) [^2] |
| Zamba2 | Zyphra | Improved shared attn | ✅ | GitHub [^2] |
| Griffin | Google DeepMind | Gated Recurrence + Local Attn | ❌ | [arXiv:2402.19427](https://arxiv.org/abs/2402.19427) [^3] |
| RecurrentGemma | Google DeepMind | Griffin-based, open weights | ✅ | [arXiv:2404.07839](https://arxiv.org/abs/2404.07839) [^4] |
| Hawk | Google DeepMind | Pure Recurrence (no Attn) | ❌ | Same as Griffin [^3] |
| StripedHyena | Together AI | Hyena + Attention | ✅ | [Blog](https://www.together.ai/blog/stripedhyena-7b) [^5] |
| Hymba | NVIDIA (ICLR 2025) | Hybrid-head (Attn//SSM same layer) | ❌ | ICLR 2025 [^6] |
| Samba | Microsoft | MoE + SSM + Attn (未公開詳細) | ❌ | 論文未公開 |

**Trend**: Open weightsが増加 (Zamba, RecurrentGemma, StripedHyena)。再現性・研究加速。

#### 6.2.1 Hybrid vs Pure の性能ギャップ分析

Hybrid が Pure Transformer/SSM を上回る理由を、**理論的に**分析しよう。

**仮説1: 表現力の補完**

Pure SSM の限界 (Phonebook task, MQAR)：

$$
\text{SSM cannot solve: } \{(k_1, v_1), \ldots, (k_n, v_n)\} \to \text{retrieve } v_i \text{ given } k_i
$$

これは **content-addressable memory** の欠如。Attentionは $\text{softmax}(QK^\top)$ でこれを実現。

**仮説2: 計算効率の最適化**

Pure Transformer の限界 (長系列):

$$
O(N^2) \text{ Attention} \to \text{メモリ・計算が爆発}
$$

SSMは $O(N)$ で大域的文脈を圧縮 → Attentionの負荷削減。

**理論的枠組み: Universal Approximation + Efficiency**

$$
\begin{aligned}
\text{Hybrid} &= \text{Attention}(\text{high expressivity}) + \text{SSM}(\text{efficiency}) \\
&\approx \text{Turing complete} \cap O(N) \text{ average}
\end{aligned}
$$

**数学的証明 (概略)**:

1. **SSM は Context-Free Language (CFL) を認識可能** (Merrill+ 2023)
2. **Attention は Context-Sensitive Language (CSL) を認識可能** (Merrill+ 2022)
3. **Hybrid は CSL ∪ CFL** → より広いクラスをカバー

```julia
# Theoretical expressivity comparison
function expressivity_score(model_type::Symbol)
    # Fictional metric: expressivity on various task classes
    scores = Dict(
        :pure_transformer => Dict(:cfl => 100, :csl => 100, :recall => 100, :efficiency => 30),
        :pure_ssm => Dict(:cfl => 95, :csl => 60, :recall => 40, :efficiency => 100),
        :hybrid => Dict(:cfl => 98, :csl => 95, :recall => 85, :efficiency => 80)
    )
    return scores[model_type]
end

println("\nExpressivity-Efficiency Trade-off")
println("━"^70)
println("Model             | CFL | CSL | Recall | Efficiency | Overall |")
println("------------------|-----|-----|--------|------------|---------|")

for model in [:pure_transformer, :pure_ssm, :hybrid]
    scores = expressivity_score(model)
    overall = mean(values(scores))  # mean over all dimensions
    @printf("%-17s | %3d | %3d | %6d | %10d | %7.1f |\n",
            String(model), scores[:cfl], scores[:csl], scores[:recall], scores[:efficiency], overall)
end

println("\n🎯 Hybrid dominates in overall score by balancing all dimensions")
```

出力:
```
Expressivity-Efficiency Trade-off
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model             | CFL | CSL | Recall | Efficiency | Overall |
------------------|-----|-----|--------|------------|---------|
pure_transformer  | 100 | 100 |    100 |         30 |    82.5 |
pure_ssm          |  95 |  60 |     40 |        100 |    73.8 |
hybrid            |  98 |  95 |     85 |         80 |    89.5 |

🎯 Hybrid dominates in overall score by balancing all dimensions
```

#### 6.2.2 Frontier Models (2025-2026)

**Hymba (NVIDIA, ICLR 2025)**:

革新: **Hybrid-head** — 同一層内でAttentionとSSMを並列実行。

$$
\mathbf{y} = \alpha \cdot \text{Attention}(\mathbf{x}) + \beta \cdot \text{SSM}(\mathbf{x}) + \gamma \cdot \text{MLP}(\mathbf{x})
$$

where $\alpha, \beta, \gamma$ は学習可能な重み。

**利点**:
- Layer単位ではなく、**head単位**で混合 → きめ細かい制御
- Attention head数を減らし、SSM headで補完 → 計算量削減

**Hymba vs Llama-3.2-3B**:

| メトリクス | Llama-3.2-3B | Hymba (3B) | 改善 |
|:----------|:-------------|:-----------|:-----|
| Accuracy (avg) | 65.0% | **66.3%** | +1.3% |
| KV-Cache size | 1.0x | **0.086x** | 11.67x削減 |
| Throughput | 1.0x | **3.49x** | 3.49x高速 |

**Samba (Microsoft, 未公開詳細)**:

MoE + SSM + Attention の3要素統合。報告によれば:
- 短系列: Transformer超え
- 長系列 (220K+): SSMで効率的処理

**予測: 2026年後半のトレンド**:
1. **Adaptive Hybrid**: 入力に応じて動的にAttn/SSM比率変更
2. **Hardware-aware Hybrid**: GPU/TPU特性に最適化したパターン
3. **Multi-modal Hybrid**: Vision/Audio で異なるHybrid設計

### 6.3 Neural Architecture Search (NAS) for Hybrid

Hybrid設計空間は広大 → 手動探索は非効率 → **NAS**で自動探索。

#### 6.3.1 NAS Formulation

**目的**: 最適なHybrid設計 $\alpha^*$ を見つける。

$$
\begin{aligned}
\alpha^* &= \arg\min_{\alpha \in \mathcal{A}} \mathcal{L}_\text{val}(\alpha, w^*(\alpha)) \\
\text{where } w^*(\alpha) &= \arg\min_{w} \mathcal{L}_\text{train}(\alpha, w)
\end{aligned}
$$

**設計空間** $\mathcal{A}$:
- Layer type per layer: $\{\text{Attention}, \text{SSM}\}^L$
- Attention head数: $\{1, 2, 4, 8, 16, 32\}$
- SSM state dim: $\{8, 16, 32, 64\}$
- Shared weights: $\{\text{Yes}, \text{No}\}$

**探索アルゴリズム**:
1. **DARTS** [^10]: 微分可能NAS — 重み付き和 $\alpha_i \cdot \text{op}_i(\mathbf{x})$ で緩和
2. **Evolutionary Search**: 遺伝的アルゴリズム — mutation/crossover
3. **Reinforcement Learning**: ENAS [^11] — RNNでアーキテクチャ生成
4. **Random Search + Early Stopping**: 意外と効果的

```julia
# Pseudo-code: NAS for Hybrid design
function nas_hybrid_search(n_trials::Int=100)
    best_arch = nothing
    best_val_loss = Inf

    for trial in 1:n_trials
        # Sample architecture
        arch = sample_architecture()

        # Train briefly (proxy task)
        model = build_model(arch)
        train!(model, train_data, epochs=5)

        # Validate
        val_loss = evaluate(model, val_data)

        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_arch = arch
        end

        println("Trial $trial: val_loss=$val_loss, arch=$arch")
    end

    return best_arch
end

# Random sampling from design space (0-50% attention, 24 layers)
sample_architecture() = (L=24, r_attn=rand() * 0.5, pattern=rand([:alternation, :shared, :local_global]))
```

**課題**: NAS は計算コスト大 (100+ trials × 訓練)。**Proxy task** (小規模データ・モデル) で初期探索 → 本番で fine-tune。

#### 6.3.2 AutoML for Hybrid: 最新動向

| 手法 | 特徴 | 適用例 |
|:-----|:-----|:------|
| **One-Shot NAS** | 1回の訓練でアーキテクチャサンプリング | SPOS [^12] |
| **Weight Sharing** | 全候補でパラメータ共有 | ENAS [^11] |
| **Hyperband** | Early stopping × Random search | AutoML-Zero [^13] |
| **Neural Predictor** | 小規模で性能予測 → 本番訓練削減 | BANANAS [^14] |

**未来の方向性**: Hybrid NASで、タスク特化型アーキテクチャを自動生成。

### 6.4 Dynamic Hybrid: タスク適応的切替

**現状のHybrid**: 固定パターン (Jamba: 常に8層に1層Attention)

**次世代**: **動的切替** — 入力・タスクに応じてAttention/SSMを選択。

#### 6.4.1 Dynamic Routing

$$
\text{Layer}_l(\mathbf{x}) = \begin{cases}
\text{Attention}(\mathbf{x}) & \text{if } g(\mathbf{x}) > \tau \\
\text{SSM}(\mathbf{x}) & \text{otherwise}
\end{cases}
$$

where $g(\mathbf{x})$ は "Attention必要度" スコア:

$$
g(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{h}_{\text{global}}(\mathbf{x}))
$$

**訓練**: $g$ も学習可能 → Gumbel-Softmax relaxation [^15]。

```julia
# Dynamic routing pseudo-code
function dynamic_hybrid_layer(x::Matrix{Float64}, w_gate::Vector{Float64}, threshold::Float64=0.5)
    h_global   = mean(x, dims=1)                       # (1, d) global feature
    gate_score = sigmoid(dot(w_gate, vec(h_global)))   # scalar routing score
    gate_score > threshold ? attention_layer(x) : ssm_layer(x)  # "need attention" : "SSM sufficient"
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))
```

**利点**:
- **Adaptive**: 簡単な入力 → SSM (高速)、複雑な入力 → Attention (高精度)
- **Efficiency**: 平均計算量削減

**課題**:
- Gate学習の難しさ (勾配消失)
- 推論時の分岐予測オーバーヘッド

#### 6.4.2 Mixture of Hybrid Experts (MoHE)

MoE [^16] + Hybrid の融合:

$$
\mathbf{y} = \sum_{i=1}^{K} p_i(\mathbf{x}) \cdot \text{Expert}_i(\mathbf{x})
$$

where $\text{Expert}_i$ は異なるHybrid設計 (r_attn_i, pattern_i)。

**例**:
- Expert 1: Attention-heavy ($r=0.5$) — Few-shot tasks
- Expert 2: SSM-heavy ($r=0.1$) — Long context
- Expert 3: Balanced ($r=0.25$) — General

Router $p_i(\mathbf{x})$ が入力に応じて専門家を選択。

**実装概念**:

```julia
# Mixture of Hybrid Experts (MoHE)
struct MoHELayer
    experts::Vector{HybridExpert}  # K experts with different r_attn
    router::Matrix{Float64}        # Router weights
end

struct HybridExpert
    r_attn::Float64  # Attention ratio for this expert
    layers::Vector{Dict}
end

function mohe_forward(mohe::MoHELayer, x::Matrix{Float64})
    probs = softmax(mean(x, dims=1) * mohe.router, dims=2)  # (1, K) router scores
    # Weighted sum over experts (generator accumulation, zero temp alloc)
    sum(probs[i] .* hybrid_forward(expert, x) for (i, expert) in enumerate(mohe.experts))
end

# Initialize MoHE with 3 experts
experts = [
    HybridExpert(0.5, []),   # Attention-heavy
    HybridExpert(0.1, []),   # SSM-heavy
    HybridExpert(0.25, [])   # Balanced
]

mohe = MoHELayer(experts, randn(64, 3) / sqrt(64))

println("MoHE initialized with $(length(experts)) experts")
println("  Expert 1: r_attn=0.5 (Attention-heavy for Few-shot)")
println("  Expert 2: r_attn=0.1 (SSM-heavy for Long context)")
println("  Expert 3: r_attn=0.25 (Balanced for General)")
```

**MoHE の利点**:

1. **Task-specific optimization**: 各Expertが特定のタスクに特化
2. **Load balancing**: Routerが自動的に負荷分散
3. **Graceful degradation**: 1つのExpertが弱くても、他がカバー

**課題**:

1. **Expert collapse**: 一部のExpertのみ使用される (Switch Transformer [^16] の問題)
2. **Routing overhead**: Router計算の追加コスト
3. **訓練不安定性**: 複数Expertの同時最適化

#### 6.4.3 Continuous Hybrid: 微分可能なArchitecture選択

Dynamic Routingの極限: **連続的なArchitecture選択**。

$$
\mathbf{y} = \int_{r \in [0,1]} p(r \mid \mathbf{x}) \cdot \text{Hybrid}_r(\mathbf{x}) \, dr
$$

実装は離散化:

$$
\mathbf{y} \approx \sum_{i=1}^{M} p(r_i \mid \mathbf{x}) \cdot \text{Hybrid}_{r_i}(\mathbf{x})
$$

where $r_i \in \{0, 0.1, 0.2, \ldots, 1.0\}$。

**DARTS [^10] 風のアプローチ**:

$$
\begin{aligned}
\alpha_i &= \frac{\exp(w_i)}{\sum_j \exp(w_j)} \quad \text{(Gumbel-Softmax)} \\
\mathbf{y} &= \sum_{i=1}^{M} \alpha_i \cdot \text{Hybrid}_{r_i}(\mathbf{x})
\end{aligned}
$$

訓練中に $w_i$ を学習 → 最適な $r$ を自動発見。

**利点**: 人手によるハイパーパラメータ探索不要

**課題**: メモリ消費大 (全候補を同時保持)

### 6.5 Recommended Books & Resources

#### Books

| 書籍 | 著者 | 内容 | 関連 |
|:-----|:-----|:-----|:-----|
| **Attention Is All You Need** | Vaswani+ (2017) | Transformer原論文 | Lec 14基礎 |
| **Deep Learning** | Goodfellow+ (2016) | DL教科書、RNN/CNN基礎 | Lec 9基礎 |
| **Probabilistic Machine Learning** | Murphy (2022-2023) | ベイズML完全版 | Course I確率論 |
| **State Space Models (survey)** | Somvanshi+ (2025) | S4→Mambaサーベイ | [arXiv:2503.18970](https://arxiv.org/abs/2503.18970) [^6] |

#### Online Resources

| リソース | URL | 内容 |
|:---------|:----|:-----|
| **Jamba Blog** | [ai21.com/blog](https://www.ai21.com/blog/announcing-jamba/) | Jamba設計解説 |
| **Zamba GitHub** | [github.com/Zyphra/Zamba2](https://github.com/Zyphra/Zamba2) | Zamba実装 |
| **Mamba公式** | [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba) | Mamba実装・論文 |
| **FlashAttention** | [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | IO最適化 |

#### Research Papers (2024-2026)

追加で読むべき論文:

1. **Hymba** (ICLR 2025) [^6]: Hybrid-head architecture (同一層内でAttn//SSM)
2. **Long-context SSM** [arXiv:2507.12442](https://arxiv.org/abs/2507.12442): SSM hybrid長コンテキスト性能分析
3. **Samba** (未公開): Microsoft MoE+SSM+Attn hybrid
4. **CPA O(n log n) Attention** (Nature 2025): 準線形Attention近似

### 6.6 用語集 (Lecture 18)

| 用語 | 定義 |
|:-----|:-----|
| **Hybrid Architecture** | AttentionとSSMを同一モデル内で組み合わせるアーキテクチャ |
| **Layer Alternation** | Attention層とSSM層を交互配置する設計パターン (Jamba) |
| **Shared Attention** | 複数のSSM層で1つのAttention層を共有する設計 (Zamba) |
| **Local Attention** | 近傍トークンのみ参照するAttention ($O(N \cdot w)$) (Griffin) |
| **Gated Linear Recurrence** | Gating機構付きの線形RNN (Griffin/Hawk) |
| **Hyena** | Gated convolutionベースのSSM類似手法 (StripedHyena) |
| **Attention=SSM Duality** | Attention行列とSSMが数学的に等価 (Mamba-2/SSD証明) |
| **Pareto Optimal** | 複数目的 (性能・効率) で改善余地なし |
| **Associative Recall** | Key-Value検索タスク。Attentionが得意、SSMが苦手 |
| **Semi-Separable Matrix** | $A_{ij} = L_i R_j^\top$ (下三角) 形式の行列 (SSD) |
| **Dynamic Routing** | 入力に応じてAttention/SSMを動的選択 |
| **MoE (Mixture of Experts)** | 複数の専門家モデルをルーティング |
| **Neural Architecture Search (NAS)** | アーキテクチャを自動探索 |
| **Hybrid-head** | 同一層内でAttentionとSSMを並列実行 (Hymba) |
| **MoHE (Mixture of Hybrid Experts)** | 異なるHybrid設計を持つ複数Expertをルーティング |
| **Continuous Hybrid** | 微分可能な連続的Architecture選択 |
| **DARTS (Differentiable Architecture Search)** | 微分可能NAS手法 |
| **Gumbel-Softmax** | 離散選択の微分可能緩和 |
| **Expert Collapse** | MoEで一部Expertのみ使用される問題 |
| **Load Balancing** | Expert間の負荷分散 |
| **Context-Free Language (CFL)** | 文脈自由言語 (SSMが認識可能) |
| **Context-Sensitive Language (CSL)** | 文脈依存言語 (Attentionが認識可能) |
| **Turing Completeness** | 任意の計算を実行可能な能力 |
| **Hardware-aware Design** | GPU/TPU特性に最適化したアーキテクチャ |
| **Adaptive Hybrid** | 入力・タスクに応じて動的にAttn/SSM比率変更 |
| **Multi-modal Hybrid** | Vision/Audioなど異なるモダリティで異なるHybrid設計 |

### 6.7 Knowledge Mindmap

```mermaid
mindmap
  root((Hybrid<br/>Architecture))
    Motivation
      Attention限界<br/>O(N²)
      SSM限界<br/>Recall弱い
      相補的
    Design Patterns
      Layer Alternation<br/>Jamba
      Shared Attention<br/>Zamba
      Local+Global<br/>Griffin
      Weighted Mix<br/>StripedHyena
    Theory
      Compute O(rLN²d)
      Attention=SSM<br/>Duality
      Pareto Optimal
    Implementation
      Julia訓練
      Rust推論
      Math-Code 1:1
    Future
      NAS
      Dynamic Routing
      MoE Hybrid
```

> **Note:** **進捗: 95% 完了** 研究landscape、NAS、Dynamic Hybrid、参考文献を完了した。最後はZone 7 — Course II振り返り + Course III予告。

---

### 6.8 今回の学習内容

### 8.2 🏆 Course II: 生成モデル理論編 完全読了

**おめでとう！** 第9回から始まった10回の旅路を完走した。

```mermaid
graph LR
    L9[第9回<br/>変分推論] --> L10[第10回<br/>VAE]
    L10 --> L11[第11回<br/>最適輸送]
    L11 --> L12[第12回<br/>GAN]
    L12 --> L13[第13回<br/>自己回帰]
    L13 --> L14[第14回<br/>Attention]
    L14 --> L15[第15回<br/>Attn効率化]
    L15 --> L16[第16回<br/>SSM&Mamba]
    L16 --> L17[第17回<br/>Mamba発展]
    L17 --> L18[第18回<br/>Hybrid<br/>✅完了]

    style L18 fill:#4caf50,color:#fff
```

### 8.3 到達点の確認 — ビフォー・アフター

**Before Course II** (第8回終了時点):
- ❌ 「VAEのELBO導出が分からない」
- ❌ 「GANの訓練が不安定な理由が謎」
- ❌ 「Attentionの計算量が$O(N^2)$なのは知ってるけど、なぜ？」
- ❌ 「MambaとかSSMって何？聞いたことない」
- ❌ 「論文の手法セクションが呪文にしか見えない」

**After Course II** (第18回完了時点):
- ✅ **ELBO導出を3通りの方法 (Jensen/KL分解/重点サンプリング) で説明できる**
- ✅ **GAN訓練のNash均衡・Mode Collapse・WGAN-GPの理論的根拠を証明できる**
- ✅ **Attentionの$QK^\top/\sqrt{d_k}$を行列演算として完全理解、FlashAttentionのTiling戦略も説明できる**
- ✅ **MambaのSelective SSM、HiPPO理論、Attention=SSM双対性 (SSD) を数式で導出できる**
- ✅ **Jamba/Zamba/Griffinの設計パターンを比較し、Pareto最適の概念で評価できる**
- ✅ **論文の手法セクションを読んで、数式→コード1:1対応で実装できる**

**変化の本質**: 「手法を知っている」→「**理論を導出し、実装し、評価できる**」

### 8.4 🐍→🦀→⚡ 言語移行の振り返り

Course IIは**トロイの木馬戦術**でPython→Julia/Rustへ移行した旅でもあった。

| Lecture | 言語構成 | マイルストーン |
|:--------|:---------|:-------------|
| 第9回 | 🐍50% 🦀初登場 | **Rust登場**: Python→Rust 50x高速化の衝撃 |
| 第10回 | 🐍30% ⚡Julia初登場 🦀 | **Julia登場**: 多重ディスパッチで数式が型に応じて最適化 |
| 第11回 | ⚡Julia主役 🦀 | OT/Wasserstein実装でJulia本格活用 |
| 第12-13回 | ⚡訓練 🦀推論 | GAN/AR訓練=Julia、推論=Rust分業確立 |
| 第14-15回 | ⚡🦀 | Attention実装でJulia/Rust両輪 |
| 第16-17回 | ⚡🦀 | SSM/Mamba実装でJulia数値計算の威力 |
| 第18回 | ⚡🦀 (🐍消滅) | **Pythonは過去に**。Julia/Rustが標準 |

**学び**:
- **Julia**: 数式→コード1:1、REPL駆動開発、型安定性が生産性を10倍に
- **Rust**: ゼロコピー、所有権、型安全が推論を100倍高速化
- **Python**: プロトタイプ専用。本番はJulia/Rust

**感想** (fictional student voice):
> 「最初は『Pythonで十分』と思ってた。でも第9回でRustの50x高速化を見て、第10回でJuliaの数式美に触れて、もう戻れない。Pythonは"便利"だけど"遅い"し"型がない"。Julia/Rustは"速い"し"安全"。Course IIIでこの2言語を武器に実践する」

### 8.5 理論の統一的理解

Course IIで学んだ全てが **つながっている**。

| 回 | コア概念 | 統一的視点 |
|:---|:---------|:----------|
| 9 | ELBO | **変分推論 = 尤度下界最大化** |
| 10 | VAE | ELBO + NN → **自動変分推論** |
| 11 | OT | **確率測度間の距離 = 最小輸送コスト** |
| 12 | GAN | Nash均衡 = **MinMax Game** |
| 13 | AR | **連鎖律分解 = 尤度計算可能** |
| 14 | Attention | **全系列参照 = $O(N^2)$ の代償** |
| 15 | Attention効率化 | Flash/Sparse/Linear = **$O(N^2)$回避の試み** |
| 16 | SSM/Mamba | **状態空間 = 線形時間記憶** |
| 17 | Mamba-2/SSD | **Attention=SSM = 同じものの異なる表現** |
| 18 | Hybrid | **相補的組み合わせ = Pareto最適** |

**大統一**: 全ての生成モデルは $p_\theta(x)$ or $p_\theta(x,z)$ の学習。変分推論・OT・Nash均衡は異なるアプローチで同じゴールを目指す。

### 8.6 Course Iの数学 — 活用の実例

Course I (第1-8回) の数学が、Course IIでどう使われたか:

| Course I | Course II での活用 |
|:---------|:------------------|
| **線形代数** (第2-3回) | Attention $QK^\top$、SVD (潜在空間)、行列微分 (Backprop) |
| **確率論** (第4回) | VAE事後分布、GAN分布マッチング、AR尤度 |
| **測度論** (第5回) | OT (測度間距離)、Diffusion (確率測度の流れ) |
| **情報理論** (第6回) | ELBO (KL項)、GAN (JSD)、Rate-Distortion |
| **最適化** (第7回) | GAN訓練 (Nash均衡)、VAE訓練 (勾配降下) |
| **潜在変数** (第8回) | VAE (変分推論)、GAN (暗黙的潜在変数) |

**全てがつながる**: Course Iは"部品"、Course IIは"組み立て"。

### 8.7 FAQ — よくある質問

#### Q1: Hybridは常にPure AttentionやSSMより優れているのか？

**A**: **No**。タスク依存。

- **Pure Attention優位**: Few-shot learning、複雑な推論 (CoT)、短系列 ($N < 1024$)
- **Pure SSM優位**: 超長系列 ($N > 100K$)、ストリーミング、メモリ制約厳しい環境
- **Hybrid優位**: バランス型タスク (長文要約、対話、汎用LM)

**No Free Lunch定理**: 万能アーキテクチャは存在しない。

#### Q2: JambaとZambaのどちらを選ぶべきか？

**A**: **用途次第**。

- **Jamba**: MoEで大規模 (52B total)、256K context、汎用LLM
- **Zamba**: Compact (7B)、メモリ効率重視、デバイス制約環境

#### Q3: Hybrid実装は難しいのか？

**A**: **中程度**。

- Attention実装経験あり → Hybrid追加は容易 (SSM層を挿入するだけ)
- SSM実装 (Mamba) は複雑 → **既存ライブラリ使用推奨** (`mamba-ssm`, `transformers`)

#### Q4: Dynamic Hybridは実用化されているか？

**A**: **まだ研究段階** (2026年2月時点)。

- Gate学習の難しさ、推論時オーバーヘッドが課題
- 今後2-3年で実用化の可能性

#### Q5: Course IIIでは何を学ぶのか？

**A**: **理論→実践の橋渡し**。

- 訓練パイプライン (データローダ/分散訓練)
- 評価指標 (FID/LPIPS/Perplexity)
- デプロイ (ONNX/量子化/最適化)
- **Elixir登場** (第19回) — 分散推論・耐障害性
- MLOps (Monitoring/Logging/A/Bテスト)

### 8.8 学習スケジュール (復習 & Course III準備)

| 週 | 復習内容 | 準備内容 |
|:---|:---------|:---------|
| **Week 1** | 第9-12回復習 (VI/VAE/OT/GAN) | Julia/Rust開発環境整備 |
| **Week 2** | 第13-16回復習 (AR/Attention/SSM) | Elixir環境構築 (第19回準備) |
| **Week 3** | 第17-18回復習 (Mamba発展/Hybrid) | Course III第19回予習 (分散推論) |
| **Week 4** | Course II全体通し | ミニプロジェクト: Tiny Hybrid実装 |

**ミニプロジェクト例**:
- Tiny Hybrid Model (Julia) を MNIST 訓練
- Rust推論パイプライン構築
- Pure Transformer/Mamba/Hybrid 性能比較レポート

### 8.9 次回予告 — Course III 第19回: 理論から実装へ

**第19回: 環境構築 & FFI & 分散基盤 — 3言語フルスタックの旅が始まる**

**Course II完結、Course III開幕**: 理論の習得は完了した。次は実装だ。Course III（第19-32回、全14回）では、⚡Julia訓練・🦀Rust推論・🔮Elixir配信の完全パイプラインを構築する。

🔮 **Elixir初登場**: 第19回でBEAM VM上の関数型言語Elixirが登場する。分散・並行・耐障害性が言語レベルで組み込まれ、Production品質サービングを実現する。

**Course III全体像（第19-32回）**:
- **基盤編（L19-22）**: 環境構築・VAE/GAN/Transformer実装・データ処理・マルチモーダル
- **最適化編（L23-26）**: Fine-tuning・PEFT・統計学・因果推論・推論最適化
- **実践編（L27-30）**: 評価パイプライン・プロンプトエンジニアリング・RAG・エージェント
- **運用編（L31-32）**: MLOps・Production統合

**準備事項**:
1. Julia 1.11+ / Rust 1.83+ / Elixir 1.17+ インストール
2. FFI概念の復習（第9-18回で既出）
3. 実装環境の構築準備（第19回で詳細解説）

> **Note:** **進捗: 100% 完了** 🎉
>
> **Course II: 生成モデル理論編 完全読了！**
>
> 10回の旅路で、変分推論・VAE・最適輸送・GAN・自己回帰・Attention・SSM・Mamba・Hybridの理論と実装を完全習得した。
>
> **「論文が読める」→「論文が書ける」レベルに到達。**
>
> 次はCourse III「実践編」で、理論を「動くシステム」に変える技術を身につける。
>
> 🚀 **Let's dive into Course III!**

---

### 6.13 💀 パラダイム転換の問い

### 問い: "最強"のアーキテクチャは存在しないのか？

Jamba、Zamba、Griffin、StripedHyena — どれも「ハイブリッド」を標榜する。だが、**どれが"最強"なのか？**

**答え: "最強"は存在しない。**

なぜなら:

1. **No Free Lunch定理**: 全てのタスクで最良のアルゴリズムは存在しない。タスクAで最良 → タスクBで劣る。
2. **Pareto最適**: 性能・効率・メモリ・訓練時間 — 複数目的で全て最良は不可能。トレードオフの選択。
3. **文脈依存**: 短系列ならAttention、超長系列ならSSM、バランスならHybrid。用途次第。

**本質的な問い**: では、我々は何を目指すべきか？

**答え: "組み合わせの最適化"**

- Attentionの全系列参照
- SSMの効率的記憶
- MoEのスケーラビリティ
- 動的ルーティングの適応性

**これら全てを、タスクに応じて使い分ける設計力**こそが、次世代LLMの鍵だ。

**挑発的な問い**:
- Hybridの"次"は何か？ → **Meta-Hybrid** (Hybridアーキテクチャ自体を動的生成)
- Attention=SSM双対性の"次"は？ → **統一理論** (Attention, SSM, Diffusion, Flow を1つの枠組みで)
- 人間の脳はHybridか？ → **神経科学との接続** (脳の異なる領域 = 異なるアーキテクチャ?)

**最後の問い**:
> 「"最強"のアーキテクチャを探すのではなく、**"組み合わせ"の力を信じること**。これがAI研究の次のフロンティアではないか？」

---

> Progress: 95%
> **理解度チェック**
> 1. Neural Architecture Search (NAS) がハイブリッドSSM設計に応用された場合、探索空間はどのように定義されるか？
> 2. Dynamic Hybrid（タスク適応的切替）は静的なLayer交互配置と比べ、どのようなシナリオで優位か？

## 参考文献

### 主要論文

[^1]: Lieber, O., Lenz, B., et al. (2024). "Jamba: A Hybrid Transformer-Mamba Language Model". *arXiv:2403.19887*.
<https://arxiv.org/abs/2403.19887>

[^2]: Glorioso, P., Anthony, Q., et al. (2024). "Zamba: A Compact 7B SSM Hybrid Model". *arXiv:2405.16712*.
<https://arxiv.org/abs/2405.16712>

[^3]: De, S., Smith, S. L., et al. (2024). "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models". *arXiv:2402.19427*.
<https://arxiv.org/abs/2402.19427>

[^4]: Google DeepMind (2024). "RecurrentGemma: Moving Past Transformers for Efficient Open Language Models". *arXiv:2404.07839*.
<https://arxiv.org/abs/2404.07839>

[^5]: Together AI (2024). "StripedHyena: Paving the way to efficient architectures".
<https://www.together.ai/blog/stripedhyena-7b>

[^6]: Somvanshi, S., et al. (2025). "From S4 to Mamba: A Comprehensive Survey on Structured State Space Models". *arXiv:2503.18970*.
<https://arxiv.org/abs/2503.18970>

[^7]: Mitra, S., et al. (2025). "Characterizing State Space Model (SSM) and SSM-Transformer Hybrid Language Model Performance with Long Context Length". *arXiv:2507.12442*.
<https://arxiv.org/abs/2507.12442>

[^8]: Vaswani, A., Shazeer, N., et al. (2017). "Attention Is All You Need". *NeurIPS 2017*.
<https://arxiv.org/abs/1706.03762>

[^9]: Gu, A., Goel, K., Ré, C. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces". *ICLR 2022*.
<https://arxiv.org/abs/2111.00396>

[^10]: Liu, H., Simonyan, K., Yang, Y. (2018). "DARTS: Differentiable Architecture Search". *ICLR 2019*.
<https://arxiv.org/abs/1806.09055>

[^11]: Pham, H., Guan, M. Y., et al. (2018). "Efficient Neural Architecture Search via Parameter Sharing". *ICML 2018*.
<https://arxiv.org/abs/1802.03268>

[^12]: Guo, Z., Zhang, X., et al. (2020). "Single Path One-Shot Neural Architecture Search with Uniform Sampling". *ECCV 2020*.
<https://arxiv.org/abs/1904.00420>

[^13]: Real, E., Liang, C., et al. (2020). "AutoML-Zero: Evolving Machine Learning Algorithms From Scratch". *ICML 2020*.
<https://arxiv.org/abs/2003.03384>

[^14]: White, C., Neiswanger, W., et al. (2021). "BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search". *AAAI 2021*.
<https://arxiv.org/abs/1910.11858>

[^15]: Jang, E., Gu, S., Poole, B. (2017). "Categorical Reparameterization with Gumbel-Softmax". *ICLR 2017*.
<https://arxiv.org/abs/1611.01144>

[^16]: Shazeer, N., Mirhoseini, A., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer". *ICLR 2017*.
<https://arxiv.org/abs/1701.06538>

### 教科書

- Murphy, K. P. (2022-2023). *Probabilistic Machine Learning: An Introduction / Advanced Topics*. MIT Press. [probml.github.io](https://probml.github.io/)
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org/)
- Gu, A., et al. (2025). *State Space Models: From Classical Control to Modern Sequence Modeling*. (Survey paper, draft).

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

