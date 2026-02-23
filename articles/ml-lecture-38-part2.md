---
title: "第38回: Flow Matching & 生成モデル統一理論: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "🌀"
type: "tech"
topics: ["machinelearning", "deeplearning", "flowmatching", "rust", "diffusion"]
published: true
slug: "ml-lecture-38-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

**→ 前編（理論編）**: [ml-lecture-38-part1](./ml-lecture-38-part1)

## 💻 Z5. 試練（実装） — Rust Flow Matching実装

理論を手を動かして確かめよう。ここでは、**Conditional Flow Matching (CFM)**の完全な実装を通じて、理論の各要素が実コードにどう対応するかを学ぶ。

---

### 4.1 実装の全体像

実装する内容：

1. **Gaussian Probability Paths**（OT Path / VP Path）
2. **Conditional Vector Field** $\mathbf{u}_t(\mathbf{x}|\mathbf{x}_1)$
3. **CFM Loss**の訓練ループ
4. **ODE Sampling**（Euler法 / RK4法）
5. **2次元玩具データセット**での可視化

実装言語：**Rust 1.11**（Candle + burn::optim + ode_solvers）

---

### 4.2 依存パッケージ

```rust:setup.rs
use candle_core::{Tensor, Device, DType, Result};
use candle_nn::{Module, VarBuilder, Linear, linear};
use ndarray::{Array1, Array2, ArrayView2, s};
use rand::{Rng, SeedableRng};

let mut rng = rand::rngs::StdRng::seed_from_u64(42); // seed=42: reproducible
```

---

### 4.3 データセット生成

2次元の**2峰ガウス混合**をターゲット分布とする：

```rust:dataset.rs
use ndarray::Array2;
use rand::Rng;
use rand_distr::StandardNormal;

/// Target distribution: mixture of 2 Gaussians
///     p_data(x) = 0.5*N([-2, 0], I) + 0.5*N([2, 0], I)
fn sample_target(n: usize, rng: &mut impl Rng) -> Array2<f32> {
    let d = 2;
    let centers: [[f32; 2]; 2] = [[-2.0, 0.0], [2.0, 0.0]];
    Array2::from_shape_fn((d, n), |(j, i)| {
        let mode = (i * 6364136223846793005u64.wrapping_add(i as u64) % 2) as usize;
        rng.sample::<f32, _>(StandardNormal) + centers[mode][j] // x₁ ~ p_data: N(centers[mode], I)
    })
}

/// Source distribution: standard Gaussian N(0, I)
fn sample_source(n: usize, d: usize, rng: &mut impl Rng) -> Array2<f32> { Array2::from_shape_fn((d, n), |_| rng.sample::<f32, _>(StandardNormal)) } // x₀ ~ N(0,I)
```

---

### 4.4 Probability Path定義

前述の理論に基づき、**Optimal Transport Path**と**VP Path**を実装する。

```rust:paths.rs
use ndarray::Array2;
use rand::Rng;
use rand_distr::StandardNormal;

/// Gaussian Probability Path: μ_t(x₁|x₀) と Σ_t
///
/// Parameters:
///   - path_type: PathType::OT (Optimal Transport) or PathType::VP (Variance Preserving)
#[derive(Clone, Copy)]
enum PathType { OT, VP }

struct GaussianPath {
    path_type: PathType,
    sigma_min: f32,
}

impl GaussianPath {
    /// Default: OT path with minimal noise
    fn new() -> Self { GaussianPath { path_type: PathType::OT, sigma_min: 1e-5 } }

    /// Compute μ_t(x₁, x₀) and σ_t at time t
    fn path_params(&self, t: f32, x1: &Array2<f32>, x0: &Array2<f32>)
        -> (Array2<f32>, f32)
    {
        match self.path_type {
            PathType::OT => {
                let mu_t = x1.mapv(|v| t * v) + x0.mapv(|v| (1.0 - t) * v); // μ_t(x₁,x₀) = t·x₁ + (1-t)·x₀  (OT straight path)
                (mu_t, self.sigma_min) // σ_t = σ_min  (OT path, constant noise)
            }
            PathType::VP => {
                let mu_t = x1.mapv(|v| t * v);           // μ_t = t·x₁  (VP path mean)
                let sigma_t = (1.0 - t * t).sqrt();      // σ_t = √(1-t²)  (VP path std)
                (mu_t, sigma_t)
            }
        }
    }

    /// Sample from conditional distribution q_t(x|x₁, x₀)
    ///     x_t ~ N(μ_t, σ_t²I)
    fn sample_conditional(&self, t: f32, x1: &Array2<f32>, x0: &Array2<f32>,
                           rng: &mut impl Rng) -> Array2<f32>
    {
        let (mu_t, sigma_t) = self.path_params(t, x1, x0);
        let eps = Array2::from_shape_fn(mu_t.raw_dim(),
            |_| rng.sample::<f32, _>(StandardNormal));
        mu_t + eps.mapv(|v| sigma_t * v) // xₜ ~ N(μ_t, σ_t²I)  (conditional path)
    }

    /// Compute conditional vector field u_t(x|x₁, x₀)
    ///     u_t = ∂μ_t/∂t + (σ_t σ'_t / σ_t²)(x - μ_t)
    fn conditional_vector_field(&self, t: f32, x_t: &Array2<f32>,
                                 x1: &Array2<f32>, x0: &Array2<f32>) -> Array2<f32>
    {
        match self.path_type {
            PathType::OT => {
                // uₜ(x|x₁,x₀) = x₁ - x₀  (constant! OT path)
                x1 - x0
            }
            PathType::VP => {
                // uₜ(x|x₁,x₀) = x₁ + σ'_t/σ_t·(x - μ_t)  (VP conditional field)
                let (mu_t, sigma_t) = self.path_params(t, x1, x0);
                let dsigma_dt = -t / (1.0 - t * t + 1e-8).sqrt(); // σ'_t = -t/√(1-t²)
                x1 + (x_t - &mu_t).mapv(|v| dsigma_dt / (sigma_t + 1e-8) * v)
            }
        }
    }
}
```

**重要なポイント**：
- OT Pathでは$\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$（定数！）
- VP Pathでは$\mathbf{u}_t$が$\mathbf{x}_t$に依存する

---

### 4.5 Vector Field Network

時刻$t$と位置$\mathbf{x}_t$から速度$\mathbf{v}_\theta(\mathbf{x}_t, t)$を予測するネットワーク。

```rust:network.rs
use candle_core::{Tensor, Device, Result};
use candle_nn::{Module, VarBuilder, Linear, linear, Activation};

// Vector field network: v_θ(xₜ, t) ≈ uₜ  (CFM target)
// Implements: trait FlowModel { fn forward(&self, x: &Tensor, t: &Tensor) -> Result<Tensor> }
/// Time-conditional MLP for vector field prediction
///     v_θ(x_t, t): (d+1) → 128 → 128 → d
struct VectorFieldNet {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl VectorFieldNet {
    fn new(d: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(d + 1, 128, vb.pp("fc1"))?,
            fc2: linear(128, 128, vb.pp("fc2"))?,
            fc3: linear(128, d, vb.pp("fc3"))?,
        })
    }

    /// Forward pass with time conditioning
    ///     v_θ(xₜ, t): R^{d+1} → R^d  (network forward)
    fn forward(&self, x_t: &Tensor, t: &Tensor) -> Result<Tensor> {
        let t_col = t.unsqueeze(1)?;
        let input = Tensor::cat(&[x_t, &t_col], 1)?; // [xₜ || t]: R^{d+1}  (time conditioning)
        let h = self.fc1.forward(&input)?.gelu()?;
        let h = self.fc2.forward(&h)?.gelu()?;
        self.fc3.forward(&h) // v_θ(xₜ, t): R^{d+1} → R^d
    }
}
```

---

### 4.6 CFM Loss実装

理論式（Zone 3.1）のLossを実装する：

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\left\|\mathbf{v}_\theta(t, \mathbf{x}_t) - \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0)\right\|^2\right]
$$

```rust:loss.rs
use ndarray::Array2;
use rand::Rng;

/// Conditional Flow Matching Loss
fn cfm_loss(
    model: &VectorFieldNet,
    path: &GaussianPath,
    batch_size: usize,
    rng: &mut impl Rng,
) -> f32 {
    let t: f32 = rng.gen(); // t ~ U(0,1)  (uniform time sampling)
    let x0 = sample_source(batch_size, 2, rng); // x₀ ~ N(0,I)
    let x1 = sample_target(batch_size, rng);    // x₁ ~ p_data
    let x_t = path.sample_conditional(t, &x1, &x0, rng); // xₜ ~ N(μ_t, σ_t²I)  (conditional path)
    let u_t = path.conditional_vector_field(t, &x_t, &x1, &x0); // uₜ: conditional target field
    let v_hat = model_predict(model, &x_t, t); // v_θ(xₜ, t): R^{d+1} → R^d
    // L_CFM = E_{t,x₀,x₁}[||v_θ(xₜ,t) - uₜ(xₜ|x₁,x₀)||²]
    let diff = &v_hat - &u_t;
    diff.iter().map(|v| v * v).sum::<f32>() / diff.len() as f32
}
```

---

### 4.7 訓練ループ

```rust:train.rs
use candle_nn::optim::{Adam, AdamConfig, Optimizer};
use std::time::Instant;

/// Train Flow Matching model
fn train_flow_matching(
    n_epochs: usize,
    batch_size: usize,
    lr: f64,
    path_type: PathType,
    rng: &mut impl Rng,
) -> Result<(VectorFieldNet, Vec<f32>)> {
    let d = 2;
    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = VectorFieldNet::new(d, vb.clone())?;
    let mut opt = Adam::new(vb.all_vars(), AdamConfig { lr, ..Default::default() })?;

    let path = GaussianPath { path_type, sigma_min: 1e-5 };
    let mut losses = Vec::with_capacity(n_epochs);

    for epoch in 0..n_epochs {
        let loss = cfm_loss(&model, &path, batch_size, rng); // L_CFM = E_{t,x₀,x₁}[||v_θ - uₜ||²]
        // Autograd backward + optimizer step handled via candle_core
        opt.backward_step(&Tensor::new(loss, &dev)?)?; // θ ← θ - α∇L_CFM  (Adam step)

        losses.push(loss);

        if (epoch + 1) % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch + 1, loss);
        }
    }

    Ok((model, losses))
}
```

---

### 4.8 ODE Sampling

訓練後、ODEを解いてサンプル生成：

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \mathbf{v}_\theta(\mathbf{x}_t, t), \quad \mathbf{x}_0 \sim \mathcal{N}(0, I)
$$

```rust:sampling.rs
use ndarray::Array2;

/// Euler ODE integrator: dx/dt = v_fn(x, t)
fn euler_integrate(
    v_fn: impl Fn(&Array2<f32>, f32) -> Array2<f32>,
    x0: &Array2<f32>,
    n_steps: usize,
) -> Array2<f32> {
    let dt = 1.0_f32 / n_steps as f32; // Euler step size Δt = 1/N
    let mut x = x0.clone();
    for step in 0..n_steps {
        let t = step as f32 * dt; // t = step × Δt
        let v = v_fn(&x, t);
        x = x + v * dt; // xₜ₊dt = xₜ + v_θ(xₜ,t)·dt  (ODE integrator)
    }
    x
}

/// Sample from learned flow via Euler ODE solving
fn sample_flow(
    model: &VectorFieldNet,
    n_samples: usize,
    n_steps: usize,
    rng: &mut impl Rng,
) -> Array2<f32> {
    let x0 = sample_source(n_samples, 2, rng);

    euler_integrate(
        |x, t| model_predict(model, x, t),
        &x0,
        n_steps,
    )
}
```

**注**：
- `Euler()`: 1次精度（速い）
- `RK4()`: 4次精度（高精度）
- Rectified Flowでは1-stepで十分（$\Delta t = 1$）

---

### 4.9 可視化

```rust:visualize.rs
/// Visualize training progress and generated samples
fn visualize_results(model: &VectorFieldNet, losses: &[f32], n_samples: usize,
                     rng: &mut impl Rng)
{
    // Plot 1: Training loss curve
    println!("Training Loss (last 10): {:?}", &losses[losses.len().saturating_sub(10)..]);

    // Plot 2: Generated samples vs real data
    let x_real = sample_target(n_samples, rng);
    let x_gen = sample_flow(model, n_samples, 100, rng);

    println!("Real samples shape: {:?}", x_real.shape());
    println!("Generated samples shape: {:?}", x_gen.shape());

    // Plot 3: Trajectory visualization (single sample)
    let x0_single = sample_source(1, 2, rng);
    let n_steps = 20;
    let dt = 1.0_f32 / n_steps as f32;
    let mut traj = vec![x0_single.clone()];
    let mut x = x0_single.clone();
    for step in 0..n_steps {
        let t = step as f32 * dt;
        let v = model_predict(model, &x, t); // v_θ(xₜ, t): R^{d+1} → R^d
        x = x + v * dt; // xₜ₊dt = xₜ + v_θ(xₜ,t)·dt  (Euler step)
        traj.push(x.clone());
    }
    println!("Trajectory length: {}", traj.len());
}
```

---

### 4.10 実行例

```rust:main.rs
fn main() -> Result<()> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Train OT-based CFM: μ_t(x₁,x₀) = t·x₁ + (1-t)·x₀  (straight path)
    let (model_ot, losses_ot) = train_flow_matching(
        1000, 256, 1e-3, PathType::OT, &mut rng
    )?;

    // Visualize
    visualize_results(&model_ot, &losses_ot, 1000, &mut rng);

    // Train VP-based CFM: μ_t = t·x₁,  σ_t = √(1-t²)  (VP path)
    let (model_vp, losses_vp) = train_flow_matching(
        1000, 256, 1e-3, PathType::VP, &mut rng
    )?;

    Ok(())
}
```

**期待される結果**：
- OT Pathの方が収束が速い（直線経路）
- VP Pathは若干迂回するが、安定性が高い
- どちらも真の分布を正確に再現

---

### 4.11 実装のポイント整理

| 理論要素 | 実装上の対応 |
|----------|--------------|
| $\mathbf{u}_t(\mathbf{x}\|\mathbf{x}_1, \mathbf{x}_0)$ | `conditional_vector_field()` |
| $\mu_t(\mathbf{x}_1, \mathbf{x}_0)$ | `path_params()` の `μ_t` |
| $q_t(\mathbf{x}\|\mathbf{x}_1, \mathbf{x}_0)$ | `sample_conditional()` |
| $\mathcal{L}_{\text{CFM}}$ | `cfm_loss()` のMSE |
| ODE Sampling | `sample_flow()` の `solve(ODEProblem)` |

> **Note:** **実装の核心**
> CFMの実装は驚くほどシンプル。Diffusion Modelのような複雑なノイズスケジュール、多段階逆過程、score networkの工夫は一切不要。**直線経路（OT Path）+ MSE Loss + ODE Solver**だけで十分だ。

---

### 🔬 実験・検証 実験ゾーン — 演習と検証

理論と実装を踏まえ、以下の演習を通じて理解を深めよう。

---

### 演習1: OT Path vs VP Pathの比較

**問題**：
Zone 4の実装で、`:ot`と`:vp`の両方を訓練し、以下を比較せよ：

1. **訓練速度**（同じlossに到達するepoch数）
2. **生成品質**（2-Wasserstein距離で定量評価）
3. **軌道の直線性**（始点→終点の直線からの平均偏差）

**ヒント**：
- Wasserstein距離：`using OptimalTransport; w2 = wasserstein(x_real, x_gen, 2)`
- 直線性：各時刻$t$での位置と直線$(1-t)\mathbf{x}_0 + t\mathbf{x}_1$の距離

**期待される観察**：
- OT Pathの方が訓練が速く、軌道も直線に近い
- VP Pathは初期段階で大きく迂回する

---

### 演習2: Rectified Flowの1-step生成

**問題**：
Rectified Flow（arXiv:2209.03003）は、OT Pathを**再学習**することで1-stepサンプリングを可能にする。次の手順で実装せよ：

**Step 1: 初期CFMの訓練**

```rust
// Step 1: train initial CFM  (OT path: μ_t = t·x₁ + (1-t)·x₀)
let (model_1, losses_1) = train_flow_matching(1000, 256, 1e-3, PathType::OT, &mut rng)?;
```

**Step 2: 軌道の再サンプリング**

訓練済みモデルで$\mathbf{x}_0 \to \mathbf{x}_1$の軌道を生成し、新しいペア$(\mathbf{x}_0', \mathbf{x}_1')$を作る：

```rust
fn resample_trajectories(model: &VectorFieldNet, n_samples: usize,
                          rng: &mut impl Rng) -> (Array2<f32>, Array2<f32>)
{
    let x0 = sample_source(n_samples, 2, rng); // x₀ ~ N(0,I)
    let x1 = sample_flow(model, n_samples, 100, rng); // x₁' = ODESolve(v_θ; x₀, t:0→1)  (Reflow)
    (x0, x1)
}
```

**Step 3: 直線経路での再訓練**

新しいペア$(\mathbf{x}_0', \mathbf{x}_1')$に対し、**完全な直線**を目標とする：

```rust
fn rectified_loss(model: &VectorFieldNet, x0: &Array2<f32>, x1: &Array2<f32>,
                   batch_size: usize, rng: &mut impl Rng) -> f32
{
    let n = x0.ncols();
    let idx: Vec<usize> = (0..batch_size).map(|_| rng.gen_range(0..n)).collect();
    let t: f32 = rng.gen();

    let x0_b = Array2::from_shape_fn((2, batch_size), |(r, c)| x0[[r, idx[c]]]);
    let x1_b = Array2::from_shape_fn((2, batch_size), |(r, c)| x1[[r, idx[c]]]);

    // x_t = t * x₁ + (1-t) * x₀
    let x_t = x1_b.mapv(|v| t * v) + x0_b.mapv(|v| (1.0 - t) * v); // μ_t(x₁,x₀) = t·x₁ + (1-t)·x₀  (OT straight path)
    let u_t = &x1_b - &x0_b; // u_t = x₁ - x₀  (straight-line target)

    let v_hat = model_predict(model, &x_t, t);
    let diff = v_hat - u_t;
    diff.iter().map(|v| v * v).sum::<f32>() / diff.len() as f32 // L_CFM = E[||v_θ(xₜ,t) - u_t||²]
}

```

**Step 4: 1-step生成のテスト**

```rust
// Resample
let (x0_new, x1_new) = resample_trajectories(&model_1, 10000, &mut rng);

// Re-train
let (model_2, _) = train_with_rectified_loss(&x0_new, &x1_new, &mut rng)?;

// 1-step sampling (Euler with Δt=1)
let x0_test = sample_source(1000, 2, &mut rng);
let t_ones: Vec<f32> = vec![1.0_f32; 1000];
let v_hat = model_predict_batch(&model_2, &x0_test, &t_ones);
let x1_gen = &x0_test + &v_hat; // x₁ ≈ x₀ + v_θ(x₀,0)·1  (1-step Euler, Δt=1)
```

**検証**：
- 1-step生成の品質が、初期モデルの50-step ODEに匹敵することを確認せよ

---

### 演習3: Score ↔ Flow等価性の数値検証

**問題**：
Zone 3.5の理論的等価性を数値的に検証せよ。

**Step 1: Diffusion Modelの訓練**

標準的なDDPMを訓練し、score function $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$を学習：

```rust
// Score network: ε_θ(x_t, t) ≈ -√(β_t) ∇log p_t(x_t)
fn train_score_model(_rng: &mut impl Rng) -> ScoreNet {
    // DDPM training (Zone 3.5の式を使用)
    todo!()
}
```

**Step 2: Score → Flowの変換**

Probability Flow ODE (3.5.3の式) を使って、scoreから速度場を計算：

```rust
// v_t(x) = -½β_t·[x + ε_θ(xₜ,t)]  (Score↔Flow equiv.)
fn score_to_flow(eps_theta: &Array2<f32>, x_t: &Array2<f32>, beta_t: f32) -> Array2<f32> { (x_t + eps_theta).mapv(|v| -0.5 * beta_t * v) }
```

**Step 3: 直接Flow Matchingとの比較**

CFMで訓練した速度場$\mathbf{v}_\theta$と、scoreから計算した速度場を比較：

```rust
// Sample test points
let x_test = sample_target(100, &mut rng);
let t_test: Vec<f32> = (0..100).map(|_| rng.gen::<f32>() * 0.9 + 0.05).collect(); // t ∈ [0.05, 0.95]

// CFM prediction
let v_cfm = model_predict_batch(&model_cfm, &x_test, &t_test);

// Score-based prediction
let eps_pred = model_predict_batch(&model_score, &x_test, &t_test);
let beta_t = 0.1_f32; // β_t = 0.1  (example diffusion coefficient)
let v_score = score_to_flow(&eps_pred, &x_test, beta_t); // v_t(x) = -½β_t·[x + ε_θ(xₜ,t)]  (Score↔Flow equiv.)

// Compute correlation
let v_cfm_flat: Vec<f32> = v_cfm.iter().cloned().collect();
let v_score_flat: Vec<f32> = v_score.iter().cloned().collect();
let n = v_cfm_flat.len() as f32;
let mean_c = v_cfm_flat.iter().sum::<f32>() / n;
let mean_s = v_score_flat.iter().sum::<f32>() / n;
let cov: f32 = v_cfm_flat.iter().zip(&v_score_flat).map(|(a, b)| (a - mean_c) * (b - mean_s)).sum::<f32>() / n;
let std_c = (v_cfm_flat.iter().map(|a| (a - mean_c).powi(2)).sum::<f32>() / n).sqrt();
let std_s = (v_score_flat.iter().map(|b| (b - mean_s).powi(2)).sum::<f32>() / n).sqrt();
let correlation = cov / (std_c * std_s + 1e-8);
println!("Score ↔ Flow correlation: {}", correlation);
```

**期待される結果**：
- 相関係数が0.95以上（ほぼ一致）
- 生成サンプルの品質も同等

---

### 演習4: DiffFlowのハイブリッド訓練

**問題**：
Zone 3.6のDiffFlowを簡易実装し、$\lambda$の効果を調べよ。

**Discriminator追加**：

```rust
// D(x,t): R^{d+1} → [0,1]  (discriminator for DiffFlow GAN term)
fn build_discriminator(d: usize, vb: VarBuilder) -> Result<Discriminator> {
    Ok(Discriminator {
        fc1: linear(d + 1, 64, vb.pp("fc1"))?,
        fc2: linear(64, 64, vb.pp("fc2"))?,
        fc3: linear(64, 1, vb.pp("fc3"))?,
    })
}
```

**DiffFlow Loss**：

```rust
fn diffflow_loss(model: &VectorFieldNet, disc: &Discriminator,
                 path: &GaussianPath, lambda: f32,
                 batch_size: usize, rng: &mut impl Rng) -> (f32, f32)
{
    // CFM term
    let loss_cfm = cfm_loss(model, path, batch_size, rng);

    // GAN term
    let x_real = sample_target(batch_size, rng);
    let x_fake = sample_flow(model, batch_size, 100, rng);

    let zeros = Array2::zeros((1, batch_size));
    let ones_arr = Array2::ones((1, batch_size));
    let d_real = disc_forward(disc, &ndarray::concatenate![ndarray::Axis(0), x_real, zeros]);
    let d_fake = disc_forward(disc, &ndarray::concatenate![ndarray::Axis(0), x_fake, ones_arr]);

    let loss_d = -(d_real.iter().map(|v| (v + 1e-8).ln()).sum::<f32>()
                 + d_fake.iter().map(|v| (1.0 - v + 1e-8).ln()).sum::<f32>())
                 / batch_size as f32;
    let loss_g = -d_fake.iter().map(|v| (v + 1e-8).ln()).sum::<f32>()
                 / batch_size as f32;

    let total_loss = loss_cfm + lambda * loss_g; // L_DiffFlow = L_CFM + λ·L_G  (hybrid CFM+GAN)
    (total_loss, loss_d)
}
```

**実験**：
- $\lambda \in \{0, 0.01, 0.1, 1.0\}$で訓練
- 各設定でFID（または2-Wasserstein距離）を計算
- 訓練安定性（lossの分散）を比較

**仮説**：
- $\lambda=0$：最も安定だが、サンプリングが遅い
- $\lambda=0.1$：品質と速度のバランスが最良
- $\lambda=1.0$：不安定化（mode collapse発生の可能性）

---

### 演習5: Wasserstein勾配流の可視化

**問題**：
JKOスキーム（Zone 3.7.5）を用いて、2次元分布の勾配流を可視化せよ。

**設定**：
- 初期分布$p_0 = \mathcal{N}([3, 3], I)$
- 目標分布$p_{\text{data}} = 0.5\mathcal{N}([-2, 0], I) + 0.5\mathcal{N}([2, 0], I)$
- 目的関数$\mathcal{F}[p] = \mathrm{KL}(p \| p_{\text{data}})$

**実装**：

```rust
fn jko_step(p_current: &Array2<f32>, p_target: &Array2<f32>, tau: f32,
             rng: &mut impl Rng) -> Array2<f32>
{
    // JKO step: min_p [KL(p||p_target) + W₂²(p, p_current)/(2τ)]
    let m = pairwise_sq_dist(p_current, p_target); // C_ij = ||xᵢ - yⱼ||²  (cost matrix)
    let gamma = sinkhorn_ot(&m.mapv(|v| v as f64), tau as f64, 100); // Entropic OT: min_π Σπᵢⱼcᵢⱼ + ε·H(π)

    // Update via transport plan: move particles toward target
    apply_transport(p_current, p_target, &gamma.mapv(|v| v as f32))
}

fn main_jko() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut p = sample_source(1000, 2, &mut rng);
    for k in 0..50 {
        let p_target = sample_target(1000, &mut rng);
        p = jko_step(&p, &p_target, 0.1, &mut rng);
        if k % 10 == 0 {
            println!("JKO step {}: p shape {:?}", k, p.shape());
        }
    }
}
```

**可視化**：
- 各ステップで分布のscatter plotをアニメーション化
- 軌道が「滑らかに」2峰ガウスに収束することを確認

---

### 実験のまとめ

| 演習 | 確認する理論 | 重要な観察 |
|------|--------------|------------|
| 演習1 | OT vs VP Path | OT = 直線 → 高効率 |
| 演習2 | Rectified Flow | 再訓練で1-step化可能 |
| 演習3 | Score ↔ Flow等価性 | 数値的にほぼ一致 |
| 演習4 | DiffFlow統一 | $\lambda$でDiffusion↔GAN連続変化 |
| 演習5 | Wasserstein勾配流 | JKO = 離散勾配降下 |

> **Note:** **実験の本質**
> 理論は美しいが、手を動かして初めて「なぜこれが革命的か」が腹落ちする。特に演習2のRectified Flowでは、**1-stepで高品質な画像が生成される瞬間**に立ち会える。これは、理論が実用に直結する稀有な例だ。

> **Progress: 85%**
> **理解度チェック**
> 1. OT-CFM 実装で `x_t = (1-t)*x0 + t*x1` を使った場合の条件付きベクトル場 `u_t = x1 - x0` が定数になる理由を、経路の微分から導け。
> 2. Rectified Flow の ReFlow アルゴリズムにおいて、訓練済みモデルから生成した軌道 $(x_0, x_1^\prime)$ をペアとして再訓練する理由（直線性改善のメカニズム）を説明せよ。

---

## 🔬 Z6. 新たな冒険へ（研究動向）

Flow Matchingは急速に進化している。ここでは、2024-2025年の最新研究と、未解決の課題を紹介する。

---

### 6.1 Flow Map Matching (Boffi+ NeurIPS 2025)

**問題意識**：
従来のCFMは、各サンプル$(\mathbf{x}_0, \mathbf{x}_1)$ごとに**独立に**条件付き速度場$\mathbf{u}_t(\mathbf{x}|\mathbf{x}_1)$を計算する。しかし、これは次の非効率を生む：

- サンプル間の**共通構造**（例：顔画像の目の位置）を活用できない
- 高次元データで計算コストが増大

**Flow Map Matchingの提案**：

「条件付き速度場」ではなく、**輸送写像**（transport map）$\mathbf{T}_t: \mathbb{R}^d \to \mathbb{R}^d$を直接学習する。

$$
\mathbf{x}_t = \mathbf{T}_t(\mathbf{x}_0), \quad \mathbf{v}_t(\mathbf{x}_t) = \frac{\partial \mathbf{T}_t}{\partial t}(\mathbf{T}_t^{-1}(\mathbf{x}_t))
$$

**利点**：
1. **Amortization**：一度$\mathbf{T}_t$を学習すれば、任意の$\mathbf{x}_0$に適用可能
2. **幾何学的制約**の統合（例：体積保存、曲率制約）
3. **逆写像**$\mathbf{T}_t^{-1}$も学習可能（双方向生成）

**実験結果**（ImageNet 64×64）：

| 手法 | FID ↓ | Sampling Steps | 訓練時間 |
|------|-------|----------------|----------|
| CFM | 2.31 | 50 | 100% |
| **Flow Map Matching** | **2.18** | **50** | **75%** |

---

### 6.2 Variational Rectified Flow (Guo+ 2025)

**問題**：
Rectified Flowの再訓練（reflow）は、軌道を直線に近づけるが、**理論的保証**がない。どの程度の再訓練で最適になるか？

**変分定式化**：

最適輸送写像を**変分問題**として定式化：

$$
\min_{\mathbf{T}} \mathbb{E}\left[\|\mathbf{T}(\mathbf{x}_0) - \mathbf{x}_1\|^2\right] + \lambda\,\mathrm{KL}(q_{\mathbf{T}} \| p_{\text{data}})
$$

ここで：
- 第1項：輸送コスト（直線性）
- 第2項：分布一致性
- $\lambda$：正則化パラメータ

**理論的成果**：
- 再訓練の**収束レート**を導出：$O(1/\sqrt{K})$（$K$=再訓練回数）
- 最適$\lambda$の選択基準を提供

**実用的インパクト**：
- 再訓練を2-3回で打ち切る理論的根拠
- 計算コスト削減

---

### 6.3 Multitask Stochastic Interpolants (Negrel+ 2025)

**動機**：
画像生成では、複数の条件（テキスト、スタイル、解像度）を同時に扱いたい。

**提案**：
Stochastic Interpolants（Zone 3.4）を**マルチタスク学習**に拡張：

$$
\mathcal{L}_{\text{multi}} = \sum_{k=1}^K w_k\,\mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1^{(k)}}\left[\left\|\mathbf{v}_\theta^{(k)}(t, \mathbf{x}_t) - \mathbf{u}_t^{(k)}\right\|^2\right]
$$

ここで：
- $k$：タスクインデックス（例：$k=1$はテキスト条件、$k=2$はスタイル条件）
- $w_k$：タスク重み
- $\mathbf{v}_\theta^{(k)}$：タスク固有の速度場

**技術的工夫**：
- **Adapter Modules**：共通バックボーン + タスク固有層
- **Task Balancing**：各タスクのlossを動的に調整（GradNormアルゴリズム）

**実験**：
- Text-to-ImageとStyle Transferを同時訓練
- 単一タスク訓練より**30%の計算削減**、品質は同等

---

### 6.4 Flow Matching for Discrete Domains

**課題**：
これまでのFlow Matchingは**連続空間**$\mathbb{R}^d$を仮定。しかし、テキスト、グラフ、分子などは**離散構造**を持つ。

**現在のアプローチ**：

1. **Embedding Space Flow**（Campbell+ 2024）
   - 離散トークンを連続embeddingに写像
   - Embedding空間でFlow Matching
   - デコード時に最近傍トークンに丸める

   **問題**：丸め誤差、embedding空間の非自然性

2. **Continuous-Time Markov Chain Flow**（Sun+ 2024）
   - 離散状態間の遷移確率をFlowとして定式化
   - Rate matrix $\mathbf{Q}_t$を学習

   $$
   \frac{\partial p_t}{\partial t} = p_t \mathbf{Q}_t
   $$

   **問題**：状態空間が大きいと$\mathbf{Q}_t$の次元爆発

**未解決問題**：
- 離散Flowの**最適輸送理論**の確立
- 効率的なサンプリングアルゴリズム

---

### 6.5 High-Resolution Image GenerationへのScale

**現状**：
- CIFAR-10 (32×32)：FID ~2
- ImageNet 64×64：FID ~2.5
- **ImageNet 256×256**：FID ~5-7（Diffusionに劣る）

**ボトルネック**：

1. **Memory**：高解像度では速度場ネットワークが巨大化
2. **ODE Stiffness**：複雑なデータでODEが「硬い」（stiff）になり、数値誤差が蓄積

**研究方向**：

**a) Latent Flow Matching**（Dao+ 2024）
- VAEの潜在空間でFlow Matching
- Stable Diffusionと同様のアプローチ
- ImageNet 256×256でFID **3.2**達成

**b) Multi-Scale Flow**（Kim+ 2024）
- 低解像度→高解像度の段階的生成
- 各スケールで独立なFlow
- メモリ効率が大幅向上

**c) Adaptive Step Size ODE Solver**
- DiffEq.jlの`Tsit5()`など、適応的ソルバーを活用
- Stiffnessを自動検出してステップサイズ調整

---

### 6.6 未解決の理論的問題

**Problem 1: 非凸最適化の保証**

CFM Lossは非凸だが、実際には局所最適に陥らない。なぜか？

**予想**：
- Over-parameterization（ニューラルネットが過剰に大きい）
- Loss landscapeが「フラット」（implicit regularization）

**必要な理論**：Neural Tangent Kernel (NTK)解析、Mean Field理論

---

**Problem 2: 最適なProbability Pathの選択**

OT Path、VP Path、General Pathのうち、**データ依存で最適な経路**を自動選択できるか？

**アイデア**：
- Meta-learning：複数のpathで訓練し、validation lossで選択
- Adaptive Path：データ分布の幾何学的特性（曲率、位相）から経路を構築

---

**Problem 3: サンプリング複雑度の下界**

Rectified Flowは1-stepを主張するが、**理論的に必要な最小ステップ数**は？

**既知の結果**：
- Lipschitz連続な速度場では、$O(\epsilon^{-1})$ステップで$\epsilon$-近似（標準的ODE理論）

**Open Question**：
- データの「複雑さ」（例：モード数、次元）と必要ステップ数の関係
- 1-stepが可能な条件の特徴づけ

---

### 6.7 応用領域の拡大

Flow Matchingは画像生成を超えて広がっている：

**a) 分子設計**（Drug Discovery）
- タンパク質の3D構造生成（AlphaFold的応用）
- 化学的制約（結合長、角度）をFlowに組み込む

**b) 音声合成**
- WaveNetの代替としてのFlow-based TTS
- リアルタイム生成（低レイテンシ）

**c) 強化学習**
- 行動ポリシーの生成モデル化
- Flow Matching + Actor-Critic

**d) 気象予測**
- 時空間データの確率的予測
- Ensemble生成（複数の未来軌道）

---

### 6.8 最新論文リスト（2024-2025）

訓練効率とスケーラビリティに関する最新研究：

1. **Flow Map Matching**（Boffi+ 2024, arXiv:2406.07507）
   - 輸送写像の直接学習

2. **Variational Rectified Flow**（Guo+ 2025, arXiv:2502.09616）
   - 変分定式化と収束保証

3. **Multitask Stochastic Interpolants**（Negrel+ 2025, arXiv:2508.04605）
   - マルチタスク学習への拡張

4. **Meta AI Flow Matching Guide**（2024, arXiv:2412.06264）
   - 実装ベストプラクティス集

5. **Discrete Flow Matching**（Campbell+ 2024）
   - テキスト生成への応用

<details><summary>深掘り: Flow Matching実装リソース</summary>

Flow Matchingのコミュニティは活発で、毎月新しい論文が登場する。以下のリソースが有用：

- **GitHub**: `atong01/conditional-flow-matching`（公式実装）
- **Papers with Code**: "Flow Matching"タグでフィルタ
- **Twitter**: #FlowMatching ハッシュタグ（研究者の議論）

特に、**ICLR 2025 Workshop on Flow-Based Models**では、未公開の最新研究が議論される。

</details>

> **Progress: 95%**
> **理解度チェック**
> 1. Flow Map Matching（arXiv:2406.07507）がなぜ反復サンプリングを不要にできるのか、Flow Consistency 条件の観点から説明せよ。
> 2. Wasserstein 勾配流の離散近似である JKO scheme $\rho^{k+1} = \arg\min_\rho \frac{1}{2\tau}W_2^2(\rho,\rho^k) + \mathcal{F}(\rho)$ において、$\tau\to0$ の極限で連続時間の Fokker-Planck 方程式が復元されることを直感的に説明せよ。

---

## 🎭 Z7. エピローグ（まとめ・FAQ・次回予告）

ここまでの長い旅を振り返り、重要なポイントを整理しよう。

---

### 7.1 この講義で学んだこと

**核心的洞察**：

1. **生成モデルの統一理論**
   - Score Matching、Diffusion Models、Flow Matching、GANsは、すべて**最適輸送理論のWasserstein勾配流**として理解できる
   - 違いは「目的関数$\mathcal{F}$」と「離散化手法」だけ

2. **Conditional Flow Matching (CFM)の革新性**
   - **周辺化トリック**により、周辺速度場$\mathbf{v}_t$を学習せずに、条件付き速度場$\mathbf{u}_t(\mathbf{x}|\mathbf{x}_1)$だけで訓練可能
   - Simulation-free（SDEを解かずに訓練できる）

3. **Optimal Transport (OT) Pathの優位性**
   - 直線経路 → 最短距離 → 少ないステップで高品質生成
   - Rectified Flowで1-step生成も可能

4. **Stochastic Interpolantsの一般性**
   - FlowとDiffusionを統一する枠組み
   - 確率的揺らぎ$\sigma_t$の選択で連続的に移行

5. **DiffFlowの統一視点**
   - SDMとGANが**同一SDE**から導出される
   - $g(t)$（拡散係数）と$\lambda$（GAN項の重み）で連続的に制御

---

### 7.2 重要な数式の総まとめ

**CFM Loss**：
$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\left\|\mathbf{v}_\theta(t, \mathbf{x}_t) - \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0)\right\|^2\right]
$$

**Gaussian Probability Path**（OT）：
$$
\mu_t(\mathbf{x}_1, \mathbf{x}_0) = t\mathbf{x}_1 + (1-t)\mathbf{x}_0, \quad \sigma_t = \sigma_{\min}
$$

**条件付き速度場**（OT Path）：
$$
\mathbf{u}_t(\mathbf{x} | \mathbf{x}_1, \mathbf{x}_0) = \mathbf{x}_1 - \mathbf{x}_0
$$

**Score ↔ Flow等価性**：
$$
\mathbf{v}_t(\mathbf{x}) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2\nabla_{\mathbf{x}}\log p_t(\mathbf{x})
$$

**Wasserstein勾配流**：
$$
\mathbf{v}_t = -\nabla \frac{\delta \mathcal{F}}{\delta p}\bigg|_{p=p_t}
$$

---

### 7.3 実装のチェックリスト

Flow Matchingを実装する際の必須要素：

- [ ] **Probability Path**の定義（`path_params()`）
- [ ] **条件付き速度場**の計算（`conditional_vector_field()`）
- [ ] **CFM Loss**の実装（MSE between $\mathbf{v}_\theta$ and $\mathbf{u}_t$）
- [ ] **時刻条件付きネットワーク**（入力に$t$を結合）
- [ ] **ODE Solver**（DifferentialEquations.jlなど）
- [ ] **可視化**（軌道、サンプル、loss curve）

---

### 7.4 よくある質問（FAQ）

**Q1: Flow MatchingとDiffusion Models、どちらを使うべき？**

**A**：
- **Flow Matching**：サンプリング速度が重要な場合（リアルタイム生成、1-step化）
- **Diffusion Models**：既存の大規模実装（Stable Diffusion）を活用したい場合
- **両者のハイブリッド**（DiffFlow）：最高品質を追求する場合

**現時点の推奨**：新規プロジェクトなら**Flow Matching**。理由：
- シンプルな実装
- 高速サンプリング
- 理論的に洗練されている

---

**Q2: なぜOT Pathが最適なのか？**

**A**：
最適輸送理論により、$p_0$から$p_1$への「最短経路」がOT Pathであることが保証される。数学的には：

$$
W_2(p_0, p_1)^2 = \inf_{\pi} \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim \pi}\left[\|\mathbf{x}_1 - \mathbf{x}_0\|^2\right]
$$

この最適解が直線経路$\mu_t = t\mathbf{x}_1 + (1-t)\mathbf{x}_0$を与える（Gaussianの場合）。

---

**Q3: Rectified Flowの再訓練は本当に必要？**

**A**：
**データ依存**。簡単な分布（MNIST、2D toy data）では初回訓練でほぼ直線。複雑な分布（ImageNet）では1-2回の再訓練で大幅改善。

**判断基準**：
- 軌道の直線性を測定（平均偏差）
- 1-step生成の品質をチェック
- 改善が見られなくなったら終了

---

**Q4: 高次元データ（例：1024×1024画像）でもFlow Matchingは有効？**

**A**：
**Latent Space Flow Matching**が有効。手順：

1. VAEで画像を低次元潜在空間に圧縮（例：1024×1024 → 64×64×4）
2. 潜在空間でFlow Matching訓練
3. デコーダで画像に戻す

Stable Diffusionと同じアプローチ。Meta AIのFlow Matching Guide（arXiv:2412.06264）に詳細あり。

---

**Q5: 実装で最もハマりやすいバグは？**

**A**：
**Top 3**：

1. **時刻$t$の範囲ミス**
   - 訓練では$t \in (0, 1)$だが、サンプリングでは$t=0$と$t=1$の境界も必要
   - 解決：`t = rand() * 0.98 + 0.01`で訓練、サンプリングは`t ∈ [0, 1]`

2. **ベクトル場の符号ミス**
   - $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$を$\mathbf{x}_0 - \mathbf{x}_1$と書いてしまう
   - 解決：Zone 1のインタラクティブ例で可視化して確認

3. **ODEの数値誤差**
   - Euler法でステップサイズが大きすぎる
   - 解決：RK4法を使う、またはステップサイズを半分に

---

**Q6: Wasserstein勾配流の理解は必須？**

**A**：
**実装には不要、理論の深い理解には必須**。

- 実装者：Zone 4のコードだけ読めばOK
- 研究者：Zone 3.7を熟読し、Jordan+ (1998) の原論文へ
- 数学的背景：測度論、変分法、PDE

---

### 7.5 次のステップ

**Level 1（初学者）**：
- [ ] Zone 4の実装を完全に再現
- [ ] 演習1-3を解く
- [ ] 2D toy datasetで可視化

**Level 2（中級者）**：
- [ ] MNIST/CIFAR-10でFlow Matching訓練
- [ ] Rectified Flow実装
- [ ] 演習4-5に挑戦

**Level 3（上級者）**：
- [ ] Latent Flow Matching実装（VAE統合）
- [ ] 最新論文（Zone 6.8）を実装
- [ ] 独自の応用領域で実験（音声、分子など）

**Level 4（研究者）**：
- [ ] 未解決問題（Zone 6.6）に取り組む
- [ ] 新しいProbability Pathを提案
- [ ] ICLR/NeurIPSに投稿

---

### 7.6 リソース集

**公式実装**：
- `atong01/conditional-flow-matching`（PyTorch、reference実装）
- `Candle/Burn`（Rust、本講義のベース）

**論文**：
- Flow Matching原論文（Lipman+ ICLR 2023, arXiv:2210.02747）
- Stochastic Interpolants（Albergo+ 2023, arXiv:2303.08797）
- DiffFlow（Zhang+ 2023, arXiv:2307.02159）

**チュートリアル**：
- Meta AI Flow Matching Guide（arXiv:2412.06264）
- Hugging Face Diffusers（Flow Matching実装例）

**数学的背景**：
- Optimal Transport（Villani, "Topics in Optimal Transportation"）
- Wasserstein Gradient Flow（Jordan+ "The Variational Formulation of the Fokker-Planck Equation", 1998）

---

## Paradigm-Breaking Question: 生成モデルの「次」は何か？

ここまでの講義で、我々は生成モデルの統一理論に到達した。Score Matching、Diffusion、Flow、GANは、すべて**Wasserstein勾配流**という同じ山の異なる登山ルートだ。

しかし、問いは残る：

> **「この統一理論の先に、さらなるパラダイムシフトはあるのか？」**

---

### 現在の限界

どれほど洗練されても、現在の生成モデルは本質的に**データの模倣**だ：

- 訓練データ$p_{\text{data}}$を近似する分布$p_\theta$を学習
- 新しい「創造」ではなく、「既存データの補間」

**具体例**：
- Stable Diffusionは、訓練データにない完全に新しい概念（例：「量子もつれを可視化した抽象画」）を生成できない
- Flow Matchingも、$p_0$から$p_{\text{data}}$への最適経路を学ぶだけ

---

### 次のパラダイムへの示唆

**方向1: 因果生成モデル**

現在のモデルは**相関**を学ぶが、**因果関係**は学ばない。

**必要な要素**：
- 構造因果モデル（SCM）とFlowの統合
- 介入（intervention）と反事実（counterfactual）の生成

**想像される応用**：
- 「この薬を投与しなかったら、どうなっていたか？」の画像生成
- 因果的に整合した未来予測

---

**方向2: アクティブ生成（Active Generation）**

現在のモデルは**受動的**（プロンプトに反応するだけ）。

**次世代**：
- 生成モデル自身が「次に何を生成すべきか」を能動的に決定
- 強化学習との深い統合（reward-conditioned flow）

**例**：
- ユーザーの意図を予測して、先回りで画像を提案
- 対話的な創造（AI: 「この色をもっと鮮やかにしますか？」）

---

**方向3: 物理法則埋め込み生成**

画像生成は自由すぎる（物理的にあり得ない画像も生成）。

**制約付き生成**：
- Navier-Stokes方程式を満たす流体シミュレーション画像
- 熱力学第二法則を満たすプロセス動画
- Flow MatchingのPathに**微分方程式制約**を埋め込む

**技術**：
- Physics-Informed Neural Networks (PINN) + Flow Matching
- Symplectic Flow（ハミルトン力学保存）

---

**方向4: 意味的連続性の探求**

OT Pathは「座標空間」で直線だが、「意味空間」では？

**問い**：
- 「猫」から「犬」への最適な変形経路は、座標の線形補間か？
- むしろ「猫 → ネコ科 → 動物 → イヌ科 → 犬」のような**概念階層**を辿るべきでは？

**研究**：
- 意味的距離（semantic distance）の定義
- 概念グラフ上のFlow

---

### あなたへの問い

このコースを修了したあなたに、最後の問いを投げかけたい：

**「Flow Matchingの次に来る、あなた自身の生成モデルは何か？」**

- それは、因果を扱うか？
- 物理法則を尊重するか？
- 意味的な構造を持つか？
- それとも、まったく別の原理に基づくか？

理論は道具だ。**真の創造は、道具を超えたところにある**。

---

**Congratulations!** 🎉

あなたは、生成モデルの最前線に到達した。ここから先は、あなた自身が道を切り拓く番だ。

---

## 7. 最新研究動向（2024-2025）

### 7.1 Conditional Variable Flow Matching (CVFM)

**問題設定**: 従来の Conditional Flow Matching (CFM) は固定条件 $c$ に対する生成 $p(x|c)$ を学習するが、**連続的な条件変数** $c \in \mathbb{R}^d$ に対する amortization（償却学習）は困難だった。

例: 温度パラメータ $T \in [0.1, 2.0]$ で生成スタイルを制御したいが、各 $T$ 値ごとに別モデルを訓練するのは非効率。

**CVFM の解決策** (Brennan et al., 2024) [^cvfm]:

Conditional OT (C²OT) を導入 — **条件依存コスト**でカップリングを学習:

$$
\pi^* = \arg\min_{\pi \in \Pi(p_0, p_1)} \mathbb{E}_{(x_0, x_1, c) \sim \pi} \left[ \| x_1 - x_0 \|^2 + \lambda \| g(c) - f(x_0, x_1) \|^2 \right]
$$

ここで:
- $g(c)$: 条件エンコーダ（例: MLP）
- $f(x_0, x_1)$: ペア特徴抽出器
- $\lambda$: アライメント強度

**直感**: 単なる OT は $c$ を無視して $p_0 \to p_1$ の最短経路を求める。C²OT は $c$ と $(x_0, x_1)$ の一貫性を罰則化 → 条件に応じた異なる経路を学習。

**Velocity Field**:

$$
v_\theta(x_t, t, c) = \text{VelocityNet}(x_t, t, g(c))
$$

訓練:

$$
\mathcal{L}_\text{CVFM} = \mathbb{E}_{t, c, (x_0, x_1) \sim \pi^*(c)} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]
$$

**実験結果** (Conditional Image Generation):

| Method | FID ↓ | Condition Fidelity (CLIP ↑) |
|:-------|:------|:----------------------------|
| CFM (per-condition) | 12.3 | 0.82 |
| Conditional Diffusion | 14.7 | 0.79 |
| **CVFM** | **11.1** | **0.85** |

**応用**: Text-to-Image で guidance scale $w \in [1, 20]$ を連続制御、分子生成で結合親和性を連続条件として学習。

### 7.2 Minibatch Optimal Transport Flow Matching

Tong et al. (2023) [^minibatch_ot] は、**ミニバッチ内で OT を解く**ことで計算量を $O(n^3)$ から $O(B^3)$ に削減（$B$ = バッチサイズ $\ll n$ = データセット全体）。

**課題**: 従来の OT-CFM は全データペア $(x_0^{(i)}, x_1^{(j)})$ の距離行列 $C_{ij} = \| x_1^{(j)} - x_0^{(i)} \|^2$ ($n \times n$) を解く必要 → メモリ $O(n^2)$、計算 $O(n^3)$。

**Minibatch OT のアイデア**:

各イテレーションでバッチ $\{x_0^{(i)}\}_{i=1}^B$ と $\{x_1^{(j)}\}_{j=1}^B$ をサンプリングし、**バッチ内 OT** を解く:

$$
\pi_B^* = \arg\min_{\pi \in \Pi(p_{B,0}, p_{B,1})} \sum_{i,j} \pi_{ij} \| x_1^{(j)} - x_0^{(i)} \|^2
$$

ここで $p_{B,0}, p_{B,1}$ はバッチの経験分布。

**理論的保証**: バッチサイズ $B$ が十分大きければ（$B \gtrsim \sqrt{n}$）、$\pi_B^*$ は真の OT $\pi^*$ に収束（Wasserstein 距離で）。

**実装** (Sinkhorn アルゴリズム):

```rust
use ndarray::{Array1, Array2};

fn sinkhorn_ot(c: &Array2<f64>, eps: f64, max_iter: usize) -> Array2<f64> {
    let b = c.nrows();
    let k = c.mapv(|v| (-v / eps).exp()); // K = exp(-C/ε)  (Gibbs kernel)
    let mut u = Array1::<f64>::ones(b);
    let mut v = Array1::<f64>::ones(b);

    for _ in 0..max_iter {
        u = k.dot(&v).mapv(|x| 1.0 / (x + 1e-8));     // u = 1/(Kv)   (Sinkhorn iteration)
        v = k.t().dot(&u).mapv(|x| 1.0 / (x + 1e-8)); // v = 1/(Kᵀu)
    }

    // π = diag(u)·K·diag(v)  (OT coupling)
    let pi = Array2::from_shape_fn((b, b), |(i, j)| u[i] * k[[i, j]] * v[j]);
    let s = pi.sum();
    pi / s // Normalize
}

fn minibatch_ot_loss(x0_batch: &Array2<f32>, x1_batch: &Array2<f32>,
                      model: &VectorFieldNet, t: f32) -> f32
{
    // L_OT-CFM = Σᵢⱼ πᵢⱼ·||v_θ(xₜ,t) - (x₁ⱼ-x₀ᵢ)||²
    let b = x0_batch.ncols();
    let c = pairwise_sq_dist_f64(x1_batch, x0_batch); // C_ij = ||x₁ⱼ - x₀ᵢ||²
    let pi = sinkhorn_ot(&c, 0.1, 100); // π = OT coupling  (ε=0.1)

    (0..b).flat_map(|i| (0..b).map(move |j| (i, j)))
        .filter(|&(i, j)| pi[[i, j]] > 1e-6)
        .map(|(i, j)| {
            let x_t = Array2::from_shape_fn((2, 1), |(r, _)|
                (1.0 - t) * x0_batch[[r, i]] + t * x1_batch[[r, j]]); // xₜ = (1-t)x₀ᵢ + t·x₁ⱼ
            let u_t = Array2::from_shape_fn((2, 1), |(r, _)|
                x1_batch[[r, j]] - x0_batch[[r, i]]); // uₜ = x₁ⱼ - x₀ᵢ  (OT straight-line)
            let v_hat = model_predict(model, &x_t, t);
            let diff = v_hat - u_t;
            pi[[i, j]] as f32 * diff.iter().map(|v| v * v).sum::<f32>()
        })
        .sum::<f32>() / b as f32
}
```

**計算量比較**:

| Method | OT Solve | Memory | Time/Iter |
|:-------|:---------|:-------|:----------|
| Full OT-CFM | $O(n^3)$ | $O(n^2)$ | 10-100s (n=50K) |
| **Minibatch OT-CFM** | $O(B^3)$ | $O(B^2)$ | **0.5s** (B=256) |

**品質**: CIFAR-10 で FID 差は 0.3 未満（ほぼ同等）。

### 7.3 Weighted Conditional Flow Matching

Liu et al. (2025) [^weighted_cfm] は、**サンプル重み付き CFM** を提案 — データの重要度に応じて学習を調整。

**動機**: データセットは不均衡（例: 医療画像で稀な疾患、テキストで低頻度語彙）。均一サンプリングは多数派バイアスを生む。

**Weighted CFM Loss**:

$$
\mathcal{L}_\text{WCFM} = \mathbb{E}_{t, x_0, x_1} \left[ w(x_0, x_1) \cdot \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

重み関数の例:

1. **Inverse Frequency**:
   $$
   w(x_1) = \frac{1}{\sqrt{\text{count}(c(x_1))}}
   $$
   $c(x_1)$ はクラスラベル。

2. **Importance Sampling**:
   $$
   w(x_0, x_1) = \frac{\| x_1 - x_0 \|^2}{\mathbb{E}[\| x_1 - x_0 \|^2]}
   $$
   難しいペア（距離が大きい）に注目。

3. **Curriculum Learning**:
   $$
   w(x_0, x_1; \text{epoch}) = \min\left(1, \frac{\text{epoch}}{T_\text{warmup}} \right) \cdot \mathbb{1}[\text{difficult}(x_0, x_1)]
   $$
   初期は簡単なサンプル、徐々に難しいサンプルへ。

**実験** (Imbalanced CIFAR-10, クラス比 1:100):

| Method | Minority Class FID ↓ | Majority Class FID ↓ |
|:-------|:---------------------|:---------------------|
| CFM (uniform) | 28.4 | 5.2 |
| Weighted Diffusion | 15.7 | 5.8 |
| **Weighted CFM** | **12.3** | **5.4** |

**Minority Class の品質が 2.3倍改善**（Majority への影響は最小）。

### 7.4 実装例: Minibatch OT-CFM (Rust)

以下は、前述の理論を統合した実装例。

```rust
use candle_core::{Tensor, Device, DType, Result};
use candle_nn::optim::{Adam, AdamConfig, Optimizer};
use ndarray::{Array1, Array2};
use rand::Rng;

// --- Minibatch OT Solver ---
fn sinkhorn_coupling(c: &Array2<f32>, eps: f32, max_iter: usize) -> Array2<f32> {
    let b = c.nrows();
    let k = c.mapv(|v| (-v / eps).exp()); // K = exp(-C/ε)  (Gibbs kernel)
    let mut u = Array1::<f32>::ones(b);
    let mut v = Array1::<f32>::ones(b);

    for _ in 0..max_iter {
        u = k.dot(&v).mapv(|x| 1.0 / (x + 1e-8));     // u = 1/(Kv)   (Sinkhorn iteration)
        v = k.t().dot(&u).mapv(|x| 1.0 / (x + 1e-8)); // v = 1/(Kᵀu)
    }

    let pi = Array2::from_shape_fn((b, b), |(i, j)| u[i] * k[[i, j]] * v[j]); // π = diag(u)·K·diag(v)  (OT coupling)
    let s = pi.sum();
    pi / s
}

// --- Velocity Network ---
fn velocity_net(d_in: usize, d_hidden: usize, vb: VarBuilder) -> Result<VectorFieldNet> { VectorFieldNet::new_with_hidden(d_in, d_hidden, vb) }

// --- Minibatch OT-CFM Training ---
fn train_minibatch_ot_cfm(
    data_source: impl Fn(&mut rand::rngs::StdRng) -> Array2<f32>,
    data_target: impl Fn(&mut rand::rngs::StdRng) -> Array2<f32>,
    n_epochs: usize,
    batch_size: usize,
    eps_sinkhorn: f32,
    rng: &mut rand::rngs::StdRng,
) -> Result<VectorFieldNet> {
    let d = 2;
    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = velocity_net(d, 128, vb.clone())?;
    let mut opt = Adam::new(vb.all_vars(), AdamConfig { lr: 1e-3, ..Default::default() })?;

    for epoch in 0..n_epochs {
        // Sample batches
        let x0 = data_source(rng); // (d, B)
        let x1 = data_target(rng); // (d, B)

        let c = pairwise_sq_dist(&x1, &x0); // C_ij = ||x₁ⱼ - x₀ᵢ||²  (cost matrix)
        let pi = sinkhorn_coupling(&c, eps_sinkhorn, 50); // π = OT coupling via Sinkhorn
        let t: f32 = rng.gen(); // t ~ U(0,1)
        // L_OT-CFM = Σᵢⱼ πᵢⱼ·||v_θ(xₜ,t) - (x₁ⱼ-x₀ᵢ)||²
        let loss = (0..batch_size).flat_map(|i| (0..batch_size).map(move |j| (i, j)))
            .filter(|&(i, j)| pi[[i, j]] > 1e-6)
            .map(|(i, j)| {
                let x_t = Array2::from_shape_fn((d, 1), |(r, _)|
                    (1.0 - t) * x0[[r, i]] + t * x1[[r, j]]); // xₜ = (1-t)x₀ᵢ + t·x₁ⱼ
                let u_t_vec: Vec<f32> = (0..d).map(|r| x1[[r, j]] - x0[[r, i]]).collect(); // uₜ = x₁ⱼ - x₀ᵢ
                let v_hat = model_predict(&model, &x_t, t); // v_θ(xₜ, t): R^{d+1} → R^d
                let diff: f32 = v_hat.iter().zip(&u_t_vec).map(|(a, b)| (a - b).powi(2)).sum();
                pi[[i, j]] * diff
            })
            .sum::<f32>() / batch_size as f32;

        opt.backward_step(&Tensor::new(loss, &dev)?)?;

        if (epoch + 1) % 10 == 0 {
            println!("Epoch {}, Loss: {}", epoch + 1, loss);
        }
    }

    Ok(model)
}

// --- ODE Sampling ---
fn sample_ot_cfm(model: &VectorFieldNet, x0: &Array2<f32>, n_steps: usize) -> Array2<f32> { euler_integrate(|x, t| model_predict(model, x, t), x0, n_steps) } // xₜ₊dt = xₜ + v_θ(xₜ,t)·dt  (ODE integrator)
```

**使用例**:

```rust
// x₀ ~ N(0,I), x₁ ~ N([3,0],I): Two Gaussians
let source = |rng: &mut rand::rngs::StdRng| -> Array2<f32> {
    sample_source(256, 2, rng) // x₀ ~ N(0,I)
};
let target = |rng: &mut rand::rngs::StdRng| -> Array2<f32> {
    sample_source(256, 2, rng).mapv(|v| v) + 3.0_f32 // x₁ ~ N([3,0],I)
};

// Train minibatch OT-CFM: L_OT-CFM = Σᵢⱼ πᵢⱼ·||v_θ(xₜ,t) - (x₁ⱼ-x₀ᵢ)||²
let model = train_minibatch_ot_cfm(source, target, 200, 256, 0.1, &mut rng)?;

// Sample
let x0_test = sample_source(500, 2, &mut rng); // x₀ ~ N(0,I)
let x1_samples = sample_ot_cfm(&model, &x0_test, 100); // ODE solve: x₁ = ODESolve(v_θ; x₀)

// Print summary
println!("Source: {:?}", x0_test.shape());
println!("Generated: {:?}", x1_samples.shape());
```

---

## 参考文献

[^cvfm]: Brennan, M., et al. (2024). "Conditional Variable Flow Matching: Transforming Conditional Densities with Amortized Conditional Optimal Transport". *arXiv:2411.08314*.

[^minibatch_ot]: Tong, A., et al. (2023). "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport". *arXiv:2302.00482*.

[^weighted_cfm]: Calvo-Ordonez, S., et al. (2025). "Weighted Conditional Flow Matching". *arXiv:2507.22270*.

---

### 7.5 Rectified Flow: Flow Matching の理論的洗練

Liu et al. (2023) は、**Rectified Flow** を提案 — Flow Matching の経路をより直線的にする手法。

**問題**: 標準 OT-CFM でも、経路 $\mathbf{x}_t$ は完全な直線ではない（データ多様体の曲率の影響）。曲がった経路 → より多くの NFE が必要。

**Rectification のアイデア**:

1. **初期 Flow** を訓練（OT-CFM）
2. **Reflow**: 訓練済み Flow でサンプルペア $(x_0', x_1')$ を生成
3. これらのペアで**再訓練** → より直線的な Flow

数学的には:

$$
(x_0^{(k+1)}, x_1^{(k+1)}) = \text{Sample from } p_\theta^{(k)}
$$

$k$ 回目の Flow で生成したペアを使い、$k+1$ 回目を訓練。

**理論的保証**: $k \to \infty$ で、経路は**ほぼ直線**に収束 → 1-step sampling が可能。

**実験** (CIFAR-10):

| Iteration | Steps for FID<5 | Training Time |
|:----------|:----------------|:--------------|
| k=0 (OT-CFM) | 20 | 1× |
| k=1 (Reflow) | 10 | 2× (累積) |
| k=2 (Reflow²) | **5** | 3× (累積) |

**2回の Reflow で 5-step 生成** を達成。

**Rust 実装**:

```rust
fn reflow_iteration(
    model_k: &VectorFieldNet,
    data_source: impl Fn(&mut rand::rngs::StdRng) -> Array2<f32>,
    n_samples: usize,
    rng: &mut rand::rngs::StdRng,
) -> Result<VectorFieldNet> {
    let x0_new: Vec<Array2<f32>> = (0..n_samples)
        .map(|_| data_source(rng)) // x₀ ~ p_source
        .collect();
    let x1_new: Vec<Array2<f32>> = x0_new.iter()
        .map(|x0| euler_integrate(|x, t| model_predict(model_k, x, t), x0, 100)) // x₁' = ODESolve(v_θ; x₀, t:0→1)  (Reflow)
        .collect();

    // Re-train with rectified pairs
    train_cfm_from_pairs(&x0_new, &x1_new, rng)
}
```

**応用**: Text-to-Image (Stable Diffusion) で Reflow² → 4-step 生成で品質維持。

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
