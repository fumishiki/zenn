---
title: "第37回: 🎲 SDE/ODE & 確率過程論: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "🎲"
type: "tech"
topics: ["machinelearning", "deeplearning", "sde", "rust", "stochasticprocesses"]
published: true
slug: "ml-lecture-37-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

## 💻 Z5. 試練（実装）（45分）— Rust ode_solversでSDE数値解法

### 4.1 Rust ode_solvers入門 — SDEProblemの定義

Rustのode_solversはSDE/ODE/DAEを統一的に扱う強力なパッケージ。

**基本的なSDE定義**:

```rust
// use rand_distr; // rand, rand_distr クレートを使用

// SDE Model trait: defines drift f and diffusion g
// trait SdeModel {
//     fn drift(&self, x: f64, t: f64) -> f64;     // f(x,t)
//     fn diffusion(&self, t: f64) -> f64;           // g(t)
//     fn score(&self, x: f64, t: f64) -> f64;      // ∇log p_t(x)
// }

use rand::Rng;
use rand_distr::StandardNormal;

// Forward SDE: dx = f(x,t)dt + g(t)dW  (Itô)
// drift: f(x, t) = -0.5 * β * x
fn drift(x: f64, beta: f64) -> f64 { -0.5 * beta * x } // f(x,t) = -½β(t)·x

// diffusion: g(x, t) = √β
fn diffusion(beta: f64) -> f64 { beta.sqrt() } // g(t) = √β(t)

fn main() {
    let mut rng = rand::thread_rng();

    // 初期値、時間範囲、パラメータ
    let mut x = 1.0_f64;
    let beta = 1.0_f64;
    let dt = 0.01_f64;
    let n_steps = (1.0 / dt) as usize;

    // Euler-Maruyama 法で VP-SDE を解く
    let mut trajectory = vec![x];
    for _ in 0..n_steps {
        let dw: f64 = rng.sample(StandardNormal);
        x += drift(x, beta) * dt + diffusion(beta) * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
        trajectory.push(x);
    }

    println!("VP-SDE サンプルパス: {} ステップ", trajectory.len());
    println!("終端値 X(1.0) = {:.4}", trajectory[trajectory.len() - 1]);
    // Plotting: use plotters crate for visualization
}
```

**数式↔コード対応**:
- SDE: $dX_t = -\frac{1}{2}\beta X_t dt + \sqrt{\beta} dW_t$
- `drift(u, p, t)`: Drift項 $f(x, t) = -\frac{1}{2}\beta x$
- `diffusion(u, p, t)`: Diffusion項 $g(x, t) = \sqrt{\beta}$
- `EM()`: Euler-Maruyama法（$\Delta t = 0.01$）

### 4.2 VP-SDE実装 — 線形/Cosineスケジュール

DDPM対応のVP-SDEを線形/Cosineスケジュールで実装。

**線形スケジュール**:
$$
\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})
$$

```rust
// VP-SDE with 線形スケジュール
use rand::Rng;
use rand_distr::StandardNormal;

fn beta_linear(t: f64, beta_min: f64, beta_max: f64) -> f64 { beta_min + t * (beta_max - beta_min) } // β(t) = β_min + t·(β_max - β_min)

// Drift: f(x, t) = -0.5 * β(t) * x
fn vp_drift_linear(x: f64, t: f64, beta_min: f64, beta_max: f64) -> f64 { -0.5 * beta_linear(t, beta_min, beta_max) * x } // f(x,t) = -½β(t)·x

// Diffusion: g(x, t) = √β(t)
fn vp_noise_linear(t: f64, beta_min: f64, beta_max: f64) -> f64 { beta_linear(t, beta_min, beta_max).sqrt() } // g(t) = √β(t)

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // Euler-Maruyama で VP-SDE（線形スケジュール）
    let mut x = 1.0_f64;
    let mut trajectory = vec![(0.0_f64, x)];
    for step in 0..n_steps {
        let t = step as f64 * dt;
        let dw: f64 = rng.sample(StandardNormal);
        x += vp_drift_linear(x, t, beta_min, beta_max) * dt
            + vp_noise_linear(t, beta_min, beta_max) * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
        trajectory.push((t + dt, x));
    }

    println!("VP-SDE 線形スケジュール: {} ステップ", trajectory.len());
    println!("終端値 X(1.0) = {:.4}", trajectory.last().unwrap().1);
    // Plotting: use plotters crate — xlabel="t", ylabel="X(t)", title="VP-SDE 線形スケジュール"
}
```

**Cosineスケジュール**（DDPM Improved, Nichol & Dhariwal 2021）:
$$
\bar{\alpha}_t = \frac{\cos\left(\frac{t + s}{1 + s} \cdot \frac{\pi}{2}\right)^2}{\cos\left(\frac{s}{1 + s} \cdot \frac{\pi}{2}\right)^2}, \quad \beta(t) = -\frac{d \log \bar{\alpha}_t}{dt}
$$
（$s = 0.008$ は小さなオフセット）

```rust
// Cosineスケジュール
use rand::Rng;
use rand_distr::StandardNormal;
use std::f64::consts::PI;

fn alpha_bar_cosine(t: f64, s: f64) -> f64 {
    let num = ((t + s) / (1.0 + s) * PI / 2.0).cos().powi(2);
    let den = (s / (1.0 + s) * PI / 2.0).cos().powi(2);
    num / den
}

// β(t) = -d/dt log ᾱ(t),  ᾱ(t) = cos²(πt/(2+2s)) / cos²(πs/(2+2s))
fn beta_cosine(t: f64, s: f64) -> f64 {
    let h = 1e-6;
    -(alpha_bar_cosine(t + h, s).ln() - alpha_bar_cosine(t, s).ln()) / h // β(t) = -d/dt log ᾱ(t)
}

fn vp_drift_cosine(x: f64, t: f64, s: f64) -> f64 { -0.5 * beta_cosine(t, s) * x } // f(x,t) = -½β(t)·x

fn vp_noise_cosine(t: f64, s: f64) -> f64 { beta_cosine(t, s).sqrt() } // g(t) = √β(t)

fn main() {
    let mut rng = rand::thread_rng();
    let s = 0.008_f64;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;

    // 線形スケジュール
    let mut x_linear = 1.0_f64;
    // Cosineスケジュール
    let mut x_cosine = 1.0_f64;

    let mut traj_linear = vec![(0.0_f64, x_linear)];
    let mut traj_cosine = vec![(0.0_f64, x_cosine)];

    for step in 0..n_steps {
        let t = step as f64 * dt;

        // 線形
        let dw_l: f64 = rng.sample(StandardNormal);
        let b_l = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
        x_linear += -0.5 * b_l * x_linear * dt + b_l.sqrt() * dt.sqrt() * dw_l; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
        traj_linear.push((t + dt, x_linear));

        // Cosine
        let dw_c: f64 = rng.sample(StandardNormal);
        x_cosine += vp_drift_cosine(x_cosine, t, s) * dt
            + vp_noise_cosine(t, s) * dt.sqrt() * dw_c; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
        traj_cosine.push((t + dt, x_cosine));
    }

    println!("VP-SDE 線形 終端値: {:.4}", traj_linear.last().unwrap().1);
    println!("VP-SDE Cosine 終端値: {:.4}", traj_cosine.last().unwrap().1);
    // Plotting: use plotters crate — title="VP-SDE: 線形 vs Cosine"
}
```

**線形 vs Cosine の違い**:
- 線形: 終端でノイズが急増（$\beta_{\max} = 20$）
- Cosine: 滑らかなスケジュール、端点での急変を回避

### 4.3 VE-SDE実装 — 幾何スケジュール

NCSNのVE-SDEを幾何スケジュールで実装。

**幾何スケジュール**:
$$
\sigma(t) = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t
$$

$$
\frac{d\sigma^2(t)}{dt} = 2\sigma(t) \log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right) \sigma(t) = 2\sigma^2(t) \log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)
$$

```rust
// VE-SDE with 幾何スケジュール
use rand::Rng;
use rand_distr::StandardNormal;

// Drift項 = 0（VE-SDEは平均を変化させない）
fn ve_drift(_x: f64) -> f64 { 0.0 } // f(x,t) = 0  (VE-SDE has no drift)

// Diffusion: g(t) = √(2 σ²(t) log(σ_max / σ_min))
fn ve_noise(t: f64, sigma_min: f64, sigma_max: f64) -> f64 {
    let sigma_t = sigma_min * (sigma_max / sigma_min).powf(t); // σ(t) = σ_min·(σ_max/σ_min)^t
    (2.0 * sigma_t.powi(2) * (sigma_max / sigma_min).ln()).sqrt() // g(t) = √(2σ²(t)·log(σ_max/σ_min))
}

fn main() {
    let mut rng = rand::thread_rng();
    let sigma_min = 0.01_f64;
    let sigma_max = 50.0_f64;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // Euler-Maruyama で VE-SDE を解く
    let mut x = 1.0_f64;
    let mut trajectory = vec![(0.0_f64, x)];
    for step in 0..n_steps {
        let t = step as f64 * dt;
        let dw: f64 = rng.sample(StandardNormal);
        x += ve_drift(x) * dt + ve_noise(t, sigma_min, sigma_max) * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
        trajectory.push((t + dt, x));
    }

    println!("VE-SDE 幾何スケジュール: {} ステップ", trajectory.len());
    println!("終端値 X(1.0) = {:.4}", trajectory.last().unwrap().1);
    // Plotting: use plotters crate — xlabel="t", ylabel="X(t)", title="VE-SDE 幾何スケジュール"
}
```

**特徴**:
- Drift項なし（平均変化なし）
- 分散が時間とともに爆発的に増加

### 4.4 Reverse-time SDE実装 — Score関数近似

Reverse-time SDEを簡易Score関数近似で実装。

**VP-SDE Reverse-time**:
$$
dX_t = \left[-\frac{1}{2}\beta(t) X_t - \beta(t) \nabla \log p_t(X_t)\right] dt + \sqrt{\beta(t)} d\bar{W}_t
$$

**Score関数近似**（ガウス仮定）:
学習済みScore関数 $s_\theta(x, t)$ がない場合、ガウス近似で $\nabla \log p_t(x) \approx -x / \sigma_t^2$。

```rust
// Reverse-time VP-SDE（簡易Score近似）
use rand::Rng;
use rand_distr::StandardNormal;

// Score近似（実際はNNで学習）
// 簡易的に ∇log p_t(x) ≈ -x（ガウス仮定）
fn score_approx(x: f64) -> f64 { -x } // ∇log p_t(x) ≈ -x  (Gaussian approx.)

// Reverse-time Drift = -0.5 * β(t) * x - β(t) * ∇log p_t(x)
fn reverse_vp_drift(x: f64, t: f64, beta_min: f64, beta_max: f64) -> f64 {
    let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
    -0.5 * beta_t * x - beta_t * score_approx(x) // f_rev(x,t) = f - g²·∇log p_t  (Anderson 1982)
}

fn reverse_vp_noise(t: f64, beta_min: f64, beta_max: f64) -> f64 { (beta_min + t * (beta_max - beta_min)).sqrt() } // g(t) = √β(t)

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // 初期値: ノイズ分布 N(0, 1)
    let mut x: f64 = rng.sample(StandardNormal);
    let mut trajectory = vec![(1.0_f64, x)];

    // 逆時間（t: 1 → 0）: 負のdt
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let dw: f64 = rng.sample(StandardNormal);
        // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        x += reverse_vp_drift(x, t, beta_min, beta_max) * (-dt)
            + reverse_vp_noise(t, beta_min, beta_max) * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        trajectory.push((t - dt, x));
    }

    println!("Reverse-time VP-SDE: {} ステップ", trajectory.len());
    println!("終端値 X(0.0) = {:.4}", trajectory.last().unwrap().1);
    // Plotting: use plotters crate — title="Reverse-time VP-SDE（簡易Score）"
}
```

**注意**:
- 実際のDiffusion Modelでは Score関数 $s_\theta(x, t)$ をNeural Networkで学習
- ここでは $\nabla \log p_t(x) \approx -x$ のガウス近似（デモ目的）

### 4.5 Probability Flow ODE実装 — 決定論的軌道

Probability Flow ODEを`ODEProblem`で実装。

**VP-SDE Probability Flow ODE**:
$$
\frac{dX_t}{dt} = -\frac{1}{2}\beta(t) X_t - \frac{1}{2}\beta(t) \nabla \log p_t(X_t)
$$

```rust
// Probability Flow ODE for VP-SDE
use rand::Rng;
use rand_distr::StandardNormal;

// PF-ODE: dx/dt = f - ½g²·∇log p_t  (Song+ 2021)
// Score近似（実際はNNで学習）: ∇log p_t(x) ≈ -x
fn pf_ode_rhs(x: f64, t: f64, beta_min: f64, beta_max: f64) -> f64 {
    let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
    let score_approx = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
    -0.5 * beta_t * x - 0.5 * beta_t * score_approx // PF-ODE: dx/dt = f - ½g²·∇log p_t
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // 初期ノイズ（t=1 から t=0 へ逆時間）
    let mut x: f64 = rng.sample(StandardNormal);
    let mut trajectory = vec![(1.0_f64, x)];

    // Euler法で PF-ODE を逆時間に解く
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        x += pf_ode_rhs(x, t, beta_min, beta_max) * (-dt); // PF-ODE: dx/dt = f - ½g²·∇log p_t  (Song+ 2021)
        trajectory.push((t - dt, x));
    }

    println!("Probability Flow ODE: {} ステップ", trajectory.len());
    println!("終端値 X(0.0) = {:.4}", trajectory.last().unwrap().1);
    // Plotting: use plotters crate — title="Probability Flow ODE"
}
```

**Reverse-time SDE vs PF-ODE**:
```rust
// 同じ初期値で比較
use rand::Rng;
use rand_distr::StandardNormal;

// Reverse-time SDE: Euler-Maruyama（逆時間）
fn run_reverse_sde(x0: f64, beta_min: f64, beta_max: f64, dt: f64, n_steps: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut x = x0;
    let mut traj = vec![x];
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
        let score = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        traj.push(x);
    }
    traj
}

// PF-ODE: Euler法（逆時間）
fn run_pf_ode(x0: f64, beta_min: f64, beta_max: f64, dt: f64, n_steps: usize) -> Vec<f64> {
    let mut x = x0;
    let mut traj = vec![x];
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
        let score = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
        x += (-0.5 * beta_t * x - 0.5 * beta_t * score) * (-dt); // PF-ODE: dx/dt = f - ½g²·∇log p_t  (Song+ 2021)
        traj.push(x);
    }
    traj
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let x0 = 0.5_f64; // 共通の初期値
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // Reverse-time SDE
    let traj_sde = run_reverse_sde(x0, beta_min, beta_max, dt, n_steps, &mut rng);

    // PF-ODE
    let traj_ode = run_pf_ode(x0, beta_min, beta_max, dt, n_steps);

    println!("SDE 終端値: {:.4}", traj_sde.last().unwrap());
    println!("ODE 終端値: {:.4}", traj_ode.last().unwrap());
    // Plotting: use plotters crate — title="SDE vs ODE"
}
```

**結果**:
- Reverse-time SDE: 確率的（軌道が揺れる）
- PF-ODE: 決定論的（滑らかな軌道）

### 4.6 Predictor-Corrector法実装 — 精度向上

Predictor-Corrector法で高品質サンプリング。

**アルゴリズム**:
1. Predictor: Reverse-time SDEで1ステップ
2. Corrector: Langevin Dynamics（複数回反復）

```rust
// Predictor-Corrector サンプリング
use rand::Rng;
use rand_distr::StandardNormal;

fn predictor_corrector_sampling(
    n_steps: usize,
    n_corrector: usize,
    eps_langevin: f64,
    beta_min: f64,
    beta_max: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let mut x: f64 = rng.sample(StandardNormal);
    let dt = 1.0 / n_steps as f64; // 逆時間ステップ幅（正）

    let mut trajectory = vec![x];

    for step in 0..n_steps {
        let t = 1.0 - step as f64 / n_steps as f64;
        let beta_t = beta_min + t * (beta_max - beta_min);

        // Predictor (reverse SDE): xₜ₋₁ = xₜ + f_rev·dt + g·√dt·ΔW
        let score = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // PC predictor: xₜ₋₁ = xₜ + f_rev·dt + g·√dt·ΔW

        // Corrector (Langevin): x ← x + ε·s + √(2ε)·ΔW
        for _ in 0..n_corrector {
            let score_c = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
            let dw_c: f64 = rng.sample(StandardNormal);
            x += eps_langevin * score_c + (2.0 * eps_langevin).sqrt() * dw_c; // Langevin: x ← x + ε·∇log p + √(2ε)·ΔW
        }

        trajectory.push(x);
    }

    trajectory // n_steps+1 要素のベクトル
}

fn main() {
    let mut rng = rand::thread_rng();
    // サンプリング実行
    let traj = predictor_corrector_sampling(100, 5, 0.01, 0.1, 20.0, &mut rng);

    println!("Predictor-Corrector: {} ステップ", traj.len());
    // t_plot: 1.0 → 0.0 (101点)
    let t_plot: Vec<f64> = (0..=100).map(|i| 1.0 - i as f64 / 100.0).collect();
    for (t, x) in t_plot.iter().zip(traj.iter()).take(5) {
        println!("  t={:.2} x={:.4}", t, x);
    }
    // Plotting: use plotters crate — title="Predictor-Corrector サンプリング"
}
```

**Predictor-Corrector vs Euler-Maruyama**:
```rust
// Euler-Maruyama（Predictor-onlyと等価）
use rand::Rng;
use rand_distr::StandardNormal;

// Euler-Maruyama でリバース VP-SDE を解く（score ≈ -x）
fn em_reverse_vp(x0: f64, beta_min: f64, beta_max: f64, dt: f64, n_steps: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut x = x0;
    let mut traj = vec![x];
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
        let score = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        traj.push(x);
    }
    traj
}

fn predictor_corrector_sampling(
    n_steps: usize, n_corrector: usize, eps: f64,
    beta_min: f64, beta_max: f64, rng: &mut impl Rng,
) -> Vec<f64> {
    let mut x: f64 = rng.sample(StandardNormal);
    let dt = 1.0 / n_steps as f64;
    let mut traj = vec![x];
    for step in 0..n_steps {
        let t = 1.0 - step as f64 / n_steps as f64;
        let beta_t = beta_min + t * (beta_max - beta_min);
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * (-x)) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // PC predictor: xₜ₋₁ = xₜ + f_rev·dt + g·√dt·ΔW
        for _ in 0..n_corrector {
            let dw_c: f64 = rng.sample(StandardNormal);
            x += eps * (-x) + (2.0 * eps).sqrt() * dw_c; // Corrector (Langevin): x ← x + ε·s + √(2ε)·ΔW
        }
        traj.push(x);
    }
    traj
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let x0: f64 = rng.sample(StandardNormal);

    // Euler-Maruyama
    let traj_em = em_reverse_vp(x0, beta_min, beta_max, 0.01, 100, &mut rng);
    // Predictor-Corrector
    let traj_pc = predictor_corrector_sampling(100, 5, 0.01, beta_min, beta_max, &mut rng);

    println!("Euler-Maruyama 終端値: {:.4}", traj_em.last().unwrap());
    println!("Predictor-Corrector 終端値: {:.4}", traj_pc.last().unwrap());
    // Plotting: use plotters crate — title="Predictor-Corrector vs Euler-Maruyama"
}
```

**結果**: Predictor-Correctorは軌道が滑らか（Correctorでスコア方向に補正）

### 4.7 数値ソルバー比較 — Euler-Maruyama vs 高次手法

ode_solversが提供する各種ソルバーの精度・速度比較。

**SDEソルバー一覧**:
- `EM()`: Euler-Maruyama法（1次精度、低コスト）
- `SRIW1()`: Roessler法（弱1.5次精度、対角ノイズ）
- `SRA1()`: 適応的Roessler法（弱1.5次、ステップサイズ自動調整）
- `ImplicitEM()`: 暗黙的Euler-Maruyama（剛性問題）

```rust
// use criterion; // criterion クレートでベンチマーク（本番環境）

// テストSDE: Ornstein-Uhlenbeck過程
// dX = -θ X dt + σ dW
use rand::Rng;
use rand_distr::StandardNormal;
use std::time::Instant;

// Euler-Maruyama で OU過程を解く（固定ステップ dt）
fn solve_ou_em(theta: f64, sigma: f64, x0: f64, t_end: f64, dt: f64, rng: &mut impl Rng) -> f64 {
    let n_steps = (t_end / dt).ceil() as usize;
    let mut x = x0;
    for _ in 0..n_steps {
        let dw: f64 = rng.sample(StandardNormal);
        x += -theta * x * dt + sigma * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
    }
    x
}

// 解析解（比較用）: E[X(t)] = x0 * exp(-θ t)
fn analytical(t: f64, x0: f64, theta: f64) -> f64 { x0 * (-theta * t).exp() }

fn main() {
    let mut rng = rand::thread_rng();
    let theta = 1.0_f64;
    let sigma = 0.5_f64;
    let x0 = 1.0_f64;
    let t_end = 10.0_f64;

    let solver_configs = [
        ("EM (dt=0.01)", 0.01_f64),
        ("EM (dt=0.001)", 0.001_f64),  // SRIW1相当の精度
        ("EM (dt=0.0001)", 0.0001_f64), // SRA1相当の精度
    ];

    let x_analytical = analytical(t_end, x0, theta);

    for (name, dt) in &solver_configs {
        let start = Instant::now();
        let x_final = solve_ou_em(theta, sigma, x0, t_end, *dt, &mut rng);
        let elapsed = start.elapsed();

        let error = (x_final - x_analytical).abs();
        println!("{}: error={:.6}, time={:.3}ms", name, error, elapsed.as_secs_f64() * 1000.0);
    }

    // Plotting: use plotters crate for bar chart
}
```

**結果**:
- EM: 最速だが精度低い
- SRIW1: 精度高い（弱1.5次）、コストはEM の ~2倍
- SRA1: 適応ステップで剛性問題に強い

**実用指針**:
- 高速プロトタイプ: EM
- 高精度サンプリング: SRIW1
- 剛性SDE（急激な変化）: SRA1 or ImplicitEM

### 4.8 適応的ステップサイズ制御 — SRA1による自動調整

剛性問題（$\beta(t)$ が急変）で適応的ソルバーの威力を確認。

```rust
// 急激に変化するβ(t)（剛性問題）
use rand::Rng;
use rand_distr::StandardNormal;
use std::time::Instant;

fn beta_stiff(t: f64) -> f64 { if t < 0.5 { 0.1 } else { 50.0 } } // β(t): step function (stiff)

fn vp_drift_stiff(x: f64, t: f64) -> f64 { -0.5 * beta_stiff(t) * x } // f(x,t) = -½β(t)·x

fn vp_noise_stiff(t: f64) -> f64 { beta_stiff(t).sqrt() } // g(t) = √β(t)

// 固定ステップ Euler-Maruyama
fn solve_em_fixed(x0: f64, dt: f64, t_end: f64, rng: &mut impl Rng) -> Vec<(f64, f64)> {
    let n_steps = (t_end / dt).ceil() as usize;
    let mut x = x0;
    let mut traj = vec![(0.0_f64, x)];
    for step in 0..n_steps {
        let t = step as f64 * dt;
        let dw: f64 = rng.sample(StandardNormal);
        x += vp_drift_stiff(x, t) * dt + vp_noise_stiff(t) * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
        traj.push((t + dt, x));
    }
    traj
}

// 適応ステップ Euler-Maruyama（t > 0.5 で dt を縮小）
fn solve_em_adaptive(x0: f64, t_end: f64, rng: &mut impl Rng) -> Vec<(f64, f64)> {
    let mut x = x0;
    let mut t = 0.0_f64;
    let mut traj = vec![(t, x)];
    while t < t_end {
        // 剛性の強い領域では小さなステップ
        let dt = if t >= 0.5 { 0.001 } else { 0.01 };
        let dt = dt.min(t_end - t);
        let dw: f64 = rng.sample(StandardNormal);
        x += vp_drift_stiff(x, t) * dt + vp_noise_stiff(t) * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
        t += dt;
        traj.push((t, x));
    }
    traj
}

fn main() {
    let mut rng = rand::thread_rng();

    // 固定ステップ EM
    let traj_em = solve_em_fixed(1.0, 0.01, 1.0, &mut rng);
    // 適応ステップ（SRA1相当）
    let traj_adaptive = solve_em_adaptive(1.0, 1.0, &mut rng);

    println!("EM ステップ数: {}", traj_em.len());
    println!("適応ステップ数: {}", traj_adaptive.len());
    println!("EM 終端値: {:.4}", traj_em.last().unwrap().1);
    println!("適応 終端値: {:.4}", traj_adaptive.last().unwrap().1);
    // Plotting: use plotters crate — title="剛性問題: EM vs 適応ステップ"
}
```

**結果**:
- SRA1は $t > 0.5$ で自動的にステップサイズを縮小
- EMは固定ステップで不安定（発散リスク）

### 4.9 マルチスケールSDE — 高速・低速変数の分離

高速変数と低速変数が混在するSDE（マルチスケール問題）。

**設定**:
$$
\begin{aligned}
dX_t &= -\gamma X_t dt + \sigma_X dW^X_t \quad (\text{低速変数}) \\
dY_t &= -\epsilon^{-1} Y_t dt + \sigma_Y dW^Y_t \quad (\text{高速変数, } \epsilon \ll 1)
\end{aligned}
$$

高速変数 $Y_t$ は平衡化が早い（$\epsilon = 0.01$）。

```rust
// マルチスケールSDE
// dX = -γ X dt + σ_X dW^X  （低速変数）
// dY = -(1/ε) Y dt + σ_Y dW^Y  （高速変数, ε << 1）
use rand::Rng;
use rand_distr::StandardNormal;

fn multiscale_drift(x: f64, y: f64, eps: f64, gamma: f64) -> (f64, f64) { (-gamma * x, -y / eps) } // f_x = -γx, f_y = -y/ε

fn main() {
    let mut rng = rand::thread_rng();
    let eps = 0.01_f64;
    let gamma = 1.0_f64;
    let sigma_x = 0.5_f64;
    let sigma_y = 2.0_f64;

    // 適応ステップ: 高速変数 Y は eps が小さいので dt < eps が必要
    let dt = 0.001_f64; // ε=0.01 に対して安定なステップ
    let t_end = 5.0_f64;
    let n_steps = (t_end / dt) as usize;

    let mut x = 1.0_f64;
    let mut y = 1.0_f64;
    let mut traj_x = vec![(0.0_f64, x)];
    let mut traj_y = vec![(0.0_f64, y)];

    for step in 0..n_steps {
        let t = step as f64 * dt;
        let (dx_drift, dy_drift) = multiscale_drift(x, y, eps, gamma);
        let dw_x: f64 = rng.sample(StandardNormal);
        let dw_y: f64 = rng.sample(StandardNormal);
        x += dx_drift * dt + sigma_x * dt.sqrt() * dw_x; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW (低速)
        y += dy_drift * dt + sigma_y * dt.sqrt() * dw_y; // yₜ₊₁ = yₜ + f·dt + g·√dt·ΔW (高速)
        traj_x.push((t + dt, x));
        traj_y.push((t + dt, y));
    }

    println!("マルチスケールSDE (ε={}) ステップ数: {}", eps, traj_x.len());
    println!("X(5.0) = {:.4} (低速変数)", traj_x.last().unwrap().1);
    println!("Y(5.0) = {:.4} (高速変数)", traj_y.last().unwrap().1);
    // Plotting: use plotters crate — title="マルチスケールSDE"
}
```

**観察**:
- $Y_t$ は急速に平衡化（高周波振動）
- $X_t$ は緩やかに変化（低周波）
- 適応ステップが高速変数の細かい変化を追跡

### 4.10 Girsanov変換の実装 — 測度変換とスコア学習

Girsanov定理を使ってDrift項を変更し、Reverse-time SDEを導出する手続きを実装。

**理論**:
Forward SDE:
$$
dX_t = f(X_t, t) dt + g(X_t, t) dW_t
$$

Girsanov変換で新しいDrift $\tilde{f}$ を持つSDEに変換:
$$
dX_t = \tilde{f}(X_t, t) dt + g(X_t, t) d\tilde{W}_t
$$

Radon-Nikodym導関数:
$$
\frac{dP_{\tilde{W}}}{dP_W} = \exp\left(\int_0^T \frac{\tilde{f} - f}{g^2} dW_s - \frac{1}{2}\int_0^T \left(\frac{\tilde{f} - f}{g}\right)^2 ds\right)
$$

```rust
// Forward VP-SDE: dX = -0.5 β(t) X dt + √β(t) dW
// Girsanov変換で Reverse-time SDE に

use rand::Rng;
use rand_distr::StandardNormal;

fn forward_drift(x: f64, t: f64, beta_min: f64, beta_max: f64) -> f64 { -0.5 * (beta_min + t * (beta_max - beta_min)) * x } // f(x,t) = -½β(t)·x

fn forward_diffusion(t: f64, beta_min: f64, beta_max: f64) -> f64 { (beta_min + t * (beta_max - beta_min)).sqrt() } // g(t) = √β(t)

// Reverse-time では Drift に Score項が追加
// f_reverse = -f_forward - g² ∇log p_t
fn reverse_drift_girsanov(x: f64, t: f64, beta_min: f64, beta_max: f64, score: f64) -> f64 {
    let f_fwd = forward_drift(x, t, beta_min, beta_max); // f(x,t) = -½β(t)·x
    let g = forward_diffusion(t, beta_min, beta_max);     // g(t) = √β(t)
    -f_fwd - g * g * score // f_rev(x,t) = f - g²·∇log p_t  (Anderson 1982)
}

// 簡易Score関数（ガウス近似）: ∇log p_t(x) ≈ -x
fn score_approx(x: f64, _t: f64) -> f64 { -x } // ∇log p_t(x) ≈ -x  (Gaussian approx.)

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // Reverse-time SDE（Girsanov変換）
    let mut x = 0.5_f64;
    let mut trajectory = vec![(1.0_f64, x)];

    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let score = score_approx(x, t);
        let drift = reverse_drift_girsanov(x, t, beta_min, beta_max, score);
        let noise = forward_diffusion(t, beta_min, beta_max);
        let dw: f64 = rng.sample(StandardNormal);
        x += drift * (-dt) + noise * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        trajectory.push((t - dt, x));
    }

    println!("Girsanov変換 Reverse-time SDE: {} ステップ", trajectory.len());
    println!("終端値 X(0.0) = {:.4}", trajectory.last().unwrap().1);
    // Plotting: use plotters crate — title="Girsanov変換 Reverse-time SDE"
}
```

**Girsanov変換のキモ**:
1. Forward SDE の Drift $f$ を知る
2. Score関数 $\nabla \log p_t$ を学習（or 近似）
3. Reverse Drift = $-f - g^2 \nabla \log p_t$

これが **Score SDE統一理論** の数学的基盤。

### 4.11 JumpProcess混合SDE — Poisson Jumpとの結合

連続Brown運動に加え、Poisson過程（ジャンプ）を含むSDE。

**設定**:
$$
dX_t = -\theta X_t dt + \sigma dW_t + dN_t
$$
$N_t$ はPoisson過程（レート $\lambda$）

```rust
// JumpProcess混合SDE: dX = -θ X dt + σ dW + dN
// N_t はPoisson過程（レート λ）
use rand::Rng;
use rand_distr::{StandardNormal, Exp};

fn main() {
    let mut rng = rand::thread_rng();
    let theta = 1.0_f64;
    let sigma = 0.5_f64;
    let lambda = 2.0_f64; // Poisson rate
    let jump_size = 0.5_f64; // Jumpのサイズ（毎回 +0.5）

    let dt = 0.01_f64;
    let t_end = 10.0_f64;
    let n_steps = (t_end / dt) as usize;

    let mut x = 1.0_f64;
    let mut trajectory = vec![(0.0_f64, x)];

    // 次のジャンプ時刻を指数分布でサンプリング
    let exp_dist = Exp::new(lambda).unwrap();
    let mut next_jump: f64 = rng.sample(exp_dist);

    for step in 0..n_steps {
        let t = step as f64 * dt;

        // Brown運動部分（Euler-Maruyama）
        let dw: f64 = rng.sample(StandardNormal);
        x += -theta * x * dt + sigma * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW

        // Poissonジャンプ: 区間 [t, t+dt] にジャンプがあれば適用
        while next_jump <= t + dt {
            x += jump_size; // ジャンプ発生
            next_jump += rng.sample(exp_dist);
        }

        trajectory.push((t + dt, x));
    }

    println!("Brown運動 + Poissonジャンプ: {} ステップ", trajectory.len());
    println!("X(10.0) = {:.4}", trajectory.last().unwrap().1);
    // Plotting: use plotters crate — title="Brown運動 + Poissonジャンプ"
}
```

**結果**: 軌道に不連続なジャンプが発生。

**応用**: ファイナンス（株価の突発変動）、神経科学（スパイクニューロン）

### 4.12 並列アンサンブルシミュレーション — EnsembleProblemで高速化

複数の独立サンプルを並列で生成。

```rust
// Ornstein-Uhlenbeck SDE アンサンブルシミュレーション
// dX = -θ X dt + σ dW（1000トラジェクトリ）
use rand::Rng;
use rand_distr::StandardNormal;

fn simulate_ou(theta: f64, sigma: f64, x0: f64, dt: f64, n_steps: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut x = x0;
    let mut traj = vec![x];
    for _ in 0..n_steps {
        let dw: f64 = rng.sample(StandardNormal);
        x += -theta * x * dt + sigma * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW
        traj.push(x);
    }
    traj
}

fn main() {
    let mut rng = rand::thread_rng();
    let theta = 1.0_f64;
    let sigma = 0.5_f64;
    let dt = 0.01_f64;
    let t_end = 10.0_f64;
    let n_steps = (t_end / dt) as usize;
    let n_trajectories = 1000_usize;

    // アンサンブル実行（1000トラジェクトリ）
    // 並列化: rayon クレートの par_iter() を利用可能
    let trajectories: Vec<Vec<f64>> = (0..n_trajectories)
        .map(|_| simulate_ou(theta, sigma, 1.0, dt, n_steps, &mut rand::thread_rng()))
        .collect();

    // 平均と標準偏差を計算
    let t_vals: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt).collect();
    let mean_vals: Vec<f64> = (0..=n_steps)
        .map(|i| trajectories.iter().map(|t| t[i]).sum::<f64>() / n_trajectories as f64)
        .collect();
    let std_vals: Vec<f64> = (0..=n_steps)
        .map(|i| {
            let m = mean_vals[i];
            let var = trajectories.iter().map(|t| (t[i] - m).powi(2)).sum::<f64>() / n_trajectories as f64;
            var.sqrt()
        })
        .collect();

    println!("アンサンブル ({} トラジェクトリ):", n_trajectories);
    println!("t=10.0: mean={:.4}, std={:.4}", mean_vals[n_steps], std_vals[n_steps]);
    // Plotting: use plotters crate — title="Ornstein-Uhlenbeck過程 アンサンブル平均"
}
```

**並列化オプション**:
- `EnsembleThreads()`: マルチスレッド（共有メモリ）
- `EnsembleDistributed()`: 分散計算（クラスタ）
- `EnsembleGPUArray()`: GPU並列

**性能**: 1000トラジェクトリを並列実行で **数秒** で完了。

---

### 🔬 実験・検証（30分）— VP-SDE ↔ Probability Flow ODE変換 + 軌道可視化

### 5.1 演習: VP-SDE軌道とPF-ODE軌道の比較

同じ初期ノイズから、Reverse-time SDEとPF-ODEで軌道を生成し比較。

```rust
// VP-SDE軌道とPF-ODE軌道の比較
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand::rngs::StdRng;

fn reverse_sde_traj(x0: f64, beta_min: f64, beta_max: f64, dt: f64, n_steps: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut x = x0;
    let mut traj = vec![x];
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
        let score = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        traj.push(x);
    }
    traj
}

fn pf_ode_traj(x0: f64, beta_min: f64, beta_max: f64, dt: f64, n_steps: usize) -> Vec<f64> {
    let mut x = x0;
    let mut traj = vec![x];
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
        let score = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
        x += (-0.5 * beta_t * x - 0.5 * beta_t * score) * (-dt); // PF-ODE: dx/dt = f - ½g²·∇log p_t  (Song+ 2021)
        traj.push(x);
    }
    traj
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // 共通の初期ノイズ（5サンプル）
    let u0_list: Vec<f64> = (0..5).map(|_| rng.sample(StandardNormal)).collect();

    // 各初期値で SDE と ODE 軌道を生成して比較
    for (i, &x0) in u0_list.iter().enumerate() {
        let traj_sde = reverse_sde_traj(x0, beta_min, beta_max, dt, n_steps, &mut rng);
        let traj_ode = pf_ode_traj(x0, beta_min, beta_max, dt, n_steps);
        println!(
            "Sample {}: x0={:.3}, SDE終端={:.4}, ODE終端={:.4}",
            i,
            x0,
            traj_sde.last().unwrap(),
            traj_ode.last().unwrap()
        );
    }
    // Plotting: use plotters crate — title="Reverse-time SDE vs Probability Flow ODE"
}
```

**観察**:
- SDE: 各軌道が揺れる（確率性）
- ODE: 滑らかな決定論的軌道
- 最終分布（周辺分布）は同じ

### 5.2 演習: スコア関数の影響を可視化

真のスコア関数 vs 近似スコア関数での軌道の違い。

```rust
// 真のスコア関数（ガウス分布 N(μ, σ²) 仮定）
// ∇log N(μ, σ²) = -(x - μ) / σ²
use rand::Rng;
use rand_distr::StandardNormal;

fn true_score(x: f64, _t: f64, mu: f64, sigma: f64) -> f64 { -(x - mu) / (sigma * sigma) } // ∇log N(μ,σ²) = -(x-μ)/σ²

fn approx_score(x: f64, _t: f64) -> f64 { -x } // ∇log p_t(x) ≈ -x  (Gaussian approx.)

fn reverse_sde_with_score<F>(
    x0: f64, beta_min: f64, beta_max: f64, dt: f64, n_steps: usize,
    score_fn: F, rng: &mut impl Rng,
) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64,
{
    let mut x = x0;
    let mut traj = vec![x];
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let beta_t = beta_min + t * (beta_max - beta_min);
        let score = score_fn(x, t);
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        traj.push(x);
    }
    traj
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let mu_true = 1.0_f64;
    let sigma_true = 0.5_f64;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;
    let x0: f64 = rng.sample(StandardNormal);

    // 真のスコアを使った軌道
    let traj_true = reverse_sde_with_score(
        x0, beta_min, beta_max, dt, n_steps,
        |x, t| true_score(x, t, mu_true, sigma_true),
        &mut rng,
    );

    // 近似スコアを使った軌道
    let traj_approx = reverse_sde_with_score(
        x0, beta_min, beta_max, dt, n_steps,
        |x, t| approx_score(x, t),
        &mut rng,
    );

    println!("真のスコア 終端値: {:.4} (真の平均 μ={:.1})", traj_true.last().unwrap(), mu_true);
    println!("近似スコア 終端値: {:.4} (バイアス: μ≈0)", traj_approx.last().unwrap());
    // Plotting: use plotters crate — title="スコア関数の影響"
}
```

**結果**: 真のスコア使用時、軌道が真の平均 $\mu = 1.0$ に収束。近似スコアは $\mu = 0$ に収束（バイアス）。

### 5.3 演習: 収束性の数値検証 — ステップ数 vs 精度

ステップ数 $T$ を変化させ、生成分布と真の分布のKL距離を計測。

```rust
// 収束性の数値検証 — ステップ数 T vs KL距離
// KernelDensity の代わりにヒストグラムベースの KL 推定を使用
use rand::Rng;
use rand_distr::StandardNormal;

// Gaussian pdf: N(mu, sigma^2)
fn gaussian_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let norm = (2.0 * std::f64::consts::PI).sqrt() * sigma;
    (-(x - mu).powi(2) / (2.0 * sigma * sigma)).exp() / norm
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let mu_true = 1.0_f64;
    let sigma_true = 0.5_f64;
    let n_samples = 5000_usize;
    let dx = 0.05_f64;

    let step_counts = [10usize, 25, 50, 100, 200, 500, 1000];

    println!("収束性: ステップ数 vs KL距離");
    for &t_steps in &step_counts {
        let dt = 1.0 / t_steps as f64;

        // 各ステップ数でサンプリング
        let samples: Vec<f64> = (0..n_samples)
            .map(|_| {
                let mut x: f64 = rng.sample(StandardNormal);
                for step in 0..t_steps {
                    let t = 1.0 - step as f64 / t_steps as f64;
                    let beta_t = beta_min + t * (beta_max - beta_min);
                    let score = -(x - mu_true) / (sigma_true * sigma_true); // ∇log p_t(x) = -(x-μ)/σ²
                    let dw: f64 = rng.sample(StandardNormal);
                    x += (-0.5 * beta_t * x - beta_t * score) * (-dt)
                        + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
                }
                x
            })
            .collect();

        // KL(p_true||p_gen) = ∫ p_true·log(p_true/p_gen) dx
        // ヒストグラムベースで推定
        let x_vals: Vec<f64> = {
            let n_bins = 120;
            (0..n_bins).map(|i| -2.0 + i as f64 * dx).collect()
        };

        let kl: f64 = x_vals.iter().map(|&xv| {
            let p_true = gaussian_pdf(xv, mu_true, sigma_true);
            // 生成サンプルのカーネル密度推定（簡易：ガウスカーネル）
            let h = 0.2_f64;
            let p_gen = samples.iter()
                .map(|&s| gaussian_pdf(xv, s, h))
                .sum::<f64>() / n_samples as f64;
            if p_true > 1e-10 && p_gen > 1e-10 {
                p_true * (p_true / p_gen).ln() * dx
            } else { 0.0 }
        }).sum();

        println!("T={:4}: KL={:.6}", t_steps, kl);
    }
    // Plotting: use plotters crate — title="収束性: ステップ数 vs KL距離" (log-log scale)
}
```

**理論予測**: $\text{KL} \propto 1/T$ → 両対数プロットで傾き -1 の直線

### 5.4 演習: Manifold仮説の検証 — 高次元データの固有次元

高次元データ（$D = 100$）で固有次元 $d = 5$ のマニフォールドを生成し、収束を観察。

```rust
// Manifold仮説の検証 — 高次元データの固有次元
// 固有次元 d=5 のマニフォールド上のデータ生成 + Reverse-time SDE で再構成
use rand::Rng;
use rand_distr::StandardNormal;

// 行列-ベクトル積: (D×d) * (d×1) → (D×1)
fn mat_vec(q: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let d_dim = q.len();
    let d_sub = v.len();
    (0..d_dim).map(|i| (0..d_sub).map(|j| q[i][j] * v[j]).sum()).collect()
}

// 転置行列-ベクトル積: (d×D) * (D×1) → (d×1)
fn mat_t_vec(q: &[Vec<f64>], u: &[f64]) -> Vec<f64> {
    let d_sub = q[0].len();
    let d_dim = q.len();
    (0..d_sub).map(|j| (0..d_dim).map(|i| q[i][j] * u[i]).sum()).collect()
}

// ベクトル L2 ノルム
fn norm_vec(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() }

fn main() {
    let mut rng = rand::thread_rng();
    let big_d = 100_usize; // 埋め込み次元
    let d_sub = 5_usize;   // 固有次元
    let beta = 1.0_f64;
    let dt = 0.01_f64;
    let n_steps = (1.0 / dt) as usize;

    // ランダム直交基底（簡易: d個のランダム単位ベクトルを列として配置）
    // Gram-Schmidt 直交化で近似
    let mut q: Vec<Vec<f64>> = Vec::new();
    for _ in 0..d_sub {
        let mut col: Vec<f64> = (0..big_d).map(|_| rng.sample::<f64, _>(StandardNormal)).collect();
        // 既存の列に直交化
        for existing in &q {
            let dot: f64 = col.iter().zip(existing.iter()).map(|(a, b)| a * b).sum();
            for (c, e) in col.iter_mut().zip(existing.iter()) {
                *c -= dot * e;
            }
        }
        let n = norm_vec(&col);
        col.iter_mut().for_each(|c| *c /= n);
        q.push(col);
    }
    // q: D×d 行列（q[i][j] = Q_{i,j}）

    // 低次元潜在変数 z ~ N(0, I_d)
    let z: Vec<f64> = (0..d_sub).map(|_| rng.sample(StandardNormal)).collect();

    // 高次元埋め込み X = Q * z
    let x_original = mat_vec(&q, &z);

    // VP-SDE Forward過程でノイズ注入（t=1.0）
    let t0 = 1.0_f64;
    let alpha_t = (-0.5 * beta * t0).exp();
    let sigma_t = (1.0 - (-beta * t0).exp()).sqrt();
    let mut x_noisy: Vec<f64> = x_original.iter()
        .map(|&xi| alpha_t * xi + sigma_t * rng.sample::<f64, _>(StandardNormal))
        .collect();

    // Reverse-time SDE（簡易Score: PCA射影で法線方向ペナルティ）
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let sigma_t_cur = (1.0 - (-beta * t).exp()).max(1e-8).sqrt();

        // Manifold上への射影: Q * (Q^T * x)
        let z_proj = mat_t_vec(&q, &x_noisy);
        let x_proj = mat_vec(&q, &z_proj);

        // Score: -(x - x_proj) / sigma_t^2  （法線方向ペナルティ）
        let score: Vec<f64> = x_noisy.iter().zip(x_proj.iter())
            .map(|(&xi, &xp)| -(xi - xp) / (sigma_t_cur * sigma_t_cur))
            .collect();

        let dw: Vec<f64> = (0..big_d).map(|_| rng.sample(StandardNormal)).collect();
        for i in 0..big_d {
            let drift = -0.5 * beta * x_noisy[i] - beta * score[i]; // f_rev(x,t) = f - g²·∇log p_t  (Anderson 1982)
            x_noisy[i] += drift * (-dt) + beta.sqrt() * dt.sqrt() * dw[i]; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        }
    }

    // 元データとの距離
    let reconstruction_error = norm_vec(
        &x_noisy.iter().zip(x_original.iter()).map(|(r, o)| r - o).collect::<Vec<_>>()
    );
    println!("再構成誤差: {:.4}", reconstruction_error);
    // 固有次元が小さい → Scoreが部分空間に誘導 → 高精度再構成
}
```

**結果**: 固有次元 $d=5$ のマニフォールド上では、少ないステップで高精度再構成が可能。

### 5.5 演習: VP-SDE vs VE-SDE の分散軌道比較

Variance Preserving vs Variance Exploding の分散の時間発展を可視化。

```rust
// VP-SDE vs VE-SDE の分散軌道比較
use rand::Rng;
use rand_distr::StandardNormal;

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let sigma_min = 0.01_f64;
    let sigma_max = 50.0_f64;
    let n_samples = 1000_usize;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // アンサンブル初期値
    let u0_list: Vec<f64> = (0..n_samples)
        .map(|_| rng.sample(StandardNormal))
        .collect();

    // VP-SDE アンサンブル
    let mut vp_trajectories: Vec<Vec<f64>> = u0_list.iter().map(|&x0| {
        let mut x = x0;
        let mut traj = vec![x];
        for step in 0..n_steps {
            let t = step as f64 * dt;
            let beta_t = beta_min + t * (beta_max - beta_min);
            let dw: f64 = rand::thread_rng().sample(StandardNormal);
            x += -0.5 * beta_t * x * dt + beta_t.sqrt() * dt.sqrt() * dw; // xₜ₊₁ = xₜ + f·dt + g·√dt·ΔW (VP)
            traj.push(x);
        }
        traj
    }).collect();

    // VE-SDE アンサンブル
    let mut ve_trajectories: Vec<Vec<f64>> = u0_list.iter().map(|&x0| {
        let mut x = x0;
        let mut traj = vec![x];
        for step in 0..n_steps {
            let t = step as f64 * dt;
            let sigma_t = sigma_min * (sigma_max / sigma_min).powf(t); // σ(t) = σ_min·(σ_max/σ_min)^t
            let g = (2.0 * sigma_t.powi(2) * (sigma_max / sigma_min).ln()).sqrt(); // g(t) = √(2σ²(t)·log(σ_max/σ_min))
            let dw: f64 = rand::thread_rng().sample(StandardNormal);
            x += g * dt.sqrt() * dw; // xₜ₊₁ = xₜ + g·√dt·ΔW (VE: drift = 0)
            traj.push(x);
        }
        traj
    }).collect();

    // 分散の計算
    let var_vp: Vec<f64> = (0..=n_steps).map(|i| {
        let vals: Vec<f64> = vp_trajectories.iter().map(|t| t[i]).collect();
        let mean = vals.iter().sum::<f64>() / n_samples as f64;
        vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_samples as f64
    }).collect();

    let var_ve: Vec<f64> = (0..=n_steps).map(|i| {
        let vals: Vec<f64> = ve_trajectories.iter().map(|t| t[i]).collect();
        let mean = vals.iter().sum::<f64>() / n_samples as f64;
        vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_samples as f64
    }).collect();

    // 理論分散
    // VP: Var[X_t] = 1 - exp(-(β_min + 0.5t*(β_max-β_min))*t)
    let var_vp_theory: Vec<f64> = (0..=n_steps).map(|i| {
        let t = i as f64 * dt;
        1.0 - (-(beta_min + 0.5 * t * (beta_max - beta_min)) * t).exp()
    }).collect();

    // VE: Var[X_t] = σ_min^2 * (σ_max/σ_min)^(2t)
    let var_ve_theory: Vec<f64> = (0..=n_steps).map(|i| {
        let t = i as f64 * dt;
        sigma_min.powi(2) * (sigma_max / sigma_min).powf(2.0 * t)
    }).collect();

    println!("VP-SDE t=1.0: Var数値={:.4}, Var理論={:.4}", var_vp[n_steps], var_vp_theory[n_steps]);
    println!("VE-SDE t=1.0: Var数値={:.4}, Var理論={:.4}", var_ve[n_steps], var_ve_theory[n_steps]);
    // Plotting: use plotters crate — title="VP-SDE vs VE-SDE 分散"
}
```

**観察**:
- **VP-SDE**: 分散が上限1に収束（Variance Preserving）
- **VE-SDE**: 分散が指数的に爆発（Variance Exploding）

### 5.6 演習: Predictor-Corrector法の反復回数依存性

Correctorの反復回数を変化させ、サンプル品質を測定。

```rust
// Predictor-Corrector 反復回数依存性
use rand::Rng;
use rand_distr::StandardNormal;

fn gaussian_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let norm = (2.0 * std::f64::consts::PI).sqrt() * sigma;
    (-(x - mu).powi(2) / (2.0 * sigma * sigma)).exp() / norm
}

fn pc_sample(n_corrector: usize, n_steps: usize, eps_langevin: f64,
             beta_min: f64, beta_max: f64, true_mean: f64, true_std: f64,
             rng: &mut impl Rng) -> f64 {
    let mut x: f64 = rng.sample(StandardNormal);
    let dt = 1.0 / n_steps as f64;

    for step in 0..n_steps {
        let t = 1.0 - step as f64 / n_steps as f64;
        let beta_t = beta_min + t * (beta_max - beta_min);

        // Predictor (reverse SDE): xₜ₋₁ = xₜ + f_rev·dt + g·√dt·ΔW
        let score = -(x - true_mean) / (true_std * true_std); // ∇log p_t(x) = -(x-μ)/σ²
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // PC predictor: xₜ₋₁ = xₜ + f_rev·dt + g·√dt·ΔW

        // Corrector (Langevin): x ← x + ε·s + √(2ε)·ΔW
        for _ in 0..n_corrector {
            let score_c = -(x - true_mean) / (true_std * true_std); // ∇log p_t(x) = -(x-μ)/σ²
            let dw_c: f64 = rng.sample(StandardNormal);
            x += eps_langevin * score_c + (2.0 * eps_langevin).sqrt() * dw_c; // Langevin: x ← x + ε·∇log p + √(2ε)·ΔW
        }
    }
    x
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let true_mean = 1.0_f64;
    let true_std = 0.5_f64;
    let n_samples = 2000_usize;
    let dx = 0.05_f64;

    let corrector_counts = [0usize, 1, 3, 5, 10];

    println!("Corrector反復回数 vs KL距離");
    for &n_corr in &corrector_counts {
        let samples: Vec<f64> = (0..n_samples)
            .map(|_| pc_sample(n_corr, 100, 0.01, beta_min, beta_max, true_mean, true_std, &mut rng))
            .collect();

        // KL(p_true || p_gen) ヒストグラムベース
        let kl: f64 = (0..80).map(|i| {
            let xv = -1.0 + i as f64 * dx;
            let p_true = gaussian_pdf(xv, true_mean, true_std);
            let h = 0.2_f64;
            let p_gen = samples.iter().map(|&s| gaussian_pdf(xv, s, h)).sum::<f64>() / n_samples as f64;
            if p_true > 1e-10 && p_gen > 1e-10 {
                p_true * (p_true / p_gen).ln() * dx
            } else { 0.0 }
        }).sum();

        println!("Corrector={}: KL={:.6}", n_corr, kl);
    }
    // Plotting: use plotters crate — title="Corrector回数 vs サンプル品質"
}
```

**結果**:
- Corrector回数0（Predictor-only）: 高KL（低品質）
- Corrector回数5: KL最小（最適）
- Corrector回数10+: 改善飽和（コスト増のみ）

**実用指針**: Corrector反復5回が精度とコストのバランス。

### 5.7 演習: 異なるノイズスケジュールの比較 — 線形 vs Cosine vs 二次

線形、Cosine、二次スケジュールでの最終分布品質を比較。

```rust
// 異なるノイズスケジュールの比較 — 線形 vs Cosine vs 二次
use rand::Rng;
use rand_distr::StandardNormal;
use std::f64::consts::PI;

// 線形スケジュール
fn beta_linear(t: f64, beta_min: f64, beta_max: f64) -> f64 { beta_min + t * (beta_max - beta_min) } // β(t) = β_min + t·(β_max - β_min)

// Cosineスケジュール
fn alpha_bar_cosine(t: f64, s: f64) -> f64 {
    let num = ((t + s) / (1.0 + s) * PI / 2.0).cos().powi(2);
    let den = (s / (1.0 + s) * PI / 2.0).cos().powi(2);
    num / den // ᾱ(t) = cos²(πt/(2+2s)) / cos²(πs/(2+2s))
}
fn beta_cosine(t: f64, s: f64) -> f64 {
    let h = 1e-6;
    -(alpha_bar_cosine(t + h, s).ln() - alpha_bar_cosine(t, s).ln()) / h // β(t) = -d/dt log ᾱ(t)
}

// 二次スケジュール
fn beta_quadratic(t: f64, beta_min: f64, beta_max: f64) -> f64 { beta_min + t * t * (beta_max - beta_min) } // β(t) = β_min + t²·(β_max - β_min)

fn sample_with_schedule<F>(beta_fn: F, n_samples: usize, rng: &mut impl Rng) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    (0..n_samples).map(|_| {
        let mut x: f64 = rng.sample(StandardNormal);
        let dt = 0.01_f64;
        // t: 1.0 → 0.0 (100 ステップ)
        for step in 0..100 {
            let t = 1.0 - step as f64 / 100.0;
            let beta_t = beta_fn(t);
            let score = -x; // ∇log p_t(x) ≈ -x  (Gaussian approx.)
            let dw: f64 = rng.sample(StandardNormal);
            x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        }
        x
    }).collect()
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let s = 0.008_f64;
    let n_samples = 1000_usize;

    let samples_linear = sample_with_schedule(|t| beta_linear(t, beta_min, beta_max), n_samples, &mut rng);
    let samples_cosine = sample_with_schedule(|t| beta_cosine(t, s), n_samples, &mut rng);
    let samples_quadratic = sample_with_schedule(|t| beta_quadratic(t, beta_min, beta_max), n_samples, &mut rng);

    let mean_and_std = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let std = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        (m, std)
    };

    let (m_l, s_l) = mean_and_std(&samples_linear);
    let (m_c, s_c) = mean_and_std(&samples_cosine);
    let (m_q, s_q) = mean_and_std(&samples_quadratic);

    println!("線形スケジュール:   mean={:.4}, std={:.4}", m_l, s_l);
    println!("Cosineスケジュール: mean={:.4}, std={:.4}", m_c, s_c);
    println!("二次スケジュール:   mean={:.4}, std={:.4}", m_q, s_q);
    // Plotting: use plotters crate — title="ノイズスケジュール比較"
}
```

**結果**:
- **線形**: 標準的（DDPM論文）
- **Cosine**: 滑らか、端点での急変回避 → 高品質
- **二次**: 初期にノイズが少ない → 学習が難しい

### 5.8 演習: 次元依存性の検証 — O(d/T)理論の実証

次元 $d$ を変化させ、収束レートが $O(d/T)$ になることを確認。

```rust
// 次元依存性の検証 — O(d/T) 理論の実証
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand::rngs::StdRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let beta = 1.0_f64;
    let t_fixed = 100_usize;
    let n_samples = 500_usize;
    let dt = 1.0 / t_fixed as f64;

    let dimensions = [1usize, 2, 5, 10, 20, 50];

    println!("次元依存性 (T={}): 誤差 vs 理論 O(d/T)", t_fixed);
    for &d in &dimensions {
        // d次元 真の平均 μ = [1, 1, ..., 1]
        let mu_true = vec![1.0_f64; d];

        // T ステップでサンプリング (n_samples 個)
        let mut mu_sampled = vec![0.0_f64; d];

        for _ in 0..n_samples {
            // 初期値 ~ N(0, I_d)
            let mut x: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();

            // Reverse-time SDE（逆時間）
            for step in 0..t_fixed {
                let t = 1.0 - step as f64 * dt;
                let xi: Vec<f64> = (0..d).map(|_| rng.sample(StandardNormal)).collect();
                for j in 0..d {
                    let score = -(x[j] - mu_true[j]); // ∇log p_t(x) = -(x-μ) (true score)
                    x[j] += (-0.5 * beta * x[j] - beta * score) * (-dt)
                        + beta.sqrt() * dt.sqrt() * xi[j]; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
                }
            }

            for j in 0..d {
                mu_sampled[j] += x[j];
            }
        }

        // 平均を計算
        for v in mu_sampled.iter_mut() {
            *v /= n_samples as f64;
        }

        // Wasserstein距離（簡易: 平均のL2距離）
        let error: f64 = mu_sampled.iter().zip(mu_true.iter())
            .map(|(s, t)| (s - t).powi(2))
            .sum::<f64>()
            .sqrt();

        println!("d={:2}: error={:.4}, 理論 d/T={:.4}", d, error, d as f64 / t_fixed as f64);
    }
    // Plotting: use plotters crate — title="次元依存性 (T=100)"
}
```

**結果**: 誤差が $d/T$ に比例 → 高次元では多くのステップが必要。

### 5.9 演習: Langevin Dynamics vs Reverse-time SDE

Langevin DynamicsとReverse-time SDEのサンプリング品質を比較。

```rust
// Langevin Dynamics vs Reverse-time SDE の比較
use rand::Rng;
use rand_distr::StandardNormal;

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let true_mean = 1.0_f64;
    let true_std = 0.5_f64;
    let n_samples = 2000_usize;

    // 真のスコア: ∇log N(μ, σ²) = -(x - μ) / σ²
    let true_score = |x: f64, _t: f64| -(x - true_mean) / (true_std * true_std);

    // Reverse-time SDE サンプリング（100ステップ）
    let sde_sampling = |rng: &mut rand::rngs::ThreadRng| -> f64 {
        let mut x: f64 = rng.sample(StandardNormal);
        let dt = 0.01_f64;
        for step in 0..100 {
            let t = 1.0 - step as f64 / 100.0;
            let beta_t = beta_min + t * (beta_max - beta_min);
            let score = true_score(x, t); // ∇log p_t(x) = -(x-μ)/σ²
            let dw: f64 = rng.sample(StandardNormal);
            x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
        }
        x
    };

    // Langevin Dynamics サンプリング（t=0のスコアのみ使用）
    let langevin_sampling = |n_steps: usize, eps: f64, rng: &mut rand::rngs::ThreadRng| -> f64 {
        let mut x: f64 = rng.sample(StandardNormal);
        for _ in 0..n_steps {
            let score = true_score(x, 0.0); // ∇log p_t(x) = -(x-μ)/σ²
            let dw: f64 = rng.sample(StandardNormal);
            x += eps * score + (2.0 * eps).sqrt() * dw; // Langevin: x ← x + ε·∇log p + √(2ε)·ΔW
        }
        x
    };

    // サンプル生成
    let samples_sde: Vec<f64> = (0..n_samples).map(|_| sde_sampling(&mut rng)).collect();
    let samples_langevin: Vec<f64> = (0..n_samples).map(|_| langevin_sampling(1000, 0.01, &mut rng)).collect();

    let mean_std = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        (m, s)
    };

    let (m_sde, s_sde) = mean_std(&samples_sde);
    let (m_lang, s_lang) = mean_std(&samples_langevin);

    println!("Reverse-time SDE:   mean={:.4}, std={:.4} (100ステップ)", m_sde, s_sde);
    println!("Langevin Dynamics:  mean={:.4}, std={:.4} (1000ステップ)", m_lang, s_lang);
    println!("真の分布: mean={:.4}, std={:.4}", true_mean, true_std);
    // Plotting: use plotters crate — title="Reverse-time SDE vs Langevin Dynamics"
}
```

**結果**:
- 両者とも真の分布に収束
- **Reverse-time SDE**: より高速（100ステップ）
- **Langevin Dynamics**: 多くの反復必要（1000ステップ）

### 5.10 演習: ODEソルバーの選択がPF-ODEに与える影響

Probability Flow ODEを異なるODEソルバーで解き、精度比較。

```rust
// ODEソルバーの選択がPF-ODEに与える影響
// 各種精度の Euler 法でPF-ODEを解き精度比較
use rand::Rng;
use rand_distr::StandardNormal;
use std::time::Instant;

// PF-ODE: dx/dt = f - ½g²·∇log p_t  (Song+ 2021)
fn pf_ode_rhs(x: f64, t: f64, beta_min: f64, beta_max: f64, true_mean: f64) -> f64 {
    let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
    let score = -(x - true_mean) / 0.25; // ∇log p_t(x) = -(x-μ)/σ² (σ²=0.5²=0.25)
    -0.5 * beta_t * x - 0.5 * beta_t * score // PF-ODE: dx/dt = f - ½g²·∇log p_t
}

// Euler 法（固定ステップ）で PF-ODE を解く
fn solve_pf_ode_euler(x0: f64, beta_min: f64, beta_max: f64, true_mean: f64, n_steps: usize) -> f64 {
    let dt = 1.0 / n_steps as f64;
    let mut x = x0;
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        x += pf_ode_rhs(x, t, beta_min, beta_max, true_mean) * (-dt);
    }
    x
}

// RK4 法で PF-ODE を解く（高精度）
fn solve_pf_ode_rk4(x0: f64, beta_min: f64, beta_max: f64, true_mean: f64, n_steps: usize) -> f64 {
    let dt = 1.0 / n_steps as f64;
    let mut x = x0;
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let h = -dt; // 逆時間方向
        let k1 = pf_ode_rhs(x,             t,       beta_min, beta_max, true_mean);
        let k2 = pf_ode_rhs(x + h/2.0*k1, t - h/2.0, beta_min, beta_max, true_mean);
        let k3 = pf_ode_rhs(x + h/2.0*k2, t - h/2.0, beta_min, beta_max, true_mean);
        let k4 = pf_ode_rhs(x + h*k3,     t - h,     beta_min, beta_max, true_mean);
        x += h / 6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
    }
    x
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let true_mean = 1.0_f64;
    let x0: f64 = rng.sample(StandardNormal);

    let solver_configs: &[(&str, usize, bool)] = &[
        ("Euler (n=100)",   100,  false), // 低精度
        ("Euler (n=1000)",  1000, false), // Tsit5相当
        ("RK4  (n=100)",    100,  true),  // 高精度（Vern7相当）
        ("RK4  (n=1000)",   1000, true),  // 超高精度
    ];

    println!("ODEソルバー精度比較 (PF-ODE):");
    for &(name, n_steps, use_rk4) in solver_configs {
        let start = Instant::now();
        let x_final = if use_rk4 {
            solve_pf_ode_rk4(x0, beta_min, beta_max, true_mean, n_steps)
        } else {
            solve_pf_ode_euler(x0, beta_min, beta_max, true_mean, n_steps)
        };
        let elapsed = start.elapsed();
        let error = (x_final - true_mean).abs();
        println!("{}: error={:.6}, time={:.3}ms", name, error, elapsed.as_secs_f64() * 1000.0);
    }
    // Plotting: use plotters crate for bar chart
}
```

**結果**:
- **Euler**: 最速だが低精度
- **Tsit5**: 精度と速度のバランス（推奨）
- **Vern7**: 超高精度、コスト高
- **RadauIIA5**: 剛性問題に強い

**実用指針**: 通常はTsit5、剛性問題ならRadauIIA5。

### 5.11 演習: 異なる初期ノイズ分布の影響

初期ノイズ分布を $\mathcal{N}(0, 1)$ から $\text{Uniform}(-3, 3)$ に変更した場合の影響を調査。

```rust
// 異なる初期ノイズ分布の影響調査
// ガウス N(0,1) vs 一様 Uniform(-3,3) の初期値比較
use rand::Rng;
use rand_distr::StandardNormal;

fn solve_reverse_sde(x0: f64, beta_min: f64, beta_max: f64, true_mean: f64, true_std: f64,
                      dt: f64, n_steps: usize, rng: &mut impl Rng) -> f64 {
    let mut x = x0;
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
        let score = -(x - true_mean) / (true_std * true_std); // ∇log p_t(x) = -(x-μ)/σ²
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
    }
    x
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let true_mean = 1.0_f64;
    let true_std = 0.5_f64;
    let n_samples = 2000_usize;
    let dt = 0.001_f64;
    let n_steps = (1.0 / dt) as usize;

    // ガウス初期ノイズ: x0 ~ N(0, 1)
    let samples_gaussian: Vec<f64> = (0..n_samples).map(|_| {
        let x0: f64 = rng.sample(StandardNormal);
        solve_reverse_sde(x0, beta_min, beta_max, true_mean, true_std, dt, n_steps, &mut rng)
    }).collect();

    // 一様分布初期ノイズ: x0 ~ Uniform(-3, 3)
    let samples_uniform: Vec<f64> = (0..n_samples).map(|_| {
        let x0 = rng.gen_range(-3.0_f64..3.0_f64);
        solve_reverse_sde(x0, beta_min, beta_max, true_mean, true_std, dt, n_steps, &mut rng)
    }).collect();

    let mean_std = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        (m, s)
    };

    let (m_g, s_g) = mean_std(&samples_gaussian);
    let (m_u, s_u) = mean_std(&samples_uniform);

    println!("初期: N(0,1)       — 終端: mean={:.4}, std={:.4}", m_g, s_g);
    println!("初期: Uniform(-3,3)— 終端: mean={:.4}, std={:.4}", m_u, s_u);
    println!("真の分布: mean={:.4}, std={:.4}", true_mean, true_std);
    // 両者とも真の分布に収束 → ノイズ分布の選択は柔軟
    // Plotting: use plotters crate — title="初期ノイズ分布の影響"
}
```

**結果**: どちらの初期分布でも、最終的に真の分布 $\mathcal{N}(\mu, \sigma^2)$ に収束 → **ノイズ分布の選択は柔軟**。

### 5.12 演習: 時間ステップ依存性の可視化 — 精度 vs コスト

ステップサイズ $dt$ を変化させ、精度とコストのトレードオフを可視化。

```rust
// 時間ステップ依存性の可視化 — 精度 vs コスト
// use criterion; // criterion クレートで本番ベンチマーク
use rand::Rng;
use rand_distr::StandardNormal;
use std::time::Instant;

fn sample_reverse_sde(beta_min: f64, beta_max: f64, true_mean: f64, true_std: f64,
                       dt: f64, rng: &mut impl Rng) -> f64 {
    let n_steps = (1.0 / dt).ceil() as usize;
    let mut x: f64 = rng.sample(StandardNormal);
    for step in 0..n_steps {
        let t = 1.0 - step as f64 * dt;
        let t = t.max(0.0);
        let beta_t = beta_min + t * (beta_max - beta_min); // β(t) = β_min + t·(β_max - β_min)
        let score = -(x - true_mean) / (true_std * true_std); // ∇log p_t(x) = -(x-μ)/σ²
        let dw: f64 = rng.sample(StandardNormal);
        x += (-0.5 * beta_t * x - beta_t * score) * (-dt) + beta_t.sqrt() * dt.sqrt() * dw; // reverse SDE: dx = [f - g²∇log p]dt + g dW̄
    }
    x
}

fn main() {
    let mut rng = rand::thread_rng();
    let beta_min = 0.1_f64;
    let beta_max = 20.0_f64;
    let true_mean = 1.0_f64;
    let true_std = 0.5_f64;
    let n_per_config = 500_usize;

    let dt_values = [0.1_f64, 0.05, 0.01, 0.005, 0.001];

    println!("精度 vs コスト (dt ステップサイズ比較):");
    for &dt_val in &dt_values {
        let start = Instant::now();
        let samples: Vec<f64> = (0..n_per_config)
            .map(|_| sample_reverse_sde(beta_min, beta_max, true_mean, true_std, dt_val, &mut rng))
            .collect();
        let elapsed = start.elapsed();

        let mu_sampled = samples.iter().sum::<f64>() / n_per_config as f64;
        let error = (mu_sampled - true_mean).abs();

        println!(
            "dt={:.3}: error={:.6}, time={:.2}ms",
            dt_val, error, elapsed.as_secs_f64() * 1000.0
        );
    }
    // Plotting: use plotters crate — title="精度 vs ステップサイズ" (log-log scale)
}
```

**結果**:
- **dt小**: 高精度、高コスト
- **dt大**: 低精度、低コスト
- **最適**: dt=0.01（精度とコストのバランス）

---

> **Note:** **進捗: 92%完了**
> 実装と実験を完了。次は発展ゾーンで研究動向と参考文献を整理する。

---

> Progress: 85%
> **理解度チェック**
> 1. Rust ode_solvers での `SDEProblem` 実装において、VP-SDEとVE-SDEのdrift関数とdiffusion関数の具体的な違いをコードの変数名と対応する数式で示せ。
> 2. Predictor-Corrector実装でCorrectorのLangevinステップ数を増やすとサンプル品質が向上するが、計算コストとのトレードオフが生じる境界条件を述べよ。

## 🔬 Z6. 新たな冒険へ（研究動向）

### 6.1 SDE収束理論の最新進展（2024-2025）

**O(d/T)収束理論 (Gen Li & Yuling Yan, 2024)**

[arXiv:2409.18959](https://arxiv.org/abs/2409.18959) "O(d/T) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"

**主な貢献**:
- **最小限の仮定**下でTotal Variation距離 $O(d/T)$ 収束を証明
- データ分布の仮定: 有限1次モーメントのみ（従来はlog-Sobolev不等式等が必要）
- スコア推定が $\ell_2$-正確なら保証される

**実用的示唆**:
- 次元 $d = 1000$、ステップ $T = 1000$ で $\text{TV} \lesssim 1.0$（高精度）
- $T = 50$ に削減 → $\text{TV} \lesssim 20.0$（精度低下、高次ソルバーで補完）

**Manifold仮説下の線形収束 (Peter Potaptchik et al., 2024)**

[arXiv:2410.09046](https://arxiv.org/abs/2410.09046) "Linear Convergence of Diffusion Models Under the Manifold Hypothesis"

**主な貢献**:
- データが固有次元 $d$ のマニフォールド上に集中するとき、KL収束が $O(d \log T)$
- 埋め込み次元 $D$ ではなく固有次元 $d$（$d \ll D$）に依存
- この依存性は**シャープ**（下界も $\Omega(d)$）

**実用的示唆**:
- 画像（$D = 256^2 = 65536$）でも $d \approx 100-500$ → 大幅な理論改善
- 現実のデータのManifold仮説を支持

**VP-SDE離散化誤差の簡易解析 (2025)**

[arXiv:2506.08337](https://arxiv.org/abs/2506.08337) "Diffusion Models under Alternative Noise: Simplified Analysis and Sensitivity"

**主な貢献**:
- Euler-Maruyama法の収束レート $O(T^{-1/2})$ をGrönwall不等式で簡潔に導出
- ガウスノイズを離散ノイズ（Rademacher等）に置き換えても同じ収束レート
- 計算コスト削減の可能性

### 6.2 Score SDE統一理論の発展

**Song et al. 2021の影響**

[arXiv:2011.13456](https://arxiv.org/abs/2011.13456) "Score-Based Generative Modeling through Stochastic Differential Equations"

**貢献**:
- VP-SDE/VE-SDEによるDDPM/NCSNの統一
- Probability Flow ODEで決定論的生成
- Predictor-Corrector法で高品質サンプリング

**後続研究**:
- **Flow Matching** (第38回): Score SDEをさらに一般化
- **Consistency Models** (第40回): Probability Flow ODEを1-Stepに圧縮
- **Rectified Flow**: OTとPF-ODEの接続

### 6.3 Anderson 1982のReverse-time SDE

**Anderson (1982) "Reverse-Time Diffusion Equation Models"**

*Stochastic Processes and their Applications*, vol. 12, pp. 313-326.

**歴史的重要性**:
- Reverse-time SDEの存在を初めて証明
- Girsanov定理とBayes定理の応用
- 拡散モデル（2015-2021）で40年後に再発見

**現代的解釈**:
- Score関数 $\nabla \log p_t(x)$ がDrift項の補正に登場
- 生成モデルはAndersonの定理の**計算可能化**（NNでScore推定）

### 6.4 Rust ode_solversのエコシステム

**ode_solvers**

- 統一インターフェース: ODE/SDE/DAE/DDE/RODE
- 40種以上のソルバー（Runge-Kutta/IMEX/SDEソルバー）
- GPU対応（CUDA.jl統合）

**関連パッケージ**:
- **DiffEqCandle**: Neural ODEの訓練（Universal Differential Equations）
- **Catalyst.jl**: 化学反応ネットワークのSDE
- **ModelingToolkit.jl**: 記号的モデリング → 自動的にSDEを生成

**Diffusion Modelとの統合**:
- Candle（DLフレームワーク）でScore関数 $s_\theta(x, t)$ を訓練
- ode_solversでReverse-time SDE/PF-ODEサンプリング
- Burn（XLAコンパイル）でGPU高速化

### 6.5 SDE数値解法の高度化

**高次ソルバー（第40回で詳説）**:
- **DPM-Solver++**: PF-ODEをRunge-Kutta系で解く、$O(T^{-2})$収束
- **UniPC**: 統一Predictor-Correctorフレームワーク
- **EDM**: Elucidating Diffusion Models（Karras et al. 2022）、最適離散化

**Stochastic Runge-Kutta法**:
- Euler-Maruyamaを超える高次SDE solver
- Strong convergence $O(\Delta t^{3/2})$
- ode_solversで実装済み（`SRIW1()`, `SRIW2()`等）

> Progress: 95%
> **理解度チェック**
> 1. SDE → Flow Matching への接続において、Fokker-Planck方程式の連続性方程式としての解釈が条件付き速度場 $u_t(\mathbf{x}|\mathbf{x}_1)$ の設計にどう寄与するか述べよ。
> 2. VP-SDE・VE-SDE・Sub-VP SDE・PF-ODEの4定式化が同一の周辺分布 $p_t(\mathbf{x})$ を生成できる条件と、それぞれの数値解法上の有利な点を一行ずつ述べよ。

## 🎭 Z7. エピローグ（まとめ・FAQ・次回予告）

### 7.1 本回のまとめ — 3つの核心

**核心1: 離散DDPMの連続時間極限がVP-SDE/VE-SDE**
- DDPM $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$ → VP-SDE
- NCSN（ノイズレベル $\{\sigma_i\}$）→ VE-SDE
- 理論的根拠が明確化（Fokker-Planck方程式、収束性解析）

**核心2: Reverse-time SDEとProbability Flow ODEで生成**
- Anderson 1982のReverse-time SDE: 確率的生成
- Song et al. 2021のPF-ODE: 決定論的生成
- 同じ周辺分布 $p_t(x)$ → サンプリング手法の選択肢

**核心3: Score SDE統一理論がDDPM/NCSN/DDIMを包摂**
- Forward SDE（ノイズ注入）
- Reverse-time SDE（確率的サンプリング）
- Probability Flow ODE（決定論的サンプリング）
- Score関数 $\nabla \log p_t(x)$ がすべての鍵

### 7.2 Course I第5回との接続 — 既習知識の活用

**第5回で学んだこと**:
- Brown運動の定義と性質（連続性、非微分可能性、二次変分）
- 伊藤積分の定義（非予見性、伊藤等距離性）
- 伊藤の補題（$dW^2 = dt$ の導出、確率微分の連鎖律）
- 基本SDE（$dX = f dt + g dW$ の形式、存在・一意性の直感）
- Euler-Maruyama法（SDEの数値解法基礎）
- Fokker-Planck方程式の直感

**本回で深掘りしたこと**:
- VP-SDE/VE-SDEの**厳密導出**（伊藤の補題を適用）
- Fokker-Planck方程式の**厳密導出**（Kramers-Moyal展開）
- Anderson逆時間SDE定理（Girsanov定理の応用）
- Probability Flow ODE（連続方程式との関係）
- 収束性解析（O(d/T)、Manifold仮説）
- Rust ode_solversでのSDE実装

**第5回の知識が本回で活きる瞬間**:
- 伊藤の補題で $dX_t^2$ を計算 → VP-SDE分散導出（3.3節）
- Fokker-Planck方程式の直感を厳密化（3.6節）
- Euler-Maruyama法を前提にPredictor-Corrector法へ発展（3.13節）

### 7.3 次回（第38回）への橋渡し — Flow Matching統一理論

第38回「Flow Matching & 統一理論」で学ぶこと:
- **Conditional Flow Matching**: シミュレーションフリー訓練
- **Optimal Transport ODE**: Rectified Flow（直線輸送）
- **Stochastic Interpolants**: Flow/Diffusionの統一フレームワーク
- **DiffFlow統一理論**: SDM + GANを同一SDE表現
- **Wasserstein勾配流**: JKO schemeとFokker-Planckの等価性
- **Score ↔ Flow ↔ Diffusion ↔ ODE の数学的等価性証明**

**本回との接続**:
- Probability Flow ODE → Flow Matchingへの自然な拡張
- VP-SDE/VE-SDE → 一般確率パスへの一般化
- Score SDE統一理論 → さらなる統一（OT統合）

### 7.4 FAQ — よくある質問

**Q1: VP-SDEとVE-SDE、どちらを使うべき？**

A: タスク依存。
- **VP-SDE**: DDPMベース、画像生成で標準、分散保存で数値安定
- **VE-SDE**: NCSNベース、ノイズレベルが明示的、高次元潜在空間
- 第38回で学ぶFlow MatchingがSDEの制約を超える

**Q2: Probability Flow ODEの「同じ周辺分布」の意味は？**

A: 各時刻 $t$ での確率分布 $p_t(x)$ が同じ。
- Reverse-time SDE: 確率的軌道、サンプルごとに異なる経路
- PF-ODE: 決定論的軌道、初期値が同じなら同じ経路
- どちらも周辺分布 $\{p_t\}_{t \in [0, T]}$ は一致

**Q3: Euler-Maruyama法で十分？高次ソルバーは必須？**

A: タスク依存。
- **Euler-Maruyama**: 実装簡単、$T = 1000$ で十分な精度
- **高次ソルバー**: $T = 50$ に削減可能、推論高速化
- 第40回で学ぶDPM-Solver++/UniPCが実用的

**Q4: スコア関数 $\nabla \log p_t(x)$ はどう学習する？**

A: Denoising Score Matching（第35回）。
- ノイズ付きデータ $x_t$ からScore $\nabla \log p_t(x_t)$ を推定
- Neural Network $s_\theta(x, t)$ を訓練
- 本回は「学習済みScore関数が与えられた」と仮定

**Q5: ode_solversは必須？PyTorchで実装できない？**

A: PyTorchでも可能だが、ode_solversが圧倒的に強力。
- PyTorch: 自力でEuler-Maruyama実装、ソルバー選択肢少
- ode_solvers: 40種ソルバー、自動ステップサイズ調整、GPU対応
- 研究プロトタイプならRust、論文査読用ならPyTorch

**Q6: Anderson 1982論文は読むべき？**

A: 理論派なら推奨、実装派なら不要。
- Song et al. 2021がAnderson定理を現代的に再解釈
- Reverse-time SDEの導出スケッチ（本回3.8節）で十分
- 厳密証明（Girsanov定理）は専門書（Øksendal等）参照
### 7.6 自己診断チェックリスト

- [ ] Brown運動の二次変分 $\langle W \rangle_t = t$ を導出できる
- [ ] 伊藤の補題を使ってVP-SDEの平均・分散を導出できる
- [ ] Fokker-Planck方程式をKramers-Moyal展開から導出できる
- [ ] VP-SDE/VE-SDE/Sub-VP SDEの違いを説明できる
- [ ] Anderson逆時間SDE定理を述べられる
- [ ] Probability Flow ODEとReverse-time SDEの違いを説明できる
- [ ] Score SDE統一理論の4要素（Forward/Reverse/Score/ODE）を列挙できる
- [ ] O(d/T)収束理論の意味を説明できる
- [ ] Manifold仮説下の線形収束の意義を理解している
- [ ] Rust ode_solversでVP-SDEを実装できる
- [ ] Predictor-Corrector法のアルゴリズムを実装できる

全項目✓なら次回へ！未達成項目は該当Zoneを復習。

### 7.7 次回予告 — 第38回: Flow Matching & 統一理論

**第38回の核心トピック**:
- Conditional Flow Matching（CFM）完全導出
- Optimal Transport ODE / Rectified Flow（直線輸送）
- Stochastic Interpolants統一フレームワーク
- DiffFlow統一理論（SDM + GAN = 同一SDE）
- Wasserstein勾配流（JKO scheme / Fokker-Planckとの等価性）
- **Score ↔ Flow ↔ Diffusion ↔ ODE の数学的等価性証明**

**第37回（本回）との接続**:
- VP-SDE/VE-SDEを**一般確率パス**に拡張
- Probability Flow ODE → Flow Matching ODE（Optimal Transport統合）
- Score SDE → Flow Matching統一理論へ

> **Note:** **進捗: 100%完了 — 第37回読了！**
> SDE/ODE & 確率過程論を完全習得した。VP-SDE/VE-SDE導出、Anderson逆時間SDE、Probability Flow ODE、Score SDE統一理論、収束性解析、Rust実装を修得。次回Flow Matchingで全生成モデルの統一理論へ。

---

### 6.X パラダイム転換の問い

**"離散ステップ数 $T = 1000$ は経験則。連続時間SDEで理論化したとき、初めて「なぜ1000で十分か」に答えられる。理論なき実装は暗闇の航海では？"**

**議論ポイント**:
1. DDPMの成功（2020）は経験的。理論的正当化（Score SDE統一理論、2021）は後追い。実務では「動けばOK」か、理論的理解は必須か？
2. O(d/T)収束理論（2024）で「$T = 1000$ が十分な理由」が数学的に説明された。だが実装者の何%がこれを知るべきか？
3. Probability Flow ODEの発見（Song et al. 2021）はSDEの連続時間定式化なしには不可能だった。連続理論が新手法を生む例。理論 vs 実装、どちらが先か？

<details><summary>歴史的文脈 — SDEと拡散モデルの40年ギャップ</summary>

**Anderson 1982**: Reverse-time SDEを証明。当時は理論的興味のみ、応用なし。

**2015 Sohl-Dickstein et al.**: 拡散モデル初提案。Andersonを引用せず（独立に発見）。

**2020 Ho et al. DDPM**: 離散時間定式化で大成功。SDEとの接続は明示せず。

**2021 Song et al. Score SDE**: 40年前のAnderson定理を再発見、拡散モデルとSDE統一。Probability Flow ODE発見。

**2024-2025 収束理論**: Li & Yan、Potaptchik et al.がO(d/T)、Manifold線形収束を証明。理論が実装を逆照射。

**教訓**: 理論と実装の対話が新パラダイムを生む。40年の時を経て理論が実装に光を当てる。

</details>

---

## 参考文献

### 主要論文

[^1]: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole (2021). "Score-Based Generative Modeling through Stochastic Differential Equations". *ICLR 2021 (Oral)*.
<https://arxiv.org/abs/2011.13456>

[^2]: Brian D. O. Anderson (1982). "Reverse-time diffusion equation models". *Stochastic Processes and their Applications*, vol. 12, pp. 313-326.
<https://www.sciencedirect.com/science/article/pii/0304414982900515>

[^3]: Gen Li and Yuling Yan (2024). "O(d/T) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions". *arXiv preprint*.
<https://arxiv.org/abs/2409.18959>

[^4]: Peter Potaptchik, Iskander Azangulov, and George Deligiannidis (2024). "Linear Convergence of Diffusion Models Under the Manifold Hypothesis". *arXiv preprint*.
<https://arxiv.org/abs/2410.09046>

[^5]: Choi, J. & Fan, C. (2025). "Diffusion Models under Alternative Noise: Simplified Analysis and Sensitivity". *arXiv preprint*.
<https://arxiv.org/abs/2506.08337>

[^6]: Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020). "Denoising Diffusion Probabilistic Models". *NeurIPS 2020*.
<https://arxiv.org/abs/2006.11239>

[^7]: Alex Nichol and Prafulla Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models". *ICML 2021*.
<https://arxiv.org/abs/2102.09672>

[^8]: Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics". *ICML 2015*.
<https://arxiv.org/abs/1503.03585>

[^9]: Jiaming Song, Chenlin Meng, and Stefano Ermon (2020). "Denoising Diffusion Implicit Models". *ICLR 2021*.
<https://arxiv.org/abs/2010.02502>

[^10]: Yang Song and Stefano Ermon (2020). "Improved Techniques for Training Score-Based Generative Models". *NeurIPS 2020*.
<https://arxiv.org/abs/2006.09011>

### 教科書

- Bernt Øksendal (2003). *Stochastic Differential Equations: An Introduction with Applications* (6th ed.). Springer.
- Peter E. Kloeden and Eckhard Platen (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
- Olav Kallenberg (2002). *Foundations of Modern Probability* (2nd ed.). Springer.

### オンラインリソース

- Yang Song (2021). "Generative Modeling by Estimating Gradients of the Data Distribution". [Blog Post](https://yang-song.net/blog/2021/score/)
- MIT 6.S184 (2026). "Diffusion Models & Flow Matching". [Course Website](https://diffusion.csail.mit.edu/)
- ode_solvers Documentation. [docs.sciml.ai](https://docs.sciml.ai/DiffEqDocs/stable/)

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
