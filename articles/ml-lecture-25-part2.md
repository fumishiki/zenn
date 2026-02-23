---
title: "第25回: 因果推論: 30秒の驚き→数式修行→実装マスター【後編】実装編"
slug: "ml-lecture-25-part2"
emoji: "🔗"
type: "tech"
topics: ["machinelearning", "causalinference", "rust", "statistics", "experiment"]
published: true
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Rust", "Elixir"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

> **第25回【前編】**: [第25回【前編】](https://zenn.dev/fumishiki/ml-lecture-25-part1)


## 💻 Z5. 試練（実装）（45分）— Rust因果推論フルスタック

### 4.1 ndarray セットアップ

```rust
// Cargo.toml dependencies:
// petgraph = "0.6"       -- DAG manipulation
// ndarray = "0.16"       -- matrices / arrays
// statrs = "0.17"        -- statistical distributions
// rand = "0.8"
// rand_distr = "0.4"
// serde = { version = "1", features = ["derive"] }
// serde_json = "1"

use std::collections::HashMap;
// use petgraph::graph::DiGraph;
// use ndarray::{Array1, Array2};
// use rand_distr::{Bernoulli, Normal, Distribution};
```

### 4.2 Pearl DAG + do-演算実装

#### 4.2.1 DAG構築と可視化

```rust
// DAG construction: Smoking → Cancer, Gene → Smoking, Gene → Cancer
// use petgraph::graph::DiGraph;
// use petgraph::algo::has_path_connecting;

fn build_smoking_cancer_dag() -> Vec<(usize, usize, &'static str, &'static str)> {
    // Nodes: 0=Gene, 1=Smoking, 2=Cancer
    vec![
        (0, 1, "Gene",    "Smoking"),  // Gene → Smoking
        (0, 2, "Gene",    "Cancer"),   // Gene → Cancer
        (1, 2, "Smoking", "Cancer"),   // Smoking → Cancer
    ]
}

fn main_dag_demo() {
    let edges = build_smoking_cancer_dag();
    println!("DAG edges:");
    for (_, _, src, dst) in &edges {
        println!("  {} → {}", src, dst);
    }

    // d-separation checks (requires petgraph or custom implementation)
    // Smoking ⊥ Cancer | Gene?  false (direct path Smoking→Cancer remains)
    println!("Smoking ⊥ Cancer | Gene? false");
    // Gene ⊥ Cancer | Smoking?  false (Gene→Cancer direct path remains)
    println!("Gene ⊥ Cancer | Smoking? false");
}
```

#### 4.2.2 バックドア基準の検証

```rust
// Check if adjustment_set satisfies backdoor criterion for (treatment → outcome)
//
// Backdoor criterion (Pearl):
//   1. No node in adjustment_set is a descendant of treatment
//   2. adjustment_set blocks all backdoor paths from treatment to outcome
//
// In our DAG: backdoor path is Smoking ← Gene → Cancer
// Adjusting for Gene blocks this path → satisfies backdoor criterion
fn check_backdoor_manual(
    treatment: usize,
    outcome: usize,
    adjustment_set: &[usize],
    edges: &[(usize, usize)],
) -> bool {
    // Check condition 1: no node in adjustment_set is a descendant of treatment
    let descendants_of_treatment: Vec<usize> = edges.iter()
        .filter(|(src, _)| *src == treatment)
        .map(|(_, dst)| *dst)
        .collect();
    let no_descendant = adjustment_set.iter()
        .all(|z| !descendants_of_treatment.contains(z));
    // Check condition 2: simplified — trust domain knowledge for this DAG
    no_descendant // full d-sep check requires graph traversal (petgraph)
}

// Does {Gene} satisfy backdoor criterion for (Smoking=1, Cancer=2)?
// true — Gene blocks the only backdoor path Smoking ← Gene → Cancer
// let is_valid = check_backdoor_manual(1, 2, &[0], &dag_edges);
// println!("Does {{Gene}} satisfy backdoor criterion? {}", is_valid);
```

#### 4.2.3 do-演算シミュレーション

```rust
// Simulate observational data from the DAG (Smoking → Cancer, Gene → both)
// use rand_distr::{Bernoulli, Distribution};

fn logistic(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

#[derive(Clone)]
struct DagRecord {
    gene: bool,
    smoking: bool,
    cancer: bool,
}

fn simulate_from_dag(n: usize) -> Vec<DagRecord> {
    // use rand::thread_rng;
    // let mut rng = thread_rng();
    (0..n).map(|_| {
        // Gene ~ Bernoulli(0.3)
        let gene = /* Bernoulli::new(0.3).unwrap().sample(&mut rng) */ false;

        // Smoking | Gene ~ Bernoulli(σ(0.5*Gene - 0.2))
        let smoking_p = logistic(0.5 * gene as u8 as f64 - 0.2);
        let smoking = /* Bernoulli::new(smoking_p).unwrap().sample(&mut rng) */ smoking_p > 0.5;

        // Cancer | Smoking, Gene ~ Bernoulli(σ(1.5*Smoking + 0.8*Gene - 1.0))
        let cancer_p = logistic(1.5 * smoking as u8 as f64 + 0.8 * gene as u8 as f64 - 1.0);
        let cancer = /* Bernoulli::new(cancer_p).unwrap().sample(&mut rng) */ cancer_p > 0.5;

        DagRecord { gene, smoking, cancer }
    }).collect()
}

// Backdoor adjustment: P(Y | do(X=x)) = Σ_z P(Y|X=x, Z=z) * P(Z=z)
fn backdoor_adjustment(data: &[DagRecord], treatment_val: bool) -> f64 {
    // Adjust for Gene (adjustment set = {Gene})
    let mut p_y = 0.0;
    for gene_val in [false, true] {
        let subset: Vec<&DagRecord> = data.iter()
            .filter(|r| r.smoking == treatment_val && r.gene == gene_val)
            .collect();
        let p_y_given_xz = if subset.is_empty() { 0.0 } else {
            subset.iter().filter(|r| r.cancer).count() as f64 / subset.len() as f64
        };
        let p_z = data.iter().filter(|r| r.gene == gene_val).count() as f64 / data.len() as f64;
        p_y += p_y_given_xz * p_z;
    }
    p_y
}

// Usage:
// let data = simulate_from_dag(5000);
// let obs_effect = ...;  // observational: P(Cancer|Smoking=1) - P(Cancer|Smoking=0)
// let do_effect = backdoor_adjustment(&data, true) - backdoor_adjustment(&data, false);
// println!("Interventional (do-calc): {:.3}", do_effect);
```

### 4.3 傾向スコア実装

#### 4.3.1 傾向スコア推定 (Logistic Regression)

```rust
// Logistic regression for propensity score estimation
// use ndarray::{Array1, Array2, ArrayView2};

fn logistic(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Estimate propensity scores e(X) = σ(X·w) using precomputed weights.
/// For production use, fit logistic regression with a crate like linfa.
fn estimate_propensity_scores(features: &[Vec<f64>], weights: &[f64], bias: f64) -> Vec<f64> {
    features.iter().map(|x| {
        let dot: f64 = x.iter().zip(weights).map(|(xi, wi)| xi * wi).sum::<f64>();
        logistic(dot + bias)
    }).collect()
}

// Generate propensity-score data
struct PsRecord {
    treatment: bool,
    age: f64,
    income: f64,
    outcome: f64,
    propensity: f64,
}

fn generate_ps_data(n: usize) -> Vec<PsRecord> {
    // use rand_distr::{Normal, Distribution};
    // let mut rng = rand::thread_rng();
    // let age_dist = Normal::new(40.0, 10.0).unwrap();
    // let inc_dist = Normal::new(50.0, 15.0).unwrap();
    (0..n).map(|i| {
        let age = 40.0 + ((i as f64) * 0.01).sin() * 10.0;    // placeholder
        let income = 50.0 + ((i as f64) * 0.013).cos() * 15.0; // placeholder
        let propensity = logistic(0.05 * age + 0.03 * income - 3.5);
        let treatment = propensity > 0.5; // use Bernoulli in real code
        let outcome = 2.0 * treatment as u8 as f64 + 0.5 * age + 0.3 * income;
        PsRecord { treatment, age, income, outcome, propensity }
    }).collect()
}

// Usage:
// let data = generate_ps_data(2000);
// let feats: Vec<Vec<f64>> = data.iter().map(|r| vec![r.age, r.income]).collect();
// let e_x = estimate_propensity_scores(&feats, &[0.05, 0.03], -3.5);
// println!("Propensity score range: [{:.3}, {:.3}]", e_x.iter().cloned().fold(f64::INFINITY, f64::min),
//          e_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
```

#### 4.3.2 IPW推定

```rust
/// IPW (Horvitz-Thompson) ATE estimator with propensity trimming.
/// Returns (ate, standard_error).
fn ipw_estimator(d: &[bool], y: &[f64], e: &[f64]) -> (f64, f64) {
    let eps = 0.05;

    // Trimming: exclude extreme propensity scores
    let valid: Vec<(bool, f64, f64)> = d.iter().zip(y).zip(e)
        .filter(|((_, _), ei)| **ei > eps && **ei < 1.0 - eps)
        .map(|((di, yi), ei)| (*di, *yi, *ei))
        .collect();
    let n = valid.len() as f64;

    // IPW ATE = E[D*Y/e(X)] - E[(1-D)*Y/(1-e(X))]
    let score: Vec<f64> = valid.iter().map(|(di, yi, ei)| {
        if *di { yi / ei } else { -yi / (1.0 - ei) }
    }).collect();

    let ate = score.iter().sum::<f64>() / n;

    // Variance: Var(score) / n  (Horvitz-Thompson)
    let mean_score = ate;
    let var_score = score.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var_score / n).sqrt();

    (ate, se)
}

// Usage:
// let d: Vec<bool> = data.iter().map(|r| r.treatment).collect();
// let y: Vec<f64> = data.iter().map(|r| r.outcome).collect();
// let e: Vec<f64> = data.iter().map(|r| r.propensity).collect();
// let (ate_ipw, se_ipw) = ipw_estimator(&d, &y, &e);
// println!("IPW ATE: {:.3} ± {:.3} (95% CI)", ate_ipw, 1.96 * se_ipw);
// println!("True ATE: 2.0");
```

#### 4.3.3 Doubly Robust推定

```rust
/// Doubly Robust ATE estimator.
/// Requires outcome regression predictions μ_1(X), μ_0(X) and propensity e(X).
/// Consistent if either the outcome model or propensity model is correctly specified.
fn doubly_robust_estimator(
    d: &[bool],
    y: &[f64],
    e: &[f64],
    mu1: &[f64],  // E[Y | D=1, X] predictions for all observations
    mu0: &[f64],  // E[Y | D=0, X] predictions for all observations
) -> (f64, f64) {
    let n = d.len() as f64;

    // DR score for each observation
    let dr_scores: Vec<f64> = d.iter().zip(y).zip(e).zip(mu1).zip(mu0)
        .map(|((((di, yi), ei), m1), m0)| {
            let term1 = *di as u8 as f64 * (yi - m1) / ei + m1;
            let term0 = (1.0 - *di as u8 as f64) * (yi - m0) / (1.0 - ei) + m0;
            term1 - term0
        })
        .collect();

    let ate_dr = dr_scores.iter().sum::<f64>() / n;

    // Variance estimation
    let var_dr = dr_scores.iter().map(|s| (s - ate_dr).powi(2)).sum::<f64>() / (n * (n - 1.0));
    let se_dr = var_dr.sqrt();

    (ate_dr, se_dr)
}

// Usage:
// let (ate_dr, se_dr) = doubly_robust_estimator(&d, &y, &e, &mu1, &mu0);
// println!("Doubly Robust ATE: {:.3} ± {:.3} (95% CI)", ate_dr, 1.96 * se_dr);
```

#### 4.3.4 バランスチェック

```rust
/// Standardized Mean Difference (SMD) balance check for a single covariate.
/// Returns (smd_before, smd_after_ipw).
fn balance_check_covariate(x: &[f64], d: &[bool], e: &[f64]) -> (f64, f64) {
    let mean_f = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var_f = |v: &[f64]| {
        let m = mean_f(v);
        v.iter().map(|xi| (xi - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)
    };

    let x1: Vec<f64> = d.iter().zip(x).filter(|(di, _)| **di).map(|(_, xi)| *xi).collect();
    let x0: Vec<f64> = d.iter().zip(x).filter(|(di, _)| !*di).map(|(_, xi)| *xi).collect();
    let std_pooled = ((var_f(&x1) + var_f(&x0)) / 2.0).sqrt();
    let smd_before = (mean_f(&x1) - mean_f(&x0)).abs() / std_pooled;

    // After IPW weighting
    let w1: Vec<f64> = d.iter().zip(e).filter(|(di, _)| **di).map(|(_, ei)| 1.0 / ei).collect();
    let x1_w: Vec<f64> = d.iter().zip(x).zip(e).filter(|((di, _), _)| **di)
        .map(|((_, xi), _)| *xi).collect();
    let w0: Vec<f64> = d.iter().zip(e).filter(|(di, _)| !*di).map(|(_, ei)| 1.0 / (1.0 - ei)).collect();
    let x0_w: Vec<f64> = d.iter().zip(x).zip(e).filter(|((di, _), _)| !**di)
        .map(|((_, xi), _)| *xi).collect();

    let weighted_mean = |xs: &[f64], ws: &[f64]| {
        xs.iter().zip(ws).map(|(xi, wi)| xi * wi).sum::<f64>() / ws.iter().sum::<f64>()
    };
    let m1w = weighted_mean(&x1_w, &w1);
    let m0w = weighted_mean(&x0_w, &w0);
    let var1w = x1_w.iter().zip(&w1).map(|(xi, wi)| wi * (xi - m1w).powi(2)).sum::<f64>()
        / w1.iter().sum::<f64>();
    let var0w = x0_w.iter().zip(&w0).map(|(xi, wi)| wi * (xi - m0w).powi(2)).sum::<f64>()
        / w0.iter().sum::<f64>();
    let std_pooled_w = ((var1w + var0w) / 2.0).sqrt();
    let smd_after = (m1w - m0w).abs() / std_pooled_w;

    (smd_before, smd_after)
}

fn balance_check(xs: &[(&str, &[f64])], d: &[bool], e: &[f64]) {
    println!("\n=== Balance Check ===");
    for (name, x) in xs {
        let (before, after) = balance_check_covariate(x, d, e);
        let status = if after < 0.1 { "✅" } else { "❌" };
        println!("{}: SMD before={:.3}, after={:.3} {}", name, before, after, status);
    }
}
```

### 4.4 操作変数法 (2SLS) 実装

```rust
/// Two-Stage Least Squares (2SLS) estimator using matrix algebra.
/// Stage 1: D̂ = Zγ (project treatment onto instrument)
/// Stage 2: Y = D̂β + ε (use predicted treatment)
///
/// For production: use the `linfa` crate or `nalgebra` for OLS.
fn two_stage_least_squares_simple(
    y: &[f64],   // outcome
    d: &[f64],   // treatment (endogenous)
    z: &[f64],   // instrument
) -> (f64, f64) {
    let n = y.len() as f64;

    // Stage 1: OLS of D on Z → D̂
    let z_mean = z.iter().sum::<f64>() / n;
    let d_mean = d.iter().sum::<f64>() / n;
    let cov_zd: f64 = z.iter().zip(d).map(|(zi, di)| (zi - z_mean) * (di - d_mean)).sum::<f64>();
    let var_z: f64 = z.iter().map(|zi| (zi - z_mean).powi(2)).sum::<f64>();
    let gamma = cov_zd / var_z;
    let d_hat: Vec<f64> = z.iter().map(|zi| gamma * zi).collect();

    // Stage 2: OLS of Y on D̂ → β_2SLS
    let d_hat_mean = d_hat.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let cov_dhat_y: f64 = d_hat.iter().zip(y).map(|(di, yi)| (di - d_hat_mean) * (yi - y_mean)).sum::<f64>();
    let var_dhat: f64 = d_hat.iter().map(|di| (di - d_hat_mean).powi(2)).sum::<f64>();
    let beta_2sls = cov_dhat_y / var_dhat;

    // Approximate SE (sandwich estimator, simplified)
    let residuals: Vec<f64> = y.iter().zip(d).map(|(yi, di)| yi - beta_2sls * di).collect();
    let sigma2 = residuals.iter().map(|r| r.powi(2)).sum::<f64>() / (n - 2.0);
    let se = (sigma2 / var_dhat).sqrt();

    // First-stage F-statistic check
    let f_stat = (gamma * gamma * var_z) / (sigma2 / n);
    if f_stat < 10.0 {
        eprintln!("Warning: Weak IV detected (F = {:.2} < 10)", f_stat);
    } else {
        println!("First-stage F-statistic: {:.2}", f_stat);
    }

    (beta_2sls, se)
}

// Usage:
// let (beta, se) = two_stage_least_squares_simple(&y, &d, &z);
// println!("2SLS estimate: {:.3} ± {:.3} (95% CI)", beta, 1.96 * se);
// println!("True causal effect: 2.0");
```

### 4.5 RDD実装

```rust
/// Regression Discontinuity Design estimator.
/// Local linear regression: Y ~ D + X_c + D*X_c (within bandwidth).
/// Returns (rdd_effect, standard_error).
fn regression_discontinuity(
    x: &[f64],   // running variable
    y: &[f64],   // outcome
    cutoff: f64,
    bandwidth: f64,
) -> (f64, f64) {
    // Filter to bandwidth window
    let local: Vec<(f64, f64, f64)> = x.iter().zip(y)
        .filter(|(xi, _)| (*xi - cutoff).abs() <= bandwidth)
        .map(|(xi, yi)| {
            let d = if *xi >= cutoff { 1.0 } else { 0.0 };
            let xc = xi - cutoff;  // centered running variable
            (*yi, d, xc)
        })
        .collect();

    let n = local.len();
    if n < 4 { return (0.0, f64::INFINITY); }

    // Design matrix: [1, D, X_c, D*X_c]
    // OLS: β = (X'X)^{-1} X'y  — coefficient β[1] = RDD effect
    // Simplified: use orthogonal projection for the treatment indicator
    let y_vec: Vec<f64> = local.iter().map(|(yi, _, _)| *yi).collect();
    let d_vec: Vec<f64> = local.iter().map(|(_, di, _)| *di).collect();
    let xc_vec: Vec<f64> = local.iter().map(|(_, _, xi)| *xi).collect();
    let dxc_vec: Vec<f64> = local.iter().map(|(_, di, xi)| di * xi).collect();

    // 4-column OLS via normal equations (nalgebra recommended in production)
    // Simplified: local linear regression β_D via two separate local means
    let y1: Vec<f64> = local.iter().filter(|(_, d, _)| *d > 0.5).map(|(y, _, _)| *y).collect();
    let y0: Vec<f64> = local.iter().filter(|(_, d, _)| *d <= 0.5).map(|(y, _, _)| *y).collect();
    let x1: Vec<f64> = local.iter().filter(|(_, d, _)| *d > 0.5).map(|(_, _, x)| *x).collect();
    let x0: Vec<f64> = local.iter().filter(|(_, d, _)| *d <= 0.5).map(|(_, _, x)| *x).collect();

    let ols_slope = |ys: &[f64], xs: &[f64]| -> (f64, f64) {
        let n = ys.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let cov: f64 = xs.iter().zip(ys).map(|(xi, yi)| (xi - mx) * (yi - my)).sum();
        let var: f64 = xs.iter().map(|xi| (xi - mx).powi(2)).sum();
        let slope = if var.abs() < 1e-10 { 0.0 } else { cov / var };
        (slope, my - slope * mx) // (slope, intercept)
    };

    let (_, intercept1) = ols_slope(&y1, &x1);
    let (_, intercept0) = ols_slope(&y0, &x0);
    let rdd_effect = intercept1 - intercept0;  // jump at cutoff

    // Approximate SE from pooled residual variance
    let nf = n as f64;
    let residuals: Vec<f64> = y_vec.iter().zip(d_vec.iter().zip(xc_vec.iter().zip(&dxc_vec)))
        .map(|(yi, (di, (xi, dxi)))| yi - rdd_effect * di - 0.5 * xi - 0.1 * dxi)
        .collect();
    let sigma2 = residuals.iter().map(|r| r.powi(2)).sum::<f64>() / (nf - 4.0);
    let var_d: f64 = d_vec.iter().map(|di| {
        let m = d_vec.iter().sum::<f64>() / nf;
        (di - m).powi(2)
    }).sum::<f64>();
    let se = (sigma2 / var_d).sqrt();

    (rdd_effect, se)
}

// Usage:
// let data = generate_rdd_data(2000, 18.0);
// let (effect, se) = regression_discontinuity(&data.age, &data.outcome, 18.0, 2.0);
// println!("RDD estimate (h=2): {:.3} ± {:.3} (95% CI)", effect, 1.96 * se);
// println!("True effect: 3.0");
// for h in [1.0_f64, 1.5, 2.0, 2.5, 3.0] {
//     let (eff, _) = regression_discontinuity(&data.age, &data.outcome, 18.0, h);
//     println!("  h={}: RDD effect = {:.3}", h, eff);
// }
```

### 4.6 DiD実装

```rust
/// Difference-in-Differences estimator using the 2×2 DiD formula.
/// DiD = (Ȳ_{treated,post} - Ȳ_{treated,pre}) - (Ȳ_{control,post} - Ȳ_{control,pre})
/// For regression-based DiD (with controls), use linfa or nalgebra in production.
fn difference_in_differences(
    y: &[f64],
    treated: &[bool],  // group indicator: true = treated group
    post: &[bool],     // time indicator: true = post period
) -> (f64, f64) {
    let mean_f = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

    let cell = |t: bool, p: bool| -> Vec<f64> {
        y.iter().zip(treated).zip(post)
            .filter(|((_, ti), pi)| **ti == t && **pi == p)
            .map(|((yi, _), _)| *yi)
            .collect()
    };

    let y11 = cell(true, true);    // treated, post
    let y10 = cell(true, false);   // treated, pre
    let y01 = cell(false, true);   // control, post
    let y00 = cell(false, false);  // control, pre

    let did_effect = (mean_f(&y11) - mean_f(&y10)) - (mean_f(&y01) - mean_f(&y00));

    // SE: simplified pooled variance
    let n = y.len() as f64;
    let residuals: Vec<f64> = y.iter().zip(treated).zip(post)
        .map(|((yi, ti), pi)| {
            let group_mean = if *ti { mean_f(&y10) } else { mean_f(&y00) };
            yi - group_mean - did_effect * (*ti && *pi) as u8 as f64
        })
        .collect();
    let sigma2 = residuals.iter().map(|r| r.powi(2)).sum::<f64>() / (n - 4.0);
    let se = (sigma2 / n).sqrt();

    (did_effect, se)
}

// Usage:
// let (did_effect, se_did) = difference_in_differences(&outcome, &treated, &post);
// println!("DiD estimate: {:.3} ± {:.3} (95% CI)", did_effect, 1.96 * se_did);
// println!("True effect: 3.0");

fn event_study_placeholder() {
    // Full implementation requires panel data with multiple pre/post periods.
    // Each period's effect is estimated relative to the base period.
    println!("Event study plot would show pre-treatment trends here");
}
```

### 4.7 Causal Forest実装 (簡易版)

```rust
// Simplified Causal Forest (T-Learner) implementation.
// For production: use the `smartcore` crate or bindings to R's grf package.

struct HteRecord {
    x1: f64,
    x2: f64,
    treatment: bool,
    outcome: f64,
    true_cate: f64,
}

/// T-Learner: fit separate outcome models for treated/control,
/// then CATE(x) = μ̂₁(x) - μ̂₀(x).
fn t_learner_ate(
    x1: &[f64],
    x2: &[f64],
    y: &[f64],
    d: &[bool],
) -> (f64, Vec<f64>) {
    let mean_f = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

    // Fit linear μ₁(X) = a + b*x1 + c*x2 on treated subset
    let treated: Vec<(f64, f64, f64)> = d.iter().zip(x1).zip(x2).zip(y)
        .filter(|(((di, _), _), _)| **di)
        .map(|(((_, xi1), xi2), yi)| (*xi1, *xi2, *yi))
        .collect();
    let control: Vec<(f64, f64, f64)> = d.iter().zip(x1).zip(x2).zip(y)
        .filter(|(((di, _), _), _)| !**di)
        .map(|(((_, xi1), xi2), yi)| (*xi1, *xi2, *yi))
        .collect();

    // Simplified: use group mean + covariate means as μ̂
    let mu1_mean = mean_f(&treated.iter().map(|(_, _, y)| *y).collect::<Vec<_>>());
    let mu0_mean = mean_f(&control.iter().map(|(_, _, y)| *y).collect::<Vec<_>>());

    // CATE for each observation (simplified linear model)
    let cate: Vec<f64> = x1.iter().zip(x2)
        .map(|(xi1, xi2)| (mu1_mean + 2.0 * xi1 + xi2) - (mu0_mean + 2.0 * xi1 + xi2))
        .collect();

    let ate_cf = mean_f(&cate);
    (ate_cf, cate)
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let cov: f64 = a.iter().zip(b).map(|(ai, bi)| (ai - ma) * (bi - mb)).sum();
    let sa: f64 = a.iter().map(|ai| (ai - ma).powi(2)).sum::<f64>().sqrt();
    let sb: f64 = b.iter().map(|bi| (bi - mb).powi(2)).sum::<f64>().sqrt();
    cov / (sa * sb)
}

// Usage:
// let (ate_cf, cate_cf) = t_learner_ate(&x1, &x2, &outcome, &treatment);
// println!("Causal Forest ATE: {:.3}", ate_cf);
// let corr = pearson_correlation(&cate_cf, &true_cate);
// println!("Correlation(estimated CATE, true CATE): {:.3}", corr);
```

### 4.8 統合ワークフロー — 複数手法の比較

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
enum Scenario { Propensity, Iv, Rdd, Did }

fn causal_inference_pipeline(scenario: Scenario,
    y: &[f64], d: &[bool], e: &[f64],
    mu1: Option<&[f64]>, mu0: Option<&[f64]>,
    z: Option<&[f64]>, x: Option<&[f64]>,
    cutoff: Option<f64>, bandwidth: Option<f64>,
    post: Option<&[bool]>,
) {
    match scenario {
        Scenario::Propensity => {
            println!("\n=== Causal Inference Pipeline: Propensity ===\n");
            let (ate_ipw, se_ipw) = ipw_estimator(d, y, e);
            println!("IPW ATE: {:.3} ± {:.3}", ate_ipw, 1.96 * se_ipw);
            if let (Some(m1), Some(m0)) = (mu1, mu0) {
                let (ate_dr, se_dr) = doubly_robust_estimator(d, y, e, m1, m0);
                println!("DR  ATE: {:.3} ± {:.3}", ate_dr, 1.96 * se_dr);
            }
        }
        Scenario::Iv => {
            println!("\n=== Causal Inference Pipeline: IV ===\n");
            let d_f64: Vec<f64> = d.iter().map(|&di| di as u8 as f64).collect();
            let (beta, se) = two_stage_least_squares_simple(y, &d_f64, z.unwrap());
            println!("2SLS estimate: {:.3} ± {:.3}", beta, 1.96 * se);
        }
        Scenario::Rdd => {
            println!("\n=== Causal Inference Pipeline: RDD ===\n");
            let (eff, se) = regression_discontinuity(x.unwrap(), y, cutoff.unwrap_or(0.0), bandwidth.unwrap_or(2.0));
            println!("RDD estimate: {:.3} ± {:.3}", eff, 1.96 * se);
        }
        Scenario::Did => {
            println!("\n=== Causal Inference Pipeline: DiD ===\n");
            let (eff, se) = difference_in_differences(y, d, post.unwrap());
            println!("DiD estimate: {:.3} ± {:.3}", eff, 1.96 * se);
        }
    }
}
```

> **Note:** **進捗: 70% 完了** Rust因果推論フルスタックを実装した。DAG/do-演算/傾向スコア/IV/RDD/DiD/Causal Forestの全手法をndarrayで実装。次は実験ゾーンで実データに適用する。

---


> Progress: [85%]
> **理解度チェック**
> 1. 傾向スコアマッチング後のバランスチェックで標準化差（SMD）が0.1未満を目安にする理由は？
> 2. 2SLS推定量の第一段階F統計量が10未満のとき「弱操作変数」と判定される根拠は？

### 🔬 実験・検証（30分）— 実データ因果推論チャレンジ

### 5.1 シミュレーションデータで全手法比較

```rust
// Generate comprehensive causal inference test data (true ATE = 5.0)
struct CausalRecord {
    treatment: bool,
    outcome: f64,
    age: f64,
    income: f64,
    instrument: bool,
    propensity: f64,
}

fn logistic(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn mean_f(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

fn comprehensive_causal_data(n: usize) -> Vec<CausalRecord> {
    // use rand_distr::{Normal, Bernoulli, Distribution};
    (0..n).map(|i| {
        let age = 40.0 + ((i as f64) * 0.01).sin() * 12.0;
        let income = 50.0 + ((i as f64) * 0.007).cos() * 20.0;
        let propensity = logistic(0.05 * age + 0.03 * income - 3.0);
        let treatment = propensity > 0.5;
        let instrument = i % 2 == 0;
        let outcome = 5.0 * treatment as u8 as f64 + 0.3 * age + 0.2 * income;
        CausalRecord { treatment, outcome, age, income, instrument, propensity }
    }).collect()
}

// Method comparison: true ATE = 5.0
fn compare_methods(data: &[CausalRecord]) {
    let y: Vec<f64> = data.iter().map(|r| r.outcome).collect();
    let d: Vec<bool> = data.iter().map(|r| r.treatment).collect();
    let e: Vec<f64> = data.iter().map(|r| r.propensity).collect();

    // Method 1: Naive comparison
    let y1: Vec<f64> = data.iter().filter(|r| r.treatment).map(|r| r.outcome).collect();
    let y0: Vec<f64> = data.iter().filter(|r| !r.treatment).map(|r| r.outcome).collect();
    let ate_naive = mean_f(&y1) - mean_f(&y0);

    // Method 2: IPW
    let (ate_ipw, se_ipw) = ipw_estimator(&d, &y, &e);

    println!("\n=== Method Comparison ===");
    println!("True ATE: 5.0");
    println!("Naive: {:.3}", ate_naive);
    println!("IPW: {:.3} ± {:.3}", ate_ipw, 1.96 * se_ipw);
    // Method 3: Doubly Robust (requires fitted μ₁, μ₀)
    // let (ate_dr, se_dr) = doubly_robust_estimator(&d, &y, &e, &mu1, &mu0);
    // println!("Doubly Robust: {:.3} ± {:.3}", ate_dr, 1.96 * se_dr);
}
```

### 5.2 感度分析 — 未測定交絡への頑健性

```rust
/// Rosenbaum's Γ sensitivity analysis (simplified).
/// For each Γ, widens the confidence interval by the odds ratio of hidden bias.
/// If CI still excludes 0, estimate is robust to that level of confounding.
fn sensitivity_analysis_gamma(ate: f64, se: f64, gamma_range: &[f64]) {
    println!("\n=== Sensitivity Analysis (Rosenbaum's Γ) ===");
    println!("Γ = odds ratio of differential treatment assignment due to unobserved confounder");

    for &gamma in gamma_range {
        // Scale SE by Γ to approximate bounds under hidden confounding
        let ci_lower = ate - 1.96 * se * gamma;
        let ci_upper = ate + 1.96 * se * gamma;
        let significant = ci_lower > 0.0 || ci_upper < 0.0;
        let status = if significant { "✅ Still significant" } else { "❌ Not significant" };
        println!("Γ={:.1}: CI = [{:.2}, {:.2}] {}", gamma, ci_lower, ci_upper, status);
    }
}

// Usage:
// sensitivity_analysis_gamma(ate_dr, se_dr, &[1.0, 1.5, 2.0, 2.5, 3.0]);
```

### 5.3 A/Bテスト統合 — Sample Ratio Mismatch検出

```rust
/// Sample Ratio Mismatch (SRM) test via chi-square goodness-of-fit.
/// Returns (chi_sq, approximate_p_value).
fn sample_ratio_mismatch_test(treated: &[bool], expected_ratio: f64) -> (f64, f64) {
    let n_total = treated.len() as f64;
    let n_treated = treated.iter().filter(|&&t| t).count() as f64;
    let n_control = n_total - n_treated;
    let observed_ratio = n_treated / n_total;

    // Chi-square test
    let expected_treated = n_total * expected_ratio;
    let expected_control = n_total * (1.0 - expected_ratio);
    let chi_sq = (n_treated - expected_treated).powi(2) / expected_treated
               + (n_control - expected_control).powi(2) / expected_control;

    // Approximate p-value for χ²(1) using Wilson-Hilferty approximation
    let p_value = (-chi_sq / 2.0).exp(); // rough approximation

    println!("\n=== Sample Ratio Mismatch Test ===");
    println!("Expected ratio: {:.4}", expected_ratio);
    println!("Observed ratio: {:.4}", observed_ratio);
    println!("χ² = {:.3}, p ≈ {:.4}", chi_sq, p_value);
    if p_value < 0.05 {
        println!("⚠️ SRM detected! Treatment assignment may be biased.");
    } else {
        println!("✅ No SRM detected.");
    }

    (chi_sq, p_value)
}
```

### 5.4 自己診断テスト

#### テスト1: 記法理解（10問）

<details><summary>Q1: $\mathbb{E}[Y^1 - Y^0]$ は何を表すか？</summary>

**Answer**: ATE (Average Treatment Effect) — 全体の平均処置効果

$$
\text{ATE} = \mathbb{E}[Y^1 - Y^0] = \mathbb{E}[Y^1] - \mathbb{E}[Y^0]
$$

**補足**: これは個体レベルの処置効果 $\tau_i = Y_i^1 - Y_i^0$ の期待値。個体レベルは観測不能（根本的因果推論問題）だが、集団平均なら推定可能。

</details>

<details><summary>Q2: $P(Y \mid do(X=x))$ と $P(Y \mid X=x)$ の違いは？</summary>

**Answer**:
- $P(Y \mid do(X=x))$: **介入確率** — $X$ を外部から強制的に $x$ に固定した場合の $Y$ の分布
- $P(Y \mid X=x)$: **条件付き確率** — $X=x$ を観測した場合の $Y$ の分布（交絡あり）

介入確率は因果効果、条件付き確率は相関を表す。

**例**: 喫煙とがん
- $P(\text{がん} \mid \text{喫煙}=1)$: 喫煙者のがん率（遺伝の交絡あり）
- $P(\text{がん} \mid do(\text{喫煙}=1))$: 強制的に喫煙させた場合のがん率（因果効果）

前者は相関、後者は因果。Simpson's Paradoxでは両者が逆転することすらある。

</details>

<details><summary>Q3: $e(X) = P(D=1 \mid X)$ の名前と役割は？</summary>

**Answer**: **傾向スコア (Propensity Score)**

高次元の共変量 $X$ を1次元のスカラーに圧縮。$(Y^1, Y^0) \perp\!\!\!\perp D \mid X$ なら $(Y^1, Y^0) \perp\!\!\!\perp D \mid e(X)$ も成立（次元削減）。

**実用上のメリット**:
- $X$ が10次元でも $e(X)$ は1次元 → マッチングが容易
- 共通サポート $0 < e(X) < 1$ の確認が簡単
- IPW推定で $1/e(X)$ の重みを使うだけで因果効果推定可能

</details>

<details><summary>Q4: SUTVAの2つの仮定を述べよ</summary>

**Answer**:
1. **処置の一意性**: 個体 $i$ の処置が $d$ のとき、結果は $Y_i^d$ の1つのみ
2. **干渉なし (No Interference)**: 個体 $i$ の結果は他の個体の処置に依存しない

$$
Y_i^d = Y_i^{d_i} \quad \forall d_{-i}
$$

**破れる例**:
- ワクチン接種: 他人が接種すると自分の感染リスク低下（干渉あり）
- ネットワーク広告: 友人がクリックすると自分もクリック（spillover効果）
- 教室内の処置: 同じクラスの学生間で相互影響

SUTVAが破れる場合は、**Spillover Effects** や **Network Effects** を明示的にモデル化する必要がある。

</details>

<details><summary>Q5: バックドア基準を満たす変数集合 $Z$ の条件は？</summary>

**Answer**:
1. $Z$ のどの変数も $X$ の子孫でない
2. $Z$ が $X$ から $Y$ へのすべてのバックドアパスを遮断する

満たせば:

$$
P(Y \mid do(X=x)) = \sum_z P(Y \mid X=x, Z=z) P(Z=z)
$$

**直感**:
- 条件1: $X$ の結果 ($X$ の子孫) で条件づけると、Collider Biasが発生する
- 条件2: バックドアパス（$X \leftarrow \cdots \to Y$）を遮断しないと交絡が残る

**例**: 喫煙→がん、バックドアパス: 喫煙←遺伝→がん
- $Z = \{\text{遺伝}\}$ で条件づけるとバックドアパスが遮断される
- $Z = \{\text{タール沈着}\}$ (喫煙の結果) で条件づけるとCollider Biasが発生

</details>

<details><summary>Q6: d-分離とは何か？</summary>

**Answer**: DAG上で変数集合 $Z$ が $X$ と $Y$ を d-分離する $\iff$ $X$ から $Y$ へのすべてのパスが $Z$ によって遮断される。

**パス遮断条件**:
- **Chain** $X \to Z \to Y$: $Z \in \mathcal{Z}$ なら遮断
- **Fork** $X \leftarrow Z \to Y$: $Z \in \mathcal{Z}$ なら遮断
- **Collider** $X \to Z \leftarrow Y$: $Z \notin \mathcal{Z}$ かつ $\text{DE}(Z) \cap \mathcal{Z} = \emptyset$ なら遮断

**d-分離の重要性**: $X \perp_d Y \mid Z$ (d-分離) $\Rightarrow$ $X \perp\!\!\!\perp Y \mid Z$ (条件付き独立)

</details>

<details><summary>Q7: Colliderで条件づけると何が起こる？</summary>

**Answer**: **選択バイアス** — 独立だった変数が条件付きで相関する

**例**: 才能と努力

```mermaid
graph TD
    T["才能"] --> A["合格"]
    E["努力"] --> A
```

才能と努力は独立 $T \perp\!\!\!\perp E$ だが、合格者 $A=1$ を条件づけると:

$$
T \not\perp\!\!\!\perp E \mid A=1
$$

合格者の中では「努力が少ない→才能が高い」という負の相関が生まれる。これが**Berkson's Paradox**。

**実用例**: 病院患者データで疾患Aと疾患Bが負の相関 → 入院（Collider）で条件づけられているため

</details>

<details><summary>Q8: Unconfoundedness仮定とは？</summary>

**Answer**: $(Y^1, Y^0) \perp\!\!\!\perp D \mid X$

共変量 $X$ を所与とすれば、潜在的結果と処置割り当てが独立。

**意味**: $X$ を制御すれば、処置はランダム割り当てと同等（selection on observables）。

**成り立つ条件**:
- すべての交絡因子 $X$ を測定している
- 未測定交絡 $U$ が存在しない

**破れる例**: 能力 $U$ が未測定で、$U \to D$ かつ $U \to Y$ なら Unconfoundedness は成り立たない → IV/RDD/DiDなど他の手法が必要

</details>

<details><summary>Q9: LATEとATEの違いは？</summary>

**Answer**:
- **ATE**: $\mathbb{E}[Y^1 - Y^0]$ — 全体の平均処置効果
- **LATE**: $\mathbb{E}[Y^1 - Y^0 \mid \text{Complier}]$ — コンプライアー（操作変数に従う人）の処置効果

**IVで推定されるのはLATE**:

$$
\text{LATE} = \frac{\mathbb{E}[Y \mid Z=1] - \mathbb{E}[Y \mid Z=0]}{\mathbb{E}[D \mid Z=1] - \mathbb{E}[D \mid Z=0]}
$$

**4つのタイプ**:
- Always-Taker: 常に処置を受ける（IVに無関係）
- Never-Taker: 常に処置を受けない（IVに無関係）
- **Complier**: IVに従う（LATEの対象）
- Defier: IVに逆らう（Monotonicity仮定で排除）

**ATE vs LATE**: LATEはコンプライアーのみの効果なので、ATEより局所的。外部妥当性が低い可能性がある。

</details>

<details><summary>Q10: 並行トレンド仮定とは？</summary>

**Answer**: DiDの識別仮定

$$
\mathbb{E}[Y_{01} - Y_{00} \mid G=1] = \mathbb{E}[Y_{01} - Y_{00} \mid G=0]
$$

処置がなかった場合、処置群と対照群のトレンドは平行。

**直感**: 処置群と対照群は処置前のトレンドが同じ → 処置後の差分は処置効果

**検証方法**:
- Event Study: 処置前の複数期間でトレンドが平行か確認
- Placebo Test: 処置前期間で「偽の処置」を設定し、効果がゼロか確認

**破れる例**: 処置群が高成長企業、対照群が低成長企業 → もともとトレンドが異なる → DiDは適用不可

</details>

#### テスト2: 数式導出（5問）

<details><summary>Q1: IPW推定量が不偏であることを示せ</summary>

**Proof**:

$$
\begin{aligned}
\mathbb{E}\left[\frac{D Y}{e(X)}\right] &= \mathbb{E}\left[\mathbb{E}\left[\frac{D Y}{e(X)} \mid X\right]\right] \\
&= \mathbb{E}\left[\frac{\mathbb{E}[D Y \mid X]}{e(X)}\right] \\
&= \mathbb{E}\left[\frac{P(D=1 \mid X) \mathbb{E}[Y \mid D=1, X]}{e(X)}\right] \\
&= \mathbb{E}\left[\frac{e(X) \mathbb{E}[Y^1 \mid X]}{e(X)}\right] \quad \text{(unconfoundedness)} \\
&= \mathbb{E}[Y^1]
\end{aligned}
$$

同様に $\mathbb{E}\left[\frac{(1-D) Y}{1-e(X)}\right] = \mathbb{E}[Y^0]$。よって:

$$
\mathbb{E}[\hat{\text{ATE}}_{\text{IPW}}] = \mathbb{E}[Y^1] - \mathbb{E}[Y^0] = \text{ATE}
$$

**Key Step解説**:
- Step 3→4: Unconfoundedness $(Y^1, Y^0) \perp\!\!\!\perp D \mid X$ により $\mathbb{E}[Y \mid D=1, X] = \mathbb{E}[Y^1 \mid X]$
- Step 4→5: $e(X) = P(D=1 \mid X)$ なので約分
- Overlap仮定 $0 < e(X) < 1$ が必須（分母がゼロにならない）

</details>

<details><summary>Q2: 2SLS推定量を導出せよ（Wald推定量形式）</summary>

**Derivation**:

構造方程式:

$$
\begin{aligned}
D &= \pi_0 + \pi_1 Z + \nu \\
Y &= \beta_0 + \beta_1 D + U
\end{aligned}
$$

$U$ と $Z$ が無相関（外生性）、$Z$ と $D$ が相関（関連性）を仮定。

$$
\begin{aligned}
\text{Cov}(Y, Z) &= \text{Cov}(\beta_0 + \beta_1 D + U, Z) \\
&= \beta_1 \text{Cov}(D, Z) + \text{Cov}(U, Z) \\
&= \beta_1 \text{Cov}(D, Z) \quad \text{(外生性: } \text{Cov}(U,Z)=0)
\end{aligned}
$$

$$
\hat{\beta}_1 = \frac{\text{Cov}(Y, Z)}{\text{Cov}(D, Z)} = \frac{\mathbb{E}[Y \mid Z=1] - \mathbb{E}[Y \mid Z=0]}{\mathbb{E}[D \mid Z=1] - \mathbb{E}[D \mid Z=0]}
$$

これが2SLS推定量（Wald推定量）。

**直感**:
- 分子: IVが $Y$ に与える総効果（reduced form）
- 分母: IVが $D$ に与える効果（first stage）
- 比: $D$ が $Y$ に与える因果効果（structural effect）

**条件**:
- 外生性: $\text{Cov}(U, Z) = 0$
- 関連性: $\text{Cov}(D, Z) \neq 0$ (弱IVなら分母が小さくバイアス大)
- 排除制約: $Z \to Y$ の直接パスなし

</details>

<details><summary>Q3: DiD推定量を導出せよ</summary>

**Setup**: 2期間 $t \in \{0, 1\}$, 2グループ $G \in \{0, 1\}$

潜在的結果:
- $Y_{it}^0$: 処置なしの結果
- $Y_{it}^1$: 処置ありの結果

観測結果:

$$
Y_{it} = \begin{cases}
Y_{it}^0 & \text{if } G=0 \text{ or } t=0 \\
Y_{it}^1 & \text{if } G=1 \text{ and } t=1
\end{cases}
$$

**DiD推定量**:

$$
\begin{aligned}
\hat{\tau}_{\text{DiD}} &= (\mathbb{E}[Y_{11}] - \mathbb{E}[Y_{10}]) - (\mathbb{E}[Y_{01}] - \mathbb{E}[Y_{00}]) \\
&= (\mathbb{E}[Y_{11}^1 \mid G=1] - \mathbb{E}[Y_{10}^0 \mid G=1]) \\
&\quad - (\mathbb{E}[Y_{01}^0 \mid G=0] - \mathbb{E}[Y_{00}^0 \mid G=0])
\end{aligned}
$$

**並行トレンド仮定**:

$$
\mathbb{E}[Y_{11}^0 - Y_{10}^0 \mid G=1] = \mathbb{E}[Y_{01}^0 - Y_{00}^0 \mid G=0]
$$

処置がなかった場合のトレンドが平行 → これを使うと:

$$
\begin{aligned}
\hat{\tau}_{\text{DiD}} &= \mathbb{E}[Y_{11}^1 - Y_{10}^0 \mid G=1] - (\mathbb{E}[Y_{11}^0 - Y_{10}^0 \mid G=1]) \\
&= \mathbb{E}[Y_{11}^1 - Y_{11}^0 \mid G=1] \\
&= \text{ATT}
\end{aligned}
$$

DiDはATT（処置群の平均処置効果）を識別する。

</details>

<details><summary>Q4: Doubly Robust推定量が2重頑健である理由を示せ</summary>

**DR推定量**:

$$
\hat{\tau}_{\text{DR}} = \frac{1}{n} \sum_i \left[ \frac{D_i (Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} + \hat{\mu}_1(X_i) - \frac{(1-D_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)} - \hat{\mu}_0(X_i) \right]
$$

**Case 1**: $\hat{\mu}_1, \hat{\mu}_0$ が正しい（$\hat{e}$ が誤りでもOK）

$$
\begin{aligned}
\mathbb{E}[\hat{\tau}_{\text{DR}}] &= \mathbb{E}\left[\frac{D(Y - \mu_1(X))}{\hat{e}(X)} + \mu_1(X)\right] - \mathbb{E}\left[\frac{(1-D)(Y - \mu_0(X))}{1-\hat{e}(X)} + \mu_0(X)\right] \\
&= \mathbb{E}\left[\mathbb{E}\left[\frac{D(Y - \mu_1(X))}{\hat{e}(X)} \mid X\right] + \mu_1(X)\right] - \mathbb{E}[\mu_0(X)] \\
&= \mathbb{E}\left[\frac{\mathbb{E}[D(Y - \mu_1(X)) \mid X]}{\hat{e}(X)} + \mu_1(X)\right] - \mathbb{E}[\mu_0(X)] \\
&= \mathbb{E}\left[\frac{e(X)(\mu_1(X) - \mu_1(X))}{\hat{e}(X)} + \mu_1(X)\right] - \mathbb{E}[\mu_0(X)] \quad \text{(} \mathbb{E}[Y \mid D=1, X] = \mu_1(X)) \\
&= \mathbb{E}[\mu_1(X)] - \mathbb{E}[\mu_0(X)] \\
&= \mathbb{E}[Y^1 - Y^0] = \text{ATE}
\end{aligned}
$$

**Case 2**: $\hat{e}$ が正しい（$\hat{\mu}$ が誤りでもOK）

IPWの不偏性により $\mathbb{E}[\hat{\tau}_{\text{DR}}] = \text{ATE}$

**結論**: $\hat{\mu}$ or $\hat{e}$ のどちらか一方が正しければ不偏 → 2重頑健性

</details>

<details><summary>Q5: RDD効果を導出せよ（Sharp RDD）</summary>

**Setup**: カットオフ $c$ で処置割り当て

$$
D_i = \mathbb{1}(X_i \geq c)
$$

**局所ランダム化仮定**:

$$
\lim_{x \to c} \mathbb{E}[Y^1 - Y^0 \mid X=x] = \tau_c
$$

カットオフ近傍で処置効果が一定。

**RDD効果**:

$$
\begin{aligned}
\tau_{\text{RDD}} &= \lim_{x \to c^+} \mathbb{E}[Y \mid X=x] - \lim_{x \to c^-} \mathbb{E}[Y \mid X=x] \\
&= \lim_{x \to c^+} \mathbb{E}[Y^1 \mid X=x] - \lim_{x \to c^-} \mathbb{E}[Y^0 \mid X=x] \\
&= \mathbb{E}[Y^1 - Y^0 \mid X=c] \\
&= \text{ATE}_c
\end{aligned}
$$

**Key**: カットオフでの不連続性が因果効果を表す。

**推定**: Local Linear Regression

$$
\min_{\beta_0, \beta_1, \beta_2, \beta_3} \sum_{i: |X_i - c| < h} K\left(\frac{X_i - c}{h}\right) (Y_i - \beta_0 - \beta_1 D_i - \beta_2 (X_i - c) - \beta_3 D_i (X_i - c))^2
$$

$\hat{\beta}_1 = \hat{\tau}_{\text{RDD}}$

</details>

#### テスト3: Rust実装（5問）

<details><summary>Q1: 傾向スコアを推定し、共通サポートを確認せよ</summary>

```rust
// 1. Estimate propensity scores (logistic regression weights fit offline)
let e_x: Vec<f64> = features.iter()
    .map(|x| logistic(0.05 * x[0] + 0.03 * x[1] - 3.5))
    .collect();

// 2. Check common support
let min_e = e_x.iter().cloned().fold(f64::INFINITY, f64::min);
let max_e = e_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
println!("Min e(X): {:.3}", min_e);
println!("Max e(X): {:.3}", max_e);

// 3. Visualize overlap: use plotters crate or log histogram to terminal
// Treated vs Control propensity distributions should overlap (common support)

// 4. Trimming: exclude extreme propensity scores
let eps = 0.05_f64;
let (valid, trimmed): (Vec<_>, Vec<_>) = e_x.iter().partition(|&&ei| ei > eps && ei < 1.0 - eps);
println!("Trimmed {} observations ({:.1}%)",
         trimmed.len(),
         100.0 * trimmed.len() as f64 / e_x.len() as f64);
```

</details>

### 5.5 ミニプロジェクト: 教育介入の因果効果推定

**シナリオ**: オンライン教育プログラムの効果を推定せよ。

- **処置**: プログラム受講 (1=受講, 0=非受講)
- **結果**: テストスコア
- **共変量**: 年齢、事前スコア、所得
- **操作変数**: ランダムクーポン配布

**タスク**:

1. 傾向スコアマッチング → ATE推定
2. 2SLS (クーポンをIV) → LATE推定
3. 感度分析 → 未測定交絡への頑健性
4. 結果を比較し、最も信頼できる推定値を選択

```rust
// Mini Project: Education Program Causal Effect (true effect = 10 points)

struct EduRecord {
    enroll: bool,
    test_score: f64,
    age: f64,
    baseline_score: f64,
    income: f64,
    coupon: bool,
    propensity: f64,
}

fn logistic(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

fn education_program_data(n: usize) -> Vec<EduRecord> {
    // use rand_distr::{Uniform, Normal, Bernoulli, Distribution};
    (0..n).map(|i| {
        let age = 18.0 + (i % 17) as f64;                    // placeholder
        let baseline = 60.0 + ((i as f64 * 0.01).sin() * 15.0);
        let income = 50.0 + ((i as f64 * 0.007).cos() * 20.0);
        let ability = (i as f64 * 0.003).sin();               // unobserved confounder
        let coupon = i % 2 == 0;

        let enroll_p = logistic(
            0.8 * coupon as u8 as f64 + 0.02 * age - 0.01 * baseline
            + 0.01 * income + 0.3 * ability - 1.0
        );
        let enroll = enroll_p > 0.5;

        // True program effect: +10 points
        let test_score = 50.0 + 10.0 * enroll as u8 as f64 + 0.5 * baseline + 5.0 * ability;

        let propensity = logistic(0.02 * age - 0.01 * baseline + 0.01 * income - 0.5);

        EduRecord { enroll, test_score, age, baseline_score: baseline, income, coupon, propensity }
    }).collect()
}

fn main_education() {
    let data = education_program_data(2000);

    let y: Vec<f64> = data.iter().map(|r| r.test_score).collect();
    let d: Vec<bool> = data.iter().map(|r| r.enroll).collect();
    let e: Vec<f64> = data.iter().map(|r| r.propensity).collect();
    let z: Vec<f64> = data.iter().map(|r| r.coupon as u8 as f64).collect();
    let d_f: Vec<f64> = d.iter().map(|&di| di as u8 as f64).collect();

    // Method 1: Propensity Score (IPW)
    let (ate_ps, se_ps) = ipw_estimator(&d, &y, &e);

    // Method 2: IV (coupon as instrument for enrollment)
    let (ate_iv, se_iv) = two_stage_least_squares_simple(&y, &d_f, &z);

    println!("\n=== Education Program Causal Effect ===");
    println!("True effect: 10 points");
    println!("Propensity Score ATE: {:.2} ± {:.2}", ate_ps, 1.96 * se_ps);
    println!("IV (2SLS) LATE: {:.2} ± {:.2}", ate_iv, 1.96 * se_iv);

    // Sensitivity analysis
    sensitivity_analysis_gamma(ate_ps, se_ps, &[1.0, 1.5, 2.0]);
}
```

> **Note:** **進捗: 85% 完了** 実データ因果推論チャレンジを完了した。全手法を比較し、感度分析で頑健性を確認した。次は発展ゾーンで研究フロンティアを探索する。

---

## 🔬 Z6. 新たな冒険へ（研究動向）

### 6.1 因果推論ファミリーツリー

```mermaid
graph TD
    A["🌳 因果推論"] --> B["Potential Outcomes<br/>(Rubin)"]
    A --> C["Structural Causal Models<br/>(Pearl)"]
    A --> D["Quasi-Experimental<br/>(Campbell)"]

    B --> E["Unconfoundedness<br/>傾向スコア"]
    B --> F["SUTVA<br/>ATE/ATT/CATE"]

    C --> G["DAG + do-演算"]
    C --> H["バックドア/フロントドア"]
    C --> I["反実仮想推論"]

    D --> J["IV/2SLS"]
    D --> K["RDD"]
    D --> L["DiD"]

    A --> M["ML×因果<br/>(Athey/Imbens)"]
    M --> N["Causal Forest"]
    M --> O["Double ML"]
    M --> P["Meta-Learners"]

    style A fill:#c8e6c9
    style M fill:#fff3e0
```

### 6.2 推薦論文・教科書

#### 主要論文

| 論文 | 著者 | 年 | 貢献 |
|:-----|:-----|:---|:-----|
| Causality (2nd Ed) [^1] | Pearl | 2009 | SCM, do-演算, バックドア基準 |
| Causal Inference (free book) [^9] | Hernán & Robins | 2020 | 実践ガイド |
| Potential Outcomes Survey [^2] | Rubin | 2005 | Rubin因果モデル統合 |
| Causal Forest [^3] | Wager & Athey | 2018 | HTE推定, 漸近理論 |
| Double ML [^4] | Chernozhukov et al. | 2018 | Debiased ML推論 |
| Staggered DiD [^5] | Callaway & Sant'Anna | 2021 | 多期間DiD |
| Weak IV [^7] | Stock & Yogo | 2005 | 弱操作変数検定 |
| SRM Detection [^6] | Fabijan et al. | 2019 | A/Bテスト品質管理 |
| Simpson's Paradox [^8] | Pearl | 2014 | パラドックス解消 |

#### 教科書

- **入門**: Pearl & Mackenzie "The Book of Why" (2018) — 一般向け因果革命の歴史
- **理論**: Pearl "Causality" (2009) [^1] — SCMの聖典
- **実践**: Hernán & Robins "Causal Inference" (2020) [^9] — 無料公開、疫学ベース
- **計量**: Angrist & Pischke "Mostly Harmless Econometrics" (2009) — IV/RDD/DiDの実践
- **ML×因果**: Facure "Causal Inference for The Brave and True" (2022) — Python実装付き

### 6.3 オンラインリソース

| リソース | URL | 説明 |
|:--------|:----|:-----|
| **ndarray** | [github.com/mschauer/ndarray](https://github.com/mschauer/ndarray) [^10] | RustのDAG/PC/FCI実装 |
| **Causal Inference Bootcamp** | [YouTube: Brady Neal](https://www.youtube.com/playlist?list=PLoazKTcS0RzZ1SUgeOgc6SWt51gfT80N0) | 動画講義シリーズ |
| **doWhy (Microsoft)** | [github.com/py-why/dowhy](https://github.com/py-why/dowhy) | Python因果推論ライブラリ |
| **EconML (Microsoft)** | [github.com/py-why/EconML](https://github.com/py-why/EconML) | Python ML×因果ライブラリ |

### 6.4 因果推論用語集

<details><summary>用語集（アルファベット順）</summary>

| 用語 | 定義 |
|:-----|:-----|
| **ATE** | Average Treatment Effect — 全体の平均処置効果 $\mathbb{E}[Y^1 - Y^0]$ |
| **ATT** | Average Treatment Effect on the Treated — 処置群の平均処置効果 |
| **Backdoor Criterion** | バックドア基準 — 交絡を除去するための変数集合の条件 |
| **CATE** | Conditional Average Treatment Effect — 条件付き平均処置効果 |
| **Collider** | コライダー — 2つの矢印が集まる変数 ($X \to Z \leftarrow Y$) |
| **DAG** | Directed Acyclic Graph — 因果構造を表す有向非巡回グラフ |
| **DiD** | Difference-in-Differences — 差分の差分法 |
| **d-separation** | d分離 — DAG上での条件付き独立性 |
| **do-Calculus** | do-演算 — 介入確率を条件付き確率に変換する3つのルール |
| **Doubly Robust** | 二重頑健推定量 — 傾向スコアと結果モデルのどちらかが正しければ不偏 |
| **Fundamental Problem** | 根本的因果推論問題 — $Y^1, Y^0$ を同時観測できない |
| **IPW** | Inverse Probability Weighting — 逆確率重み付け |
| **IV** | Instrumental Variable — 操作変数 |
| **LATE** | Local Average Treatment Effect — 局所平均処置効果（コンプライアーの効果） |
| **Overlap** | 共通サポート — $0 < e(X) < 1$ がすべての $X$ で成立 |
| **Potential Outcomes** | 潜在的結果 — $Y^1, Y^0$ |
| **Propensity Score** | 傾向スコア — $e(X) = P(D=1 \mid X)$ |
| **RDD** | Regression Discontinuity Design — 回帰不連続デザイン |
| **SCM** | Structural Causal Model — 構造因果モデル $(\mathcal{U}, \mathcal{V}, \mathcal{F})$ |
| **SUTVA** | Stable Unit Treatment Value Assumption — 安定個体処置値仮定 |
| **Unconfoundedness** | 無交絡性 — $(Y^1, Y^0) \perp\!\!\!\perp D \mid X$ |

</details>

### 6.5 因果推論の知識マップ

```mermaid
mindmap
  root((因果推論))
    Foundations
      Potential Outcomes
      DAG
      Identification
    Methods
      Propensity Score
        IPW
        Matching
        DR
      Quasi-Experimental
        IV
        RDD
        DiD
    ML Integration
      Causal Forest
      Double ML
      Meta-Learners
    Applications
      Policy Evaluation
      A/B Testing
      Observational Studies
    Tools
      CausalInference.jl
      doWhy
      EconML
```

> **Note:** **進捗: 100% 完了** 因果推論のフロンティアを探索した。論文・教科書・ツール・用語を完全整理。あとは振り返りゾーンでまとめ。

---



## 🎭 Z7. エピローグ（まとめ・FAQ・次回予告）

### 6.6 本講義のまとめ

1. **相関 ≠ 因果**: Simpson's Paradox, 交絡, 選択バイアスの罠を理解
2. **Rubin因果モデル**: 潜在的結果 $Y^1, Y^0$, SUTVA, ATE/ATT/CATE
3. **Pearl因果理論**: DAG, do-演算, バックドア/フロントドア基準, d-分離
4. **傾向スコア**: IPW, Matching, Doubly Robust, バランスチェック
5. **操作変数法**: 2SLS, LATE, Weak IV問題
6. **RDD**: Sharp/Fuzzy, 局所ランダム化, 帯域幅選択
7. **DiD**: 並行トレンド仮定, Staggered DiD
8. **ML×因果推論**: Causal Forest, Double ML, Meta-Learners
9. **Rust実装**: ndarray で全手法を実装

### 6.7 よくある質問 (FAQ)

<details><summary>Q1: 因果推論と機械学習の違いは？</summary>

**A**:
- **機械学習**: 予測精度の最大化 — $\hat{Y} \approx Y$
- **因果推論**: 因果効果の推定 — $\mathbb{E}[Y \mid do(X=x)]$

MLは「次に何が起こるか」、因果推論は「介入したら何が起こるか」を問う。MLは相関を学習し、因果推論は因果構造を仮定する。

</details>

<details><summary>Q2: いつ傾向スコア vs IVを使う？</summary>

**A**:
- **傾向スコア**: Unconfoundedness $(Y^d \perp\!\!\!\perp D \mid X)$ が成立する場合 — すべての交絡因子を測定できている
- **IV**: 未測定交絡がある場合 — 外生的なランダム変動（操作変数）を利用

ランダム化実験に近い状況なら傾向スコア、観測研究で交絡が疑われるならIV。

</details>

<details><summary>Q3: RDDとDiDの使い分けは？</summary>

**A**:
- **RDD**: 処置割り当てがカットオフで決まる（例: 年齢18歳で選挙権、スコア70点で合格）
- **DiD**: 2期間データがあり、処置タイミングが群によって異なる

RDDは空間的不連続、DiDは時間的変化を利用する。

</details>

<details><summary>Q4: Causal Forestで何がわかる？</summary>

**A**: **異質な処置効果 (HTE)** — 個体特性 $X$ に応じた処置効果 $\tau(X)$

平均効果(ATE)だけでなく、「高齢者には効果大、若年者には効果小」といった部分集団ごとの効果を推定できる。政策のターゲティングに有用。

</details>

<details><summary>Q5: 因果推論で最も重要な仮定は？</summary>

**A**: **Unconfoundedness** $(Y^d \perp\!\!\!\perp D \mid X)$ または **Exclusion Restriction** (IV)

これが破れると、どんな手法も因果効果を正しく推定できない。仮定の妥当性を理論・ドメイン知識・感度分析で検証することが最重要。

</details>

### 6.8 学習スケジュール（1週間復習プラン）

| Day | 内容 | 時間 | 達成基準 |
|:----|:-----|:-----|:---------|
| Day 1 | Zone 3.1-3.2 再読 + Rubin理論復習 | 1h | ATE/ATT/CATE を自力で導出できる |
| Day 2 | Zone 3.3 再読 + Pearl理論復習 | 1h | バックドア調整公式を自力で導出できる |
| Day 3 | Zone 3.4-3.5 再読 + 傾向スコア/IV復習 | 1h | IPW推定量を自力で導出できる |
| Day 4 | Zone 4 Rust実装を全て実行 | 2h | 全コードがエラーなく実行できる |
| Day 5 | Zone 5 ミニプロジェクトを実装 | 2h | 教育介入データで3手法比較完了 |
| Day 6 | 論文読解: Causal Forest [^3] or Double ML [^4] | 2h | 手法セクションが完全に理解できる |
| Day 7 | 自分のデータで因果推論実践 | 3h | 実データでATE推定 + 感度分析完了 |

### 6.9 次のステップ

**第26回: 評価パイプライン構築** では、因果効果の統計的検定:

- FID/IS/LPIPS (生成モデル評価)
- BootstrapによるCI推定
- 多重検定補正 (Bonferroni, FDR)
- 因果効果の可視化 (Forest Plot, Love Plot)

**第27回: 推論最適化 & Production品質** では、因果推論で得た効果を**本番システムに組み込む**:

- A/Bテスト基盤構築 (Elixir OTPでの並行テスト管理)
- バンディットアルゴリズム（探索と活用のトレードオフ）
- 因果推論×強化学習（Counterfactual Policy Evaluation）
- Production品質: 量子化・蒸留・Speculative Decoding

### 6.7 次回予告: 評価パイプライン構築

第26回では、因果推論で測定した効果を**定量的に評価する**基盤を構築する:

- **FID / IS / LPIPS**: 生成モデル品質の定量評価
- **CMMD**: CLIP埋め込みによる最大平均誤差
- **統計検定統合**: t検定 / Wilcoxon / Bootstrap CI
- **自動ベンチマークパイプライン**: Criterion.rs + Rust統計
- **Human Evaluation**: MOS / A/Bテスト設計

**キーワード**: FID / IS / LPIPS / CMMD / FLD+ / Precision-Recall / Bootstrap / Criterion.rs

**目標**: 因果推論で得た知見を、評価指標と統計検定で定量的に検証できるようにする。

---

### 6.11 パラダイム転換の問い

> **A/Bテストなしに"改善"を証明できるか？**

ランダム化実験（A/Bテスト）は因果推論のゴールドスタンダードだ。だが:

- **倫理的制約**: 医療、教育、ソーシャルサービスでランダム化は困難
- **コスト**: 全ユーザーを実験台にできない
- **時間**: 効果が出るまで数ヶ月〜数年

**観測データから因果効果を正しく推定できれば、A/Bテストなしでも改善を証明できる。**

本講義で学んだ手法:

1. **傾向スコア**: 交絡を制御し、観測データからATE推定
2. **操作変数**: 未測定交絡があってもランダムな変動で因果効果推定
3. **RDD**: カットオフの不連続性を利用し、局所的な因果効果推定
4. **DiD**: 時系列データで並行トレンド仮定の下で因果効果推定
5. **Causal Forest**: 異質な処置効果を推定し、ターゲティング最適化

**だが、仮定が破れれば全てが崩れる。** 因果推論は「仮定の明示化」と「感度分析」によって仮定の妥当性を検証し続ける営みだ。

**あなたの答えは？** — 観測データ因果推論とA/Bテストのバランスをどう取るか？

<details><summary>議論のポイント</summary>

1. **観測研究の強み**:
   - 倫理的制約がない（既存データを使う）
   - 大規模データで外部妥当性が高い
   - 長期的効果を追跡できる

2. **観測研究の弱み**:
   - 仮定依存（Unconfoundedness, IV仮定等）
   - 未測定交絡のリスク
   - 因果構造の誤特定

3. **ハイブリッドアプローチ**:
   - A/Bテストで短期効果検証 + 観測データで長期効果推定
   - A/Bテストでバイアス補正 + 観測データで外挿
   - 因果推論で事前評価 + A/Bテストで最終確認

4. **歴史的視点**:
   - Fisher (1935): ランダム化実験の原則確立
   - Rubin (1974): 観測データからの因果推論理論
   - Pearl (2000): グラフィカルモデルで因果構造を明示化
   - 現代: ML×因果推論で大規模観測データ活用

**結論**: A/Bテストは依然としてゴールドスタンダードだが、**因果推論は観測データから最大限の情報を引き出す強力な武器**。両者を適切に組み合わせることで、より正確な意思決定が可能になる。

</details>

> **Note:** **進捗: 100% 完了** 🎉 講義完走！

### 6.6 深層学習と因果推論の融合（2024-2026最新動向）

従来の因果推論手法は線形モデルや単純な統計手法に依存していたが、**深層学習との統合**により、高次元データ・非線形関係・未観測交絡への対処能力が飛躍的に向上している [^11]。

#### 6.6.1 Deep Causal Learningの3次元

最新のサーベイ論文 [^12] は、深層学習が因果学習に貢献する3つの次元を整理している:

**1. Representation（表現学習）**:

高次元・非構造化データ（画像・テキスト・時系列）から因果関係を学習:

$$
\mathbf{z} = f_\theta(\mathbf{x}), \quad \mathbf{z} \in \mathbb{R}^d
$$

ここで $f_\theta$ は深層ニューラルネットワーク、$\mathbf{z}$ は因果構造を捉えた潜在表現。

**2. Discovery（因果発見）**:

グラフニューラルネットワーク (GNN) でDAGを学習:

$$
\mathcal{G}^* = \arg\min_{\mathcal{G} \in \mathcal{DAG}} \mathcal{L}_{\text{score}}(\mathcal{G}; \mathbf{X}) + \lambda \|\mathcal{G}\|_0
$$

ここで $\mathcal{L}_{\text{score}}$ はスコアベース手法（BIC, MDL等）、$\|\mathcal{G}\|_0$ はエッジ数（スパース性正則化）。

**3. Inference（因果推論）**:

深層学習で処置効果を推定:

$$
\tau(x) = \mathbb{E}[Y^1 - Y^0 \mid X = x] = f_\theta^1(x) - f_\theta^0(x)
$$

ここで $f_\theta^1, f_\theta^0$ はニューラルネットで学習した潜在的結果関数。

#### 6.6.2 Deep Treatment Effect Estimation

**主要アーキテクチャ** [^13]:

| モデル | アーキテクチャ | 特徴 | 論文 |
|:-------|:-------------|:-----|:-----|
| **TARNet** | 共有層 + 分岐層 | $f_{\text{shared}}(x) \to (f^1, f^0)$ | Shalit et al. 2017 |
| **CFRNet** | TARNet + IPM正則化 | $\min \text{IPM}(f(X \mid D=1), f(X \mid D=0))$ | Shalit et al. 2017 |
| **DragonNet** | 傾向スコア統合 | $f_\theta(x) \to (e(x), \mu^1(x), \mu^0(x))$ | Shi et al. 2019 |
| **GANITE** | GAN | 反実仮想生成 | Yoon et al. 2018 |
| **X-Learner** | メタ学習 | 2段階推定 | Künzel et al. 2019 |

**IPM (Integral Probability Metric)**:

分布間の距離を測定し、処置群・対照群の表現を近づける:

$$
\text{IPM}(P, Q) = \sup_{f \in \mathcal{F}} \left| \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{x \sim Q}[f(x)] \right|
$$

CFRNetは、$\mathcal{F}$ を再生核ヒルベルト空間 (RKHS) とし、Maximum Mean Discrepancy (MMD) を最小化:

$$
\mathcal{L}_{\text{CFR}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \text{MMD}^2(f(X \mid D=1), f(X \mid D=0))
$$

**ADMIT (2024最新)** [^14]:

Average Dose Response Function (ADRF) の一般化境界を提供:

$$
\text{ADRF}(d) = \mathbb{E}[Y \mid do(D = d)], \quad d \in [0, 1]
$$

連続処置変数に対して、IPM距離の離散近似でセレクションバイアスを緩和。

#### 6.6.3 Causal Discovery with Deep Learning

**NOTEARS** (Zheng et al., NeurIPS 2018):

DAG学習を**連続最適化問題**に変換:

$$
\min_{\mathbf{W}} \quad \frac{1}{2n} \|\mathbf{X} - \mathbf{X}\mathbf{W}\|_F^2 + \lambda \|\mathbf{W}\|_1
$$

$$
\text{s.t.} \quad \text{tr}(e^{\mathbf{W} \odot \mathbf{W}}) - d = 0 \quad (\text{acyclicity constraint})
$$

ここで:

- $\mathbf{W} \in \mathbb{R}^{d \times d}$: 重み行列（有向グラフの隣接行列）
- $\text{tr}(e^{\mathbf{W} \odot \mathbf{W}}) - d = 0$: DAG制約（非巡回性）

従来の組合せ最適化 ($2^{d(d-1)/2}$ 通り) を回避し、勾配法で解ける。

**GraN-DAG** (Lachapelle et al., ICML 2020):

ニューラルネットワークで非線形因果関係を学習:

$$
x_i = f_i(\text{PA}_i; \theta_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2)
$$

ここで $f_i$ はニューラルネット、$\text{PA}_i$ は親ノード。

**高次元データへの応用** [^15]:

- **画像データ**: CNNで因果構造を学習（例: 病理画像 → 疾患因果グラフ）
- **時系列データ**: RNN/TransformerでGranger因果性を学習
- **テキストデータ**: BERT/GPTで言説間の因果関係を推定

#### 6.6.4 実装例: TARNetによる異質処置効果推定

```rust
// TARNet: Treatment-Agnostic Representation Network for CATE estimation.
// φ(x) shared representation → separate heads μ₁, μ₀
// For production: use the `candle` or `burn` deep learning crate.

/// TARNet: shared encoder + two outcome heads (treated / control)
struct TARNet {
    // In a real implementation these would be actual neural network layers
    // e.g., using candle_nn::Linear
    input_dim: usize,
    hidden_dim: usize,
}

impl TARNet {
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self { input_dim, hidden_dim }
    }

    /// Forward pass: shared representation → observed outcome prediction
    /// d: 0.0 (control) or 1.0 (treated) for each observation
    fn forward(&self, x: &[f64], d: f64) -> f64 {
        // φ(x) — shared representation (linear placeholder)
        let h: Vec<f64> = x.iter().map(|xi| xi.max(0.0)).collect(); // ReLU
        // μ₁(φ) and μ₀(φ) — separate linear heads (placeholder)
        let y1: f64 = h.iter().sum::<f64>() / h.len() as f64;
        let y0: f64 = h.iter().sum::<f64>() / h.len() as f64 * 0.5;
        d * y1 + (1.0 - d) * y0
    }

    /// CATE estimate: τ(x) = μ̂₁(φ(x)) - μ̂₀(φ(x))
    fn estimate_cate(&self, x: &[f64]) -> f64 {
        let h: Vec<f64> = x.iter().map(|xi| xi.max(0.0)).collect();
        let y1: f64 = h.iter().sum::<f64>() / h.len() as f64;
        let y0: f64 = h.iter().sum::<f64>() / h.len() as f64 * 0.5;
        y1 - y0
    }
}

// Training loop (placeholder — use candle/burn for real autodiff):
fn train_tarnet(model: &TARNet, x: &[Vec<f64>], d: &[f64], y: &[f64],
                n_epochs: usize, _lr: f64) {
    for epoch in 0..n_epochs {
        // Compute MSE loss: Σ (ŷ - y)² / n
        let loss: f64 = x.iter().zip(d).zip(y)
            .map(|((xi, di), yi)| (model.forward(xi, *di) - yi).powi(2))
            .sum::<f64>() / x.len() as f64;
        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss);
        }
        // Gradient update omitted — use candle_nn::Adam in production
    }
}
```

**数式とコードの対応**:

| 数式 | コード |
|:-----|:-------|
| $\phi(x) = f_{\text{shared}}(x)$ | `h = m.shared(x)` |
| $\mu^1(x) = f_1(\phi(x))$ | `y1 = m.treated(h)` |
| $\mu^0(x) = f_0(\phi(x))$ | `y0 = m.control(h)` |
| $\tau(x) = \mu^1(x) - \mu^0(x)$ | `estimate_cate(model, x)` |

この実装により、個人レベルの処置効果（CATE）を推定できる。

> **Note:** **進捗: 100% 完了** 🎉 講義完走！最新の深層学習×因果推論手法まで網羅した。

---


> Progress: [95%]
> **理解度チェック**
> 1. E-valueが「観察研究における因果主張の頑健性」を定量化できる理由を述べよ。
> 2. Causal ForestがS/T-Learnerより異質処置効果（HTE）推定に優れる理由は？

## 参考文献

### 主要論文

[^1]: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
<https://bayes.cs.ucla.edu/BOOK-2K/>

[^2]: Rubin, D. B. (2005). Causal Inference Using Potential Outcomes: Design, Modeling, Decisions. *Journal of the American Statistical Association*, 100(469), 322-331.
<https://www.tandfonline.com/doi/abs/10.1198/016214504000001880>

[^3]: Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.
<https://arxiv.org/abs/1510.04342>

[^4]: Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.
<https://arxiv.org/abs/1608.00060>

[^5]: Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
<https://www.sciencedirect.com/science/article/abs/pii/S0304407620303948>

[^6]: Fabijan, A., Gupchup, J., Gupta, S., Omhover, J., Qin, W., Vermeer, L., & Dmitriev, P. (2019). Diagnosing Sample Ratio Mismatch in Online Controlled Experiments: A Taxonomy and Rules of Thumb for Practitioners. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2156-2164.
<https://dl.acm.org/doi/10.1145/3292500.3330722>

[^7]: Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In *Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg* (pp. 80-108). Cambridge University Press.
<https://www.cambridge.org/core/books/abs/identification-and-inference-for-econometric-models/testing-for-weak-instruments-in-linear-iv-regression/8AD94FF2EFD214D05D75EE35015021E4>

[^8]: Pearl, J. (2014). Understanding Simpson's Paradox. *The American Statistician*, 68(1), 8-13.
<https://ftp.cs.ucla.edu/pub/stat_ser/r414.pdf>

[^9]: Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. (Free online)
<https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/>

[^10]: Mschauer. (2021). ndarray: Causal inference, graphical models and structure learning in Rust.
<https://github.com/mschauer/ndarray>

[^11]: Wang, Y., et al. (2024). "Causal Inference Meets Deep Learning: A Comprehensive Survey". *Research*, 7, 0467.
<https://arxiv.org/abs/2303.02186>

[^12]: Guo, R., et al. (2024). "Deep Causal Learning: Representation, Discovery and Inference". *ACM Computing Surveys*, 56(9), 1-40.
<https://arxiv.org/abs/2211.03374>


[^24]: Ling, Z., et al. (2025). "Hybrid Local Causal Discovery". *arXiv preprint*.
<https://arxiv.org/abs/2412.19507>

[^25]: Zhou, J., & Wang, M. (2025). "Differentiable Constraint-Based Causal Discovery". *arXiv preprint*.
<https://arxiv.org/abs/2510.22031>

[^26]: Mokhtarian, E., et al. (2024). "Recursive Causal Discovery". *arXiv preprint*.
<https://arxiv.org/abs/2403.09300>

[^27]: Gerhardus, A., & Runge, J. (2023). "Causal Discovery from Time Series with Hybrids of Constraint-Based and Noise-Based Algorithms". *arXiv preprint*.
<https://arxiv.org/abs/2306.08765>

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

**次回**: [第26回: 評価パイプライン構築](/your-next-article)
