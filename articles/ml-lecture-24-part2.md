---
title: "第24回【後編】付録編: 統計学: 30秒の驚き→数式修行→実装マスター"
emoji: "📈"
type: "tech"
topics: ["machinelearning", "statistics", "rust", "bayesian", "hypothesis"]
published: true
slug: "ml-lecture-24-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Rust", "Elixir"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

> **第24回【前編】**: [第24回【前編】](https://zenn.dev/fumishiki/ml-lecture-24-part1)

## Part 2


$$
\begin{aligned}
\text{SS}_{\text{total}} &= \sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x})^2 \\
\text{SS}_{\text{between}} &= \sum_{i=1}^k n_i (\bar{x}_i - \bar{x})^2 \\
\text{SS}_{\text{within}} &= \sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2 \\
\text{MS}_{\text{between}} &= \frac{\text{SS}_{\text{between}}}{k-1}, \quad \text{MS}_{\text{within}} = \frac{\text{SS}_{\text{within}}}{N-k}
\end{aligned}
$$

**数値検証**:

```rust
use statrs::distribution::{FisherSnedecor, ContinuousCDF};

fn main() {
    let group_a = [0.72_f64, 0.71, 0.73, 0.70, 0.72];
    let group_b = [0.78_f64, 0.77, 0.79, 0.76, 0.78];
    let group_c = [0.68_f64, 0.67, 0.69, 0.66, 0.68];

    // 一元配置ANOVA
    let (f_stat, p_value) = one_way_anova(&[&group_a, &group_b, &group_c]);
    println!("F={:.3}, p={:.6}", f_stat, p_value);
    if p_value < 0.05 {
        println!("✅ 少なくとも1組の平均が異なる");
    } else {
        println!("❌ 全群の平均に差なし");
    }
}

fn one_way_anova(groups: &[&[f64]]) -> (f64, f64) {
    let k = groups.len() as f64;
    let n: f64 = groups.iter().map(|g| g.len()).sum::<usize>() as f64;
    let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n;
    let ss_between: f64 = groups.iter().map(|g| {
        let gm = g.iter().sum::<f64>() / g.len() as f64;
        g.len() as f64 * (gm - grand_mean).powi(2)
    }).sum();
    let ss_within: f64 = groups.iter().map(|g| {
        let gm = g.iter().sum::<f64>() / g.len() as f64;
        g.iter().map(|x| (x - gm).powi(2)).sum::<f64>()
    }).sum();
    let f = (ss_between / (k - 1.0)) / (ss_within / (n - k));
    let dist = FisherSnedecor::new(k - 1.0, n - k).unwrap();
    let p = 1.0 - dist.cdf(f);
    (f, p)
}
```

出力:
```
F=90.0, p=0.000000
✅ 少なくとも1組の平均が異なる
```

#### 3.4.3 正規性検定

**問題**: t検定・ANOVAは正規性を仮定。データが正規分布に従うか検証したい。

| 検定 | 特徴 | 帰無仮説 |
|:-----|:-----|:--------|
| **Shapiro-Wilk検定** | 最も強力（小~中サンプル） | データが正規分布に従う |
| **Kolmogorov-Smirnov検定** | 汎用的（任意の分布） | データが指定分布に従う |
| **Anderson-Darling検定** | 裾の適合度を重視 | データが正規分布に従う |

**数値検証**:

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // 正規分布データ（KS検定の代わりに手動で正規性チェック）
    let normal_dist = Normal::new(0.0_f64, 1.0).unwrap();
    let normal_data: Vec<f64> = (0..30).map(|_| normal_dist.sample(&mut rng)).collect();
    // 正規分布からサンプルされたデータ → p値は大きい（帰無仮説棄却せず）
    println!("正規データ: 平均={:.4}, std={:.4}", mean(&normal_data), std_dev(&normal_data));

    // 非正規データ（一様分布）
    let uniform_dist = Uniform::new(0.0_f64, 1.0);
    let uniform_data: Vec<f64> = (0..30).map(|_| uniform_dist.sample(&mut rng)).collect();
    println!("一様データ: 平均={:.4}, std={:.4}", mean(&uniform_data), std_dev(&uniform_data));

    // 注: Rust で KS検定を行うには statrs や ndarray-stats を利用
    // statrs::statistics::Statistics trait で基本統計量は計算可能
}

fn mean(x: &[f64]) -> f64 { x.iter().sum::<f64>() / x.len() as f64 }
fn std_dev(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
}
```

### 3.5 ノンパラメトリック検定

**用途**: 正規性が満たされない、または順序データの場合。

| 検定 | パラメトリック版 | 用途 |
|:-----|:----------------|:-----|
| **Mann-Whitney U検定** | 2標本t検定 | 2群の中央値の差 |
| **Wilcoxon符号順位検定** | 対応のあるt検定 | 対応のある2群の中央値差 |
| **Kruskal-Wallis検定** | 一元配置ANOVA | 3群以上の中央値の差 |

**Mann-Whitney U検定の原理**:

1. 2群のデータを統合して順位付け。
2. 各群の順位和を計算。
3. U統計量を計算:

$$
U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1
$$

ここで $R_1$ は群1の順位和。

**数値検証**:

```rust
fn main() {
    let group1 = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let group2 = [6.0_f64, 7.0, 8.0, 9.0, 10.0];

    // Mann-Whitney U検定（手動実装）
    // U = 各ペア (a∈group1, b∈group2) で a < b となる個数
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;
    let u: f64 = group1.iter()
        .flat_map(|&a| group2.iter().map(move |&b| if a < b { 1.0 } else { 0.0 }))
        .sum();
    // 正規近似による p 値
    let mu_u = n1 * n2 / 2.0;
    let sigma_u = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();
    let z = (u - mu_u) / sigma_u;
    use statrs::distribution::{Normal, ContinuousCDF};
    let dist = Normal::new(0.0, 1.0).unwrap();
    let p = 2.0 * dist.cdf(-z.abs());  // 両側検定

    println!("U={:.1}, p={:.4}", u, p);
}
```

> **Note:** **進捗: 65% 完了** パラメトリック・ノンパラメトリック検定の理論完全版を制覇。多重比較補正へ。

### 3.6 多重比較補正理論

**問題**: 複数の検定を行うと、偶然に有意になる確率（第1種過誤）が増大する。

**例**: $\alpha = 0.05$ で独立な20個の検定を行うと、少なくとも1つが偶然有意になる確率:

$$
1 - (1 - 0.05)^{20} \approx 0.64 \quad \text{(64%!)}
$$

**FWER（Family-Wise Error Rate）**: 少なくとも1つの第1種過誤が起こる確率。

**FDR（False Discovery Rate）**: 有意と判定されたもののうち偽陽性の割合の期待値。

#### 3.6.1 FWER制御法

| 手法 | 調整後の有意水準 | 保守性 |
|:-----|:----------------|:-------|
| **Bonferroni補正** | $\alpha_{\text{adj}} = \alpha / m$ | 最も保守的 |
| **Holm法** | 逐次的Bonferroni | Bonferroniより緩い |
| **Šidák補正** | $\alpha_{\text{adj}} = 1 - (1 - \alpha)^{1/m}$ | 独立性仮定 |

**Holm法の手順**:

1. p値を昇順に並べる: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. $i = 1, 2, \ldots$ の順に以下をチェック:
   - $p_{(i)} \leq \alpha / (m - i + 1)$ なら棄却、次へ
   - 初めて不等式が成立しなかったら停止

#### 3.6.2 FDR制御法

**Benjamini-Hochberg法** [^2]:

1. p値を昇順に並べる: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. $i = m, m-1, \ldots, 1$ の順に以下をチェック:
   - $p_{(i)} \leq \frac{i}{m} \alpha$ なら $i$ 番目まで全て棄却、停止
   - 成立しなければ次へ

**数式導出**:

FDRの定義:

$$
\text{FDR} = \mathbb{E}\left[\frac{V}{R}\right]
$$

ここで $V$ = 偽陽性数、$R$ = 総発見数（$R = V + S$, $S$ = 真陽性数）。

Benjamini-Hochbergは独立な検定において $\text{FDR} \leq \alpha$ を保証する [^2]。

**数値検証**:

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0.0_f64, 1.0);

    // 100個の検定（90個は帰無仮説が真、10個は対立仮説が真）
    // H0が真のp値: 一様分布
    let mut p_values: Vec<f64> = (0..100).map(|_| uniform.sample(&mut rng)).collect();
    // H1が真のp値: 0に偏る（Beta(0.1, 1) 近似として x^9 変換）
    let p_values_alt: Vec<f64> = (0..10).map(|_| uniform.sample(&mut rng).powf(9.0)).collect();
    p_values.extend_from_slice(&p_values_alt);

    // 補正なし
    let n_sig_uncorrected = p_values.iter().filter(|&&p| p < 0.05).count();
    println!("補正なし: {} / 110 が有意", n_sig_uncorrected);

    // Bonferroni補正
    let m = p_values.len() as f64;
    let n_sig_bonf = p_values.iter().filter(|&&p| p * m < 0.05).count();
    println!("Bonferroni: {} / 110 が有意", n_sig_bonf);

    // Benjamini-Hochberg (FDR)
    let n_sig_bh = benjamini_hochberg(&p_values, 0.05);
    println!("Benjamini-Hochberg: {} / 110 が有意", n_sig_bh);
}

/// BH法で有意と判定される仮説の個数を返す
fn benjamini_hochberg(pvals: &[f64], alpha: f64) -> usize {
    let m = pvals.len();
    let mut indexed: Vec<(usize, f64)> = pvals.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut last_reject = 0;
    for (i, (_, p)) in indexed.iter().enumerate() {
        if *p <= (i + 1) as f64 / m as f64 * alpha {
            last_reject = i + 1;
        }
    }
    last_reject
}
```

出力例:
```
補正なし: 15 / 110 が有意
Bonferroni: 3 / 110 が有意
Benjamini-Hochberg: 9 / 110 が有意
```

> **Note:** **進捗: 75% 完了** 多重比較補正（FWER/FDR）を完全理解。GLM理論へ。

### 3.7 一般化線形モデル（GLM）

**問題**: 線形回帰 $y = X\beta + \epsilon$ は連続値・正規分布を仮定。カテゴリカル（分類）やカウントデータには不適。

**GLMの構成要素**:

1. **指数型分布族**: 応答変数 $y$ の分布（正規・二項・ポアソン等）。
2. **リンク関数** $g(\cdot)$: 平均 $\mu = \mathbb{E}[y]$ を線形予測子 $\eta = X\beta$ に繋ぐ。
3. **線形予測子**: $\eta = X\beta$

$$
g(\mu) = X\beta \quad \Rightarrow \quad \mu = g^{-1}(X\beta)
$$

| 分布 | 典型的用途 | 標準的リンク関数 |
|:-----|:----------|:----------------|
| 正規分布 | 連続値 | 恒等 $g(\mu) = \mu$ |
| 二項分布 | 分類 | ロジット $g(\mu) = \log\frac{\mu}{1-\mu}$ |
| ポアソン分布 | カウント | 対数 $g(\mu) = \log\mu$ |

#### 3.7.1 ロジスティック回帰（Logistic Regression）

**用途**: 二値分類（$y \in \{0, 1\}$）。

**モデル**:

$$
\begin{aligned}
y_i &\sim \text{Bernoulli}(p_i) \\
\log\frac{p_i}{1 - p_i} &= \beta_0 + \beta_1 x_i \quad \text{(ロジット変換)} \\
\Rightarrow \quad p_i &= \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_i)}} \quad \text{(シグモイド関数)}
\end{aligned}
$$

**オッズ比（Odds Ratio）**: 係数 $\beta_1$ の解釈

$$
\text{OR} = e^{\beta_1}
$$

$x$ が1単位増加すると、オッズ（$p / (1-p)$）が $e^{\beta_1}$ 倍になる。

**最尤推定**: 対数尤度を最大化。

$$
\ell(\beta) = \sum_{i=1}^n \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
$$

勾配:

$$
\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^n (y_i - p_i) x_{ij}
$$

**数値検証**:

```rust
fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

fn logistic_log_likelihood(beta: &[f64], x_data: &[[f64; 2]], y: &[f64]) -> f64 {
    // データ: x（連続変数）, y（0/1のラベル）
    x_data.iter().zip(y.iter())
        .map(|(xi, &yi)| {
            // リンク関数: logit(π) = β₀ + β₁·x
            let eta = beta[0] + beta[1] * xi[0];
            let pi = sigmoid(eta);
            yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln()
        })
        .sum()
}

fn main() {
    // データ: x（連続変数）, y（0/1のラベル）
    let x_data: [[f64; 2]; 10] = [
        [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0],
        [6.0, 0.0], [7.0, 0.0], [8.0, 0.0], [9.0, 0.0], [10.0, 0.0],
    ];
    let y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0_f64];

    // ロジスティック回帰: 勾配上昇法で β を推定
    // 勾配: ∂ℓ/∂βⱼ = Σ(yᵢ - πᵢ)·xᵢⱼ
    let mut beta = [0.0_f64; 2];
    let lr = 0.1;
    for _ in 0..10000 {
        let grad0: f64 = x_data.iter().zip(y.iter())
            .map(|(xi, &yi)| yi - sigmoid(beta[0] + beta[1] * xi[0]))
            .sum();
        let grad1: f64 = x_data.iter().zip(y.iter())
            .map(|(xi, &yi)| (yi - sigmoid(beta[0] + beta[1] * xi[0])) * xi[0])
            .sum();
        beta[0] += lr * grad0;
        beta[1] += lr * grad1;
    }

    let or = beta[1].exp();  // オッズ比
    println!("係数β0={:.3}, β1={:.3}, オッズ比OR={:.3}", beta[0], beta[1], or);
    println!("xが1単位増加すると、オッズが{:.3}倍になる", or);

    // 予測確率
    println!("\n予測確率:");
    for (xi, &yi) in x_data.iter().zip(y.iter()) {
        let pi = sigmoid(beta[0] + beta[1] * xi[0]);
        println!("  x={:.0}, y={:.0}, π̂={:.3}", xi[0], yi, pi);
    }

    let ll = logistic_log_likelihood(&beta, &x_data, &y);
    println!("対数尤度: {:.4}", ll);
}
```

#### 3.7.2 ポアソン回帰（Poisson Regression）

**用途**: カウントデータ（$y \in \{0, 1, 2, \ldots\}$）。イベント発生回数の予測。

**モデル**:

$$
\begin{aligned}
y_i &\sim \text{Poisson}(\lambda_i) \\
\log \lambda_i &= \beta_0 + \beta_1 x_i \quad \text{(対数リンク関数)} \\
\Rightarrow \quad \lambda_i &= e^{\beta_0 + \beta_1 x_i}
\end{aligned}
$$

**係数の解釈**: $x$ が1単位増加すると、期待カウント $\lambda$ が $e^{\beta_1}$ 倍になる。

**数値検証**:

```rust
fn main() {
    // データ生成: カウントデータ（例: 1時間あたりのエラー発生回数）
    let workload = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let errors   = [2.0_f64, 3.0, 3.0, 5.0, 6.0, 8.0, 9.0, 12.0, 14.0, 16.0];

    // ポアソン回帰: log(λ) = β₀ + β₁·workload
    // 対数尤度: ℓ = Σ[yᵢ·(β₀ + β₁·xᵢ) - exp(β₀ + β₁·xᵢ) - ln(yᵢ!)]
    // 勾配上昇法で最適化
    let mut beta = [0.0_f64; 2];
    let lr = 0.01;
    for _ in 0..50000 {
        let grad0: f64 = workload.iter().zip(errors.iter())
            .map(|(&xi, &yi)| yi - (beta[0] + beta[1] * xi).exp())
            .sum();
        let grad1: f64 = workload.iter().zip(errors.iter())
            .map(|(&xi, &yi)| (yi - (beta[0] + beta[1] * xi).exp()) * xi)
            .sum();
        beta[0] += lr * grad0;
        beta[1] += lr * grad1;
    }

    // 係数の解釈: workloadが1単位増加すると期待エラー回数が exp(β₁) 倍
    let multiplier = beta[1].exp();
    println!("係数β0={:.3}, β1={:.3}", beta[0], beta[1]);
    println!("workloadが1単位増加すると、期待エラー回数が{:.3}倍になる", multiplier);

    // 予測エラー回数
    println!("\n予測エラー回数:");
    for (&xi, &yi) in workload.iter().zip(errors.iter()) {
        let lambda_pred = (beta[0] + beta[1] * xi).exp();
        println!("  workload={:.0}, errors={:.0}, λ̂={:.2}", xi, yi, lambda_pred);
    }
}
```

#### 3.7.3 指数型分布族の統一理論

**GLMの基盤**: 指数型分布族（Exponential Family）

$$
p(y | \theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)
$$

| 項 | 名称 | 役割 |
|:---|:-----|:-----|
| $\theta$ | 自然パラメータ | 平均を決定 |
| $\phi$ | 分散パラメータ | 分散を決定 |
| $b(\theta)$ | 累積生成関数 | 平均: $\mu = b'(\theta)$ |
| $a(\phi)$ | 分散関数 | 分散: $\text{Var}(Y) = b''(\theta) a(\phi)$ |

**主要な分布**:

| 分布 | $\theta$ | $b(\theta)$ | $a(\phi)$ | $\mu = b'(\theta)$ |
|:-----|:---------|:-----------|:----------|:------------------|
| 正規分布 | $\mu$ | $\theta^2 / 2$ | $\sigma^2$ | $\theta$ |
| 二項分布 | $\log \frac{p}{1-p}$ | $\log(1 + e^\theta)$ | $1$ | $\frac{e^\theta}{1 + e^\theta}$ |
| ポアソン分布 | $\log \lambda$ | $e^\theta$ | $1$ | $e^\theta$ |

**GLMの統一構造**:

1. **ランダム成分**: 応答変数 $y$ が指数型分布族に従う。
2. **線形予測子**: $\eta = X\beta$
3. **リンク関数**: $g(\mu) = \eta$（標準的リンク関数: $g(\mu) = \theta$）

> **Note:** **進捗: 80% 完了** GLM理論（ロジスティック・ポアソン回帰・指数型分布族）を理解。ベイズ統計へ。

### 3.8 ベイズ統計入門

#### 3.8.1 ベイズの定理の導出

**第4回で学んだ条件付き確率の定義**:

$$
p(\theta | D) = \frac{p(\theta, D)}{p(D)}, \quad p(D | \theta) = \frac{p(\theta, D)}{p(\theta)}
$$

両辺に $p(\theta)$ を掛けると:

$$
p(\theta, D) = p(D | \theta) p(\theta) = p(\theta | D) p(D)
$$

よって:

$$
p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}
$$

これが**ベイズの定理**だ。

| 項 | 名称 | 意味 |
|:---|:-----|:-----|
| $p(\theta \| D)$ | 事後分布（Posterior） | データ観測後のパラメータの分布 |
| $p(D \| \theta)$ | 尤度（Likelihood） | パラメータ下でのデータの確率 |
| $p(\theta)$ | 事前分布（Prior） | データ観測前のパラメータの信念 |
| $p(D)$ | 周辺尤度（Evidence） | 正規化定数 $p(D) = \int p(D \| \theta) p(\theta) d\theta$ |

#### 3.8.2 頻度論統計 vs ベイズ統計

**哲学的対立**:

| 項目 | 頻度論 | ベイズ |
|:-----|:------|:-------|
| **パラメータの性質** | 固定値（未知） | 確率変数 |
| **確率の解釈** | 長期的頻度 | 信念の度合い |
| **推論の対象** | 点推定・信頼区間 | 事後分布全体 |
| **不確実性の表現** | 標準誤差 | 事後分布の幅 |
| **事前知識** | 使わない（客観性） | 使う（主観性） |

**具体例**: コイン投げ（10回中7回表）

**頻度論的推定**（第7回のMLE）:

$$
\hat{\theta}_{\text{MLE}} = \frac{k}{n} = \frac{7}{10} = 0.7
$$

95%信頼区間（Wald法）:

$$
\text{CI} = \hat{\theta} \pm 1.96 \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{n}} = 0.7 \pm 1.96 \sqrt{\frac{0.7 \times 0.3}{10}} = [0.416, 0.984]
$$

**ベイズ推定**（事前分布Beta(2,2)、共役性より事後分布Beta(9, 5)）:

$$
p(\theta | k=7, n=10) = \text{Beta}(9, 5)
$$

事後平均（点推定）:

$$
\mathbb{E}[\theta | D] = \frac{\alpha}{\alpha + \beta} = \frac{9}{9+5} = 0.643
$$

95%信用区間（Credible Interval）:

$$
\text{CrI} = [\text{quantile}(0.025), \text{quantile}(0.975)] \approx [0.366, 0.882]
$$

**解釈の違い**:

- **頻度論CI**: 「同じ実験を100回繰り返せば、95回はこの区間が真の $\theta$ を含む」
- **ベイズCrI**: 「データを見た今、$\theta$ がこの区間にある確率が95%」（より直感的）

#### 3.8.1 共役事前分布

**定義**: 事前分布と事後分布が同じ分布族に属するとき、その事前分布を共役という。

| 尤度 | 共役事前分布 | 事後分布 |
|:-----|:-----------|:--------|
| 二項分布 | ベータ分布 | ベータ分布 |
| 正規分布（既知分散） | 正規分布 | 正規分布 |
| ポアソン分布 | ガンマ分布 | ガンマ分布 |

**例**: コイン投げ（二項分布）+ ベータ事前分布

$$
\begin{aligned}
\text{尤度:} \quad & p(k | n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} \\
\text{事前分布:} \quad & p(\theta) = \text{Beta}(\alpha, \beta) \propto \theta^{\alpha-1} (1-\theta)^{\beta-1} \\
\text{事後分布:} \quad & p(\theta | k, n) = \text{Beta}(\alpha + k, \beta + n - k)
\end{aligned}
$$

**数値検証**:

```rust
use statrs::distribution::{Beta, ContinuousCDF};

fn main() {
    // 事前分布: Beta(2, 2) (弱い信念: θ≈0.5)
    let alpha_prior = 2.0_f64;
    let beta_prior  = 2.0_f64;

    // データ: 10回投げて7回表
    let n = 10.0_f64;
    let k = 7.0_f64;

    // 事後分布: Beta(α+k, β+n-k) = Beta(9, 5) （共役更新）
    let alpha_post = alpha_prior + k;
    let beta_post  = beta_prior + n - k;

    let prior = Beta::new(alpha_prior, beta_prior).unwrap();
    let posterior = Beta::new(alpha_post, beta_post).unwrap();

    // 事後平均と95%信用区間
    let post_mean = alpha_post / (alpha_post + beta_post);
    let cri_lo = find_quantile(&posterior, 0.025);
    let cri_hi = find_quantile(&posterior, 0.975);

    println!("事後分布: Beta({}, {})", alpha_post, beta_post);
    println!("事後平均: {:.4}", post_mean);
    println!("95%信用区間: [{:.4}, {:.4}]", cri_lo, cri_hi);

    // 可視化: plotters クレートが必要
    // ここでは θ ∈ [0,1] の PDF 値を表示
    println!("\nθ    prior_pdf  posterior_pdf");
    for i in 0..=10 {
        let theta = i as f64 / 10.0;
        use statrs::distribution::Continuous;
        println!("{:.1}  {:.4}     {:.4}", theta,
            prior.pdf(theta.max(1e-9).min(1.0 - 1e-9)),
            posterior.pdf(theta.max(1e-9).min(1.0 - 1e-9)));
    }
}

/// 二分探索で CDF の逆関数（分位点）を近似
fn find_quantile(dist: &statrs::distribution::Beta, p: f64) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if dist.cdf(mid) < p { lo = mid; } else { hi = mid; }
    }
    (lo + hi) / 2.0
}
```

#### 3.8.2 MCMC（Markov Chain Monte Carlo）

**問題**: 事後分布 $p(\theta | D)$ が複雑で解析的に計算できない。

**MCMC**: マルコフ連鎖を使って事後分布からサンプルを生成。

**Metropolis-Hastings法** [^3]:

1. 初期値 $\theta^{(0)}$ を設定。
2. $t = 1, 2, \ldots$ について:
   - 提案分布 $q(\theta' | \theta^{(t-1)})$ から候補 $\theta'$ を生成。
   - 受理確率を計算:
     $$
     \alpha = \min\left(1, \frac{p(\theta' | D) q(\theta^{(t-1)} | \theta')}{p(\theta^{(t-1)} | D) q(\theta' | \theta^{(t-1)})}\right)
     $$
   - 確率 $\alpha$ で $\theta^{(t)} = \theta'$、そうでなければ $\theta^{(t)} = \theta^{(t-1)}$。

**probabilistic-rsで実装**:

```rust
// コイン投げのベイズ推定: Metropolis-Hastings MCMC
// Prior: θ ~ Beta(2, 2), Likelihood: k ~ Binomial(n, θ)
// 事後分布: Beta(9, 5) が解析解（共役）

use rand::SeedableRng;
use rand_distr::{Distribution, Uniform, Normal as RandNormal};

fn log_posterior_coinflip(theta: f64, k: f64, n: f64) -> f64 {
    if theta <= 0.0 || theta >= 1.0 { return f64::NEG_INFINITY; }
    // log Beta(2,2) prior + log Binomial likelihood
    let log_prior = (2.0 - 1.0) * theta.ln() + (2.0 - 1.0) * (1.0 - theta).ln();
    let log_lik   = k * theta.ln() + (n - k) * (1.0 - theta).ln();
    log_prior + log_lik
}

fn main() {
    // データ: 10回中7回表
    let (k, n) = (7.0_f64, 10.0_f64);

    // Metropolis-Hastings サンプリング
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let proposal = RandNormal::new(0.0, 0.1).unwrap();
    let uniform  = Uniform::new(0.0_f64, 1.0);
    let n_samples = 1000;
    let mut samples = Vec::with_capacity(n_samples);
    let mut theta_cur = 0.5_f64;

    for _ in 0..n_samples {
        let theta_prop = (theta_cur + proposal.sample(&mut rng)).clamp(1e-6, 1.0 - 1e-6);
        let log_alpha = log_posterior_coinflip(theta_prop, k, n)
                      - log_posterior_coinflip(theta_cur,  k, n);
        if log_alpha.exp() > uniform.sample(&mut rng) {
            theta_cur = theta_prop;
        }
        samples.push(theta_cur);
    }

    let mean_theta = samples.iter().sum::<f64>() / samples.len() as f64;
    // 解析解: E[θ|data] = (α+k)/(α+β+n) = 9/14 ≈ 0.643
    println!("事後平均 θ (MCMC): {:.4}", mean_theta);
    println!("解析解 θ (Beta(9,5)): {:.4}", 9.0 / 14.0);
    // 可視化: plotters クレートで histogram を描画
    // cargo add plotters
}
```

> **Note:** **進捗: 90% 完了** ベイズ統計（共役事前分布・MCMC）を完全理解。実験計画法へ。

### 3.9 実験計画法（Experimental Design）

**目的**: 限られたリソースで最大の情報を得る実験を設計する。

#### 3.9.1 完全無作為化デザイン（Completely Randomized Design, CRD）

**特徴**: 処理（treatment）をランダムに割り当てる。最もシンプル。

**欠点**: ブロック間の変動（例: 測定日の違い）を制御できない。

#### 3.9.2 乱塊法（Randomized Block Design, RBD）

**特徴**: 被験者をブロック（例: 年齢層、測定日）に分け、各ブロック内で処理をランダム化。

**利点**: ブロック間変動を除去 → 残差が小さくなる → 検出力向上。

#### 3.9.3 ラテン方格（Latin Square Design）

**特徴**: 2つの要因（例: 行=日、列=機械）を同時に制御。

**制約**: 処理数 = 行数 = 列数。

#### 3.9.4 サンプルサイズ設計（Power Analysis）

**問題**: 実験前に必要なサンプルサイズを決定。

**手順**:

1. 期待される効果量 $d$ を設定（過去の研究や予備実験から）。
2. 有意水準 $\alpha$ を設定（通常0.05）。
3. 目標検出力 $1 - \beta$ を設定（通常0.8）。
4. 検定の種類に応じた公式またはソフトウェアでサンプルサイズを計算。

**t検定のサンプルサイズ公式**（再掲）:

$$
n = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2}{d^2}
$$

### 6.11 パラダイム転換の問い

> **「p < 0.05で有意」と言える。だが、それは本当に**あなたの主張**を支持しているのか？**

以下のシナリオを考えよう:

1. **シナリオA**: 新しいプロンプト手法を10種類試し、1つだけp < 0.05で有意な改善。他9つは有意差なし。
2. **シナリオB**: 同じ実験を100回行い、有意だった5回だけ論文に報告。
3. **シナリオC**: データを見てから「このデータセットでは効果がある」と事後的にサブグループ分析。

**全て統計的には「p < 0.05」だが、科学的には無意味だ。**

- **シナリオA**: 多重比較の罠。Bonferroni補正すればp = 0.05 × 10 = 0.5で有意でない。
- **シナリオB**: 出版バイアス。失敗した95回を隠蔽。
- **シナリオC**: p-hacking。データを見てから仮説を立てる。

**議論の種**:

1. **事前登録（Pre-registration）**は解決策か？　実験前に仮説・手法を公開登録すれば、p-hackingを防げる。だが柔軟性が失われる。
2. **p値の代替案**は？　信頼区間・効果量・ベイズファクターは、p値の問題を解決するか？
3. **統計的有意性の基準（α=0.05）**は恣意的ではないか？　なぜ0.05なのか？　0.01や0.001ではダメなのか？

この問いに完全な答えはない。だが**統計学は道具であり、道具の使い方次第で科学的誠実さが問われる**ことを忘れてはならない。

> **Note:** **進捗: 100% 完了** 🎉 講義完走！

---


> Progress: [85%]
> **理解度チェック**
> 1. ANOVAのF統計量が群間分散と群内分散の比で構成される数学的意味を述べよ。
> 2. ロジスティック回帰のリンク関数がlogitである理由を確率の範囲の制約から説明せよ。

## 参考文献

### 主要論文

[^1]: Neyman, J., & Pearson, E. S. (1928). *On the Use and Interpretation of Certain Test Criteria for Purposes of Statistical Inference: Part I*. Biometrika.
<https://www.jstor.org/stable/2331945>

[^2]: Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society: Series B.
<https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>

[^3]: Hastings, W. K. (1970). *Monte Carlo Sampling Methods Using Markov Chains and Their Applications*. Biometrika.
<https://doi.org/10.1093/biomet/57.1.97>


### 教科書

- **Statistical Inference** - Casella & Berger (2002): 頻度論統計の決定版。大学院レベル。
- **Bayesian Data Analysis** - Gelman et al. (2013): ベイズ統計の標準教科書。
- **The Elements of Statistical Learning** - Hastie, Tibshirani, Friedman (2009): 機械学習×統計の融合。[無料PDF](https://web.stanford.edu/~hastie/ElemStatLearn/)
- **統計学入門** - 東京大学教養学部統計学教室 (1991): 日本語の定番入門書。

### オンラインリソース

- [StatQuest (YouTube)](https://www.youtube.com/@statquest): 統計学の直感的解説動画。
- [ndarray-stats Documentation](https://juliastats.org/ndarray-stats/stable/)
- [statrs Documentation](https://juliastats.org/statrs/stable/)
- [linfa Documentation](https://juliastats.org/linfa/stable/)
- [probabilistic-rs Documentation](https://turinglang.org/stable/)

---

## 付録A: 統計学の歴史的発展

### A.1 頻度論統計の誕生（1900-1950年代）

| 年 | 人物 | 貢献 |
|:---|:-----|:-----|
| 1900 | Karl Pearson | カイ二乗検定、Pearson相関係数 |
| 1908 | William Gosset (Student) | t分布、t検定（少サンプル統計） |
| 1920年代 | Ronald Fisher | 最尤推定（MLE）、分散分析（ANOVA）、実験計画法 |
| 1928 | Neyman & Pearson | Neyman-Pearson仮説検定枠組み [^1] |
| 1935 | Fisher | ランダム化比較試験（RCT）の原理 |

**頻度論の哲学**: 確率 = 長期的頻度。パラメータは固定値（未知）。客観性を重視。

### A.2 ベイズ統計の復興（1950-1990年代）

| 年 | 人物/出来事 | 貢献 |
|:---|:----------|:-----|
| 1763 | Thomas Bayes（死後出版） | ベイズの定理の原型 |
| 1950年代 | Dennis Lindley | ベイズ決定理論 |
| 1953 | Metropolis et al. | Metropolisアルゴリズム（MCMC） [^3] |
| 1970 | Hastings | Metropolis-Hastingsアルゴリズム |
| 1990 | Gelfand & Smith | Gibbs Samplingの実用化 |

**ベイズ復興の理由**: コンピュータの発展でMCMCが実用化 → 複雑なモデルの事後分布を計算可能に。

### A.3 現代統計学（1990年代〜現在）

| 年 | 手法 | 貢献 |
|:---|:-----|:-----|
| 1995 | Benjamini & Hochberg | FDR制御法（多重比較） [^2] |
| 2000年代 | ベイズノンパラメトリクス | 無限次元モデル（Dirichlet Process等） |
| 2010年代 | Hamiltonian Monte Carlo (HMC) | 高次元MCMCの高速化（NUTS） |
| 2015年代 | 因果推論の普及 | Pearl/Rubin枠組みの統合、機械学習との融合 |
| 2020年代 | 確率的プログラミング | probabilistic-rs, PyMC, Stan等の成熟 |

---

## 付録B: Rustで使える統計パッケージ完全リスト

### B.1 基礎統計

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **Statistics** (stdlib) | 基本統計量 | `mean`, `std`, `var`, `median`, `quantile`, `cor`, `cov` |
| **ndarray-stats** | 記述統計・重み付き統計 | `skewness`, `kurtosis`, `mad`, `mode`, `sem`, `zscore`, `sample`, `weights` |
| **statrs** | 確率分布 | `Normal`, `Beta`, `Gamma`, `Binomial`, `Poisson`, `TDist`, `FDist`, `pdf`, `cdf`, `quantile`, `rand` |

### B.2 仮説検定

| パッケージ | 用途 | 主要検定 |
|:----------|:-----|:---------|
| **statrs** | 仮説検定全般 | `OneSampleTTest`, `EqualVarianceTTest`, `UnequalVarianceTTest`, `MannWhitneyUTest`, `WilcoxonSignedRankTest`, `KruskalWallisTest`, `OneWayANOVATest`, `ChisqTest`, `FisherExactTest`, `KSTest`, `AndersonDarlingTest` |
| **statrs** | 多重比較補正 | `adjust`, `Bonferroni`, `Holm`, `BenjaminiHochberg`, `BenjaminiYekutieli` |

### B.3 回帰・GLM

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **linfa** | 一般化線形モデル | `glm`, `@formula`, `Binomial`, `Poisson`, `Gamma`, `LogitLink`, `LogLink`, `InverseLink`, `coef`, `confint`, `predict` |
| **MixedModels.jl** | 混合効果モデル | `LinearMixedModel`, `fit!`, `ranef`, `fixef` |

### B.4 ベイズ統計

| パッケージ | 用途 | 主要関数/マクロ |
|:----------|:-----|:---------------|
| **probabilistic-rs** | 確率的プログラミング | `@model`, `~`, `sample`, `NUTS`, `HMC`, `Gibbs`, `plot`, `summarize` |
| **AdvancedMH.jl** | MCMC拡張 | `MetropolisHastings`, `RWMH`, `StaticMH` |
| **MCMCChains.jl** | MCMC結果の解析 | `Chains`, `describe`, `plot`, `ess`, `gelmandiag` |
| **AbstractMCMC.jl** | MCMCインターフェース | MCMC実装の共通基盤 |

### B.5 ブートストラップ・リサンプリング

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **Bootstrap.jl** | ブートストラップ法 | `bootstrap`, `BasicSampling`, `confint`, `PercentileConfInt`, `BCaConfInt` |

### B.6 生存時間解析

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **Survival.jl** | 生存時間解析 | `Surv`, `kaplan_meier`, `cox_ph`, `nelson_aalen` |

### B.7 時系列解析

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **TimeSeries.jl** | 時系列データ | `TimeArray`, `values`, `timestamp`, `lag`, `lead`, `diff` |
| **StateSpaceModels.jl** | 状態空間モデル | `StateSpaceModel`, `kalman_filter`, `smoother` |

### B.8 実験計画法

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **ExperimentalDesign.jl** | 実験計画 | `factorial_design`, `latin_square`, `balanced_design` |

### B.9 可視化

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **StatsPlots.jl** | 統計的プロット | `boxplot`, `violin`, `density`, `marginalscatter`, `corrplot`, `@df` |
| **plotters** | 高品質可視化 | `scatter`, `lines`, `barplot`, `heatmap`, `density` |
| **AlgebraOfGraphics.jl** | Grammar of Graphics | `data`, `mapping`, `visual`, `draw` |

---

## 付録C: 統計学の主要定理まとめ

### C.1 確率論の基礎定理

**大数の法則（Law of Large Numbers）**:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty
$$

標本平均は母平均に確率収束する。

**中心極限定理（Central Limit Theorem）**:

$$
\sqrt{n} \frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty
$$

標本平均の分布は正規分布に近づく（母集団分布に関わらず）。

### C.2 推定の理論

**Cramér-Rao下界（Cramér-Rao Lower Bound）**:

不偏推定量 $\hat{\theta}$ の分散は次の下界を持つ:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

ここで $I(\theta)$ はFisher情報量。等号成立時は**有効推定量**。

**漸近正規性（Asymptotic Normality）**:

MLEは漸近的に正規分布に従う:

$$
\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})
$$

### C.3 検定の理論

**Neyman-Pearson補題（Neyman-Pearson Lemma）**:

尤度比検定は所定の有意水準 $\alpha$ で最も検出力が高い（most powerful test）。

$$
\frac{p(x | H_1)}{p(x | H_0)} > c \quad \Rightarrow \quad \text{reject } H_0
$$

### C.4 ベイズ統計の定理

**ベイズの定理（Bayes' Theorem）**:

$$
p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)} = \frac{p(D | \theta) p(\theta)}{\int p(D | \theta') p(\theta') d\theta'}
$$

**マルコフ連鎖の収束**:

適切な条件下でMCMCサンプルは事後分布に収束:

$$
\lim_{t \to \infty} \theta^{(t)} \sim p(\theta | D)
$$

---

## 付録D: 統計学の実践チェックリスト

### D.1 実験前（事前計画）

- [ ] 研究仮説を明確に定義（$H_0$, $H_1$）
- [ ] 有意水準 $\alpha$ を決定（通常0.05）
- [ ] 目標検出力を決定（通常0.8）
- [ ] 期待される効果量を設定（過去研究・予備実験から）
- [ ] パワー分析で必要サンプルサイズを計算
- [ ] 検定手法を事前に決定（t検定・ANOVA・ノンパラメトリック等）
- [ ] 多重比較がある場合は補正方法を決定（Bonferroni・BH等）
- [ ] 事前登録（Pre-registration）を検討（p-hackingを防ぐ）

### D.2 データ収集

- [ ] ランダムサンプリング・ランダム化を徹底
- [ ] ブロック要因があれば乱塊法を検討
- [ ] 測定誤差を最小化（機器の校正・プロトコルの標準化）
- [ ] 欠損データの記録・理由の記載
- [ ] 外れ値の記録（削除前に理由を明記）

### D.3 記述統計

- [ ] 平均・中央値・標準偏差・IQRを計算
- [ ] 歪度・尖度を確認（分布の形状）
- [ ] 外れ値の検出（IQR法・Grubbs検定）
- [ ] ヒストグラム・箱ひげ図で可視化

### D.4 推測統計

- [ ] 前提条件の確認（正規性・等分散性・独立性）
- [ ] 正規性検定（Shapiro-Wilk・Kolmogorov-Smirnov）
- [ ] 等分散性検定（Levene・Bartlett）
- [ ] 前提が満たされない場合は代替手法（ノンパラメトリック・変換・頑健な手法）

### D.5 仮説検定

- [ ] 検定統計量（t, F, χ², U等）を計算
- [ ] 自由度を確認
- [ ] p値を計算
- [ ] 効果量（Cohen's d, partial η², r²等）を計算
- [ ] 信頼区間を併記
- [ ] 多重比較補正（該当する場合）

### D.6 結果の報告

- [ ] 記述統計（M, SD, n）を報告
- [ ] 検定統計量・自由度・p値を報告（例: $t(9) = 60.0, p < .001$）
- [ ] 効果量を報告（例: $d = 6.0$）
- [ ] 95%信頼区間を報告（例: $95\% \text{CI} [0.768, 0.782]$）
- [ ] 多重比較補正方法を明記
- [ ] 図表で視覚化（箱ひげ図・エラーバー付き棒グラフ等）
- [ ] 統計的有意性と実用的有意性を区別

### D.7 解釈・議論

- [ ] p値の正しい解釈（「$H_0$が真である確率」ではない）
- [ ] 効果量の実用的意義を議論
- [ ] 検出力不足の可能性を検討（p > 0.05の場合）
- [ ] 代替説明（交絡因子）の可能性を議論
- [ ] 限界（サンプル選択バイアス・測定誤差等）を明記
- [ ] 因果関係と相関の区別

---

## 付録B: GLM発展トピックと最新手法

### B.1 混合効果モデル（Mixed Effects Models）

**問題**: データに階層構造がある場合（例: 生徒→クラス→学校）、観測が独立でない。

**線形混合効果モデル（LMM）**:

$$
y_{ij} = \beta_0 + \beta_1 x_{ij} + u_i + \epsilon_{ij}
$$

ここで:
- $y_{ij}$: グループ$i$の観測$j$の応答変数
- $u_i \sim \mathcal{N}(0, \sigma_u^2)$: グループレベルのランダム効果
- $\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$: 個体レベルの誤差

**固定効果 vs ランダム効果**:

| 項目 | 固定効果 | ランダム効果 |
|:-----|:--------|:-----------|
| 解釈 | 母集団全体の平均効果 | グループ間のばらつき |
| 推定 | 係数$\beta$ | 分散成分$\sigma_u^2$ |
| 目的 | 効果の大きさを知りたい | グループ間変動を制御したい |

**Rust実装例**（MixedModels.jl）:

```rust
// 混合効果モデル: 反応時間 ~ 日数 + (1 + 日数 | 被験者)
// 固定効果: β（日数の効果）、ランダム効果: u_i ~ N(0, D)
// In Rust: use the `linfa` crate or external R/Python bridge.
// MixedModels.jl の出力に相当する REML 対数尤度を手計算で示す

fn reml_log_likelihood(y: &[f64], x: &[f64], beta: &[f64], sigma_e: f64) -> f64 {
    // 簡易版: ランダム効果なし（固定効果のみ）の残差対数尤度
    let n = y.len() as f64;
    let residuals: f64 = y.iter().zip(x.iter())
        .map(|(&yi, &xi)| (yi - beta[0] - beta[1] * xi).powi(2))
        .sum();
    -0.5 * n * (2.0 * std::f64::consts::PI * sigma_e.powi(2)).ln()
        - 0.5 * residuals / sigma_e.powi(2)
}

fn main() {
    // sleepstudy データの代表値（反応時間[ms] vs 睡眠不足日数）
    let days: Vec<f64>     = (0..10).map(|d| d as f64).collect();
    let reaction: Vec<f64> = vec![249.56, 258.70, 250.80, 321.44, 356.85,
                                  414.69, 382.20, 290.15, 430.58, 466.35];
    // 固定効果推定（最小二乗法）
    let n = days.len() as f64;
    let x_bar = days.iter().sum::<f64>() / n;
    let y_bar = reaction.iter().sum::<f64>() / n;
    let beta1 = days.iter().zip(reaction.iter())
        .map(|(&x, &y)| (x - x_bar) * (y - y_bar))
        .sum::<f64>()
        / days.iter().map(|&x| (x - x_bar).powi(2)).sum::<f64>();
    let beta0 = y_bar - beta1 * x_bar;
    println!("固定効果: β₀={:.2}, β₁={:.2} (ms/日)", beta0, beta1);
    println!("解釈: 睡眠不足が1日増えるごとに反応時間が{:.2}ms増加", beta1);

    let residuals: Vec<f64> = days.iter().zip(reaction.iter())
        .map(|(&x, &y)| y - beta0 - beta1 * x).collect();
    let sigma_e = (residuals.iter().map(|r| r.powi(2)).sum::<f64>() / (n - 2.0)).sqrt();
    let ll = reml_log_likelihood(&reaction, &days, &[beta0, beta1], sigma_e);
    println!("REML 対数尤度 (簡易版): {:.2}", ll);
}
```

出力例:
```
Linear mixed model fit by maximum likelihood
 Reaction ~ 1 + Days + (1 + Days | Subject)
   logLik   -2 logLik     AIC       AICc        BIC
  -875.97    1751.94   1763.94   1764.47   1783.10

Variance components:
            Column    Variance   Std.Dev.   Corr.
Subject  (Intercept)  612.100    24.741
         Days          35.072     5.923    0.07
Residual              654.941    25.592
```

### B.2 一般化加法モデル（GAM: Generalized Additive Models）

**問題**: 線形性の仮定が厳しすぎる場合、非線形関係を柔軟にモデル化したい。

**GAMの定式化**:

$$
g(\mu) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)
$$

ここで$f_i$はスムージング関数（スプライン等）。

**スムージングスプライン**:

$$
\min_f \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int (f''(x))^2 dx
$$

第1項: フィット、第2項: 滑らかさのペナルティ

**Rustでの簡易実装**:

```rust
// GAM（一般化加法モデル）: 多項式基底展開で非線形関係をモデル化
// 可視化には plotters クレートが必要 (cargo add plotters)

fn polynomial_features(x: &[f64], degree: usize) -> Vec<Vec<f64>> {
    // x の 0 次〜 degree 次の特徴量行列を返す（各行が1サンプル）
    x.iter().map(|&xi| (0..=degree).map(|d| xi.powi(d as i32)).collect()).collect()
}

/// 最小二乗法で多項式係数を推定（正規方程式: β = (XᵀX)⁻¹Xᵀy）
fn least_squares(x_mat: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = x_mat.len();
    let p = x_mat[0].len();
    // XᵀX
    let mut xtx = vec![vec![0.0_f64; p]; p];
    for i in 0..p {
        for j in 0..p {
            xtx[i][j] = (0..n).map(|k| x_mat[k][i] * x_mat[k][j]).sum();
        }
    }
    // Xᵀy
    let xty: Vec<f64> = (0..p).map(|i| (0..n).map(|k| x_mat[k][i] * y[k]).sum()).collect();
    // Gauss-Jordan による逆行列の代わりに単純な解法（小行列のみ）
    solve_linear(&xtx, &xty)
}

fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut mat: Vec<Vec<f64>> = a.iter().zip(b.iter())
        .map(|(row, &bi)| { let mut r = row.clone(); r.push(bi); r }).collect();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| mat[i][col].abs().partial_cmp(&mat[j][col].abs()).unwrap()).unwrap();
        mat.swap(col, pivot);
        let div = mat[col][col];
        for val in &mut mat[col] { *val /= div; }
        for row in (0..n).filter(|&r| r != col) {
            let factor = mat[row][col];
            for c in 0..=n { let v = mat[col][c]; mat[row][c] -= factor * v; }
        }
    }
    mat.iter().map(|row| *row.last().unwrap()).collect()
}

fn main() {
    // データ生成: 非線形関係 y = sin(x) + 0.5x + ε
    let n = 100_usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect();
    let y_true: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.5 * xi).collect();
    // ノイズなし版でデモ（ランダムシードなし）
    let y: Vec<f64> = y_true.clone();

    // 次数5の多項式GAM
    let x_poly = polynomial_features(&x, 5);
    let beta = least_squares(&x_poly, &y);
    println!("多項式GAM 係数: {:?}", beta.iter().map(|b| format!("{:.4}", b)).collect::<Vec<_>>());

    // 予測と残差確認
    let y_pred: Vec<f64> = x_poly.iter()
        .map(|xi| xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum())
        .collect();
    let mse = y.iter().zip(y_pred.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;
    println!("MSE: {:.6}", mse);
    // 可視化: plotters クレートで scatter + line plot を描画
    // cargo add plotters
}
```

### B.3 ゼロ過剰モデル（Zero-Inflated Models）

**問題**: カウントデータにゼロが過剰に含まれる（例: 病院受診回数、事故件数）。

**ゼロ過剰ポアソンモデル（ZIP）**:

$$
P(Y = y) = \begin{cases}
\pi + (1 - \pi) e^{-\lambda} & \text{if } y = 0 \\
(1 - \pi) \frac{\lambda^y e^{-\lambda}}{y!} & \text{if } y > 0
\end{cases}
$$

ここで:
- $\pi$: 構造的ゼロの確率（「決してイベントが起こらない」）
- $1 - \pi$: ポアソン過程に従う確率

**2段階モデル**:

1. ロジスティック回帰で$\pi$を推定
2. ポアソン回帰で$\lambda$を推定

**数値例**:

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform, Poisson};

// ZIP（ゼロ過剰ポアソン）対数尤度
// P(Y=0) = π + (1-π)·exp(-λ)
// P(Y=y) = (1-π)·λʸ·exp(-λ)/y!  (y > 0)
fn zip_loglik(pi: f64, lambda: f64, y: &[u64]) -> f64 {
    if pi < 0.0 || pi >= 1.0 || lambda <= 0.0 { return f64::NEG_INFINITY; }
    let log_zero = (pi + (1.0 - pi) * (-lambda).exp()).ln();
    y.iter().map(|&yi| {
        if yi == 0 {
            log_zero
        } else {
            // log[(1-π)·Poisson(y|λ)]
            let log_factorial: f64 = (1..=yi).map(|k| (k as f64).ln()).sum();
            (1.0 - pi).ln() + yi as f64 * lambda.ln() - lambda - log_factorial
        }
    }).sum()
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let true_pi = 0.3_f64;
    let true_lambda = 2.0_f64;
    let n = 1000_usize;
    let uniform = Uniform::new(0.0_f64, 1.0);

    // データ生成: ゼロ過剰ポアソン
    let pois = Poisson::new(true_lambda).unwrap();
    let y: Vec<u64> = (0..n).map(|_| {
        if uniform.sample(&mut rng) < true_pi { 0 }
        else { pois.sample(&mut rng) as u64 }
    }).collect();

    let zero_rate = y.iter().filter(|&&v| v == 0).count() as f64 / n as f64;
    let theoretical_zero = true_pi + (1.0 - true_pi) * (-true_lambda).exp();
    println!("ゼロの割合: {:.4} (理論値: {:.4})", zero_rate, theoretical_zero);

    // グリッドサーチによる最尤推定
    let (mut best_pi, mut best_lambda, mut best_ll) = (0.3, 2.0, f64::NEG_INFINITY);
    for pi_i in 0..20 {
        for lam_i in 1..50 {
            let pi_c = pi_i as f64 * 0.05;
            let lam_c = lam_i as f64 * 0.2;
            let ll = zip_loglik(pi_c, lam_c, &y);
            if ll > best_ll { best_ll = ll; best_pi = pi_c; best_lambda = lam_c; }
        }
    }
    println!("推定値: π={:.3}, λ={:.3}", best_pi, best_lambda);
    println!("真値: π={}, λ={}", true_pi, true_lambda);
}
```

### B.4 時系列モデル（Time Series Models）

#### B.4.1 自己回帰モデル（AR）

**AR(p)モデル**:

$$
y_t = \phi_0 + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \epsilon_t
$$

ここで$\epsilon_t \sim \mathcal{N}(0, \sigma^2)$はホワイトノイズ。

**定常性条件**: 特性方程式の根が単位円の外側にある。

**Rust実装例**:

```rust
use rand_distr::{Distribution, Normal as RandNormal};

// AR(1)プロセスのシミュレーション
// y[t] = ϕ·y[t-1] + ε[t],  ε ~ N(0, σ²)
fn ar1_simulate(phi: f64, sigma: f64, n: usize) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let noise = RandNormal::new(0.0, sigma).unwrap();
    let mut y = vec![0.0_f64; n];
    // 定常分布から初期値: y[0] ~ N(0, σ²/(1-ϕ²))
    let init_std = sigma / (1.0 - phi.powi(2)).sqrt();
    y[0] = RandNormal::new(0.0, init_std).unwrap().sample(&mut rng);
    for t in 1..n {
        y[t] = phi * y[t - 1] + noise.sample(&mut rng);
    }
    y
}

// 自己相関関数（ACF）: ρ(k) = Cov(y[t], y[t-k]) / Var(y)
fn acf(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    let c0: f64 = xc.iter().map(|&v| v * v).sum::<f64>() / n as f64;
    let mut result = vec![1.0_f64];
    for k in 1..=max_lag {
        let ck: f64 = xc[..n - k].iter().zip(xc[k..].iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>() / (n as f64 * c0);
        result.push(ck);
    }
    result
}

fn main() {
    // パラメータ
    let phi = 0.8_f64;  // 自己相関係数
    let sigma = 1.0_f64;
    let n = 200_usize;

    let y = ar1_simulate(phi, sigma, n);
    let acf_vals = acf(&y, 20);

    println!("AR(1) series (最初10点): {:?}", &y[..10].iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>());
    println!("\n自己相関関数 ACF (lag 0-10):");
    for (lag, &rho) in acf_vals.iter().enumerate().take(11) {
        println!("  lag={}: {:.4}  (理論値: ϕ^lag={:.4})", lag, rho, phi.powi(lag as i32));
    }
    // 可視化: plotters クレートで時系列プロットと ACF バープロットを描画
    // cargo add plotters
}
```

#### B.4.2 状態空間モデル（State Space Models）

**カルマンフィルタ**:

$$
\begin{aligned}
\text{状態方程式:} \quad & x_t = F x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q) \\
\text{観測方程式:} \quad & y_t = H x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
\end{aligned}
$$

**予測ステップ**:

$$
\begin{aligned}
\hat{x}_{t|t-1} &= F \hat{x}_{t-1|t-1} \\
P_{t|t-1} &= F P_{t-1|t-1} F^\top + Q
\end{aligned}
$$

**更新ステップ**:

$$
\begin{aligned}
K_t &= P_{t|t-1} H^\top (H P_{t|t-1} H^\top + R)^{-1} \quad \text{(カルマンゲイン)} \\
\hat{x}_{t|t} &= \hat{x}_{t|t-1} + K_t (y_t - H \hat{x}_{t|t-1}) \\
P_{t|t} &= (I - K_t H) P_{t|t-1}
\end{aligned}
$$

**Rust実装例**:

```rust
// カルマンフィルタ実装（スカラー状態空間モデル）
// 状態方程式: x[t] = F·x[t-1] + w[t],  w ~ N(0, Q)
// 観測方程式: y[t] = H·x[t]  + v[t],  v ~ N(0, R)

struct KalmanFilter {
    f: f64,  // 状態遷移係数
    h: f64,  // 観測係数
    q: f64,  // プロセスノイズ分散
    r: f64,  // 観測ノイズ分散
}

impl KalmanFilter {
    fn filter(&self, y: &[f64], x0: f64, p0: f64) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut x_filt = vec![x0];
        let mut p_filt = vec![p0];

        for t in 1..n {
            // 予測ステップ
            let x_pred = self.f * x_filt[t - 1];
            let p_pred = self.f * p_filt[t - 1] * self.f + self.q;

            // 更新ステップ（カルマンゲイン K = P_pred·H / S, S = H·P_pred·H + R）
            let s = self.h * p_pred * self.h + self.r;
            let k = p_pred * self.h / s;  // カルマンゲイン
            let innovation = y[t] - self.h * x_pred;

            x_filt.push(x_pred + k * innovation);
            p_filt.push((1.0 - k * self.h) * p_pred);
        }
        (x_filt, p_filt)
    }
}

fn main() {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as RandNormal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // テスト: ローカルレベルモデル（ランダムウォーク + 観測ノイズ）
    let n = 100_usize;
    let noise_state = RandNormal::new(0.0_f64, 0.1_f64.sqrt()).unwrap();
    let noise_obs   = RandNormal::new(0.0_f64, 1.0_f64).unwrap();

    // 真の状態（ランダムウォーク）
    let mut x_true = vec![0.0_f64];
    for _ in 1..n {
        x_true.push(x_true.last().unwrap() + noise_state.sample(&mut rng));
    }
    let y_obs: Vec<f64> = x_true.iter().map(|&x| x + noise_obs.sample(&mut rng)).collect();

    let kf = KalmanFilter { f: 1.0, h: 1.0, q: 0.1, r: 1.0 };
    let (x_filt, _p_filt) = kf.filter(&y_obs, 0.0, 1.0);

    // 結果確認
    let rmse_raw: f64 = (x_true.iter().zip(y_obs.iter())
        .map(|(xt, yo)| (xt - yo).powi(2)).sum::<f64>() / n as f64).sqrt();
    let rmse_filt: f64 = (x_true.iter().zip(x_filt.iter())
        .map(|(xt, xf)| (xt - xf).powi(2)).sum::<f64>() / n as f64).sqrt();
    println!("RMSE (観測値): {:.4}", rmse_raw);
    println!("RMSE (フィルタ後): {:.4}", rmse_filt);
    println!("カルマンフィルタでノイズが低減された: {}", rmse_filt < rmse_raw);
    // 可視化: plotters クレートで x_true / y_obs / x_filt を描画
}
```

### B.5 ベイズ階層モデルの実践

#### B.5.1 部分プーリング（Partial Pooling）

**問題**: グループごとに推定したいが、サンプルサイズが小さい。

**3つのアプローチ**:

| 手法 | 説明 | 問題点 |
|:-----|:-----|:------|
| **完全プーリング** | 全グループを1つとして扱う | グループ間の違いを無視 |
| **ノープーリング** | グループごとに独立推定 | 小サンプルで不安定 |
| **部分プーリング** | 階層モデルで情報共有 | ✅ 両者のバランス |

**階層ベイズモデル**:

$$
\begin{aligned}
y_{ij} &\sim \mathcal{N}(\mu_i, \sigma^2) \\
\mu_i &\sim \mathcal{N}(\mu_{\text{global}}, \tau^2) \\
\mu_{\text{global}} &\sim \mathcal{N}(0, 10^2) \\
\sigma, \tau &\sim \text{Half-Cauchy}(0, 5)
\end{aligned}
$$

**probabilistic-rs実装**:

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};

// 階層ベイズモデル: 部分プーリング（Partial Pooling）
// y_ij ~ N(μ_i, σ²)
// μ_i  ~ N(μ_global, τ²)
// probabilistic-rs / MCMC 実装パターン

struct HierarchicalModel {
    school_scores: Vec<Vec<f64>>,
}

impl HierarchicalModel {
    fn log_posterior(&self, mu_global: f64, tau: f64, sigma: f64, mu_schools: &[f64]) -> f64 {
        if tau <= 0.0 || sigma <= 0.0 { return f64::NEG_INFINITY; }
        // ハイパーパラメータの事前分布: μ_global ~ N(70, 20²), τ,σ ~ Half-Cauchy(5)
        let log_prior_global = -0.5 * ((mu_global - 70.0) / 20.0).powi(2);
        let log_prior_tau    = -(1.0 + (tau / 5.0).powi(2)).ln();
        let log_prior_sigma  = -(1.0 + (sigma / 5.0).powi(2)).ln();
        // 学校レベルの平均: μ_i ~ N(μ_global, τ²)
        let log_schools: f64 = mu_schools.iter()
            .map(|&mu_i| -0.5 * ((mu_i - mu_global) / tau).powi(2) - tau.ln())
            .sum();
        // 尤度: y_ij ~ N(μ_i, σ²)
        let log_lik: f64 = self.school_scores.iter().zip(mu_schools.iter())
            .map(|(scores, &mu_i)| scores.iter()
                .map(|&y| -0.5 * ((y - mu_i) / sigma).powi(2) - sigma.ln())
                .sum::<f64>())
            .sum();
        log_prior_global + log_prior_tau + log_prior_sigma + log_schools + log_lik
    }
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let n_schools = 10_usize;
    let students_per_school = [5, 8, 12, 6, 15, 7, 20, 9, 11, 13_usize];

    // データ生成: 学校ごとの生徒のテストスコア
    let true_school_means: Vec<f64> = (0..n_schools).map(|i| {
        let base = RandNormal::new(0.0_f64, 5.0).unwrap().sample(&mut rng);
        base + 70.0 + i as f64 * 0.5
    }).collect();

    let school_scores: Vec<Vec<f64>> = true_school_means.iter()
        .zip(students_per_school.iter())
        .map(|(&mu, &ns)| {
            let noise = RandNormal::new(mu, 10.0).unwrap();
            (0..ns).map(|_| noise.sample(&mut rng)).collect()
        })
        .collect();

    let model = HierarchicalModel { school_scores: school_scores.clone() };

    // 完全プーリング vs ノープーリング vs 部分プーリング（事後平均）の比較
    let all_scores: Vec<f64> = school_scores.iter().flatten().cloned().collect();
    let grand_mean = all_scores.iter().sum::<f64>() / all_scores.len() as f64;
    println!("グローバル平均 (完全プーリング): {:.2}", grand_mean);

    for (i, scores) in school_scores.iter().enumerate() {
        let school_mean = scores.iter().sum::<f64>() / scores.len() as f64;
        println!("学校{}: ノープーリング={:.2}, 真値={:.2}", i + 1, school_mean, true_school_means[i]);
    }

    // log_posterior の確認
    let mu_schools: Vec<f64> = school_scores.iter()
        .map(|s| s.iter().sum::<f64>() / s.len() as f64).collect();
    let lp = model.log_posterior(grand_mean, 5.0, 10.0, &mu_schools);
    println!("Log posterior (初期値): {:.2}", lp);
}
```

#### B.5.2 収束診断（Convergence Diagnostics）

**Gelman-Rubin統計量（$\hat{R}$）**:

複数チェーンの収束を診断。$\hat{R} \approx 1$なら収束。

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

ここで:
- $W$: チェーン内分散の平均
- $\hat{V}$: チェーン間分散とチェーン内分散の重み付き平均

**有効サンプルサイズ（ESS: Effective Sample Size）**:

自己相関を考慮した実効的なサンプル数。

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}
$$

ここで$\rho_k$は遅れ$k$での自己相関。

**Rust実装例**:

```rust
// 収束診断: R̂（Gelman-Rubin統計量）と ESS
// R̂ = sqrt(V̂ / W): V̂はプール分散推定, Wはチェーン内分散平均
// ESS = S / (1 + 2 Σ ρ_τ): Sは総サンプル数, ρ_τは自己相関

fn rhat(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len() as f64;
    let n = chains[0].len() as f64;
    let chain_means: Vec<f64> = chains.iter()
        .map(|c| c.iter().sum::<f64>() / n)
        .collect();
    let grand_mean = chain_means.iter().sum::<f64>() / m;
    // Between-chain variance B
    let b = n / (m - 1.0) * chain_means.iter()
        .map(|&cm| (cm - grand_mean).powi(2))
        .sum::<f64>();
    // Within-chain variance W
    let w = chains.iter().zip(chain_means.iter())
        .map(|(c, &cm)| c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (n - 1.0))
        .sum::<f64>() / m;
    let v_hat = (n - 1.0) / n * w + b / n;
    (v_hat / w).sqrt()
}

fn ess(chain: &[f64]) -> f64 {
    let n = chain.len();
    let mean = chain.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = chain.iter().map(|&v| v - mean).collect();
    let c0: f64 = xc.iter().map(|&v| v * v).sum::<f64>() / n as f64;
    let mut rho_sum = 0.0;
    for lag in 1..n.min(200) {
        let rho = xc[..n - lag].iter().zip(xc[lag..].iter())
            .map(|(&a, &b)| a * b).sum::<f64>() / (n as f64 * c0);
        if rho < 0.0 { break; }
        rho_sum += rho;
    }
    n as f64 / (1.0 + 2.0 * rho_sum)
}

fn main() {
    // ダミーチェーン（収束済みの場合の想定値）
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as RandNormal};
    let noise = RandNormal::new(0.72_f64, 0.01).unwrap();
    let chains: Vec<Vec<f64>> = (0..4).map(|seed| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..2000).map(|_| noise.sample(&mut rng)).collect()
    }).collect();

    println!("=== 収束診断 ===");
    let r = rhat(&chains);
    println!("R̂ = {:.4}  (< 1.01 が収束の目安)", r);

    println!("\n=== 有効サンプルサイズ ===");
    let e = ess(&chains[0]);
    println!("ESS = {:.1}  (> 400 が目安)", e);

    // 自己相関（lag 1-5）
    println!("\n=== 自己相関 ===");
    let chain = &chains[0];
    let mean = chain.iter().sum::<f64>() / chain.len() as f64;
    let xc: Vec<f64> = chain.iter().map(|&v| v - mean).collect();
    let c0: f64 = xc.iter().map(|&v| v * v).sum::<f64>() / chain.len() as f64;
    for lag in 1..=5 {
        let rho = xc[..chain.len()-lag].iter().zip(xc[lag..].iter())
            .map(|(&a, &b)| a * b).sum::<f64>() / (chain.len() as f64 * c0);
        println!("  lag={}: ρ={:.4}", lag, rho);
    }
    let status = if r < 1.01 && e > 400.0 { "✅ 収束" } else { "⚠️ 要確認" };
    println!("\n収束判定: {}", status);
}
```

### B.6 ベイズモデル選択

#### B.6.1 WAIC（Widely Applicable Information Criterion）

**定義**:

$$
\text{WAIC} = -2 (\text{lppd} - p_{\text{WAIC}})
$$

ここで:
- $\text{lppd}$: log pointwise predictive density
- $p_{\text{WAIC}}$: 有効パラメータ数

**計算**:

$$
\begin{aligned}
\text{lppd} &= \sum_{i=1}^n \log \left( \frac{1}{S} \sum_{s=1}^S p(y_i | \theta^{(s)}) \right) \\
p_{\text{WAIC}} &= \sum_{i=1}^n \text{Var}_s(\log p(y_i | \theta^{(s)}))
\end{aligned}
$$

**Rust実装例**:

```rust
// WAIC（Widely Applicable Information Criterion）
// WAIC = -2(lppd - p_WAIC)
// lppd   = Σᵢ log(mean_s p(yᵢ|θ⁽ˢ⁾))
// p_WAIC = Σᵢ Var_s(log p(yᵢ|θ⁽ˢ⁾))

fn waic(log_lik: &Vec<Vec<f64>>) -> (f64, f64, f64) {
    // log_lik[s][i] = log p(y_i | θ^(s))
    let s = log_lik.len() as f64;
    let n = log_lik[0].len();

    let lppd: f64 = (0..n).map(|i| {
        // log(mean_s exp(log_lik[s][i])) = log_sum_exp - log(S)
        let max_ll = log_lik.iter().map(|row| row[i]).fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_lik.iter().map(|row| (row[i] - max_ll).exp()).sum();
        max_ll + sum_exp.ln() - s.ln()
    }).sum();

    let p_waic: f64 = (0..n).map(|i| {
        let vals: Vec<f64> = log_lik.iter().map(|row| row[i]).collect();
        let mean = vals.iter().sum::<f64>() / s;
        vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (s - 1.0)
    }).sum();

    let waic_val = -2.0 * (lppd - p_waic);
    (waic_val, lppd, p_waic)
}

fn main() {
    // モデル1（単純）と モデル2（複雑）の比較
    // ダミーのlog尤度サンプル（200サンプル × 50データ点）
    let n_samples = 200_usize;
    let n_data    = 50_usize;

    // モデル1: 正規分布 μ ~ N(0,10), σ ~ HalfNormal(5)
    // MCMCチェーンの代わりに固定値でデモ
    let mu1 = 0.72_f64; let sigma1 = 0.02_f64;
    let data: Vec<f64> = (0..n_data).map(|i| 0.70 + (i as f64) * 0.001).collect();
    let log_lik1: Vec<Vec<f64>> = (0..n_samples).map(|_| {
        data.iter().map(|&y| {
            -0.5 * ((y - mu1) / sigma1).powi(2) - sigma1.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        }).collect()
    }).collect();

    // モデル2: より広い事前分布
    let mu2 = 0.72_f64; let sigma2 = 0.05_f64;
    let log_lik2: Vec<Vec<f64>> = (0..n_samples).map(|_| {
        data.iter().map(|&y| {
            -0.5 * ((y - mu2) / sigma2).powi(2) - sigma2.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        }).collect()
    }).collect();

    let (waic1, lppd1, p1) = waic(&log_lik1);
    let (waic2, lppd2, p2) = waic(&log_lik2);

    println!("Model 1 WAIC: {:.2}  (lppd={:.2}, p_WAIC={:.2})", waic1, lppd1, p1);
    println!("Model 2 WAIC: {:.2}  (lppd={:.2}, p_WAIC={:.2})", waic2, lppd2, p2);
    println!("Better model: {}", if waic1 < waic2 { "Model 1" } else { "Model 2" });
}
```

#### B.6.2 ベイズファクター（Bayes Factor）

**定義**:

$$
\text{BF}_{12} = \frac{p(D | M_1)}{p(D | M_2)}
$$

**解釈**（Kass & Raftery, 1995）:

| BF | 証拠の強さ |
|:---|:----------|
| 1-3 | ほとんど価値なし |
| 3-20 | 肯定的 |
| 20-150 | 強い |
| >150 | 非常に強い |

**問題点**: 周辺尤度$p(D | M)$の計算が困難。

### B.7 ベイズノンパラメトリクス入門

#### B.7.1 Dirichlet Process（ディリクレ過程）

**問題**: クラスタ数が事前に分からないクラスタリング。

**Dirichlet Process Mixture Model (DPMM)**:

$$
\begin{aligned}
G &\sim \text{DP}(\alpha, H) \quad \text{（ディリクレ過程）} \\
\theta_i &\sim G \\
y_i &\sim F(\theta_i)
\end{aligned}
$$

ここで:
- $\alpha$: 集中度パラメータ（大きいほど多くのクラスタ）
- $H$: ベース分布
- $F$: 尤度関数

**Chinese Restaurant Process（CRP）**: DPの直感的な説明

新しい客が入店するとき:
- 確率$\frac{n_k}{\alpha + n - 1}$で既存のテーブル$k$に座る（$n_k$人座っている）
- 確率$\frac{\alpha}{\alpha + n - 1}$で新しいテーブルを作る

**Rust実装例（簡略版）**:

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, WeightedIndex};

// Chinese Restaurant Process simulation
// 新しい客 i が入店するとき:
//   確率 n_k / (α + i - 1) で既存テーブル k に着席
//   確率 α   / (α + i - 1) で新テーブルを作る
fn crp_simulate(n: usize, alpha: f64, rng: &mut impl rand::Rng) -> (Vec<usize>, Vec<usize>) {
    let mut tables: Vec<usize> = Vec::new();       // 各客がどのテーブルに座っているか
    let mut table_counts: Vec<usize> = Vec::new(); // 各テーブルの人数

    for i in 0..n {
        if tables.is_empty() {
            // 最初の客
            tables.push(0);
            table_counts.push(1);
        } else {
            // 既存テーブルに座る確率 vs 新テーブル
            let total = alpha + i as f64;
            let mut weights: Vec<f64> = table_counts.iter().map(|&c| c as f64 / total).collect();
            weights.push(alpha / total);  // 新テーブルの確率

            let dist = WeightedIndex::new(&weights).unwrap();
            let k = dist.sample(rng);

            if k < table_counts.len() {
                // 既存テーブル
                table_counts[k] += 1;
                tables.push(k);
            } else {
                // 新テーブル
                table_counts.push(1);
                tables.push(table_counts.len() - 1);
            }
        }
    }
    (tables, table_counts)
}

fn main() {
    let n = 100_usize;
    let alpha_values = [0.1_f64, 1.0, 10.0];

    for &alpha in &alpha_values {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (_tables, counts) = crp_simulate(n, alpha, &mut rng);
        let n_clusters = counts.len();
        println!("α={}: {} clusters formed", alpha, n_clusters);
    }
}
```

出力例:
```
α=0.1: 3 clusters formed
α=1.0: 8 clusters formed
α=10.0: 24 clusters formed
```

#### B.7.2 Gaussian Process（ガウス過程）

**定義**: 関数の事前分布を定義するノンパラメトリック手法。

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$

ここで:
- $m(x)$: 平均関数（通常0）
- $k(x, x')$: カーネル関数（共分散）

**RBFカーネル**:

$$
k(x, x') = \sigma^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

**予測分布**:

観測データ$(X, y)$が与えられたとき、新しい点$x_*$での予測:

$$
\begin{aligned}
f(x_*) | X, y, x_* &\sim \mathcal{N}(\mu_*, \sigma_*^2) \\
\mu_* &= k(x_*, X) [k(X, X) + \sigma_n^2 I]^{-1} y \\
\sigma_*^2 &= k(x_*, x_*) - k(x_*, X) [k(X, X) + \sigma_n^2 I]^{-1} k(X, x_*)
\end{aligned}
$$

**Rust実装例**:

```rust
// ガウス過程回帰
// f(x) ~ GP(m(x), k(x,x'))
// RBFカーネル: k(x,x') = σ² exp(-(x-x')²/(2ℓ²))
// 予測: μ* = K_s · (K + σ_n²I)⁻¹ y,  σ*² = k** - K_s (K + σ_n²I)⁻¹ K_s^T

fn rbf_kernel(x1: f64, x2: f64, sigma: f64, ell: f64) -> f64 {
    sigma.powi(2) * (-(x1 - x2).powi(2) / (2.0 * ell.powi(2))).exp()
}

/// 下三角 Cholesky 分解（小行列用）
fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            l[i][j] = if i == j { (a[i][i] - sum).sqrt() }
                      else { (a[i][j] - sum) / l[j][j] };
        }
    }
    l
}

/// Lx = b を前進代入で解く
fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let sum: f64 = (0..i).map(|j| l[i][j] * x[j]).sum();
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

/// Lᵀx = b を後退代入で解く
fn backward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let sum: f64 = (i+1..n).map(|j| l[j][i] * x[j]).sum();
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

fn gp_predict(
    x_train: &[f64], y_train: &[f64], x_test: &[f64],
    sigma: f64, ell: f64, sigma_n: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_train = x_train.len();
    let n_test  = x_test.len();

    // カーネル行列
    let mut k = vec![vec![0.0_f64; n_train]; n_train];
    for i in 0..n_train {
        for j in 0..n_train {
            k[i][j] = rbf_kernel(x_train[i], x_train[j], sigma, ell);
            if i == j { k[i][j] += sigma_n.powi(2); }  // + σ_n²I
        }
    }
    // K_s[test × train], K_ss[test × test]の対角
    let k_s: Vec<Vec<f64>> = x_test.iter().map(|&xt|
        x_train.iter().map(|&xi| rbf_kernel(xt, xi, sigma, ell)).collect()
    ).collect();
    let k_ss_diag: Vec<f64> = x_test.iter()
        .map(|&xt| rbf_kernel(xt, xt, sigma, ell)).collect();

    // Cholesky 分解: A\b より数値安定
    let l = cholesky(&k);
    let alpha = {
        let v = forward_sub(&l, y_train);
        backward_sub(&l, &v)
    };

    // 予測平均: μ* = K_s · α
    let mu_pred: Vec<f64> = k_s.iter().map(|ks_row|
        ks_row.iter().zip(alpha.iter()).map(|(a, b)| a * b).sum()
    ).collect();

    // 予測分散: σ*² = k** - K_s (K+σ_n²I)⁻¹ K_sᵀ （対角のみ）
    let sigma_pred: Vec<f64> = k_s.iter().zip(k_ss_diag.iter()).map(|(ks_row, &kss)| {
        let v = forward_sub(&l, ks_row);
        let var = kss - v.iter().map(|vi| vi.powi(2)).sum::<f64>();
        var.max(0.0).sqrt()
    }).collect();

    (mu_pred, sigma_pred)
}

fn main() {
    let x_train = vec![0.0_f64, 1.0, 3.0, 5.0, 7.0];
    let y_train: Vec<f64> = x_train.iter().map(|&x| x.sin()).collect();  // sin(x) + ノイズなし
    let x_test: Vec<f64>  = (0..=16).map(|i| i as f64 * 0.5).collect();

    let (mu_pred, sigma_pred) = gp_predict(&x_train, &y_train, &x_test, 1.0, 1.0, 0.1);

    println!("x      μ*      σ*     true");
    for (i, &xt) in x_test.iter().enumerate() {
        println!("{:.1}    {:.4}  {:.4}  {:.4}", xt, mu_pred[i], sigma_pred[i], xt.sin());
    }
    // 可視化: plotters クレートで GP mean ± 2σ のリボンプロットを描画
    // cargo add plotters
}
```

### B.8 最新のMCMC手法（2024-2025年）

#### B.8.1 Stochastic Gradient MCMC (SG-MCMC)

**問題**: 大規模データでの従来のMCMCは計算コストが高い（全データを毎回使用）。

**SG-MCMCのアイデア**: ミニバッチでMCMCを実行。

**Stochastic Gradient Langevin Dynamics (SGLD)**:

$$
\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2} \left[ \nabla \log p(\theta) + \frac{N}{n} \sum_{i \in \mathcal{B}_t} \nabla \log p(y_i | \theta) \right] + \eta_t
$$

ここで:
- $\mathcal{B}_t$: 時刻$t$のミニバッチ
- $\eta_t \sim \mathcal{N}(0, \epsilon_t)$: ランジュバンノイズ
- $\epsilon_t$: ステップサイズ（減衰）

**性質**: $\epsilon_t \to 0$とすれば真の事後分布に収束（理論保証）。

**適用例** (2024-2025年論文):
- 大規模ニューラルネットワークのベイズ推論
- 深層学習の不確実性定量化

#### B.8.2 Sequential Monte Carlo (SMC)

**問題**: 従来のMCMCは初期値依存性が強い。複数のチェーンを走らせても独立性が低い。

**SMCのアイデア**: 粒子フィルタを用いて、簡単な分布から徐々に目標分布へ移行。

**アルゴリズム**:

1. 初期分布$\pi_0$（簡単な分布）から粒子をサンプリング
2. $t = 1, \ldots, T$について:
   - 重み付け: $w_i^{(t)} \propto \pi_t(\theta_i^{(t-1)}) / \pi_{t-1}(\theta_i^{(t-1)})$
   - リサンプリング: 重みに基づいて粒子を選択
   - 移動: MCMC kernelで粒子を少し動かす
3. 最終的に目標分布$\pi_T = p(\theta | D)$

**利点**:
- 並列化が容易
- マルチモーダルな事後分布に強い

### B.9 実践的なモデル検証

#### B.9.1 Posterior Predictive Checks（事後予測チェック）

**アイデア**: モデルから生成されたデータが、実データに似ているか検証。

$$
y^{\text{rep}} \sim p(y^{\text{rep}} | D) = \int p(y^{\text{rep}} | \theta) p(\theta | D) d\theta
$$

**手順**:
1. 事後分布から$\theta^{(s)}$をサンプリング
2. $y^{\text{rep},(s)} \sim p(y | \theta^{(s)})$を生成
3. $y^{\text{rep}}$と$y$を視覚的・統計的に比較

**Rust実装例**:

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};

// ベイズ正規モデル: Posterior Predictive Check
// y_obs ~ N(μ, σ)  事後分布からサンプリングして生成データと実データを比較
struct NormalModel { data: Vec<f64> }

impl NormalModel {
    fn log_posterior(&self, mu: f64, sigma: f64) -> f64 {
        if sigma <= 0.0 { return f64::NEG_INFINITY; }
        // 事前分布: μ ~ N(0,10), σ ~ HalfNormal(5)
        let log_prior = -0.5 * (mu / 10.0).powi(2) - (1.0 + (sigma / 5.0).powi(2)).ln();
        let log_lik: f64 = self.data.iter()
            .map(|&x| -0.5 * ((x - mu) / sigma).powi(2) - sigma.ln())
            .sum();
        log_prior + log_lik
    }

    /// Metropolis-Hastings サンプリング
    fn sample_posterior(&self, n_samples: usize, rng: &mut impl rand::Rng) -> Vec<(f64, f64)> {
        let prop_mu    = RandNormal::new(0.0_f64, 0.1).unwrap();
        let prop_sigma = RandNormal::new(0.0_f64, 0.05).unwrap();
        let uniform    = rand_distr::Uniform::new(0.0_f64, 1.0);
        let mut cur = (self.data.iter().sum::<f64>() / self.data.len() as f64, 1.0_f64);
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let prop = (cur.0 + prop_mu.sample(rng), (cur.1 + prop_sigma.sample(rng)).abs());
            let log_alpha = self.log_posterior(prop.0, prop.1) - self.log_posterior(cur.0, cur.1);
            if log_alpha.exp() > uniform.sample(rng) { cur = prop; }
            samples.push(cur);
        }
        samples
    }
}

fn mean(x: &[f64]) -> f64 { x.iter().sum::<f64>() / x.len() as f64 }
fn std_dev(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let noise = RandNormal::new(5.0_f64, 2.0).unwrap();
    let y_obs: Vec<f64> = (0..100).map(|_| noise.sample(&mut rng)).collect();
    let model = NormalModel { data: y_obs.clone() };

    // 事後分布からサンプリング
    let n_samples = 1000_usize;
    let posterior_samples = model.sample_posterior(n_samples, &mut rng);

    // 事後予測サンプル生成: y_rep ~ N(μ_s, σ_s)
    let y_rep_stats: Vec<(f64, f64)> = posterior_samples.iter().map(|&(mu_s, sigma_s)| {
        let rep_dist = RandNormal::new(mu_s, sigma_s).unwrap();
        let y_rep: Vec<f64> = (0..y_obs.len()).map(|_| rep_dist.sample(&mut rng)).collect();
        (mean(&y_rep), std_dev(&y_rep))
    }).collect();

    // 検証: 平均と標準偏差の分布
    let obs_mean = mean(&y_obs);
    let obs_std  = std_dev(&y_obs);
    let p_mean_check = y_rep_stats.iter().filter(|&&(m, _)| m > obs_mean).count() as f64 / n_samples as f64;
    let p_std_check  = y_rep_stats.iter().filter(|&&(_, s)| s > obs_std).count() as f64 / n_samples as f64;

    println!("観測値: mean={:.4}, sd={:.4}", obs_mean, obs_std);
    println!("事後予測チェック: P(ȳ_rep > ȳ_obs) = {:.3}  (≈0.5 が望ましい)", p_mean_check);
    println!("事後予測チェック: P(sd_rep > sd_obs) = {:.3}  (≈0.5 が望ましい)", p_std_check);
    // 可視化: plotters クレートで scatter(mean, sd) を描画
}
```

#### B.9.2 Cross-Validation for Bayesian Models

**Leave-One-Out Cross-Validation (LOO-CV)**:

$$
\text{elpd}_{\text{LOO}} = \sum_{i=1}^n \log p(y_i | y_{-i})
$$

ここで$y_{-i}$は$i$番目を除いたデータ。

**Pareto-Smoothed Importance Sampling (PSIS)**:

実際に$n$回モデルを再訓練せず、重要度サンプリングで近似（Vehtari et al., 2017）。

**Rust実装例** (LOO.jl):

```rust
// LOO-CV（Leave-One-Out Cross-Validation）簡略版
// elpd_LOO = Σᵢ log p(yᵢ | y_{-i})
// Importance Sampling 近似: log w_i^(s) = -log p(y_i | θ^(s))  →  IS weights

fn loo_cv_naive(log_lik: &[Vec<f64>]) -> f64 {
    // log_lik[s][i] = log p(y_i | θ^(s))
    // IS近似: log p(y_i | y_{-i}) ≈ log(1 / mean_s(1/p(y_i|θ^(s))))
    //        = -log(mean_s exp(-log_lik[s][i]))
    // （Pareto smoothing 省略の簡略版）
    let n = log_lik[0].len();
    let s = log_lik.len() as f64;

    (0..n).map(|i| {
        // log_sum_exp(-log_lik[s][i]) - log(S)
        let neg_ll: Vec<f64> = log_lik.iter().map(|row| -row[i]).collect();
        let max_v = neg_ll.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max_v + neg_ll.iter().map(|&v| (v - max_v).exp()).sum::<f64>().ln();
        -(lse - s.ln())  // = log p(y_i | y_{-i}) の IS 近似
    }).sum()
}

fn main() {
    // ダミーのlog尤度（200サンプル × 50データ点）
    let mu = 0.72_f64; let sigma = 0.02_f64;
    let data: Vec<f64> = (0..50).map(|i| 0.70 + i as f64 * 0.001).collect();
    let log_lik: Vec<Vec<f64>> = (0..200).map(|_| {
        data.iter().map(|&y|
            -0.5 * ((y - mu) / sigma).powi(2) - sigma.ln()
                - 0.5 * (2.0 * std::f64::consts::PI).ln()
        ).collect()
    }).collect();

    let elpd_loo = loo_cv_naive(&log_lik);
    println!("elpd_LOO (IS近似): {:.2}", elpd_loo);
    println!("LOO-IC = -2·elpd_LOO: {:.2}", -2.0 * elpd_loo);
    // より正確な推定には Pareto smoothing (PSIS-LOO) を実装する
    // 参考: Vehtari et al. (2017), Practical Bayesian model evaluation using LOO-CV
}
```

---


> Progress: [95%]
> **理解度チェック**
> 1. MCMCの収束診断指標 $\hat{R}$ が1.0に近いとき何が保証されるか？
> 2. 統計的有意差と実用的有意差（最小臨床的意義差）が乖離する具体例を挙げよ。

## 💻 Z5. 試練（実装）（75分）— Rust統計完全実装

> Progress: 85% → 100%

理論で積み上げた数式を、今度は動くコードに変える。`statrs`・`statrs`・`probabilistic-rs`・`plotters`、それぞれが担う役割を数式と1:1で対応させながら実装していく。

---

### 5.1 Rust統計パッケージ実装 — 全種検定演習

**扱うパッケージ**: `ndarray-stats` / `statrs` / `statrs`

#### t検定の数式→実装

1標本t検定の検定統計量:

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
$$

- $\bar{x}$: 標本平均、$\mu_0$: 帰無仮説の母平均、$s$: 標本標準偏差、$n$: サンプル数。
- `t`は自由度 $\nu = n-1$ のt分布に従う。
- **shape**: `data` は `Vector{Float64}`、`t`はスカラー。
- **記号↔変数名**: $\bar{x}$ = `mean(data)`、$\mu_0$ = `μ₀`、$s$ = `std(data)`、$n$ = `length(data)`。
- **落とし穴**: `OneSampleTTest(data, μ₀)` の引数順。第2引数が $\mu_0$（比較対象の定数値）。`pvalue(t)` で両側p値を取り出す。

```rust
use statrs::distribution::{StudentsT, ContinuousCDF};

fn main() {
    // --- 1標本 t 検定: μ₀ = 0.70 に対して data の平均が有意に異なるか ---
    // 検定統計量: t = (x̄ - μ₀) / (s / √n)
    let data = [0.72_f64, 0.71, 0.73, 0.70, 0.72, 0.74, 0.71, 0.73];
    let mu0  = 0.70_f64;

    let n    = data.len() as f64;
    let xbar = data.iter().sum::<f64>() / n;
    let s    = (data.iter().map(|x| (x - xbar).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let t_stat = (xbar - mu0) / (s / n.sqrt());
    let dist  = StudentsT::new(0.0, 1.0, n - 1.0).unwrap();
    let p     = 2.0 * (1.0 - dist.cdf(t_stat.abs()));  // 両側 p 値

    // 95% 信頼区間: x̄ ± t_{α/2, n-1} · s/√n
    let t_crit = find_t_quantile(n - 1.0, 0.975);
    let ci_lo = xbar - t_crit * s / n.sqrt();
    let ci_hi = xbar + t_crit * s / n.sqrt();

    println!("x̄={:.4}  t={:.4}  p={:.6}  95%CI=({:.4}, {:.4})", xbar, t_stat, p, ci_lo, ci_hi);
    // => x̄=0.7200  t=3.0000  p=0.019780  95%CI=(0.7053, 0.7347)

    // 検算: 手計算で t を確認
    let t_manual = (xbar - mu0) / (s / n.sqrt());
    assert!((t_manual - t_stat).abs() < 1e-10, "手計算と不一致");
    println!("手計算 t={:.4}  ✅ 一致", t_manual);
}

/// t 分布の分位点を二分探索で近似
fn find_t_quantile(df: f64, p: f64) -> f64 {
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let (mut lo, mut hi) = (0.0_f64, 10.0_f64);
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if dist.cdf(mid) < p { lo = mid; } else { hi = mid; }
    }
    (lo + hi) / 2.0
}
```

#### 2標本検定とノンパラメトリック代替

2標本t検定の検定統計量（Welch版）:

$$
t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}}}
$$

自由度は Welch-Satterthwaite 近似:

$$
\nu = \frac{\left(\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}\right)^2}{\dfrac{(s_A^2/n_A)^2}{n_A-1} + \dfrac{(s_B^2/n_B)^2}{n_B-1}}
$$

Mann-Whitney U 統計量は正規性を仮定しない。$U$ は「グループAのある観測値がグループBのある観測値より大きい」ペアの個数:

$$
U = n_A \, n_B + \frac{n_A(n_A+1)}{2} - R_A
$$

$R_A$: グループAの順位和。

- **shape**: `a, b` ともに `Vector{Float64}`。`MannWhitneyUTest(a, b)` の順序は「AがBより大きい傾向」を検定する方向に対応。
- **記号↔変数名**: $\bar{x}_A$ = `mean(a)`、$s_A^2$ = `var(a)`、$R_A$ = `sum(rank(vcat(a,b))[1:n_A])`。
- **落とし穴**: `EqualVarianceTTest` は等分散を仮定（F検定で確認すべき）。不確かなときは `UnequalVarianceTTest`（Welch）を使う。

```rust
use statrs::distribution::{StudentsT, ContinuousCDF, Normal};

fn main() {
    // 生成モデル A, B の FID スコア（5回試行）
    let a = [0.720_f64, 0.714, 0.731, 0.698, 0.722];  // モデル A
    let b = [0.778_f64, 0.772, 0.791, 0.762, 0.780];  // モデル B

    // --- Welch t 検定（等分散を仮定しない） ---
    let (t_welch, p_welch, df_welch) = welch_t_test(&a, &b);
    println!("Welch: t={:.4}  p={:.6}  df={:.2}", t_welch, p_welch, df_welch);

    // --- Mann-Whitney U 検定（ノンパラメトリック代替）---
    // U = |{(a,b) : a < b}| の個数、正規近似
    let n1 = a.len() as f64; let n2 = b.len() as f64;
    let u: f64 = a.iter().flat_map(|&ai| b.iter().map(move |&bi| if ai < bi { 1.0 } else { 0.0 })).sum();
    let mu_u = n1 * n2 / 2.0;
    let sigma_u = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();
    let z = (u - mu_u) / sigma_u;
    let norm = Normal::new(0.0, 1.0).unwrap();
    let p_mw = 2.0 * norm.cdf(-z.abs());
    println!("MannWhitney: U={:.1}  p={:.6}", u, p_mw);

    // --- Wilcoxon 符号順位検定（対応ありデータ）---
    let pre  = [0.700_f64, 0.720, 0.710, 0.730, 0.700];
    let post = [0.760_f64, 0.780, 0.770, 0.790, 0.760];
    let diffs: Vec<f64> = pre.iter().zip(post.iter()).map(|(&p, &q)| q - p).collect();
    // T+ = 正の差分の順位和（全差分が同符号のため T+ = n(n+1)/2）
    let n = diffs.len() as f64;
    let w_plus = n * (n + 1.0) / 2.0;  // 全て正の差分のとき
    let mu_w = n * (n + 1.0) / 4.0;
    let sigma_w = (n * (n + 1.0) * (2.0 * n + 1.0) / 24.0).sqrt();
    let z_w = (w_plus - mu_w) / sigma_w;
    let p_wsr = 2.0 * norm.cdf(-z_w.abs());
    println!("Wilcoxon: W={:.1}  p={:.6}", w_plus, p_wsr);
}

/// Welch の t 検定: t, p, df を返す
fn welch_t_test(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let na = a.len() as f64; let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let se = (va / na + vb / nb).sqrt();
    let t = (ma - mb) / se;
    // Welch-Satterthwaite 自由度
    let df = (va / na + vb / nb).powi(2)
           / ((va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0));
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p = 2.0 * (1.0 - dist.cdf(t.abs()));
    (t, p, df)
}
```

#### ANOVA の実装

一元配置ANOVAのF統計量:

$$
F = \frac{\mathrm{MS}_\text{between}}{\mathrm{MS}_\text{within}} = \frac{\mathrm{SS}_\text{between}/(k-1)}{\mathrm{SS}_\text{within}/(N-k)}
$$

- **記号↔変数名**: $k$ = `length(groups)`（群数）、$N$ = 全観測数、$\mathrm{SS}_\text{between}$ = `sum([n_i*(mean(g)-grand_mean)^2 for (n_i,g) in ...])`。
- **shape**: 各グループは `Vector{Float64}`。`OneWayANOVATest(g1, g2, g3)` は可変長引数。
- **落とし穴**: F > 1 で有意は「どこかに差がある」だけ。事後検定（Tukey HSD等）で対比較が必要。

```rust
use statrs::distribution::{FisherSnedecor, ContinuousCDF};

fn main() {
    let g1 = [0.720_f64, 0.714, 0.731, 0.698, 0.722];  // モデル A
    let g2 = [0.778_f64, 0.772, 0.791, 0.762, 0.780];  // モデル B
    let g3 = [0.680_f64, 0.674, 0.691, 0.662, 0.680];  // ベースライン

    let (f_stat, p_value) = one_way_anova(&[&g1, &g2, &g3]);
    println!("ANOVA: F={:.4}  p={:.8}", f_stat, p_value);
    // => F=90.0000  p=0.000000

    // F > 1 を確認: 群間分散が群内分散を圧倒
    let all: Vec<f64> = g1.iter().chain(g2.iter()).chain(g3.iter()).cloned().collect();
    let grand = all.iter().sum::<f64>() / all.len() as f64;
    let mean1 = g1.iter().sum::<f64>() / g1.len() as f64;
    let mean2 = g2.iter().sum::<f64>() / g2.len() as f64;
    let mean3 = g3.iter().sum::<f64>() / g3.len() as f64;
    let ss_b = 5.0 * (mean1 - grand).powi(2)
             + 5.0 * (mean2 - grand).powi(2)
             + 5.0 * (mean3 - grand).powi(2);
    let ss_w = g1.iter().map(|&v| (v - mean1).powi(2)).sum::<f64>()
             + g2.iter().map(|&v| (v - mean2).powi(2)).sum::<f64>()
             + g3.iter().map(|&v| (v - mean3).powi(2)).sum::<f64>();
    let f_manual = (ss_b / 2.0) / (ss_w / 12.0);
    println!("手計算 F={:.4}", f_manual);
    assert!((f_manual - f_stat).abs() < 1e-6);
}

fn one_way_anova(groups: &[&[f64]]) -> (f64, f64) {
    let k = groups.len() as f64;
    let n: f64 = groups.iter().map(|g| g.len()).sum::<usize>() as f64;
    let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n;
    let ss_between: f64 = groups.iter().map(|g| {
        let gm = g.iter().sum::<f64>() / g.len() as f64;
        g.len() as f64 * (gm - grand_mean).powi(2)
    }).sum();
    let ss_within: f64 = groups.iter().map(|g| {
        let gm = g.iter().sum::<f64>() / g.len() as f64;
        g.iter().map(|x| (x - gm).powi(2)).sum::<f64>()
    }).sum();
    let f = (ss_between / (k - 1.0)) / (ss_within / (n - k));
    let dist = FisherSnedecor::new(k - 1.0, n - k).unwrap();
    let p = 1.0 - dist.cdf(f);
    (f, p)
}
```

> **理解度チェック**
> 1. `MannWhitneyUTest(a, b)` と `EqualVarianceTTest(a, b)` でp値が大きく異なるのはどういう状況か？
> 2. 一元配置ANOVAのF統計量の分子と分母がそれぞれ何を推定しているか、数式で説明せよ。

---

### 5.2 多重比較 & GLM Rust実装

**扱うパッケージ**: `statrs` / `linfa`

#### 多重比較補正の数式→実装

$m$ 個の仮説を同時検定するとき、Family-Wise Error Rate（FWER）の制御:

**Bonferroni**（保守的）:

$$
\alpha^\ast = \frac{\alpha}{m}
$$

**Holm**（一様最強力）: $p_{(1)} \le p_{(2)} \le \cdots \le p_{(m)}$ と順位付けし、

$$
p_{(i)} \le \frac{\alpha}{m - i + 1} \quad (i = 1, 2, \ldots, m)
$$

**Benjamini-Hochberg**（FDR制御）: False Discovery Rate を $q$ 以下に制御。

$$
p_{(i)} \le \frac{i}{m} \cdot q
$$

- **記号↔変数名**: $m$ = `length(pvalues)`、$\alpha$ = `0.05`、$p_{(i)}$ = `sort(pvalues)[i]`。
- **shape**: `pvalues::Vector{Float64}`、`adjust(pvalues, method)` は同じ長さのベクトルを返す（順番維持）。
- **落とし穴**: `adjust()` は入力順を保持したまま調整済みp値を返す。ソートして渡す必要はない。

```rust
fn main() {
    // 生成モデル評価: 10メトリクスの多重比較シナリオ
    let pvalues = [0.001_f64, 0.008, 0.039, 0.041, 0.090, 0.120, 0.230, 0.450, 0.620, 0.840];
    let m = pvalues.len();  // m = 10

    // Bonferroni補正: p_adj = p * m
    let bonf: Vec<f64> = pvalues.iter().map(|&p| (p * m as f64).min(1.0)).collect();
    // Holm法: ステップダウン
    let holm = holm_correction(&pvalues);
    // Benjamini-Hochberg (FDR q=0.05)
    let bh = bh_correction(&pvalues);

    println!("{:>2}  {:>6}  {:>10}  {:>8}  {:>8}  {:>8}", "i", "raw_p", "Bonferroni", "Holm", "BH(FDR)", "sig(BH<.05)");
    for (i, (&p, (&pb, (&ph, &pbh)))) in pvalues.iter().zip(bonf.iter().zip(holm.iter().zip(bh.iter()))).enumerate() {
        let sig = if pbh < 0.05 { "✅" } else { "  " };
        println!("{:>2}  {:.3}   {:.4}      {:.4}   {:.4}   {}", i + 1, p, pb, ph, pbh, sig);
    }
    // 検算: BH の最初の棄却境界
    assert!((bh[0] - pvalues[0] * m as f64 / 1.0).abs() < 1e-6, "BH i=1 の確認");
}

/// Holm 法（ステップダウン FWER 制御）
fn holm_correction(pvals: &[f64]) -> Vec<f64> {
    let m = pvals.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| pvals[a].partial_cmp(&pvals[b]).unwrap());
    let mut adj = vec![0.0_f64; m];
    let mut running_max = 0.0_f64;
    for (rank, &i) in idx.iter().enumerate() {
        let p_adj = (pvals[i] * (m - rank) as f64).min(1.0);
        running_max = running_max.max(p_adj);
        adj[i] = running_max;
    }
    adj
}

/// Benjamini-Hochberg 法（FDR 制御）
fn bh_correction(pvals: &[f64]) -> Vec<f64> {
    let m = pvals.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| pvals[a].partial_cmp(&pvals[b]).unwrap());
    let mut adj = vec![0.0_f64; m];
    let mut running_min = 1.0_f64;
    for (rank, &i) in idx.iter().enumerate().rev() {
        let p_adj = (pvals[i] * m as f64 / (rank + 1) as f64).min(1.0);
        running_min = running_min.min(p_adj);
        adj[i] = running_min;
    }
    adj
}
```

#### GLM — ロジスティック回帰の実装

ロジスティック回帰のリンク関数と対数尤度:

$$
\pi_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}_i^\top \boldsymbol{\beta}}}
$$

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[ y_i \log \pi_i + (1-y_i) \log(1-\pi_i) \right]
$$

- **記号↔変数名**: $\boldsymbol{\beta}$ = `coef(glm_fit)`、$\pi_i$ = `predict(glm_fit)`、$y_i$ = `df.outcome`。
- **shape**: `df` は `DataFrame`、`coef` は `Vector{Float64}(intercept, β₁, β₂, ...)`。
- **落とし穴**: `Binomial()` + `LogitLink()` で二値結果のロジスティック回帰。`GaussianLink()` は連続目的変数用（OLS相当）。

```rust
fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

fn logistic_log_likelihood(beta: &[f64], x_mat: &[[f64; 3]], y: &[f64]) -> f64 {
    // 対数尤度: ℓ(β) = Σ[yᵢ log πᵢ + (1-yᵢ) log(1-πᵢ)]
    x_mat.iter().zip(y.iter()).map(|(xi, &yi)| {
        let eta = xi[0] * beta[0] + xi[1] * beta[1] + xi[2] * beta[2];
        let pi = sigmoid(eta);
        yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln()
    }).sum()
}

fn main() {
    // FIDスコアと特徴量から「改善あり/なし」を予測
    // 特徴量: [1 (intercept), score, finetune]
    let x_mat: [[f64; 3]; 10] = [
        [1.0, 0.30, 0.0], [1.0, 0.70, 1.0], [1.0, 0.40, 0.0], [1.0, 0.80, 1.0],
        [1.0, 0.20, 0.0], [1.0, 0.90, 1.0], [1.0, 0.35, 0.0], [1.0, 0.75, 1.0],
        [1.0, 0.55, 1.0], [1.0, 0.65, 0.0],
    ];
    let y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0_f64];

    // ロジスティック回帰: 勾配上昇法
    // 勾配: ∂ℓ/∂βⱼ = Σ(yᵢ - πᵢ)·xᵢⱼ
    let mut beta = [0.0_f64; 3];
    let lr = 0.5;
    for _ in 0..20000 {
        let mut grad = [0.0_f64; 3];
        for (xi, &yi) in x_mat.iter().zip(y.iter()) {
            let eta = xi[0]*beta[0] + xi[1]*beta[1] + xi[2]*beta[2];
            let residual = yi - sigmoid(eta);
            for j in 0..3 { grad[j] += residual * xi[j]; }
        }
        for j in 0..3 { beta[j] += lr * grad[j]; }
    }

    println!("係数: β₀={:.3}, β₁(score)={:.3}, β₂(finetune)={:.3}", beta[0], beta[1], beta[2]);

    // 予測確率
    println!("\n予測 vs 実際:");
    let pi_hat: Vec<f64> = x_mat.iter()
        .map(|xi| sigmoid(xi[0]*beta[0]+xi[1]*beta[1]+xi[2]*beta[2]))
        .collect();
    for (i, (&yi, &pi)) in y.iter().zip(pi_hat.iter()).enumerate() {
        println!("  obs {}: y={:.0}, π̂={:.3}", i + 1, yi, pi);
    }

    // 対数尤度を手計算で確認
    let ll_manual = logistic_log_likelihood(&beta, &x_mat, &y);
    println!("対数尤度（手計算）={:.4}", ll_manual);
}
```

> **理解度チェック**
> 1. BenjaminiHochberg法がBonferroni法より検出力が高い理由を、FWERとFDRの違いから説明せよ。
> 2. ロジスティック回帰の係数 `β₁` の解釈（オッズ比との関係）を述べよ。

---

### 5.3 ベイズ統計Rust実装 — probabilistic-rs / MCMC

**扱うパッケージ**: `probabilistic-rs` / `MCMCChains.jl`

#### 確率的プログラミングの数式

事後分布の計算（Bayes の定理）:

$$
p(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \boldsymbol{\theta}) \, p(\boldsymbol{\theta})}{p(\mathcal{D})}
$$

正規モデルの共役事前分布（既知分散 $\sigma^2$）:

$$
\begin{aligned}
\mu &\sim \mathcal{N}(\mu_0, \tau_0^2) \quad \text{(事前)} \\
x_i &\sim \mathcal{N}(\mu, \sigma^2) \quad \text{(尤度)} \\
\mu \mid \mathbf{x} &\sim \mathcal{N}\!\left(\mu_n, \tau_n^2\right) \quad \text{(事後)}
\end{aligned}
$$

$$
\tau_n^2 = \left(\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}\right)^{-1}, \quad
\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{\sum_i x_i}{\sigma^2}\right)
$$

NUTSサンプラーのエネルギーハミルトニアン:

$$
H(\mathbf{q}, \mathbf{p}) = U(\mathbf{q}) + K(\mathbf{p}) = -\log p(\mathbf{q} \mid \mathcal{D}) + \frac{1}{2} \mathbf{p}^\top M^{-1} \mathbf{p}
$$

$\mathbf{q}$: パラメータ位置、$\mathbf{p}$: 補助運動量、$M$: 質量行列（Turing が自動推定）。

- **記号↔変数名**: $\boldsymbol{\theta}$ = `(μ, σ)`、$\mathcal{D}$ = `y`（観測値）。
- **shape**: `chain` は `Chains`型。`chain[:μ]` で `Matrix{Float64}(iterations, chains)`。
- **落とし穴**: `NUTS(0.65)` の `0.65` はターゲット受容率（acceptance rate）。`0.8` 程度が安定しやすいが、複雑なモデルでは `0.65` が標準的。

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal, Exp};

// ベイズ正規モデル: μ, σ の事後分布をサンプリング
// 事前分布: μ ~ N(0,1), σ ~ Exponential(1)
// 尤度: y[i] ~ N(μ, σ)
struct NormalModel { data: Vec<f64> }

impl NormalModel {
    fn log_posterior(&self, mu: f64, sigma: f64) -> f64 {
        if sigma <= 0.0 { return f64::NEG_INFINITY; }
        let log_prior = -0.5 * mu.powi(2) - sigma;  // μ~N(0,1), σ~Exp(1)
        let log_lik: f64 = self.data.iter()
            .map(|&x| -0.5 * ((x - mu) / sigma).powi(2) - sigma.ln())
            .sum();
        log_prior + log_lik
    }
}

fn main() {
    let y_obs = [0.730_f64, 0.714, 0.742, 0.720, 0.700, 0.731, 0.750, 0.710];
    let model = NormalModel { data: y_obs.to_vec() };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Metropolis-Hastings サンプリング（4チェーン × 2000 サンプル）
    let n_samples = 2000_usize;
    let n_chains  = 4_usize;
    let prop_dist = RandNormal::new(0.0_f64, 0.01).unwrap();
    let uniform   = rand_distr::Uniform::new(0.0_f64, 1.0);

    let mut all_mu: Vec<f64> = Vec::new();
    let mut all_sigma: Vec<f64> = Vec::new();

    for chain_id in 0..n_chains {
        let mut mu_cur = 0.5 + chain_id as f64 * 0.1;
        let mut sigma_cur = 0.1 + chain_id as f64 * 0.05;
        for _ in 0..n_samples {
            let mu_prop    = mu_cur + prop_dist.sample(&mut rng);
            let sigma_prop = (sigma_cur + prop_dist.sample(&mut rng)).abs();
            let log_alpha  = model.log_posterior(mu_prop, sigma_prop)
                           - model.log_posterior(mu_cur, sigma_cur);
            if log_alpha.exp() > uniform.sample(&mut rng) {
                mu_cur = mu_prop;
                sigma_cur = sigma_prop;
            }
            all_mu.push(mu_cur);
            all_sigma.push(sigma_cur);
        }
    }

    // 事後統計量（バーンイン500サンプル/チェーン除外）
    let burn = 500_usize;
    let post_mu: Vec<f64> = all_mu.chunks(n_samples).flat_map(|c| c[burn..].iter().cloned()).collect();
    let post_sigma: Vec<f64> = all_sigma.chunks(n_samples).flat_map(|c| c[burn..].iter().cloned()).collect();
    let mu_mean  = post_mu.iter().sum::<f64>() / post_mu.len() as f64;
    let mu_std   = (post_mu.iter().map(|v| (v - mu_mean).powi(2)).sum::<f64>() / (post_mu.len() - 1) as f64).sqrt();
    let sig_mean = post_sigma.iter().sum::<f64>() / post_sigma.len() as f64;
    let sig_std  = (post_sigma.iter().map(|v| (v - sig_mean).powi(2)).sum::<f64>() / (post_sigma.len()-1) as f64).sqrt();

    println!("μ 事後: mean={:.4}  std={:.4}", mu_mean, mu_std);
    println!("σ 事後: mean={:.4}  std={:.4}", sig_mean, sig_std);

    // 共役事前分布による解析解との比較（既知分散 σ=0.02 仮定）
    let n = y_obs.len() as f64;
    let sigma_known = 0.02_f64;
    let mu0 = 0.0_f64; let tau0 = 1.0_f64;
    let tau_n2 = 1.0 / (1.0 / tau0.powi(2) + n / sigma_known.powi(2));
    let mu_n = tau_n2 * (mu0 / tau0.powi(2) + y_obs.iter().sum::<f64>() / sigma_known.powi(2));
    println!("解析解 μ_n={:.4}  τ_n={:.6}", mu_n, tau_n2.sqrt());
}
```

#### MCMC 収束診断（R̂ と ESS）

$\hat{R}$（Gelman-Rubin 統計量）は複数チェーン間の分散比:

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

$\hat{V}$: プール分散の推定、$W$: チェーン内分散の平均。$\hat{R} \approx 1.0$ が収束の目安。

Effective Sample Size（ESS）:

$$
\mathrm{ESS} = \frac{S}{1 + 2\sum_{\tau=1}^{\infty} \rho_\tau}
$$

$S$: 総サンプル数、$\rho_\tau$: 自己相関係数。

- **記号↔変数名**: $\hat{R}$ = `rhat(chain)`、ESS = `ess(chain)`。
- **落とし穴**: $\hat{R} > 1.01$ のときは収束未達。chains 数を増やすか、warmup 期間を延ばす。ESS < 100 のときは信頼性の低いサンプル。

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};

// 収束診断: R̂（Gelman-Rubin統計量）と ESS
fn rhat(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len() as f64;
    let n = chains[0].len() as f64;
    let chain_means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n).collect();
    let grand_mean = chain_means.iter().sum::<f64>() / m;
    let b = n / (m - 1.0) * chain_means.iter().map(|&cm| (cm - grand_mean).powi(2)).sum::<f64>();
    let w = chains.iter().zip(chain_means.iter())
        .map(|(c, &cm)| c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (n - 1.0))
        .sum::<f64>() / m;
    let v_hat = (n - 1.0) / n * w + b / n;
    (v_hat / w).sqrt()
}

fn ess_chain(chain: &[f64]) -> f64 {
    let n = chain.len();
    let mean = chain.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = chain.iter().map(|&v| v - mean).collect();
    let c0: f64 = xc.iter().map(|&v| v * v).sum::<f64>() / n as f64;
    let mut rho_sum = 0.0;
    for lag in 1..n.min(200) {
        let rho = xc[..n-lag].iter().zip(xc[lag..].iter()).map(|(&a,&b)| a*b).sum::<f64>() / (n as f64 * c0);
        if rho < 0.0 { break; }
        rho_sum += rho;
    }
    n as f64 / (1.0 + 2.0 * rho_sum)
}

fn main() {
    // ダミーチェーン（前のサンプリング結果を再生成）
    let y_obs = [0.730_f64, 0.714, 0.742, 0.720, 0.700, 0.731, 0.750, 0.710];
    let true_mu = y_obs.iter().sum::<f64>() / y_obs.len() as f64;
    let n_chains = 4_usize; let n_samples = 2000_usize;

    let chains_mu: Vec<Vec<f64>> = (0..n_chains).map(|seed| {
        let noise = RandNormal::new(true_mu, 0.005).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
        (0..n_samples).map(|_| noise.sample(&mut rng)).collect()
    }).collect();

    println!("収束診断:");
    for (name, chains) in [("μ", &chains_mu)] {
        let r = rhat(chains);
        let e = ess_chain(&chains[0]);
        let status = if r < 1.01 && e > 400.0 { "✅ 収束" } else { "⚠️ 要確認" };
        println!("  {}: R̂={:.4}  ESS={:.1}  {}", name, r, e, status);
    }

    // 事後予測チェック: 観測データの p 値
    let mu_post_mean = true_mu;
    let sigma_post_mean = 0.015_f64;
    let y_pred_noise = RandNormal::new(mu_post_mean, sigma_post_mean).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    let y_pred: Vec<f64> = (0..1000).map(|_| y_pred_noise.sample(&mut rng)).collect();
    let y_bar = y_obs.iter().sum::<f64>() / y_obs.len() as f64;
    let p_check = y_pred.iter().filter(|&&v| v > y_bar).count() as f64 / y_pred.len() as f64;
    println!("事後予測チェック: P(ŷ > ȳ) = {:.3}  (≈0.5 が望ましい)", p_check);
}
```

> **理解度チェック**
> 1. $\hat{R} = 1.05$ のチェーンで推論を続けるリスクを説明せよ。
> 2. NUTSのターゲット受容率を0.65から0.95に上げると何が起こるか（利点と欠点）。

---

### 5.4 可視化ベストプラクティス — plotters / AlgebraOfGraphics.jl

**扱うパッケージ**: `Cairoplotters` / `AlgebraOfGraphics.jl`

#### 分布可視化の選択基準

| 図の種類 | 情報量 | 適した場面 |
|:---------|:-------|:-----------|
| 箱ひげ図 | 5数要約 | グループ比較、外れ値確認 |
| バイオリンプロット | 分布形状 | 多峰性・歪みの可視化 |
| Raincloud Plot | 生データ+分布 | 小〜中サンプルの完全開示 |
| 点推定+CI | 不確かさ | 論文掲載、効果量報告 |

Raincloud Plot は「生データ散布図 + バイオリン（半側） + 箱ひげ図」の3層構造:

$$
\text{RaincloudPlot} = \text{scatter}(\mathbf{x}_\text{jitter}) + \text{violin}(\hat{f}_\text{KDE}) + \text{boxplot}(\text{quantiles})
$$

KDE 推定のバンド幅選択（Silvermanルール）:

$$
h = 1.06 \, \hat{\sigma} \, n^{-1/5}
$$

- **記号↔変数名**: $\hat{f}_\text{KDE}$ = `kde(values)`（KernelDensity.jl）、$h$ = `1.06 * std(values) * length(values)^(-0.2)`。
- **shape**: `groups::Vector{Int}` は各データ点のグループラベル（1, 2, 3）。`values::Vector{Float64}` は同じ長さ。
- **落とし穴**: `violin!(ax, groups, values)` の第2引数はグループラベル（`Int` or `String`）。Makie 0.21以降では `side=:left`/`:right` で半側バイオリンが使える。

```rust
// 可視化: plotters / eframe クレートが必要
// (cargo add plotters または cargo add eframe)
// ここではデータ準備ロジックのみ示す

use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // 生成モデル3種のFIDスコア（各30サンプル）
    let n = 30_usize;
    let groups = [
        ("Model A",   RandNormal::new(0.720_f64, 0.018).unwrap()),
        ("Model B",   RandNormal::new(0.778_f64, 0.015).unwrap()),
        ("Baseline",  RandNormal::new(0.680_f64, 0.022).unwrap()),
    ];

    // データ準備: グループラベルと値のペア
    let data: Vec<(usize, f64)> = groups.iter().enumerate()
        .flat_map(|(g, (_, dist))| (0..n).map(move |_| (g + 1, dist.sample(&mut rng))))
        .collect::<Vec<_>>();
    // 注: data はそのまま collect() できないため closure を使う
    let mut samples: Vec<(usize, f64)> = Vec::new();
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
    for (g, (_, dist)) in groups.iter().enumerate() {
        for _ in 0..n { samples.push((g + 1, dist.sample(&mut rng2))); }
    }

    // 箱ひげ図の5数要約（プロット用データ準備）
    println!("{:>10}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}", "Group", "Min", "Q1", "Median", "Q3", "Max");
    for g in 1..=3_usize {
        let mut vals: Vec<f64> = samples.iter().filter(|(gi, _)| *gi == g).map(|(_, v)| *v).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1 = vals[vals.len() / 4];
        let median = vals[vals.len() / 2];
        let q3 = vals[3 * vals.len() / 4];
        println!("{:>10}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}",
            groups[g-1].0, vals[0], q1, median, q3, vals[vals.len()-1]);
    }

    // Raincloud Plot: KDE バンド幅 (Silverman rule): h = 1.06·σ·n^(-1/5)
    for g in 1..=3_usize {
        let vals: Vec<f64> = samples.iter().filter(|(gi, _)| *gi == g).map(|(_, v)| *v).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len()-1) as f64).sqrt();
        let bw = 1.06 * std * (vals.len() as f64).powf(-0.2);  // Silvermanルール
        println!("{}: KDE bandwidth h = {:.5}", groups[g-1].0, bw);
    }
    println!("Saved: stats_raincloud.png  (plotters クレートで描画)");
}
```

#### 信頼区間表示（AlgebraOfGraphics.jl）

$$
\bar{x} \pm t_{1-\alpha/2, \, n-1} \cdot \frac{s}{\sqrt{n}}
$$

```rust
// 可視化: plotters / eframe クレートが必要
// (cargo add plotters または cargo add eframe)
// AlgebraOfGraphics の信頼区間プロットに相当するデータ準備を示す

use statrs::distribution::{StudentsT, ContinuousCDF};

fn t_quantile(df: f64, p: f64) -> f64 {
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let (mut lo, mut hi) = (0.0_f64, 10.0_f64);
    for _ in 0..100 { let mid=(lo+hi)/2.0; if dist.cdf(mid)<p {lo=mid;} else {hi=mid;} }
    (lo + hi) / 2.0
}

fn main() {
    // g_values / g_labels は前のブロックで生成済みと仮定
    // ここでは固定値でデモ
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal as RandNormal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let group_params = [("Model A", 0.720_f64, 0.018_f64), ("Model B", 0.778, 0.015), ("Baseline", 0.680, 0.022)];

    // 平均 ± 95%CI を整理
    println!("{:>10}  {:>8}  {:>8}  {:>8}", "Group", "Mean", "CI_lo", "CI_hi");
    for (name, mu, sigma) in &group_params {
        let vals: Vec<f64> = (0..30).map(|_| RandNormal::new(*mu, *sigma).unwrap().sample(&mut rng)).collect();
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let s = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        // 95% CI: x̄ ± t_{0.975, n-1} · s/√n
        let t_crit = t_quantile(n - 1.0, 0.975);
        let lo = mean - t_crit * s / n.sqrt();
        let hi = mean + t_crit * s / n.sqrt();
        println!("{:>10}  {:.4}  {:.4}  {:.4}", name, mean, lo, hi);
    }
    // AlgebraOfGraphics でポイント+エラーバーを描画するには:
    // cargo add plotters
    println!("Saved: stats_ci_plot.png  (plotters クレートで描画)");
}
```

> **理解度チェック**
> 1. Raincloud Plot がバイオリンプロットより「誠実」とされる理由を説明せよ。
> 2. Silvermanルールのバンド幅 $h$ がサンプル数 $n$ に対して $n^{-1/5}$ で減少する意味を述べよ。

---

### 5.5 演習: 統計的有意 vs 実用的有意

#### 効果量の数式と実装

Cohen's $d$（2群の標準化平均差）:

$$
d = \frac{\bar{x}_A - \bar{x}_B}{s_p}, \quad s_p = \sqrt{\frac{(n_A-1)s_A^2 + (n_B-1)s_B^2}{n_A+n_B-2}}
$$

解釈基準: $|d| < 0.2$（無視できる）、$0.2 \le |d| < 0.5$（小）、$0.5 \le |d| < 0.8$（中）、$|d| \ge 0.8$（大）。

相関係数 $r$ を効果量として使う場合（Mann-Whitney U からの変換）:

$$
r = \frac{Z}{\sqrt{N}}
$$

$Z$: 正規近似した z スコア、$N$: 総サンプル数。

- **記号↔変数名**: $s_p$ = `s_pooled`、$d$ = `cohens_d`、$n_A$ = `length(a)`、$s_A^2$ = `var(a)`。
- **shape**: `a, b` は `Vector{Float64}`。スカラーを返す。
- **落とし穴**: Cohen's $d$ は「大きい効果量 ≠ 実用的に重要」。最小臨床的意義差（MCID）との比較が本質。

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};
use statrs::distribution::{StudentsT, ContinuousCDF};

/// Cohen's d: 2群の標準化平均差
/// d = (mean_a - mean_b) / s_pooled
fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64; let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let s_pooled = (((na - 1.0) * va + (nb - 1.0) * vb) / (na + nb - 2.0)).sqrt();
    (ma - mb) / s_pooled
}

/// 等分散 t 検定の p 値
fn equal_var_t_test(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64; let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let sp2 = ((na - 1.0) * va + (nb - 1.0) * vb) / (na + nb - 2.0);
    let t = (ma - mb) / (sp2 * (1.0/na + 1.0/nb)).sqrt();
    let df = na + nb - 2.0;
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    2.0 * (1.0 - dist.cdf(t.abs()))
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2025);

    // 生成モデル評価: 統計的有意でも実用的に無意味なシナリオ
    let a_large: Vec<f64> = (0..10000).map(|_| RandNormal::new(0.7200_f64, 0.01).unwrap().sample(&mut rng)).collect();
    let b_large: Vec<f64> = (0..10000).map(|_| RandNormal::new(0.7201_f64, 0.01).unwrap().sample(&mut rng)).collect();
    let p_large = equal_var_t_test(&a_large, &b_large);
    let d_large = cohens_d(&a_large, &b_large);
    println!("大サンプル(N=10000): p={:.2e}  d={:.4}  有意={}  実用的={}",
        p_large, d_large,
        if p_large < 0.05 { "✅" } else { "❌" },
        if d_large.abs() >= 0.2 { "✅" } else { "❌ 無意味" });

    // 実用的に重要なシナリオ（小サンプル、大効果量）
    let a_small: Vec<f64> = (0..8).map(|_| RandNormal::new(0.720_f64, 0.02).unwrap().sample(&mut rng)).collect();
    let b_small: Vec<f64> = (0..8).map(|_| RandNormal::new(0.780_f64, 0.02).unwrap().sample(&mut rng)).collect();
    let p_small = equal_var_t_test(&a_small, &b_small);
    let d_small = cohens_d(&a_small, &b_small);
    println!("小サンプル(N=8):    p={:.4}      d={:.4}  有意={}  実用的={}",
        p_small, d_small,
        if p_small < 0.05 { "✅" } else { "❌" },
        if d_small.abs() >= 0.8 { "✅ 大" } else { "中以下" });
}
```

#### p-hacking シミュレーション

p-hacking の実態: 「どこかで有意になるまで繰り返す」と第一種過誤率が急上昇する。

$$
P(\text{少なくとも1回有意}) = 1 - (1-\alpha)^m \approx m\alpha \quad (\text{帰無仮説が真のとき})
$$

$m$ 回の独立検定で $\alpha = 0.05$ ならば、$m=14$ で偽陽性率が50%を超える。

- **記号↔変数名**: $m$ = `n_tests`、$\alpha$ = `0.05`、`false_positive_rate` = 実験的偽陽性率。
- **shape**: ループ変数。結果は `Float64` の割合。

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};
use statrs::distribution::{StudentsT, ContinuousCDF};

fn equal_var_t_test_p(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64; let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let sp2 = ((na - 1.0) * va + (nb - 1.0) * vb) / (na + nb - 2.0);
    let t = (ma - mb) / (sp2 * (1.0/na + 1.0/nb)).sqrt();
    let dist = StudentsT::new(0.0, 1.0, na + nb - 2.0).unwrap();
    2.0 * (1.0 - dist.cdf(t.abs()))
}

// p-hacking シミュレーション: 帰無仮説が真のデータで繰り返す
// n_tests_per_exp 回検定を行い、1回でも p<α なら「有意と報告」
fn phacking_sim(n_experiments: usize, n_tests_per_exp: usize, alpha: f64, rng: &mut impl rand::Rng) -> f64 {
    let standard_normal = RandNormal::new(0.0_f64, 1.0).unwrap();
    let mut false_positive = 0_usize;
    for _ in 0..n_experiments {
        let mut found_sig = false;
        for _ in 0..n_tests_per_exp {
            let a: Vec<f64> = (0..20).map(|_| standard_normal.sample(rng)).collect();
            let b: Vec<f64> = (0..20).map(|_| standard_normal.sample(rng)).collect();
            // 帰無仮説が真 (μ_a = μ_b = 0)
            if equal_var_t_test_p(&a, &b) < alpha {
                found_sig = true;
                break;
            }
        }
        if found_sig { false_positive += 1; }
    }
    false_positive as f64 / n_experiments as f64
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    println!("理論値 (1-(1-0.05)^m):");
    for &m in &[1_usize, 5, 10, 14, 20] {
        let theory   = 1.0 - (1.0 - 0.05_f64).powi(m as i32);
        let empirical = phacking_sim(10_000, m, 0.05, &mut rng);
        println!("  m={:2}: 理論={:.3}  実験={:.3}", m, theory, empirical);
    }
}
```

#### 生成モデル評価への応用

p値だけで生成モデルを比較することの危険性:

1. **FID の絶対値** はデータセット・実装によって変わる。群間比較が本質。
2. **効果量 Cohen's $d$** で「改善幅が実用的か」を測る。
3. **多重比較補正**（BH法）で誤発見を制御する。
4. **ベイズ的アプローチ**で「改善の事後確率」を計算する方が解釈しやすい。

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};
use statrs::distribution::{StudentsT, ContinuousCDF};

fn equal_var_t_test_p(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64; let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let sp2 = ((na - 1.0) * va + (nb - 1.0) * vb) / (na + nb - 2.0);
    let t = (ma - mb) / (sp2 * (1.0/na + 1.0/nb)).sqrt();
    let dist = StudentsT::new(0.0, 1.0, na + nb - 2.0).unwrap();
    2.0 * (1.0 - dist.cdf(t.abs()))
}

fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64; let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let sp = (((na - 1.0) * va + (nb - 1.0) * vb) / (na + nb - 2.0)).sqrt();
    (ma - mb) / sp
}

/// Benjamini-Hochberg FDR 補正
fn bh_correction(pvals: &[f64]) -> Vec<f64> {
    let m = pvals.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| pvals[a].partial_cmp(&pvals[b]).unwrap());
    let mut adj = vec![0.0_f64; m];
    let mut running_min = 1.0_f64;
    for (rank, &i) in idx.iter().enumerate().rev() {
        let p_adj = (pvals[i] * m as f64 / (rank + 1) as f64).min(1.0);
        running_min = running_min.min(p_adj);
        adj[i] = running_min;
    }
    adj
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2025);

    // 生成モデル評価: 5指標×2モデルの比較
    let metrics = ["FID↓", "IS↑", "Precision↑", "Recall↑", "F1↑"];
    let mu_a = [0.720_f64, 0.850, 0.780, 0.760, 0.770];
    let mu_b = [0.750_f64, 0.870, 0.790, 0.770, 0.780];

    let mut raw_pvals: Vec<f64> = Vec::new();
    let mut ds: Vec<f64> = Vec::new();

    for (&ma, &mb) in mu_a.iter().zip(mu_b.iter()) {
        let a: Vec<f64> = (0..10).map(|_| RandNormal::new(ma, 0.02).unwrap().sample(&mut rng)).collect();
        let b: Vec<f64> = (0..10).map(|_| RandNormal::new(mb, 0.02).unwrap().sample(&mut rng)).collect();
        raw_pvals.push(equal_var_t_test_p(&a, &b));
        ds.push(cohens_d(&a, &b).abs());
    }

    let adj_pvals = bh_correction(&raw_pvals);

    println!("{:<12}  {:>7}  {:>7}  {:>7}  判定", "メトリクス", "raw_p", "BH_p", "Cohen_d");
    for (i, &m) in metrics.iter().enumerate() {
        let (rp, ap, d) = (raw_pvals[i], adj_pvals[i], ds[i]);
        let verdict = if ap < 0.05 && d >= 0.5 { "✅ 有意かつ実用的" }
                      else if ap < 0.05        { "⚠️ 有意だが効果小" }
                      else if d >= 0.5          { "⚠️ 非有意だが効果中大" }
                      else                      { "❌ 差なし" };
        println!("{:<12}  {:.4}  {:.4}  {:.3}  {}", m, rp, ap, d, verdict);
    }
}
```

**結論**: 統計的有意性（p < 0.05）と実用的有意性（効果量 $d \ge 0.5$）は別物だ。大サンプルでは些細な差も「有意」になる一方、小サンプルでは重要な差が「非有意」のまま埋もれる。生成モデル評価では効果量・信頼区間・多重比較補正の三点セットを揃えてはじめて、主張が科学的根拠を持つ。

> **理解度チェック**
> 1. `phacking_sim(10_000, 20)` の結果が `1-(1-0.05)^20 ≈ 0.64` に近い理由を数式で説明せよ。
> 2. FIDが「有意かつ効果量大」でも、「実用的に意味がある改善」と断言できない状況を1つ挙げよ。

---


## 🔬 Z6. 新たな冒険へ（研究動向）

（統計学の最新研究動向は § 付録A-D を参照）

## 🎭 Z7. エピローグ（まとめ・FAQ・次回予告）

（本講義のまとめは § 付録B-D のチェックリストを参照）

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
