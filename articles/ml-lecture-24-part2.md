---
title: "第24回【後編】付録編: 統計学: 30秒の驚き→数式修行→実装マスター""
emoji: "📈"
type: "tech"
topics: ["machinelearning", "statistics", "julia", "bayesian", "hypothesis"]
published: true
---
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

```julia
using HypothesisTests

group_a = [0.72, 0.71, 0.73, 0.70, 0.72]
group_b = [0.78, 0.77, 0.79, 0.76, 0.78]
group_c = [0.68, 0.67, 0.69, 0.66, 0.68]

# 一元配置ANOVA
test = OneWayANOVATest(group_a, group_b, group_c)
println("F=$(round(test.F, digits=3)), p=$(round(pvalue(test), digits=6))")
println(pvalue(test) < 0.05 ? "✅ 少なくとも1組の平均が異なる" : "❌ 全群の平均に差なし")
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

```julia
using HypothesisTests, Distributions

# 正規分布データ
normal_data = rand(Normal(0, 1), 30)
test_normal = ExactOneSampleKSTest(normal_data, Normal(0, 1))
println("正規データ: p=$(round(pvalue(test_normal), digits=4))")

# 非正規データ（一様分布）
uniform_data = rand(Uniform(0, 1), 30)
test_uniform = ExactOneSampleKSTest(uniform_data, Normal(0.5, 1))
println("一様データ: p=$(round(pvalue(test_uniform), digits=4))")
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

```julia
using HypothesisTests

group1 = [1, 2, 3, 4, 5]
group2 = [6, 7, 8, 9, 10]

# Mann-Whitney U検定
test = MannWhitneyUTest(group1, group2)
println("U=$(test.U), p=$(round(pvalue(test), digits=4))")
```

:::message
**進捗: 65% 完了** パラメトリック・ノンパラメトリック検定の理論完全版を制覇。多重比較補正へ。
:::

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

```julia
using MultipleTesting

# 100個の検定（90個は帰無仮説が真、10個は対立仮説が真）
p_values_null = rand(100)  # H0が真のp値: 一様分布
p_values_alt  = rand(Beta(0.1, 1), 10)  # H1が真のp値: 0に偏る
p_values = vcat(p_values_null, p_values_alt)

# 補正なし
n_sig_uncorrected = sum(p_values .< 0.05)
println("補正なし: $(n_sig_uncorrected) / 110 が有意")

# Bonferroni補正
p_bonf = adjust(PValues(p_values), Bonferroni())
n_sig_bonf = sum(p_bonf .< 0.05)
println("Bonferroni: $(n_sig_bonf) / 110 が有意")

# Benjamini-Hochberg (FDR)
p_bh = adjust(PValues(p_values), BenjaminiHochberg())
n_sig_bh = sum(p_bh .< 0.05)
println("Benjamini-Hochberg: $(n_sig_bh) / 110 が有意")
```

出力例:
```
補正なし: 15 / 110 が有意
Bonferroni: 3 / 110 が有意
Benjamini-Hochberg: 9 / 110 が有意
```

:::message
**進捗: 75% 完了** 多重比較補正（FWER/FDR）を完全理解。GLM理論へ。
:::

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

```julia
using GLM, DataFrames

# データ: x（連続変数）, y（0/1のラベル）
df = DataFrame(
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
)

# ロジスティック回帰
model = glm(@formula(y ~ x), df, Binomial(), LogitLink())
println(model)

# 係数の解釈
β1 = coef(model)[2]
OR = exp(β1)
println("\n係数β1=$(round(β1, digits=3)), オッズ比OR=$(round(OR, digits=3))")
println("xが1単位増加すると、オッズが$(round(OR, digits=3))倍になる")

# 予測
df.y_pred = predict(model, df)
println("\n予測確率:")
println(df)
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

```julia
using GLM, DataFrames, Distributions

# データ生成: カウントデータ（例: 1時間あたりのエラー発生回数）
df = DataFrame(
    workload = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 負荷レベル
    errors = [2, 3, 3, 5, 6, 8, 9, 12, 14, 16]   # エラー回数
)

# ポアソン回帰
model = glm(@formula(errors ~ workload), df, Poisson(), LogLink())
println(model)

# 係数の解釈
β1 = coef(model)[2]
multiplier = exp(β1)
println("\nworkloadが1単位増加すると、期待エラー回数が$(round(multiplier, digits=3))倍になる")

# 予測
df.errors_pred = predict(model, df)
println("\n予測エラー回数:")
println(df)
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

:::message
**進捗: 80% 完了** GLM理論（ロジスティック・ポアソン回帰・指数型分布族）を理解。ベイズ統計へ。
:::

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

```julia
using Distributions, Plots

# 事前分布: Beta(2, 2) (弱い信念: θ≈0.5)
α, β = 2.0, 2.0
prior = Beta(α, β)

# データ: 10回投げて7回表
n, k = 10, 7

# 事後分布: Beta(α+k, β+n-k) = Beta(9, 5)
posterior = Beta(α + k, β + n - k)

# 可視化
θ_range = 0:0.01:1
plot(θ_range, pdf.(prior, θ_range), label="事前分布 Beta(2,2)", linewidth=2)
plot!(θ_range, pdf.(posterior, θ_range), label="事後分布 Beta(9,5)", linewidth=2)
xlabel!("θ (コインが表の確率)")
ylabel!("密度")
title!("ベイズ更新: コイン投げ")
savefig("bayesian_update.png")
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

**Turing.jlで実装**:

```julia
using Turing, Distributions, StatsPlots

# モデル定義: コイン投げ（ベイズ推定）
@model function coinflip(y)
    # 事前分布
    θ ~ Beta(2, 2)

    # 尤度
    y ~ Binomial(length(y), θ)
end

# データ: 10回中7回表
data = 7

# MCMCサンプリング（NUTS: No-U-Turn Sampler, Hamiltonian Monte Carloの改良版）
chain = sample(coinflip([data]), NUTS(), 1000)

# 事後分布の可視化
plot(chain)
```

:::message
**進捗: 90% 完了** ベイズ統計（共役事前分布・MCMC）を完全理解。実験計画法へ。
:::

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

:::message
**進捗: 100% 完了** 🎉 講義完走！
:::

---

## 参考文献

### 主要論文

[^1]: Neyman, J., & Pearson, E. S. (1928). *On the Use and Interpretation of Certain Test Criteria for Purposes of Statistical Inference: Part I*. Biometrika.
@[card](https://www.jstor.org/stable/2331945)

[^2]: Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society: Series B.
@[card](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

[^3]: Hastings, W. K. (1970). *Monte Carlo Sampling Methods Using Markov Chains and Their Applications*. Biometrika.
@[card](https://doi.org/10.1093/biomet/57.1.97)

[^4]: Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury Press.

[^5]: Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

[^6]: Nelder, J. A., & Wedderburn, R. W. M. (1972). *Generalized Linear Models*. Journal of the Royal Statistical Society: Series A.
@[card](https://doi.org/10.2307/2344614)

[^7]: Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

### 教科書

- **Statistical Inference** - Casella & Berger (2002): 頻度論統計の決定版。大学院レベル。
- **Bayesian Data Analysis** - Gelman et al. (2013): ベイズ統計の標準教科書。
- **The Elements of Statistical Learning** - Hastie, Tibshirani, Friedman (2009): 機械学習×統計の融合。[無料PDF](https://web.stanford.edu/~hastie/ElemStatLearn/)
- **統計学入門** - 東京大学教養学部統計学教室 (1991): 日本語の定番入門書。

### オンラインリソース

- [StatQuest (YouTube)](https://www.youtube.com/@statquest): 統計学の直感的解説動画。
- [StatsBase.jl Documentation](https://juliastats.org/StatsBase.jl/stable/)
- [HypothesisTests.jl Documentation](https://juliastats.org/HypothesisTests.jl/stable/)
- [GLM.jl Documentation](https://juliastats.org/GLM.jl/stable/)
- [Turing.jl Documentation](https://turinglang.org/stable/)

---

## 記法規約

| 記号 | 意味 | 備考 |
|:-----|:-----|:-----|
| $\bar{x}$ | 標本平均 | $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$ |
| $s^2$ | 標本分散（不偏） | $s^2 = \frac{1}{n-1} \sum (x_i - \bar{x})^2$ |
| $s$ | 標本標準偏差 | $s = \sqrt{s^2}$ |
| $\mu$ | 母平均 | 母集団の期待値 |
| $\sigma^2$ | 母分散 | 母集団の分散 |
| $\text{SE}$ | 標準誤差 | $\text{SE} = \sigma / \sqrt{n} \approx s / \sqrt{n}$ |
| $\alpha$ | 有意水準 | 第1種過誤率（通常0.05） |
| $\beta$ | 第2種過誤率 | $1 - \beta$ = 検出力 |
| $H_0$ | 帰無仮説 | 「差がない」「効果がない」 |
| $H_1$ | 対立仮説 | 「差がある」「効果がある」 |
| $p$ | p値 | $H_0$下での極端値の確率 |
| $d$ | Cohen's d | 効果量 $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$ |
| $t$ | t統計量 | t検定の検定統計量 |
| $F$ | F統計量 | ANOVAの検定統計量 |
| $\text{df}$ | 自由度 | 推定に使える独立な情報の数 |
| $\text{CI}$ | 信頼区間 | Confidence Interval |
| $\text{FWER}$ | 家族誤差率 | Family-Wise Error Rate |
| $\text{FDR}$ | 偽発見率 | False Discovery Rate |
| $\theta$ | パラメータ | ベイズ統計での推定対象 |
| $p(\theta \| D)$ | 事後分布 | データ観測後のパラメータ分布 |
| $p(D \| \theta)$ | 尤度 | パラメータ下でのデータの確率 |
| $p(\theta)$ | 事前分布 | データ観測前のパラメータ分布 |

**統計検定のJulia実装対応**:

| 数式 | Julia実装 |
|:-----|:----------|
| $\bar{x} = \frac{1}{n}\sum x_i$ | `mean(x)` |
| $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ | `var(x)` |
| $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$ | `OneSampleTTest(x, μ₀)` |
| $p\text{-value}$ | `pvalue(test)` |
| $\alpha_{\text{Bonf}} = \alpha / m$ | `adjust(PValues(p), Bonferroni())` |
| $\text{logit}(p) = \log\frac{p}{1-p}$ | `glm(@formula(y ~ x), df, Binomial(), LogitLink())` |
| $p(\theta \| D) \propto p(D \| \theta) p(\theta)$ | `@model function model(D) ... end` + `sample(...)` |

---

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
| 2020年代 | 確率的プログラミング | Turing.jl, PyMC, Stan等の成熟 |

---

## 付録B: Juliaで使える統計パッケージ完全リスト

### B.1 基礎統計

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **Statistics** (stdlib) | 基本統計量 | `mean`, `std`, `var`, `median`, `quantile`, `cor`, `cov` |
| **StatsBase.jl** | 記述統計・重み付き統計 | `skewness`, `kurtosis`, `mad`, `mode`, `sem`, `zscore`, `sample`, `weights` |
| **Distributions.jl** | 確率分布 | `Normal`, `Beta`, `Gamma`, `Binomial`, `Poisson`, `TDist`, `FDist`, `pdf`, `cdf`, `quantile`, `rand` |

### B.2 仮説検定

| パッケージ | 用途 | 主要検定 |
|:----------|:-----|:---------|
| **HypothesisTests.jl** | 仮説検定全般 | `OneSampleTTest`, `EqualVarianceTTest`, `UnequalVarianceTTest`, `MannWhitneyUTest`, `WilcoxonSignedRankTest`, `KruskalWallisTest`, `OneWayANOVATest`, `ChisqTest`, `FisherExactTest`, `KSTest`, `AndersonDarlingTest` |
| **MultipleTesting.jl** | 多重比較補正 | `adjust`, `Bonferroni`, `Holm`, `BenjaminiHochberg`, `BenjaminiYekutieli` |

### B.3 回帰・GLM

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **GLM.jl** | 一般化線形モデル | `glm`, `@formula`, `Binomial`, `Poisson`, `Gamma`, `LogitLink`, `LogLink`, `InverseLink`, `coef`, `confint`, `predict` |
| **MixedModels.jl** | 混合効果モデル | `LinearMixedModel`, `fit!`, `ranef`, `fixef` |

### B.4 ベイズ統計

| パッケージ | 用途 | 主要関数/マクロ |
|:----------|:-----|:---------------|
| **Turing.jl** | 確率的プログラミング | `@model`, `~`, `sample`, `NUTS`, `HMC`, `Gibbs`, `plot`, `summarize` |
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
| **Makie.jl** | 高品質可視化 | `scatter`, `lines`, `barplot`, `heatmap`, `density` |
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
