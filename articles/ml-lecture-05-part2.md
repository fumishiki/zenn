---
title: "第5回: 測度論的確率論・確率過程入門: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "📏"
type: "tech"
topics: ["machinelearning", "deeplearning", "measuretheory", "stochasticprocesses", "python"]
published: true
slug: "ml-lecture-05-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Python"]
keywords: ["MCMC", "importance sampling", "SDE", "Langevin dynamics", "Fokker-Planck equation"]
---

> **📘 本記事は後編（実装編）です**: [前編（理論編）はこちら](/articles/ml-lecture-05-part1)

## Learning Objectives

この後編を修了すると、以下のスキルが身につきます:

- [ ] Monte Carlo積分を実装し、 `$O(1/\sqrt{N})$` の収束レートを確認できる
- [ ] 分散低減法（重点サンプリング、層化サンプリング、制御変量法）を使いこなせる
- [ ] Kernel Density Estimationを実装し、Radon-Nikodym導関数として理解できる
- [ ] Metropolis-Hastings法でMCMCサンプリングを実装できる
- [ ] Brown運動の5つの性質をコードで検証できる
- [ ] Euler-Maruyama法でSDEを数値的に解ける
- [ ] Ornstein-Uhlenbeck過程を実装し、定常分布への収束を確認できる
- [ ] Langevin dynamicsでスコア関数を用いたサンプリングができる
- [ ] Fokker-Planck方程式を理解し、SDEと密度時間発展の関係を説明できる

---

> **Note:** Part1（理論編）と合わせて読むことを推奨。特に §4.5 Radon-Nikodym, §4.8 Markov連鎖, §4.10 伊藤積分は本Part2で直接実装する内容と1:1対応している。

## 💻 Z5. 実装ゾーン（45分）— 測度論を Python に翻訳する

> **Zone 5 目標**: 測度論の抽象概念を具体的なコードに落とし込む。Monte Carlo積分、KDE、Markov連鎖シミュレーション、Brown運動パス生成を実装する。

### 5.1 Monte Carlo 積分 — Lebesgue積分の近似

理論では `$\int f \, d\mu$` と書くが、実務ではMonte Carlo法で近似する。大数の法則が収束を保証する。

```math
\int f(x) \, p(x) \, dx \approx \frac{1}{N} \sum_{i=1}^{N} f(X_i), \quad X_i \sim p
```

**記号対応**:

| 数式 | コード変数 | shape |
|:-----|:----------|:------|
| `$f(X_i)$` | `f(x)` | `(N,)` |
| `$\hat{I}_N = \frac{1}{N}\sum_i f(X_i)$` | `np.mean(f(x))` | scalar |
| `$\text{Var}[\hat{I}_N] = \sigma^2/N$` | `f(x).var()/n` | scalar |
| `$X_i \sim p$` | `sampler(n)` | `(N,)` |

**収束速度**: 標準誤差 `$\text{SE} = \sigma/\sqrt{N}$`。`$N$` を10倍にするとSEは `$\sqrt{10}$` 倍減少。次元に依存しない（次元の呪いからの解放）。


```python
import numpy as np

def monte_carlo_integrate(f, sampler, n_samples: int, n_trials: int = 20):
    """Monte Carlo integration.

    E[f(X)] ≈ (1/N) Σ f(X_i)
    Variance: Var[estimate] = Var[f(X)] / N
    """
    est = []
    for _ in range(n_trials):
        x = sampler(n_samples)
        est.append(float(np.mean(f(x))))
    est = np.array(est, dtype=np.float64)
    return float(est.mean()), float(est.std(ddof=1))

# E[X^2] where X ~ N(0,1) should be 1
f = lambda x: x * x
rng = np.random.default_rng(42)
sampler = lambda n: rng.standard_normal(n)

for n in [100, 1_000, 10_000]:
    mean, std = monte_carlo_integrate(f, sampler, n)
    print(f"N={n:>6d}  mean={mean:.4f}  std={std:.4f}")
```

> **観察**: `$N$` が10倍になるとStdが `$\sqrt{10} \approx 3.16$` 倍小さくなる — Monte Carloの `$O(1/\sqrt{N})$` 収束レート。

**収束速度の実証的検証**:

```python
import numpy as np

rng = np.random.default_rng(0)
print("N         mean    std     SE_theory")
print("-" * 45)
for N in [100, 1_000, 10_000, 100_000, 1_000_000]:
    x = rng.standard_normal((50, N))  # 50 trials
    f_vals = x**2
    estimates = f_vals.mean(axis=1)  # shape (50,)
    se_empiric = estimates.std(ddof=1)
    se_theory = 1.0 / np.sqrt(N)  # sigma^2=Var[X^2]=2 for X~N(0,1), /sqrt(N)
    # exact: Var[X^2] = E[X^4] - (E[X^2])^2 = 3 - 1 = 2
    se_theory_exact = np.sqrt(2) / np.sqrt(N)
    print(f"N={N:>8d}: mean={estimates.mean():.4f}  SE_emp={se_empiric:.5f}  SE_th={se_theory_exact:.5f}")
```

`$\text{SE} = \sqrt{\text{Var}[f(X)]/N}$`。`$f(X) = X^2$` で `$X \sim \mathcal{N}(0,1)$` のとき `$\text{Var}[X^2] = E[X^4] - (E[X^2])^2 = 3 - 1 = 2$`（4次モーメントを使う）。

**収束証明（中心極限定理）**:

`$Y_i = f(X_i)$` とすると `$\hat{I}_N = \bar{Y}_N = (1/N)\sum_i Y_i$`。CLTより:

```math
\sqrt{N}(\hat{I}_N - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2), \quad \sigma^2 = \text{Var}[f(X)]
```

標準誤差 `$\text{SE} = \sigma/\sqrt{N}$`。精度 `$\epsilon$` を達成するのに必要なサンプル数: `$N = \sigma^2/\epsilon^2$`。**次元数に非依存** — これがMonte Carloが高次元積分に使われる理由。数値積分法（Simpson則など）は `$O(N^{-k/d})$`（`$d$`：次元数）で次元の呪いを受ける。

**分散の推定**:

```math
\hat{\sigma}^2 = \frac{1}{N-1}\sum_{i=1}^N (f(X_i) - \hat{I}_N)^2
```

区間推定（95%信頼区間）: `$\hat{I}_N \pm 1.96 \hat{\sigma}/\sqrt{N}$`。Lebesgue積分の近似として: 測度 `$p d\lambda$` のモンテカルロ近似は測度を経験測度 `$\hat{P}_N = (1/N)\sum_i \delta_{X_i}$` で置換することと等価。

### 5.2 `%timeit` デビュー — パフォーマンス計測

第5回から `%timeit` を使い始める。計算コストの感覚を養おう。


```python
import numpy as np, time
N = 1_000_000
rng = np.random.default_rng(0)
x = rng.standard_normal(N)
def sum_loop(arr):
    s = 0.0
    for v in arr: s += v * v
    return s / len(arr)
def sum_vec(arr): return (arr * arr).mean()
t0=time.perf_counter(); r1=sum_loop(x); t1=time.perf_counter()
t2=time.perf_counter(); r2=sum_vec(x);  t3=time.perf_counter()
print(f"loop={1000*(t1-t0):.1f}ms  vec={1000*(t3-t2):.1f}ms  result={r2:.4f}")
print(f"speedup={(t1-t0)/(t3-t2):.0f}x")  # E[X^2]=1
```


> **教訓**: ベクトル化は通常 **50-100倍** 高速。測度論の理論ではsummation orderは無関係だが、実装では**メモリアクセスパターン**が支配的。

### 5.2.1 分散低減法 — Monte Carloを賢くする

Monte Carloの `$O(1/\sqrt{N})$` 収束は変えられないが、**分散の定数因子**を減らせる。



**層化サンプリング**: 積分域を K 層に分割し各層から均等サンプル。

```math
\hat{I}_{\text{strat}} = \sum_{k=1}^K \frac{1}{K} \cdot \frac{K}{N} \sum_{i \in \text{layer}\,k} f(X_i)
```

```python
import numpy as np

def stratified_mc(f, lo, hi, n_total=100_000, n_strata=100):
    n_each = n_total // n_strata
    total = 0.0
    for k in range(n_strata):
        a = lo + k*(hi-lo)/n_strata
        b = a + (hi-lo)/n_strata
        total += f(np.random.uniform(a, b, n_each)).mean() * (hi-lo)/n_strata
    return total

f = lambda x: np.exp(-x**2)
crude = f(np.random.uniform(0, 1, 100_000)).mean()
strat = stratified_mc(f, 0, 1)
print(f"crude={crude:.5f}  stratified={strat:.5f}  exact=0.74682")
```


### 5.3 重点サンプリング (Importance Sampling) — 測度の変換

Radon-Nikodym導関数の実用版。`$p$` からサンプリングが難しい場合、別の分布 `$q$` を使う:

```math
\mathbb{E}_p[f(X)] = \mathbb{E}_q\left[f(X) \frac{p(X)}{q(X)}\right] = \mathbb{E}_q\left[f(X) \frac{dP}{dQ}(X)\right]
```

`$\frac{p(x)}{q(x)}$` がまさに **Radon-Nikodym導関数** `$\frac{dP}{dQ}(x)$` である。



**記号対応**:

| 数式 | コード変数 | 意味 |
|:-----|:----------|:-----|
| `$w(x) = p(x)/q(x)$` | `np.exp(p_logpdf - q_logpdf)` | Radon-Nikodym導関数 |
| `$\tilde{w}(x) = w(x)/\sum w$` | `w / w.sum()` | 正規化重み |
| `$\hat{I}_{\text{IS}} = \sum_i \tilde{w}_i f(X_i)$` | `w @ f(x)` | IS推定量 |
| ESS | `1 / sum(w_tilde^2)` | 有効サンプルサイズ |


```python
import numpy as np
from scipy.stats import norm

def importance_sampling(f, p_logpdf, q_sampler, q_logpdf, n=50_000):
    x = q_sampler(n)
    log_w = p_logpdf(x) - q_logpdf(x)
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    est = float(w @ f(x))
    ess_pct = 1.0 / float((w**2).sum()) / n * 100
    return est, ess_pct

est, ess = importance_sampling(
    f=lambda x: x,
    p_logpdf=lambda x: norm.logpdf(x, 5, 1),
    q_sampler=lambda n: norm.rvs(0, 3, size=n),
    q_logpdf=lambda x: norm.logpdf(x, 0, 3))
print(f"IS={est:.4f}  (true=5.0)  ESS={ess:.1f}%")
```

> **Note:** `$w(x) = p(x)/q(x)$` が Radon-Nikodym 導関数 `$dP/dQ(x)$` そのもの。ESS < 10% なら提案分布 `$q$` が `$p$` のサポートをカバーできていない。

**Self-Normalized IS (SNIS)**: 正規化定数 `$Z = \int p^*(x)dx$` が未知の場合:

```math
\hat{I}_{\text{SNIS}} = \frac{\sum_i w_i f(X_i)}{\sum_j w_j}, \quad w_i = \frac{p^*(X_i)}{q(X_i)}
```

SNISはバイアスを持つが（`$\mathbb{E}[\hat{I}_{\text{SNIS}}] \neq \mu$`）、`$N \to \infty$` で一致推定量になる。VAEのELBOをモンテカルロ推定するとき、このSNISがIMPORTANCE WEIGHTED AE (IWAE) の基礎になる。

**SNISとKLダイバージェンスの関係**:

```math
D_{\mathrm{KL}}(q \| p) = \mathbb{E}_q\left[\log\frac{q(X)}{p(X)}\right] = -\mathbb{E}_q[\log w(X)] + \text{const}
```

Importance weight の対数平均がKLダイバージェンスと直結。VAEの変分下界 ELBO = `$-D_{\mathrm{KL}}(q \| p) + \mathbb{E}_q[\log p(x|z)]$` はこの構造から来ている。

**IWAE（Importance Weighted Autoencoder）**:

```math
\mathcal{L}_K^{\text{IWAE}} = \mathbb{E}_{z_1, \ldots, z_K \sim q_\phi(z|x)}\left[\log \frac{1}{K}\sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}\right]
```

これはK個のSNIS推定量の対数。`$K=1$` でELBO、`$K \to \infty$` で `$\log p(x)$`（真の対数尤度）に収束。測度論的には: K個のサンプルから測度 `$p(z|x)$` を推定し、その正規化定数 `$\log p(x) = \log \int p(x,z)dz$` を近似している。


### 5.4 カーネル密度推定 (KDE) — Radon-Nikodym導関数の推定

データから確率密度関数（= Lebesgue測度に関するRadon-Nikodym導関数）を推定する。

```math
\hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - X_i}{h}\right)
```

バンド幅 `$h$` は「測度の解像度」を決める。



**記号対応**:

| 数式 | コード変数 | 意味 |
|:-----|:----------|:-----|
| `$h$` | `h` | バンド幅（解像度） |
| `$K(u) = \phi(u)$` | `exp(-0.5*d^2)/sqrt(2pi)` | ガウシアンカーネル |
| `$\hat{f}_h(x) = \frac{1}{nh}\sum K((x-X_i)/h)$` | `kernels.mean(axis=1)/h` | KDE推定値 |
| `$h_{\text{Silverman}} = 1.06\hat{\sigma}n^{-1/5}$` | `1.06*std*n**(-0.2)` | 最適バンド幅 |


```python
import numpy as np

def gaussian_kde(data, h=None):
    n = len(data)
    h = h or 1.06 * data.std(ddof=1) * n**(-0.2)
    def estimate(x_eval):
        d = (x_eval[:, None] - data[None, :]) / h
        return (np.exp(-0.5*d**2) / np.sqrt(2*np.pi)).mean(axis=1) / h
    return estimate, h

rng = np.random.default_rng(42)
n = 500
data = np.where(rng.random(n)<0.7, rng.standard_normal(n), rng.normal(4, 0.5, n))
kde_fn, h = gaussian_kde(data)
x_eval = np.linspace(-4, 7, 200)
density = kde_fn(x_eval)
dx = x_eval[1] - x_eval[0]
print(f"h={h:.3f}  integral={float(density.sum()*dx):.4f}  (should=1.0)")
```


### 5.5 Markov連鎖シミュレーション — 定常分布への収束

定常分布 `$\boldsymbol{\pi}$` への収束を可視化する。



```python
import numpy as np

P = np.array([[0.7, 0.2, 0.1],[0.3, 0.4, 0.3],[0.1, 0.3, 0.6]])
vals, vecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(vals - 1)); pi = np.abs(vecs[:, idx]); pi /= pi.sum()
print(f"exact   pi = {pi}")
print(f"P^100 row0 = {np.linalg.matrix_power(P,100)[0]}")

def simulate_markov(P, n_steps=100_000, x0=0):
    n = len(P); x = x0; hist = np.zeros(n, int)
    for _ in range(n_steps):
        x = np.random.choice(n, p=P[x]); hist[x] += 1
    return hist / n_steps
print(f"empiric pi = {simulate_markov(P)}")
```


### 5.6 Metropolis-Hastings — MCMC の基礎

詳細釣り合い条件を使って、任意の目標分布からサンプリングする。

```math
\alpha(x, x') = \min\left(1, \frac{\pi(x') q(x \mid x')}{\pi(x) q(x' \mid x)}\right)
```

```python
import numpy as np

def metropolis_hastings(log_target, proposal_std, x0, n_samples, burnin=1000):
    """Metropolis-Hastings MCMC sampler.

    Detailed balance: π(x) P(x→x') = π(x') P(x'→x)
    Acceptance: α = min(1, π(x')q(x|x') / π(x)q(x'|x))
    For symmetric proposal: α = min(1, π(x')/π(x))
    """
    x = x0
    samples = []
    accepted = 0

    for i in range(n_samples + burnin):
        # Symmetric proposal: q(x'|x) = N(x, σ²)
        x_proposed = x + proposal_std * np.random.randn()

        # Log acceptance ratio (symmetric → simplifies)
        log_alpha = log_target(x_proposed) - log_target(x)

        if np.log(np.random.rand()) < log_alpha:
            x = x_proposed
            if i >= burnin:
                accepted += 1

        if i >= burnin:
            samples.append(x)

    acceptance_rate = accepted / n_samples
    return np.array(samples), acceptance_rate

# Target: mixture of Gaussians (unnormalized)
def log_target_mixture(x):
    """Log of unnormalized mixture density."""
    return np.logaddexp(
        -0.5 * (x + 2)**2 / 0.5**2,
        -0.5 * (x - 3)**2 / 1.0**2
    )

np.random.seed(42)
samples, rate = metropolis_hastings(log_target_mixture, proposal_std=1.0, x0=0.0, n_samples=20_000)
print(f"accept%={rate*100:.1f}%  mean={samples.mean():.3f}  std={samples.std():.3f}")
```

`$\pi$` の正規化定数を知らなくてもサンプリングできる — これがベイズ推論で重要。

**詳細釣り合い条件の確認**:

```math
\pi(x) \cdot \alpha(x, x') \cdot q(x' | x) = \pi(x') \cdot \alpha(x', x) \cdot q(x | x')
```

これが成立するのは定義から: `$\alpha(x, x') = \min(1, \pi(x')q(x|x')/\pi(x)q(x'|x))$` と設定したから。対称提案 `$q(x'|x) = q(x|x')$` のとき `$\alpha(x, x') = \min(1, \pi(x')/\pi(x))$` に簡略化される。

**最適受理率**: Roberts et al. (1997) [^5] は高次元ガウス目標分布に対して最適受理率 `$\approx 23.4\%$` を示した。提案分布の幅を受理率が 20-25% になるよう調整するのが実践的なヒューリスティクス。

**MALA (Metropolis-Adjusted Langevin Algorithm)**: Langevin Dynamicsにメトロポリス補正を加え、バイアスを除いたもの。ULAに比べてステップ数を大幅削減できる:

```math
x' = x + \frac{\epsilon}{2} \nabla \log \pi(x) + \sqrt{\epsilon} Z, \quad \text{then accept/reject with } \alpha(x, x')
```

MALAはULAより効率的（`$d$ 次元での最適スケーリングが `$\epsilon = O(d^{-1/3})$` vs ULAの `$O(d^{-1})$`）。拡散モデルのサンプリングアルゴリズムの設計に直接影響する。

**Gibbs Samplerとの比較**: Gibbs Samplerは高次元の場合に全変数を一度にサンプルせず、各変数 `$x_i$` を他の変数を固定して条件付き `$p(x_i | \mathbf{x}_{-i})$` からサンプルする:

```math
x_i^{(t+1)} \sim p(x_i \mid x_1^{(t+1)}, \ldots, x_{i-1}^{(t+1)}, x_{i+1}^{(t)}, \ldots, x_d^{(t)})
```

**特性比較**:

| アルゴリズム | 受理判定 | 必要情報 | 高次元 | 相関変数 |
|-------------|---------|---------|--------|----------|
| MH (球形提案) | あり | `$\log \pi$` | △ ステップ小さく | △ |
| MALA | あり | `$\nabla \log \pi$` | ○ `$O(d^{-1/3})$` | ○ |
| HMC/NUTS | あり | `$\nabla \log \pi$` | ◎ `$O(d^{-1/4})$` | ◎ |
| Gibbs | なし（常に受理） | 条件付き密度 | ○（独立成分） | ✕ |
| ULA | なし（バイアスあり） | `$\nabla \log \pi$` | ○ | ○ |

Gibbs Samplerの**詳細釣り合い証明**: 条件付き分布からサンプルするので、一つの成分を更新するステップの詳細釣り合いは自明に成立（`$\pi(x_i|\mathbf{x}_{-i})$` から直接サンプルするから）。全成分を一周すると（Systematic Gibbs）定常分布 `$\pi$` に収束。

**拡散モデルとの接続**: DDPM のデノイジング `$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$` は、時系列を条件付き確率の積に分解するGibbs的構造だ。ただし各ステップで独立にサンプルするため Gibbs Sampler とは異なり、Score SDE の逆過程と同値。

### 5.7 Brown運動パス生成 — 離散近似

`$W(t_{k+1}) = W(t_k) + \sqrt{\Delta t} \cdot Z_k, \quad Z_k \sim \mathcal{N}(0,1)$`



```python
import numpy as np

T, n_steps, n_paths = 1.0, 1000, 200
rng = np.random.default_rng(42)
dt = T / n_steps
dW = rng.standard_normal((n_steps, n_paths)) * np.sqrt(dt)
W = np.vstack([np.zeros(n_paths), np.cumsum(dW, axis=0)])
qv = (dW**2).sum(axis=0)  # quadratic variation: should -> T
print(f"W(T): mean={W[-1].mean():.3f}  std={W[-1].std():.3f}  (theory: 0, 1)")
print(f"[W]_T: mean={qv.mean():.4f}  std={qv.std():.4f}  (theory: 1.0, 0)")
```


### 5.8 幾何Brown運動 (GBM) — Itôの公式の実践

株価モデルの古典:

```math
dS = \mu S \, dt + \sigma S \, dW
```

Itôの公式により解析解が得られる:

```math
S(t) = S(0) \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W(t)\right)
```

`$-\frac{\sigma^2}{2}$` の **Itô補正項** に注意 — これが伊藤積分の非直感的な部分。


```python
import numpy as np

S0, mu, sigma, T, n_steps = 100.0, 0.05, 0.20, 1.0, 252
n_paths = 5000
rng = np.random.default_rng(0)
dt = T / n_steps
dW = rng.standard_normal((n_steps, n_paths)) * np.sqrt(dt)
log_S = np.log(S0) + ((mu - 0.5*sigma**2)*dt + sigma*dW).sum(axis=0)
S_T = np.exp(log_S)
print(f"E[S(T)] empiric={S_T.mean():.2f}  analytic={S0*np.exp(mu*T):.2f}")
log_ret = np.log(S_T/S0)
print(f"log-ret mean={log_ret.mean():.4f}  std={log_ret.std():.4f}")
print(f"  theory  mean={(mu-0.5*sigma**2):.4f}  std={sigma:.4f}")
```

**Itô補正の必要性**:

素朴な `$d(\log S) = dS/S$` の計算では Itô 補正 `$-\sigma^2/2$` が出ない。コードで確認すると `E[S(T)] = S0 * exp(mu*T)` が成立する（正しい）が、`$\sigma^2/2$` を落とすと `E[S(T)]` が誤った値になる。



素朴な `$d(\log S) = dS/S = \mu dt + \sigma dW$` と積分すると `$S(T) = S_0 \exp(\mu T + \sigma W_T)$` となり、`$\mathbb{E}[S(T)] = S_0 e^{\mu T} e^{\sigma^2 T/2} \neq S_0 e^{\mu T}$`。Itô補正 `$-\sigma^2/2$` は `$\mathbb{E}[S(T)] = S_0 e^{\mu T}$`（リスクニュートラル評価）を保証するために必要。この補正なしに金融派生商品のプライシングは成り立たない。

**対数正規性の検証**: `$\log(S_T/S_0) \sim \mathcal{N}((\mu-\sigma^2/2)T, \sigma^2 T)$` が成立することをコードで確認した。平均 `$\mu T$` でなく `$(\mu-\sigma^2/2)T$` になるのがItô積分の非直感的な核心だ。

### 5.9 Ornstein-Uhlenbeck過程 — DDPMの連続極限

Diffusion modelの連続極限はOrnstein-Uhlenbeck (OU) 過程:

```math
dX_t = -\theta X_t \, dt + \sigma \, dW_t
```

平均回帰性（mean-reverting）を持ち、定常分布は `$\mathcal{N}(0, \sigma^2/(2\theta))$`。



```python
import numpy as np

theta, sigma, x0, T, n_steps, n_paths = 1.0, 1.0, 5.0, 10.0, 10_000, 2000
rng = np.random.default_rng(42)
dt = T / n_steps
X = np.full(n_paths, x0, dtype=float)
for k in range(n_steps):
    X += -theta * X * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
stat_var = sigma**2 / (2*theta)  # N(0, sigma^2/(2*theta))
print(f"final: mean={X.mean():.3f}  var={X.var():.3f}  (stat.var={stat_var:.3f})")
```


### 5.10 Langevin Dynamics — Score関数でサンプリング

Score function `$\nabla_x \log p(x)$` を使って目標分布からサンプリングするLangevin Monte Carlo法:

```math
X_{k+1} = X_k + \frac{\epsilon}{2} \nabla_x \log p(X_k) + \sqrt{\epsilon} \, Z_k, \quad Z_k \sim \mathcal{N}(0, I)
```

`$\epsilon \to 0$`、`$K \to \infty$` で `$X_K \sim p$` に収束する[^2]。



```python
import numpy as np
from scipy.stats import norm

def ula(score_fn, x0=0.0, eps=0.005, n=100_000, burnin=10_000, seed=42):
    rng = np.random.default_rng(seed)
    x = float(x0); samples = []
    for i in range(n + burnin):
        x += 0.5*eps*score_fn(x) + np.sqrt(eps)*rng.standard_normal()
        if i >= burnin: samples.append(x)
    return np.array(samples)

def log_p(x):
    return float(np.logaddexp(norm.logpdf(x, -2, 0.5), norm.logpdf(x, 3, 1.0)))
def score(x, h=1e-4): return (log_p(x+h)-log_p(x-h))/(2*h)

s = ula(score)
print(f"mean={s.mean():.3f}  std={s.std():.3f}")
```

**Fokker-Planck接続**: Langevin SDE `$dX = \nabla\log p(X)dt + \sqrt{2}dW$` のFP定常解は `$q_\infty = p$`。


### 5.11 Euler-Maruyama法 — SDEの数値解法

SDEの厳密解が得られるケース（GBM、OU過程）は少数派だ。一般のSDEでは**数値解法**が必要になる。最も基本的な手法がEuler-Maruyama法 — ODE のEuler法をSDEに拡張したもの。

#### 離散化スキーム

SDE `$dX_t = f(X_t) \, dt + g(X_t) \, dW_t$` を時間幅 `$\Delta t$` で離散化する:

```math
X_{n+1} = X_n + f(X_n) \Delta t + g(X_n) \sqrt{\Delta t} \, Z_n, \quad Z_n \sim \mathcal{N}(0, 1)
```

`$\sqrt{\Delta t} \, Z_n$` が Brown運動増分 `$\Delta W_n = W_{t_{n+1}} - W_{t_n} \sim \mathcal{N}(0, \Delta t)$` に対応。

これは Euler 法の確率版だ。

**記号対応**:

| 数式 | コード変数 | shape |
|:-----|:----------|:------|
| `$f(X_n)$` | `f(X)` | `(n_paths,)` |
| `$g(X_n)$` | `g(X)` | `(n_paths,)` |
| `$\Delta t$` | `dt` | scalar |
| `$Z_n \sim \mathcal{N}(0,1)$` | `rng.standard_normal(n_paths)` | `(n_paths,)` |

```python
import numpy as np

def euler_maruyama(f, g, x0, T=1.0, n_steps=1000, n_paths=2000, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    X = np.full(n_paths, x0, dtype=float)
    for _ in range(n_steps):
        Z = rng.standard_normal(n_paths)
        X = X + f(X)*dt + g(X)*sqrt_dt*Z
    return X

# OU: dX = -X dt + dW -> stationary N(0, (1-e^{-2T})/2)
X_T = euler_maruyama(f=lambda x: -x, g=lambda x: np.ones_like(x), x0=5.0)
stat_var = (1 - np.exp(-2.0)) / 2
print(f"X(T) mean={X_T.mean():.3f}  var={X_T.var():.3f}  (stat.var={stat_var:.3f})")
```

#### 強収束と弱収束

| 収束の種類 | 定義 | Euler-Maruyama | 意味 |
|:---------|:----|:-------------|:-----|
| 強収束 | `$\mathbb{E}[\|X_N - X(T)\|] \leq C \Delta t^{1/2}$` | `$O(\sqrt{\Delta t})$` | パスが近い |
| 弱収束 | `$\|\mathbb{E}[h(X_N)] - \mathbb{E}[h(X(T))]\| \leq C \Delta t$` | `$O(\Delta t)$` | 統計量が近い |

- **強収束**: 個々のパスが真の解に近い（シミュレーション・可視化に重要）
- **弱収束**: 期待値や分布の性質が正しい（統計量の推定に十分）

拡散モデルでは多くの場合、**弱収束で十分**（生成画像の分布が正しければよい）。DDPM の離散ステップ数 `$T = 1000$` は弱収束の精度を確保するため。

**Milstein法（1次精度）**: Euler-Maruyamaを改善した高精度スキーム:

```math
X_{n+1} = X_n + f(X_n)\Delta t + g(X_n)\Delta W_n + \frac{1}{2}g(X_n)g'(X_n)[(\Delta W_n)^2 - \Delta t]
```

追加項 `$\frac{1}{2}g g'[(\Delta W)^2 - \Delta t]$` がItô補正から来ている（`$(dW)^2 = dt$` の次の項）。強収束が `$O(\Delta t)$` に改善（Euler-Maruyamaの `$O(\sqrt{\Delta t})$` から）。

`$g$` が定数（OU過程、DDPM）の場合: `$g' = 0$` なのでMilstein = Euler-Maruyama。つまりDDPMでは両者が等価で、Euler-Maruyamaで十分。



### 5.12 収束定理の数値検証 — MCT vs DCT vs Fatou

3つの収束定理を同時に検証する。

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 200_000)
print("MCT (-> 50):")
for n in [1, 2, 5, 10]:
    print(f"  n={n}: {(x*(x<=n)).mean()*10:.2f}")

x2 = rng.uniform(0, 20, 200_000)
print("DCT (-> 1.0):")
for n in [2, 10, 100]:
    gn = (1+x2/n)**(-n)
    print(f"  n={n}: {gn.mean()*20:.4f}")

x3 = rng.uniform(0, 5, 200_000)
print("No domination (stays ~0.5):")
for n in [1, 5, 50]:
    hn = n * x3 * np.exp(-n * x3**2)
    print(f"  n={n}: {hn.mean()*5:.4f}")
```


### Quick Check — Z5

<details><summary>Q1: Importance Samplingでw(x)=p(x)/q(x)が「Radon-Nikodym導関数」になる理由を説明せよ。</summary>

**A**: Radon-Nikodym定理は「`$P \ll Q$` のとき `$P(A) = \int_A \frac{dP}{dQ} dQ$` を満たす可測関数が一意存在する」と言う。Importance weightingの等式:

```math
\mathbb{E}_P[f] = \int f \, dP = \int f \frac{dP}{dQ} dQ = \mathbb{E}_Q\left[f \cdot \frac{p}{q}\right]
```

の `$p(x)/q(x)$` がまさに `$dP/dQ(x)$`。`$p \ll q$`（サポートの包含）が Radon-Nikodym の前提条件に対応し、これが崩れると ESS が 0 に近づく。

</details>

<details><summary>Q2: Brown運動の二次変動 [W]_T = T を数値的に検証するコードの意図を説明せよ。</summary>

**A**: 二次変動の定義は `$[W]_T = \lim_{\|P\| \to 0} \sum_k (W_{t_{k+1}} - W_{t_k})^2$`。コード中の `(dW**2).sum(axis=0)` はこの和の離散近似。`$\Delta t \to 0$` のとき `$\sum (\Delta W)^2 \to T$`（確率収束）。これが `$(dW)^2 = dt$` という伊藤の補題の2次項の起源であり、通常の微積分では消える `$dx^2 = 0$` との本質的違い。

</details>

<details>
<summary>Quick Check 答え合わせ</summary>

以下を確認してみましょう:

1. Monte Carlo積分の収束レートは `$O(1/\sqrt{N})$` — サンプル数を100倍にすると誤差は10倍小さくなる
2. 重点サンプリングでESS < 10%の場合、推定結果は信頼できない
3. KDEのバンド幅 `$h$` は「測度の解像度」を決める — 小さすぎるとノイジー、大きすぎるとぼやける
4. Metropolis-Hastingsの受理率は23%前後が最適（多次元ガウス目標の場合）
5. Brown運動の二次変動 `$[W]_T = T$` — これがItô補正の源泉
6. Euler-Maruyama法は強収束 `$O(\sqrt{\Delta t})$`、弱収束 `$O(\Delta t)$`

</details>

<details><summary>Q3: Euler-Maruyama法でΔtを半分にすると誤差はどう変わるか？強収束と弱収束で答えよ。</summary>

**A**:
- **強収束** (`$\mathbb{E}[|X_T^{\Delta t} - X_T|^2]^{1/2}$`): `$O(\sqrt{\Delta t})$`。`$\Delta t$` を半分にすると誤差は `$1/\sqrt{2} \approx 0.707$` 倍。
- **弱収束** (`$|\mathbb{E}[f(X_T^{\Delta t})] - \mathbb{E}[f(X_T)]|$`): `$O(\Delta t)$`。`$\Delta t$` を半分にすると誤差は `$1/2$` 倍。

生成モデルでは弱収束（分布の近似）で十分なため、DDPMの `$T=1000$` は弱収束精度 `$O(1/T) = O(10^{-3})$` を狙っている。強収束は各サンプルパスの精度に関係し、オプション価格計算のような用途で重要。
</details>

<details><summary>Q4: KDEのバンド幅 h を小さくしすぎるとどうなるか？測度論的に説明せよ。</summary>

**A**: KDE は `$\hat{p}_h(x) = \frac{1}{Nh}\sum_{i=1}^N K\left(\frac{x-X_i}{h}\right)$` で定義される。`$h \to 0$` のとき、各カーネル `$K(\cdot/h)/h$` はデータ点 `$X_i$` に集中する Dirac delta `$\delta_{X_i}$` に収束（分布収束の意味で）。つまり `$\hat{p}_h \to \frac{1}{N}\sum_i \delta_{X_i}$`（経験測度）になり、連続密度が推定できなくなる。`$h$` は「Lebesgue測度に対する経験測度の平滑化パラメータ」で、Silvermanルール `$h = 1.06\hat{\sigma}N^{-1/5}$` はMISE（平均積分二乗誤差）最小化の漸近最適解。
</details>



### 5.13 数式→コード翻訳パターン集

| 数式 | Python | 注意点 |
|:--|:--|:--|
| `$\int f \, d\mu$` | `np.mean(f(samples))` | Monte Carlo近似 |
| `$\frac{dP}{dQ}(x)$` | `p.pdf(x) / q.pdf(x)` | Importance weight |
| `$\hat{f}_h(x)$` | `kde_estimate(data, x, h)` | バンド幅選択が重要 |
| `$P^n$` | `np.linalg.matrix_power(P, n)` | 定常分布へ収束 |
| `$W(t)$` | `np.cumsum(np.sqrt(dt)*Z)` | `$Z \sim \mathcal{N}(0,1)$` |
| `$\sum (\Delta W)^2$` | `np.sum(np.diff(W)**2)` | `$\to T$`（二次変動） |
| `$dX = a \, dt + b \, dW$` | `X[i+1] = X[i] + a*dt + b*dW` | Euler-Maruyama |
| `$e^{-\theta t}$` | `np.exp(-theta*t)` | OU過程の平均回帰 |
| `$\frac{1}{nh}\sum K(\cdot)$` | `np.mean(kernel) / h` | KDE |
| `$\boldsymbol{\pi} P = \boldsymbol{\pi}$` | `eig(P.T)` で固有値1の固有ベクトル | 左固有ベクトル |

### 5.14 Monte Carlo 信頼区間の構成

**中心極限定理による区間推定**: 推定量 `$\hat{\mu}_N = \frac{1}{N}\sum_{i=1}^N f(X_i)$` に対する 95% 信頼区間:

```math
\hat{\mu}_N \pm z_{0.025} \cdot \frac{\hat{\sigma}}{\sqrt{N}}, \quad \hat{\sigma}^2 = \frac{1}{N-1}\sum_{i=1}^N (f(X_i) - \hat{\mu}_N)^2
```

- `$\hat{\mu}_N$`: 標本平均（MC推定値） — コードの `f_vals.mean()`
- `$z_{0.025} = 1.96$`: 標準正規分布の97.5%点
- `$\hat{\sigma}^2$`: 標本分散（不偏推定量、`ddof=1`）
- `$\hat{\sigma}/\sqrt{N}$`: 標準誤差（SE）— `f_vals.std(ddof=1) / np.sqrt(N)`

```python
import numpy as np

rng = np.random.default_rng(0)
N = 10_000
X = rng.standard_normal(N)
# f(X) = exp(-X^2/2)/sqrt(2pi) を N(0,1) で積分 → integral phi^2 dx = 1/(2*sqrt(pi))
f_vals = np.exp(-X**2 / 2) / np.sqrt(2 * np.pi)
true_val = 1.0 / (2 * np.sqrt(np.pi))  # = 0.28209...

mean_est = f_vals.mean()            # mu_hat
se = f_vals.std(ddof=1) / np.sqrt(N)  # sigma_hat / sqrt(N)
ci_lo = mean_est - 1.96 * se
ci_hi = mean_est + 1.96 * se
print(f"Estimate : {mean_est:.5f}")
print(f"95% CI   : [{ci_lo:.5f}, {ci_hi:.5f}]")
print(f"True val : {true_val:.5f}  in CI: {ci_lo <= true_val <= ci_hi}")
# → 95回/100試行でCIが真値を含む
```

> **⚠️ Warning:** `$f(X)^2$` が `$q$` に関して可積分（`$\mathbb{E}_q[f^2] < \infty$`）でないとCLTが適用不可。例えば重要度サンプリングで `$p/q$` が裾で爆発する場合。

---

> Progress: 85%

---

## 🔬 Z6. 研究フロンティア（20分）— 測度論の最前線

> **Zone 6 目標**: 本講義で学んだ測度論・確率過程を基盤とする最新研究を俯瞰する。

### 6.1 Score SDE の理論的完成 — Song et al. 2020

Score SDE [^2] はDDPMをVP-SDE（Variance Preserving SDE）として定式化した金字塔だ。

```math
d\mathbf{x} = -\frac{\beta(t)}{2} \mathbf{x} \, dt + \sqrt{\beta(t)} \, d\mathbf{W}
```

**VP-SDE の測度論的意味**: この SDE は、標本 `$\mathbf{x}_0 \sim p_0$` から始まり `$t \to \infty$` で `$\mathcal{N}(\mathbf{0}, \mathbf{I})$` に収束するOU過程。各時刻の分布 `$p_t$` は Fokker-Planck 方程式に従う。Score SDE の革新は **この連続族 `$\{p_t\}_{t \in [0,T]}$` 全体を1本のSDEで記述できる** 点にある。DDPM は離散近似でしかなかったが、Score SDE では任意の時刻 `$t$` で `$\nabla \log p_t$` が定義される。

Anderson（1982）[^9] のReverse SDE定理を使うと、逆時間過程は:

```math
d\mathbf{x} = \left[-\frac{\beta(t)}{2} \mathbf{x} - \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + \sqrt{\beta(t)} \, d\bar{\mathbf{W}}
```

Score関数 `$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$` を**ニューラルネットワーク** `$s_\theta(\mathbf{x}, t)$` で近似し、逆SDEを解くことで `$p_0$`（データ分布）からサンプリングできる。

**学習目的関数（Denoising Score Matching）**:

```math
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\lambda(t) \left\| s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t | \mathbf{x}_0) \right\|^2 \right]
```

ガウス遷移核の場合、`$\nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t | \mathbf{x}_0) = -\boldsymbol{\epsilon}/\sigma_t$`（`$\boldsymbol{\epsilon}$`はノイズ）となり、DDPMの `$\epsilon$`-predictionと等価になる。この事実はRadon-Nikodym導関数がガウス密度の対数微分に帰着することから直接導かれる。

**導出**: ガウス遷移核 `$q_t(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$` の対数:

```math
\log q_t(\mathbf{x}_t|\mathbf{x}_0) = -\frac{d}{2}\log(2\pi(1-\bar{\alpha}_t)) - \frac{\|\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0\|^2}{2(1-\bar{\alpha}_t)}
```

`$\mathbf{x}_t$` で微分:

```math
\nabla_{\mathbf{x}_t} \log q_t = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}
```

ここで `$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$`（再パラメータ化）を使った。つまり `$s_\theta \approx -\boldsymbol{\epsilon}/\sigma_t$`、`$\epsilon$`-predictionとScore関数の1:1対応が明確になった。

### 6.2 VP-SDE収束理論 — Grönwall不等式の応用

最新の理論研究 [^10] はEuler-Maruyama離散化の誤差を制御するためにGrönwall不等式を使う。

**Grönwall不等式**: 非負関数 `$u(t)$` が:

```math
u(t) \leq \alpha(t) + \int_0^t \beta(s) u(s) \, ds
```

を満たすならば:

```math
u(t) \leq \alpha(t) + \int_0^t \alpha(s) \beta(s) \exp\left(\int_s^t \beta(r) \, dr\right) ds
```

これをVP-SDEのKL divergence誤差に適用すると、ステップ幅 `$\Delta t$` に対する離散化誤差の上界:

**証明スケッチ** (by induction):

`$u_n = D_{\mathrm{KL}}(p_n \| q_n)$`（`$n$`ステップ後のKL）とすると、1ステップのKL誤差 `$\delta_n \leq C \cdot \Delta t^2$` より:

```math
u_{n+1} \leq (1 + \beta \Delta t) u_n + C \Delta t^2
```

これを繰り返し適用（`$N = T/\Delta t$` 回）:

```math
u_N \leq (1 + \beta \Delta t)^N u_0 + C \Delta t^2 \sum_{k=0}^{N-1} (1+\beta \Delta t)^k \leq e^{\beta T} \cdot C \Delta t^2 \cdot \frac{e^{\beta T}-1}{\beta \Delta t}
```

最終的に `$D_{\mathrm{KL}} \leq O(\Delta t)$`（弱収束の直接証明）。

```math
D_{\mathrm{KL}}(p_{\theta,\Delta t} \| p_{\text{data}}) \leq C \cdot \Delta t^2 \cdot \int_0^T \mathbb{E}[\|\nabla \log p_t\|^2] \, dt
```

が導出される。これは **Euler-Maruyama法の弱収束 `$O(\Delta t)$`** の理論的根拠であり、DDPMのステップ数 `$T$` を増やすほど精度が上がる理由だ。

**スコア誤差への接続**: 式の右辺 `$\int_0^T \mathbb{E}[\|\nabla \log p_t\|^2] dt$` は、Score Matchingの損失関数の積分版だ。つまり「学習されたスコア関数の精度が生成品質のボトルネック」であることが理論的に保証される。スコア誤差を `$\epsilon$` 以下にすれば、最終KLは `$O(\epsilon + \Delta t)$` — 学習誤差と離散化誤差の和。

**Grönwall不等式の一般形**（連続版）:

```math
\frac{d}{dt} u(t) \leq \beta(t) u(t) + \gamma(t) \implies u(t) \leq e^{\int_0^t \beta(s)ds} u(0) + \int_0^t e^{\int_s^t \beta(r)dr} \gamma(s) ds
```

これはSDE収束解析に限らず、ODE安定性解析・偏微分方程式の一意性証明・機械学習の一般化誤差バウンドなど幅広く使われる不等式。微分不等式の積分を指数関数で上から抑えるという、「情報量の制御」の基本技術。

### 6.3 離散拡散モデルのKL収束保証

連続拡散モデルに対して、離散状態空間（テキストのトークンなど）での拡散過程 [^11] のKL収束:

**離散拡散の測度論的基礎**: 離散状態空間 `$\mathcal{X}$` 上の確率測度はPMFで表現されるが、Chapman-Kolmogorov方程式と遷移核の積としての同時分布という構造は連続の場合と全く同じだ。重要なのは:

```math
q(x_t | x_0) = \sum_{x_1, \ldots, x_{t-1}} \prod_{s=1}^t q(x_s | x_{s-1})
```

これは `$Q_t = Q_1^t$`（遷移行列の `$t$` 乗）で表現でき、DDPM の closed-form `$q(\mathbf{x}_t | \mathbf{x}_0)$` の離散類似だ。

**VQDM, MaskDiffusion, MDLM**: テキスト向け離散拡散の最近の系譜。Maskトークンを「吸収状態」とするMarkov連鎖を使い、各トークンが独立に mask → demask される。測度論的には `$q_t(x_t | x_0) = \text{Cat}((1-\beta_t)\delta_{x_t=x_0} + \beta_t \delta_{x_t=[\text{MASK}]})$`。

**KL収束証明の測度論的核心**: [^11] の収束証明は以下の分解を使う:

```math
D_{\mathrm{KL}}(q(x_{0:T}) \| p_\theta(x_{0:T})) = \sum_{t=1}^T \mathbb{E}_{q(x_{t+1})}[D_{\mathrm{KL}}(q(x_t|x_{t+1}, x_0) \| p_\theta(x_t|x_{t+1}))]
```

このステップ毎KL分解は **Chain Ruleの測度論的版** — 結合測度のKLが条件付きKLの和に等しい:

```math
D_{\mathrm{KL}}(P(X,Y) \| Q(X,Y)) = D_{\mathrm{KL}}(P(X) \| Q(X)) + \mathbb{E}_{P(X)}[D_{\mathrm{KL}}(P(Y|X) \| Q(Y|X))]
```

これはRadon-Nikodym導関数の連鎖律 `$\frac{dP}{dQ} = \frac{dP_X}{dQ_X} \cdot \frac{dP_{Y|X}}{dQ_{Y|X}}$` の期待値を取った結果だ。



```math
D_{\mathrm{KL}}(q_t(x_t) \| p_\theta(x_t)) \leq \sum_{s=1}^{t} D_{\mathrm{KL}}(q(x_s | x_{s-1}, x_0) \| p_\theta(x_s | x_{s+1}))
```

この不等式はMarkov連鎖の測度論的構造 — 具体的には遷移核の積と条件付き期待値のタワー性質 — から直接導かれる。「離散」でも「連続」でも、測度論の言語は同一だ。

### 6.4 Flow Matching の測度論的基礎

Flow Matching [^7] は確率パス `$p_t$` を直接設計する。

**条件付き確率パス**: 各 `$\mathbf{x}_1 \sim p_1$`（データ点）に対し:

```math
p_t(\mathbf{x} | \mathbf{x}_1) = \mathcal{N}(t \mathbf{x}_1, (1 - (1-\sigma_{\min})t)^2 \mathbf{I})
```

条件付き速度場 `$u_t(\mathbf{x} | \mathbf{x}_1)$` で確率フローODEを定義:

```math
d\mathbf{x} = u_t(\mathbf{x}) \, dt, \quad u_t(\mathbf{x}) = \mathbb{E}[u_t(\mathbf{x} | \mathbf{x}_1) | \mathbf{x}_t = \mathbf{x}]
```

周辺速度場 `$u_t$` は条件付き速度場の期待値 — これは測度論的条件付き期待値の射影解釈が本質的に使われている。

**Flow Matching の損失関数**:

```math
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\| v_\theta(\mathbf{x}_t, t) - u_t(\mathbf{x}_t | \mathbf{x}_1) \|^2\right]
```

ここで `$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$`（線形補間）、条件付き速度場 `$u_t(\mathbf{x}_t | \mathbf{x}_1) = \mathbf{x}_1 - \mathbf{x}_0$`（定数！）。これを学習した `$v_\theta$` で ODE `$d\mathbf{x}/dt = v_\theta(\mathbf{x}_t, t)$` を積分すれば `$p_0 \to p_1$` の輸送が得られる。

**Rectified Flow との比較**: Rectified Flow は `$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$` の同じ構造だが、パスの「まっすぐさ」を訓練後のreflowで改善する。FLUX.1 (Black Forest, 2024) がこのアーキテクチャを採用している。

**なぜFlow MatchingはSDEより速いか**: SDEはランジュバン力学的なノイズを持つが、Flow MatchingはODE（確定論的）。サンプリング時のステップ数を10-30倍削減できる。しかし数学的基盤（確率パスの構成・収束保証）はFokker-Planck方程式と同様の測度論が必要。

**周辺速度場の測度論的正当化**: 損失関数で条件付き速度場 `$u_t(\mathbf{x}|\mathbf{x}_1)$` の期待値が周辺速度場 `$u_t(\mathbf{x})$` と一致することの証明:

```math
\mathbb{E}_{\mathbf{x}_1 | \mathbf{x}_t = \mathbf{x}}[u_t(\mathbf{x} | \mathbf{x}_1)] = u_t(\mathbf{x})
```

これはContinuity Equation:

```math
\partial_t p_t + \nabla \cdot (p_t u_t) = 0
```

の線形性から来る。条件付きバージョンを `$\mathbf{x}_1$` で積分するとき、Fubiniの定理で積分と微分を交換できる（`$p_t$` の可積分性が条件）。この「条件付き→周辺への射影」はPart1で学んだ条件付き期待値の射影性質そのものだ。

### 6.4b Stochastic Interpolants — 測度論的最終統一

Albergo & Vanden-Eijnden (2023) の Stochastic Interpolants は Flow Matching と拡散モデルを統一する框架だ。

**定義（Stochastic Interpolant）**: ソース分布 `$\rho_0$` とターゲット分布 `$\rho_1$` の間の補間:

```math
\mathbf{x}(t) = \alpha(t) \mathbf{x}_0 + \beta(t) \mathbf{x}_1 + \gamma(t) \boldsymbol{\xi}, \quad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
```

- `$\alpha(0)=1, \alpha(1)=0$`（ソースを消す）
- `$\beta(0)=0, \beta(1)=1$`（ターゲットに至る）
- `$\gamma(t) \geq 0$`（ノイズの大きさ。`$\gamma=0$` で Flow Matching、`$\gamma > 0$` で拡散的）

**統一性**: 適切な `$\alpha, \beta, \gamma$` を選ぶと:
- `$\gamma = 0$`: Flow Matching / Rectified Flow
- `$\gamma = \sqrt{t(1-t)}$`: Bridge Matching
- `$\gamma(t) = \sqrt{1-\bar{\alpha}_t}$`: DDPM / Score SDE

**測度論的視点**: `$\mathbf{x}(t)$` の各時刻の分布 `$\rho_t = \text{Law}(\mathbf{x}(t))$` がパスの族（確率カーネル）を定義する。ベクトル場 `$b_t$` は条件付き速度場の条件付き期待値として定まる — これはRadon-Nikodym定理と条件付き期待値の射影性質の直接応用だ。

**学習目的関数の導出**: 訓練する量はベクトル場 `$b_\theta(\mathbf{x}, t)$`:

```math
\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1, \boldsymbol{\xi}}\left[\| b_\theta(\mathbf{x}(t), t) - \dot{\mathbf{x}}(t) \|^2\right]
```

ここで `$\dot{\mathbf{x}}(t) = \dot{\alpha}(t)\mathbf{x}_0 + \dot{\beta}(t)\mathbf{x}_1 + \dot{\gamma}(t)\boldsymbol{\xi}$`（補間の時間微分）。`$\gamma=0$` のとき Flow Matching の損失に帰着。`$\gamma > 0$` のとき `$\boldsymbol{\xi}$ が加わりスコア関数的な成分が現れる。

**スコア関数との接続**: `$\gamma(t) > 0$` のとき、条件付き期待値の射影から:

```math
b_t(\mathbf{x}) = v_t(\mathbf{x}) - \frac{\dot{\gamma}(t)}{\gamma(t)} \cdot \sigma_t^2 \nabla_\mathbf{x} \log \rho_t(\mathbf{x})
```

第1項が速度場（Flow Matchingの寄与）、第2項がスコア関数（拡散の寄与）。`$\gamma \to 0$` でスコア項が消え純粋なFlow Matchingに、`$v_t \to 0$` で純粋なScore SDEに退化する。Stochastic Interpolantsは「Flow Matchingと拡散モデルの間を連続的に補間するパラメータ族」として理解できる。

### 6.5 研究系譜図

```mermaid
graph TD
    RN["Radon-Nikodym<br/>定理 (1913/1930)"] --> KL["KL Divergence<br/>Kullback-Leibler 1951"]
    ITO["伊藤積分<br/>Itô 1944"] --> SDE["SDE理論<br/>1950s-"]
    SDE --> REVS["Reverse SDE<br/>Anderson 1982"]
    SDE --> FP["Fokker-Planck<br/>方程式"]
    
    REVS --> DDPM["DDPM<br/>Ho+ 2020"]
    FP --> SCORE["Score SDE<br/>Song+ 2020"]
    RN --> SCORE
    SCORE -->|"ODE sampler"| FLOW["Flow Matching<br/>Lipman+ 2022"]
    SCORE -->|"直線化"| RF["Rectified Flow<br/>Liu+ 2022"]
    
    KL --> VAE["VAE<br/>Kingma+ 2013"]
    KL --> GAN["GAN<br/>Goodfellow+ 2014"]
    
    FLOW --> STABLE["Stable Diffusion 3<br/>Esser+ 2024"]
    RF --> FLUX["FLUX<br/>Black Forest 2024"]
    
    style ITO fill:#fff9c4
    style REVS fill:#e3f2fd
    style SCORE fill:#c8e6c9
    style STABLE fill:#f3e5f5
    style FLUX fill:#f3e5f5
    SI["Stochastic Interpolants<br/>Albergo+ 2023"] --> UNIFIED["統一框架"]
    FLOW --> SI
    DDPM --> SCORE
    GAN -->|"GAN死亡?"| DDPM
```

**系譜の読み方**: 縦軸は時間（上=古い）。色は: 黄=数学基礎、青=理論突破、緑=実用化、紫=応用システム。

各ノードの測度論的核心:
- **Itô積分 (1944)**: 適合過程の確率積分 — Brownian filtrationに対するmartingale
- **Reverse SDE (1982)**: Girsanov変換 + Radon-Nikodym — 時間反転の測度変換
- **Score SDE (2020)**: Fokker-Planck + スコア関数 — 密度の対数微分
- **Flow Matching (2022)**: Continuity Equation + 条件付き期待値 — 測度輸送のODE記述
- **Stochastic Interpolants (2023)**: SDEとODEの統一 — Girsanov + Pushforward

> Progress: 95%

### Z6 理解度チェック

**チェック 1**: Score SDE の逆時間過程を生成に使うには、各時刻 `$t$` のスコア `$\nabla \log p_t(\mathbf{x})$` が必要だ。しかしデータから `$p_t$` が分からない場合、どうやってスコアを近似するか？

<details><summary>ヒント: Tweedie公式</summary>

**Tweedie公式**: `$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$` のとき:

```math
\nabla \log p_t(\mathbf{x}_t) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\,\mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t]}{1 - \bar{\alpha}_t}
```

ニューラルネット `$\epsilon_\theta(\mathbf{x}_t, t)$` で `$\mathbb{E}[\epsilon | \mathbf{x}_t]$` を予測 → スコア `$\approx -\epsilon_\theta / \sqrt{1-\bar{\alpha}_t}$`。Denoising Score Matchingの本質はこれ。
</details>

**チェック 2**: Flow Matchingで `$(\mathbf{x}_0, \mathbf{x}_1)$` を独立サンプル（カップリングなし）で直線補間すると、生成品質が下がる理由を測度論的に説明せよ。

<details><summary>答え</summary>

独立カップリングでは `$p_{0 \times 1}(\mathbf{x}_0, \mathbf{x}_1) = p_0(\mathbf{x}_0) p_1(\mathbf{x}_1)$`。直線補間 `$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$` の軌跡が **交差**（同じ `$\mathbf{x}_t$` から異なる `$\mathbf{x}_1$` に向かう複数の軌跡）するため、速度場 `$u_t(\mathbf{x})$` が「平均化」され直線的でなくなる。Conditional OT カップリング（`$W_2$` 距離最小化）は交差を最小化し、「まっすぐな」軌跡を与える。
</details>


## 🚀 Z7. 振り返りゾーン（30分）— まとめと次回予告

### 7.1 Fokker-Planck方程式の直感 — SDEから確率密度のPDEへ

SDEは**個々のパス**（サンプル軌道）を記述する。だが生成モデルの本質的な問いは「確率密度 `$p(x, t)$` が時間とともにどう変化するか」だ。Fokker-Planck方程式（Kolmogorov前向き方程式）は、SDEをパスの集団（確率密度）の言葉に翻訳する。

#### SDEからFokker-Planckへの対応

SDEとFokker-Planck方程式は1対1対応する「双対言語」だ。

| SDE視点 | Fokker-Planck視点 | 意味 |
|:--------|:-----------------|:-----|
| `$X_t(\omega)$` は確率的パス | `$p(x, t)$` は確率密度 | 1粒子 vs 粒子の雲 |
| `$f(X_t) dt$` はdrift | `$-\partial_x(f \cdot p)$` は確率フラックス | 流れの源 |
| `$g(X_t) dW_t$` はdiffusion | `$\frac{1}{2}\partial_{xx}(g^2 p)$` は拡散項 | 広がりの源 |
| Itô補正 | 拡散項の出現 | 同一現象の2つの顔 |

SDE:
```math
dX_t = f(X_t) \, dt + g(X_t) \, dW_t
```

に対応するFokker-Planck方程式 (FPE):

```math
\frac{\partial p}{\partial t}(x, t) = -\frac{\partial}{\partial x}\big[f(x) \, p(x, t)\big] + \frac{1}{2}\frac{\partial^2}{\partial x^2}\big[g^2(x) \, p(x, t)\big]
```

- 第1項: `$-\partial_x(fp)$` — **ドリフト項**（確率の流れ）
- 第2項: `$\frac{1}{2}\partial_x^2(g^2 p)$` — **拡散項**（確率の広がり）

> **一言で言えば**: SDEが「1つの粒子がどう動くか」を記述するのに対し、Fokker-Planck方程式は「粒子の雲（確率密度）がどう変形するか」を記述する。

#### 導出の直感（多次元は第30回）

確率の保存則（連続の方程式）から出発する。`$J(x, t)$` を確率フラックス（確率の流れ）とすると:

```math
\frac{\partial p}{\partial t} = -\frac{\partial J}{\partial x}
```

Itôの公式から、フラックスは:

```math
J(x, t) = f(x) p(x, t) - \frac{1}{2}\frac{\partial}{\partial x}\big[g^2(x) p(x, t)\big]
```

ドリフトによる流れ `$fp$` と、拡散による広がり `$-\frac{1}{2}\partial_x(g^2 p)$` の和。これを連続の方程式に代入するとFPEが得られる。

#### OU過程の場合

`$dX_t = -\theta X_t \, dt + \sigma \, dW_t$` のFPE:

**定常解の導出**:

`$\partial_t p = 0$` とすると:

```math
0 = \theta \partial_x(xp_\infty) + \frac{\sigma^2}{2} \partial_{xx} p_\infty
```

試験解 `$p_\infty(x) \propto \exp(-\theta x^2/\sigma^2)$` を代入:

```math
\partial_x p_\infty = -\frac{2\theta x}{\sigma^2} p_\infty, \quad \partial_{xx} p_\infty = \left(-\frac{2\theta}{\sigma^2} + \frac{4\theta^2 x^2}{\sigma^4}\right) p_\infty
```

FPEに代入して確認:

```math
\theta \partial_x(x p_\infty) + \frac{\sigma^2}{2}\partial_{xx} p_\infty = \left[\theta - \frac{2\theta^2 x^2}{\sigma^2} + \frac{\sigma^2}{2}\left(-\frac{2\theta}{\sigma^2} + \frac{4\theta^2 x^2}{\sigma^4}\right)\right] p_\infty = 0 \checkmark
```

正規化: `$p_\infty(x) = \mathcal{N}(0, \sigma^2/(2\theta))$`。シミュレーションで確認した定常分散 `$\sigma^2/(2\theta)$` が厳密に導出された。

```math
\frac{\partial p}{\partial t} = \theta \frac{\partial}{\partial x}(x \, p) + \frac{\sigma^2}{2}\frac{\partial^2 p}{\partial x^2}
```

定常解: `$p_\infty(x) = \mathcal{N}(0, \sigma^2/(2\theta))$`。Zone 5.9で数値確認したOU定常分布がFPE解として厳密導出。

#### SDE ↔ Fokker-Planck ↔ Score SDE の三角関係

```mermaid
graph TD
    SDE["SDE<br/>dX = f dt + g dW<br/>パスの記述"] -->|Itô's formula| FPE["Fokker-Planck<br/>∂p/∂t = -∂(fp) + ½∂²(g²p)<br/>密度の時間発展"]
    FPE -->|定常解 ∂p/∂t=0| STAT["定常分布<br/>p∞(x)"]
    SDE -->|Anderson 1982| REV["Reverse SDE<br/>dX = [f - g²∇log p]dt + g dW̄"]
    FPE -->|∇log p_t| SCORE["Score function<br/>∇ log p_t(x)"]
    SCORE --> REV
    REV -->|generative model| GEN["Score SDE<br/>Song+ 2020"]

    style SDE fill:#e3f2fd
    style FPE fill:#fff9c4
    style GEN fill:#c8e6c9
```

| 視点 | 記述対象 | 数学的対象 | 生成モデルでの役割 |
|:-----|:--------|:---------|:---------------|
| SDE | 1つのパス | `$X_t(\omega)$` | Forward/Reverse process |
| Fokker-Planck | 確率密度の時間発展 | `$p(x, t)$` | ノイズスケジュール設計 |
| Score function | 密度の勾配 | `$\nabla \log p_t$` | NN で学習する対象 |

**数値的Fokker-Planck検証**:

FP方程式の定常解 `$p_\infty(x) \propto \exp(-\theta x^2/\sigma^2)$` をシミュレーションで確認する:

```python
import numpy as np
from scipy.stats import norm

theta, sigma = 1.0, 1.0
stat_var = sigma**2 / (2*theta)
stat_std = np.sqrt(stat_var)

# FP predicts: p_inf(x) = N(0, sigma^2/(2*theta))
rng = np.random.default_rng(0)
X = np.full(5000, 0.0)  # start at 0 (already stationary)
dt = 0.01
for _ in range(10_000):
    X += -theta*X*dt + sigma*np.sqrt(dt)*rng.standard_normal(5000)

# Chi-square goodness of fit test: bins
bins = np.linspace(-4, 4, 20)
counts, _ = np.histogram(X, bins=bins)
expected = norm.cdf(bins[1:], 0, stat_std) - norm.cdf(bins[:-1], 0, stat_std)
expected *= len(X)
chi2 = float(((counts - expected)**2 / expected).sum())
print(f"X(inf): mean={X.mean():.3f}  std={X.std():.3f}  stat_std={stat_std:.3f}")
print(f"chi2 statistic={chi2:.1f}  (expected ~18 for 18 dof)")
```

> **Note:** **第30回への予告**: ここでは1次元・OU過程の場合のFokker-Planckを味見した。第30回「Diffusion Models II」では、多次元FPE の完全導出、reverse SDE の厳密証明（Girsanov変換）、そしてFPEからScore SDEの学習目的関数（denoising score matching）を導く。Fokker-Planckは拡散モデル理論の「裏ボス」だ。

### 7.2 生成モデルの測度論的統一

**Pushforward測度**:

可測写像 `$T: (\mathcal{X}, \mathcal{F}) \to (\mathcal{Y}, \mathcal{G})$` と測度 `$\mu$` に対し、Pushforward測度 `$T_\# \mu$` は:

```math
(T_\# \mu)(B) = \mu(T^{-1}(B)) \quad \forall B \in \mathcal{G}
```

直感: `$T$` で変換した後の測度。`$T$` が可逆かつ微分可能なら変数変換公式:

```math
\int_\mathcal{Y} f \, d(T_\# \mu) = \int_\mathcal{X} (f \circ T) \, d\mu
```

Normalizing Flowsの確率密度変換（`$p_z$` → `$p_x = |\det J_T|^{-1} p_z \circ T^{-1}$`）はこの公式の直接適用。

**すべての生成モデルは測度輸送**: ソース測度 `$\mu_0$`（ガウスノイズ）からターゲット測度 `$\mu_1$`（データ分布）へ。

- Normalizing Flow: 決定論的・可逆な写像 `$T$`（`$T_\# \mu_0 = \mu_1$`）
- VAE: 確率的エンコーダ `$q_\phi(z|x)` と デコーダ `$p_\theta(x|z)$` の間接的輸送
- Diffusion: SDEの forward/reverse で測度を変形
- Flow Matching: ODEのベクトル場 `$v_t$` で確率パス `$\mu_t$` を設計（`$\mu_0 \to \mu_1$`）

```mermaid
graph TD
    A["測度輸送<br/>T#p₀ = p₁"] --> B["Normalizing Flows<br/>可逆変換 T"]
    A --> C["VAE<br/>潜在空間の測度"]
    A --> D["Diffusion<br/>SDE forward/reverse"]
    A --> E["Flow Matching<br/>確率パス p_t"]

    D --> F["Score SDE<br/>∇log p_t"]
    E --> G["Rectified Flow<br/>直線化パス"]
    E --> H["Stochastic Interpolants<br/>一般化補間"]

    I["Radon-Nikodym<br/>dP/dQ"] -.-> D
    I -.-> F
    J["Pushforward<br/>T#μ"] -.-> B
    J -.-> E
    K["Markov Chain<br/>遷移核"] -.-> D
```

> すべての生成モデルは、**ソース測度 `$p_0$`（通常はガウスノイズ）をターゲット測度 `$p_1$`（データ分布）に輸送する写像**として統一的に理解できる。測度論はこの統一的視点を与える言語である。

**Wasserstein距離**: 測度間の距離として最も自然なのが `$W_p$` 距離:

```math
W_p(\mu, \nu) = \left(\inf_{\gamma \in \Gamma(\mu, \nu)} \int \|x - y\|^p \, d\gamma(x, y)\right)^{1/p}
```

ここで `$\Gamma(\mu, \nu)$` は `$\mu$`, `$\nu$` を周辺分布に持つ結合分布（カップリング）全体の集合。`$W_2$`（`$p=2$`）は最適輸送コスト（地球を動かすコスト）。KLと異なりサポートが重ならなくても有限値を持つ（GANの訓練に有利）。

**各モデルの使うカップリング**:
- GAN: 偶然のカップリング（GANは暗黙的に最適輸送をしている、という視点）
- Flow Matching (COT): `$W_2$` 最適カップリング → まっすぐな軌跡
- 拡散モデル: ガウス加算ノイズ（確率的カップリング）
- Normalizing Flow: 決定論的カップリング（可逆写像）

Wasserstein距離の計算は一般に `$O(n^3)$` の線形計画問題だが、Sinkhorn algorithm（エントロピー正則化）で `$O(n^2/\epsilon^2)$` に削減できる。これもLebesgue積分・測度論の言語なしには定式化できない。

### 7.3 今回の冒険の収穫

| Zone | 何を学んだか | キーワード |
|:--:|:--|:--|
| 0 (Part1) | なぜ測度論が必要か | Cantor集合、Riemann積分の限界、混合分布 |
| 1-4 (Part1) | 測度空間と理論 | `$\sigma$`-加法族、Lebesgue積分、MCT/DCT、Radon-Nikodym、pushforward、収束、確率過程、伊藤解析 |
| 5 (Part2) | 実装 | Monte Carlo `$O(1/\sqrt{N})$`、IS (Radon-Nikodym)、KDE (Silvermanルール)、MH法 (詳細釣り合い)、Brown運動 (二次変動)、GBM (Itô補正)、OU過程 (平均回帰)、Langevin (Score)、Euler-Maruyama (強/弱収束) |
| 6 (Part2) | 研究動向 | Score SDE (VP-SDE)、VP-SDE収束 (Grönwall)、離散拡散 (KL保証)、Flow Matching (条件付きベクトル場) |
| 7 (Part2) | まとめ | Fokker-Planck (SDE↔密度)、測度輸送統一、FAQ |

**今回の本質的洞察5選**:

1. **測度論はコードのバグ予防接種** — 測度ゼロ、絶対連続、Radon-Nikodym、Fatouの補題を知ることで「なぜNaNが出るか」が分かる
2. **$O(1/\sqrt{N})$ はMonte Carloの壁** — これを超えるには分散削減（IS/層化）か解析的計算が必要。次元の呪いと組み合わさると `$O(N^{-1/d})$` に落ちる
3. **SDE ↔ 確率密度のPDE** — Fokker-Planck方程式は「個々の粒子の軌跡（SDE）」と「集団の密度進化（PDE）」の橋渡し
4. **Score関数 = 確率密度の勾配** — 生成モデルの本質は「どこに確率密度が高いか」を知ること。Langevin dynamicsは確率の「上り坂」を登る
5. **深層生成モデルは確率空間間の写像** — VAE/GAN/拡散/Flowは全てpushforward測度の言語で統一して理解できる

### 7.4 数式記号対照表

| 記号 | 意味 | 初出 |
|:-----|:-----|:-----|
| `$(\Omega, \mathcal{F}, P)$` | 確率空間（標本空間、σ-加法族、確率測度） | Z1 |
| `$P \ll Q$` | 絶対連続 `$Q(A)=0 \Rightarrow P(A)=0$` | Z1 |
| `$\frac{dP}{dQ}$` | Radon-Nikodym導関数（確率密度の厳密定義） | Z1 |
| `$X_n \xrightarrow{a.s.} X$` | 概収束 `$P(\lim X_n = X) = 1$` | Z1 |
| `$X_n \xrightarrow{d} X$` | 分布収束（最弱、CLTはこれ） | Z1 |
| `$[W]_t = t$` | Brown運動の二次変動（伊藤補正の源泉） | Z1 |
| `$dX = \mu dt + \sigma dW$` | 確率微分方程式（SDE） | Z1 |
| `$\boldsymbol{\pi} P = \boldsymbol{\pi}$` | 定常分布の固有方程式 | Z1 |
| `$\nabla_x \log p(x)$` | Score関数（Langevin / Score SDE の核心） | Z5 |
| `$v_t(x)$` | Flow Matchingの速度場 | Z6 |
| `$\text{ESS}$` | 有効サンプルサイズ（IS品質指標） | Z5 |
| `$\alpha(x, x')$` | MH法の受理確率 | Z5 |
| `$\partial_t p = -\partial_x(fp) + \frac{1}{2}\partial_{xx}(g^2 p)$` | Fokker-Planck方程式 | Z6 |

### 7.5 数式→コード 1:1 対照

| 数式操作 | Python | 数値的落とし穴 |
|:---------|:-------|:--------------|
| `$\int f \, d\mu \approx \frac{1}{N}\sum_i f(X_i)$` | `np.mean(f(x))` | Nは1e4以上推奨 |
| `$w(x) = p(x)/q(x)$` | `np.exp(logp - logq)` | log空間で計算（overflow防止）|
| `$[W]_T = \sum (\Delta W)^2$` | `(dW**2).sum(axis=0)` | dtが小さいほど精確 |
| `$X_{n+1} = X_n + f\Delta t + g\sqrt{\Delta t}Z$` | `X + f(X)*dt + g(X)*sqrt_dt*Z` | Brownian incrementはN(0,dt)|
| `$-\theta X dt + \sigma dW$` | `-theta*X*dt + sigma*sqrt_dt*Z` | 平均回帰は正のthetaで保証 |
| `$\min(1, \pi(x')/\pi(x))$` | `min(0, log_pi_new - log_pi_old)` | log比較で overflow 回避 |
| `$\partial_t p + \nabla \cdot (pu) = 0$` | `(dp_dt + np.gradient(p*u, dx)).sum()` | 連続性方程式の数値検証 |
| `$\mathbb{E}[f(X)] \pm 1.96 \hat{\sigma}/\sqrt{N}$` | `mean ± 1.96*std(ddof=1)/sqrt(N)` | CLT前提、`$N \geq 30$` 推奨 |
| `$e^{\mu T + \sigma W_T - \sigma^2T/2}$` | `S0*np.exp((mu - 0.5*sigma**2)*T + sigma*W_T)` | Itô補正 `-sigma²/2` 必須 |
| `$\sum_i w_i^2 / (\sum_i w_i)^2$` | `1 / ((w/w.sum())**2).sum()` | ESS = effective sample size |
| `$\sigma(\{A\}) = \{\emptyset, A, A^c, \Omega\}$` | `frozenset({frozenset(), A, Omega-A, Omega})` | 最小σ-加法族 |

### 7.7 最重要テイクアウェイ

> **⚠️ Warning:** **3つの核心メッセージ**
>
> 1. **測度論は「積分できる対象」を最大限に広げる言語** — Riemann積分では扱えない関数（Dirichlet関数、混合分布）をLebesgue積分が処理する。確率論はこの上に構築される。
>
> 2. **Radon-Nikodym導関数は測度の「比較」を可能にする** — PDFは `$dP/d\lambda$`、尤度比は `$dP/dQ$`、importance weightも `$dP/dQ$`。生成モデルのlossは常に測度間の「距離」を最小化している。
>
> 3. **確率過程は「時間的に繋がった測度の族」** — Markov連鎖は離散時間、Brown運動は連続時間。DDPMは離散Markov連鎖、Score SDEは連続SDE。測度論が両者を統一する。

**実装への直接示唆**:

| 測度論の概念 | 実装上の意味 | 無視した場合のバグ |
|:------------|:------------|:-----------------|
| `$P \ll Q$`（絶対連続） | IS重みが有限 | NaN / Inf 重み |
| DCT | 勾配と期待値の交換 | 誤った勾配推定 |
| 二次変動 `$[W]_t = t$` | `dW ~ N(0, dt)` | `dt`忘れ（`sqrt(dt)`の欠如） |
| Itô補正 | GBMの `$-\sigma^2/2$` 項 | `E[S_T] ≠ S_0 e^{μT}` |
| Radon-Nikodym | 対数空間でIS計算 | 数値オーバーフロー |
| Fokker-Planck | 定常分布 `$p_\infty \propto e^{-U}$` | 非定常サンプルでの偏り |
| Girsanov変換 | 測度変換の尤度比 | Novikov条件未確認で発散 |

> **Note:** 上記のバグパターンは全て「測度論的概念を無視した実装」が原因だ。測度論の学習コストは「バグ修正にかかるコスト」への先行投資と考えられる。

実際、実装のバグを追いかけていると「なぜこうなるのか」という問いは必ず測度論的な概念に行き着く。

### 7.8 FAQ

<details><summary>Q1: 測度論を学ばなくても深層生成モデルの論文は読めますか？</summary>
**A**: 実装レベルでは可能。しかしScore SDE [^2]、Flow Matching [^7]、Rectified Flow [^6] のような理論的に深い論文は、測度論なしでは「なぜこの式が正しいか」が理解できない。特にRadon-Nikodym導関数とpushforward measureは必須の概念。
</details>

<details><summary>Q2: Itô積分とStratonovich積分の違いは？</summary>
**A**: Itô積分は左端点評価、Stratonovichは中点評価。Itôは「未来を知らない」（適合過程）が連鎖律にItô補正が必要。Stratonovichは連鎖律が通常通りだがマルチンゲール性を失う。金融・MLではItôが標準。
</details>

<details><summary>Q3: DDPMでMarkov連鎖を使う理由は？</summary>
**A**: Markov性により (1) 同時分布が遷移核の積に分解、(2) 各ステップ独立設計、(3) reverse processもMarkov。非Markovだと全ステップ同時最適化が必要で計算不可能。
</details>

<details><summary>Q4: 絶対連続 `$P \ll Q$` の重要性は？</summary>
**A**: `$P \ll Q$` のとき `$dP/dQ$` が存在。生成モデルで `$p_\theta$` と `$p_{\text{data}}$` が相互絶対連続でないとKL divergenceが `$+\infty$`。GANのmode collapse の一因。
</details>

<details><summary>Q5: Euler-Maruyama法の時間幅Δtをどう選ぶか？</summary>
**A**: 弱収束 `$O(\Delta t)$` より、精度 `$\epsilon$` を達成するには `$\Delta t = O(\epsilon)$`、ステップ数 `$T/\Delta t = O(T/\epsilon)$`。DDPMの `$T=1000$` は `$\epsilon = 10^{-3}$` 程度の精度に対応。実際には学習された逆過程の品質がボトルネックになるので、`$T$` が大きすぎても品質は飽和する。DDIM [^12] は `$T$` を10-50に削減できる「弱収束で十分」の好例。
</details>

<details><summary>Q6: Score関数 ∇log p(x) は何を表すか？</summary>
**A**: 確率密度の対数微分。高確率領域に向かう方向を指す。直感的には「今いる場所から最も確率が高い場所への勾配」。Fisher情報量 `$I(\theta) = \mathbb{E}[(\nabla \log p_\theta)^2]$` の被積分関数でもある。Stein Identity: `$\mathbb{E}_p[s(x)f(x)] = -\mathbb{E}_p[\nabla f(x)]$`（`$s = \nabla \log p$`）がScore Matchingの理論的基礎。
</details>

<details><summary>Q7: Girsanov変換を実装する際の注意点は？</summary>

**A**: Girsanov変換は測度変換であり、実装では **尤度比（Radon-Nikodym導関数）の数値安定性** が最大の問題。尤度比は:

```math
\frac{dQ}{dP}\bigg|_{\mathcal{F}_T} = \exp\left(\int_0^T \theta_t \, dW_t - \frac{1}{2}\int_0^T \theta_t^2 \, dt\right)
```

問題点: 期待値 `$\mathbb{E}_P[dQ/dP] = 1$` が成り立つはずだが、有限サンプルでは爆発しやすい。`$\theta_t^2$` が大きいとき、指数の分散が爆発する（lognormal分布の分散は `$e^{\sigma^2}(e^{\sigma^2}-1)$` で`$\sigma$` 大で爆発）。

**実装的解決**: `log-sum-exp` で対数空間で計算する。Novikov条件 `$\mathbb{E}[\exp(\frac{1}{2}\int_0^T \theta_t^2 dt)] < \infty$` が成立するかを事前確認すること。
</details>

<details><summary>Q8: 深層生成モデルのバグの多くは測度論的エラーという主張について</summary>

**A**: 誇張ではない。実際によくある3パターン:

1. **Trap: `$dP/dQ$` が存在しない状況でKLを計算**: `$\text{support}(p) \not\subseteq \text{support}(q)$` のとき KL = +∞。実装では NaN/Inf が出る。GANの訓練初期不安定の原因の一つ。

2. **Trap: score関数の評価点が対数密度の定義外**: `$\nabla_x \log p(x)$` は `$p(x) > 0$` の点でのみ定義。境界付近でスコアが爆発する。DDPM は小さな `$\sigma_{\min} > 0$` で回避。

3. **Trap: Fokker-Planckの境界条件忘れ**: 半無限区間 `$[0, \infty)$` のFPは `$x=0$` での境界条件（Neumann or absorbing）が必要。忘れると定常解が収束しない。

測度論をマスターすることは「バグの予防接種」と言える。
</details>



### 7.9 よくある罠

> **⚠️ Warning:** **Trap 1**: 測度ゼロ ≠ 空集合。`$\mathbb{Q}$` も Cantor集合も測度ゼロだが稠密・非可算。
>
> **Trap 2**: Riemann可 ⇒ Lebesgue可 だが逆は×。Dirichlet関数 `$1_\mathbb{Q}$` はLebesgue積分=0 だがRiemann不可。
>
> **Trap 3**: 概収束 ⇒ 確率収束 だが逆は×。Typewriter sequenceが反例。
>
> **Trap 4**: `$d(W^2) = 2W \, dW + dt$`。最後の `$+dt$` （二次変動）を忘れるとItô補正を見逃す。
>
> **Trap 5**: 重点サンプリングでESS < 10%なら結果は信頼できない。
>
> **Trap 6**: Euler-Maruyamaで `$g(X) \cdot \Delta t \cdot Z$` と書くと間違い。正しくは `$g(X) \cdot \sqrt{\Delta t} \cdot Z$`。Brown運動増分 `$\Delta W \sim \mathcal{N}(0, \Delta t)$` の標準偏差は `$\sqrt{\Delta t}$`。
>
> **Trap 7**: GBMで `$S(T) = S_0 \exp(\mu T + \sigma W_T)$` と書くとItô補正を忘れている。正しくは `$S(T) = S_0 \exp((\mu - \sigma^2/2)T + \sigma W_T)$`。`$\mathbb{E}[S(T)]$` が `$S_0 e^{\mu T}$` にならないことで確認できる。
>
> **Trap 8**: `$\sigma$`-加法族の閉包性を直感的に「全ての部分集合を含む」と思うと間違い。`$\sigma(\mathcal{C})$` は `$\mathcal{C}$` を含む **最小の** `$\sigma$`-加法族であり、任意の部分集合は含まない。例: `$\sigma(\{a\}) = \{\emptyset, \{a\}, \{a\}^c, \Omega\}$` は `$|\Omega| \geq 3$` のとき全部分集合より小さい。
>
> **Trap 9**: 重点サンプリングで `$q(x)$` が `$p(x)f(x)$` の「重い尾」をカバーしていない場合、重み `$w_i = p(x_i)/q(x_i)$` が少数の点に集中してESS → 1 になる。`$\text{ESS} = (\sum w_i)^2 / \sum w_i^2$` を常に報告すること。ESSが有効サンプル数を表す指標として広く使われる。
>
> **Trap 10**: Flow Matchingで Conditional OT パスを使わずに直線パスを使うと、交差が起きて学習困難になる。`$x_t = (1-t)x_0 + tx_1$` は `$x_0 \sim p_0$`, `$x_1 \sim p_1$` が独立のとき軌跡が交差する。Conditional OT [^7] は `$(x_0, x_1)$` を最適輸送カップリングから同時にサンプルすることで交差を最小化する。





### 7.6 第5回まとめ図 — 理論と実装の橋

```mermaid
graph LR
    subgraph Theory["理論（Part1）"]
        RN["Radon-Nikodym<br/>dP/dQ"]
        LEB["Lebesgue積分<br/>∫f dμ"]
        CONV["収束定理<br/>MCT/DCT/Fatou"]
        ITO["伊藤積分<br/>∫f dW"]
        FP["Fokker-Planck<br/>∂p/∂t = L†p"]
    end
    subgraph Impl["実装（Part2）"]
        IS["重点サンプリング<br/>w=p/q"]
        MC["Monte Carlo<br/>1/N Σf(X_i)"]
        DCT_VERIFY["DCT検証<br/>g_n→e^{-x}"]
        EM["Euler-Maruyama<br/>X_{n+1}=X_n+fdt+g√dtZ"]
        LANG["Langevin Dynamics<br/>X+=ε/2·∇logp+√ε·Z"]
    end
    RN --> IS
    LEB --> MC
    CONV --> DCT_VERIFY
    ITO --> EM
    FP --> LANG
    style Theory fill:#e3f2fd
    style Impl fill:#c8e6c9
```

### 7.10 理解度の自己診断

以下の問いに答えられるか確認しよう。

<details><summary>診断問1: Lebesgue積分のよさをRiemannと比較して説明せよ</summary>

**Riemann積分の弱点**: 積分を「x軸を分割して細長い長方形で近似」する。これは関数が「ほぼ連続」でないと機能しない。例: Dirichlet関数 `$1_\mathbb{Q}(x)$` はRiemannで積分不可。

**Lebesgue積分**: 「y軸を分割して対応するxの集合の測度を使う」。`$\int f \, d\mu = \int_0^\infty \mu(\{x: f(x) > t\}) \, dt$` (層別表現)。Dirichlet関数: `$\int 1_\mathbb{Q} \, d\mu = \mu(\mathbb{Q}) = 0$`（有理数の測度ゼロ）。

**核心的優位性**: 積分と極限の交換が保証される（DCT/MCT）。これが確率論・測度論ベースのMLの理論証明で必須。
</details>

<details><summary>診断問2: 伊藤補題を使ってOU過程の解析解を求めよ</summary>

OU過程 `$dX = -\theta X dt + \sigma dW$` に `$f(t, X) = e^{\theta t} X$` を適用。

`$df = \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial X}dX + \frac{1}{2}\frac{\partial^2 f}{\partial X^2}(dX)^2$`

`$= \theta e^{\theta t} X dt + e^{\theta t}(-\theta X dt + \sigma dW) + 0$`

`$= \sigma e^{\theta t} dW$`

両辺積分: `$e^{\theta t}X_t - X_0 = \sigma \int_0^t e^{\theta s} dW_s$`

`$\therefore X_t = X_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dW_s$`

確率積分の平均ゼロ性より `$\mathbb{E}[X_t] = X_0 e^{-\theta t} \to 0$`（平均回帰）。
</details>

<details><summary>診断問3: Langevin dynamicsで目標分布が `$p^*(x) \propto e^{-U(x)}$` のとき、定常分布が `$p^*$` に収束することを示せ</summary>

Langevin SDE: `$dX = -\nabla U(X) dt + \sqrt{2} dW$`（`$\nabla \log p^* = -\nabla U$` を使った）

対応するFokker-Planck方程式: `$\partial_t p = \nabla \cdot (p \nabla U) + \Delta p$`

定常解 `$p^*$` の検証: `$\nabla \cdot (p^* \nabla U) + \Delta p^* = ?$`

`$= \nabla p^* \cdot \nabla U + p^* \Delta U + \Delta p^*$`

`$p^* = Z^{-1}e^{-U}$ より `$\nabla p^* = -p^* \nabla U$`、`$\Delta p^* = p^*(|\nabla U|^2 - \Delta U)$`

代入: `$-p^*|\nabla U|^2 + p^*\Delta U + p^*|\nabla U|^2 - p^*\Delta U = 0$ ✓
</details>

### 7.11 実装チェックリスト

研究・実務で測度論の知識が必要になる場面と、実装前に確認すべき問い:

**確率過程シミュレーション前チェック**:
- [ ] SDEの係数 `$f, g$` は可測か（Borel可測性）
- [ ] 定常分布が存在するか（Fokker-Planck定常解が存在するか）
- [ ] EM法のタイムステップ `$\Delta t$` は十分小さいか（弱収束 `$O(\Delta t)$`）
- [ ] `$g(X)$` が `$X$` に依存する場合: Milstein補正が必要か確認

**生成モデル実装前チェック**:
- [ ] `$p_\theta$` と `$p_{\text{data}}$` は絶対連続か（KLが有限か）
- [ ] スコア関数 `$\nabla \log p_t(x)$` の計算点 `$x$` は `$p_t > 0$` の領域か
- [ ] Importance Sampling使用時: `$\text{ESS} > N/10$` か
- [ ] 条件付き期待値のタワー性質を仮定しているか（Markov性が崩れていないか）

**理論的根拠の確認**:
- [ ] 積分と微分の交換: DCTの仮定（優関数 `$g$` で `$|f_n| \leq g$`、`$\mathbb{E}[g] < \infty$`）を確認
- [ ] CLTを使う前: `$\text{Var}[f(X)] < \infty$` か（対数正規など裾が重い場合は危険）
- [ ] KL分解でタワー性質を使う前: 適切なフィルトレーション `$\mathcal{F}_t$` が設定されているか

### 7.14 次回予告 — 第6回: 情報理論・最適化理論

次の第6回では **情報理論と最適化理論** に進む。KLダイバージェンスとSGDで武装する回だ。

> **Note:** **第6回のハイライト**
> - Shannon Entropy: `$H(X) = -\sum p(x) \log p(x)$`
> - KL Divergence: `$D_{\text{KL}}(p \| q) = \int p \log \frac{p}{q} \, d\mu$` — Radon-Nikodym導関数再び!
> - Mutual Information: `$I(X;Y)$` — 依存の測度
> - f-Divergence: KLの統一的一般化
> - 勾配降下法: SGD・Adam — パラメータ最適化の決定版
> - 損失関数設計: Cross-Entropy = KL最小化の等価性

> **第4回** の確率分布 → **第5回** の測度論的基礎 → **第6回** の情報理論・最適化理論。3つの講義で確率論の「三角形」が完成する。

**第6回の数学的位置づけ**: 情報理論はLebesgue積分の応用だ。Shannon Entropy:

```math
H(X) = -\int p(x) \log p(x) \, d\mu(x)
```

はLebesgue積分そのもの。KL Divergence:

```math
D_{\mathrm{KL}}(P \| Q) = \int \frac{dP}{dQ} \log \frac{dP}{dQ} \, dQ
```

はRadon-Nikodym導関数の積分。第5回で学んだことが情報理論の土台に直接なっている。**Mutual Information** `$I(X; Y) = D_{\mathrm{KL}}(P_{XY} \| P_X \otimes P_Y)$` は結合分布と周辺分布の積のKL距離 — これも測度論の言語でしか厳密には定義できない。

**最適化理論との接続**: 勾配降下法の収束解析（第6回後半）では、損失関数の凸性（Hessianの固有値条件）と収束レート `$O(1/\sqrt{T})$`（確率的SGD）の証明に確率論的技法が必要。具体的には確率変数の和の集中不等式（Hoeffdingの不等式、Azuma-Hoeffding）を使う。これも測度論の応用だ。

---


### 7.15 💀 パラダイム転換の問い

**【問い】「確率論なんて深層学習に必要ない」という主張に反論できるか？**

この問いは、実装優先の実務家から繰り返し聞こえてくる。事実、PyTorchで大規模モデルを訓練するだけなら測度論の知識はほぼ不要だ。だが、以下の状況に直面したとき、その主張は崩壊する:

- **Score SDE [^2]** を読んで「なぜReverse SDEが成り立つか」を理解しようとしたとき — Anderson (1982) のRadon-Nikodym引数がなければ読めない
- **Flow Matching [^7]** でConditional OTカップリングが「なぜ必要か」を説明しようとしたとき — 測度輸送の基礎がなければ答えられない
- **学習の収束証明** を書こうとしたとき — 収束定理（MCT/DCT）の交換可能性がなければ証明できない

<details><summary>歴史的背景: 確率論の工学への浸透</summary>

確率論の厳密な基礎（コルモゴロフの公理化）が確立したのは1933年。それ以前は「確率とは何か」に複数の不整合な定義が混在していた。Shannonの情報理論（1948）、Wienerのノイズ理論（1948）、伊藤清の確率積分（1944）が急速に実用化され、工学に測度論が浸透した。

深層学習の爆発期（2012年以降）は「測度論なしでも動く」という幻想を生んだが、理論的突破（Score SDE: 2020、Flow Matching: 2022）が「厳密な確率論なしでは理解できない」という現実を復活させた。歴史は繰り返す。
</details>

**反論の核心**: 「動く実装」と「理論的理解」は別物だ。機械が動けば十分、という立場は **次世代のアーキテクチャを設計する能力を放棄する** ことと同義だ。確率論は「インフラ」として表に出ないが、消えてはいない。

---

> **⚠️ Warning:** **PB Question**: Lebesgue積分なくして確率密度なし。測度を知らずに生成モデルを語れるか？
>
> Riemann積分の世界では、`$\mathbb{Q}$` 上の一様分布のような「病的な」分布を扱えない。Lebesgue積分はこの制限を取り払い、Radon-Nikodym導関数として確率密度関数を厳密に定義する。
>
> DDPMのforward processは、ガウスの遷移核を持つMarkov連鎖であり、その分布の変化は pushforward measure の系列として記述される。Score SDE は、この離散過程を連続のSDEに拡張し、Brown運動のItô積分を使って定式化する。Flow Matching は、測度輸送の最適化問題として生成モデルを再定式化する。
>
> **すべての道は測度論に通じる。**
>
> 測度論を学ぶことは、個々の手法の背後にある統一的な構造を見ることである。それは単なる数学的厳密性のためではなく、**新しい生成モデルを設計するための言語**を手に入れることを意味する。
>
> 次の第6回では、この測度の言語の上に「情報」の概念を構築する。KLダイバージェンスは `$\frac{dP}{dQ}$` の対数の期待値 — まさにRadon-Nikodym導関数が主役だ。

---
> Progress: 100%

---

## 参考文献

[^1]: Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. arXiv:2006.11239 — DDPMの原論文。ガウス遷移核を持つMarkov連鎖として拡散過程を定義。

[^2]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. arXiv:2011.13456 — Score SDEの原論文。DDPMを連続SDEに拡張し、reverse SDEでサンプリング。

[^3]: Levin, D. A., & Peres, Y. (2017). *Markov Chains and Mixing Times* (2nd ed.). American Mathematical Society. — Markov連鎖理論の標準教科書。エルゴード定理・混合時間の詳細。

[^4]: Itô, K. (1944). *Stochastic Integral*. Proceedings of the Imperial Academy, 20(8), 519-524. — 確率積分の原論文。Brown運動に対する積分を定義。

[^5]: Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). *Weak convergence and optimal scaling of random walk Metropolis algorithms*. Annals of Applied Probability, 7(1), 110-120. — MH法の最適受理率23.4%の理論。

[^6]: Liu, X., Gong, C., & Liu, Q. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023. arXiv:2209.03003 — Rectified Flowの原論文。パスの直線化による高速生成。

[^7]: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). *Flow Matching for Generative Modeling*. ICLR 2023. arXiv:2210.02747 — Flow Matchingの原論文。条件付き速度場の回帰で生成モデルを構築。

[^9]: Anderson, B. D. O. (1982). *Reverse-time diffusion equation models*. Stochastic Processes and their Applications, 12(3), 313-326. — Reverse SDEの理論。Score SDEの基礎。

[^10]: Tao, M. (2025). VP-SDE Discretization Error Analysis via Gronwall Inequality. arXiv:2506.08337 — Grönwall不等式によるEuler-Maruyama離散化誤差の上界。

[^11]: Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces*. NeurIPS 2021. arXiv:2107.03006 — 離散状態空間拡散モデルの原論文。

[^12]: Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR 2021. arXiv:2010.02502 — DDIMの原論文。ステップ数を大幅削減しながら品質維持。

[^13]: Albergo, M. S., & Vanden-Eijnden, E. (2022). *Building Normalizing Flows with Stochastic Interpolants*. ICLR 2023. arXiv:2209.15571 — Stochastic Interpolantsの原論文。Flow MatchingとDiffusionの統一。

---

## 著者リンク

- Blog: https://fumishiki.dev
- X: https://x.com/fumishiki
- LinkedIn: https://www.linkedin.com/in/fumitakamurakami
- GitHub: https://github.com/fumishiki
- Hugging Face: https://huggingface.co/fumishiki

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
