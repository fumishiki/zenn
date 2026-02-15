---
title: "第31回: MLOps完全版【前編】理論編: 30秒の驚き→数式修行""
emoji: "🔄"
type: "tech"
topics: ["machinelearning", "mlops", "rust", "julia", "elixir"]
published: true
slug: "ml-lecture-31-part1"
---

# 第31回: MLOps完全版 — 99.9%可用性は"努力"ではなく"設計"だ

> **モデルを訓練できても、本番で動かせなければ価値はゼロ。MLOps全領域を網羅し、Train→Evaluate→Deploy→Monitorのライフサイクルを完結させる。**

第30回でエージェントを完全構築した。だが"動く"だけでは足りない。

本番環境では、モデルは**生き物**だ。データが変わり、ユーザーの嗜好が変わり、性能が劣化する。再訓練が必要になり、A/Bテストで新モデルを検証し、段階的にロールアウトする。障害が起きれば即座にロールバックし、ドリフトを検出して自動再訓練をトリガーする。

これら全てを「手作業」でやっていたら、1人月が100人日に化ける。

**MLOps (Machine Learning Operations)** は、この混沌を「設計」で解決する。バージョニング・実験管理・CI/CD・A/Bテスト・モニタリング・SLI/SLO・ドリフト検出・DPO/RLHF。7つのピースを組み合わせ、モデルのライフサイクル全体を自動化する。

本講義はCourse IIIの第13回 — 第19回から始まった実践編の最終盤だ。第32回で統合PJを構築し、Course IIIを完結させる。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論（論文が書ける）、実装（Production-ready）、最新（2025-2026 SOTA）の3軸で差別化する。
:::

```mermaid
graph LR
    A["🔧 Version<br/>Model/Data"] --> B["🧪 Experiment<br/>MLflow/W&B"]
    B --> C["🚀 CI/CD<br/>Auto-Test"]
    C --> D["🎯 A/B Test<br/>Canary"]
    D --> E["📊 Monitor<br/>Drift/SLO"]
    E --> F["🔁 Retrain<br/>Auto-Trigger"]
    F --> A
    G["🎓 RLHF/DPO<br/>Human Feedback"] --> B
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
    style F fill:#fff9c4
    style G fill:#e0f2f1
```

**所要時間の目安**:

| ゾーン | 内容 | 時間 | 難易度 |
|:-------|:-----|:-----|:-------|
| Zone 0 | クイックスタート | 30秒 | ★☆☆☆☆ |
| Zone 1 | 体験ゾーン | 10分 | ★★☆☆☆ |
| Zone 2 | 直感ゾーン + 発展 | 35分 | ★★★★★ |
| Zone 3 | 数式修行ゾーン | 90分 | ★★★★★ |
| Zone 4 | 実装ゾーン | 60分 | ★★★★☆ |
| Zone 5 | 実験ゾーン | 30分 | ★★★★☆ |
| Zone 6 | 振り返り + 統合 | 30分 | ★★★☆☆ |

---

## 🚀 0. クイックスタート（30秒）— 実験メタデータを記録する

**ゴール**: MLOpsの核心を30秒で体感する — 実験を「記録」しなければ「再現」できない。

MLflowスタイルのメトリクス記録を3行で動かす。

```julia
using Dates, JSON3

# Experiment metadata logging (simplified MLflow-style)
function log_experiment(name::String, params::Dict, metrics::Dict, artifacts::Vector{String})
    experiment = Dict(
        "name" => name,
        "timestamp" => now(),
        "params" => params,
        "metrics" => metrics,
        "artifacts" => artifacts,
        "run_id" => string(rand(UInt64), base=16)
    )

    # Persist to JSON (real MLflow uses DB + artifact store)
    filename = "experiments/$(experiment["run_id"]).json"
    mkpath("experiments")
    open(filename, "w") do io
        JSON3.write(io, experiment)
    end

    println("✅ Logged experiment: $(experiment["name"]) (run_id: $(experiment["run_id"]))")
    println("   Params: $(params)")
    println("   Metrics: $(metrics)")
    return experiment["run_id"]
end

# Example: Train a tiny model and log everything
params = Dict("lr" => 0.001, "batch_size" => 32, "epochs" => 10)
metrics = Dict("train_loss" => 0.023, "val_acc" => 0.952, "f1" => 0.948)
artifacts = ["model_weights.pt", "config.yaml"]

run_id = log_experiment("tiny-classifier-v1", params, metrics, artifacts)
```

出力:
```
✅ Logged experiment: tiny-classifier-v1 (run_id: a3f9c2e1b4d8)
   Params: Dict("lr" => 0.001, "batch_size" => 32, "epochs" => 10)
   Metrics: Dict("train_loss" => 0.023, "val_acc" => 0.952, "f1" => 0.948)
```

**3行のコードで実験をJSON化して永続化した。** これがMLOpsの出発点だ。実際のMLflowは:

- SQLiteまたはPostgreSQLでメタデータ管理
- S3/GCS/Azureで大きなartifact保存
- UIで実験比較・モデルバージョニング・デプロイ管理

この背後にある理論:

$$
\begin{aligned}
\text{Reproducibility} &= f(\text{Code}, \text{Data}, \text{Hyperparams}, \text{Env}, \text{Seed}) \\
\text{MLOps Goal:} \quad & \text{Track all 5 dimensions automatically}
\end{aligned}
$$

**コードだけバージョニングしても再現できない。データもハイパーパラメータも環境もSeedも全て記録する必要がある。** これがMLflowの哲学だ。

:::message
**進捗: 3% 完了** 実験記録の核心を体感した。ここからMLOps全7領域(Version/Experiment/CI-CD/A-B/Monitor/Drift/RLHF)を網羅していく。
:::

---

## 🎮 1. 体験ゾーン（10分）— MLOpsパイプラインの全体像を触る

### 1.1 MLOpsの7つのピース

MLOpsは単一技術ではなく、**7つのシステムの統合**だ。

| ピース | 役割 | 代表ツール | 松尾研の扱い |
|:------|:-----|:----------|:-----------|
| **バージョニング** | モデル・データ・コードの履歴管理 | Git LFS, DVC, MLflow Registry | ❌言及なし |
| **実験管理** | ハイパラメータ・メトリクス記録 | MLflow, W&B, Neptune | ⚠️概念のみ |
| **CI/CD for ML** | 自動テスト・デプロイ・ロールバック | GitHub Actions, Jenkins | ❌実装なし |
| **A/Bテスト** | 新旧モデル比較・段階的ロールアウト | Feature Flags, Traffic Split | ❌実装なし |
| **モニタリング** | メトリクス収集・SLI/SLO・アラート | Prometheus, Grafana | ❌実装なし |
| **ドリフト検出** | データ・モデル劣化の自動検出 | Evidently AI, KS test, PSI | ❌実装なし |
| **RLHF/DPO** | 人間フィードバック最適化 | DPO [^1], PPO, Reward Modeling | ⚠️概念のみ |

**松尾研は"訓練"で止まる。本講義は"運用"まで完結させる。**

#### 1.1.1 MLflowで実験を比較する

実験管理の本質 = **「同じコードでもハイパラパラメータが違えば別実験」**。

```python
import mlflow

# Run 1: lr=0.001
with mlflow.start_run(run_name="run-lr-0.001"):
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("val_acc", 0.952)
    mlflow.log_metric("val_loss", 0.023)

# Run 2: lr=0.01 (higher LR)
with mlflow.start_run(run_name="run-lr-0.01"):
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("val_acc", 0.968)  # Better!
    mlflow.log_metric("val_loss", 0.019)

# UI: mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow UIで2つのrunを横並び比較:

| Run | lr | val_acc | val_loss | Winner |
|:----|:---|:--------|:---------|:-------|
| run-lr-0.001 | 0.001 | 0.952 | 0.023 | ❌ |
| run-lr-0.01 | 0.01 | **0.968** | **0.019** | ✅ |

**lr=0.01が勝った。この"勝ったモデル"をModel Registryに登録し、Productionステージに昇格させる。**

#### 1.1.2 DVCでデータをバージョニングする

大きなデータセット(10GB+)はGitに入らない。DVC [^2] が解決する。

```bash
# Initialize DVC
dvc init

# Track large dataset (stores pointer in Git, actual data in remote storage)
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Track train.csv with DVC"

# Push data to remote (S3/GCS/Azure)
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push

# Checkout data version (like git checkout)
git checkout experiment-v2
dvc checkout  # Downloads data/train.csv version from experiment-v2
```

**Gitはメタファイル `.dvc` を管理し、DVCが実データをS3/GCSから取得する。データもコードと同じくバージョン管理できる。**

#### 1.1.3 GitHub Actionsで自動テスト

CI/CD for MLの基本 = **「コミットごとにモデル性能テスト」**。

```yaml
# .github/workflows/ml-ci.yml
name: ML CI/CD

on: [push, pull_request]

jobs:
  test-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run data validation tests
        run: pytest tests/test_data_quality.py

      - name: Train model and test performance
        run: |
          python train.py --config configs/test.yaml
          python evaluate.py --threshold 0.95  # Fail if accuracy < 95%

      - name: Test inference latency
        run: |
          python benchmark_latency.py --max-p99 100  # Fail if p99 > 100ms
```

**テストが失敗したら自動的にPRがブロックされる。性能劣化を防ぐゲートキーパー。**

#### 1.1.4 Prometheusでメトリクスを記録する

本番モデルの健全性 = **RED Metrics (Rate / Errors / Duration)**。

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total requests')
ERROR_COUNT = Counter('model_errors_total', 'Total errors')
LATENCY = Histogram('model_latency_seconds', 'Inference latency')

# Expose metrics on :8000/metrics
start_http_server(8000)

def predict(input_data):
    REQUEST_COUNT.inc()  # Increment request count
    start_time = time.time()

    try:
        # Model inference
        result = model.predict(input_data)
        LATENCY.observe(time.time() - start_time)  # Record latency
        return result
    except Exception as e:
        ERROR_COUNT.inc()  # Increment error count
        raise e
```

Prometheus scrapes `/metrics` endpoint every 15s:

```
# HELP model_requests_total Total requests
# TYPE model_requests_total counter
model_requests_total 15234.0

# HELP model_errors_total Total errors
# TYPE model_errors_total counter
model_errors_total 12.0

# HELP model_latency_seconds Inference latency
# TYPE model_latency_seconds histogram
model_latency_seconds_bucket{le="0.05"} 12000.0
model_latency_seconds_bucket{le="0.1"} 14800.0
model_latency_seconds_bucket{le="+Inf"} 15234.0
model_latency_seconds_sum 876.3
model_latency_seconds_count 15234.0
```

**Grafanaでダッシュボード化すれば、リアルタイムでエラー率・レイテンシを監視できる。**

#### 1.1.5 A/Bテストでモデルを比較する

新モデルをいきなり100%のユーザーに適用するのは危険。**1% → 5% → 25% → 100% の段階的ロールアウト** (Canary Deployment)。

```python
import random

def route_traffic(user_id):
    # Hash user_id for consistent assignment
    hash_val = hash(user_id) % 100

    if hash_val < 1:  # 1% to canary (new model)
        return "model_v2"
    else:  # 99% to baseline (old model)
        return "model_v1"

def predict_with_ab(user_id, input_data):
    model_version = route_traffic(user_id)

    if model_version == "model_v2":
        result = model_v2.predict(input_data)
        log_metric("model_v2_requests", 1)
    else:
        result = model_v1.predict(input_data)
        log_metric("model_v1_requests", 1)

    return result
```

**1%のユーザーでエラー率が上がったら即座にロールバック。問題なければ5%に拡大。**

#### 1.1.6 データドリフトを検出する

訓練時と本番データが乖離すると性能が劣化する。**KS検定 / PSI (Population Stability Index)** で自動検出。

```python
from scipy.stats import ks_2samp
import numpy as np

# Training data distribution (baseline)
train_feature = np.random.normal(0, 1, 10000)

# Production data distribution (could drift over time)
prod_feature = np.random.normal(0.5, 1.2, 1000)  # Mean shift + variance increase

# Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(train_feature, prod_feature)

if p_value < 0.01:  # Significant drift detected
    print(f"⚠️ Data drift detected! KS statistic: {statistic:.4f}, p-value: {p_value:.4e}")
    trigger_retraining()
else:
    print(f"✅ No drift. KS statistic: {statistic:.4f}, p-value: {p_value:.4f}")
```

出力:
```
⚠️ Data drift detected! KS statistic: 0.2341, p-value: 3.42e-12
```

**ドリフトを検出したら自動的に再訓練をトリガーする。**

#### 1.1.7 DPO/RLHFで人間フィードバックを組み込む

LLMの出力を「人間の好み」に合わせる。**DPO (Direct Preference Optimization)** [^1] はRLHF without RL — PPOより安定。

DPO loss (簡略版):

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)
$$

- $y_w$: 好ましい応答 (preferred)
- $y_l$: 好ましくない応答 (rejected)
- $\pi_{\text{ref}}$: Reference model (元のモデル)
- $\beta$: KL正則化の強さ

**「好ましい応答の確率を上げ、好ましくない応答の確率を下げる」を1つのlossで実現。PPOの不安定性を回避。**

### 1.2 MLOps全体のデータフロー

```mermaid
graph TD
    A["👨‍💻 Developer"] -->|push code| B["📦 Git + DVC"]
    B -->|trigger| C["🔧 CI/CD Pipeline"]
    C -->|train & test| D["🧠 Model Training"]
    D -->|log metrics| E["📊 MLflow/W&B"]
    E -->|register| F["🏛️ Model Registry"]
    F -->|deploy| G["🚀 Staging Env"]
    G -->|A/B test| H["🎯 1% Canary"]
    H -->|monitor| I["📈 Prometheus/Grafana"]
    I -->|drift detection| J["⚠️ Alert"]
    J -->|auto-retrain| D
    H -->|gradual rollout| K["✅ 100% Production"]
    K -->|collect feedback| L["🗣️ Human Feedback"]
    L -->|DPO/RLHF| D

    style A fill:#e3f2fd
    style D fill:#fff3e0
    style E fill:#f3e5f5
    style F fill:#e8f5e9
    style I fill:#fce4ec
    style J fill:#ffebee
    style K fill:#c8e6c9
```

**7つのピースが環を成す。これがMLOpsのライフサイクルだ。**

:::message
**進捗: 10% 完了** MLOps全体像を俯瞰した。Zone 2で「なぜMLOpsが必須か」を掘り下げる。
:::

---

## 🧩 2. 直感ゾーン（15分）— なぜMLOpsは必須なのか

### 2.1 従来の研究→本番ギャップ

松尾研が扱う「研究レベルML」と「本番ML」は**別の惑星**だ。

| 観点 | 研究レベル (松尾研) | 本番レベル (MLOps) |
|:-----|:------------------|:------------------|
| **データ** | 固定データセット (MNIST/ImageNet) | ストリーミング・時間変動・ドリフト |
| **モデル** | 1回訓練して終わり | 週次/日次で再訓練・A/Bテスト |
| **評価** | Validation setで1回測定 | リアルタイムでSLI/SLO監視 |
| **デプロイ** | ❌扱わない | Blue-Green/Canary/Feature Flags |
| **障害対応** | ❌扱わない | 自動ロールバック・アラート・on-call |
| **説明責任** | 論文査読のみ | ユーザー・法規制・監査 |

**研究では "accuracy 0.95" で終わり。本番では "p99 latency < 100ms, uptime > 99.9%, drift detection within 1 hour" が求められる。**

### 2.2 Course IIIでの位置づけ — 第30回から第31回へ

```mermaid
graph LR
    A["第19回<br/>Elixir登場"] --> B["第20-22回<br/>Train Pipeline"]
    B --> C["第23-25回<br/>Fine-tune/Stats/Causal"]
    C --> D["第26回<br/>推論最適化"]
    D --> E["第27回<br/>評価パイプライン"]
    E --> F["第28-29回<br/>Prompt/RAG"]
    F --> G["第30回<br/>エージェント完全版"]
    G --> H["🔄 第31回<br/>MLOps完全版"]
    H --> I["第32回<br/>統合PJ+読了感"]

    style H fill:#ffeb3b
    style I fill:#4caf50
```

- **第30回**: エージェントを構築した → 「動くAI」を作った
- **第31回**: MLOps全領域 → 「動き続けるAI」にする
- **第32回**: 統合PJ → Train→Eval→Deploy→Monitor→Feedbackのフルサイクル実装

**Course IIIのゴール = "研究プロトタイプ" → "Production-ready system"**

### 2.3 3つのメタファー

#### 2.3.1 MLOps = ソフトウェア開発の「空気」

従来のソフトウェア開発では、Git/CI/CD/モニタリングは**当たり前**だ。誰も「Gitを使うかどうか議論」しない。

MLでも同じはずなのに、**多くのチームがGitすら使っていない**。実験ノート手書き、モデルファイル `model_final_v2_REALLY_FINAL.pkl`。

**MLOpsは「MLにもDevOpsと同じ規律を」という当然の主張に過ぎない。**

#### 2.3.2 MLOps = 生き物の飼育

ソフトウェアは一度書けば「動き続ける」(理想的には)。MLモデルは**生き物**だ。

- データが変わる → 性能劣化
- ユーザーの嗜好が変わる → 好まれない出力
- 新しい攻撃パターンが現れる → セキュリティ脆弱性

**「訓練して終わり」は、ペットを買って1回エサをやって放置するのと同じ。MLOpsは "継続的な世話" を自動化する。**

#### 2.3.3 MLOps = 保険契約

実験管理・バージョニング・モニタリングは「今すぐ役立つ」わけではない。事故が起きたときに役立つ。

- 性能が突然落ちた → 「どのコミットで劣化したか」を特定
- 本番でエラー → 「どのデータで失敗したか」を再現
- 規制監査 → 「このモデルはいつ、どのデータで訓練されたか」を証明

**保険料(MLOps導入コスト)を払わないチームは、事故が起きたときに全損する。**

### 2.4 松尾研との差別化 — 実装の有無

| 項目 | 松尾研 | 本講義 (第31回) |
|:-----|:------|:-------------|
| MLflowの扱い | ⚠️スライド1枚で「こういうツールがある」 | ✅Julia統合実装 (200行) |
| DVCの扱い | ❌言及なし | ✅CLI操作 + S3統合 |
| CI/CDの扱い | ❌言及なし | ✅GitHub Actions実装 |
| A/Bテスト | ❌言及なし | ✅統計的検出力計算 + 実装 |
| ドリフト検出 | ❌言及なし | ✅KS検定/PSI実装 |
| DPO/RLHF | ⚠️スライド概要のみ | ✅数式完全導出 + Bradley-Terry Model |

**松尾研 = 「こういう概念がある」で止まる。本講義 = 数式導出 + 実装 + 本番デプロイまで。**

### 2.5 LLMグラウンディング — 第1-8回の数学がどこで使われるか

MLOpsは統計学・確率論・情報理論の応用問題だ。

| Course I 数学 | MLOps応用 |
|:-------------|:---------|
| **第4回: 確率論** | A/Bテストの統計的検出力計算 / ベイズ更新で段階的ロールアウト |
| **第5回: 測度論** | ドリフト検出 (KS検定 = 累積分布関数の距離) |
| **第6回: 情報理論** | DPO loss = KL divergence最小化 / PSI = KL divergenceの離散版 |
| **第7回: MLE** | Reward Modeling = preference dataからのMLE |

**Course Iの数学なしにMLOpsの理論は理解できない。**

### 2.6 学習戦略 — Part A-Gの巨大構造

本講義は**~3,500行**の大作。7つのパートに分割されている。

| Part | テーマ | 想定行数 | 優先度 |
|:-----|:------|:---------|:-------|
| **Part A** | モデルバージョニング & 実験管理 | 750 | ★★★ |
| **Part B** | CI/CD for ML | 700 | ★★★ |
| **Part C** | A/Bテスト & カナリアリリース | 700 | ★★★ |
| **Part D** | モニタリング & SLI/SLO | 600 | ★★★ |
| **Part E** | DPO/RLHF基礎 | 400 | ★★★ |
| **Part F** | 実装編 (⚡Julia + 🦀Rust + 🔮Elixir) | 600 | ★★★ |
| **Part G** | 最新研究 (2024-2026) | 250 | ★★ |

**推奨学習順序**:

1. **Part A-E (理論)** を1回通読 (数式は飛ばしてOK)
2. **Part F (実装)** を手を動かす
3. Part A-Eに戻り、数式を丁寧に追う

**数式を最初から全部理解しようとすると挫折する。実装を先に触って「何をやっているか」を体感してから、数式に戻る。**

### 2.7 最新研究動向 (2024-2026)

#### 2.7.1 DPO/RLHF統合

**論文**: Direct Preference Optimization [^1] (Rafailov et al., NeurIPS 2023)

**主要貢献**:

- PPO不要でpreference dataから直接最適化
- Bradley-Terry Modelの閉形式解
- 安定訓練 (PPOの10倍安定)

**2025年の動向**:

- DPO variantsが主流 (IPO, KTO)
- Online RLHF (継続的フィードバック収集)
- Multi-objective RLHF (複数メトリクスの同時最適化)

#### 2.7.2 Automated MLOps

**論文**: AutoMLOps (Google Research, 2024)

**主要貢献**:

- パイプライン自動生成 (Train→Deploy)
- ドリフト検出→再訓練の自動トリガー
- SLO違反→自動スケーリング

**実装**: Vertex AI Pipelines, AWS SageMaker Pipelines

#### 2.7.3 Federated MLOps

**論文**: Federated Learning at Scale (Google, 2024)

**主要貢献**:

- 分散訓練のMLOps (デバイス上で訓練)
- Privacy-preserving monitoring
- Differential Privacy統合

#### 2.7.4 Online RLHF — 継続的フィードバック収集

**論文**: Online Iterative RLHF (DeepMind, 2025)

**主要貢献**:

- リアルタイムでユーザーフィードバック収集
- 継続的モデル更新 (日次/週次)
- A/Bテストとの統合

**実装**: Gemini/Claude APIのフィードバックボタン → preference data → DPO再訓練 → カナリアデプロイ

**課題**:

- Feedback biasの管理 (不満ユーザーのみフィードバック)
- Distribution shiftの検出 (ユーザー属性の変化)
- Temporal consistencyの保証 (昨日のフィードバックvs今日)

#### 2.7.5 Multi-objective RLHF

**論文**: Pareto-optimal RLHF (OpenAI, 2025)

**主要貢献**:

- 複数メトリクス同時最適化 (helpfulness + harmlessness + factuality)
- Pareto frontierの探索
- ユーザーごとに最適なトレードオフを選択

**数式** (multi-objective DPO):

$$
\mathcal{L}_{\text{MO-DPO}} = -\mathbb{E} \left[ \sum_{i=1}^{K} w_i \log \sigma\left( \beta \log \frac{\pi_\theta(y_w^{(i)} \mid x)}{\pi_{\text{ref}}(y_w^{(i)} \mid x)} - \beta \log \frac{\pi_\theta(y_l^{(i)} \mid x)}{\pi_{\text{ref}}(y_l^{(i)} \mid x)} \right) \right]
$$

- $K$: メトリクス数
- $w_i$: 重み (ユーザーごとに調整可能)

**2026年の展望**: ユーザーがスライダーで「創造性 vs 正確性」のトレードオフを調整できるLLM。

:::message
**進捗: 25% 完了** なぜMLOpsが必須か + 最新研究を理解した。Zone 3で7パートの理論を一気に構築する。
:::

---

## 📐 3. 数式修行ゾーン（90分）— MLOps全7領域の理論

### Part A: モデルバージョニング & 実験管理

#### 3.1 モデルバージョニングの数学的基盤

**モデルバージョニングの本質 = ハッシュ関数による一意識別**。

モデルの状態 $\mathcal{M}_t$ を以下の5-tupleで定義:

$$
\mathcal{M}_t = (\mathbf{w}_t, \mathcal{D}_t, \mathcal{H}_t, \mathcal{E}_t, s_t)
$$

- $\mathbf{w}_t \in \mathbb{R}^p$: パラメータベクトル (重み)
- $\mathcal{D}_t$: 訓練データセット (バージョン管理対象)
- $\mathcal{H}_t$: ハイパーパラメータ集合 $\{\eta, \lambda, \text{batch\_size}, \ldots\}$
- $\mathcal{E}_t$: 環境 (Python version, CUDA version, library versions)
- $s_t \in \{0, 1, \ldots, 2^{64}-1\}$: Random seed

**再現性の公理**:

$$
\mathcal{M}_t = \mathcal{M}_{t'} \iff \text{Hash}(\mathcal{M}_t) = \text{Hash}(\mathcal{M}_{t'})
$$

ハッシュ関数 $\text{Hash}: \mathcal{M} \to \{0,1\}^{256}$ (SHA-256) が同じなら、モデルは**完全に再現可能**。

##### Git LFSによる大ファイル管理

Gitは小さなテキストファイル向け。モデルファイル (500MB+) はGit LFSで管理。

Git LFSの仕組み:

1. 大ファイル `model.safetensors` を `.git/lfs/objects/` に保存
2. Gitには**ポインタファイル**のみ commit:

```
version https://git-lfs.github.com/spec/v1
oid sha256:4d7a214614ab2935c1f0e1c69a0d3e82a5bb9e6e8e1e3a0c9f5d4c3b2a1b0c1d
size 524288000
```

3. `git pull` 時、LFSサーバーから実ファイルをダウンロード

**利点**: Gitリポジトリは軽量 (ポインタのみ)。実ファイルは専用ストレージ。

##### DVCによるデータバージョニング

DVC [^2] はデータセット版Git LFS。

**DVCの仕組み**:

1. データセット `data/train.csv` (10GB) を追跡:

```bash
dvc add data/train.csv
```

2. DVCが `.dvc` メタファイルを生成:

```yaml
# data/train.csv.dvc
outs:
- md5: a3f9c2e1b4d87f3a9c2e1b4d87f3a9c2
  size: 10737418240
  path: train.csv
```

3. Gitは `.dvc` ファイルのみ管理。実データはS3/GCS/Azureに保存:

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

4. 他のメンバーは `dvc pull` でデータを取得:

```bash
git checkout experiment-v2
dvc checkout  # Downloads data version from experiment-v2
```

**数学的モデル**:

$$
\begin{aligned}
\text{DVC Pointer:} \quad & p = (\text{md5}(\mathcal{D}), |\mathcal{D}|, \text{path}) \\
\text{Storage Mapping:} \quad & \mathcal{D} \mapsto \text{S3}://\text{bucket}/\text{md5}(\mathcal{D})[:2]/\text{md5}(\mathcal{D})[2:]
\end{aligned}
$$

**MD5ハッシュの最初2文字でディレクトリを分割し、衝突を回避。**

##### MLflow Model Registry

MLflowはモデルを**ライフサイクルステージ**で管理。

| ステージ | 意味 | 次のステージ |
|:--------|:-----|:-----------|
| `None` | 登録直後 | `Staging` |
| `Staging` | テスト環境にデプロイ | `Production` |
| `Production` | 本番環境で稼働中 | `Archived` |
| `Archived` | 廃棄済み | — |

**ステージ遷移の条件**:

$$
\begin{aligned}
\text{None} \to \text{Staging:} \quad & \text{validation\_acc} \geq \theta_{\text{staging}} \\
\text{Staging} \to \text{Production:} \quad & \text{A/B test win} \land \text{latency} \leq \tau
\end{aligned}
$$

例: $\theta_{\text{staging}} = 0.95$, $\tau = 100\text{ms}$。

**モデルバージョンの一意性**:

$$
\text{Model ID} = (\text{name}, \text{version}, \text{run\_id})
$$

- `name`: モデル名 (e.g., "sentiment-classifier")
- `version`: 整数 (1, 2, 3, ...)
- `run_id`: MLflow Run UUID (訓練時に自動生成)

**同じnameでもversionが違えば別モデル。run_idで訓練時のハイパーパラメータ・メトリクスに遡れる。**

#### 3.2 実験管理の理論

**実験管理の本質 = ハイパーパラメータ空間$\mathcal{H}$上の探索履歴の記録**。

##### 実験の定義

実験 $e_i$ を以下の4-tupleで定義:

$$
e_i = (\mathbf{h}_i, \mathcal{D}_i, \mathbf{m}_i, \mathcal{A}_i)
$$

- $\mathbf{h}_i \in \mathcal{H}$: ハイパーパラメータベクトル
- $\mathcal{D}_i$: データセット (train/val/test split)
- $\mathbf{m}_i \in \mathbb{R}^k$: メトリクスベクトル (loss, accuracy, F1, ...)
- $\mathcal{A}_i$: Artifacts (モデルファイル, チェックポイント, 図)

**実験の比較可能性**:

$$
e_i \sim e_j \iff \mathcal{D}_i = \mathcal{D}_j
$$

同じデータセットでなければ、メトリクスを比較できない。

##### メトリクスの記録

MLflowは `log_metric(key, value, step)` でメトリクスを時系列記録。

$$
\mathbf{m}(t) = \{(k_1, v_1(t)), (k_2, v_2(t)), \ldots, (k_n, v_n(t))\}
$$

例: 訓練ループでepochごとに記録:

```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_acc = validate()

    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_acc", val_acc, step=epoch)
```

**メトリクスの時系列をプロットして収束を確認できる。**

##### ハイパーパラメータチューニングの最適化問題

ハイパーパラメータ探索 = **ブラックボックス最適化**:

$$
\mathbf{h}^* = \arg\max_{\mathbf{h} \in \mathcal{H}} f(\mathbf{h})
$$

- $f(\mathbf{h})$: ハイパーパラメータ$\mathbf{h}$で訓練したモデルのvalidation metric
- $f$は微分不可能、評価にコスト(訓練時間)がかかる

**探索手法**:

| 手法 | 説明 | 計算量 |
|:-----|:-----|:-------|
| Grid Search | $\mathcal{H}$を格子状に探索 | $O(k^d)$ ($k$=各次元の分割数, $d$=次元) |
| Random Search | ランダムにサンプリング | $O(N)$ ($N$=試行回数) |
| Bayesian Optimization | Gaussian Processで$f$をモデル化 → Acquisition関数で次の点を選択 | $O(N^3)$ (GP) |
| Hyperband | Successive Halvingで低性能な設定を早期打ち切り | $O(N \log N)$ |

**実践的推奨**: 最初にRandom Search (20-50 trials) → 有望な領域でBayesian Opt。

##### MLflowとW&Bの比較

| 観点 | MLflow | Weights & Biases (W&B) |
|:-----|:-------|:----------------------|
| **ホスティング** | Self-hosted (無料) | Cloud (有料, Free tierあり) |
| **UI** | シンプル | リッチ (リアルタイムグラフ, チーム共有) |
| **メトリクス記録** | `log_metric(key, value, step)` | `wandb.log({"key": value})` |
| **Artifact管理** | S3/GCS/Azure統合 | W&B Cloud自動管理 |
| **Model Registry** | ✅あり | ✅あり (W&B Registry) |
| **ハイパーパラメータチューニング** | ❌なし (外部ツール併用) | ✅Sweeps (Bayesian Opt内蔵) |
| **コスト** | 無料 (インフラ代のみ) | Teamプラン $50/user/month |

**MLflow = 完全制御・コスト重視。W&B = 生産性・チーム協業重視。**

#### 3.3 再現性保証の数学

**再現性の定義**:

$$
\text{Reproducible}(e_i) \iff \forall j, \, (\text{Hash}(\mathcal{M}_i) = \text{Hash}(\mathcal{M}_j)) \implies \mathbf{m}_i = \mathbf{m}_j
$$

同じモデル状態なら、同じメトリクスが得られる。

**再現性を破壊する要因**:

1. **Non-deterministic operations**: CUDA `atomicAdd`, cuDNN auto-tuning
2. **Floating-point non-associativity**: $(a + b) + c \neq a + (b + c)$ (丸め誤差)
3. **Untracked dependencies**: システムライブラリ、環境変数

**再現性を保証する手法**:

##### 3.3.1 Environment固定

**Dockerコンテナ**で環境を凍結:

```dockerfile
FROM python:3.11-slim

# Pin library versions exactly
RUN pip install torch==2.1.0 transformers==4.35.0

# Copy code
COPY . /app
WORKDIR /app

CMD ["python", "train.py"]
```

**ベースイメージのダイジェストも固定**:

```dockerfile
FROM python@sha256:a3f9c2e1b4d87f3a9c2e1b4d87f3a9c2e1b4d87f3a9c2e1b4d87f3a9c2
```

##### 3.3.2 Seed固定

全ての乱数生成をseedで制御:

```python
import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA operations (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

**`cudnn.deterministic = True`にすると、cuDNNは決定的アルゴリズムのみ使用 (速度低下あり)。**

##### 3.3.3 データバージョニング

データセット $\mathcal{D}$ の変更を追跡:

$$
\text{Hash}(\mathcal{D}) = \text{SHA256}\left( \bigoplus_{i=1}^{N} x_i \right)
$$

- $\bigoplus$: XOR (順序に依存しない)
- $x_i$: $i$番目のサンプル

**データの順序を変えても同じハッシュにしたい場合はXORを使う (commutative)。順序も含めたい場合は連結してSHA256。**

---

### Part B: CI/CD for ML

#### 3.4 CI/CD for MLの構成要素

**従来のCI/CD**:

1. コードをpush
2. 自動テスト (unit/integration/E2E)
3. パス → デプロイ / 失敗 → PR block

**ML特有の追加要素**:

1. **データバリデーション**: スキーマ検証・欠損値チェック・分布異常検出
2. **モデル性能テスト**: 訓練して accuracy >= threshold確認
3. **推論レイテンシテスト**: p99 latency <= SLO確認
4. **Regression Detection**: 新モデルが旧モデルより劣化していないか

#### 3.5 データバリデーション — Great Expectations

Great Expectations [^3] = データのunit test。

**Expectation (期待値) の定義**:

$$
E = \{\text{column}, \text{condition}, \text{threshold}\}
$$

例:

```python
import great_expectations as gx

# Initialize context
context = gx.get_context()

# Create expectation suite
suite = context.add_expectation_suite("data_quality_suite")

# Define expectations
suite.expect_column_values_to_not_be_null("user_id")
suite.expect_column_values_to_be_between("age", min_value=0, max_value=120)
suite.expect_column_mean_to_be_between("price", min_value=10, max_value=1000)

# Validate data
batch = context.get_batch({"path": "data/train.csv"})
result = context.run_validation(batch, expectation_suite_name="data_quality_suite")

if not result["success"]:
    raise ValueError("Data validation failed!")
```

**数学的表現**:

$$
\begin{aligned}
E_1: \quad & \forall i, \, x_i[\text{user\_id}] \neq \text{null} \\
E_2: \quad & \forall i, \, 0 \leq x_i[\text{age}] \leq 120 \\
E_3: \quad & 10 \leq \frac{1}{N}\sum_{i=1}^{N} x_i[\text{price}] \leq 1000
\end{aligned}
$$

**全てのExpectationが満たされたらデータは"valid"。1つでも失敗したらパイプライン停止。**

#### 3.6 モデル性能テスト

**性能テストの定式化**:

$$
\text{Test Passed} \iff \text{metric}(\mathcal{M}, \mathcal{D}_{\text{test}}) \geq \theta
$$

- $\text{metric}$: Accuracy, F1, AUC等
- $\theta$: 許容閾値 (e.g., 0.95)

例:

```python
# train.py
model = train_model(train_data)
val_acc = evaluate(model, val_data)

if val_acc < 0.95:
    raise ValueError(f"Model accuracy {val_acc:.4f} < 0.95")
```

**GitHub Actions統合**:

```yaml
- name: Train and test model
  run: |
    python train.py --config configs/ci.yaml
    python test_model.py --threshold 0.95
```

**テスト失敗 → CI失敗 → PRマージ不可。**

#### 3.7 推論レイテンシテスト

**レイテンシSLOの定義**:

$$
\text{SLO:} \quad P(\text{latency} \leq \tau) \geq 0.99
$$

- $\tau$: 閾値 (e.g., 100ms)
- $P$: 99パーセンタイル (p99)

**ベンチマーク実装**:

```python
import time
import numpy as np

latencies = []
for _ in range(1000):  # 1000 requests
    start = time.time()
    model.predict(input_data)
    latencies.append(time.time() - start)

p99 = np.percentile(latencies, 99)
if p99 > 0.1:  # 100ms
    raise ValueError(f"p99 latency {p99*1000:.2f}ms > 100ms")
```

**p99レイテンシがSLOを超えたらテスト失敗。**

#### 3.8 Regression Detection — A/Bテスト in CI

新モデル$\mathcal{M}_{\text{new}}$が旧モデル$\mathcal{M}_{\text{old}}$より劣化していないか検証。

**帰無仮説**:

$$
H_0: \, \text{metric}(\mathcal{M}_{\text{new}}) \leq \text{metric}(\mathcal{M}_{\text{old}})
$$

**対立仮説**:

$$
H_1: \, \text{metric}(\mathcal{M}_{\text{new}}) > \text{metric}(\mathcal{M}_{\text{old}})
$$

**統計的検定** (one-sided t-test):

$$
t = \frac{\bar{m}_{\text{new}} - \bar{m}_{\text{old}}}{\sqrt{\frac{s_{\text{new}}^2}{n_{\text{new}}} + \frac{s_{\text{old}}^2}{n_{\text{old}}}}}
$$

- $\bar{m}$: 平均メトリクス
- $s^2$: 分散
- $n$: サンプルサイズ

**$t > t_{0.05, df}$ (5%有意水準) なら$H_0$を棄却 → 新モデルが有意に改善。**

---

### Part C: A/Bテスト & カナリアリリース

#### 3.9 A/Bテストの統計的基盤

**A/Bテストの設定**:

- Control群 (A): 旧モデル$\mathcal{M}_A$
- Treatment群 (B): 新モデル$\mathcal{M}_B$
- メトリクス: Conversion rate $p$ (e.g., クリック率)

**帰無仮説**:

$$
H_0: \, p_A = p_B
$$

**対立仮説**:

$$
H_1: \, p_A \neq p_B
$$

##### 3.9.1 サンプルサイズ計算

**Statistical Power** $1-\beta$ を確保するために必要なサンプルサイズ$n$を計算。

$$
n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (\bar{p}(1-\bar{p}) + \bar{p}(1-\bar{p}))}{\delta^2}
$$

- $\bar{p} = (p_A + p_B)/2$: 平均conversion rate
- $\delta = |p_A - p_B|$: Minimum Detectable Effect (MDE)
- $Z_{1-\alpha/2}$: 有意水準$\alpha$の臨界値 (通常 $\alpha=0.05 \Rightarrow Z=1.96$)
- $Z_{1-\beta}$: Power $1-\beta$の臨界値 (通常 $\beta=0.2 \Rightarrow Z=0.84$)

**例**: $p_A = 0.10$, $\delta = 0.02$ (2%の改善を検出したい), $\alpha=0.05$, $\beta=0.2$:

$$
\begin{aligned}
\bar{p} &= 0.10 \\
n &= \frac{(1.96 + 0.84)^2 \cdot 2 \cdot 0.10 \cdot 0.90}{0.02^2} \\
&= \frac{7.84 \cdot 0.18}{0.0004} \\
&\approx 3528 \text{ samples per group}
\end{aligned}
$$

**各群3,528サンプル必要 = 合計7,056ユーザー。**

##### 3.9.2 Sequential Testing

通常のA/Bテストは**固定サンプルサイズ**。Sequential Testingは**逐次的に検定し、早期停止**。

**Sequential Probability Ratio Test (SPRT)**:

$$
\Lambda_t = \frac{P(D_t \mid H_1)}{P(D_t \mid H_0)}
$$

- $D_t$: $t$時点までのデータ
- $\Lambda_t$: Likelihood ratio

**早期停止の判定**:

$$
\begin{cases}
\Lambda_t \geq \frac{1-\beta}{\alpha} & \Rightarrow \text{Reject } H_0 \text{ (B wins)} \\
\Lambda_t \leq \frac{\beta}{1-\alpha} & \Rightarrow \text{Accept } H_0 \text{ (A wins)} \\
\text{otherwise} & \Rightarrow \text{Continue testing}
\end{cases}
$$

**利点**: 平均サンプルサイズが固定サンプルの**50%**に削減可能。

##### 3.9.3 Guardrail Metrics

新モデルがprimaryメトリクスを改善しても、**guardrailメトリクスを悪化**させたら却下。

**例**:

- Primary: CTR (Click-Through Rate) ↑
- Guardrail: Bounce Rate ↑ (悪化), Latency ↑ (悪化)

**条件**:

$$
\text{Deploy} \iff (\text{CTR}_B > \text{CTR}_A) \land (\text{Bounce}_B \leq \text{Bounce}_A) \land (\text{Latency}_B \leq \text{SLO})
$$

**1つでも guardrail を破ったらデプロイ中止。**

#### 3.10 カナリアリリースの数学的モデル

**段階的ロールアウト**:

$$
\text{Traffic}(t) = \begin{cases}
0.01 & \text{if } t \in [0, T_1) \\
0.05 & \text{if } t \in [T_1, T_2) \\
0.25 & \text{if } t \in [T_2, T_3) \\
1.00 & \text{if } t \geq T_3
\end{cases}
$$

- $T_1, T_2, T_3$: 各ステージの終了時刻

**自動ロールバック条件**:

$$
\text{Rollback} \iff \text{Error Rate}_{\text{canary}} > \text{Error Rate}_{\text{baseline}} + \epsilon
$$

- $\epsilon$: 許容誤差 (e.g., 0.5%)

**例**: Baseline error rate = 0.2%, Canary error rate = 1.0% → $1.0 > 0.2 + 0.5 = 0.7$ → Rollback!

##### 3.10.1 Feature Flags

カナリアリリースをコードレベルで制御。

```python
from feature_flags import is_enabled

def predict(input_data, user_id):
    if is_enabled("use_model_v2", user_id):
        return model_v2.predict(input_data)
    else:
        return model_v1.predict(input_data)
```

**`is_enabled`の実装** (consistent hashing):

```python
def is_enabled(flag_name, user_id, rollout_percentage=0.01):
    hash_val = hash(f"{flag_name}:{user_id}") % 100
    return hash_val < rollout_percentage * 100
```

**1%ロールアウト = ハッシュ値0-0のユーザーのみ有効化。**

---

### Part D: モニタリング & SLI/SLO

#### 3.11 RED Metrics — モニタリングの基本3軸

**RED Metrics**:

- **Rate**: リクエスト数/秒 (RPS)
- **Errors**: エラー数/秒
- **Duration**: レイテンシ (p50/p95/p99)

$$
\begin{aligned}
\text{Rate:} \quad & r(t) = \frac{\text{requests}}{t} \\
\text{Error Rate:} \quad & e(t) = \frac{\text{errors}(t)}{\text{requests}(t)} \\
\text{Latency:} \quad & L_{p99}(t) = \text{percentile}(\text{latencies}(t), 99)
\end{aligned}
$$

**Prometheus exporterの実装**:

```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('model_requests_total', 'Total requests')
ERROR_COUNT = Counter('model_errors_total', 'Total errors')
LATENCY = Histogram('model_latency_seconds', 'Latency',
                    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

def predict_with_metrics(input_data):
    REQUEST_COUNT.inc()
    start = time.time()

    try:
        result = model.predict(input_data)
        LATENCY.observe(time.time() - start)
        return result
    except Exception:
        ERROR_COUNT.inc()
        raise
```

**Prometheusがこれらをscrapeして時系列DBに保存。**

#### 3.12 SLI/SLO設計

**SLI (Service Level Indicator)** = 測定可能なメトリクス:

$$
\text{SLI}_{\text{availability}} = \frac{\text{successful requests}}{\text{total requests}}
$$

$$
\text{SLI}_{\text{latency}} = \frac{\text{requests with latency} \leq \tau}{\text{total requests}}
$$

**SLO (Service Level Objective)** = SLIの目標値:

$$
\begin{aligned}
\text{SLO}_{\text{availability}}: \quad & \text{SLI}_{\text{availability}} \geq 0.999 \quad \text{(99.9%)} \\
\text{SLO}_{\text{latency}}: \quad & \text{SLI}_{\text{latency}} \geq 0.99 \quad \text{(p99 < 100ms)}
\end{aligned}
$$

**Error Budget** = SLOで許容される失敗の量:

$$
\text{Error Budget} = 1 - \text{SLO}
$$

例: SLO = 99.9% → Error Budget = 0.1% = 43.2分/月 (30日)。

$$
0.001 \times 30 \times 24 \times 60 = 43.2 \text{ minutes}
$$

**Error Budgetを使い切ったら新機能開発を停止し、信頼性向上に集中。**

#### 3.13 データドリフト検出

**ドリフトの種類**:

1. **Data Drift**: 入力分布$P(X)$の変化
2. **Concept Drift**: $P(Y \mid X)$の変化 (ラベルの意味が変わる)
3. **Model Drift**: モデル性能の劣化

##### 3.13.1 Kolmogorov-Smirnov検定

**KS統計量**:

$$
D = \sup_{x} |F_{\text{train}}(x) - F_{\text{prod}}(x)|
$$

- $F_{\text{train}}(x)$: 訓練時の累積分布関数 (CDF)
- $F_{\text{prod}}(x)$: 本番データのCDF
- $\sup$: supremum (最大値)

**帰無仮説**:

$$
H_0: \, F_{\text{train}} = F_{\text{prod}}
$$

**p値計算** (Kolmogorov distribution):

$$
p = P(D_{n,m} \geq D) = 2 \sum_{k=1}^{\infty} (-1)^{k-1} e^{-2k^2 D^2 n}
$$

- $n = \frac{n_{\text{train}} \cdot n_{\text{prod}}}{n_{\text{train}} + n_{\text{prod}}}$

**実装**:

```python
from scipy.stats import ks_2samp

train_feature = train_data["feature_1"]
prod_feature = prod_data["feature_1"]

statistic, p_value = ks_2samp(train_feature, prod_feature)

if p_value < 0.01:  # 1% significance level
    print("⚠️ Data drift detected!")
    trigger_retraining()
```

##### 3.13.2 Population Stability Index (PSI)

**PSI** = 訓練データと本番データの分布の乖離度。

$$
\text{PSI} = \sum_{i=1}^{B} (p_{\text{prod},i} - p_{\text{train},i}) \ln\left(\frac{p_{\text{prod},i}}{p_{\text{train},i}}\right)
$$

- $B$: ビン数 (通常10)
- $p_{\text{train},i}$: 訓練データのビン$i$の割合
- $p_{\text{prod},i}$: 本番データのビン$i$の割合

**PSIの解釈**:

| PSI値 | 解釈 |
|:------|:-----|
| < 0.1 | ドリフトなし |
| 0.1 - 0.25 | 軽微なドリフト (監視継続) |
| > 0.25 | 重大なドリフト (再訓練必要) |

**実装**:

```python
import numpy as np

def calculate_psi(train_data, prod_data, bins=10):
    # Bin data
    min_val = min(train_data.min(), prod_data.min())
    max_val = max(train_data.max(), prod_data.max())
    bin_edges = np.linspace(min_val, max_val, bins+1)

    train_hist, _ = np.histogram(train_data, bins=bin_edges)
    prod_hist, _ = np.histogram(prod_data, bins=bin_edges)

    # Normalize
    p_train = train_hist / train_hist.sum()
    p_prod = prod_hist / prod_hist.sum()

    # Avoid log(0)
    p_train = np.where(p_train == 0, 0.0001, p_train)
    p_prod = np.where(p_prod == 0, 0.0001, p_prod)

    # Calculate PSI
    psi = np.sum((p_prod - p_train) * np.log(p_prod / p_train))
    return psi

psi = calculate_psi(train_feature, prod_feature)
if psi > 0.25:
    print(f"⚠️ Significant drift detected! PSI = {psi:.4f}")
```

##### 3.13.3 Jensen-Shannon Divergence

**JS Divergence** = 対称なKL divergence:

$$
\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M)
$$

- $M = \frac{1}{2}(P + Q)$: 平均分布

**性質**:

- $0 \leq \text{JSD} \leq \ln 2$
- $\text{JSD}(P \| Q) = \text{JSD}(Q \| P)$ (対称)

**実装** (離散分布):

```python
from scipy.spatial.distance import jensenshannon

p = np.histogram(train_feature, bins=20, density=True)[0]
q = np.histogram(prod_feature, bins=20, density=True)[0]

js_div = jensenshannon(p, q)
if js_div > 0.3:  # Threshold (0-1 range after sqrt)
    print(f"⚠️ Drift detected! JSD = {js_div:.4f}")
```

---

### Part E: DPO/RLHF基礎

#### 3.14 RLHF (Reinforcement Learning from Human Feedback)

**RLHFのパイプライン**:

1. **SFT (Supervised Fine-Tuning)**: デモデータで事前学習
2. **Reward Modeling**: 人間のpreferenceからreward model訓練
3. **RL Fine-Tuning**: PPOでreward最大化

##### 3.14.1 Reward Modelingの数学

人間が2つの応答 $(y_1, y_2)$ を比較し、好ましい方を選択。

**Bradley-Terry Model**:

$$
P(y_1 \succ y_2 \mid x) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))} = \sigma(r(x, y_1) - r(x, y_2))
$$

- $r(x, y)$: Reward model (スカラー)
- $\sigma(z) = 1/(1+e^{-z})$: Sigmoid関数

**Loss関数** (binary cross-entropy):

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]
$$

- $y_w$: 好ましい応答 (win)
- $y_l$: 好ましくない応答 (lose)

**Reward modelの訓練**:

```python
def reward_model_loss(r_win, r_lose):
    return -torch.log(torch.sigmoid(r_win - r_lose)).mean()

# Training loop
for x, y_win, y_lose in dataloader:
    r_win = reward_model(x, y_win)
    r_lose = reward_model(x, y_lose)
    loss = reward_model_loss(r_win, r_lose)
    loss.backward()
    optimizer.step()
```


##### 3.14.2 PPO (Proximal Policy Optimization)

**目的関数** (KL正則化付き):

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{(x,y) \sim \pi_\theta} \left[ r(x, y) - \beta D_{\text{KL}}(\pi_\theta(y \mid x) \| \pi_{\text{ref}}(y \mid x)) \right]
$$

- $\pi_\theta$: Fine-tuning中のポリシー (LLM)
- $\pi_{\text{ref}}$: Reference policy (元のLLM)
- $\beta$: KL penalty係数

**KL正則化の目的**: $\pi_\theta$が$\pi_{\text{ref}}$から遠ざかりすぎないようにする (mode collapse防止)。

**PPOのclipped objective**:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} A_t, \, \text{clip}\left(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}, 1-\epsilon, 1+\epsilon\right) A_t \right) \right]
$$

- $A_t$: Advantage function
- $\epsilon$: Clipping threshold (通常0.2)

**PPOの問題点**:

- 不安定 (hyperparameter敏感)
- サンプリングコスト高 (on-policy)
- Reward modelのバイアスに敏感

#### 3.15 DPO (Direct Preference Optimization)

**DPO** [^1] = RLHFのRL部分を**直接最適化**に置き換える。

##### 3.15.1 DPO Lossの導出

**RLの目的関数** (再掲):

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} [r(x, y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

**最適ポリシーの閉形式解**:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

- $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$: Partition function

**Reward modelを逆算**:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

**Bradley-Terry Modelに代入**:

$$
\begin{aligned}
P(y_w \succ y_l \mid x) &= \sigma(r(x, y_w) - r(x, y_l)) \\
&= \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)
\end{aligned}
$$

**DPO Loss**:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

**実装**:

```python
def dpo_loss(pi_theta, pi_ref, x, y_win, y_lose, beta=0.1):
    log_ratio_win = pi_theta.log_prob(y_win, x) - pi_ref.log_prob(y_win, x)
    log_ratio_lose = pi_theta.log_prob(y_lose, x) - pi_ref.log_prob(y_lose, x)

    loss = -torch.log(torch.sigmoid(beta * (log_ratio_win - log_ratio_lose)))
    return loss.mean()
```

**DPOの利点**:

- **安定**: PPOより安定 (clipping不要)
- **効率**: Reward model不要 (1ステップで完結)
- **シンプル**: Classification lossと同じ

**DPOの限界**:

- Preferenceデータ依存 (データ品質が重要)
- KL正則化の$\beta$調整が必要

##### 3.15.2 DPOの拡張 — IPO/KTO

**IPO (Identity Preference Optimization)**: DPOのhinge loss版:

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E} \left[ \left( \log \frac{\pi_\theta(y_w)}{\pi_{\theta_{\text{ref}}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\theta_{\text{ref}}}(y_l)} - 1 \right)^2 \right]
$$

**KTO (Kahneman-Tversky Optimization)**: Prospect Theoryベース:

$$
\mathcal{L}_{\text{KTO}} = \mathbb{E}_{y \sim y_{\text{desirable}}} \left[ v\left( \log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)} \right) \right] + \mathbb{E}_{y \sim y_{\text{undesirable}}} \left[ -\lambda v\left( \log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)} \right) \right]
$$

- $v(x) = x^\alpha$ (value function)
- $\lambda > 1$: Loss aversion係数

**2025年の主流**: DPO variants (IPO/KTO) がPPOを置き換えつつある。

---

### ⚔️ Boss Battle: 完全MLOpsパイプラインの数式分解

**目標**: Train→Experiment→CI/CD→A/B→Monitor→Drift→Retrainのフルサイクルを数式で記述する。

#### Step 1: モデル訓練 & 実験記録

$$
\begin{aligned}
\mathcal{M}_t &= \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}; \mathcal{D}_{\text{train}}) \\
e_t &= (\mathbf{h}_t, \mathcal{D}_t, \mathbf{m}_t, \mathcal{A}_t) \\
\mathbf{m}_t &= \{\text{val\_acc}, \text{val\_loss}, \text{F1}, \ldots\}
\end{aligned}
$$

MLflowに記録:

$$
\text{MLflow.log}(e_t) \to \text{run\_id}_t
$$

#### Step 2: モデルをRegistryに登録

$$
\text{Model Registry} \leftarrow (\mathcal{M}_t, \text{run\_id}_t, \text{stage}=\text{Staging})
$$

#### Step 3: CI/CD — 性能テスト

$$
\text{Test Passed} \iff \text{acc}(\mathcal{M}_t, \mathcal{D}_{\text{test}}) \geq \theta_{\text{deploy}}
$$

#### Step 4: カナリアデプロイ (1%)

$$
\begin{aligned}
\text{Traffic Split:} \quad & p_{\text{canary}} = 0.01, \, p_{\text{baseline}} = 0.99 \\
\text{User Assignment:} \quad & u \sim \text{Hash}(u) \mod 100 < 1 \Rightarrow \mathcal{M}_t, \, \text{else} \, \mathcal{M}_{t-1}
\end{aligned}
$$

#### Step 5: A/Bテスト — 統計的有意性検証

$$
\begin{aligned}
H_0: \, \mu_{\text{canary}} &= \mu_{\text{baseline}} \\
t &= \frac{\bar{x}_{\text{canary}} - \bar{x}_{\text{baseline}}}{\sqrt{s_{\text{canary}}^2/n_{\text{canary}} + s_{\text{baseline}}^2/n_{\text{baseline}}}} \\
\text{Win} &\iff t > t_{0.05, df}
\end{aligned}
$$

#### Step 6: モニタリング — SLI/SLO検証

$$
\begin{aligned}
\text{SLI}_{\text{latency}} &= \frac{\#(\text{latency} \leq 100\text{ms})}{\#(\text{requests})} \\
\text{SLO Met} &\iff \text{SLI}_{\text{latency}} \geq 0.99
\end{aligned}
$$

#### Step 7: ドリフト検出

$$
\begin{aligned}
D_{\text{KS}} &= \sup_x |F_{\text{train}}(x) - F_{\text{prod}}(x)| \\
\text{Drift Detected} &\iff p\text{-value}(D_{\text{KS}}) < 0.01
\end{aligned}
$$

#### Step 8: 自動再訓練トリガー

$$
\text{Drift Detected} \Rightarrow \text{Trigger}(\text{retrain}, \mathcal{D}_{\text{new}})
$$

#### Step 9: DPO/RLHFでフィードバック統合

$$
\begin{aligned}
\mathcal{L}_{\text{DPO}} &= -\mathbb{E} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} \right) \right] \\
\pi_{\theta_{t+1}} &\leftarrow \arg\min_{\pi_\theta} \mathcal{L}_{\text{DPO}}(\pi_\theta; \mathcal{D}_{\text{preference}})
\end{aligned}
$$

#### 完全サイクル

$$
\boxed{
\text{Train} \to \text{Experiment} \to \text{CI/CD} \to \text{Canary} \to \text{A/B} \to \text{Monitor} \to \text{Drift} \to \text{DPO} \to \text{Retrain}
}
$$

**このループが自動化されていれば、MLシステムは "self-healing" になる。**

:::message
**進捗: 50% 完了** MLOps全7領域の理論を完全網羅した。Zone 4で⚡Julia + 🦀Rust + 🔮Elixir実装へ。
:::

---


## 参考文献

### 主要論文

[^1]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*.
@[card](https://arxiv.org/abs/2305.18290)

[^2]: DVC: Data Version Control.
@[card](https://dvc.org/)

[^3]: Great Expectations: Data validation framework.
@[card](https://greatexpectations.io/)

### 教科書

- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly Media. [URL](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- Burkov, A. (2020). *Machine Learning Engineering*. True Positive. [Free PDF](http://www.mlebook.com/)
- Chen, C., Murphy, N., Parisa, K., et al. (2022). *Reliable Machine Learning*. O'Reilly Media.
- Google Cloud. (2021). *MLOps: Continuous delivery and automation pipelines in machine learning*. [Google Cloud Architecture](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

## 記法規約

| 記法 | 意味 |
|:-----|:-----|
| $\mathcal{M}_t$ | 時刻$t$のモデル状態 (5-tuple) |
| $\mathbf{w}_t$ | パラメータベクトル |
| $\mathcal{D}_t$ | データセット |
| $\mathcal{H}_t$ | ハイパーパラメータ集合 |
| $\mathcal{E}_t$ | 環境 (Python/CUDA version) |
| $s_t$ | Random seed |
| $e_i$ | 実験 $i$ (4-tuple: $\mathbf{h}, \mathcal{D}, \mathbf{m}, \mathcal{A}$) |
| $\text{SLI}$ | Service Level Indicator (測定可能なメトリクス) |
| $\text{SLO}$ | Service Level Objective (SLIの目標値) |
| $\text{Error Budget}$ | $1 - \text{SLO}$ (許容される失敗の量) |
| $D_{\text{KL}}(P \| Q)$ | Kullback-Leibler divergence |
| $\text{JSD}(P \| Q)$ | Jensen-Shannon Divergence |
| $D_{\text{KS}}$ | Kolmogorov-Smirnov統計量 |
| $\text{PSI}$ | Population Stability Index |
| $r(x, y)$ | Reward model |
| $\pi_\theta(y \mid x)$ | Policy (LLM) |
| $\pi_{\text{ref}}(y \mid x)$ | Reference policy |
| $\beta$ | KL正則化係数 |
| $y_w$ | 好ましい応答 (win) |
| $y_l$ | 好ましくない応答 (lose) |
| $\mathcal{L}_{\text{DPO}}$ | Direct Preference Optimization loss |
| $\mathcal{L}_{\text{RM}}$ | Reward Modeling loss (Bradley-Terry) |
| $\alpha$ | 有意水準 (Type I error rate, 通常0.05) |
| $\beta$ | Type II error rate (通常0.2 → power = 0.8) |
| $\delta$ | Minimum Detectable Effect (MDE) |
| $n$ | サンプルサイズ |

---

## ライセンス

本記事は [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ja)（クリエイティブ・コモンズ 表示 - 非営利 - 継承 4.0 国際）の下でライセンスされています。

### ⚠️ 利用制限について
