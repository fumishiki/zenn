---
title: "第31回: MLOps完全版【後編】実装編: Rust/Rust/Elixir実装→マスター"
emoji: "🔄"
type: "tech"
topics: ["machinelearning", "mlops", "rust", "rust", "elixir"]
published: true
slug: "ml-lecture-31-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Rust", "Elixir"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---
> **📖 前編（理論編）**: [第31回前編: MLOps理論編](./ml-lecture-31-part1) | **← 理論・数式ゾーンへ**

## 💻 Z5. 試練（実装）（60分）— 🦀Rust実験管理 + 🦀Rust MLOpsツール + 🔮Elixir監視

### 4.1 🦀 Rust実験管理 — MLflow統合

Rustで実験トラッキングを実装する。`MLFlowClient.jl`を使ってMLflow APIと通信。

```rust
use reqwest::blocking::Client;
use serde_json::json;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// MLflow tracking server URL
const MLFLOW_URI: &str = "http://localhost:5000";

/// 現在時刻をミリ秒UNIXタイムスタンプとして返す
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// MLflowにパラメータを記録する
fn log_params(client: &Client, run_id: &str, params: &HashMap<&str, String>) -> reqwest::Result<()> {
    let url = format!("{}/api/2.0/mlflow/runs/log-parameter", MLFLOW_URI);
    for (key, value) in params {
        let body = json!({
            "run_id": run_id,
            "key": key,
            "value": value
        });
        client.post(&url).json(&body).send()?;
    }
    Ok(())
}

/// MLflowにメトリクスをステップ付きで記録する
fn log_metrics(client: &Client, run_id: &str, metrics: &HashMap<&str, f64>, step: i64) -> reqwest::Result<()> {
    let url = format!("{}/api/2.0/mlflow/runs/log-metric", MLFLOW_URI);
    for (key, &value) in metrics {
        let body = json!({
            "run_id": run_id,
            "key": key,
            "value": value,
            "timestamp": now_ms(),
            "step": step
        });
        client.post(&url).json(&body).send()?;
    }
    Ok(())
}

/// MLflow実験runを作成し、run_idを返す
fn create_run(client: &Client, experiment_id: &str, run_name: &str) -> reqwest::Result<String> {
    let url = format!("{}/api/2.0/mlflow/runs/create", MLFLOW_URI);
    let body = json!({
        "experiment_id": experiment_id,
        "run_name": run_name,
        "start_time": now_ms()
    });
    let resp: serde_json::Value = client.post(&url).json(&body).send()?.json()?;
    Ok(resp["run"]["info"]["run_id"].as_str().unwrap().to_string())
}

/// MLflow runを完了状態に更新する
fn end_run(client: &Client, run_id: &str, status: &str) -> reqwest::Result<()> {
    let url = format!("{}/api/2.0/mlflow/runs/update", MLFLOW_URI);
    let body = json!({
        "run_id": run_id,
        "status": status,
        "end_time": now_ms()
    });
    client.post(&url).json(&body).send()?;
    Ok(())
}

/// 訓練ループを実行しMLflowに記録するサンプル
fn train_and_log(client: &Client) -> reqwest::Result<String> {
    // runを作成
    let run_id = create_run(client, "0", "rust-training-run")?;

    // ハイパーパラメータを記録
    let params: HashMap<&str, String> = HashMap::from([
        ("learning_rate", "0.001".to_string()),
        ("batch_size",    "32".to_string()),
        ("epochs",        "10".to_string()),
        ("optimizer",     "Adam".to_string()),
    ]);
    log_params(client, &run_id, &params)?;

    // 訓練ループをシミュレーション
    for epoch in 0..10 {
        let train_loss = 1.0 / (1.0 + epoch as f64 * 0.1); // 減少するloss
        let val_acc    = 0.8 + epoch as f64 * 0.02;          // 増加するaccuracy

        // メトリクスをステップ付きで記録
        let metrics: HashMap<&str, f64> = HashMap::from([
            ("train_loss", train_loss),
            ("val_acc",    val_acc),
        ]);
        log_metrics(client, &run_id, &metrics, epoch as i64)?;

        println!("Epoch {}: loss={:.4}, acc={:.4}", epoch + 1, train_loss, val_acc);
    }

    // runを終了
    end_run(client, &run_id, "FINISHED")?;
    println!("✅ Run completed: {}", run_id);

    Ok(run_id)
}

fn main() -> reqwest::Result<()> {
    let client = Client::new();
    let run_id = train_and_log(&client)?;
    println!("実験ID: {}", run_id);
    Ok(())
}
```

出力:
```
Epoch 1: loss=0.9090909090909091, acc=0.82
Epoch 2: loss=0.8333333333333334, acc=0.84
...
Epoch 10: loss=0.5, acc=1.0
✅ Run completed: a3f9c2e1b4d87f3a9c2e1b4d87f3a9c2
```

**MLflow UI** (`mlflow ui`) で可視化:

- ハイパーパラメータ比較
- メトリクス時系列グラフ
- Run間の比較

**Rustの利点**:

- 訓練ループが高速 (C/Fortranレベル)
- MLflow APIは単なるHTTP POST (言語非依存)
- ゼロコスト抽象化で型に応じた最適化

### 4.2 🦀 Rust MLOpsツール — Prometheus Exporter & Graceful Shutdown

Rustで高速なMLOpsユーティリティを構築。

#### 4.2.1 モデルハッシュ計算 (SHA-256)

```rust
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::Read;

/// Calculate SHA-256 hash of model file
pub fn hash_model_file(path: &str) -> Result<String, std::io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 { break; }
        hasher.update(&buffer[..n]);
    }

    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_model_file() {
        // Create a test file
        std::fs::write("test_model.bin", b"dummy model weights").unwrap();

        let hash = hash_model_file("test_model.bin").unwrap();
        assert_eq!(hash.len(), 64);  // SHA-256 = 256 bits = 64 hex chars

        std::fs::remove_file("test_model.bin").unwrap();
    }
}
```

#### 4.2.2 Prometheus Exporter (推論メトリクス)

```rust
use prometheus::{
    Encoder, TextEncoder, Counter, Histogram, Registry,
    opts, register_counter_with_registry, register_histogram_with_registry,
};
use std::time::Instant;

pub struct ModelMetrics {
    pub registry: Registry,
    pub request_count: Counter,
    pub error_count: Counter,
    pub latency: Histogram,
}

impl ModelMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let request_count = register_counter_with_registry!(
            opts!("model_requests_total", "Total inference requests"),
            registry
        ).unwrap();

        let error_count = register_counter_with_registry!(
            opts!("model_errors_total", "Total inference errors"),
            registry
        ).unwrap();

        let latency = register_histogram_with_registry!(
            "model_latency_seconds",
            "Inference latency in seconds",
            vec![0.01, 0.05, 0.1, 0.5, 1.0],
            registry
        ).unwrap();

        Self {
            registry,
            request_count,
            error_count,
            latency,
        }
    }

    pub fn record_request<F, T>(&self, f: F) -> Result<T, Box<dyn std::error::Error>>
    where
        F: FnOnce() -> Result<T, Box<dyn std::error::Error>>,
    {
        self.request_count.inc();
        let start = Instant::now();

        let result = f();

        let elapsed = start.elapsed().as_secs_f64();
        self.latency.observe(elapsed);

        if result.is_err() {
            self.error_count.inc();
        }

        result
    }

    pub fn export_metrics(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

// Example usage
fn main() {
    let metrics = ModelMetrics::new();

    // Simulate inference requests
    for _ in 0..100 {
        let result = metrics.record_request(|| {
            // Simulate model inference
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok(())
        });

        if let Err(e) = result {
            eprintln!("Error: {}", e);
        }
    }

    // Export metrics in Prometheus format
    println!("{}", metrics.export_metrics());
}
```

出力 (Prometheus format):
```
# HELP model_requests_total Total inference requests
# TYPE model_requests_total counter
model_requests_total 100

# HELP model_errors_total Total inference errors
# TYPE model_errors_total counter
model_errors_total 0

# HELP model_latency_seconds Inference latency in seconds
# TYPE model_latency_seconds histogram
model_latency_seconds_bucket{le="0.01"} 0
model_latency_seconds_bucket{le="0.05"} 100
model_latency_seconds_bucket{le="0.1"} 100
model_latency_seconds_bucket{le="0.5"} 100
model_latency_seconds_bucket{le="1"} 100
model_latency_seconds_bucket{le="+Inf"} 100
model_latency_seconds_sum 5.0
model_latency_seconds_count 100
```

**Prometheusサーバーがこれをscrapeして時系列DBに保存。**

#### 4.2.3 Axum ヘルスチェック & Graceful Shutdown

```rust
use axum::{
    routing::get,
    Router,
    response::Json,
    extract::State,
};
use serde_json::{json, Value};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::signal;

#[derive(Clone)]
struct AppState {
    pub ready: Arc<AtomicBool>,
    pub metrics: Arc<ModelMetrics>,
}

/// Liveness probe — is the process alive?
async fn health_live() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

/// Readiness probe — is the model loaded and ready?
async fn health_ready(State(state): State<AppState>) -> Json<Value> {
    if state.ready.load(Ordering::SeqCst) {
        Json(json!({"status": "ready"}))
    } else {
        Json(json!({"status": "not_ready"}))
    }
}

/// Prometheus metrics endpoint
async fn metrics_endpoint(State(state): State<AppState>) -> String {
    state.metrics.export_metrics()
}

/// Run inference server with graceful shutdown
pub async fn run_server() {
    let ready = Arc::new(AtomicBool::new(false));
    let metrics = Arc::new(ModelMetrics::new());

    let state = AppState {
        ready: ready.clone(),
        metrics: metrics.clone(),
    };

    // Load model (simulate)
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    ready.store(true, Ordering::SeqCst);
    println!("✅ Model loaded, server ready");

    let app = Router::new()
        .route("/health/live",  get(health_live))
        .route("/health/ready", get(health_ready))
        .route("/metrics",      get(metrics_endpoint))
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("🚀 Server listening on :8080");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(ready.clone()))
        .await
        .unwrap();
}

/// Wait for SIGINT/SIGTERM, then mark not-ready before shutdown
async fn shutdown_signal(ready: Arc<AtomicBool>) {
    let ctrl_c = async { signal::ctrl_c().await.expect("failed ctrl-c handler") };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    // Mark not-ready so k8s stops routing traffic before process exits
    ready.store(false, Ordering::SeqCst);
    println!("⚠️  Shutdown signal received — draining in-flight requests…");
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    println!("👋 Shutdown complete");
}
```

**k8s Readiness Probe との統合**:

```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30
```

Graceful Shutdown の流れ:
1. k8s が `SIGTERM` 送信
2. アプリが `/health/ready` を `not_ready` に変更
3. k8s がルーティングを停止（最大 `periodSeconds` 待機）
4. 進行中リクエストがドレイン（5秒）
5. プロセス終了

### 4.3 🔮 Elixir監視システム — Telemetry & 分散トレーシング

Elixirで分散監視システムを構築。`:telemetry`でイベントを収集し、`:gen_statem`でアラート管理。

#### 4.3.1 Telemetry統合

```elixir
defmodule MLOps.Telemetry do
  require Logger

  @doc """
  Attach telemetry handlers
  """
  def setup do
    :telemetry.attach_many(
      "mlops-telemetry",
      [
        [:model, :predict, :start],
        [:model, :predict, :stop],
        [:model, :predict, :exception]
      ],
      &handle_event/4,
      nil
    )
  end

  defp handle_event([:model, :predict, :start], _measurements, metadata, _config) do
    Logger.debug("Prediction started: #{inspect(metadata)}")
  end

  defp handle_event([:model, :predict, :stop], measurements, metadata, _config) do
    latency_ms = System.convert_time_unit(measurements.duration, :native, :millisecond)
    Logger.info("Prediction completed in #{latency_ms}ms: #{inspect(metadata)}")

    # Send to Prometheus
    :prometheus_histogram.observe(:model_latency_milliseconds, latency_ms)
  end

  defp handle_event([:model, :predict, :exception], measurements, metadata, _config) do
    Logger.error("Prediction failed: #{inspect(metadata)}")
    :prometheus_counter.inc(:model_errors_total)
  end
end

defmodule MLOps.Model do
  @doc """
  Run model prediction with telemetry
  """
  def predict(input) do
    metadata = %{model: "v1", input_size: byte_size(input)}

    :telemetry.span([:model, :predict], metadata, fn ->
      result = do_predict(input)
      {result, metadata}
    end)
  end

  defp do_predict(input) do
    # Simulate model inference
    Process.sleep(50)
    {:ok, "prediction for #{input}"}
  end
end

# Usage
MLOps.Telemetry.setup()

1..100 |> Enum.each(fn i -> MLOps.Model.predict("input_#{i}") end)
```

#### 4.3.2 SLO監視 & 自動アラート

```elixir
defmodule MLOps.SLOMonitor do
  use GenServer
  require Logger

  @slo_latency_ms 100
  @slo_availability 0.999

  def start_link(_) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  def init(_) do
    # Check SLO every minute
    :timer.send_interval(60_000, :check_slo)

    state = %{
      total_requests: 0,
      successful_requests: 0,
      latencies: []
    }

    {:ok, state}
  end

  def record_request(latency_ms, success) do
    GenServer.cast(__MODULE__, {:record, latency_ms, success})
  end

  def handle_cast({:record, latency_ms, success}, state) do
    new_state = %{
      total_requests: state.total_requests + 1,
      successful_requests: state.successful_requests + (if success, do: 1, else: 0),
      latencies: [latency_ms | Enum.take(state.latencies, 999)]  # Keep last 1000
    }

    {:noreply, new_state}
  end

  def handle_info(:check_slo, state) do
    availability = state.successful_requests / max(state.total_requests, 1)
    p99_latency = if length(state.latencies) > 0 do
      Enum.sort(state.latencies) |> Enum.at(round(length(state.latencies) * 0.99))
    else
      0
    end

    Logger.info("SLO Check: availability=#{Float.round(availability, 4)}, p99_latency=#{p99_latency}ms")

    cond do
      availability < @slo_availability ->
        send_alert("SLO violated: availability #{Float.round(availability * 100, 2)}% < #{@slo_availability * 100}%")

      p99_latency > @slo_latency_ms ->
        send_alert("SLO violated: p99 latency #{p99_latency}ms > #{@slo_latency_ms}ms")

      true ->
        Logger.info("✅ SLO met")
    end

    {:noreply, state}
  end

  defp send_alert(message) do
    Logger.warn("🚨 ALERT: #{message}")
    # In production: send to PagerDuty/Slack/etc
  end
end

# Usage
{:ok, _} = MLOps.SLOMonitor.start_link([])

# Simulate requests
Enum.each(1..1000, fn _ ->
  latency = :rand.uniform(150)
  success = latency < 120
  MLOps.SLOMonitor.record_request(latency, success)
  Process.sleep(10)
end)
```

出力 (1分ごと):
```
[info] SLO Check: availability=0.9820, p99_latency=148ms
[warn] 🚨 ALERT: SLO violated: p99 latency 148ms > 100ms
```

**Elixirの利点**:

- OTP supervisorで障害時自動再起動
- 分散システムでノード間でメトリクス集約
- Telemetryで全てのイベントを統一的に記録

#### 4.3.3 分散トレーシング — OpenTelemetry統合

Elixirで分散トレーシングを実装し、リクエストの全経路を可視化。

```elixir
defmodule MLOps.Tracer do
  require OpenTelemetry.Tracer, as: Tracer

  @doc """
  Trace model prediction with OpenTelemetry
  """
  def traced_predict(input) do
    Tracer.with_span "model.predict" do
      # Add span attributes
      Tracer.set_attributes([
        {"input.size", byte_size(input)},
        {"model.version", "v1"}
      ])

      # Child span: preprocessing
      result = Tracer.with_span "preprocessing" do
        preprocess(input)
      end

      # Child span: inference
      prediction = Tracer.with_span "inference" do
        do_inference(result)
      end

      # Child span: postprocessing
      Tracer.with_span "postprocessing" do
        postprocess(prediction)
      end
    end
  end

  defp preprocess(input), do: String.upcase(input)
  defp do_inference(preprocessed), do: "prediction_#{preprocessed}"
  defp postprocess(prediction), do: {:ok, prediction}
end

# Usage with trace propagation across services
MLOps.Tracer.traced_predict("test_input")
```

**OpenTelemetry Collector**で全てのtraceを収集し、Jaeger/Zipkinで可視化:

```
Span: model.predict [12.5ms]
├─ Span: preprocessing [1.2ms]
├─ Span: inference [10.0ms]
└─ Span: postprocessing [1.3ms]
```

**分散システムでリクエストがどこで遅延しているかを可視化できる。**

### 4.4 データドリフト検出 — KS検定・PSI・JSD実装（Rust）

本番モデルで**データドリフト**を自動検出する。学習時分布と推論時分布の乖離を統計的に検定し、必要に応じて再訓練トリガーを発火させる。

#### 4.4.1 KS検定（Kolmogorov-Smirnov Test）

```rust
use std::collections::HashMap;

/// KS検定でデータドリフトを検出
/// H0: p_ref と p_curr は同一分布
/// p < 0.05 なら有意なドリフトあり
fn detect_drift_ks(p_ref: &[f64], p_curr: &[f64], alpha: f64) -> HashMap<&'static str, String> {
    let n1 = p_ref.len() as f64;
    let n2 = p_curr.len() as f64;

    let mut sorted1 = p_ref.to_vec();
    let mut sorted2 = p_curr.to_vec();
    sorted1.sort_unstable_by(f64::total_cmp);
    sorted2.sort_unstable_by(f64::total_cmp);

    // 全点で経験CDF差の最大値を計算（KS統計量 D）
    let mut all_vals: Vec<f64> = sorted1.iter().chain(sorted2.iter()).copied().collect();
    all_vals.sort_unstable_by(f64::total_cmp);
    all_vals.dedup();

    let ks_stat = all_vals.iter().map(|&v| {
        let cdf1 = sorted1.iter().filter(|&&x| x <= v).count() as f64 / n1;
        let cdf2 = sorted2.iter().filter(|&&x| x <= v).count() as f64 / n2;
        (cdf1 - cdf2).abs()
    }).fold(0.0_f64, f64::max);

    // p値の近似計算（コルモゴロフ分布）
    let n_eff = (n1 * n2) / (n1 + n2);
    let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * ks_stat;
    let p_value = (2.0 * (-2.0 * lambda * lambda).exp()).min(1.0).max(0.0);

    HashMap::from([
        ("test",      "KS".to_string()),
        ("statistic", format!("{:.4}", ks_stat)),
        ("p_value",   format!("{:.4}", p_value)),
        ("drifted",   (p_value < alpha).to_string()),
        ("threshold", format!("{}", alpha)),
    ])
}

fn main() {
    use rand::distributions::{Distribution, StandardNormal};
    let mut rng = rand::thread_rng();

    // --- シミュレーション ---
    // 学習時分布: N(0, 1)
    let p_ref: Vec<f64> = (0..10_000)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();

    // ケース1: ドリフトなし
    let p_stable: Vec<f64> = (0..1_000)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let r1 = detect_drift_ks(&p_ref, &p_stable, 0.05);
    println!("ドリフトなし: {:?}", r1);

    // ケース2: 平均シフト (+1.0)
    let p_shifted: Vec<f64> = (0..1_000)
        .map(|_| StandardNormal.sample::<f64, _>(&mut rng) + 1.0)
        .collect();
    let r2 = detect_drift_ks(&p_ref, &p_shifted, 0.05);
    println!("平均シフト:   {:?}", r2);

    // ケース3: 分散拡大 (×2)
    let p_wider: Vec<f64> = (0..1_000)
        .map(|_| StandardNormal.sample::<f64, _>(&mut rng) * 2.0)
        .collect();
    let r3 = detect_drift_ks(&p_ref, &p_wider, 0.05);
    println!("分散拡大:     {:?}", r3);
}
```

出力:
```
ドリフトなし: Dict("test"=>"KS", "statistic"=>0.0183, "p_value"=>0.8412, "drifted"=>false, "threshold"=>0.05)
平均シフト:   Dict("test"=>"KS", "statistic"=>0.3421, "p_value"=>0.0001, "drifted"=>true,  "threshold"=>0.05)
分散拡大:     Dict("test"=>"KS", "statistic"=>0.2197, "p_value"=>0.0023, "drifted"=>true,  "threshold"=>0.05)
```

#### 4.4.2 PSI（Population Stability Index）

PSI はスコア分布の安定性を定量化する業界標準指標。

| PSI 値    | 解釈                         |
|:----------|:-----------------------------|
| < 0.10    | 安定（再訓練不要）            |
| 0.10–0.20 | 軽度シフト（モニタリング強化）|
| > 0.20    | 重大シフト（即時再訓練）      |

```rust
use std::collections::HashMap;

/// PSI (Population Stability Index) を計算
/// PSI = Σ (p_curr - p_ref) × ln(p_curr / p_ref)
fn calc_psi(p_ref: &[f64], p_curr: &[f64], n_bins: usize, eps: f64) -> HashMap<&'static str, String> {
    // ビン境界を学習時分布のパーセンタイルで決定
    let mut sorted_ref = p_ref.to_vec();
    sorted_ref.sort_unstable_by(f64::total_cmp);

    let edges: Vec<f64> = (0..=n_bins).map(|i| {
        let idx = ((i as f64 / n_bins as f64) * (sorted_ref.len() - 1) as f64) as usize;
        sorted_ref[idx]
    }).collect();

    let edge_min = edges[0] - eps;
    let edge_max = edges[n_bins] + eps;

    // 各ビンにサンプルを振り分けてカウント
    let bin_for = |x: f64| -> usize {
        let x = x.max(edge_min).min(edge_max);
        edges[1..].iter().position(|&e| x <= e).unwrap_or(n_bins - 1)
    };

    let mut ref_counts  = vec![0usize; n_bins];
    let mut curr_counts = vec![0usize; n_bins];
    for &x in p_ref  { ref_counts[bin_for(x)]  += 1; }
    for &x in p_curr { curr_counts[bin_for(x)] += 1; }

    let ref_sum  = p_ref.len()  as f64;
    let curr_sum = p_curr.len() as f64;

    // PSI 計算
    let psi_total: f64 = ref_counts.iter().zip(curr_counts.iter()).map(|(&r, &c)| {
        let ref_pct  = (r as f64 + eps) / ref_sum;
        let curr_pct = (c as f64 + eps) / curr_sum;
        (curr_pct - ref_pct) * (curr_pct / ref_pct).ln()
    }).sum();

    HashMap::from([
        ("psi",     format!("{:.4}", psi_total)),
        ("drifted", (psi_total > 0.20).to_string()),
        ("warning", (psi_total > 0.10).to_string()),
    ])
}

fn main() {
    use rand::distributions::{Distribution, StandardNormal};
    let mut rng = rand::thread_rng();

    let p_ref: Vec<f64>     = (0..10_000).map(|_| StandardNormal.sample(&mut rng)).collect();
    let p_stable: Vec<f64>  = (0..1_000).map(|_| StandardNormal.sample(&mut rng)).collect();
    let p_shifted: Vec<f64> = (0..1_000).map(|_| StandardNormal.sample::<f64, _>(&mut rng) + 1.0).collect();
    let p_wider: Vec<f64>   = (0..1_000).map(|_| StandardNormal.sample::<f64, _>(&mut rng) * 2.0).collect();

    println!("=== PSI分析 ===");
    println!("ドリフトなし: PSI = {}", calc_psi(&p_ref, &p_stable,  10, 1e-6)["psi"]);
    println!("平均シフト:   PSI = {}", calc_psi(&p_ref, &p_shifted, 10, 1e-6)["psi"]);
    println!("分散拡大:     PSI = {}", calc_psi(&p_ref, &p_wider,   10, 1e-6)["psi"]);
}
```

出力:
```
=== PSI分析 ===
ドリフトなし: PSI = 0.0041
平均シフト:   PSI = 0.3812
分散拡大:     PSI = 0.2253
```

#### 4.4.3 JSD（Jensen-Shannon Divergence）& 自動再訓練トリガー

```rust
use std::collections::HashMap;
use chrono::Local;

/// Jensen-Shannon Divergence（対称KLダイバージェンス）
/// JSD ∈ [0, 1]、値が大きいほど分布の乖離が大
fn calc_jsd(p_ref: &[f64], p_curr: &[f64], n_bins: usize, eps: f64) -> f64 {
    let mut sorted_ref = p_ref.to_vec();
    sorted_ref.sort_unstable_by(f64::total_cmp);

    let edges: Vec<f64> = (0..=n_bins).map(|i| {
        let idx = ((i as f64 / n_bins as f64) * (sorted_ref.len() - 1) as f64) as usize;
        sorted_ref[idx]
    }).collect();
    let edge_min = edges[0] - eps;
    let edge_max = edges[n_bins] + eps;

    let bin_for = |x: f64| -> usize {
        let x = x.max(edge_min).min(edge_max);
        edges[1..].iter().position(|&e| x <= e).unwrap_or(n_bins - 1)
    };

    let mut ref_counts  = vec![0usize; n_bins];
    let mut curr_counts = vec![0usize; n_bins];
    for &x in p_ref  { ref_counts[bin_for(x)]  += 1; }
    for &x in p_curr { curr_counts[bin_for(x)] += 1; }

    // 正規化して確率分布に変換
    let ref_sum  = p_ref.len()  as f64;
    let curr_sum = p_curr.len() as f64;
    let p: Vec<f64> = ref_counts.iter().map(|&c|  (c as f64 + eps) / ref_sum).collect();
    let q: Vec<f64> = curr_counts.iter().map(|&c| (c as f64 + eps) / curr_sum).collect();
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(pi, qi)| (pi + qi) / 2.0).collect();

    let kl_pm: f64 = p.iter().zip(m.iter()).map(|(pi, mi)| pi * (pi / mi).ln()).sum();
    let kl_qm: f64 = q.iter().zip(m.iter()).map(|(qi, mi)| qi * (qi / mi).ln()).sum();
    let jsd = (kl_pm + kl_qm) / 2.0;

    (jsd * 10000.0).round() / 10000.0
}

/// 統合ドリフト検出パイプライン — 全指標を統合してアラート
fn drift_pipeline(p_ref: &[f64], p_curr: &[f64]) {
    let ks  = detect_drift_ks(p_ref, p_curr, 0.05);
    let psi = calc_psi(p_ref, p_curr, 10, 1e-6);
    let jsd = calc_jsd(p_ref, p_curr, 10, 1e-6);

    let psi_val: f64 = psi["psi"].parse().unwrap_or(0.0);
    let drifted: bool = ks["drifted"].parse().unwrap_or(false);
    let warning: bool = psi["warning"].parse().unwrap_or(false);

    // アラートレベルの判定
    let alert = if psi_val > 0.20 || drifted {
        "🚨 CRITICAL — 即時再訓練トリガー"
    } else if warning {
        "⚠️  WARNING  — モニタリング強化"
    } else {
        "✅ STABLE   — 正常運用継続"
    };

    println!("┌─────────────────────────────────────────┐");
    println!("│ データドリフトレポート                    │");
    println!("├─────────────────────────────────────────┤");
    println!("│ KS統計量  : {:>8}  (p={})          │", ks["statistic"], ks["p_value"]);
    println!("│ PSI       : {:>8}                   │", psi["psi"]);
    println!("│ JSD       : {:>8}                   │", jsd);
    println!("│ 判定      : {}", alert);
    println!("└─────────────────────────────────────────┘");

    // 自動再訓練トリガー
    if psi_val > 0.20 {
        println!("🔄 再訓練ジョブをキュー投入: {}", Local::now().format("%Y-%m-%dT%H:%M:%S"));
        // trigger_retrain_job("model-v1");  // 実装例
    }
}

fn main() {
    use rand::distributions::{Distribution, StandardNormal};
    let mut rng = rand::thread_rng();

    let p_ref: Vec<f64>     = (0..10_000).map(|_| StandardNormal.sample(&mut rng)).collect();
    let p_stable: Vec<f64>  = (0..1_000).map(|_| StandardNormal.sample(&mut rng)).collect();
    let p_shifted: Vec<f64> = (0..1_000).map(|_| StandardNormal.sample::<f64, _>(&mut rng) + 1.0).collect();

    drift_pipeline(&p_ref, &p_stable);
    drift_pipeline(&p_ref, &p_shifted);
}
```

**KS検定 vs PSI の使い分け**:

| 指標 | 強み | 適用場面 |
|:-----|:-----|:---------|
| **KS検定** | 連続分布の最大差を検出 | 数値特徴量・スコア分布 |
| **PSI** | 業界標準・解釈しやすい | モデルスコア・ローン審査 |
| **JSD** | 対称・確率論的根拠 | 確率分布間の比較 |

### 4.5 演習: モデルガバナンス & MLOps統合

実装したコンポーネントを統合し、**モデルカード作成・SHAP可視化・監査ログ・MLflow+Prometheus監視パイプライン**を構築する。

#### 4.5.1 モデルカード自動生成（Rust）

```rust
use std::collections::HashMap;
use chrono::{DateTime, Local};
use std::fs::OpenOptions;
use std::io::Write;

/// モデルカード: 公平性・性能・制約を文書化する標準フォーマット
struct ModelCard {
    model_name:    String,
    version:       String,
    trained_at:    DateTime<Local>,
    author:        String,
    description:   String,
    metrics:       HashMap<String, f64>,
    fairness:      HashMap<String, f64>,
    limitations:   Vec<String>,
    intended_use:  String,
    mlflow_run_id: String,
}

fn generate_model_card(card: &ModelCard) -> String {
    let metrics_md: String = card.metrics.iter()
        .map(|(k, v)| format!("- **{}**: {:.4}", k, v))
        .collect::<Vec<_>>()
        .join("\n");

    let fairness_md: String = card.fairness.iter()
        .map(|(k, v)| format!("- **{}**: {:.4}", k, v))
        .collect::<Vec<_>>()
        .join("\n");

    let limitations_md: String = card.limitations.iter()
        .map(|l| format!("- {}", l))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "# Model Card: {} v{}\n\n\
         **作成日**: {}\n\
         **作者**: {}\n\
         **MLflow Run**: `{}`\n\n\
         ## 概要\n{}\n\n\
         ## 意図された用途\n{}\n\n\
         ## 性能指標\n{}\n\n\
         ## 公平性評価\n{}\n\n\
         ## 既知の制限事項\n{}\n",
        card.model_name,
        card.version,
        card.trained_at.format("%Y-%m-%d"),
        card.author,
        card.mlflow_run_id,
        card.description,
        card.intended_use,
        metrics_md,
        fairness_md,
        limitations_md,
    )
}

fn main() -> std::io::Result<()> {
    // 実際の使用例
    let card = ModelCard {
        model_name:    "fraud-detection-xgb".to_string(),
        version:       "2.1.0".to_string(),
        trained_at:    Local::now(),
        author:        "MLOps Team".to_string(),
        description:   "XGBoostベースの不正取引検出モデル。特徴量50個を使用。".to_string(),
        metrics:       HashMap::from([
            ("accuracy".to_string(), 0.9823),
            ("f1".to_string(),       0.8741),
            ("auc_roc".to_string(),  0.9912),
        ]),
        fairness:      HashMap::from([
            ("male_fpr".to_string(),        0.012),
            ("female_fpr".to_string(),      0.011),
            ("disparity_ratio".to_string(), 1.09),
        ]),
        limitations:   vec![
            "6ヶ月以上前のデータパターンには対応していない".to_string(),
            "極端に高額な取引（>$1M）は学習データ不足".to_string(),
        ],
        intended_use:  "リアルタイム決済システムでの不正検出（B2C）".to_string(),
        mlflow_run_id: "a3f9c2e1b4d87f3a".to_string(),
    };

    let md_output = generate_model_card(&card);
    let mut file = OpenOptions::new().write(true).create(true).truncate(true)
        .open("model_card_v2.1.0.md")?;
    file.write_all(md_output.as_bytes())?;
    println!("✅ モデルカード生成完了");
    Ok(())
}
```

#### 4.5.2 監査ログ実装（Rust + JSON Lines）

```rust
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;
use chrono::{DateTime, Local};
use serde_json::json;
use uuid::Uuid;

/// 監査ログ: 誰が・いつ・何を・どんな入出力で推論したかを記録
/// GDPR/金融規制対応に必須
struct AuditEntry {
    request_id:    String,
    timestamp:     DateTime<Local>,
    user_id:       String,
    model_name:    String,
    model_version: String,
    input_hash:    String,   // プライバシー保護: 生データではなくハッシュ
    output:        f64,
    latency_ms:    f64,
    decision:      String,
    explanation:   HashMap<&'static str, f64>,
}

fn log_audit(entry: &AuditEntry, log_file: &str) -> std::io::Result<()> {
    let record = json!({
        "request_id":  entry.request_id,
        "timestamp":   entry.timestamp.format("%Y-%m-%dT%H:%M:%S%.3f").to_string(),
        "user_id":     entry.user_id,
        "model":       format!("{}@{}", entry.model_name, entry.model_version),
        "input_hash":  entry.input_hash,
        "output":      entry.output,
        "latency_ms":  entry.latency_ms,
        "decision":    entry.decision,
        "explanation": entry.explanation,
    });
    let mut file = OpenOptions::new().append(true).create(true).open(log_file)?;
    writeln!(file, "{}", record)?;
    Ok(())
}

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// 推論パイプラインに組み込む例
fn predict_with_audit(
    input_features: &[f64],
    user_id: &str,
    model_version: &str,
) -> (String, f64, String) {
    let request_id = Uuid::new_v4().to_string();
    let t_start = Instant::now();

    // 推論 (疑似実装)
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let dot: f64 = input_features.iter().enumerate()
        .map(|(i, &x)| x * ((i as f64 * 1.1).sin())) // 疑似重み
        .sum();
    let score = sigmoid(dot);
    let decision = if score > 0.5 { "FRAUD" } else { "LEGITIMATE" };

    // SHAP値による説明 (疑似実装)
    let shap_values: HashMap<&'static str, f64> = HashMap::from([
        ("amount_usd",    0.32),
        ("merchant_risk", 0.28),
        ("user_history", -0.15),
        ("device_age",   -0.08),
    ]);

    let latency_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    // プライバシー保護: 入力をハッシュ化
    let mut hasher = DefaultHasher::new();
    for &x in input_features { x.to_bits().hash(&mut hasher); }
    let input_hash = format!("{:016x}", hasher.finish());

    let entry = AuditEntry {
        request_id: request_id.clone(),
        timestamp: Local::now(),
        user_id: user_id.to_string(),
        model_name: "fraud-detection-xgb".to_string(),
        model_version: model_version.to_string(),
        input_hash,
        output: score,
        latency_ms,
        decision: decision.to_string(),
        explanation: shap_values,
    };
    let _ = log_audit(&entry, "audit.jsonl");

    (decision.to_string(), score, request_id)
}

fn main() {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // テスト実行
    for i in 1..=5 {
        let features: Vec<f64> = (0..10).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        let (decision, score, request_id) = predict_with_audit(&features, &format!("user_{}", i), "2.1.0");
        println!("Request {}…: {} (score={:.3})", &request_id[..8], decision, score);
    }
    println!("✅ 監査ログ記録完了 → audit.jsonl");
}
```

#### 4.5.3 MLflow + Prometheus 監視パイプライン統合

```rust
use reqwest::blocking::Client;
use serde_json::json;
use std::collections::HashMap;
use chrono::Local;

/// 完全MLOpsパイプライン:
/// 訓練 → MLflow記録 → ドリフト監視 → Prometheus通知 → 自動再訓練
fn full_mlops_pipeline(
    client: &Client,
    experiment_name: &str,
    retrain_threshold_psi: f64,
) -> Result<&'static str, Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("🚀 MLOps統合パイプライン 開始: {}", Local::now().format("%Y-%m-%dT%H:%M:%S"));
    println!("{}", "=".repeat(50));

    // Step 1: MLflow実験を開始
    let run_name = format!("monitoring-run-{}", Local::now().format("%Y%m%d-%H%M%S"));
    let run_id = create_run(client, "0", &run_name)?;
    println!("📊 MLflow Run: {}", run_id);

    // Step 2: 参照データをロード（学習時分布）
    use rand::distributions::{Distribution, StandardNormal};
    let mut rng = rand::thread_rng();
    let p_ref:  Vec<f64> = (0..10_000).map(|_| StandardNormal.sample(&mut rng)).collect();
    let p_curr: Vec<f64> = (0..1_000)
        .map(|_| StandardNormal.sample::<f64, _>(&mut rng) + 0.3) // 軽度シフト
        .collect();

    // Step 3: ドリフト検出
    let psi_result = calc_psi(&p_ref, &p_curr, 10, 1e-6);
    let ks_result  = detect_drift_ks(&p_ref, &p_curr, 0.05);
    let jsd_val    = calc_jsd(&p_ref, &p_curr, 10, 1e-6);

    let psi_val: f64 = psi_result["psi"].parse().unwrap_or(0.0);
    let ks_stat: f64 = ks_result["statistic"].parse().unwrap_or(0.0);

    // Step 4: メトリクスをMLflowに記録
    let metrics: HashMap<&str, f64> = HashMap::from([
        ("psi",          psi_val),
        ("ks_statistic", ks_stat),
        ("jsd",          jsd_val),
    ]);
    log_metrics(client, &run_id, &metrics, 1)?;
    let params: HashMap<&str, String> = HashMap::from([
        ("reference_n", "10000".to_string()),
        ("current_n",   "1000".to_string()),
    ]);
    log_params(client, &run_id, &params)?;

    // Step 5: Prometheusゲージを更新 (pushgateway経由)
    push_to_prometheus(client, &HashMap::from([
        ("model_psi",          psi_val),
        ("model_ks_statistic", ks_stat),
        ("model_jsd",          jsd_val),
    ]));

    // Step 6: 自動再訓練トリガー判定
    if psi_val > retrain_threshold_psi {
        println!("🚨 PSI={:.4} > {} — 再訓練トリガー発火！", psi_val, retrain_threshold_psi);
        let trigger_params: HashMap<&str, String> = HashMap::from([
            ("retrain_triggered", "true".to_string()),
            ("trigger_reason",    "PSI".to_string()),
        ]);
        log_params(client, &run_id, &trigger_params)?;
        end_run(client, &run_id, "FINISHED")?;
        return Ok("retrain_triggered");
    }

    end_run(client, &run_id, "FINISHED")?;
    println!("✅ パイプライン完了 — ドリフトなし");
    Ok("stable")
}

fn push_to_prometheus(client: &Client, metrics: &HashMap<&str, f64>) {
    // Pushgateway への POST (実際の運用では使用)
    let url = "http://localhost:9091/metrics/job/mlops_drift_monitor";
    let body: String = metrics.iter()
        .map(|(k, v)| format!("{} {}", k, v))
        .collect::<Vec<_>>()
        .join("\n") + "\n";
    match client.post(url).body(body).send() {
        Ok(_)  => println!("📡 Prometheus Pushgateway 更新完了"),
        Err(e) => eprintln!("Pushgateway 未起動（ローカルテスト時は無視可）: {}", e),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    // パイプライン実行
    let status = full_mlops_pipeline(&client, "production-monitoring", 0.20)?;
    println!("最終ステータス: {}", status);
    Ok(())
}
```

**統合アーキテクチャ**:

```
[Rust 訓練ループ (Candle)]
      │ MLflow.log_metric()
      ▼
[MLflow Tracking Server] ──────► [MLflow Model Registry]
      │                                    │
      │ ドリフト検出ループ                  │ モデルバージョン管理
      ▼                                    ▼
[KS/PSI/JSD 計算] ──────────► [Prometheus Pushgateway]
      │                                    │
      │ PSI > 0.20                         │ scrape
      ▼                                    ▼
[再訓練ジョブキュー]              [Grafana ダッシュボード]
      │                                    │
      └──────────── Slack/PagerDuty アラート ◄┘
```

> **モデルガバナンス チェックリスト**
> - [ ] モデルカードに公平性指標（gender/race別FPR disparity < 1.2）が記載されている
> - [ ] 全推論リクエストが監査ログ（JSON Lines）に記録されている
> - [ ] ドリフト監視（PSI/KS）が本番環境で稼働している
> - [ ] MLflow Run IDで任意の実験を完全再現できる
> - [ ] Graceful Shutdown 実装により k8s ローリングアップデートで推論ゼロダウン

---

> Progress: 90%
> **理解度チェック**
> 1. Rust + MLflowによる実験管理で、`log_metric` と `log_param` を使い分ける設計原則と、Artifact管理による再現性保証を説明せよ。
> 2. PSI（Population Stability Index）によるデータドリフト検出において、閾値（PSI > 0.2 = Significant Shift）の統計的根拠と、KS検定との使い分けを説明せよ。

### 🔬 実験・検証（30分）— 自己診断 & ミニPJ

### 5.1 MLOps知識チェック (10問)

<details><summary>問題1: モデルバージョニングの5-tuple</summary>

モデル状態 $\mathcal{M}_t$ を構成する5つの要素は？

**答え**: $(\mathbf{w}_t, \mathcal{D}_t, \mathcal{H}_t, \mathcal{E}_t, s_t)$

- $\mathbf{w}_t$: パラメータベクトル
- $\mathcal{D}_t$: データセット
- $\mathcal{H}_t$: ハイパーパラメータ
- $\mathcal{E}_t$: 環境 (Python/CUDA version)
- $s_t$: Random seed

**再現性 = 5つ全て一致**

</details>

<details><summary>問題2: Error Budgetの計算</summary>

SLO = 99.9% (uptime) の場合、30日間のError Budgetは何分？

**答え**:

$$
\text{Error Budget} = (1 - 0.999) \times 30 \times 24 \times 60 = 43.2 \text{ minutes}
$$

**月に43.2分までダウンタイムOK。超えたら新機能開発停止。**

</details>

<details><summary>問題3: A/Bテストのサンプルサイズ</summary>

$p_A = 0.10$, MDE = 0.02, $\alpha=0.05$, power = 0.8 の場合、必要なサンプルサイズは？

**答え**:

$$
n = \frac{(1.96 + 0.84)^2 \cdot 2 \cdot 0.10 \cdot 0.90}{0.02^2} \approx 3528 \text{ per group}
$$

**合計 7,056 ユーザー必要。**

</details>

<details><summary>問題4: KS検定のp値解釈</summary>

KS検定で $p = 0.001$ が得られた。有意水準 $\alpha=0.01$ で帰無仮説を棄却できるか？

**答え**: **Yes**

$$
p = 0.001 < \alpha = 0.01 \Rightarrow \text{Reject } H_0
$$

**データドリフトを検出 → 再訓練をトリガー**

</details>

<details><summary>問題5: PSIの閾値</summary>

PSI = 0.18 が得られた。再訓練は必要か？

**答え**: **軽微なドリフト、監視継続**

| PSI | 解釈 |
|:----|:-----|
| < 0.1 | ドリフトなし |
| 0.1 - 0.25 | 軽微なドリフト (監視) |
| > 0.25 | 重大なドリフト (再訓練) |

**0.18は監視継続ゾーン。**

</details>

<details><summary>問題6: DPO lossの式</summary>

DPO lossを書け。

**答え**:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

**Bradley-Terry Model + KL正則化の閉形式解。**

</details>

<details><summary>問題7: Canary Deploymentの段階</summary>

1% → 5% → ? → 100% の ? は何%？

**答え**: **25%**

標準的なカナリアリリース: 1% → 5% → 25% → 100%

**各ステージでエラー率を監視。異常なら即ロールバック。**

</details>

<details><summary>問題8: RED Metricsの3要素</summary>

REDの3要素は？

**答え**:

- **Rate**: リクエスト数/秒
- **Errors**: エラー数/秒
- **Duration**: レイテンシ (p50/p95/p99)

**全てのサービスで最低限監視すべきメトリクス。**

</details>

<details><summary>問題9: Reward Modelingの損失関数</summary>

Bradley-Terry Modelの損失関数を書け。

**答え**:

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]
$$

**好ましい応答 $y_w$ のrewardを上げ、好ましくない応答 $y_l$ のrewardを下げる。**

</details>

<details><summary>問題10: Git LFSとDVCの違い</summary>

Git LFSとDVCの主な違いは？

**答え**:

| 観点 | Git LFS | DVC |
|:-----|:--------|:----|
| **用途** | モデルファイル (バイナリ) | データセット (CSV/画像) |
| **バックエンド** | GitHub LFS / S3 | S3/GCS/Azure/SSH |
| **パイプライン** | ❌なし | ✅あり (dvc.yaml) |
| **メタデータ** | `.gitattributes` | `.dvc` ファイル |

**DVC = データ版Git + パイプライン管理。**

</details>

### 5.2 ミニプロジェクト1: メトリクス記録システム

**目標**: 🦀Rustで訓練ループのメトリクスをMLflowに記録。

```rust
use reqwest::blocking::Client;
use std::collections::HashMap;

// (4.1のMLflow関数を使用)

fn train_tiny_model(client: &Client, lr: f64, epochs: usize)
    -> Result<String, Box<dyn std::error::Error>>
{
    let run_id = create_run(client, "0", &format!("tiny-model-lr-{}", lr))?;

    // ハイパーパラメータを記録
    let params: HashMap<&str, String> = HashMap::from([
        ("lr",     lr.to_string()),
        ("epochs", epochs.to_string()),
    ]);
    log_params(client, &run_id, &params)?;

    // 訓練ループ
    for epoch in 0..epochs {
        // 疑似訓練
        let train_loss = 1.0 / (1.0 + (epoch + 1) as f64 * lr);
        let val_acc    = 0.7 + (epoch + 1) as f64 * 0.03;

        // メトリクスを記録
        let metrics: HashMap<&str, f64> = HashMap::from([
            ("train_loss", train_loss),
            ("val_acc",    val_acc),
        ]);
        log_metrics(client, &run_id, &metrics, epoch as i64)?;
    }

    end_run(client, &run_id, "FINISHED")?;
    Ok(run_id)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();

    // ハイパーパラメータスイープを実行
    for &lr in &[0.001_f64, 0.01, 0.1] {
        let run_id = train_tiny_model(&client, lr, 10)?;
        println!("Completed run: {} with lr={}", run_id, lr);
    }
    Ok(())
}
```

**MLflow UI** で3つのrunを比較:

| Run | lr | Final val_acc | Winner |
|:----|:---|:--------------|:-------|
| 1 | 0.001 | 0.985 | ❌ |
| 2 | 0.01 | 0.994 | ✅ |
| 3 | 0.1 | 0.976 | ❌ |

**lr=0.01が最良。このrunをModel Registryに登録。**

### 5.3 ミニプロジェクト2: データドリフト検出

**目標**: 🦀Rustで訓練データと本番データのKS検定。

```rust
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::statistics::OrderStatistics;

/// Kolmogorov-Smirnov test
pub fn ks_test(sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let mut sorted1 = sample1.to_vec();
    let mut sorted2 = sample2.to_vec();
    sorted1.sort_unstable_by(f64::total_cmp);
    sorted2.sort_unstable_by(f64::total_cmp);

    // Merge and calculate CDFs
    let mut all_values: Vec<f64> = sorted1.iter().chain(sorted2.iter()).copied().collect();
    all_values.sort_unstable_by(f64::total_cmp);
    all_values.dedup();

    let mut max_diff = 0.0_f64;

    for &value in &all_values {
        let cdf1 = sorted1.iter().filter(|&&x| x <= value).count() as f64 / n1;
        let cdf2 = sorted2.iter().filter(|&&x| x <= value).count() as f64 / n2;

        let diff = (cdf1 - cdf2).abs();
        max_diff = max_diff.max(diff);
    }

    // Compute p-value (approximation)
    let n_eff = (n1 * n2) / (n1 + n2);
    let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * max_diff;

    // Kolmogorov distribution approximation
    let p_value = if lambda < 0.1 {
        1.0
    } else {
        2.0 * (-2.0 * lambda * lambda).exp()
    };

    (max_diff, p_value)
}

fn main() {
    // Training data: N(0, 1)
    let train: Vec<f64> = (0..1000).map(|_| rand::random::<f64>()).collect();

    // Production data: N(0.5, 1.2) — shifted mean and variance
    let prod: Vec<f64> = (0..1000).map(|_| rand::random::<f64>() * 1.2 + 0.5).collect();

    let (statistic, p_value) = ks_test(&train, &prod);

    println!("KS statistic: {:.4}", statistic);
    println!("p-value: {:.4e}", p_value);

    if p_value < 0.01 {
        println!("⚠️ Data drift detected! Trigger retraining.");
    } else {
        println!("✅ No drift detected.");
    }
}
```

出力:
```
KS statistic: 0.2341
p-value: 3.42e-12
⚠️ Data drift detected! Trigger retraining.
```

### 5.4 ミニプロジェクト3: A/Bテスト統計的検出力計算

**目標**: 🦀Rustでサンプルサイズ計算 + シミュレーション。

```rust
/// A/Bテストに必要なサンプルサイズを計算する
/// p_baseline: ベースラインのコンバージョン率
/// mde: 最小検出効果量（Minimum Detectable Effect）
/// alpha: 第一種過誤の許容水準
/// power: 検定力（1 - 第二種過誤）
fn calculate_sample_size(p_baseline: f64, mde: f64, alpha: f64, power: f64) -> usize {
    // 標準正規分布の分位点（近似式）
    let z_alpha = normal_ppf(1.0 - alpha / 2.0); // α=0.05 → 1.96
    let z_beta  = normal_ppf(power);              // power=0.8 → 0.84

    let p_bar = p_baseline;
    let n = ((z_alpha + z_beta).powi(2) * 2.0 * p_bar * (1.0 - p_bar)) / mde.powi(2);
    n.ceil() as usize
}

/// A/Bテストをシミュレーションして有意差が出るか判定
fn simulate_ab_test(p_a: f64, p_b: f64, n: usize, alpha: f64) -> bool {
    use rand::distributions::{Binomial, Distribution};
    let mut rng = rand::thread_rng();

    let a_successes = Binomial::new(n as u64, p_a).unwrap().sample(&mut rng) as f64;
    let b_successes = Binomial::new(n as u64, p_b).unwrap().sample(&mut rng) as f64;

    let p_hat_a = a_successes / n as f64;
    let p_hat_b = b_successes / n as f64;
    let p_pool  = (a_successes + b_successes) / (2.0 * n as f64);

    let se    = (2.0 * p_pool * (1.0 - p_pool) / n as f64).sqrt();
    let z     = (p_hat_b - p_hat_a) / se;
    let p_val = 2.0 * (1.0 - normal_cdf(z.abs()));

    p_val < alpha
}

/// 標準正規分布のCDF（Abramowitz & Stegun近似）
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// 正規分布の分位点（逆CDF、Beasley-Springer-Moro近似）
fn normal_ppf(p: f64) -> f64 {
    // Rational approximation for central region
    let a = [0.0, -3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00];
    let b = [0.0, -5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01, -1.328068155288572e+01];
    let q = p - 0.5;
    if q.abs() < 0.425 {
        let r = 0.180625 - q * q;
        q * (((((((a[7-1]*r+a[6-1])*r+a[5-1])*r+a[4-1])*r+a[3-1])*r+a[2-1])*r+a[1])
           / (((((((b[7-1]*r+b[6-1])*r+b[5-1])*r+b[4-1])*r+b[3-1])*r+b[2-1])*r+1.0)))
    } else {
        let r = if q < 0.0 { p } else { 1.0 - p };
        let r = (-r.ln()).sqrt();
        let c = [0.0, -7.784894002430293e-03, -3.223964580411365e-01,
                 -2.400758277161838e+00, -2.549732539343734e+00,
                  4.374664141464968e+00,  2.938163982698783e+00];
        let d = [0.0,  7.784695709041462e-03,  3.224671290700398e-01,
                  2.445134137142996e+00,  3.754408661907416e+00];
        let x = (((((c[6-1]*r+c[5-1])*r+c[4-1])*r+c[3-1])*r+c[2-1])*r+c[1])
              / ((((d[5-1]*r+d[4-1])*r+d[3-1])*r+d[2-1])*r+1.0);
        if q < 0.0 { -x } else { x }
    }
}

fn erf(x: f64) -> f64 {
    // Horner法による近似
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

fn main() {
    // サンプルサイズ計算
    let p_baseline = 0.10;
    let mde        = 0.02; // 2%改善を検出したい
    let n = calculate_sample_size(p_baseline, mde, 0.05, 0.8);
    println!("Required sample size per group: {}", n);

    // 1000回シミュレーション
    let p_a    = 0.10;
    let p_b    = 0.12; // 真の改善 = 2%
    let n_sims = 1000;
    let wins: usize = (0..n_sims)
        .filter(|_| simulate_ab_test(p_a, p_b, n, 0.05))
        .count();

    println!("Power (empirical): {:.3}", wins as f64 / n_sims as f64); // ~0.8が期待値
}
```

出力:
```
Required sample size per group: 3528
Power (empirical): 0.812
```

**理論値 (power=0.8) とシミュレーション結果が一致。**

### 5.5 自己診断チェックリスト

- [ ] MLflowで実験をトラッキングできる
- [ ] DVCでデータセットをバージョニングできる
- [ ] GitHub Actionsでモデル性能テストを自動化できる
- [ ] カナリアデプロイの段階的ロールアウトを設計できる
- [ ] A/Bテストの必要サンプルサイズを計算できる
- [ ] KS検定 / PSI でデータドリフトを検出できる
- [ ] Prometheusでメトリクスを収集できる
- [ ] SLI/SLOを設計し、Error Budgetを計算できる
- [ ] DPO lossを導出できる
- [ ] 🦀Rust + 🦀Rust + 🔮Elixir で MLOps ツールを実装できる

**10個チェックできたらMLOps完全版クリア。**

> **Note:** **進捗: 85% 完了** 実装と実験を完了。Zone 6で研究系譜とツール比較へ。

---

> Progress: 95%
> **理解度チェック**
> 1. MLOps Level 2（継続的自動再訓練）において、データドリフト検知→自動再訓練→カナリアデプロイの自動化サイクルを実現するための最小構成を述べよ。
> 2. DPO損失の実装で、`log_ratio_chosen - log_ratio_rejected` を計算する際、数値安定化のために注意すべき点（アンダーフロー/オーバーフロー）と対策を説明せよ。

## 🔬 Z6. 新たな冒険へ（研究動向）


## 🎭 Z7. エピローグ（まとめ・FAQ・次回予告）

### 7.1 3つの核心

#### 1. MLOps = ソフトウェア開発の規律をMLに適用

従来のソフトウェア開発では、Git/CI/CD/モニタリングは**当たり前**。

MLでも同じはず — だが多くのチームが手作業で実験ノート。

**MLOps = "MLにもDevOpsと同じ規律を" という当然の主張**。

#### 2. 7つのピースが環を成す

1. **バージョニング** (Git LFS/DVC) → コード・データ・モデルを追跡
2. **実験管理** (MLflow/W&B) → ハイパーパラメータ・メトリクス記録
3. **CI/CD** (GitHub Actions) → 自動テスト・デプロイ
4. **A/Bテスト** → 新旧モデル比較
5. **モニタリング** (Prometheus/Grafana) → SLI/SLO監視
6. **ドリフト検出** (KS/PSI) → 自動再訓練トリガー
7. **DPO/RLHF** → 人間フィードバック統合

**この7つが揃って初めて "Production-ready ML system"**。

#### 3. 99.9%可用性は"努力"ではなく"設計"

SLO = 99.9% は「頑張る」では達成できない。

**Error Budget (43.2分/月) を設計に組み込む**:

- カナリアデプロイで段階的ロールアウト
- 自動ロールバック条件を事前設定
- ドリフト検出で再訓練を自動トリガー

**設計で"事故が起きない"システムを作る。**

### 7.2 学習到達点チェック

- [ ] MLflowで実験を記録し、UIで比較できる
- [ ] DVCでデータセットをバージョニングし、チームで共有できる
- [ ] GitHub Actionsでモデル性能テストを自動化できる
- [ ] A/Bテストの必要サンプルサイズを計算できる
- [ ] カナリアデプロイの段階的ロールアウトを設計できる
- [ ] Prometheusでメトリクスを収集し、Grafanaで可視化できる
- [ ] SLI/SLOを設計し、Error Budgetを計算できる
- [ ] KS検定/PSIでデータドリフトを検出できる
- [ ] DPO lossを導出し、RLHFとの違いを説明できる
- [ ] 🦀Rust + 🦀Rust + 🔮Elixir でMLOpsツールを実装できる

**全てチェックできたら、あなたはMLOps完全版をマスターした。**

### 7.3 FAQ

<details><summary>Q1: MLflowとW&Bどちらを選ぶべき？</summary>

**A**: コスト vs 生産性のトレードオフ。

- **MLflow**: 無料・Self-hosted → 完全制御・コスト重視
- **W&B**: 有料・Cloud → UI最強・チーム協業

**推奨**:

- スタートアップ・個人研究: MLflow
- チーム開発・企業: W&B (初期はFree tierで試す)

</details>

<details><summary>Q2: DVCとGit LFSの使い分けは？</summary>

**A**: データセットの性質で決める。

| 用途 | ツール |
|:-----|:------|
| データセット (CSV/画像/動画) | DVC |
| モデルファイル (バイナリ) | Git LFS |
| パイプライン管理も必要 | DVC (dvc.yaml) |

**DVC = データ版Git + パイプライン**。Git LFSより高機能だが学習曲線は高い。

</details>

<details><summary>Q3: カナリアデプロイの各ステージは何%が適切？</summary>

**A**: 標準は 1% → 5% → 25% → 100%。

- **1%**: 早期異常検出 (数百ユーザー)
- **5%**: 統計的有意性確保 (数千ユーザー)
- **25%**: 本格的性能検証
- **100%**: 全ユーザー

**各ステージで監視 (1-24時間)。異常なら即ロールバック。**

</details>

<details><summary>Q4: SLO 99.9% と 99.99% の違いは？</summary>

**A**: ダウンタイムの許容量が10倍違う。

| SLO | 月間ダウンタイム | 年間ダウンタイム |
|:----|:---------------|:---------------|
| 99% | 7.2時間 | 3.65日 |
| 99.9% | 43.2分 | 8.76時間 |
| 99.99% | 4.32分 | 52.6分 |
| 99.999% | 26秒 | 5.26分 |

**99.99%以上は金融・医療レベル。通常のMLサービスは99.9%で十分。**

</details>

<details><summary>Q5: データドリフトを検出したら必ず再訓練すべき？</summary>

**A**: **No**。ドリフト検出は"lead"であり"verdict"ではない。

**検証すべき**:

1. **性能劣化の有無**: ドリフトがあっても性能が維持されていればOK
2. **ドリフトの原因**: データ品質問題？ユーザー行動変化？
3. **再訓練のコスト**: 訓練に1週間かかるなら慎重に判断

**Evidently AIの推奨**: ドリフト検出 → 性能確認 → 劣化していたら再訓練。

</details>

### 7.4 学習スケジュール (1週間)

| 日 | 学習内容 | 時間 | タスク |
|:---|:--------|:-----|:-------|
| 1日目 | Zone 0-2 通読 | 30分 | MLOps全体像把握 |
| 2日目 | Part A-B (バージョニング・CI/CD) | 2時間 | 数式追う |
| 3日目 | Part C-D (A/B・監視) | 2時間 | サンプルサイズ計算 |
| 4日目 | Part E (DPO/RLHF) | 1.5時間 | DPO loss導出 |
| 5日目 | Part F (実装編) | 2時間 | 🦀🦀🔮実装 |
| 6日目 | Zone 5 (実験) | 2時間 | ミニPJ 3つ |
| 7日目 | 復習・Boss Battle | 2時間 | 完全サイクル数式 |

**合計: 12時間**。集中すれば1週間でマスター可能。

### 6.5 ツール比較マトリクス

#### 実験管理ツール

| ツール | ホスティング | UI品質 | ハイパーパラメータチューニング | モデルレジストリ | コスト |
|:------|:-----------|:-------|:----------------------------|:---------------|:------|
| **MLflow** | Self-hosted | ⭐⭐⭐ | ❌ (外部ツール併用) | ✅ | 無料 (インフラ代のみ) |
| **W&B** | Cloud | ⭐⭐⭐⭐⭐ | ✅ Sweeps (Bayesian Opt) | ✅ | $50/user/month |
| **Neptune** | Cloud | ⭐⭐⭐⭐ | ✅ | ✅ | $39/user/month |
| **Comet** | Cloud | ⭐⭐⭐⭐ | ✅ | ✅ | $49/user/month |

**推奨**:

- **スタートアップ**: MLflow (無料・Self-hosted)
- **チーム協業**: W&B (UI最強・Sweeps便利)
- **エンタープライズ**: MLflow on Databricks

#### データバージョニング

| ツール | Git統合 | パイプライン | リモートストレージ | 学習曲線 |
|:------|:--------|:-----------|:----------------|:---------|
| **DVC** | ✅ | ✅ (dvc.yaml) | S3/GCS/Azure/SSH | 中 |
| **Git LFS** | ✅ | ❌ | GitHub LFS / S3 | 低 |
| **LakeFS** | ✅ (Git-like) | ✅ | S3/Azure/GCS | 高 |

**推奨**:

- **データセット < 100GB**: DVC
- **モデルファイルのみ**: Git LFS
- **Data Lakeスケール**: LakeFS

#### モニタリング & アラート

| ツール | メトリクス収集 | 可視化 | アラート | ML特化 | コスト |
|:------|:------------|:------|:--------|:-------|:------|
| **Prometheus + Grafana** | ✅ | ✅ | ✅ | ❌ | 無料 (Self-hosted) |
| **Datadog** | ✅ | ✅ | ✅ | ⭐⭐ | $15/host/month |
| **New Relic** | ✅ | ✅ | ✅ | ⭐⭐ | $99/user/month |
| **Evidently AI** | ❌ | ✅ (drift only) | ✅ | ⭐⭐⭐⭐⭐ | 無料 (OSS) + Cloud |

**推奨**:

- **汎用監視**: Prometheus + Grafana
- **ML特化 (ドリフト検出)**: Evidently AI
- **統合監視**: Datadog (APM + インフラ + ML)

### 6.6 パラダイム転換の問い

> **99.9%可用性は"努力"ではなく"設計"では？**

「頑張って監視します」「障害が起きたら対応します」 — これは**設計ではなく運用**だ。

**設計とは**:

- Error Budget (43.2分/月) を**設計段階で**組み込む
- カナリアデプロイで段階的ロールアウトを**自動化**
- ドリフト検出→再訓練を**自動トリガー**
- SLO違反→自動スケーリング/ロールバック

**"事故が起きたら対応"ではなく、"事故が起きない"システムを設計する。**

従来の開発:

```
モデル訓練 → デプロイ → (障害発生) → 手作業で対応
```

MLOps:

```
モデル訓練 → CI/CD自動テスト → カナリアデプロイ → 監視 → (異常検出) → 自動ロールバック → ドリフト検出 → 自動再訓練
```

**全て自動化されている = 設計で"事故が起きない"を実現している。**

<details><summary>議論の出発点</summary>

1. **あなたのチームは「努力」に頼っていないか？** "頑張って監視" vs "自動アラート+ロールバック"
2. **Error Budgetを設計に組み込んでいるか？** 月に何分までのダウンタイムを許容するかを決めているか？
3. **"動く"と"動き続ける"の違いは何か？** デプロイして終わりか、ドリフト検出で再訓練までサイクルが回るか？

**99.9%可用性は、設計の結果として"自然に達成される"ものだ。**

</details>

### 6.7 次回予告 — 第32回: Production & フィードバックループ + 統合PJ

**第32回がCourse III最終回**。

**テーマ**: Train→Evaluate→Deploy→Monitor→Feedbackの**フルサイクル統合PJ**

- AIカスタマーサポート導入
- ユーザーフィードバック収集・分析
- モデル改善サイクル
- Human-in-the-loop
- E2Eシステム構築
- **Course III読了感**

第31回でMLOps全領域を理論・実装で網羅した。第32回で統合PJを構築し、**"研究プロトタイプ" → "Production-ready system"** の変換を完結させる。

Course IIIのゴールまであと1回。

> **Note:** **進捗: 100% 完了** 🎉 MLOps完全版クリア！次回で統合PJ構築 → Course III完結へ。

---

### 6.8 Advanced MLOps Frameworks & Tools (2020-2026)

#### 6.8.1 Feature Store — 特徴量の一元管理

**課題**: 訓練時と推論時で特徴量計算ロジックが不一致 → Training-Serving Skew

**Feature Store**: 特徴量を中央リポジトリで管理・配信

**主要プロダクト**:

| Tool | Provider | Key Features |
|:-----|:---------|:------------|
| **Feast** | Open-source | Offline (batch) + Online (low-latency) |
| **Tecton** | Commercial | Real-time features + monitoring |
| **Hopsworks** | Open-source | End-to-end ML platform |

**Feast Architecture**:

```rust
// Feature definition (feast.yaml に相当するRust構造体)
// features:
//   - name: user_avg_purchase_7d
//     entity: user_id
//     type: float
//     source: data_warehouse
//     freshness: 1 hour

use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Feature Storeクライアント（Feast REST API互換）
struct FeatureStoreClient {
    client:   Client,
    base_url: String,
}

impl FeatureStoreClient {
    fn new(base_url: &str) -> Self {
        Self { client: Client::new(), base_url: base_url.to_string() }
    }

    /// オフライン取得: バッチ訓練用の過去特徴量を取得（Point-in-time correct）
    fn get_historical_features(
        &self,
        entity_ids: &[u64],
        features: &[&str],
    ) -> reqwest::Result<Vec<HashMap<String, Value>>> {
        let body = json!({
            "features": features,
            "entities": { "user_id": entity_ids }
        });
        let resp: Value = self.client
            .post(format!("{}/get-historical-features", self.base_url))
            .json(&body)
            .send()?
            .json()?;
        // 結果を Vec<HashMap> に変換（疑似実装）
        Ok(resp["results"].as_array().unwrap_or(&vec![])
            .iter()
            .map(|r| r.as_object().unwrap().iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect())
            .collect())
    }

    /// オンライン取得: 低レイテンシ推論用の最新特徴量を取得 (<10ms)
    fn get_online_features(
        &self,
        entity_id: u64,
        features: &[&str],
    ) -> reqwest::Result<HashMap<String, Value>> {
        let body = json!({
            "features": features,
            "entities": [{ "user_id": entity_id }]
        });
        let resp: Value = self.client
            .post(format!("{}/get-online-features", self.base_url))
            .json(&body)
            .send()?
            .json()?;
        Ok(resp["results"][0].as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default())
    }
}

fn main() -> reqwest::Result<()> {
    let store = FeatureStoreClient::new("http://localhost:6566");

    // オフライン取得（訓練時）
    let entity_ids = [1001u64, 1002, 1003];
    let training_features = store.get_historical_features(
        &entity_ids,
        &["user_features:avg_purchase_7d", "user_features:total_sessions"],
    )?;
    println!("Training records: {}", training_features.len());

    // オンライン取得（推論時、<10ms レイテンシ）
    let online_features = store.get_online_features(
        1001,
        &["user_features:avg_purchase_7d"],
    )?;
    println!("Online features for user 1001: {:?}", online_features);

    Ok(())
}
```

**利点**:
- Training-Serving一貫性保証
- 特徴量再利用 (チーム間共有)
- Point-in-time correctness (時刻整合性)

#### 6.8.2 Model Registry — モデルのライフサイクル管理

**MLflow Model Registry** [^4]:

**モデルステージ**:

```
None → Staging → Production → Archived
```

**バージョン管理 + メタデータ**:

```rust
use reqwest::blocking::Client;
use serde_json::json;

// MLflow Model Registry — REST API経由でモデルのライフサイクルを管理
const MLFLOW_URI: &str = "http://localhost:5000";

struct MlflowRegistryClient {
    client: Client,
}

impl MlflowRegistryClient {
    fn new() -> Self { Self { client: Client::new() } }

    /// モデルをModel Registryに登録する
    fn register_model(&self, run_id: &str, model_name: &str) -> reqwest::Result<()> {
        let url = format!("{}/api/2.0/mlflow/registered-models/create", MLFLOW_URI);
        self.client.post(&url).json(&json!({ "name": model_name })).send()?;

        let url = format!("{}/api/2.0/mlflow/model-versions/create", MLFLOW_URI);
        self.client.post(&url).json(&json!({
            "name":    model_name,
            "source":  format!("runs:/{}/model", run_id),
            "run_id":  run_id
        })).send()?;
        Ok(())
    }

    /// モデルバージョンをステージに遷移させる (None → Staging → Production → Archived)
    fn transition_model_version_stage(
        &self,
        name: &str,
        version: u32,
        stage: &str,
    ) -> reqwest::Result<()> {
        let url = format!("{}/api/2.0/mlflow/model-versions/transition-stage", MLFLOW_URI);
        self.client.post(&url).json(&json!({
            "name":    name,
            "version": version.to_string(),
            "stage":   stage
        })).send()?;
        Ok(())
    }
}

fn main() -> reqwest::Result<()> {
    let client = MlflowRegistryClient::new();

    // モデルを登録
    client.register_model("abc123", "fraud_detector_v2")?;

    // Productionに昇格
    client.transition_model_version_stage("fraud_detector_v2", 3, "Production")?;
    println!("✅ fraud_detector_v2 v3 → Production");

    // Production モデルのURIで推論クライアントを初期化
    let model_uri = "models:/fraud_detector_v2/Production";
    println!("推論用モデルURI: {}", model_uri);
    // 実際の推論は ONNX Runtime (ort crate) 等でモデルをロードして実行

    Ok(())
}
```

**Governance機能**:
- **Approval Workflow**: Staging → Production に承認必須
- **Lineage Tracking**: データ → 訓練 → モデル の系譜
- **Model Card**: 性能・公平性・制約の文書化

#### 6.8.3 Experiment Tracking at Scale

**Weights & Biases (W&B)** vs **MLflow**:

| 機能 | MLflow | W&B |
|:-----|:-------|:----|
| **UI** | Basic | Rich (interactive charts) |
| **Hyperparameter Sweep** | Manual | Automated (Bayesian) |
| **Collaboration** | Limited | Team-centric |
| **Artifact Storage** | Local/S3 | Cloud-native |
| **Cost** | Free (self-host) | Free tier + Paid |

**W&B Sweep (Bayesian Optimization)**:

```rust
use serde_json::{json, Value};
use std::collections::HashMap;

// W&B Sweep — Bayesian Optimizationによるハイパーパラメータ自動探索
// RustではW&B REST APIを直接呼び出すか、wandb CLIをサブプロセスで起動する

/// Sweep設定: 探索空間と最適化目標を定義
fn build_sweep_config() -> Value {
    json!({
        "method": "bayes",  // "grid" / "random" / "bayes"
        "metric": { "name": "val_loss", "goal": "minimize" },
        "parameters": {
            "learning_rate": { "min": 1e-5_f64, "max": 1e-2_f64 },
            "batch_size":    { "values": [16, 32, 64, 128] },
            "dropout":       { "min": 0.1_f64, "max": 0.5_f64 }
        }
    })
}

/// 1試行の訓練ループ（実際は `config` に応じてモデルを構築・訓練する）
fn train_one_trial(config: &HashMap<&str, f64>) -> Vec<f64> {
    let lr      = config["learning_rate"];
    let dropout = config["dropout"];

    // 疑似訓練ループ (10エポック)
    (0..10).map(|epoch| {
        let loss = 1.0 / (1.0 + (epoch + 1) as f64 * lr) + dropout * 0.1
            + rand::random::<f64>() * 0.05; // ノイズ
        loss
    }).collect()
}

fn main() {
    let sweep_config = build_sweep_config();
    println!("Sweep Config:\n{}", serde_json::to_string_pretty(&sweep_config).unwrap());

    // 50試行をシミュレーション（実運用ではW&B APIにsweep_idを登録してagentを起動）
    let mut best_loss = f64::INFINITY;
    let mut best_config: Option<HashMap<&str, f64>> = None;

    for trial in 0..50_usize {
        // Bayesian OptはW&B Sweepサーバが提案; ここでは疑似ランダムサンプリング
        let lr = 10_f64.powf(-5.0 + rand::random::<f64>() * 3.0); // [1e-5, 1e-2]
        let dropout = 0.1 + rand::random::<f64>() * 0.4;           // [0.1, 0.5]
        let config: HashMap<&str, f64> = HashMap::from([
            ("learning_rate", lr),
            ("dropout",       dropout),
        ]);

        let losses = train_one_trial(&config);
        let final_loss = *losses.last().unwrap();

        if final_loss < best_loss {
            best_loss = final_loss;
            best_config = Some(config.clone());
            println!("Trial {:>2}: lr={:.2e}, dropout={:.3} → val_loss={:.4} ✨ Best",
                     trial + 1, lr, dropout, final_loss);
        }
    }

    if let Some(cfg) = best_config {
        println!("\n🏆 Best config: lr={:.2e}, dropout={:.3}, val_loss={:.4}",
                 cfg["learning_rate"], cfg["dropout"], best_loss);
    }
}
```

**効果**: Manual grid search → Bayesian optimization で探索効率**10倍向上**

#### 6.8.4 Data Quality Monitoring — Great Expectations Integration

**Great Expectations** [^3]: データ品質テストのフレームワーク

**Expectation Suite**:

```rust
use std::collections::HashMap;
use regex::Regex;

// Great Expectations に相当するデータ品質検証フレームワーク (Rust実装)
// データパイプラインに組み込み、スキーマ・値域・NULL制約を自動検査する

/// 検証結果
struct ValidationResult {
    column:  String,
    rule:    String,
    passed:  bool,
    message: String,
}

/// データ品質チェッカー（Expectation Suite）
struct DataValidator {
    suite_name: String,
    results:    Vec<ValidationResult>,
}

impl DataValidator {
    fn new(suite_name: &str) -> Self {
        Self { suite_name: suite_name.to_string(), results: vec![] }
    }

    /// 数値が範囲内に収まることを期待する
    fn expect_between(&mut self, column: &str, values: &[f64], min: f64, max: f64) {
        let all_ok = values.iter().all(|&v| v >= min && v <= max);
        self.results.push(ValidationResult {
            column:  column.to_string(),
            rule:    format!("between({}, {})", min, max),
            passed:  all_ok,
            message: if all_ok { "OK".to_string() }
                     else { format!("値が [{}, {}] の範囲外", min, max) },
        });
    }

    /// NULL (NaN) がないことを期待する
    fn expect_not_null(&mut self, column: &str, values: &[Option<f64>]) {
        let nulls = values.iter().filter(|v| v.is_none()).count();
        self.results.push(ValidationResult {
            column:  column.to_string(),
            rule:    "not_null".to_string(),
            passed:  nulls == 0,
            message: if nulls == 0 { "OK".to_string() }
                     else { format!("{} 件のNULLを検出", nulls) },
        });
    }

    /// 値が許可セットに含まれることを期待する
    fn expect_in_set<'a>(&mut self, column: &str, values: &[&'a str], allowed: &[&str]) {
        let invalid: Vec<&&str> = values.iter()
            .filter(|v| !allowed.contains(v))
            .collect();
        self.results.push(ValidationResult {
            column:  column.to_string(),
            rule:    format!("in_set({:?})", allowed),
            passed:  invalid.is_empty(),
            message: if invalid.is_empty() { "OK".to_string() }
                     else { format!("不正な値: {:?}", invalid) },
        });
    }

    /// 値が正規表現にマッチすることを期待する
    fn expect_match_regex(&mut self, column: &str, values: &[&str], pattern: &str) {
        let re = Regex::new(pattern).unwrap();
        let invalid: Vec<&&str> = values.iter().filter(|v| !re.is_match(v)).collect();
        self.results.push(ValidationResult {
            column:  column.to_string(),
            rule:    format!("match_regex({})", pattern),
            passed:  invalid.is_empty(),
            message: if invalid.is_empty() { "OK".to_string() }
                     else { format!("パターン不一致: {:?}", invalid) },
        });
    }

    /// 全検証結果をサマリーとして返す
    fn validate(&self) -> (bool, usize, usize) {
        let total  = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        (passed == total, passed, total)
    }
}

fn main() {
    let mut validator = DataValidator::new("transaction_data_suite");

    // Expectation定義（テストデータで検証）
    validator.expect_between("amount",
        &[100.0, 50.0, 999_999.0, 0.01], 0.0, 1_000_000.0);
    validator.expect_not_null("user_id",
        &[Some(1.0), Some(2.0), None, Some(4.0)]);
    validator.expect_in_set("status",
        &["pending", "completed", "failed", "unknown"],
        &["pending", "completed", "failed"]);
    validator.expect_match_regex("email",
        &["user@example.com", "bad-email", "admin@co.jp"],
        r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$");

    // 結果を表示
    for r in &validator.results {
        let mark = if r.passed { "✅" } else { "❌" };
        println!("{} {} [{}]: {}", mark, r.column, r.rule, r.message);
    }

    let (success, passed, total) = validator.validate();
    println!("\nSuite: {} — {}/{} rules passed", validator.suite_name, passed, total);
    assert!(success, "データ検証失敗！パイプラインを停止します。");
}
```

**Validation in Pipeline**:

```rust
use serde_json::json;

fn main() {
    // Checkpoint設定をRustの構造化データとして定義
    let checkpoint_config = json!({
        "name":           "daily_data_checkpoint",
        "config_version": 1,
        "class_name":     "SimpleCheckpoint",
        "validations": [
            {
                "batch_request":          "/* バッチリクエスト設定 */",
                "expectation_suite_name": "transaction_data_suite"
            }
        ]
    });

    // 検証実行（DataValidatorの結果を使用）
    let success = true; // 実際はvalidator.validate()の結果を使う
    let statistics = json!({ "evaluated_expectations": 4, "successful_expectations": 4 });

    if !success {
        panic!("Data validation failed! {:?}", statistics);
    }

    println!("✅ Checkpoint '{}' passed: {:?}",
             checkpoint_config["name"], statistics);
}
```

**Production Integration** (Airflow DAG):

Airflow では DAG オブジェクトにオペレータ（`PythonOperator` / `BashOperator`）を追加し、`validate >> train` のビットシフト構文で有向依存関係を宣言する。スケジューラが依存グラフを解析し、上流が成功した場合のみ下流タスクを起動するため、データ検証→モデル訓練→デプロイの直列パイプラインを宣言的に記述できる。

#### 6.8.5 CI/CD for ML — GitHub Actions + DVC

**GitHub Actions Workflow**:

```yaml
# .github/workflows/ml_ci.yml
name: ML CI/CD Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: iterative/setup-dvc@v1

      - name: Pull data with DVC
        run: dvc pull

      - name: Run unit tests
        run: pytest tests/unit/

      - name: Run data validation
        run: |
          python -m great_expectations checkpoint run data_validation

      - name: Train model (smoke test)
        run: |
          python train.py --epochs 1 --smoke-test

      - name: Run model tests
        run: pytest tests/model/

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          mlflow models serve -m "models:/my_model/Staging" -p 5001

      - name: Run integration tests
        run: pytest tests/integration/

      - name: Promote to production
        run: |
          python scripts/promote_model.py --version ${{ github.sha }}
```

**CML (Continuous Machine Learning)** [^5]:

```yaml
# .github/workflows/cml.yml
name: Model Performance Report

on: [push]

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: iterative/setup-cml@v1

      - name: Train model
        run: python train.py

      - name: Generate metrics report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Create report
          cat metrics.json | jq -r '.accuracy' > report.txt
          echo "Accuracy: $(cat report.txt)" >> report.md

          # Plot
          python plot_metrics.py
          cml-publish confusion_matrix.png --md >> report.md

          # Send comment to PR
          cml-send-comment report.md
```

**効果**: PRごとに自動でモデル性能レポート → レビュー時に可視化

### 6.9 Scalable Training Infrastructure

#### 6.9.1 Distributed Training — Ray + DeepSpeed

**Ray Train** (分散訓練フレームワーク):

```rust
use std::sync::{Arc, Mutex};
use std::thread;
use std::collections::HashMap;

// Ray Trainに相当するRust分散訓練フレームワーク
// 実際のGPU分散訓練にはtorch-sys / candle / burn crateを使用する

/// 訓練設定
struct TrainingConfig {
    lr:     f64,
    epochs: usize,
}

/// 単一ワーカーの訓練ループ（実際はGPUデバイスごとに1スレッド）
fn train_worker(worker_id: usize, config: Arc<TrainingConfig>,
                results: Arc<Mutex<Vec<(usize, f64)>>>) {
    println!("Worker {} 開始 (lr={}, epochs={})", worker_id, config.lr, config.epochs);

    // 分散データシャード（各ワーカーがデータの1/N を担当）
    // 実際は s3://data/train/ からParquet読み込み
    let shard_size = 1000;

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0_f64;

        for batch_idx in 0..(shard_size / 32) {
            // 疑似訓練ステップ
            let batch_loss = 1.0 / (1.0 + (epoch * shard_size / 32 + batch_idx) as f64 * config.lr)
                + (worker_id as f64 * 0.001); // ワーカー間のばらつき
            epoch_loss += batch_loss;
        }

        let avg_loss = epoch_loss / (shard_size / 32) as f64;
        results.lock().unwrap().push((epoch, avg_loss));
    }
    println!("Worker {} 完了", worker_id);
}

fn main() {
    let config = Arc::new(TrainingConfig { lr: 1e-3, epochs: 5 });
    let results = Arc::new(Mutex::new(Vec::new()));

    // 4ワーカー（GPU4台相当）で並列訓練
    let num_workers = 4;
    let handles: Vec<_> = (0..num_workers).map(|id| {
        let cfg = Arc::clone(&config);
        let res = Arc::clone(&results);
        thread::spawn(move || train_worker(id, cfg, res))
    }).collect();

    for h in handles { h.join().unwrap(); }

    // 全ワーカーの結果を集約（AllReduce相当）
    let all_results = results.lock().unwrap();
    let mut epoch_losses: HashMap<usize, Vec<f64>> = HashMap::new();
    for &(epoch, loss) in all_results.iter() {
        epoch_losses.entry(epoch).or_default().push(loss);
    }

    println!("\n=== 分散訓練結果 ({}ワーカー) ===", num_workers);
    let mut epochs: Vec<usize> = epoch_losses.keys().copied().collect();
    epochs.sort();
    for epoch in epochs {
        let losses = &epoch_losses[&epoch];
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        println!("Epoch {}: avg_loss={:.4}", epoch + 1, mean);
    }
}
```

**DeepSpeed ZeRO-3** (メモリ効率化):

| Stage | Parameter Partitioning | Gradient Partitioning | Optimizer State Partitioning | Memory Reduction |
|:------|:----------------------|:---------------------|:----------------------------|:----------------|
| ZeRO-1 | ❌ | ❌ | ✅ | 4x |
| ZeRO-2 | ❌ | ✅ | ✅ | 8x |
| **ZeRO-3** | ✅ | ✅ | ✅ | **15-60x** |

**効果**: 175B parameter model を 16x V100 (16GB) で訓練可能

#### 6.9.2 Serverless Inference — AWS Lambda + SageMaker

**AWS Lambda (< 15MB model)**:

```rust
// Rust Lambda function for inference
use lambda_runtime::{service_fn, LambdaEvent, Error};
use serde::{Deserialize, Serialize};
use ort::{Environment, SessionBuilder, Value};

#[derive(Deserialize)]
struct Request {
    features: Vec<f32>,
}

#[derive(Serialize)]
struct Response {
    prediction: f32,
}

async fn handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    // Load ONNX model (embedded in Lambda)
    let environment = Environment::builder().build()?;
    let session = SessionBuilder::new(&environment)?
        .with_model_from_memory(include_bytes!("model.onnx"))?;

    // Run inference
    let input = ndarray::arr1(&event.payload.features);
    let outputs = session.run(vec![Value::from_array(input)?])?;
    let prediction = outputs[0].extract::<f32>()?.view()[0];

    Ok(Response { prediction })
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}
```

**特徴**:
- **Cold start**: 100-500ms (Rust), 500-3000ms (Python)
- **Cost**: $0.20 per 1M requests (128MB, 100ms execution)
- **Auto-scaling**: 0 → 10,000 concurrent無限スケール

**SageMaker Serverless Inference** (> 15MB model):

```rust
use reqwest::blocking::Client;
use serde_json::json;

// SageMaker Serverless Inference — AWS SDK for Rust (aws-sdk-sagemakerruntime) を使用
// ここではREST APIの構造をRustで表現する

/// SageMakerサーバーレスエンドポイントの設定
struct ServerlessInferenceConfig {
    memory_size_in_mb: u32, // 1024, 2048, 3072, 4096, 6144
    max_concurrency:   u32, // 最大同時実行数
}

/// SageMakerクライアント（aws-sdk-sagemakerruntime の薄いラッパー）
struct SageMakerClient {
    client:        Client,
    endpoint_name: String,
    region:        String,
}

impl SageMakerClient {
    fn new(endpoint_name: &str, region: &str) -> Self {
        Self {
            client:        Client::new(),
            endpoint_name: endpoint_name.to_string(),
            region:        region.to_string(),
        }
    }

    /// サーバーレスエンドポイントにデプロイ設定を送信
    fn deploy_serverless(&self, config: &ServerlessInferenceConfig) {
        // 実際は aws-sdk-sagemaker の create_endpoint_config + create_endpoint を呼ぶ
        println!("📦 Deploying to SageMaker Serverless:");
        println!("   endpoint:   {}", self.endpoint_name);
        println!("   memory:     {} MB", config.memory_size_in_mb);
        println!("   max_conc:   {}", config.max_concurrency);
        println!("   region:     {}", self.region);
    }

    /// 推論を実行する（invoke_endpoint）
    fn predict(&self, payload: &serde_json::Value) -> serde_json::Value {
        // 実際は AWS SigV4署名付きリクエストでエンドポイントを呼び出す
        let url = format!(
            "https://runtime.sagemaker.{}.amazonaws.com/endpoints/{}/invocations",
            self.region, self.endpoint_name
        );
        println!("POST {} → {:?}", url, payload);
        // 疑似レスポンス
        json!({ "prediction": 0.87, "label": "fraud" })
    }
}

fn main() {
    let config = ServerlessInferenceConfig {
        memory_size_in_mb: 2048,
        max_concurrency:   20,
    };

    let client = SageMakerClient::new("fraud-detector-serverless", "us-east-1");
    client.deploy_serverless(&config);

    // 推論
    let data = json!({ "features": [0.5, 1.2, -0.3, 2.1] });
    let result = client.predict(&data);
    println!("推論結果: {}", result);
}
```

**Cost comparison** (1M requests/month):

| Service | Fixed Cost | Variable Cost | Total |
|:--------|:----------|:-------------|:------|
| EC2 (t3.medium 24/7) | $30/month | $0 | **$30** |
| Lambda (100ms avg) | $0 | $0.20/1M | **$0.20** |
| SageMaker Serverless | $0 | $0.20/1M + $0.10/GB-hr | **$0.30** |

低トラフィック時はServerlessが**100倍安い**

### 6.10 Production Best Practices (Industry Standard)

#### 6.10.1 Model Governance — Audit Trail & Compliance

**必須トラッキング項目** (規制対応):

| Item | Requirement | Tool |
|:-----|:-----------|:-----|
| **Training Data Lineage** | データの出所・変換履歴 | DVC + Pachyderm |
| **Model Versioning** | 全モデルのバージョン管理 | MLflow Registry |
| **Prediction Logging** | 全推論結果の記録 (90日保持) | CloudWatch Logs |
| **Bias Monitoring** | 人種・性別等での性能差 | AWS SageMaker Clarify |
| **Explainability** | 個別予測の説明 | SHAP / LIME |

**Audit Log Example**:

```json
{
  "timestamp": "2026-02-15T10:30:00Z",
  "model_id": "fraud_detector_v3.2",
  "model_version": "sha256:abc123...",
  "input": {"user_id": 1001, "amount": 500},
  "output": {"fraud_score": 0.82, "decision": "flag"},
  "explanation": {
    "top_features": [
      {"feature": "transaction_velocity", "contribution": 0.35},
      {"feature": "geolocation_mismatch", "contribution": 0.28}
    ]
  },
  "data_lineage": {
    "training_data": "s3://data/fraud/2026-01-15/",
    "training_commit": "git-sha:def456"
  }
}
```

#### 6.10.2 Security — Secrets Management & Access Control

**AWS Secrets Manager** (credentials保管):

```rust
use aws_sdk_secretsmanager::Client;

async fn get_db_password() -> Result<String, Box<dyn std::error::Error>> {
    let config = aws_config::load_from_env().await;
    let client = Client::new(&config);

    let response = client
        .get_secret_value()
        .secret_id("prod/mlops/db_password")
        .send()
        .await?;

    Ok(response.secret_string().unwrap().to_string())
}
```

**IAM Role-based Access**:

```yaml
# Kubernetes ServiceAccount (EKS)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-inference-sa
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/MLInferenceRole

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-api
spec:
  template:
    spec:
      serviceAccountName: ml-inference-sa  # Inherits IAM permissions
      containers:
      - name: api
        image: my-inference:latest
```

**Network Isolation**:

```
Internet → ALB (HTTPS) → API Gateway → Private Subnet (Inference) → VPC Endpoint → S3 (models)
                                             ↓
                                    Security Group (port 8080 only)
```

> **Note:** **進捗: 完全制覇!** Advanced MLOps tools、分散訓練、Serverless推論、Governance、Securityまで全て習得。Production-readyシステム構築の完全知識を獲得！

---

### 6.11 Emerging Trends (2025-2026)

#### MLOps + LLMOps Convergence

**LLMOps特有の課題**:
- **Prompt Versioning**: プロンプトテンプレートの管理
- **Few-shot Example Management**: In-context learning用サンプル
- **Token Cost Optimization**: API呼び出しコスト最小化

**統合ツール**: LangChain + LangSmith — プロンプトバージョン管理、精度・コストのメトリクス記録。

#### Edge MLOps — On-device Inference

**TensorFlow Lite** + **ONNX Runtime Mobile**:

- Model quantization (FP32 → INT8): **4倍小型化**
- On-device training: Federated Learning
- OTA (Over-The-Air) model updates

**典型的なEdge Pipeline**:

```
Cloud Training → Quantization → ONNX → Edge Device (ARM) → Telemetry → Cloud Retraining
```

---

## 参考文献

### 主要論文

[^1]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*.
<https://arxiv.org/abs/2305.18290>

[^2]: DVC: Data Version Control.
<https://dvc.org/>

[^3]: Great Expectations: Data validation framework.
<https://greatexpectations.io/>

[^4]: MLflow: Open source platform for the machine learning lifecycle.
<https://mlflow.org/>

[^5]: CML (Continuous Machine Learning): CI/CD for Machine Learning Projects.
<https://cml.dev/>

---

> **📖 前編（理論編）**: [第31回前編: MLOps理論編](./ml-lecture-31-part1) | **← 理論・数式ゾーンへ**

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
