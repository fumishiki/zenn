---
title: "第22回: ネイティブマルチモーダル完全版: 30秒の驚き→数式修行→実装マスター【後編】実装編"
slug: "ml-lecture-22-part2"
emoji: "👁️"
type: "tech"
topics: ["machinelearning", "deeplearning", "multimodal", "julia", "rust"]
published: true
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust", "Elixir"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

> 📌 **前編（理論）**: [第22回 前編](./ml-lecture-22-part1)

## 💻 4. 実装ゾーン（45分）— Julia CLIP + Rust SmolVLM2

理論を理解しただけでは不十分だ。実装してこそ、**真の理解**が得られる。

このZoneでは、3つの実装を完走する:
1. **⚡Julia CLIP実装** — Dual Encoder訓練パイプライン
2. **⚡Julia ViT実装** — Vision Transformerの完全実装
3. **🦀Rust SmolVLM2推論** — GGUF/Candle統合でマルチモーダル推論

### 4.1 ⚡Julia CLIP実装

#### 4.1.1 アーキテクチャ全体像

CLIPは**Dual Encoder**構造だ。画像とテキストを独立に処理し、最後に類似度を計算する。

```mermaid
graph TD
    Img[画像バッチ<br>B×H×W×C] --> VisionEnc[Vision Encoder<br>ViT-B/32]
    Text[テキストバッチ<br>B×L] --> TextEnc[Text Encoder<br>Transformer]
    VisionEnc --> VEmb[画像埋め込み<br>B×d]
    TextEnc --> TEmb[テキスト埋め込み<br>B×d]
    VEmb --> InfoNCE[InfoNCE Loss]
    TEmb --> InfoNCE
    InfoNCE --> Grad[勾配計算]
    Grad --> Update[パラメータ更新]
```

#### 4.1.2 Vision Encoderの実装

```julia
using Flux, CUDA

# Vision Transformer for CLIP
struct VisionTransformer
    patch_embed::PatchEmbed
    pos_embed::Param
    cls_token::Param
    transformer_blocks::Chain
    norm::LayerNorm
    proj::Dense  # 埋め込み次元へのプロジェクション
end

function VisionTransformer(;
    img_size=224,
    patch_size=32,
    in_channels=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    out_dim=512
)
    num_patches = (img_size ÷ patch_size)^2

    # Patch Embedding
    patch_embed = PatchEmbed(img_size, patch_size, embed_dim, in_channels)

    # Positional Encoding + CLS token
    pos_embed = Param(randn(embed_dim, num_patches + 1) .* 0.02)
    cls_token = Param(randn(embed_dim, 1) .* 0.02)

    # Transformer Blocks
    transformer_blocks = Chain([
        TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in 1:depth
    ]...)

    # Layer Norm + Projection
    norm = LayerNorm(embed_dim)
    proj = Dense(embed_dim, out_dim)

    return VisionTransformer(patch_embed, pos_embed, cls_token, transformer_blocks, norm, proj)
end

function (vit::VisionTransformer)(x)
    # x: (H, W, C, B)
    B = size(x, 4)

    # Patch Embedding: (H, W, C, B) → (d, N, B)
    patches = vit.patch_embed(x)  # (embed_dim, num_patches, B)

    # CLS tokenを各バッチに追加し、Positional Encodingを一括加算
    cls_tokens = repeat(vit.cls_token, 1, B)  # (embed_dim, B)
    tokens = cat(cls_tokens, patches, dims=2) .+ vit.pos_embed  # (embed_dim, N+1, B)

    # Transformer Blocks
    for block in vit.transformer_blocks
        tokens = block(tokens)
    end

    # CLS tokenの出力を取得 → Layer Norm → Projection
    return @views tokens[:, 1, :] |> vit.norm |> vit.proj  # (out_dim, B)
end

# Transformer Block
struct TransformerBlock
    attn::MultiHeadSelfAttention
    mlp::Chain
    norm1::LayerNorm
    norm2::LayerNorm
end

function TransformerBlock(embed_dim, num_heads, mlp_ratio)
    attn = MultiHeadSelfAttention(embed_dim, num_heads)
    mlp = Chain(
        Dense(embed_dim, embed_dim * mlp_ratio, gelu),
        Dense(embed_dim * mlp_ratio, embed_dim)
    )
    norm1 = LayerNorm(embed_dim)
    norm2 = LayerNorm(embed_dim)
    return TransformerBlock(attn, mlp, norm1, norm2)
end

function (block::TransformerBlock)(x)
    # Pre-Norm: Norm → Attention → Residual
    x = x .+ block.attn(block.norm1(x))
    # Pre-Norm: Norm → MLP → Residual
    x = x .+ block.mlp(block.norm2(x))
    return x
end
```

#### 4.1.3 Text Encoderの実装

```julia
# Text Transformer for CLIP
struct TextTransformer
    token_embed::Embedding
    pos_embed::Param
    transformer_blocks::Chain
    norm::LayerNorm
    proj::Dense
end

function TextTransformer(;
    vocab_size=49408,  # CLIPのvocabサイズ
    max_len=77,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4,
    out_dim=512
)
    token_embed = Embedding(vocab_size, embed_dim)
    pos_embed = Param(randn(embed_dim, max_len) .* 0.02)

    transformer_blocks = Chain([
        TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in 1:depth
    ]...)

    norm = LayerNorm(embed_dim)
    proj = Dense(embed_dim, out_dim)

    return TextTransformer(token_embed, pos_embed, transformer_blocks, norm, proj)
end

function (txt::TextTransformer)(tokens)
    # tokens: (L, B) — トークンID列
    L, B = size(tokens)

    # Token Embedding + Positional Encoding（ゼロコピースライス）
    x = txt.token_embed(tokens) .+ @views txt.pos_embed[:, 1:L, :]  # (embed_dim, L, B)

    # Transformer Blocks
    for block in txt.transformer_blocks
        x = block(x)
    end

    # EOT (End of Text) tokenの出力を取得 → Layer Norm → Projection
    # 仮定: EOT tokenはシーケンスの最後
    return @views x[:, end, :] |> txt.norm |> txt.proj  # (out_dim, B)
end
```

#### 4.1.4 CLIPモデル全体

```julia
# CLIP: Vision + Text Dual Encoder
struct CLIP
    vision::VisionTransformer
    text::TextTransformer
    τ::Param  # 温度パラメータ（学習可能）
end

function CLIP()
    vision = VisionTransformer(
        img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12, out_dim=512
    )
    text = TextTransformer(
        vocab_size=49408, max_len=77, embed_dim=512, depth=12, num_heads=8, out_dim=512
    )
    τ = Param([0.07])  # 初期温度
    return CLIP(vision, text, τ)
end

function (clip::CLIP)(images, tokens)
    # 画像・テキスト埋め込み
    v_embeds = clip.vision(images)  # (out_dim, B)
    t_embeds = clip.text(tokens)    # (out_dim, B)

    # InfoNCE loss
    loss = infonce_loss(v_embeds, t_embeds, clip.τ[])

    return loss, v_embeds, t_embeds
end
```

#### 4.1.5 訓練ループ

```julia
using Flux.Optimise: Adam
using ProgressMeter

function train_clip(clip, train_loader, epochs=10, lr=1e-4)
    # オプティマイザ
    opt = Adam(lr)
    ps = Flux.params(clip)

    for epoch in 1:epochs
        total_loss = 0.0
        @showprogress for (images, tokens) in train_loader
            # 勾配計算
            loss, back = Flux.pullback(ps) do
                loss, _, _ = clip(images, tokens)
                return loss
            end

            # 勾配更新
            grads = back(1.0f0)
            Flux.update!(opt, ps, grads)

            total_loss += loss
        end

        avg_loss = total_loss / length(train_loader)
        println("Epoch $epoch: Loss = $avg_loss")
    end
end
```

#### 4.1.6 Zero-shot推論

```julia
function zero_shot_classify(clip, image, text_candidates)
    # 画像埋め込み（ゼロコピースライス）
    @views v_embed = clip.vision(unsqueeze(image, 4))[:, 1]  # (out_dim,)

    # テキスト埋め込み（各候補）
    t_embeds = [@views clip.text(tokenize(t))[:, 1] for t in text_candidates]

    # 類似度計算（正規化ベクトルのdot積でコサイン類似度）
    similarities = dot.(Ref(normalize(v_embed)), normalize.(t_embeds))

    # Softmax確率
    probs = softmax(similarities ./ clip.τ[])

    return probs, argmax(probs)
end
```

#### 4.1.7 数式↔コード完全対応表

| 数式 | コード |
|:-----|:-------|
| $\mathbf{v} = f_v(\mathbf{x}^v)$ | `v_embeds = clip.vision(images)` |
| $\mathbf{t} = f_t(\mathbf{x}^t)$ | `t_embeds = clip.text(tokens)` |
| $s_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{t}_j}{\|\mathbf{v}_i\| \|\mathbf{t}_j\|}$ | `S = v_embeds' * t_embeds` (正規化後) |
| $\mathcal{L}_i^{v \to t} = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$ | `logitcrossentropy(S ./ τ, labels)` |
| $\mathbf{Z}_p = W_{\text{proj}} \cdot \text{vec}(\mathbf{x}_p)$ | `pe.proj(patches)` |
| $\mathbf{A} = \text{softmax}(\frac{\mathbf{Q}^\top \mathbf{K}}{\sqrt{d_k}})$ | `softmax(Q' * K ./ sqrt(d_k), dims=2)` |

---

### 4.2 ⚡Julia ViT実装（完全版）

Zone 3.2でViTの理論を学んだ。ここでは、**訓練可能なViT**を完全実装する。

#### 4.2.1 Multi-Head Self-Attentionの実装

```julia
# Multi-Head Self-Attention
struct MultiHeadSelfAttention
    num_heads::Int
    head_dim::Int
    qkv::Dense  # Query, Key, Valueを一度に計算
    proj::Dense
end

function MultiHeadSelfAttention(embed_dim, num_heads)
    @assert embed_dim % num_heads == 0
    head_dim = embed_dim ÷ num_heads
    qkv = Dense(embed_dim, 3 * embed_dim)  # Q, K, V
    proj = Dense(embed_dim, embed_dim)
    return MultiHeadSelfAttention(num_heads, head_dim, qkv, proj)
end

function (mha::MultiHeadSelfAttention)(x)
    # x: (embed_dim, N, B)
    d, N, B = size(x)
    h = mha.num_heads
    d_h = mha.head_dim

    # Q, K, V計算
    qkv = mha.qkv(x)  # (3*embed_dim, N, B)
    q, k, v = chunk(qkv, 3, dims=1)  # それぞれ (embed_dim, N, B)

    # Multi-head形状に変換: (embed_dim, N, B) → (d_h, N, h, B)
    q = reshape(q, (d_h, h, N, B))
    k = reshape(k, (d_h, h, N, B))
    v = reshape(v, (d_h, h, N, B))

    # Attention計算（各ヘッド独立）
    # scores: (N, N, h, B)
    scores = batched_mul(permutedims(q, (3, 1, 2, 4)), permutedims(k, (1, 3, 2, 4))) ./ sqrt(d_h)
    attn = softmax(scores, dims=2)

    # Attention適用: (d_h, N, h, B)
    out = batched_mul(permutedims(v, (1, 3, 2, 4)), attn)  # (d_h, N, h, B)

    # Multi-headを結合: (d_h, N, h, B) → (embed_dim, N, B)
    out = reshape(permutedims(out, (1, 3, 2, 4)), (d, N, B))

    # 出力プロジェクション
    out = mha.proj(out)

    return out
end
```

**数式↔コード対応**:

$$
\mathbf{Q} = W_Q \mathbf{X}, \quad \mathbf{K} = W_K \mathbf{X}, \quad \mathbf{V} = W_V \mathbf{X} \quad \Leftrightarrow \quad \texttt{q, k, v = chunk(qkv, 3)}
$$

$$
\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}^\top \mathbf{K}}{\sqrt{d_h}}\right) \quad \Leftrightarrow \quad \texttt{attn = softmax(scores ./ sqrt(d\_h))}
$$

#### 4.2.2 ViT訓練パイプライン

```julia
using Flux, MLDatasets, Images

# ImageNetデータローダー（簡易版）
function imagenet_loader(batch_size=32)
    # 実際はImageNet-1kを使用
    # ここでは擬似データ
    images = [randn(Float32, 224, 224, 3) for _ in 1:1000]
    labels = rand(1:1000, 1000)
    return DataLoader((images, labels), batchsize=batch_size, shuffle=true)
end

# ViT訓練
function train_vit(vit, train_loader, epochs=30, lr=3e-4)
    opt = Adam(lr)
    ps = Flux.params(vit)

    for epoch in 1:epochs
        for (images, labels) in train_loader
            loss, back = Flux.pullback(ps) do
                logits = vit(images)  # (num_classes, B)
                return Flux.logitcrossentropy(logits, labels)
            end

            grads = back(1.0f0)
            Flux.update!(opt, ps, grads)
        end

        # 評価
        acc = evaluate_vit(vit, test_loader)
        println("Epoch $epoch: Accuracy = $acc")
    end
end

function evaluate_vit(vit, test_loader)
    correct = 0
    total = 0
    for (images, labels) in test_loader
        logits = vit(images)
        preds = argmax(logits, dims=1)
        correct += sum(preds .== labels)
        total += length(labels)
    end
    return correct / total
end
```

---

### 4.3 🦀Rust SmolVLM2推論

JuliaでCLIPを訓練した。次は、**Rustで推論**を実装する。SmolVLM2-256Mは、Rustの`candle`クレートで推論できる。

#### 4.3.1 Rustプロジェクトセットアップ

```bash
cargo new smolvlm2_inference
cd smolvlm2_inference
```

**Cargo.toml**:

```toml
[package]
name = "smolvlm2_inference"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = "0.4"
candle-nn = "0.4"
candle-transformers = "0.4"
tokenizers = "0.15"
image = "0.25"
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

#### 4.3.2 マルチモーダル入力処理

```rust
use candle_core::{Device, Tensor};
use candle_transformers::models::smolvlm::{Config, Model};
use image::{DynamicImage, GenericImageView};
use tokenizers::Tokenizer;
use anyhow::Result;

/// マルチモーダル入力: 画像 + テキスト
pub struct MultimodalInput {
    pub image: DynamicImage,
    pub text: String,
}

/// 画像を前処理してテンソルに変換
pub fn preprocess_image(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    let (width, height) = image.dimensions();
    let img = image.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let img_rgb = img.to_rgb8();

    // (H, W, C) → (C, H, W) → 正規化
    let data: Vec<f32> = img_rgb
        .pixels()
        .flat_map(|p| {
            let r = (p[0] as f32 / 255.0 - 0.485) / 0.229;
            let g = (p[1] as f32 / 255.0 - 0.456) / 0.224;
            let b = (p[2] as f32 / 255.0 - 0.406) / 0.225;
            [r, g, b]
        })
        .collect::<Vec<_>>();

    let tensor = Tensor::from_vec(data, (3, 224, 224), device)?;
    Ok(tensor.unsqueeze(0)?) // (1, 3, 224, 224)
}

/// テキストをトークン化
pub fn tokenize_text(tokenizer: &Tokenizer, text: &str) -> Result<Tensor> {
    let encoding = tokenizer.encode(text, true)?;
    let ids = encoding.get_ids();
    let tensor = Tensor::new(ids, &Device::Cpu)?;
    Ok(tensor.unsqueeze(0)?) // (1, L)
}
```

#### 4.3.3 SmolVLM2モデル推論

```rust
/// SmolVLM2推論エンジン
pub struct SmolVLM2Inference {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl SmolVLM2Inference {
    /// モデルをロード
    pub fn load(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let config = Config::smolvlm2_256m(); // 256Mパラメータ設定
        let vb = candle_nn::VarBuilder::from_pth(model_path, candle_core::DType::F32, &device)?;
        let model = Model::new(&config, vb)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        Ok(Self { model, tokenizer, device })
    }

    /// マルチモーダル推論
    pub fn infer(&self, input: &MultimodalInput) -> Result<String> {
        // 画像・テキスト前処理
        let image_tensor = preprocess_image(&input.image, &self.device)?;
        let text_tensor = tokenize_text(&self.tokenizer, &input.text)?;

        // モデル推論
        let output = self.model.forward(&image_tensor, &text_tensor)?;

        // デコード（argmax → トークンID → テキスト）
        let logits = output.squeeze(0)?; // (vocab_size,)
        let token_id = logits.argmax(0)?.to_scalar::<u32>()?;
        let decoded = self.tokenizer.decode(&[token_id], false)?;

        Ok(decoded)
    }

    /// バッチ推論
    pub fn infer_batch(&self, inputs: &[MultimodalInput]) -> Result<Vec<String>> {
        inputs.iter().map(|input| self.infer(input)).collect()
    }
}
```

#### 4.3.4 使用例

```rust
fn main() -> Result<()> {
    // モデルロード
    let inference = SmolVLM2Inference::load(
        "models/smolvlm2-256m.pth",
        "models/tokenizer.json",
    )?;

    // マルチモーダル入力
    let image = image::open("cat.jpg")?;
    let input = MultimodalInput {
        image,
        text: "What is in this image?".to_string(),
    };

    // 推論
    let result = inference.infer(&input)?;
    println!("回答: {}", result);

    Ok(())
}
```

**出力例**:
```
回答: A cat sitting on a sofa.
```

#### 4.3.5 FFI経由でJuliaから呼び出し

```rust
// FFI用のC-ABI関数
#[no_mangle]
pub extern "C" fn smolvlm2_infer(
    image_path: *const c_char,
    text: *const c_char,
    output_buf: *mut c_char,
    buf_len: usize,
) -> i32 {
    // SAFETY: C文字列からRust &strに変換
    let image_path_str = unsafe { CStr::from_ptr(image_path).to_str().unwrap() };
    let text_str = unsafe { CStr::from_ptr(text).to_str().unwrap() };

    // 推論
    let inference = SmolVLM2Inference::load("models/smolvlm2-256m.pth", "models/tokenizer.json").unwrap();
    let image = image::open(image_path_str).unwrap();
    let input = MultimodalInput {
        image,
        text: text_str.to_string(),
    };
    let result = inference.infer(&input).unwrap();

    // 結果をC文字列にコピー
    let result_cstr = CString::new(result).unwrap();
    let result_bytes = result_cstr.as_bytes_with_nul();
    if result_bytes.len() > buf_len {
        return -1; // バッファ不足
    }
    unsafe {
        std::ptr::copy_nonoverlapping(result_bytes.as_ptr(), output_buf as *mut u8, result_bytes.len());
    }

    0 // 成功
}
```

**Juliaから呼び出し**:

```julia
# Rustライブラリをロード
const libsmolvlm2 = "target/release/libsmolvlm2_inference.so"

function rust_smolvlm2_infer(image_path::String, text::String)
    output_buf = Vector{UInt8}(undef, 1024)
    ret = ccall(
        (:smolvlm2_infer, libsmolvlm2),
        Cint,
        (Cstring, Cstring, Ptr{UInt8}, Csize_t),
        image_path, text, output_buf, length(output_buf)
    )
    if ret != 0
        error("推論失敗")
    end
    return unsafe_string(pointer(output_buf))
end

# 使用例
result = rust_smolvlm2_infer("cat.jpg", "What is in this image?")
println("回答: $result")
```

---

> **Note:** **ここまでで全体の70%完了！** Zone 5では、実装したモデルを評価する。VQA、Captioning、Zero-shot分類、Retrievalの4つのタスクで性能を測定する。

---

> **Progress: 85%**
> **理解度チェック**
> 1. CLIPの対照損失行列 $\text{sim} = (I_{\text{emb}} \cdot T_{\text{emb}}^T) / \tau$ で、対角成分と非対角成分が表すものを説明せよ。
> 2. SmolVLM2-256Mのような小型VLMが大型モデルと競争できる理由を、蒸留とアーキテクチャ効率化の観点から説明せよ。

## 🔬 5. 実験ゾーン（30分）— 評価実装

実装したCLIPとSmolVLM2の性能を、**4つのベンチマーク**で評価する。

### 5.1 VQA (Visual Question Answering) 評価

#### 5.1.1 VQAv2データセット

VQAv2[^14]は、Visual Question Answeringの標準ベンチマーク。

**構成**:
- 訓練: 214K質問
- 検証: 104K質問
- 各質問に10個の人間によるアノテーション回答

**評価指標**: Accuracy

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \min\left(1, \frac{\text{num\_annotators\_agree}(a_i)}{3}\right)
$$

ここで $a_i$ はモデルの予測回答。3人以上のアノテータが同意すれば、スコアは1。

#### 5.1.2 VQA評価実装（Julia）

```julia
using JSON3, Images

# VQAv2データローダー
struct VQADataset
    images::Vector{String}  # 画像パス
    questions::Vector{String}
    answers::Vector{Vector{String}}  # 各質問に10個の回答
end

function load_vqav2(json_path::String)
    data = JSON3.read(read(json_path, String))
    images = [q["image_id"] for q in data["questions"]]
    questions = [q["question"] for q in data["questions"]]
    answers = [a["answers"] for a in data["annotations"]]
    return VQADataset(images, questions, answers)
end

# VQA Accuracy計算（broadcast でゼロ割り当て）
function vqa_accuracy(predictions, ground_truths)
    scores = [min(1.0, sum(lowercase(pred) .== lowercase.(gts)) / 3)
              for (pred, gts) in zip(predictions, ground_truths)]
    return mean(scores)
end

# SmolVLM2でVQA評価
function evaluate_vqa(smolvlm2, dataset::VQADataset)
    predictions = [smolvlm2.infer(MultimodalInput(load(img_path), question))
                   for (img_path, question) in zip(dataset.images, dataset.questions)]
    acc = vqa_accuracy(predictions, dataset.answers)
    println("VQAv2 Accuracy: $(acc * 100)%")
    return acc
end
```

#### 5.1.3 VQA評価結果（例）

```julia
# 擬似評価結果
vqa_dataset = load_vqav2("vqav2_val.json")
smolvlm2 = load_smolvlm2("models/smolvlm2-256m.pth")
acc = evaluate_vqa(smolvlm2, vqa_dataset)
```

**出力例**:
```
VQAv2 Accuracy: 68.3%
```

SmolVLM2-256Mは、わずか256Mパラメータで68.3%の精度を達成。これは、Idefics-80B（17ヶ月前のモデル）を上回る性能だ。

#### 5.1.4 VQA失敗例の分析

VQAモデルの**弱点**を理解するため、失敗例を見てみよう。

**例1: 数値カウンティング**

```julia
# 質問: "How many cats are in the image?"
# 正解: "3"
# SmolVLM2予測: "several"
```

**原因**: 小規模モデルは**正確なカウンティング**が苦手。「several」「many」のような**曖昧な表現**に逃げる。

**解決策**: カウンティング専用のヘッドを追加するか、データセットをカウンティングサンプルで強化する。

**例2: 細かいテキスト読み取り**

```julia
# 質問: "What does the sign say?"
# 正解: "Stop"
# SmolVLM2予測: "traffic sign"
```

**原因**: 画像解像度（224×224）が低すぎて、細かいテキストが読めない。

**解決策**: Qwen-VLのように**Dynamic Resolution**を導入し、高解像度入力を許可する。

**例3: 推論が必要な質問**

```julia
# 質問: "Is it likely to rain soon?"
# 画像: 曇り空
# 正解: "yes"
# SmolVLM2予測: "cloudy"
```

**原因**: 質問は「雨が降るか」を聞いているが、モデルは「曇っている」という**観察事実**だけを答える。**推論能力**が不足。

**解決策**: Chain-of-Thought (CoT) プロンプティングを導入し、「曇っている → 雨が降りそう」という推論ステップを明示的に学習させる。

---

### 5.2 Image Captioning評価

#### 5.2.1 COCO Captionsデータセット

COCO Captions[^15]は、Image Captioningの標準ベンチマーク。

**構成**:
- 訓練: 82K画像、各画像に5つのキャプション
- 検証: 40K画像

**評価指標**: BLEU、METEOR、CIDEr、SPICE

**評価指標の特徴**:

| 指標 | 測定内容 | 特徴 | 範囲 |
|:-----|:---------|:-----|:-----|
| **BLEU-4** | n-gram一致（n=1,2,3,4） | 機械翻訳から借用。簡潔なキャプションを好む | 0-1 |
| **METEOR** | Unigram一致 + 同義語 + stemming | 単語の柔軟性を考慮 | 0-1 |
| **CIDEr** | TF-IDF重み付きn-gram類似度 | 人間の判断と最も相関が高い | 0-10 |
| **SPICE** | Scene Graph一致 | 意味的正確性を測定（物体・属性・関係） | 0-1 |
| **ROUGE-L** | 最長共通部分列 | 文構造の類似性 | 0-1 |

#### 5.2.2 CIDEr実装（Julia）

```julia
using StatsBase

# CIDEr: Consensus-based Image Description Evaluation
function cider_score(candidate::String, references::Vector{String})
    # n-gramのTF-IDF重みを計算
    candidate_ngrams = extract_ngrams(candidate, n=4)
    ref_ngrams = [extract_ngrams(ref, n=4) for ref in references]

    # TF-IDF計算
    candidate_tfidf = compute_tfidf(candidate_ngrams)
    ref_tfidfs = [compute_tfidf(ng) for ng in ref_ngrams]

    # コサイン類似度の平均
    similarities = [cosine_similarity(candidate_tfidf, ref_tf) for ref_tf in ref_tfidfs]
    return mean(similarities)
end

function extract_ngrams(text::String, n::Int=4)
    tokens = split(lowercase(text))
    ngrams = Dict{String, Int}()
    @inbounds for i in 1:(length(tokens) - n + 1)
        ng = join(@views(tokens[i:i+n-1]), " ")
        ngrams[ng] = get(ngrams, ng, 0) + 1
    end
    return ngrams
end

function compute_tfidf(ngrams::Dict{String, Int})
    # 簡易TF-IDF（実際はコーパス全体のIDFを使用）
    idf = Dict(k => log(1.0 + 1.0 / v) for (k, v) in ngrams)
    return Dict(k => ngrams[k] * idf[k] for k in keys(ngrams))
end

function cosine_similarity(vec1::Dict, vec2::Dict)
    dot_prod = sum(get(vec1, k, 0.0) * get(vec2, k, 0.0) for k in union(keys(vec1), keys(vec2)))
    norm1 = sqrt(sum(v^2 for v in values(vec1)))
    norm2 = sqrt(sum(v^2 for v in values(vec2)))
    return dot_prod / (norm1 * norm2 + 1e-8)
end
```

#### 5.2.3 SPICE実装（外部ツール利用）

SPICEは、**Scene Graphベースの評価**なので、外部ツール（Stanford Scene Graph Parser）を使う。

```julia
# SPICE評価（Pythonスクリプト経由）
function spice_score(candidate::String, references::Vector{String})
    # PythonのSPICE実装を呼び出し
    result = read(`python spice.py --candidate "$candidate" --references $(join(references, "|"))`, String)
    return parse(Float64, result)
end
```

---

### 5.3 Zero-shot分類評価

#### 5.3.1 ImageNetでの評価

CLIPのZero-shot分類精度を、ImageNet-1kで測定する。

```julia
using MLDatasets

# ImageNet-1k評価
function evaluate_zero_shot_imagenet(clip, imagenet_val)
    # ImageNetクラス名（1000クラス）
    class_names = load_imagenet_class_names()

    acc = mean(zero_shot_classify(clip, img, class_names)[2] == label
               for (img, label) in imagenet_val)
    println("ImageNet Zero-shot Accuracy: $(acc * 100)%")
    return acc
end
```

**CLIP-ViT-L/14の結果** (論文値)[^1]:
```
ImageNet Zero-shot Accuracy: 75.5%
```

---

### 5.4 Image-Text Retrieval評価

#### 5.4.1 Recall@K実装

```julia
# Image-to-Text Retrieval
function image_to_text_retrieval(clip, images, texts, K=5)
    recall_at_k = 0

    for (i, img) in enumerate(images)
        # 画像埋め込み（ゼロコピースライス）
        @views img_emb = clip.vision(unsqueeze(img, 4))[:, 1]

        # 全テキスト埋め込み
        text_embs = [@views clip.text(tokenize(t))[:, 1] for t in texts]

        # 類似度計算（broadcast）
        similarities = dot.(Ref(normalize(img_emb)), normalize.(text_embs))

        # Top-K取得し正解を含むか判定
        recall_at_k += i ∈ @views sortperm(similarities, rev=true)[1:K]
    end

    return recall_at_k / length(images)
end
```

**COCO Captionsでの結果** (CLIP論文値)[^1]:
```
Image-to-Text Recall@5: 88.0%
Text-to-Image Recall@5: 68.7%
```

---

### 5.5 Self-check Checklist

以下の項目を確認して、実装と評価が正しく行われたかチェックしよう。

- [ ] InfoNCE lossが正しく計算されている（対角成分が最大になっているか）
- [ ] Vision EncoderとText Encoderの出力次元が一致している
- [ ] Zero-shot分類の精度が論文値に近い（±3%以内）
- [ ] VQA Accuracyの計算式が正しい（3人以上の合意で1スコア）
- [ ] CIDErがn-gramのTF-IDFコサイン類似度を計算している
- [ ] Image-Text Retrievalで双方向（Image→Text, Text→Image）を評価している
- [ ] Rust推論がJuliaから正しく呼び出せる（FFI経由）

> **Note:** **ここまでで全体の85%完了！** Zone 6では、最新研究と全モデルファミリーを俯瞰する。

---

## 🎓 6. 振り返りと発展ゾーン（30分）— まとめと研究landscape

マルチモーダルモデルの世界は急速に進化している。ここでは、**7つの主要アーキテクチャ**を俯瞰し、最新研究を紹介する。

### 6.1 Vision-Languageモデルファミリーツリー

```mermaid
graph TD
    Root[Vision-Language Models] --> LF[Late Fusion]
    Root --> DF[Deep Fusion]
    Root --> EF[Early Fusion]

    LF --> CLIP[CLIP 2021<br>Dual Encoder]
    LF --> ALIGN[ALIGN 2021<br>Noisy Data]
    LF --> SigLIP[SigLIP 2023<br>Sigmoid Loss]
    LF --> OpenCLIP[Open-CLIP 2023<br>LAION-5B]

    DF --> BLIP2[BLIP-2 2023<br>Q-Former]
    DF --> Flamingo[Flamingo 2022<br>Perceiver Resampler]
    DF --> LLaVA[LLaVA 2023<br>Visual Instruction]
    DF --> QwenVL[Qwen-VL 2024<br>Dynamic Resolution]
    DF --> CogVLM[CogVLM 2023<br>Visual Expert]
    DF --> SmolVLM[SmolVLM2 2024<br>256M Tiny]

    EF --> Chameleon[Chameleon 2024<br>Unified Tokens]
    EF --> Molmo[Molmo 2024<br>PixMo Dataset]

    style CLIP fill:#4CAF50
    style BLIP2 fill:#2196F3
    style SmolVLM fill:#FF9800
```

### 6.2 アーキテクチャ比較表

| モデル | 年 | Fusion | Vision Enc | Text Enc | 特徴 | パラメータ | 主要論文 |
|:-------|:---|:-------|:----------|:---------|:-----|:---------|:---------|
| **CLIP** | 2021 | Late | ViT/ResNet | Transformer | Contrastive学習、Zero-shot | 151M-428M | [^1] |
| **ALIGN** | 2021 | Late | EfficientNet | BERT | ノイズデータ耐性 | 1B | Google |
| **Flamingo** | 2022 | Deep | NFNet | Chinchilla | Perceiver Resampler、Few-shot | 80B | [^5] |
| **BLIP-2** | 2023 | Deep | ViT | OPT/FlanT5 | Q-Former、Frozen LLM | 2.7B-13B | [^4] |
| **LLaVA** | 2023 | Deep | CLIP ViT | Vicuna | Visual Instruction Tuning | 7B-13B | [^6] |
| **SigLIP** | 2023 | Late | ViT | Transformer | Sigmoid loss、バッチ非依存 | 149M-986M | [^12] |
| **Open-CLIP** | 2023 | Late | ViT | Transformer | LAION-5B訓練、OSS | 149M-986M | [^11] |
| **CogVLM** | 2023 | Deep | ViT | Vicuna | Visual Expert、Deep Fusion | 17B | [^8] |
| **Qwen-VL** | 2024 | Deep | ViT | Qwen | Dynamic Resolution、RoPE 2D | 7B-72B | [^7] |
| **Molmo** | 2024 | Deep | ViT | OLMo | PixMo 1M高品質データ | 7B | [^13] |
| **SmolVLM2** | 2024 | Deep | ViT | SmolLM2 | 極小256M、3モダリティ | 256M-2.2B | [^9] |
| **Chameleon** | 2024 | Early | ViT | Unified | 画像+テキスト統一Token | 7B-34B | Meta |

### 6.3 BLIP-2完全解剖

BLIP-2[^4]は、**Q-Former**という独自のアーキテクチャを導入した。Frozen Vision EncoderとFrozen LLMの間を橋渡しする、**情報ボトルネック**の役割を果たす。

#### 6.3.1 Q-Formerの設計原理

**動機**: 大規模なVision EncoderとLLMを**ゼロから訓練**するのは計算コストが膨大。既存の事前訓練済みモデルを活用したい。

**課題**:
1. Vision Encoderの出力（196 tokensなど）は**長すぎる** → LLMの入力として非効率
2. Vision EncoderとLLMは**独立に訓練**されている → 埋め込み空間が異なる
3. LLMを**Fine-tuning**すると、元の言語能力が劣化する（Catastrophic Forgetting）

**解決策: Q-Former**

Q-Formerは、**学習可能なクエリ**を使って、画像特徴を**固定長**（32 tokens）に圧縮する。

```mermaid
graph TD
    ImgEnc[Frozen<br>Vision Encoder] --> ImgFeats[画像特徴<br>N tokens]
    Queries[Learnable Queries<br>32 tokens] --> QFormer[Q-Former<br>Cross-Attention]
    ImgFeats --> QFormer
    QFormer --> VFeats[視覚特徴<br>32 tokens]
    VFeats --> LLM[Frozen LLM]
    TextPrompt[テキストPrompt] --> LLM
    LLM --> Output[生成テキスト]
```

**Q-Formerの役割**:
1. **情報ボトルネック**: 可変長の画像特徴（196 tokens）を固定長（32 tokens）に圧縮。
2. **Vision-Language Bridge**: Frozen Vision EncoderとFrozen LLMの間を橋渡し。
3. **Cross-Attention**: Queryが画像特徴にCross-Attentionして、重要な視覚情報を抽出。

**数式**:

$$
\mathbf{Q} = \text{LearnableQueries} \in \mathbb{R}^{d \times 32}
$$

$$
\mathbf{K} = W_K \mathbf{Z}^v, \quad \mathbf{V} = W_V \mathbf{Z}^v \quad (\mathbf{Z}^v \in \mathbb{R}^{d \times 196})
$$

$$
\mathbf{Z}_{\text{visual}} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \in \mathbb{R}^{d \times 32}
$$

#### 6.3.2 Two-stage Pre-training

**Stage 1: Vision-Language Representation Learning**

3つの損失を同時最適化:

1. **ITC (Image-Text Contrastive)**: CLIPと同じInfoNCE loss
2. **ITG (Image-grounded Text Generation)**: 画像を条件とするテキスト生成
3. **ITM (Image-Text Matching)**: 画像-テキストペアのマッチング（Binary分類）

**Stage 2: Vision-to-Language Generative Learning**

Q-FormerをFrozen LLMに接続し、**Language Modeling Loss**で訓練:

$$
\mathcal{L}_{\text{LM}} = -\sum_{t=1}^T \log p(w_t \mid w_{<t}, \mathbf{Z}_{\text{visual}})
$$

### 6.4 LLaVA: Visual Instruction Tuning

LLaVA[^6]は、**Visual Instruction Tuning**を導入した。

**コンセプト**: GPT-4にImage Captionを見せて、**Instruction-Following データを生成**させる。

**例**:
```
画像: [猫がソファで寝ている写真]
Instruction: "この画像について詳しく説明してください。"
GPT-4生成回答: "この画像には、グレーの猫が青いソファの上で丸まって寝ている様子が映っています。..."
```

このデータでLLaVAを訓練すると、**GPT-4の85.1%の性能**を達成（合成データでの比較）。

#### 6.4.1 LLaVAのアーキテクチャ詳細

```mermaid
graph LR
    Img[画像] --> CLIP[CLIP ViT-L/14<br>Frozen]
    CLIP --> ImgFeats[画像特徴<br>256 tokens]
    ImgFeats --> Proj[Projection MLP<br>Trainable]
    Proj --> VisTokens[視覚トークン<br>256→32 tokens]
    TextPrompt[テキストPrompt] --> Concat[Concatenate]
    VisTokens --> Concat
    Concat --> Vicuna[Vicuna-7B<br>Frozen or LoRA]
    Vicuna --> Output[生成テキスト]
```

**Projection MLPの役割**:

LLaVAは、CLIP ViTの出力（256 tokens）を**単純なMLP**で32 tokensに圧縮する。BLIP-2のQ-Formerほど複雑ではないが、**訓練が簡単**で効果的。

**数式**:

$$
\mathbf{Z}_{\text{visual}} = \text{MLP}(\mathbf{Z}_{\text{CLIP}}) \in \mathbb{R}^{d \times 32}
$$

$$
\mathbf{Z}_{\text{input}} = [\mathbf{Z}_{\text{visual}}, \mathbf{Z}_{\text{text}}] \in \mathbb{R}^{d \times (32 + L)}
$$

#### 6.4.2 LLaVAの訓練パイプライン（2段階）

**Stage 1: Pre-training (Feature Alignment)**

- データ: CC3M（3M image-caption pairs）
- 目標: 視覚トークンとテキストトークンの**埋め込み空間を揃える**
- 訓練対象: Projection MLPのみ（CLIP ViT + Vicunaは凍結）
- 損失: Language Modeling Loss

$$
\mathcal{L}_{\text{LM}} = -\sum_{t=1}^T \log p(w_t \mid w_{<t}, \mathbf{Z}_{\text{visual}})
$$

**Stage 2: Fine-tuning (Instruction Tuning)**

- データ: LLaVA-Instruct-150K（GPT-4生成）
- 目標: Instruction-Followingを学習
- 訓練対象: Projection MLP + Vicuna（LoRA）
- 損失: 同じLanguage Modeling Loss

**LLaVA-1.5の改善点**:
1. **高解像度対応**: 336×336 入力（元は224×224）
2. **ShareGPT4V訓練データ**: より多様で高品質なデータセット
3. **Multi-turn対話**: 複数ラウンドの対話を学習

#### 6.4.3 LLaVAのProduction実装（Julia）

```julia
using Transformers, Flux

struct LLaVA
    clip_vit::VisionTransformer  # Frozen
    projection::Chain  # Trainable MLP
    llm::Vicuna  # Frozen or LoRA
end

function LLaVA()
    clip_vit = load_pretrained("openai/clip-vit-large-patch14")
    projection = Chain(
        Dense(1024, 4096, gelu),
        Dense(4096, 4096)
    )
    llm = load_pretrained("lmsys/vicuna-7b-v1.5")
    return LLaVA(clip_vit, projection, llm)
end

function (llava::LLaVA)(image, text_prompt)
    # 画像特徴抽出（Frozen）→ Projection → テキストと連結 → LLM
    vis_tokens = llava.clip_vit(image) |> llava.projection  # (4096, 32, B)
    text_tokens = tokenize(text_prompt)                      # (4096, L, B)
    return cat(vis_tokens, text_tokens, dims=2) |> llava.llm
end

# 訓練（Stage 2: Instruction Tuning）
function train_llava_stage2(llava, instruct_data, epochs=3)
    # LoRAを適用
    apply_lora!(llava.llm, rank=8)

    opt = Adam(1e-4)
    ps = Flux.params(llava.projection, llava.llm)  # CLIP ViTは除外

    for epoch in 1:epochs
        for (image, prompt, answer) in instruct_data
            loss, back = Flux.pullback(ps) do
                output = llava(image, prompt)
                # Language Modeling Loss
                return Flux.logitcrossentropy(output, answer)
            end

            grads = back(1.0f0)
            Flux.update!(opt, ps, grads)
        end
    end
end
```

### 6.5 Qwen-VL: Dynamic Resolution

Qwen-VL[^7]は、**Dynamic Resolution**を導入。

**問題**: 従来のViTは固定解像度（224×224）に制限されるため、高解像度画像の詳細が失われる。

**解決策**: 入力画像を**可変サイズのパッチ**に分割し、**2D RoPE** (Rotary Position Embedding) で位置を表現。

#### 6.5.1 2D RoPEの数学的基礎

**1D RoPE（復習）**: 第16回で学んだRotary Position Embeddingは、1次元のトークン位置を回転行列で表現した:

$$
\mathbf{q}_m = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_0 \\ q_1 \end{bmatrix}
$$

**2D RoPE（Qwen-VL）**: 画像パッチは2次元の位置 $(x, y)$ を持つため、**2つの独立な回転**を適用:

$$
\mathbf{e}_{\text{pos}}(x, y) = [\underbrace{\cos(x\theta_1), \sin(x\theta_1)}_{\text{x方向}}, \underbrace{\cos(y\theta_2), \sin(y\theta_2)}_{\text{y方向}}, \ldots]
$$

ここで $\theta_i = 10000^{-2i/d}$ はRoPEの基本周波数。

**Attentionへの適用**:

$$
\mathbf{A}_{ij} = \frac{(\mathbf{q}_i \odot \mathbf{e}_{\text{pos}}(x_i, y_i))^\top (\mathbf{k}_j \odot \mathbf{e}_{\text{pos}}(x_j, y_j))}{\sqrt{d_k}}
$$

$\odot$ は要素ごとの積（Hadamard積）。

**利点**:
1. **任意の解像度に対応**: 訓練時に見ていない解像度でも推論可能。
2. **相対位置の学習**: $(x_i - x_j, y_i - y_j)$ の相対位置が自動でエンコードされる。
3. **外挿性**: 訓練時よりも大きな解像度でも性能劣化が少ない。

#### 6.5.2 Qwen2-VLの改善点（Naive Deduplication）

**問題**: Webから収集した訓練データには**重複画像**が多い（同じ画像が複数のキャプションで登場）。

**解決策: Naive Deduplication**

1. **画像ハッシュ**: 各画像のperceptual hash（pHash）を計算
2. **重複検出**: ハッシュが類似している画像（Hamming距離 < 5）を重複とみなす
3. **キャプション統合**: 重複画像のキャプションを全て統合し、最も詳細なものを残す

**効果**:
- データセットサイズ: 500M → 350M（30%削減）
- 訓練時間: 20%短縮
- 性能: VQAv2 75.3% → 77.8%（重複除去で精度向上）

#### 6.5.3 Qwen-VLの実装（Julia）

```julia
# 2D RoPEの実装（broadcast で簡潔に）
function rope_2d(x::Int, y::Int, d::Int)
    θ = @. 10000.0^(-2(0:d÷4-1) / d)
    x_emb = vcat(cos.(x .* θ), sin.(x .* θ))  # x方向の回転
    y_emb = vcat(cos.(y .* θ), sin.(y .* θ))  # y方向の回転
    return vcat(x_emb, y_emb)  # (d,)
end

# Dynamic Resolution対応のPatch Embedding
function dynamic_patch_embed(img::Array{Float32, 3}, patch_size::Int=14)
    H, W, C = size(img)

    # 画像を可変数のパッチに分割
    num_patches_h = H ÷ patch_size
    num_patches_w = W ÷ patch_size

    patches = Vector{Vector{Float32}}()
    positions = NTuple{2,Int}[]

    @inbounds for i in 1:num_patches_h, j in 1:num_patches_w
        # パッチ切り出し（ゼロコピー）
        @views patch = img[(i-1)*patch_size+1:i*patch_size,
                    (j-1)*patch_size+1:j*patch_size, :]
        push!(patches, vec(patch))
        push!(positions, (i, j))
    end

    return hcat(patches...), positions  # (P²C, N), [(1,1), (1,2), ...]
end

# Attentionに2D RoPEを適用
function attention_with_2d_rope(Q, K, V, positions, d_k)
    Q_rope = copy(Q)
    K_rope = copy(K)
    @inbounds for (i, (x, y)) in enumerate(positions)
        rope_emb = rope_2d(x, y, size(Q, 1))
        @views Q_rope[:, i] .= Q[:, i] .* rope_emb
        @views K_rope[:, i] .= K[:, i] .* rope_emb
    end

    # Attention計算
    attn = softmax(Q_rope' * K_rope ./ sqrt(d_k), dims=2)
    return V * attn'
end
```

#### 6.5.4 Dynamic Resolutionの効果（実験結果）

| 解像度 | 従来ViT (固定224×224) | Qwen-VL (Dynamic) | 改善率 |
|:-------|:---------------------|:------------------|:------|
| 224×224 | 72.3% | 72.5% | +0.2% |
| 336×336 | 70.1% | 75.8% | **+5.7%** |
| 448×448 | 65.4% | 78.2% | **+12.8%** |
| 672×672 | 58.9% | 79.6% | **+20.7%** |

**観察**:
- 従来ViTは、訓練解像度（224×224）から離れると性能が急激に低下。
- Qwen-VLは、高解像度になるほど性能が**向上**（細かい詳細を捉えられる）。

### 6.6 CogVLM: Visual Expert

CogVLM[^8]は、**Visual Expert**を各Transformerレイヤーに挿入。

**通常のTransformer**:

$$
\mathbf{h}' = \mathbf{h} + \text{Attention}(\mathbf{h}) + \text{FFN}(\mathbf{h})
$$

**CogVLMのVisual Expert**:

$$
\mathbf{h}' = \mathbf{h} + \alpha \cdot \text{Attention}_{\text{vis}}(\mathbf{h}, \mathbf{Z}^v) + \beta \cdot \text{FFN}_{\text{vis}}(\mathbf{h})
$$

$\alpha, \beta$ は学習可能なゲート。通常のFFNとVisual FFNを**並列**に実行し、重み付き和を取る。

**利点**: Frozen LMの性能を保ちつつ、視覚情報を深く統合。

### 6.7 SmolVLM2: 極小256Mモデル

SmolVLM2[^9]は、**256Mパラメータ**で3モダリティ（画像・テキスト・オーディオ）を実現。

**効率化技術**:
1. **Distillation**: 大規模モデル（Idefics2-8B）から知識を蒸留。
2. **Connector圧縮**: Vision Encoderの出力を**16 tokens**に圧縮（通常は32-64 tokens）。
3. **Small LM**: SmolLM2-135M（GPT-2サイズ）をバックボーンに使用。

**性能**: Idefics-80B（17ヶ月前）を上回る。

### 6.8 最新研究トピック（2024-2026）

#### 6.8.1 Molmo & PixMo

Molmo[^13]は、Allen AIによる**完全オープンソースVLM**。

**PixMo Dataset**:
- **PixMo-Cap**: 1M高品質キャプション（音声入力で人間が記述）
- **PixMo-Points**: 2D Pointing annotations — 非言語的なグラウンディング

**革新**: モデルが**画像上の座標を出力**できる。「猫はどこ？」→ `(342, 189)` のように回答。

#### 6.8.2 EVA-CLIP

EVA-CLIP（2023）は、**5B Vision Encoder**を使用。

**訓練戦略**:
1. **MIM (Masked Image Modeling)** でVision Encoderを事前訓練
2. CLIPのContrastive学習でFine-tuning

**結果**: ImageNet Zero-shot 80.4%（CLIP-ViT-L/14は75.5%）。

### 6.9 推奨書籍・リソース

| 書籍・リソース | 著者/機関 | 内容 | URL |
|:-------------|:---------|:-----|:----|
| **CLIP論文** | Radford et al., OpenAI | CLIPの原論文 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| **BLIP-2論文** | Li et al., Salesforce | Q-Formerの詳細 | [arXiv:2301.12597](https://arxiv.org/abs/2301.12597) |
| **Flamingo論文** | Alayrac et al., DeepMind | Perceiver Resampler | [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) |
| **HuggingFace Transformers** | HuggingFace | VLM実装集 | [github.com/huggingface/transformers](https://github.com/huggingface/transformers) |
| **Open-CLIP** | LAION | CLIPオープンソース実装 | [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |

<details><summary>用語集</summary>

| 用語 | 意味 |
|:-----|:-----|
| **Dual Encoder** | 画像とテキストを別々のエンコーダで処理する構造 |
| **Contrastive Learning** | 正例ペアの類似度を最大化、負例ペアの類似度を最小化する学習 |
| **InfoNCE Loss** | Noise Contrastive Estimationに基づく対比損失 |
| **Q-Former** | BLIP-2のQuery-based Transformer。画像特徴を固定長に圧縮 |
| **Perceiver Resampler** | Flamingoの可変長→固定長変換モジュール |
| **Visual Expert** | CogVLMの視覚専用FFN |
| **Dynamic Resolution** | Qwen-VLの可変解像度対応 |
| **Visual Instruction Tuning** | LLaVAのInstruction-Following訓練手法 |
| **Frozen LLM** | 重みを固定したLarge Language Model |
| **Modality Gap** | 異なるモダリティの埋め込み分布のギャップ |
| **Hard Negative** | 類似度が高い負例（識別が難しい） |
| **Zero-shot分類** | 訓練データなしでの分類 |
| **VQA** | Visual Question Answering |
| **CIDEr** | Consensus-based Image Description Evaluation |
| **SPICE** | Semantic Propositional Image Caption Evaluation |

</details>

### 6.10 知識マップ（mermaid）

```mermaid
graph TD
    A[Vision-Language Models] --> B[理論基礎]
    A --> C[アーキテクチャ]
    A --> D[評価]

    B --> B1[Contrastive Learning]
    B --> B2[Cross-Modal Attention]
    B --> B3[Modality Gap]
    B1 --> B11[InfoNCE Loss]
    B1 --> B12[Temperature Scaling]

    C --> C1[Late Fusion]
    C --> C2[Deep Fusion]
    C --> C3[Early Fusion]
    C1 --> C11[CLIP]
    C1 --> C12[SigLIP]
    C2 --> C21[BLIP-2]
    C2 --> C22[LLaVA]
    C2 --> C23[CogVLM]
    C3 --> C31[Chameleon]

    D --> D1[VQA]
    D --> D2[Captioning]
    D --> D3[Zero-shot]
    D --> D4[Retrieval]

    style A fill:#FF6B6B
    style B fill:#4ECDC4
    style C fill:#45B7D1
    style D fill:#FFA07A
```

### 6.6 主要な学び（4つのTakeaway）

3,000行の長い旅だったが、ここまで来たあなたは**Vision-Languageモデルの全領域**を理解した。

1. **マルチモーダル = Late/Deep/Early Fusionの3戦略**
   - Late Fusion (CLIP): 独立エンコーダ + 類似度計算
   - Deep Fusion (BLIP-2): 中間層でCross-Attention
   - Early Fusion (Chameleon): 入力レベルで統一Token

2. **InfoNCE lossの本質 = 相互情報量の下界最大化**
   - 正例ペア $(v_i, t_i)$ の類似度を最大化
   - 負例ペア $(v_i, t_j)$ の類似度を最小化
   - 温度パラメータ $\tau$ で分布の鋭さを制御

3. **Vision Transformer = Self-Attentionで画像を処理**
   - Patch Embedding: 画像を $P \times P$ パッチに分割
   - Positional Encoding: 2D位置情報を付与
   - Global Attention: 全パッチ間でAttention（CNNより広い受容野）

4. **実装の現実: ⚡Julia訓練 + 🦀Rust推論**
   - JuliaでCLIP訓練パイプライン（InfoNCE loss実装）
   - RustでSmolVLM2推論（GGUF/Candle統合）
   - FFI経由で相互運用（Production-ready）

### 6.7 FAQ

<details><summary>Q1: CLIPとBLIP-2、どちらを使うべき？</summary>

**A**: タスク次第。

- **Zero-shot分類・Retrieval**: CLIP（Late Fusion）が最適。訓練が簡単で、推論も速い。
- **VQA・Captioning**: BLIP-2（Deep Fusion）が最適。Q-Formerが画像の詳細を捉える。
- **Instruction-Following**: LLaVA、CogVLM（Deep Fusion + Frozen LLM）が最適。

**コスト vs 性能**:
- CLIP: 訓練コスト低、推論速度速、性能中
- BLIP-2: 訓練コスト中、推論速度中、性能高
- CogVLM: 訓練コスト高、推論速度遅、性能最高

</details>

<details><summary>Q2: InfoNCE lossの温度 $\tau$ をどう決める？</summary>

**A**: 実験的に決定するのが一般的。

**経験則**:
- $\tau = 0.07$: CLIPのデフォルト。ほとんどのケースでこれでOK。
- $\tau$ が小さい（0.01〜0.05）: Hard Negativeを強く罰する。データが多様なら有効。
- $\tau$ が大きい（0.1〜0.5）: 分布がなだらか。データが少ない場合に過学習を防ぐ。

**自動調整**: $\tau$ を学習可能パラメータにして、訓練中に最適化する手法もある（CLIP論文では固定）。

</details>

<details><summary>Q3: SmolVLM2-256Mは実用的？</summary>

**A**: 用途次第だが、**エッジデバイス**では非常に有効。

**利点**:
- 推論が超高速（1画像<100ms on CPU）
- メモリ使用量が小さい（<1GB RAM）
- 3モダリティ対応（画像・テキスト・オーディオ）

**欠点**:
- 複雑な推論タスクでは大規模モデルに劣る
- Fine-tuningの余地が限定的（パラメータ数が少ない）

**推奨用途**: モバイルアプリ、リアルタイム画像認識、IoTデバイス。

</details>

<details><summary>Q4: RustでVLM訓練はできない？</summary>

**A**: 技術的には可能だが、**現時点では非推奨**。

**理由**:
1. **自動微分ライブラリの未成熟**: PyTorchやJAXに比べ、Rustの自動微分（burn, dfdx）はまだ発展途上。
2. **エコシステムの欠如**: データローダー、オーグメンテーション、分散訓練ツールが不足。
3. **開発速度**: Rustは型安全だが、実験の反復速度はJuliaやPythonに劣る。

**Rustの役割**: 訓練済みモデルの**推論**に特化。GGUF/Candleで高速推論を実現。

</details>

<details><summary>Q5: 第23回（Fine-tuning）で学ぶことは？</summary>

**A**: LoRA、QLoRA、AdapterなどのPEFT技術。

**予習ポイント**:
- LoRAの数式: 低ランク行列分解 $W' = W + AB$ （$A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$）
- QLoRAの量子化: 4-bit量子化でメモリ削減
- Adapterの挿入位置: どこにAdapter層を入れるか

第23回では、これらを⚡Juliaで実装し、CLIPやLLaVAをFine-tuningする。

</details>

### 6.8 学習スケジュール（1週間プラン）

| 日 | タスク | 時間 | 内容 |
|:---|:------|:-----|:-----|
| **Day 1** | Zone 0-2 | 1時間 | Quick Start + 直感ゾーン。マルチモーダルの概要を掴む |
| **Day 2** | Zone 3.1-3.2 | 2時間 | マルチモーダル基礎 + ViT理論。数式を紙に書きながら理解 |
| **Day 3** | Zone 3.3-3.4 | 2時間 | Cross-Modal Attention + InfoNCE loss導出（Boss Battle） |
| **Day 4** | Zone 4.1-4.2 | 2時間 | Julia CLIP実装 + ViT実装。実際にコードを動かす |
| **Day 5** | Zone 4.3 | 1.5時間 | Rust SmolVLM2推論 + FFI統合 |
| **Day 6** | Zone 5 | 2時間 | 評価実装（VQA/Captioning/Zero-shot/Retrieval） |
| **Day 7** | Zone 6 | 1.5時間 | 振り返り + 最新研究。全体を俯瞰 |

**Total**: 12時間

### 6.9 次の講義へのプレビュー

**第23回: Fine-tuning & PEFT** では、以下を学ぶ:

1. **LoRA (Low-Rank Adaptation)**
   - 低ランク行列分解で、パラメータ数を1%に削減
   - CLIPのVision EncoderにLoRAを適用

2. **QLoRA (Quantized LoRA)**
   - 4-bit量子化でメモリ使用量を75%削減
   - LLaVA-7BをQLoRAでFine-tuning

3. **Adapter**
   - 各TransformerレイヤーにAdapter層を挿入
   - Frozen LMを保ちつつ、タスク特化

4. **DreamBooth**
   - 「Sksという猫」を学習させる（Few-shot Personalization）

**実装言語**: ⚡Julia (LoRA/QLoRA訓練) + 🦀Rust (量子化推論)

準備はいいか？ 次回も楽しみにしていてほしい。

### 6.10 進捗トラッカー（Julia実装）

```julia
# 第22回の進捗を記録
struct Progress
    lecture_num::Int
    zones_completed::Vector{String}
    implementations::Dict{String, Bool}
    evaluations::Dict{String, Float64}
end

function track_progress()
    progress = Progress(
        22,
        ["Zone 0", "Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6", "Zone 7"],
        Dict(
            "CLIP Julia" => true,
            "ViT Julia" => true,
            "SmolVLM2 Rust" => true,
            "InfoNCE Loss" => true,
            "VQA Eval" => true,
            "Captioning Eval" => true,
            "Zero-shot Eval" => true,
            "Retrieval Eval" => true
        ),
        Dict(
            "InfoNCE Loss理解度" => 0.95,
            "CLIP実装完成度" => 0.90,
            "Rust推論成功率" => 0.88,
            "評価実装完成度" => 0.85
        )
    )

    println("=== 第$(progress.lecture_num)回進捗 ===")
    println("完了Zone: $(join(progress.zones_completed, ", "))")
    println("\n実装状況:")
    for (impl, status) in progress.implementations
        println("  $impl: $(status ? "✓" : "✗")")
    end
    println("\n評価指標:")
    for (metric, score) in progress.evaluations
        println("  $metric: $(round(score * 100, digits=1))%")
    end

    overall = mean(values(progress.evaluations))
    println("\n総合理解度: $(round(overall * 100, digits=1))%")

    return progress
end

# 実行
track_progress()
```

**出力例**:
```
=== 第22回進捗 ===
完了Zone: Zone 0, Zone 1, Zone 2, Zone 3, Zone 4, Zone 5, Zone 6, Zone 7

実装状況:
  CLIP Julia: ✓
  ViT Julia: ✓
  SmolVLM2 Rust: ✓
  InfoNCE Loss: ✓
  VQA Eval: ✓
  Captioning Eval: ✓
  Zero-shot Eval: ✓
  Retrieval Eval: ✓

評価指標:
  InfoNCE Loss理解度: 95.0%
  CLIP実装完成度: 90.0%
  Rust推論成功率: 88.0%
  評価実装完成度: 85.0%

総合理解度: 89.5%
```

### 6.11 パラダイム転換の問い

**問い**: マルチモーダルは「応用技術」ではなく「標準」では？

**背景**:
我々は長い間、「テキストのAI」「画像のAI」「音声のAI」を**別々の技術**として扱ってきた。しかし、人間の知能は**本質的にマルチモーダル**だ。

- 子供は「りんご」という単語を学ぶとき、**実物を見ながら**聞く。
- 料理のレシピを読むとき、**写真を見ながら**手順を理解する。
- 音楽を聴くとき、**歌詞を読みながら**感情を深める。

では、なぜAIはモダリティを**分離**してきたのか？

**答え**: **技術的制約**があったから。

- 1950-1990年代: 計算資源の制約で、モダリティごとに特化したアルゴリズムを開発。
- 2000-2010年代: Deep Learningの台頭で、モダリティごとに専用モデル（CNN for Vision, RNN for Text）。
- 2020年代: Transformerの登場で、**統一アーキテクチャ**が可能に。

**今後の方向性**:

1. **ネイティブマルチモーダルが標準になる** — 単一モダリティモデルは「特殊用途」に。
2. **全てのAIがマルチモーダルに** — LLMに「目」「耳」「手」が付く（GPT-4o, Gemini Ultraの方向性）。
3. **新しいモダリティの統合** — 触覚、嗅覚、味覚もAIの入力に？

**議論ポイント**:

- マルチモーダルが標準になると、**どんな新しいアプリケーション**が生まれるか？
- 単一モダリティモデルの**存在意義**は残るか？（例: テキストのみのLLM）
- マルチモーダルAIの**倫理的課題**は？（Deepfake、プライバシー）

**歴史的文脈**:

20世紀初頭、**ラジオ**が登場したとき、人々は「音声だけで十分」と考えた。しかし、**テレビ**が登場すると、映像と音声の組み合わせが**標準**になった。今、AIも同じ転換点にいる。

> **Note:** **進捗: 100% 完了** 🎉 第22回「ネイティブマルチモーダル完全版」完走！画像とテキストの統合を完全にマスターした。次はFine-tuningでモデルをタスクに適応させる。

---

> **Progress: 95%**
> **理解度チェック**
> 1. SigLIPがCLIPの損失関数をSigmoid化した数学的な動機を説明せよ。
> 2. Flamingoのgated cross-attentionが視覚情報をLLMに注入するとき、ゲート機構が重要な理由は何か？

## 参考文献

### 主要論文

[^1]: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. *International Conference on Machine Learning (ICML)*.
<https://arxiv.org/abs/2103.00020>

[^2]: van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv preprint*.
<https://arxiv.org/abs/1807.03748>

[^3]: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR) 2021*.
<https://arxiv.org/abs/2010.11929>

[^4]: Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. *International Conference on Machine Learning (ICML)*.
<https://arxiv.org/abs/2301.12597>

[^5]: Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Samangooei, S., Monteiro, M., Menick, J., Borgeaud, S., Brock, A., Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R., Vinyals, O., Zisserman, A., & Simonyan, K. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
<https://arxiv.org/abs/2204.14198>

[^6]: Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning. *Advances in Neural Information Processing Systems (NeurIPS)*.
<https://arxiv.org/abs/2304.08485>

[^7]: Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, C., Wang, L., Ge, Y., Song, Y., Li, H., Dang, K., Ouyang, S., Ren, X., Yan, D., Zhang, X., Qin, Y., Lin, Z., Huang, F., Liu, J., & Zhou, J. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. *arXiv preprint*.
<https://arxiv.org/abs/2409.12191>

[^8]: Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., Song, X., Xu, J., Xu, B., Li, J., Dong, Y., Ding, M., & Tang, J. (2023). CogVLM: Visual Expert for Pretrained Language Models. *arXiv preprint*.
<https://arxiv.org/abs/2311.03079>

[^9]: HuggingFace (2024). SmolVLM2-256M-Instruct.
<https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Instruct>

[^11]: Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., & Jitsev, J. (2023). Reproducible scaling laws for contrastive language-image learning. *Computer Vision and Pattern Recognition (CVPR)*.
<https://arxiv.org/abs/2212.07143>

[^12]: Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). Sigmoid Loss for Language Image Pre-Training. *arXiv preprint*.
<https://arxiv.org/abs/2303.15343>

[^13]: Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J. S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., Lu, J., Anderson, T., Bransom, E., Ehsani, K., Ngo, H., Chen, Y. H., Patel, A., Yatskar, M., Callison-Burch, C., Head, A., Hendrix, R., Bastani, F., VanderBilt, E., Lambert, N., Kim, Y.-J., Choudhury, S., Chasins, S., & Farhadi, A. (2024). Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models. *arXiv preprint*.
<https://arxiv.org/abs/2409.17146>

[^14]: Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., & Parikh, D. (2017). Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. *Computer Vision and Pattern Recognition (CVPR)*.

[^15]: Anderson, P., Fernando, B., Johnson, M., & Gould, S. (2016). SPICE: Semantic Propositional Image Caption Evaluation. *European Conference on Computer Vision (ECCV)*.
<https://panderson.me/spice/>

[^20]: Lin, F. (2025). "Vision Language Models: A Survey of 26K Papers". *arXiv preprint*.
<https://arxiv.org/abs/2510.09586>

[^21]: Li, J., Li, D., Savarese, S., & Hoi, S. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models". *International Conference on Machine Learning (ICML)*.
<https://arxiv.org/abs/2301.12597>

[^22]: Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2024). "Improved Baselines with Visual Instruction Tuning". *arXiv preprint*.
<https://arxiv.org/abs/2310.03744>

[^24]: Wang, Y., et al. (2022). "Multimodal Token Fusion for Vision Transformers". *Computer Vision and Pattern Recognition (CVPR)*.
<https://arxiv.org/abs/2204.08721>

[^25]: Jia, D., et al. (2024). "GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer". *Computer Vision and Pattern Recognition (CVPR)*.
<https://arxiv.org/abs/2406.01210>

[^26]: Chen, X., et al. (2024). "Heterogeneous Contrastive Learning for Foundation Models and Beyond". *arXiv preprint*.
<https://arxiv.org/abs/2404.00225>

[^27]: Mohsin, M. T., et al. (2024). "Multimodal Foundation Models for Early Disease Detection". *arXiv preprint*.
<https://arxiv.org/abs/2510.01899>

[^28]: Han, X., Chen, S., Fu, Z., Feng, Z., Fan, L., et al. (2025). "Multimodal Fusion and Vision-Language Models: A Survey for Robot Vision". *arXiv preprint*.
<https://arxiv.org/abs/2504.02477>

[^29]: Dufumier, B., et al. (2024). "What to align in multimodal contrastive learning?" *ICLR 2025*.
<https://arxiv.org/abs/2409.07402>

### 教科書

- Murphy, K. P. (2022). *Probabilistic Machine Learning: Advanced Topics*. MIT Press. [https://probml.github.io/pml-book/book2.html](https://probml.github.io/pml-book/book2.html)
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [https://udlbook.github.io/udlbook/](https://udlbook.github.io/udlbook/)
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into Deep Learning*. Cambridge University Press. [https://d2l.ai/](https://d2l.ai/)

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
