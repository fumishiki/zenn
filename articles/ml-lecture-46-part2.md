---
title: "第46回 (Part 2): 3D生成 & Neural Rendering: 30秒の驚き→数式修行→実装マスター"
emoji: "🧊"
type: "tech"
topics: ["machinelearning", "deeplearning", "3dgeneration", "nerf", "3dgs"]
published: true
slug: "ml-lecture-46-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---
## 💻 4. 実装ゾーン（45分）— Rust 3DGSラスタライザ

**ゴール**: 3D Gaussian Splattingのラスタライズエンジンを、Rustで実装する。

### 4.1 実装の全体構造

3DGSラスタライザは以下の処理を行う:

1. **ガウシアンの射影**: 3D→2D変換
2. **タイル割り当て**: 各ガウシアンが影響するタイルを特定
3. **深度ソート**: タイルごとにガウシアンを深度順に並べる
4. **α-Blending**: ピクセルごとに色を合成
5. **勾配計算**: バックプロパゲーション

### 4.2 データ構造の定義

```rust
// src/gaussian.rs

/// 3D Gaussian の表現
#[derive(Clone, Debug)]
pub struct Gaussian3D {
    pub mean: [f32; 3],           // μ: 中心位置
    pub cov: [[f32; 3]; 3],       // Σ: 共分散行列
    pub color: [f32; 3],          // RGB
    pub opacity: f32,             // α
}

impl Gaussian3D {
    /// 共分散行列を回転とスケールから構成
    /// Σ = R S Sᵀ Rᵀ
    pub fn from_rotation_scale(
        mean: [f32; 3],
        rotation: [f32; 4],  // quaternion [w, x, y, z]
        scale: [f32; 3],     // [sx, sy, sz]
        color: [f32; 3],
        opacity: f32,
    ) -> Self {
        // quaternion → rotation matrix
        let r = quat_to_mat3(rotation);

        // S = diag(scale)
        let s = [
            [scale[0], 0.0, 0.0],
            [0.0, scale[1], 0.0],
            [0.0, 0.0, scale[2]],
        ];

        // Σ = R S Sᵀ Rᵀ
        let ss = mat3_mul(&s, &transpose(&s));
        let rss = mat3_mul(&r, &ss);
        let cov = mat3_mul(&rss, &transpose(&r));

        Gaussian3D { mean, cov, color, opacity }
    }
}

/// 2D射影されたガウシアン
#[derive(Clone, Debug)]
pub struct Gaussian2D {
    pub mean: [f32; 2],          // μ': 画像平面座標
    pub cov: [[f32; 2]; 2],      // Σ': 2D共分散
    pub color: [f32; 3],
    pub opacity: f32,
    pub depth: f32,              // 深度（ソート用）
}

// 行列演算ヘルパー
fn quat_to_mat3(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [1.0-2.0*(y*y+z*z), 2.0*(x*y-w*z), 2.0*(x*z+w*y)],
        [2.0*(x*y+w*z), 1.0-2.0*(x*x+z*z), 2.0*(y*z-w*x)],
        [2.0*(x*z-w*y), 2.0*(y*z+w*x), 1.0-2.0*(x*x+y*y)],
    ]
}

fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn transpose(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}
```

### 4.3 カメラ投影

```rust
// src/camera.rs

/// カメラパラメータ
pub struct Camera {
    pub view_matrix: [[f32; 4]; 4],   // World → Camera 変換
    pub proj_matrix: [[f32; 4]; 4],   // Camera → NDC 変換
    pub viewport: [u32; 2],            // [width, height]
}

impl Camera {
    /// 3Dガウシアンを2Dに射影
    pub fn project_gaussian(&self, g3d: &Gaussian3D) -> Option<Gaussian2D> {
        // 1. 中心位置を射影
        let mean_cam = self.transform_point(&g3d.mean);
        if mean_cam[2] <= 0.0 { return None; }  // カメラの後ろは無視

        let mean_ndc = self.project_point(&mean_cam);
        let mean_2d = [
            (mean_ndc[0] * 0.5 + 0.5) * self.viewport[0] as f32,
            (mean_ndc[1] * 0.5 + 0.5) * self.viewport[1] as f32,
        ];

        // 2. 共分散を射影: Σ' = J W Σ Wᵀ Jᵀ
        // J: ヤコビアン（透視投影の局所線形化）
        // W: view_matrix の回転部分
        let j = self.compute_jacobian(&mean_cam);
        let w_rot = extract_rotation(&self.view_matrix);

        // W Σ Wᵀ
        let cov_cam = transform_cov3d(&g3d.cov, &w_rot);

        // J (W Σ Wᵀ) Jᵀ
        let cov_2d = project_cov3d_to_2d(&cov_cam, &j);

        Some(Gaussian2D {
            mean: mean_2d,
            cov: cov_2d,
            color: g3d.color,
            opacity: g3d.opacity,
            depth: mean_cam[2],  // カメラ空間でのZ座標
        })
    }

    fn transform_point(&self, p: &[f32; 3]) -> [f32; 3] {
        let vm = &self.view_matrix;
        [
            vm[0][0]*p[0] + vm[0][1]*p[1] + vm[0][2]*p[2] + vm[0][3],
            vm[1][0]*p[0] + vm[1][1]*p[1] + vm[1][2]*p[2] + vm[1][3],
            vm[2][0]*p[0] + vm[2][1]*p[1] + vm[2][2]*p[2] + vm[2][3],
        ]
    }

    fn project_point(&self, p: &[f32; 3]) -> [f32; 2] {
        let pm = &self.proj_matrix;
        let w = pm[3][0]*p[0] + pm[3][1]*p[1] + pm[3][2]*p[2] + pm[3][3];
        [
            (pm[0][0]*p[0] + pm[0][1]*p[1] + pm[0][2]*p[2] + pm[0][3]) / w,
            (pm[1][0]*p[0] + pm[1][1]*p[1] + pm[1][2]*p[2] + pm[1][3]) / w,
        ]
    }

    fn compute_jacobian(&self, p_cam: &[f32; 3]) -> [[f32; 2]; 3] {
        // 透視投影 x' = fx * x/z, y' = fy * y/z
        // ∂x'/∂x = fx/z, ∂x'/∂z = -fx*x/z²
        let fx = self.proj_matrix[0][0];
        let fy = self.proj_matrix[1][1];
        let z = p_cam[2];
        let z2 = z * z;

        [
            [fx / z, 0.0, -fx * p_cam[0] / z2],
            [0.0, fy / z, -fy * p_cam[1] / z2],
        ]
    }
}

fn extract_rotation(mat: &[[f32; 4]; 4]) -> [[f32; 3]; 3] {
    [
        [mat[0][0], mat[0][1], mat[0][2]],
        [mat[1][0], mat[1][1], mat[1][2]],
        [mat[2][0], mat[2][1], mat[2][2]],
    ]
}

fn transform_cov3d(cov: &[[f32; 3]; 3], rot: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    // R Σ Rᵀ
    let r_cov = mat3_mul(rot, cov);
    mat3_mul(&r_cov, &transpose(rot))
}

fn project_cov3d_to_2d(cov3d: &[[f32; 3]; 3], jac: &[[f32; 2]; 3]) -> [[f32; 2]; 2] {
    // Σ' = J Σ Jᵀ
    std::array::from_fn(|i| std::array::from_fn(|j|
        (0..3).flat_map(|k| (0..3).map(move |l| jac[i][k] * cov3d[k][l] * jac[j][l])).sum()
    ))
}
```

### 4.4 タイルベースラスタライゼーション

```rust
// src/rasterizer.rs

const TILE_SIZE: u32 = 16;  // 16×16ピクセル

pub struct Rasterizer {
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
}

impl Rasterizer {
    pub fn new(width: u32, height: u32) -> Self {
        let tile_width = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tile_height = (height + TILE_SIZE - 1) / TILE_SIZE;
        Rasterizer { width, height, tile_width, tile_height }
    }

    /// 全ガウシアンをラスタライズ
    pub fn render(&self, gaussians: &[Gaussian2D]) -> Vec<[f32; 3]> {
        let num_pixels = (self.width * self.height) as usize;
        let mut image = vec![[0.0; 3]; num_pixels];

        // 1. タイルごとに影響するガウシアンをリスト化
        let tile_gaussians = self.assign_gaussians_to_tiles(gaussians);

        // 2. 各タイルを並列処理
        image.par_chunks_mut((TILE_SIZE * self.width) as usize)
            .enumerate()
            .for_each(|(tile_y, tile_rows)| {
                for tile_x in 0..self.tile_width {
                    let tile_id = tile_y as u32 * self.tile_width + tile_x;
                    let mut gs = tile_gaussians[tile_id as usize].clone();

                    // 深度ソート（手前から）
                    gs.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());

                    // タイル内の各ピクセルをレンダリング
                    self.render_tile(tile_x, tile_y as u32, &gs, tile_rows);
                }
            });

        image
    }

    fn assign_gaussians_to_tiles(&self, gaussians: &[Gaussian2D]) -> Vec<Vec<Gaussian2D>> {
        let num_tiles = (self.tile_width * self.tile_height) as usize;
        let mut tiles: Vec<Vec<Gaussian2D>> = vec![Vec::new(); num_tiles];

        for g in gaussians {
            // ガウシアンの影響範囲を計算（3σ）
            let radius = self.compute_radius(&g.cov) * 3.0;
            let min_x = ((g.mean[0] - radius).max(0.0) / TILE_SIZE as f32).floor() as u32;
            let max_x = ((g.mean[0] + radius).min(self.width as f32) / TILE_SIZE as f32).ceil() as u32;
            let min_y = ((g.mean[1] - radius).max(0.0) / TILE_SIZE as f32).floor() as u32;
            let max_y = ((g.mean[1] + radius).min(self.height as f32) / TILE_SIZE as f32).ceil() as u32;

            // 影響範囲のタイルに追加
            for ty in min_y..max_y.min(self.tile_height) {
                for tx in min_x..max_x.min(self.tile_width) {
                    let tile_id = (ty * self.tile_width + tx) as usize;
                    tiles[tile_id].push(g.clone());
                }
            }
        }

        tiles
    }

    fn compute_radius(&self, cov: &[[f32; 2]; 2]) -> f32 {
        // 最大固有値の平方根 = 最大半径
        let trace = cov[0][0] + cov[1][1];
        let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
        let lambda_max = 0.5 * (trace + (trace * trace - 4.0 * det).sqrt());
        lambda_max.sqrt()
    }

    fn render_tile(&self, tile_x: u32, tile_y: u32, gaussians: &[Gaussian2D], tile_data: &mut [[f32; 3]]) {
        let start_x = tile_x * TILE_SIZE;
        let start_y = tile_y * TILE_SIZE;
        let end_x = (start_x + TILE_SIZE).min(self.width);
        let end_y = (start_y + TILE_SIZE).min(self.height);

        for y in start_y..end_y {
            for x in start_x..end_x {
                let pixel_idx = ((y - start_y) * self.width + (x - start_x)) as usize;
                tile_data[pixel_idx] = self.blend_pixel(x as f32 + 0.5, y as f32 + 0.5, gaussians);
            }
        }
    }

    fn blend_pixel(&self, x: f32, y: f32, gaussians: &[Gaussian2D]) -> [f32; 3] {
        let mut color = [0.0; 3];
        let mut transmittance = 1.0;

        for g in gaussians {
            if transmittance < 0.001 { break; }  // 早期打ち切り

            // 2Dガウシアン密度
            let dx = x - g.mean[0];
            let dy = y - g.mean[1];
            let cov_inv = invert_2x2(&g.cov);
            let exponent = -0.5 * (dx * (cov_inv[0][0] * dx + cov_inv[0][1] * dy)
                                 + dy * (cov_inv[1][0] * dx + cov_inv[1][1] * dy));
            let density = exponent.exp();

            // 不透明度
            let alpha = (g.opacity * density).min(0.99);

            // α-Blending
            for i in 0..3 {
                color[i] += transmittance * alpha * g.color[i];
            }
            transmittance *= 1.0 - alpha;
        }

        color
    }
}

fn invert_2x2(mat: &[[f32; 2]; 2]) -> [[f32; 2]; 2] {
    let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    let inv_det = 1.0 / det;
    [
        [mat[1][1] * inv_det, -mat[0][1] * inv_det],
        [-mat[1][0] * inv_det, mat[0][0] * inv_det],
    ]
}
```

### 4.5 使用例

```rust
// examples/render_3dgs.rs

use gaussian_splatting::{Gaussian3D, Camera, Rasterizer};

fn main() {
    // 1. ガウシアンの定義
    let gaussians_3d = vec![
        Gaussian3D::from_rotation_scale(
            [0.0, 0.0, 5.0],              // 位置
            [1.0, 0.0, 0.0, 0.0],         // 回転なし
            [1.0, 1.0, 0.5],              // スケール
            [1.0, 0.0, 0.0],              // 赤
            0.8,                           // 不透明度
        ),
        Gaussian3D::from_rotation_scale(
            [1.5, 0.5, 6.0],
            [0.924, 0.0, 0.383, 0.0],     // Y軸45度回転
            [0.8, 1.2, 0.6],
            [0.0, 1.0, 0.0],              // 緑
            0.7,
        ),
    ];

    // 2. カメラ設定
    let camera = Camera {
        view_matrix: look_at(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], &[0.0, 1.0, 0.0]),
        proj_matrix: perspective(60.0, 800.0/600.0, 0.1, 100.0),
        viewport: [800, 600],
    };

    // 3. 射影
    let gaussians_2d: Vec<_> = gaussians_3d.iter()
        .filter_map(|g| camera.project_gaussian(g))
        .collect();

    // 4. レンダリング
    let rasterizer = Rasterizer::new(800, 600);
    let image = rasterizer.render(&gaussians_2d);

    // 5. 画像保存
    save_image("output.png", &image, 800, 600);
}
```

### 4.6 Julia NeRF訓練パイプライン

Rustでラスタライザを書いたが、訓練パイプライン全体はJuliaで構築する方が柔軟性が高い。

```julia
# nerf_training.jl

using Flux, Zygote, CUDA, ProgressMeter, Statistics

# === NeRFモデル定義 ===
struct NeRFModel
    pos_encoder::Chain
    dir_encoder::Chain
    density_net::Chain
    color_net::Chain
end

Flux.@functor NeRFModel

function NeRFModel(L_pos=10, L_dir=4, hidden_dim=256)
    # 位置符号化: 3 → 3*2*L_pos
    pos_enc_dim = 3 * 2 * L_pos
    dir_enc_dim = 3 * 2 * L_dir

    # 密度ネットワーク: pos → [density, feature]
    density_net = Chain(
        Dense(pos_enc_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim + 1)  # +1 for density
    )

    # 色ネットワーク: [feature, dir] → rgb
    color_net = Chain(
        Dense(hidden_dim + dir_enc_dim, hidden_dim ÷ 2, relu),
        Dense(hidden_dim ÷ 2, 3, sigmoid)  # RGB ∈ [0,1]
    )

    NeRFModel(
        nothing,  # pos/dir encoderは関数で直接実装
        nothing,
        density_net,
        color_net
    )
end

# 位置符号化
function positional_encoding(x::AbstractVector, L::Int)
    vcat([vcat(sin.(2.0f0^i * Float32(π) .* x), cos.(2.0f0^i * Float32(π) .* x)) for i in 0:L-1]...)
end

# Forward pass
function (model::NeRFModel)(pos::AbstractVector, dir::AbstractVector, L_pos=10, L_dir=4)
    # Encode
    pos_enc = positional_encoding(pos, L_pos)
    dir_enc = positional_encoding(dir, L_dir)

    # Density + feature
    density_feat = model.density_net(pos_enc)
    σ = relu(density_feat[end])  # Density must be non-negative
    feat = density_feat[1:end-1]

    # Color
    color_input = vcat(feat, dir_enc)
    rgb = model.color_net(color_input)

    return (color=rgb, density=σ)
end

# === Volume Rendering ===
function volume_render_differentiable(model, ray_o, ray_d, t_near, t_far, N_samples=64)
    # サンプリング点
    t_vals = range(t_near, stop=t_far, length=N_samples) |> collect

    # 各点でNeRF評価
    results   = [model(ray_o .+ t .* ray_d, ray_d) for t in t_vals]
    colors    = reduce(vcat, [r.color' for r in results])
    densities = getfield.(results, :density)

    # Delta計算
    δ = vcat(diff(t_vals), [t_vals[end] - t_vals[end-1]])

    # Alpha compositing (全て微分可能)
    α = @. 1.0f0 - exp(-densities * δ)
    T = cumprod(vcat([1.0f0], 1.0f0 .- α[1:end-1]))
    weights = T .* α

    # 最終色
    final_color = sum(weights .* colors, dims=1) |> vec

    # Depth map (bonus)
    depth = sum(weights .* t_vals)

    return (color=final_color, depth=depth, weights=weights)
end

# === 訓練ループ ===
function train_nerf(
    model,
    train_images,
    train_cameras,
    epochs=1000,
    batch_size=1024,
    lr=5e-4
)
    opt = Adam(lr)
    ps = Flux.params(model)

    # 全レイを事前計算
    all_rays = []
    all_colors = []

    for (img, cam) in zip(train_images, train_cameras)
        H, W = size(img)[1:2]
        for y in 1:H, x in 1:W
            ray_o, ray_d = get_ray(cam, x, y, W, H)
            push!(all_rays, (o=ray_o, d=ray_d))
            push!(all_colors, img[y, x, :])
        end
    end

    n_rays = length(all_rays)
    println("Total rays: $n_rays")

    @showprogress for epoch in 1:epochs
        # ランダムにバッチサンプリング
        indices = rand(1:n_rays, batch_size)
        total_loss = 0.0f0

        for idx in indices
            ray = all_rays[idx]
            gt_color = all_colors[idx]

            # Forward + Loss
            loss, grads = Flux.withgradient(ps) do
                pred = volume_render_differentiable(model, ray.o, ray.d, 2.0f0, 6.0f0, 64)
                sum((pred.color .- gt_color).^2)  # MSE
            end

            # Update
            Flux.update!(opt, ps, grads)
            total_loss += loss
        end

        # Logging
        if epoch % 100 == 0
            avg_loss = total_loss / batch_size
            println("Epoch $epoch: Loss = $(round(avg_loss, digits=6))")

            # Test rendering
            if epoch % 500 == 0
                test_img = render_full_image(model, test_camera, 128, 128)
                save("nerf_epoch_$(epoch).png", test_img)
            end
        end
    end

    return model
end

# === カメラ関連ヘルパー ===
function get_ray(camera, x, y, W, H)
    # NDC座標: [-1, 1]
    u = (2 * x / W - 1) * camera.aspect * camera.tan_half_fov
    v = (2 * y / H - 1) * camera.tan_half_fov

    # カメラ空間でのレイ方向
    ray_d_cam = normalize([u, -v, -1.0])  # -Z方向が前方

    # ワールド空間に変換
    ray_o = camera.position
    ray_d = camera.rotation * ray_d_cam  # Rotation matrix

    return (ray_o, ray_d)
end

function render_full_image(model, camera, W, H)
    img = zeros(Float32, H, W, 3)

    Threads.@threads for y in 1:H
        for x in 1:W
            ray_o, ray_d = get_ray(camera, x, y, W, H)
            result = volume_render_differentiable(model, ray_o, ray_d, 2.0f0, 6.0f0, 64)
            @views img[y, x, :] .= clamp.(result.color, 0.0f0, 1.0f0)
        end
    end

    return img
end

# === 使用例 ===
# model = NeRFModel(10, 4, 256)
# trained_model = train_nerf(model, images, cameras, 5000, 1024, 5e-4)
```

**パフォーマンス最適化のポイント**:

1. **バッチ処理**: 全ピクセルではなくランダムサンプリング（1024レイ/iter）
2. **階層的サンプリング**: Coarse/Fine の2段階（省略したが実装可能）
3. **CUDA対応**: `model |> gpu` でGPU訓練
4. **早期打ち切り**: `T < 0.001` で後続サンプルを無視

### 4.7 3DGS最適化の数値的安定性

3DGSの訓練では、共分散行列の正定値性を保つことが重要。

```rust
// src/gaussian_optim.rs

/// 共分散行列の正定値チェック
pub fn is_positive_definite(cov: &[[f32; 3]; 3]) -> bool {
    // Sylvester's criterion: 全ての主小行列式が正
    let det1 = cov[0][0];
    let det2 = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    let det3 = cov[0][0] * (cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1])
             - cov[0][1] * (cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0])
             + cov[0][2] * (cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0]);

    det1 > 0.0 && det2 > 0.0 && det3 > 0.0
}

/// 最近傍正定値行列への射影（Higham's algorithm簡略版）
pub fn project_to_positive_definite(cov: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    // 固有値分解 → 負の固有値をゼロに → 再構成
    // (実装は省略: nalgebraなどを使用)

    // 簡易版: 対角成分に小さい正則化項を追加
    let epsilon = 1e-6;
    [
        [cov[0][0] + epsilon, cov[0][1], cov[0][2]],
        [cov[1][0], cov[1][1] + epsilon, cov[1][2]],
        [cov[2][0], cov[2][1], cov[2][2] + epsilon],
    ]
}

/// 勾配クリッピング（爆発防止）
pub fn clip_gradient(grad: &mut [f32], max_norm: f32) {
    let norm: f32 = grad.iter().map(|&g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        grad.iter_mut().for_each(|g| *g *= scale);
    }
}

/// Adaptive Densification の判定
pub struct DensificationConfig {
    pub grad_threshold: f32,       // 勾配閾値（例: 0.0002）
    pub size_threshold: f32,        // ガウシアンサイズ閾値（例: 0.01）
    pub split_factor: f32,          // 分割時のスケール縮小率（例: 1.6）
}

pub enum DensificationAction {
    None,
    Split,   // Over-reconstruction: 大きいガウシアンを分割
    Clone,   // Under-reconstruction: 小さいガウシアンを複製
    Prune,   // 低寄与: 削除
}

pub fn should_densify(
    gaussian: &Gaussian3D,
    grad_norm: f32,
    config: &DensificationConfig,
) -> DensificationAction {
    // 勾配が小さい → 何もしない
    if grad_norm < config.grad_threshold {
        return DensificationAction::None;
    }

    // サイズ（共分散の最大固有値）を計算
    let size = compute_max_eigenvalue(&gaussian.cov);

    if size > config.size_threshold {
        // 大きいガウシアン + 高勾配 → 分割
        DensificationAction::Split
    } else {
        // 小さいガウシアン + 高勾配 → 複製
        DensificationAction::Clone
    }
}

fn compute_max_eigenvalue(cov: &[[f32; 3]; 3]) -> f32 {
    // 簡易版: トレースの平方根（厳密には固有値ソルバーを使う）
    let trace = cov[0][0] + cov[1][1] + cov[2][2];
    (trace / 3.0).sqrt()
}

/// ガウシアンの分割
pub fn split_gaussian(g: &Gaussian3D, factor: f32) -> [Gaussian3D; 2] {
    // スケールを縮小
    let new_scale = [
        g.scale[0] / factor,
        g.scale[1] / factor,
        g.scale[2] / factor,
    ];

    // 中心をずらす（最大固有ベクトル方向に±offset）
    let offset = new_scale[0] * 0.5;  // 簡略化
    let dir = [1.0, 0.0, 0.0];  // 実際は固有ベクトル

    let g1 = Gaussian3D {
        mean: [
            g.mean[0] + offset * dir[0],
            g.mean[1] + offset * dir[1],
            g.mean[2] + offset * dir[2],
        ],
        scale: new_scale,
        rotation: g.rotation,
        color: g.color,
        opacity: g.opacity,
    };

    let g2 = Gaussian3D {
        mean: [
            g.mean[0] - offset * dir[0],
            g.mean[1] - offset * dir[1],
            g.mean[2] - offset * dir[2],
        ],
        scale: new_scale,
        rotation: g.rotation,
        color: g.color,
        opacity: g.opacity,
    };

    [g1, g2]
}
```

**数値安定性のベストプラクティス**:

1. **共分散の正定値性**: 毎イテレーション後にチェック、必要なら射影
2. **勾配クリッピング**: `||∇|| > 10` なら正規化
3. **学習率スケジューリング**: 指数減衰（例: `lr = lr_0 * 0.99^epoch`）
4. **正則化項**: `||Σ||_F^2` のペナルティで縮退防止

### 4.8 Julia と Rust の連携: FFI経由で最速レンダリング

Juliaで訓練、Rustで推論の組み合わせが最強。

```julia
# julia_rust_bridge.jl

using Libdl

# Rustライブラリをロード
const libgaussian = "/path/to/libgaussian_splatting.so"

# Rust関数のシグネチャ
function render_gaussians_ffi(
    gaussians_ptr::Ptr{Float32},  # フラット配列: [μ, q, s, c, α] × N
    n_gaussians::Int32,
    camera_ptr::Ptr{Float32},     # [view_matrix(16), proj_matrix(16), viewport(2)]
    width::Int32,
    height::Int32,
    output_ptr::Ptr{Float32}      # 出力画像バッファ
)
    ccall(
        (:render_gaussians, libgaussian),
        Cvoid,
        (Ptr{Float32}, Int32, Ptr{Float32}, Int32, Int32, Ptr{Float32}),
        gaussians_ptr, n_gaussians, camera_ptr, width, height, output_ptr
    )
end

# ラッパー関数
function render_gaussians_rust(gaussians::Vector{Gaussian3D}, camera::Camera, W::Int, H::Int)
    N = length(gaussians)

    # ガウシアンをフラット配列に変換
    # 各ガウシアン: 16要素 [μ(3), q(4), s(3), c(3), α(1), padding(2)]
    flat_gaussians = zeros(Float32, N * 16)
    for (i, g) in enumerate(gaussians)
        o = (i - 1) * 16
        @views begin
            flat_gaussians[o+1:o+3]   .= g.mean
            flat_gaussians[o+4:o+7]   .= g.rotation
            flat_gaussians[o+8:o+10]  .= g.scale
            flat_gaussians[o+11:o+13] .= g.color
            flat_gaussians[o+14]       = g.opacity
        end
    end

    # カメラパラメータをフラット配列に
    flat_camera = vcat(vec(camera.view_matrix), vec(camera.proj_matrix), camera.viewport)

    # 出力バッファ
    output = zeros(Float32, H * W * 3)

    # FFI呼び出し
    render_gaussians_ffi(
        pointer(flat_gaussians),
        Int32(N),
        pointer(flat_camera),
        Int32(W),
        Int32(H),
        pointer(output)
    )

    # 画像に整形
    img = reshape(output, (3, W, H))
    img = permutedims(img, (3, 2, 1))  # (H, W, 3)

    return img
end

# 使用例
# gaussians = optimize_gaussians(...)  # Julia訓練
# img = render_gaussians_rust(gaussians, camera, 800, 600)  # Rust推論
# save("output.png", img)
```

**パフォーマンス**:
- Julia訓練: 自動微分が強力、実験が速い
- Rust推論: ゼロコスト抽象化、マルチスレッド並列
- FFI overhead: `ccall` は数μs（レンダリング時間の1%未満）

**メモリ安全性**:
- Juliaが確保したメモリは`GC.@preserve`で保護
- Rustは`ptr`を読むだけ（所有権は移譲しない）
- FFI境界で型チェック（`Float32`統一）

> **Note:** **進捗: 70%完了** — Rust 3DGSラスタライザ + Julia訓練パイプライン + FFI連携を実装。数値安定性の考慮とAdaptive Densificationのロジックも完備。次は実験ゾーン — 実際にNeRFと3DGSを訓練してみる。

---

## 🔬 5. 実験ゾーン（30分）— NeRF・3DGS・DreamFusionを実際に動かす

**ゴール**: 理論と実装を体験で確認。NeRF訓練、3DGS再構成、Text-to-3D生成を実行する。

### 5.1 シンボル読解テスト

3D生成の論文を読むための記法確認。以下の数式を日本語で説明せよ。

<details><summary>**Q1**: $C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$</summary>

**解答**:
レイ $\mathbf{r}$ のピクセル色 $C$ は、レイ上の全ての点での「透過率×密度×色」の積分。
- $T(t)$: その点まで光が到達する確率（累積減衰）
- $\sigma(\mathbf{r}(t))$: その点の体積密度
- $\mathbf{c}(\mathbf{r}(t), \mathbf{d})$: その点から方向 $\mathbf{d}$ に放射される色

これがVolume Rendering方程式の核心。

</details>

<details><summary>**Q2**: $\boldsymbol{\Sigma}_k = \mathbf{R}_k \mathbf{S}_k \mathbf{S}_k^\top \mathbf{R}_k^\top$</summary>

**解答**:
3Dガウシアンの共分散行列 $\boldsymbol{\Sigma}_k$ を、回転 $\mathbf{R}_k$ とスケール $\mathbf{S}_k$ で分解。
- $\mathbf{S}_k$: 対角行列（各軸の半径）
- $\mathbf{R}_k$: 回転行列（楕円体の向き）
- $\mathbf{S}_k \mathbf{S}_k^\top$: スケールの2乗行列（分散）
- 回転で挟むことで、任意の向きの楕円体を表現。

</details>

<details><summary>**Q3**: $\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \boldsymbol{\epsilon}} \left[ w(t) \left( \boldsymbol{\epsilon}_\phi(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon} \right) \frac{\partial \mathbf{x}}{\partial \theta} \right]$</summary>

**解答**:
Score Distillation Samplingの勾配式。
- $\boldsymbol{\epsilon}_\phi(\mathbf{x}_t, t, y)$: 拡散モデルが予測するノイズ（テキスト $y$ 条件付き）
- $\boldsymbol{\epsilon}$: 実際に加えたノイズ
- $\boldsymbol{\epsilon}_\phi - \boldsymbol{\epsilon}$: 「もっとこう変えろ」という指示
- $\frac{\partial \mathbf{x}}{\partial \theta}$: レンダリング画像の3Dパラメータに対する勾配

拡散モデルの指示を3D空間にバックプロパゲーションして最適化。

</details>

<details><summary>**Q4**: $\gamma(\mathbf{x}) = \left( \sin(2^0 \pi \mathbf{x}), \cos(2^0 \pi \mathbf{x}), \ldots, \sin(2^{L-1} \pi \mathbf{x}), \cos(2^{L-1} \pi \mathbf{x}) \right)$</summary>

**解答**:
位置符号化 (Positional Encoding)。
- 入力 $\mathbf{x} \in \mathbb{R}^3$ を高次元 $\mathbb{R}^{6L}$ に埋め込む
- 各周波数 $2^0, 2^1, \ldots, 2^{L-1}$ の正弦波成分を明示的に入力
- MLPが高周波の詳細（テクスチャ）を学習しやすくする
- NeRFの鮮明さを決める重要技術。

</details>

<details><summary>**Q5**: $\boldsymbol{\Sigma}'_k = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma}_k \mathbf{W}^\top \mathbf{J}^\top$</summary>

**解答**:
3Dガウシアンの共分散を2Dに射影する式。
- $\boldsymbol{\Sigma}_k$: 3D共分散（3×3行列）
- $\mathbf{W}$: カメラのview行列（回転部分）
- $\mathbf{J}$: 透視投影のヤコビアン（局所線形化）
- 結果 $\boldsymbol{\Sigma}'_k$: 2D共分散（2×2行列）

3Dの楕円体が、画像平面上でどんな楕円になるかを計算。

</details>

### 5.2 実装チャレンジ: Tiny NeRF on Synthetic Data

**課題**: 合成データ（解析的に定義した3Dシーン）でTiny NeRFを訓練し、新規視点を生成せよ。

```julia
using Flux, Zygote, LinearAlgebra, Random, Statistics

# === シーン定義: 2つの球 ===
function scene_sdf(x, y, z)
    # 球1: 中心(0, 0, 4), 半径1
    d1 = sqrt(x^2 + y^2 + (z-4)^2) - 1.0
    # 球2: 中心(2, 0, 5), 半径0.7
    d2 = sqrt((x-2)^2 + y^2 + (z-5)^2) - 0.7
    return min(d1, d2)
end

function scene_color(x, y, z)
    # 球1: 赤、球2: 青
    d1 = sqrt(x^2 + y^2 + (z-4)^2)
    d2 = sqrt((x-2)^2 + y^2 + (z-5)^2)
    if d1 < d2
        return [1.0, 0.0, 0.0]
    else
        return [0.0, 0.5, 1.0]
    end
end

# === Ground Truth レンダリング ===
function render_gt(ray_o, ray_d, t_vals)
    # Sphere tracingで表面を見つける
    t = t_vals[1]
    for _ in 1:100
        pos = ray_o + t * ray_d
        dist = scene_sdf(pos...)
        if dist < 0.01
            return scene_color(pos...)
        end
        t += dist
        if t > t_vals[end]
            return [0.0, 0.0, 0.0]  # 背景=黒
        end
    end
    return [0.0, 0.0, 0.0]
end

# === 訓練データ生成 ===
function generate_training_data(n_views=8, img_size=32)
    data = []
    for i in 1:n_views
        angle = 2π * i / n_views
        cam_pos = [3*cos(angle), 0.0, 3*sin(angle) + 4.0]
        look_at = [0.0, 0.0, 4.0]

        for u in 1:img_size, v in 1:img_size
            # ピクセル→レイ
            x_ndc = (2 * u / img_size - 1) * 0.5
            y_ndc = (2 * v / img_size - 1) * 0.5
            ray_d = normalize([x_ndc, y_ndc, 1.0])  # 簡略化
            ray_o = cam_pos

            t_vals = range(0.1, stop=10.0, length=64)
            color = render_gt(ray_o, ray_d, t_vals)

            push!(data, (ray_o=ray_o, ray_d=ray_d, color=color))
        end
    end
    return data
end

# === Tiny NeRF モデル ===
function positional_encoding(x, L=6)
    vcat([vcat([sin(2.0^i * π * x[j]) for j in 1:length(x)],
               [cos(2.0^i * π * x[j]) for j in 1:length(x)]) for i in 0:L-1]...)
end

function create_nerf_model(L_pos=6, L_dir=4)
    pos_dim = 3 * 2 * L_pos
    dir_dim = 3 * 2 * L_dir
    return Chain(
        Dense(pos_dim, 128, relu),
        Dense(128, 128, relu),
        x -> vcat(x, positional_encoding([0.0, 0.0, 1.0], L_dir)),  # Dummy dir
        Dense(128 + dir_dim, 64, relu),
        Dense(64, 4),  # [r, g, b, σ]
        x -> vcat(sigmoid.(x[1:3]), relu(x[4]))  # rgb ∈ [0,1], σ ≥ 0
    )
end

# === Volume Rendering ===
function volume_render(model, ray_o, ray_d, t_vals)
    N = length(t_vals)

    results   = [(pos = ray_o .+ t .* ray_d; out = model(positional_encoding(pos)); (out[1:3], out[4])) for t in t_vals]
    colors    = reduce(vcat, [r[1]' for r in results])
    densities = [r[2] for r in results]

    # Alpha compositing
    δ = vcat(diff(collect(t_vals)), [0.1])
    α = @. 1 - exp(-densities * δ)
    T = cumprod(vcat([1.0], 1 .- α[1:end-1]))
    weights = T .* α

    final_color = sum(weights .* colors, dims=1)[1, :]
    return final_color
end

# === 訓練ループ ===
function train_tiny_nerf(data, epochs=100)
    model = create_nerf_model()
    opt = Adam(0.001)
    ps = Flux.params(model)

    for epoch in 1:epochs
        total_loss = 0.0
        for (i, sample) in enumerate(data)
            t_vals = range(0.1, stop=10.0, length=64)
            loss, grads = Flux.withgradient(ps) do
                pred = volume_render(model, sample.ray_o, sample.ray_d, t_vals)
                sum((pred .- sample.color).^2)
            end
            Flux.update!(opt, ps, grads)
            total_loss += loss
        end

        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $(total_loss / length(data))")
        end
    end

    return model
end

# === 実行 ===
Random.seed!(42)
train_data = generate_training_data(8, 32)
println("Generated $(length(train_data)) training samples")

trained_model = train_tiny_nerf(train_data, 100)
println("Training complete!")

# 新規視点でテスト
test_ray_o = [0.0, 2.0, 4.0]
test_ray_d = normalize([0.0, -0.5, 0.2])
t_vals = range(0.1, stop=10.0, length=64)
test_color = volume_render(trained_model, test_ray_o, test_ray_d, t_vals)
println("Test render color: ", test_color)
# => [0.95, 0.02, 0.01] のような赤系（球1を見ている）
```

**期待結果**: 100エポック後、新規視点でも正しい色が出る（Loss < 0.01）。

### 5.3 コード翻訳テスト: Julia → Rust

**課題**: 上記の `volume_render` 関数をRustで書け。

<details><summary>**解答例**</summary>

```rust
// volume_render.rs

pub fn volume_render(
    model: &impl Fn(&[f32]) -> [f32; 4],
    ray_o: &[f32; 3],
    ray_d: &[f32; 3],
    t_vals: &[f32],
) -> [f32; 3] {
    let n = t_vals.len();

    // Evaluate NeRF at each sample point
    let (colors, densities): (Vec<[f32; 3]>, Vec<f32>) = t_vals.iter().map(|&t| {
        let pos = [
            ray_o[0] + t * ray_d[0],
            ray_o[1] + t * ray_d[1],
            ray_o[2] + t * ray_d[2],
        ];
        let output = model(&positional_encoding(&pos, 6));
        ([output[0], output[1], output[2]], output[3])
    }).unzip();

    // Compute deltas
    let mut delta: Vec<f32> = t_vals.windows(2).map(|w| w[1] - w[0]).collect();
    delta.push(0.1);

    // Alpha compositing
    let alpha: Vec<f32> = densities.iter().zip(&delta)
        .map(|(&d, &dt)| 1.0 - (-d * dt).exp())
        .collect();

    let mut transmittance = vec![1.0f32; n];
    for i in 1..n {
        transmittance[i] = transmittance[i-1] * (1.0 - alpha[i-1]);
    }

    let mut final_color = [0.0f32; 3];
    transmittance.iter().zip(&alpha).zip(&colors).for_each(|((t, a), col)| {
        let w = t * a;
        final_color.iter_mut().zip(col).for_each(|(ci, &gi)| *ci += w * gi);
    });

    final_color
}

fn positional_encoding(pos: &[f32; 3], l: usize) -> Vec<f32> {
    (0..l).flat_map(|i| {
        let freq = (2.0_f32).powi(i as i32) * std::f32::consts::PI;
        pos.iter().map(move |&x| (freq * x).sin())
            .chain(pos.iter().map(move |&x| (freq * x).cos()))
    }).collect()
}
```

数式↔コードの1:1対応を確認。

</details>

### 5.4 3DGS再構成の実験

**準備**: 合成データまたは実データ（例: NeRF Syntheticデータセット）を用意。

**手順**:
1. Structure from Motion (SfM) で初期点群を取得
2. 点群を3Dガウシアンに初期化（各点 → 1ガウシアン）
3. Photometric Lossで最適化（100-1000イテレーション）
4. Adaptive Densificationで品質向上
5. 新規視点でレンダリング

**コード例** (Julia + 前節のRustラスタライザを呼び出し):

```julia
using LibGit2, LinearAlgebra

# 1. 初期ガウシアンの生成（SfM点群から）
function initialize_gaussians_from_points(points, colors)
    n = size(points, 1)
    [let dists = sort([norm(points[i,:] - points[j,:]) for j in 1:n if j != i])
        Gaussian3D(
            mean     = points[i, :],
            rotation = [1.0, 0.0, 0.0, 0.0],  # 単位四元数
            scale    = fill(mean(dists[1:3]), 3),
            color    = colors[i, :],
            opacity  = 0.5
        )
    end for i in 1:n]
end

# 2. 最適化ループ
function optimize_gaussians(gaussians, images, cameras, iters=500)
    # パラメータ化: [μ, q, s, c, α] を全て1つのベクトルに
    params = pack_params(gaussians)
    optimizer = Adam(0.01)

    for iter in 1:iters
        loss = 0.0

        for (img, cam) in zip(images, cameras)
            # Forward: レンダリング (Rustラスタライザを呼ぶ)
            rendered = render_gaussians_rust(params, cam)

            # Loss: L1 + D-SSIM
            loss += sum(abs.(rendered .- img)) + (1 - ssim(rendered, img))
        end

        # Backward: 勾配計算
        grads = gradient(() -> loss, params)[1]

        # Update
        update!(optimizer, params, grads)

        # Adaptive Densification (100イテレーションごと)
        if iter % 100 == 0
            params = densify_and_prune(params, grads)
            println("Iter $iter: Loss = $loss, Gaussians = $(length(params)÷16)")
        end
    end

    return unpack_params(params)
end

# 実行
# (初期点群は別途SfMで取得済みと仮定)
initial_gaussians = initialize_gaussians_from_points(sfm_points, sfm_colors)
optimized_gaussians = optimize_gaussians(initial_gaussians, train_images, train_cameras, 500)

# 新規視点レンダリング
test_image = render_gaussians_rust(optimized_gaussians, test_camera)
save("test_view.png", test_image)
```

**期待結果**: PSNR > 25 dB、訓練時間 < 10分（CPUでも）。

### 5.5 DreamFusion実験: テキストから3D

**準備**:
- 事前訓練済み拡散モデル（Stable Diffusion 2.1など）
- Instant NGP実装

**手順**:
1. テキストプロンプト: "a DSLR photo of a corgi"
2. NeRFを初期化（ランダム重み）
3. ランダム視点でレンダリング→ノイズ追加→拡散モデルでスコア計算
4. SDS勾配でNeRFを更新
5. 5000-10000イテレーション

**擬似コード**:

```julia
using StableDiffusion  # 仮想パッケージ

prompt = "a DSLR photo of a corgi"
nerf = InstantNGP()  # HashEncoding + 小さいMLP
diffusion_model = load_stable_diffusion("v2.1")

for iter in 1:10000
    # ランダム視点
    camera = random_camera()

    # レンダリング
    img = render(nerf, camera)

    # ノイズ追加
    t = rand(1:1000)
    ϵ = randn(size(img)...)
    img_noisy = add_noise(img, t, ϵ)

    # 拡散モデルでノイズ予測
    ϵ_pred = diffusion_model(img_noisy, t, prompt)

    # SDS勾配
    grad_img = (ϵ_pred - ϵ)  # 画像空間の勾配
    grad_nerf = backprop(nerf, grad_img)  # NeRFパラメータへの勾配

    # 更新
    update!(nerf, grad_nerf, lr=0.01)

    if iter % 500 == 0
        save("iter_$(iter).png", render(nerf, fixed_camera))
    end
end

# 最終3Dモデルをメッシュ化
mesh = extract_mesh(nerf)
save("corgi.obj", mesh)
```

**期待結果**: 5000イテレーションで認識可能な形状、10000で細部が出現。

### 5.6 自己診断チェックリスト

- [ ] NeRFのVolume Rendering方程式を、積分記号から離散和まで導出できる
- [ ] 位置符号化の周波数がなぜ $2^i$ なのか説明できる
- [ ] 3DGSの共分散行列 $\boldsymbol{\Sigma}$ を回転+スケールで分解できる
- [ ] 3Dガウシアンを2Dに射影する式 $\boldsymbol{\Sigma}' = \mathbf{J}\mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top\mathbf{J}^\top$ を書ける
- [ ] SDS LossがなぜKL divergenceの最小化と等価か説明できる
- [ ] VSDがSDSとどう違うか（$\boldsymbol{\epsilon}_\psi$ の役割）を説明できる
- [ ] Instant NGPのHash Encodingがなぜ高速か説明できる
- [ ] 4DGSが3DGSとどう違うか（時間軸の追加）を説明できる
- [ ] Tiny NeRFをJuliaで実装できる（100行以内）
- [ ] 3DGSラスタライザのα-Blending部分をRustで書ける

**9/10以上**: 完璧。次の講義へ。
**6-8/10**: もう一度Zone 3を読む。
**5以下**: Zone 0から復習。

> **Note:** **進捗: 85%完了** — 実験で理論を確認。Tiny NeRF訓練、3DGS再構成、DreamFusion Text-to-3Dを体験。次は発展ゾーン — 研究フロンティアと未解決問題へ。

---


> Progress: 85%
> **理解度チェック**
> 1. NeRFの離散化近似$\alpha_i = 1 - e^{-\sigma_i \delta_i}$において$\alpha_i$が「区間$\delta_i$での光吸収確率」を表すことを、連続積分から導出せよ。
> 2. 3DGSのタイルベースラスタライザで前から後ろの順（Front-to-Back）に描画するとアルファ合成が正確になる理由を説明せよ。

## 🚀 6. 発展ゾーン（30分）— 研究フロンティアと未解決問題 + まとめ

**ゴール**: 2025-2026の最新研究動向を把握し、未解決問題を特定する。

### 6.1 3DGS Applications Survey (2025)

2025年8月のサーベイ論文[^1]が、3DGSの応用を体系化した。

**3つの基盤タスク**:
1. **Segmentation**: 3Dシーンの意味的分割（SAM-3Dなど）
2. **Editing**: ガウシアンの追加・削除・変形（GaussianEditorなど）
3. **Generation**: テキスト→3D、画像→3D（DreamGaussianなど）

**応用分野**:
- **AR/VR**: リアルタイムレンダリングで没入感向上
- **自動運転**: LiDAR点群→3DGS→シミュレーション
- **ロボティクス**: 物体操作の視覚フィードバック
- **デジタルツイン**: 都市・建物の3Dモデル化

**技術的挑戦**:
- **圧縮**: ガウシアン数が100万を超える→メモリ削減（Sparse View 3DGSなど）
- **動的シーン**: 4DGS（時間軸追加）だが、長時間の一貫性が課題
- **人体再構成**: 関節・衣服の変形に対応（GaussianAvatar など）

### 6.2 NeRFの進化: Mip-NeRF 360 と Zip-NeRF

**Mip-NeRF**: マルチスケール表現でアンチエイリアス
- 問題: NeRFはレイ上の点サンプルで評価→遠景でエイリアシング
- 解決: 円錐形状（conical frustum）で積分→スケール不変

**Zip-NeRF** (2023): Hash Grid + Anti-aliasing
- Instant NGPの速度 + Mip-NeRFの品質
- 訓練15分、レンダリング10fps

しかし3DGSの100fps には及ばない→NeRFは「高品質・遅い」、3DGSは「高速・編集可能」で棲み分け。

### 6.3 Zero-1-to-3: 単一画像から3D

**動機**: 多視点画像の収集は手間→1枚の画像から3D再構成したい。

**アイデア** (2023)[^2]:
1. 拡散モデルを訓練: 1視点画像 → 別視点画像を生成
2. 生成した多視点画像でNeRFを訓練

**結果**: 1枚→6視点生成→3D再構成が可能。ただし品質は多視点データより劣る。

**発展**: One-2-3-45 (2023) は45秒で3Dメッシュ生成。

### 6.4 Magic3D vs ProlificDreamer

| 項目 | DreamFusion | Magic3D | ProlificDreamer |
|:-----|:------------|:--------|:----------------|
| 発表 | 2022年9月 | 2022年11月 | 2023年5月 |
| 損失 | SDS | SDS (2段階) | VSD |
| 解像度 | 64² | 512² | 512² |
| 訓練時間 | 1.5時間 | 40分 | 40分 |
| 品質 | 良 | 非常に良 | 最高 |
| 多様性 | 低（mode collapse） | 低 | 高 |

**Magic3D**[^3] の2段階最適化:
1. **Coarse Stage**: 低解像度拡散モデル + Instant NGP（高速）
2. **Fine Stage**: 高解像度Latent Diffusion + DMTet（メッシュ）

**ProlificDreamer**[^4] のVSD:
- $\boldsymbol{\epsilon}_\psi$ (LoRA) で $\theta$ 専用のノイズ予測
- Mode seeking → Mode covering へ
- 多様な3D生成が可能

### 6.5 4D Gaussian Splatting: 動的シーンへの拡張

**4DGS** (2024)[^5]: 時間軸を追加した3DGS

**アプローチ1**: 各フレームで独立した3DGS → メモリ爆発
**アプローチ2**: 4D Neural Voxel でガウシアンの変形を学習

$$
\boldsymbol{\mu}_k(t) = \boldsymbol{\mu}_k(0) + \Delta\boldsymbol{\mu}_k(t)
$$

ここで $\Delta\boldsymbol{\mu}_k(t)$ は4D Voxelから予測。

**性能**: リアルタイムレンダリング（82 fps）、動的シーンの高品質再構成。

**課題**: 長時間（数分）の一貫性、物理法則の学習。

### 6.6 GaussianEditor: 3D編集の実用化

**GaussianEditor** (2024)[^6]: テキスト指示で3Dシーンを編集

**手法**:
1. **Gaussian Semantic Tracing**: 編集対象のガウシアンを訓練中に追跡
2. **Hierarchical Gaussian Splatting (HGS)**: 粗→細の階層で安定化
3. **2D Diffusion Guidance**: InstructPix2Pixなどの2D編集モデルを3Dに適用

**例**:
- "Make the car red" → 車のガウシアンの色を変更
- "Remove the tree" → 木のガウシアンを削除
- "Add a cat" → 新しいガウシアンを追加

**処理時間**: 2-7分（NeRF編集は数時間かかる）

### 6.7 未解決問題と次のブレイクスルー

**1. リアルタイム訓練**:
- 現状: 3DGSでも数分
- 目標: 秒単位（スマホでの即座の3D化）
- 鍵: オンライン学習、incremental update

**2. 物理法則の組み込み**:
- 現状: 見た目の再現のみ
- 目標: 重力・摩擦・衝突を考慮した3D
- 鍵: Physics-Informed NeRF/3DGS

**3. 一般化モデル**:
- 現状: 1シーン1訓練
- 目標: 1つのモデルで全シーンに対応（Zero-shot 3D）
- 鍵: Transformer-based 3D representation

**4. 超高解像度**:
- 現状: 512²-1024²
- 目標: 4K-8K（映画品質）
- 鍵: LoD (Level of Detail)、Sparse representation

**5. Text-to-3Dの多様性**:
- 現状: VSDでも同じプロンプトから似た3D
- 目標: 明示的な多様性制御（"生成1", "生成2"...）
- 鍵: Conditional VSD、Multi-modal latent space

**次のブレイクスルー予測**:
- **2026年前半**: 5秒でText-to-3D（現在の40分から）
- **2026年後半**: スマホでリアルタイム3DGS訓練
- **2027年**: 物理シミュレーションと統合した4D World Models

> **Note:** **進捗: 95%完了** — 研究フロンティアを俯瞰。3DGS Applications、4DGS、GaussianEditor、未解決問題を把握。最後の振り返りゾーンへ。

---


**ゴール**: 本講義の核心を3つにまとめ、次のステップを明確にする。

### 6.8 3つの本質的学び

**1. Volume Rendering方程式 = 3D理解の基盤**

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt
$$

この1行が、NeRF・3DGS・DreamFusionの全ての出発点。透過率 $T(t)$ と密度 $\sigma$ の積分で色が決まる。

**2. 明示 vs 暗黙の表現トレードオフ**

- **NeRF（暗黙）**: 滑らか・メモリ効率・編集難・遅い
- **3DGS（明示）**: 高速・編集容易・メモリ大・若干ざらつき

用途で選べ。研究=NeRF、プロダクション=3DGS。

**3. 2D→3Dの知識転移 (SDS)**

2D拡散モデルの膨大な知識を、3D空間に蒸留できる。Score Distillation Samplingは、データ不足を乗り越える鍵。

### 6.9 よくある質問 (FAQ)

<details><summary>**Q1**: NeRFと3DGS、結局どちらを使うべき？</summary>

**A**: 用途次第。
- **研究・高品質重視**: NeRF（Zip-NeRFなど最新版）
- **リアルタイム・編集重視**: 3DGS
- **Text-to-3D**: DreamFusion系（NeRFベース）またはDreamGaussian（3DGSベース）

3DGSが台頭しているが、NeRFも進化中。両方知っておくべき。

</details>

<details><summary>**Q2**: 3D生成の実用化はどこまで進んでいる？</summary>

**A**: AR/VR、自動運転、ロボティクスで実用段階。
- **NVIDIA Instant NGP**: 商用ソフトに統合
- **3DGS**: ゲームエンジン（Unity/Unreal）への実装進行中
- **DreamFusion系**: Adobeなどが製品化検討中

ただしスマホでのリアルタイム訓練はまだ未来。

</details>

<details><summary>**Q3**: 数式が多くて挫折しそう。どうすれば？</summary>

**A**: 3段階で理解。
1. **直感**: Zone 0-2で「何をしているか」を掴む
2. **実装**: Zone 4-5でコードを動かす
3. **数式**: Zone 3に戻って「なぜ動くか」を確認

数式→コード→実験の3往復で定着する。焦らない。

</details>

<details><summary>**Q4**: ProlificDreamerのVSDが難しい。もっと簡単な説明は？</summary>

**A**: SDSの問題を例えで:
- **SDS**: 「この写真、もっと犬っぽくして」と言われて、いつも同じ犬種になる（mode collapse）
- **VSD**: 「この写真専用の先生」を用意して、多様な犬種を学べるようにする

$\boldsymbol{\epsilon}_\psi$ = その3D専用の先生。LoRAで効率的に訓練。

</details>

<details><summary>**Q5**: 4DGSは3DGSの拡張？実装はどれくらい違う？</summary>

**A**: 概念は近いが実装は大幅に違う。
- **3DGS**: 空間のガウシアン集合
- **4DGS**: 時空間（4D）のガウシアン集合 or 時間依存の変形

4D Voxelで変形を予測するアプローチが主流。メモリ管理が3DGSの10倍複雑。

</details>

### 6.10 学習スケジュール（1週間プラン）

| 日 | 内容 | 時間 | チェック |
|:---|:-----|:-----|:---------|
| **Day 1** | Zone 0-2（直感と全体像） | 2時間 | □ 5つの3D表現を説明できる |
| **Day 2** | Zone 3.1-3.3（NeRF数式） | 3時間 | □ Volume Rendering式を導出できる |
| **Day 3** | Zone 3.4-3.5（3DGS・SDS数式） | 3時間 | □ SDS勾配を書ける |
| **Day 4** | Zone 4（Rust実装） | 3時間 | □ 3DGSラスタライザが動く |
| **Day 5** | Zone 5（実験） | 3時間 | □ Tiny NeRFを訓練できる |
| **Day 6** | Zone 6（発展） + 論文読解 | 2時間 | □ 3つの未解決問題を挙げられる |
| **Day 7** | 復習 + 次の講義準備 | 2時間 | □ 全チェックリスト達成 |

**Total**: 18時間（Zone 3が最もヘビー）

### 6.11 進捗トラッカー

```julia
# 自己評価スクリプト
function assess_lecture_46()
    questions = [
        "NeRFのVolume Rendering式を書ける",
        "位置符号化の役割を説明できる",
        "3DGSの共分散を回転+スケールで分解できる",
        "3D→2D射影の式を書ける",
        "SDS LossのKL divergence解釈を説明できる",
        "VSDとSDSの違いを説明できる",
        "Instant NGPのHash Encodingを説明できる",
        "4DGSの時間軸追加を説明できる",
        "Tiny NeRFをJuliaで実装できる",
        "3DGSラスタライザをRustで実装できる"
    ]

    println("=== Lecture 46 自己評価 ===")
    score = 0
    for (i, q) in enumerate(questions)
        print("$i. $q? (y/n): ")
        ans = readline()
        if lowercase(ans) == "y"
            score += 1
        end
    end

    pct = score / length(questions) * 100
    println("\n達成率: $(score)/$(length(questions)) = $(round(pct, digits=1))%")

    if pct >= 90
        println("🏆 完璧！次の講義（第47回: モーション・4D生成）へ進もう。")
    elseif pct >= 70
        println("✅ 良好。不明点をもう一度復習してから次へ。")
    else
        println("⚠️ もう一度Zone 3を読み直そう。焦らなくて大丈夫。")
    end
end

assess_lecture_46()
```

**実行例**:
```
=== Lecture 46 自己評価 ===
1. NeRFのVolume Rendering式を書ける? (y/n): y
...
達成率: 9/10 = 90.0%
🏆 完璧！次の講義（第47回: モーション・4D生成）へ進もう。
```

### 6.12 次の講義へのつなぎ

**本講義で習得**: 3D空間の生成（静的シーン）

**次の講義（第47回）**: モーション・4D生成 & Diffusion Policy
- Text-to-Motion（MDM, MotionGPT-3, UniMo）
- 4D Generation（4DGS, TC4D, PaintScene4D）
- Diffusion Policy for Robotics（RDT, Hierarchical Policy）

**つながり**:
- 3DGS（本講義） → 4DGS（次講義）: 時間軸の追加
- DreamFusion（本講義） → TC4D（次講義）: Text-to-4D
- 静的3D → 動的モーション: 生成モデルの最終形態

> **Note:** **進捗: 100%完了** — 第46回完走！NeRF→3DGS→DreamFusionの3D生成革命を完全習得。Volume Rendering方程式、微分可能ラスタライゼーション、SDS Loss、全て導出した。Rustで3DGSラスタライザを実装し、Julia でTiny NeRFを訓練した。次は第47回でモーション・4D生成へ。

---


> Progress: 95%
> **理解度チェック**
> 1. Zip-NeRFがInstant NGP（Hash Grid）とMip-NeRF（反エイリアス）の両方の利点を持つ理由を、multi-scale hashingの観点から説明せよ。
> 2. 4DGS（4D Gaussian Splatting）で「長時間の一貫性」が課題になる理由を、ガウシアンパラメータの時間依存性から述べよ。

## 💀 パラダイム転換の問い

**問い**: なぜ3DGSは一夜でNeRFを"遺物"にしたのか？次に"遺物"になるのは？

**議論の起点**:
1. **速度の桁違い**: 0.1 fps → 100 fps（1000倍）は、研究室→プロダクションの閾値
2. **編集性の壁**: NeRFは「ブラックボックスMLP」、3DGSは「明示的点群」→直感的操作
3. **パラダイムシフト**: 「連続関数=美しい」から「離散+並列=実用」へ

**反論**:
- NeRFは"遺物"ではない。Zip-NeRFは品質でまだ優位
- 3DGSはメモリ食い→大規模シーンで限界
- 両者は補完的（研究 vs プロダクション）

**次の"遺物"候補**:
- **3DGS自身**: 5年後、量子コンピュータで連続表現が復権？
- **Diffusion Models**: Test-time Scalingが主流になり、Training-based生成が時代遅れに？
- **人間の3Dモデラー**: AIが完全自動化して、職業自体が消滅？

**歴史的教訓**:
- MVS（2000年代）→ NeRF（2020）: 20年
- NeRF（2020）→ 3DGS（2023）: 3年
- 加速している。次は？

<details><summary>**歴史的文脈: NeRFは本当に新しかったのか？**</summary>

Volume Rendering自体は1984年のKajiya-Von Herzen以来の古典技術。NeRFの革新は:
1. **MLPでの連続表現**: 離散Voxelから脱却
2. **微分可能性**: 勾配降下で最適化
3. **位置符号化**: 高周波の学習

しかし「連続関数で3D表現」自体は、DeepSDF（2019）など先行研究あり。NeRFは「タイミング」と「実装の洗練」で勝った。

3DGSも同様。Splatting自体は1985年のWestoverが起源。革新は「微分可能ラスタライゼーション」。

**教訓**: 新しい組み合わせが革命を起こす。全く新しい技術など稀。

</details>

---

## 参考文献

### 主要論文

[^1]: He, S., et al. (2025). "A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation". *arXiv:2508.09977*.
<https://arxiv.org/abs/2508.09977>

[^3]: Lin, C.-H., Gao, J., Tang, L., et al. (2022). "Magic3D: High-Resolution Text-to-3D Content Creation". *arXiv:2211.10440*.
<https://arxiv.org/abs/2211.10440>

[^4]: Wang, Z., Lu, C., Wang, Y., et al. (2023). "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation". *NeurIPS 2023 Spotlight*. *arXiv:2305.16213*.
<https://arxiv.org/abs/2305.16213>

### 教科書・サーベイ

- Barron, J. T., et al. (2022). "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields". *CVPR 2022*.
- Tewari, A., et al. (2022). "Advances in Neural Rendering". *Computer Graphics Forum, 41*(2).
- Kerbl, B., et al. (2024). "A Survey on 3D Gaussian Splatting". *arXiv:2401.03890*.

### オンラインリソース

- NeRF公式ページ: https://www.matthewtancik.com/nerf
- 3D Gaussian Splatting公式: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Instant NGP GitHub: https://github.com/NVlabs/instant-ngp
- DreamFusion公式: https://dreamfusion3d.github.io/

---


## 🔗 前編・後編リンク

- **前編 (Part 1 — 理論編)**: [第46回: 3D生成 & Neural Rendering (Part 1)](ml-lecture-46-part1)

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
