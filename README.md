# ImageToImage

3D医用画像向けのImage-to-image translation学習コードです。CT/MRの独自raw/hdr形式を読み込み、source volumeからtarget volumeへ変換するモデルを学習します。

学習アルゴリズムは **Image-to-Image Rectified Flow Reformulation (I2I-RFR)** です。

https://arxiv.org/abs/2603.20186

## 概要

このリポジトリは、もともとの3Dセグメンテーション用データパイプラインをできるだけ維持したまま、ペア画像変換の学習に変更したものです。

- source/targetの同名ペア画像を読み込む
- crop、spacing正規化、affine augmentationなど既存の3D前処理を利用する
- UNet backboneで3D volumeを変換する
- I2I-RFRにより、targetにノイズを混ぜた状態からtarget画像を復元する
- 推論時は少数ステップのEuler更新で出力を生成する

## 単一ディレクトリでdeblurを学習する

`training_mode: self_supervised_deblur` を指定すると、`source_data_dir` の画像だけで学習できます。各画像をclean targetとして再利用し、入力側にだけ設定した劣化を合成して、劣化画像から元画像へ戻すdeblurを学習します。`target_data_dir`はこのモードでは使いません。

```bash
python main.py --overrides \
  training_mode=self_supervised_deblur \
  source_data_dir=datasets_images \
  exp_dir=results/self_deblur
```

従来のGaussian blurを使う場合は `degradation_type: gaussian` を指定します。
blurの強さはvoxel単位のsigmaで設定し、学習時は画像ごとに範囲内から
ランダムに選びます。

```yaml
self_supervised_deblur:
  degradation_type: gaussian
  sigma_range: [0.5, 2.0]
  validation_sigma: 1.25
```

### 心臓CTのmotion blur近似

`degradation_type: cardiac_motion` では、clean画像に対して平行移動、回転、
収縮・拡張を加えた複数の心拍位相を作り、その平均をsourceにします。
投影データを使わない画像空間での近似ですが、一様なGaussian blurよりも
二重輪郭や局所的な心拍ブレを再現できます。

```bash
python main.py --overrides \
  training_mode=self_supervised_deblur \
  source_data_dir=datasets_images \
  exp_dir=results/self_deblur_cardiac \
  self_supervised_deblur.degradation_type=cardiac_motion
```

```yaml
self_supervised_deblur:
  degradation_type: cardiac_motion
  cardiac_motion:
    num_phases: 5
    num_phases_range: null
    max_translation_mm_yx: [3.0, 3.0]
    max_rotation_deg: 3.0
    max_scale_delta: 0.04
    roi_center_yx: [0.5, 0.5]
    roi_sigma_ratio_yx: [0.25, 0.25]
    validation_translation_mm_yx: [2.0, -2.0]
    validation_rotation_deg: 2.0
    validation_scale_delta: 0.025
```

- 移動量は正規化後spacingに対するmm単位です。
- `num_phases_range: [3, 7]` とすると、学習時は画像ごとに3、5、7時相から
  ランダムに選びます。validationでは再現性のため `num_phases` を固定使用します。
- `roi_center_yx` はcrop内の心臓中心、`roi_sigma_ratio_yx` はmotion範囲を
  Y/Xサイズに対する比率で指定します。
- ROI重みは元画像と変形画像の画素値混合には使わず、変位ベクトルを外側へ
  滑らかに減衰させます。各時相には単一の連続した変形画像だけが生成されます。
- 全Zスライスに同じ心拍軌跡を適用し、スライスごとのランダムな位置ずれは
  発生させません。
- padding境界はwarped maskで正規化し、ゼロ値の混入による暗い縁と
  その逆補正による高信号haloを抑えます。
- `cardiac_motion_gaussian` を指定すると、心拍motion後にGaussian blurも
  追加できます。まずは `cardiac_motion` 単独での比較を推奨します。

これは画像空間の近似なので、CT投影角ごとのmotionに由来するstreak artifactを
完全には再現しません。実データに合わせる際は、TensorBoardの `Source Images` と
`Target Images` を比較し、移動量とROIを調整してください。

### スライス厚の追加劣化

`slice_thickness.enabled: true` は `degradation_type` と独立して適用されるため、
Gaussian、cardiac motion、または両方と併用できます。例えばclean画像の
スライス厚が3 mmで、5 mm相当へ劣化させてから元の3 mm格子へ線形補間する場合は
次のように設定します。

```yaml
self_supervised_deblur:
  degradation_type: gaussian # cardiac_motionなども使用可能
  slice_thickness:
    enabled: true
    clean_thickness_mm: 3.0
    degraded_thickness_mm: 5.0
    profile_model: gaussian_fwhm # gaussian_fwhm / box_variance
    gaussian_truncate: 3.0
```

`profile_model`では追加Gaussianのσ計算を選択できます。

- `gaussian_fwhm`: スライス厚をGaussian SSPのFWHMとみなす従来方式
- `box_variance`: スライス厚を矩形平均幅とみなし、その分散に合わせる方式

3 mm→5 mm、Z spacing 3 mmの場合、`gaussian_fwhm`は
`sigma=1.699 mm = 0.566 voxel`、`box_variance`は
`sigma=1.155 mm = 0.385 voxel`です。1 mm→5 mm、Z spacing 1 mmで
`box_variance`を使うと、`sigma=1.414 mm = 1.414 voxel`になります。

処理は次の順序です。

```text
既存degradation
  → Z方向Gaussian（3 mmから5 mmへ広げる追加PSF）
  → 5 mm間隔でサンプリング
  → 元の3 mm格子へ線形補間
```

Gaussianのslice sensitivity profileを仮定し、追加GaussianのFWHMは
`sqrt(5² - 3²) = 4 mm` として計算します。出力のshapeとspacingはclean画像から
変わりません。合成結果は `debug_dataloader.py` のcomparisonで確認できます。

データ配置は、ルート直下に `.hdr/.raw` を置くか、`train` / `val` に分けます。自己教師ありモードでは各ディレクトリ以下を再帰探索するため、spacingなどでさらにサブフォルダへ分けても利用できます。

```text
datasets_images/
  train/
    mri_2.0_2.0_2.0/
      case001.hdr
      case001.raw
  val/
    mri_2.0_2.0_2.0/
      case101.hdr
      case101.raw
```

`val` が無い場合は、`train` の画像をvalidationにも使用します。合成劣化後の
sourceとclean targetは、学習時のTensorBoardに記録される `Source Images` と
`Target Images` で確認できます。

自己教師ありモードでは、正規化、gamma、sharpness、noiseなど既存の信号augmentationをsource/targetで共有します。clean targetへ共有augmentationを一度だけ適用してsourceへコピーし、そのsourceだけに選択した劣化を加えます。

なお、この方式は「元画像よりさらにぼかした画像 → 元画像」という再劣化ペアを作る自己教師あり学習です。入力画像自体に強いblurが含まれている場合、その完全にsharpな正解画像を直接与える方式ではありません。

## ペア画像のデータ配置

通常の `training_mode: paired` では、`conf/config.yaml` で2つのデータルートを指定します。

```yaml
source_data_dir: datasets_source
target_data_dir: datasets_target
```

複数のペアフォルダを使う場合は、sourceとtargetを同じ順番・同じ数のリストで指定します。

```yaml
source_data_dir:
  - datasets_source_a
  - datasets_source_b
target_data_dir:
  - datasets_target_a
  - datasets_target_b
```

コマンドラインから指定する場合は以下のように書けます。

```bash
python main.py --overrides \
  "source_data_dir=[datasets_source_a,datasets_source_b]" \
  "target_data_dir=[datasets_target_a,datasets_target_b]"
```

各フォルダは画像枚数に応じた重みでサンプリングされます。

sourceとtargetには、同じ相対パス・同じファイル名のペア画像を置きます。
同名のtarget画像が無いsource画像はwarningを出してスキップします。

指定したフォルダ直下に `.hdr/.raw` がある場合は、その一覧をそのまま使います。

```text
datasets_source/
  case001.hdr
  case001.raw
  case002.hdr
  case002.raw

datasets_target/
  case001.hdr
  case001.raw
  case002.hdr
  case002.raw
```

この形式ではvalidation用の別フォルダが無いため、同じペア一覧をtrainingとvalidationの両方に使用します。

train/validationを分けたい場合は、以下のように `train` / `val` の直下に画像を置きます。

```text
datasets_source/
  train/
    case001.hdr
    case001.raw
  val/
    case101.hdr
    case101.raw

datasets_target/
  train/
    case001.hdr
    case001.raw
  val/
    case101.hdr
    case101.raw
```

pairedモードではサブフォルダを再帰探索しません。sourceとtargetの画像サイズ・spacingは一致している必要があります。

## 現在の標準設定

デフォルトでは、zyx spacingとcrop sizeは以下です。

```yaml
aug:
  crop_size_zyx: [8,192,192]
  affine:
    norm_spacing_zyx: [3.0,0.5,0.5]
```

Z方向は8 sliceと薄いため、通常の各stageではXY方向だけをdownsampleします。
一方で、選択した中間stageだけZ方向をdownsampleし、一定間隔の3D畳み込みを
XY畳み込みとZ畳み込みへ分解できます。

```yaml
model:
  unet:
    conv_kernel_size_zyx: [1,3,3]
    z_conv_kernel_size_zyx: [3,3,3]
    z_conv_interval: 3
    factorized_z_conv: true
    factorized_residual: true
    z_downsample_stages: [1,2]
    z_down_kernel_size_zyx: [4,1,1]
    z_down_strides_zyx: [2,1,1]
    z_upsample_type: transpose_conv
    z_up_kernel_size_zyx: [4,1,1]
    downsample_type: max_pool
    pool_size_zyx: [1,2,2]
    down_kernel_size_zyx: [1,3,3]
    upsample_type: transpose_conv
    up_kernel_size_zyx: [1,4,4]
    up_strides_zyx: [1,2,2]
    resize_conv_kernel_size_zyx: [1,3,3]
```

`z_conv_interval: 3` は「3個に1個のConv blockでZ方向も畳み込む」という意味です。`0` にするとZ方向の間引き畳み込みを無効化します。

`factorized_z_conv: true`では、上記のscheduled `[3,3,3]` Convを
`[1,3,3]` Convと`[3,1,1]` Convへ分解します。
`factorized_residual: true`では、この分解blockへidentityまたは1x1 projectionの
残差接続を追加します。

`z_downsample_stages`は0始まりのencoder stage番号です。`[1,2]`ではstage 1と2の
XY downsample後に、`Conv3D(kernel=[4,1,1], stride=[2,1,1])`でZだけを
downsampleします。decoderでは対応するstageに
`Conv3DTranspose(kernel=[4,1,1], stride=[2,1,1])`を適用してskip connectionと
同じZサイズへ戻します。この構成の全体downsample倍率は`[4,16,16]`なので、
標準crop `[8,192,192]`をそのまま使用できます。

従来U-Netへ戻す場合は次のように指定します。

```bash
python main.py --overrides \
  model.unet.factorized_z_conv=false \
  model.unet.factorized_residual=false \
  model.unet.z_downsample_stages=[] \
  model.unet.start_ch=32
```

新旧構成ではlayer形状が異なるため、既存checkpointの重みを新構成へ直接ロード
することはできません。

downsampling/upsamplingは独立に切り替えられます。従来構成は
`max_pool + transpose_conv`です。Stride Convとresize-convolutionを比較する場合は
次のように指定します。`resize_conv`は`UpSampling3D`によるnearest-neighbor resize後に
Conv3Dを適用します。

```bash
python main.py --overrides \
  model.unet.downsample_type=stride_conv \
  model.unet.upsample_type=resize_conv
```

`stride_conv`は`pool_size_zyx`をstrideとして使用し、入力channel数を維持します。
kernelは`down_kernel_size_zyx`で指定します。`resize_conv`は
`up_strides_zyx`で拡大し、`resize_conv_kernel_size_zyx`のConv3Dを適用します。

従来の実験スクリプトとの互換性のため、次の別名も使用できます。

```bash
python main.py --overrides \
  model.unet.conv_type=StridedConv \
  model.unet.up_type=ResizeUpConv
```

この場合はそれぞれ`downsample_type=stride_conv`、
`upsample_type=resize_conv`として扱われます。

### 軽量Pix2Pix Generator

`model.name=pix2pix_generator`を指定すると、既存U-Netの代わりに軽量な
Pix2Pix風Generatorを使用できます。I2I-RFRの入出力とlossは共通なので、
ネットワーク形状だけを比較できます。

```bash
python main.py --overrides \
  model.name=pix2pix_generator \
  exp_dir=results/pix2pix_generator
```

標準設定は、Encoderを`Conv3D(kernel=[1,4,4], stride=[1,2,2]) + LeakyReLU`、
Decoderを`Conv3DTranspose + ReLU + skip connection`で構成します。Z方向は
downsampleせず、出力層はI2I-RFRに合わせてtanhではなくlinearです。

```yaml
model:
  name: pix2pix_generator
  pix2pix_generator:
    start_ch: 16
    depth: 4
    max_ch: 128
    down_kernel_size_zyx: [1,4,4]
    strides_zyx: [1,2,2]
    up_kernel_size_zyx: [1,4,4]
    dropout_depth: 2
    dropout_rate: 0.3
    leaky_relu_alpha: 0.2
```

`[8,192,192,2]`入力の標準設定では、軽量Generatorは387,415 parameters、
factorized U-Net（`start_ch=40`）は15,663,579 parametersです。
軽量Generatorは約2.5%（約40分の1）
なので、速度・GPUメモリと復元性能の比較に使用できます。checkpointの形状は
異なるため、U-Netの重みをPix2Pix Generatorへ直接ロードすることはできません。

さらに小さい構成は、例えば次のように比較できます。

```bash
python main.py --overrides \
  model.name=pix2pix_generator \
  model.pix2pix_generator.start_ch=8 \
  model.pix2pix_generator.depth=3 \
  model.pix2pix_generator.max_ch=64 \
  model.pix2pix_generator.dropout_depth=1 \
  exp_dir=results/pix2pix_tiny
```

## I2I-RFR

学習時は、target画像 `y` とノイズ `eps` から以下の状態を作ります。

```text
y_t = (1 - t) * y + t * eps
```

モデルにはsource画像 `x` と noisy target `y_t` をチャンネル方向に結合して入力します。

```text
f([x; y_t]) -> y
```

損失は論文に従い、`t` で重み付けしたpixel lossです。

```text
|y - f([x; y_t])| / t
```

推論時はノイズから開始し、設定された `i2i_rfr.inference_steps` 回のEuler更新でtarget側へ戻します。デフォルトは3 stepです。

## 学習

まずDataLoaderとaugmentationの出力を確認します。

```bash
python debug_dataloader.py
```

source、target、およびX方向に `source | target` を並べたcomparisonが
それぞれ次へ保存されます。comparisonの間には4 voxelの区切りが入ります。

```text
results/exp_0001/sample_train/source/
results/exp_0001/sample_train/target/
results/exp_0001/sample_train/comparison/
results/exp_0001/sample_val/source/
results/exp_0001/sample_val/target/
results/exp_0001/sample_val/comparison/
```

学習cropは、affine augmentation後の全Zスライスに心臓マスクのbit 6が
1 voxel以上残るまで再抽選します。これにより、cropの一部が完全な背景
スライスになることを防ぎます。

```yaml
aug:
  foreground_crop:
    enabled: true
    min_voxels_per_slice: 1
    max_attempts: 20
```

条件を厳しくする場合は `min_voxels_per_slice` を増やします。20回試しても
条件を満たせない場合は、背景cropを学習へ渡さずエラーにします。その場合は
`crop_size_zyx[0]`、X/Y軸回転、`margin`、`organ_crop`を調整してください。
この機能を有効にした学習データには、画像と同じプレフィックスの
`.mask.hdr`が必要です。

学習を開始します。

```bash
python main.py
```

### validation評価指標

各validationでは、従来のMAE、MSE、PSNR、lossに加えて次の指標を
TensorBoardへ記録します。これらはランダム時刻のtrain予測ではなく、
I2I-RFRの最終生成画像に対して症例単位で計算されます。

- `val_ssim_xy_global`: axial 2D SSIM（paddingを除く画像全体、高いほど良い）
- `val_ssim_xy_heart`: 心臓マスク内のaxial 2D SSIM（高いほど良い）
- `val_psnr_heart`: 心臓マスク内のPSNR（高いほど良い）
- `val_mae_hu_heart`: 心臓マスク内のHU誤差（低いほど良い）
- `val_z_gradient_mae`: 隣接スライス差の誤差（低いほど良い）
- `val_xy_edge_strength_ratio`: 出力/targetのXY輪郭強度比（1が理想）

```yaml
evaluation_metrics:
  validation_seed: 0
  ssim_filter_size: 11
  ssim_filter_sigma: 1.5
  edge_epsilon: 1.0e-6
```

validationの初期ノイズとTensorBoardのvalidation画像は`validation_seed`で
固定されるため、epoch間の差はモデル更新による変化として比較できます。
心臓領域の指標には`bit_info.heart_bit`のマスクを使用します。

### 正解なしtest画像のTensorBoardログ

正解画像のない実データを学習中に定点観測する場合は、`test_data_dir`を
指定します。フォルダ以下の`.hdr/.raw`画像を再帰的に検索します。

```bash
python main.py --overrides \
  test_data_dir=datasets_non_gated_test \
  test_image_log.max_images=3 \
  test_image_log.seed=0 \
  test_image_log.require_heart_mask=true \
  test_image_log.heart_bit=3
```

各validation時に、TensorBoardへ次の画像が記録されます。

- `Test/Source Images`: test入力の中央Zスライス（初回のみ）
- `Test/Translated Images`: 推論結果の中央Zスライス

test画像は損失やvalidation metricには使用されません。また、epoch間でモデルの
変化だけを比較できるように、I2I-RFRの初期ノイズは`seed`で固定されます。
`test_data_dir: ""`のときは無効です。

`require_heart_mask: true`の場合、各test画像と同じプレフィックスの`.mask.hdr`から
`test_image_log.heart_bit`のbounding boxを計算し、その中心でcrop
します。マスクがない画像を画像中心へフォールバックさせず、ファイル名を示して
エラーにします。test画像ではランダムcrop、回転、拡大縮小を適用しません。
`test_image_log.heart_bit: null`の場合は、学習用の`bit_info.heart_bit`を使用します。
心臓boxのcacheは`.heart-bit3.box.txt`のようにbit別に保存されるため、学習用と
test用で異なるbitのboxが混ざることはありません。

実験ごとに設定を変える場合は、`--overrides` を使います。

```bash
python main.py --overrides \
  exp_dir=results/exp_0001 \
  source_data_dir=datasets_source \
  target_data_dir=datasets_target \
  aug.crop_size_zyx=[8,192,192] \
  aug.affine.norm_spacing_zyx=[3.0,0.5,0.5]
```

サンプルの実験スクリプトは以下です。

```bash
bash submit/exp1.sh
```

## 推論

学習済みcheckpointから検証データを変換します。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras
```

predictは基本的に学習時に保存された `output.yaml` を参照します。推論時だけI2I-RFRの更新回数などを変えたい場合は、以下のオプションで上書きできます。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --inference-steps 5 \
  --t-min 0.01 \
  --clip-output
```

自己教師ありdeblurモデルで任意のフォルダを推論する場合は、
`--source-data-dir` で指定します。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --source-data-dir /path/to/images
```

### 入力画像全体のsliding-window推論

`--sliding-window` を指定すると、対象フォルダ以下の各volume全体を推論します。
window間は中央重み付きで加重平均され、I2I-RFRの初期ノイズも全volumeで共有されます。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --source-data-dir /path/to/images \
  --sliding-window \
  --window-overlap 0.5
```

windowサイズには学習時の `aug.crop_size_zyx` を使用します。入力spacingが
学習spacingと異なる場合は内部でリサンプルし、推論後に入力と同じsize・spacingへ
戻して保存します。結果はデフォルトで次に保存されます。

```text
results/exp_0001/preds_full/
```

保存先とI2I-RFR初期ノイズは変更できます。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --source-data-dir /path/to/images \
  --sliding-window \
  --output-dir /path/to/output \
  --seed 123
```

心臓境界の過剰補正を調べる場合は、同じ入力をseed 0/1で推論し、
符号付きHU差分 `prediction - source` を同時に保存できます。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --source-data-dir /path/to/images \
  --sliding-window \
  --seeds 0 1 \
  --save-difference
```

複数seedを指定した場合、結果は上書きを避けるため次のように分けて保存されます。

```text
preds_full/seed_0/case001.hdr
preds_full/seed_0/case001.difference.hdr
preds_full/seed_1/case001.hdr
preds_full/seed_1/case001.difference.hdr
```

`*.difference.hdr/.raw` は符号付きint16のHU差分です。画像ビューアではまず
WL=0、WW=200～400程度で表示すると、正負の境界リムを確認しやすくなります。
seedを1つだけ指定した場合は、従来どおり`seed_*`サブフォルダを作りません。

sliding-window推論ではtarget画像を使用しないため、pairedモデルでも
`--source-data-dir` だけで実行できます。通常のcrop推論を行うpairedモデルでは、
sourceとtargetを両方指定してください。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --source-data-dir /path/to/source \
  --target-data-dir /path/to/target
```

オプションを省略した場合は、これまでどおり `output.yaml` の
`source_data_dir` と `target_data_dir` を使用します。

`predict.py` は推論時に不要なoptimizer状態をロードせず、GPUメモリを
段階的に確保します。それでも `ResourceExhaustedError` が出る場合は、まず
データローダと比較画像のメモリを抑えて確認してください。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --num-workers 1 \
  --prefetch-size 1 \
  --no-save-comparison
```

GPU OOMの場合はcropのY/Xを小さくすると、特徴マップのメモリ使用量を
大きく削減できます。サイズはUNetのdownsample倍率で割り切れる値を推奨します。
例えば学習時の `[8,192,192]` を次のように縮小できます。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --crop-size-zyx 8 128 128
```

GPU自体の空き容量が不足している場合は、CPU推論でも切り分けできます。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras --gpu -1
```

通常のcrop推論では、cropを小さくすると出力視野も小さくなります。
sliding-window推論ではwindow数が増えますが、最終出力は画像全体のままです。

出力は以下に保存されます。

```text
results/exp_0001/preds/
```

各症例はcrop空間の `.hdr/.raw` として保存されます。

```text
case001.input.hdr  # 入力source画像
case001.input.raw
case001.hdr        # 変換後の出力画像
case001.raw
case001.target.hdr # 正解target画像
case001.target.raw
case001.comparison.hdr # input | output | targetをX方向に結合した比較用画像
case001.comparison.raw
```

出力spacingは `aug.affine.norm_spacing_zyx` です。出力画像の画素値は
target側のclip値で逆正規化した `int16`、入力画像はcrop後のsource画像を
`int16` で保存します。正解画像がある場合はtarget画像も保存し、
`comparison` は入力、出力、正解を横並びで確認するための3D volumeです。

## 主要設定

```yaml
i2i_rfr:
  p: 1.0
  t_min: 0.01
  inference_steps: 3
  clip_output: True
```

```yaml
optimizer:
  name: adamw
```

```yaml
image:
  modality: CT
  CT:
    window_level: 60
    window_width: 600
```

MRの場合は `image.modality: MR` にし、percentileベースの正規化値を使います。

## 学習が改善しない場合

まず `debug_dataloader.py` と `predict.py` の `comparison` 出力で、入力、出力、正解が同じcrop位置に並んでいるか確認してください。位置がずれている場合は、source/targetのspacing、size、ファイル対応が一致していない可能性があります。

## 学習中にOOMが出る場合

I2I-RFRではsource画像に加えてtarget画像、ノイズ画像、noisy target、2チャンネル入力を保持するため、以前のsegmentation学習よりメモリ使用量が大きくなります。

まず `batch_size` を下げてください。CPU側のメモリ不足が疑わしい場合は、DataLoaderの並列数も下げます。

```bash
python main.py --overrides \
  batch_size=2 \
  num_workers=2 \
  prefetch_size=1
```

GPUメモリに空きがあるように見える場合でも、`tf.data` の並列crop/affine処理やprefetchでCPU RAM側がOOMになることがあります。

画像改善がほとんど見えない場合は、最初はaugmentationを弱めた設定で過学習できるか確認するのがおすすめです。

```bash
python main.py --overrides \
  aug.thick2thin_rate_zyx=[0.0,0.0,0.0] \
  aug.random_normalize.prob=0.0 \
  aug.random_gamma_correction.prob=0.0 \
  aug.random_sharpness.prob=0.0 \
  aug.random_gauss_filter.prob=0.0 \
  aug.random_gauss_noise.prob=0.0
```

少数症例で `mae` や `val_mae` が下がり、`comparison` で出力が正解に近づくことを確認してから、augmentationを戻してください。I2I-RFRの推論はノイズから開始するため、確認時は `--inference-steps 5` や `--inference-steps 10` も試す価値があります。

## 出力と除外対象

学習結果、TensorBoard log、checkpoint、予測結果、rawデータはGit管理から除外しています。

```text
results/
tensorboard_logs/
checkpoints/
preds/
datasets*/
*.raw
*.keras
```

## 開発メモ

- Keras 3 / TensorFlow環境を想定しています。
- `models/unet.py` のUNetは、Conv/Pool/TransposeConvのzyx kernel/strideをconfigから変更できます。
- `data/dataloader.py` はsource画像を基準にcrop中心とaffineを決め、target画像に同じ幾何変換を適用します。
- source側に既存のbody/prostate maskがある場合はcrop中心決定に使います。無い場合は画像全体を有効領域として扱います。
