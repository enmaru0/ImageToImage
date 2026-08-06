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

`training_mode: self_supervised_deblur` を指定すると、`source_data_dir` の画像だけで学習できます。各画像をclean targetとして再利用し、入力側にだけランダムなGaussian blurを合成して、blur画像から元画像へ戻すdeblurを学習します。`target_data_dir`はこのモードでは使いません。

```bash
python main.py --overrides \
  training_mode=self_supervised_deblur \
  source_data_dir=datasets_images \
  exp_dir=results/self_deblur
```

blurの強さはvoxel単位のsigmaで設定します。学習時は画像ごとに範囲内からランダムに選び、validation時は比較可能なように固定値を使います。

```yaml
self_supervised_deblur:
  sigma_range: [0.5, 2.0]
  validation_sigma: 1.25
```

データ配置は通常モードと同じで、直下に `.hdr/.raw` を置くか、`train` / `val` に分けます。

```text
datasets_images/
  train/
    case001.hdr
    case001.raw
  val/
    case101.hdr
    case101.raw
```

`val` が無い場合は、`train` の画像をvalidationにも使用します。`debug_dataloader.py` の `source` 出力には合成blur後の画像、`target` 出力には元画像が保存されるため、学習前にblur強度を確認できます。

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

サブフォルダは再帰的に探索しません。sourceとtargetの画像サイズ・spacingは一致している必要があります。

## 現在の標準設定

デフォルトでは、zyx spacingとcrop sizeは以下です。

```yaml
aug:
  crop_size_zyx: [8,192,192]
  affine:
    norm_spacing_zyx: [3.0,0.5,0.5]
```

Z方向は8 sliceと薄いため、UNetではZ方向のdownsampleは行いません。一方で、Z方向の情報も完全には捨てず、一定間隔でのみZ方向を含む3D畳み込みを行います。

```yaml
model:
  unet:
    conv_kernel_size_zyx: [1,3,3]
    z_conv_kernel_size_zyx: [3,3,3]
    z_conv_interval: 3
    pool_size_zyx: [1,2,2]
    up_kernel_size_zyx: [1,4,4]
    up_strides_zyx: [1,2,2]
```

`z_conv_interval: 3` は「3個に1個のConv blockでZ方向も畳み込む」という意味です。`0` にするとZ方向の間引き畳み込みを無効化します。

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

学習を開始します。

```bash
python main.py
```

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

GPUメモリに空きがあるのにOOMが出る場合は、CPU側のデータローダや
比較画像作成が原因のことがあります。まず次の設定で確認してください。

```bash
python predict.py results/exp_0001/checkpoints/model_latest.keras \
  --num-workers 1 \
  --prefetch-size 1 \
  --no-save-comparison
```

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
