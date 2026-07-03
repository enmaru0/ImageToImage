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

## データ配置

`conf/config.yaml` で2つのデータルートを指定します。

```yaml
source_data_dir: datasets_source
target_data_dir: datasets_target
```

sourceとtargetには、同じ相対パス・同じファイル名のペア画像を置きます。

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

出力は以下に保存されます。

```text
results/exp_0001/preds/
```

各症例はcrop空間の `.hdr/.raw` として保存されます。出力spacingは
`aug.affine.norm_spacing_zyx`、画素値はtarget側のclip値で逆正規化した
`int16` です。

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
