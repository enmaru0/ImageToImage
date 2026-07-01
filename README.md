# Med3D-DL
本プロジェクトは、可能な限りKerasのコンポーネントを活用して書いています。
前立腺抽出のタスクにおいて本プロジェクトを使用し、性能面とC++実装しても問題ないことまで確認済みです。

## 使い方
### 環境構築
本リポジトリはKeras 3.0以降を基盤としています。
- Docker イメージ: `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`

```bash
# 必要なPythonライブラリをインストール
# ~/.localにインストールされます。環境を分けたい場合はvenvなどを使用してください。
pip install -r requirements.txt --user
```

#### リンター & フォーマッターについて
コードの整形や解析には `Ruff` を使用しています。特に VS Code で開発する方には、`Ruff` の拡張機能のインストールを推奨します。これにより、ファイル保存時に自動でコードが整形されます。

### 環境設定ファイル
- `pyproject.toml`: `Ruff` の設定ファイルです。
- `./.vscode/settings.json`: VS Code 用の設定ファイルです。`"jupyter.notebookFileRoot": "${workspaceFolder}"` を指定しているため、ノートブック実行時に、README.md の配置と同じフォルダ階層を基準として動作します。


## サポートしている機能
- シングルクラス・マルチクラスのセグメンテーション
- 最適化アルゴリズム
  - Cosine Annealing with Warmup
  - SGD、AdamW
- チェックポイントの保存と復元
  - 最新モデルおよび最適な評価スコア（例: DICE）を基準に保存
  - 学習の途中での復元（cfg.restoreを使用）cfgは変更しないこと
  - ファインチューニング（cfg.finetuneを使用）エポック数とoptimizerがリセットされます。
- TensorBoard 統合
  - 損失関数とメトリクスの記録
  - 検証データ入力およびモデル出力画像の記録（`callbacks/image_logger.py`）
- モデル
  - UNet
  - マスクを用いたバッチ正規化範囲の限定
  - Batch Renormalization(マルチGPUは未検証)
- 損失関数
  - DICEロスおよびCrossEntropyロスの実装
    - `tf.gather`を使わずにignore領域を効率的に指定
  - 複雑な形状の入力（正解マスクや予測以外）に対応
- データローダー
  - 複数のデータセット割合に基づく確率的サンプリングとバッチ作成
  - 必要な画像とマスク領域のみを効率的に読み込み
  - アフィン変換によるスケーリング、フリップ、回転、シフト、クロップ処理
  - `thin -> thick` 変換（スライス感度プロフィールに近い形で計算）
- 前処理およびデータ拡張
  - シャープネス、ガウシアンノイズ、ガウシアンブラーのGPU実装（`data/gpu_aug/random_noise.py`）
  - CT: ウインドウレベルと幅を活用した正規化
  - MR: パーセンタイルを用いた正規化
  - ランダム中心値と範囲を使用した正規化
  - ガンマ補正
- モデルエクスポート
  - `.cpp`と`.h`形式のモデル出力（`.bin`形式も対応可、ただし非推奨）

## 注意点
- [Keras 3.0 では `tf.Variable`を`keras.Model`の中で使用しないようにしてください。](https://keras.io/guides/migrating_to_keras_3/#tensorflow-variables-tracking)
- [`tf.data.Dataset`内でtf関数を使用してもCPUのみで実行されます。](https://www.tensorflow.org/guide/data_performance_analysis?hl=ja#3_cpu_%E4%BD%BF%E7%94%A8%E7%8E%87%E3%81%8C%E9%AB%98%E3%81%8F%E3%81%AA%E3%81%A3%E3%81%A6%E3%81%84%E3%82%8B%E3%81%8B%EF%BC%9F)

## 実験方法
### 実験ファイルの作成
`./submit/exp1.sh`や`./submit/exp2.sh`にサンプルスクリプトがあります。
`conf/config.yaml`を直接変更せず、これらのシェルスクリプトの内容を変更してください。この方法により、複数条件での実験実施およびパラメータの追跡が容易になります。
### ジョブの投入
`./submit/submit.sh`はCloudHPCにジョブ投入する際のサンプルスクリプトです。必要に応じて `JOB_SCRIPT`を編集して使用してください。
### 結果の確認
結果は`./results`も保存されます。
`./submit/exp1.sh`の初めにある`python utils/debug_dataloader.py ${OPTIONS}`が実行することによって、モデルの入力画像（データ拡張などの処理後の画像）の一部を生成されるので、これを確認することでデータローダーの動作を確認できます。

## フォルダ構成（一部抜粋）
```
.
├── README.md
├── callbacks               # カスタムコールバック
├── conf                    # 実験設定ファイル
├── data
│   └── dataloader.py       # データの読み込みコード
├── datasets_prostate       # サンプルデータセット
├── debug_dataloader.py     # データ拡張などを実施後の画像を作成するコード
├── export_params.py        # パラメータの出力コード
├── losses                  # カスタム損失関数
├── main.py
├── models
├── notebooks
├── predict.py              # 学習後に推論するためのコード
├── pyproject.toml
├── requirements.txt
├── results
├── submit                  # 実験用スクリプト
├── trainer.py              # カスタムトレーニングループ
└── utils
    └── rescale_dataset.py  # 学習データのスペーシングをあらかじめ揃えるためのコード
```

## TODO
- [ ] ファイルの直接バックアップ
- [ ] マルチGPU学習のサポート


## Known Issues
