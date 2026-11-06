# 12: 全10群削除とLoRAの順序 × DE

`notebooks/12_de_decile_finetune_order.ipynb` をVS CodeでColab GPUカーネルに接続して使用します。既存06/11は変更していません。本番モデルの学習・評価は作成時には実行していません。

## 設定

- `MODEL_KEY='llama'` または `'gemma'`：1つだけ選択。未選択モデルのトークン準備・モデル取得・学習・評価はしません。
- `SEEDS=[42]`：seedごとに独立保存。後からseedを追加できます。
- `PHASES`：既定では `finetune_then_prune` と `prune_then_finetune` の両方。
- 全7モジュール・全10群・DE有無。1モデル・1seedにつき共通学習1回＋削除後学習140回、最終比較280条件。削除直後の追加評価140回とbaseline2回もあります。
- `RUN_EXPERIMENTS=False` が納品時の既定。設定・入力・軽量チェックを確認し、Trueにして本番セルを実行してください。
- `INSTALL_DEPENDENCIES` は必要なときだけTrue。HFキーは環境変数か非表示入力。W&B送信は実装していません。

LoRAは既存06の基本条件（rank 8 / alpha 16 / dropout 0.05、1 epoch、learning rate 2e-4、microbatch 1、勾配蓄積8）を踏襲し、全7種の射影を学習します。学習設定はtest PPLで調整せず固定します。削除した方向の再獲得を許すため、学習後のrankを固定する圧縮実験ではありません。

学習→削除では、共通の無削除モデルのLoRA更新をmergeした後、SVDとDEを再計算します。削除→学習では逆DE変換した重みを学習します。同じ群番号でもDE前後で異なる特異方向を削除する点に注意してください。

## 入力・保存先

学習：`Salesforce/wikitext`, `wikitext-2-raw-v1`, train。dataset/tokenizer revisionとトークン列を保存します。

評価：従来08/11の保存済みtest tokensを使います。`LEGACY_OUTPUT` の `llama_tokens.npy/json` または `gemma_tokens.npy/json` が必要です。見つからない場合は保存先を修正し、別データへ黙って置換しないでください。

出力：`MyDrive/TUS/hashiguchi/data/ibis2026_de_lra/decile-finetune-order-wikitext2-v1/{llama|gemma}/`

- `manifest.json`：条件・実装・ライブラリ・入力の指紋。
- `seed-42/baseline_raw/`, `baseline_ft/`：無削除の学習前後PPL。
- `seed-42/shared_training/`：共通学習モデルのadapterと再開checkpoint。
- `seed-42/{phase}/{module}/de{0|1}_decile{01..10}/`：条件別結果・削除行列の記録。削除→学習には条件別adapter/checkpointも保存。
- `results.csv`, `paired.csv`, PDF：選択モデルの集計。PPL増分は無削除・学習済みモデルとの差。DE差は負ならDEありのPPLが低い。

## 再開

同じ設定で「すべて実行」。完了条件は保存損失と行列計画のチェックサムを検証してスキップします。全条件完了ならモデル・SVD・学習をスキップします。モデルを切り替えるにはMODEL_KEYだけ変更します。

学習途中は保存済みの完全なcheckpointから再開し、最後の保存以降は再計算します。完了adapterがあれば再学習せず評価を再開します。ローカルSVDキャッシュはColab切断で消える場合があります。削除後の重みのハッシュを確認して、以前と異なるbaseにadapterをロードしないよう停止します。

条件、コード、ライブラリ、入力が変わった場合は同じ保存先への混在を拒否します。元の設定・環境へ戻すか、別の出力タグを指定してください。同じ保存先へ複数カーネルから同時に書き込まないでください。多数のoptimizer checkpointを保持するのでDriveの空き容量を確認してください。

## 検証

ローカルCPUで再現する検証：

```bash
.venv/bin/python tools/check_decile_finetune_notebook.py
```

小型のランダムLlamaモデルのみを使い、事前学習済みモデルや実データは取得しません。10群分割・DE逆変換、LoRAの実学習・merge・adapter再読込、途中checkpointからの学習再開と重みの一致を検証します。実行制御は学習を軽量な模擬処理に置き換え、1モジュールの全40条件、21学習開始、完了スキップと破損検出を検証します。選択したモデルだけの入力で準備できることも検証します。

未検証：実Llama/GemmaのCUDA/BF16学習、GPUメモリ使用量、実際のDrive上の再開、全条件完走。最初のモデルの共通学習からGPUで確認してください。結果が出る前にDEの有効性を主張しないでください。

BoolQは今回含みません。BoolQ専用学習を別途行う場合、後日追加しても必要な学習条件数は変わりませんが、モデルロードなどの準備コストは増えます。BoolQ評価だけならBoolQ学習は必須ではありません。
