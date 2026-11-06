# 11_1：PPL評価条件とGemma Q/K削除の再検証

対象：[11_1_gemma_qk_ppl_settings_grid.ipynb](../notebooks/11_1_gemma_qk_ppl_settings_grid.ipynb)

既存の `experiments/11_1_llama_decile_analysis.ipynb` はそのまま保存している。

## 実行順序

1. 必要ならセットアップセルで `INSTALL_DEPENDENCIES=True` にして依存を入れ、カーネルを再起動しFalseに戻す。Transformers 4.57.6必須。
2. `RUN_EXPERIMENTS=True`、`PHASE='nishikawa_baseline'`で上から実行。Llama/Gemmaの未圧縮PPLを2048・全文先頭のみBOS・AutoModelForCausalLM・model.float()で測定し、7.81／10.7との差を表示。`BASELINE_MODELS`で1モデルだけにもできる。
3. 結果を確認して `PHASE='grid_baseline'`で上から再実行。Gemma未圧縮48設定。
4. `PHASE='qk_grid'`で上から再実行。未圧縮を含む計1,968評価。前段階の48結果は同一設定・環境ならスキップ。

西川氏の初期ロード精度・batch・attention・モデルrevisionは未確定。設定は明記された範囲を合わせた再現候補であり、元コードと完全一致したと断定しない。`NISHIKAWA_LOAD_PRECISION='default'`はtorch_dtype指定なしでロード後float。bf16ロード後floatも設定可能で、別結果IDになる。GemmaのAuto経路でキー不一致ならエラーを記録し、別ローダーへ黙って置換しない。

## グリッド

|軸|値|
|---|---|
|文脈長|1024、2048|
|BOS|なし、全文先頭のみ、各ブロック先頭|
|推論精度|BF16、同じBF16重みをFP32に変換|
|読み込み|既存11のConditionalGeneration→テキスト部分、AutoModelForCausalLM|
|batch|1、2|

48設定 ×（未圧縮1＋Q/K × 10群 × DE2）=1,968回。西川氏比較の2回は別。

各モジュールの全ブロックで同じ順位群を削除。第1群が最小側、第10群が最大側。FP32評価でもSVD・DE・削除重みはBF16元重みを基準とした既存11の方式に固定し、推論精度と削除操作を混同しない。RoPE等の浮動小数点bufferは精度切替時に元の値から復元する。

各ブロックBOS条件は内容L-1＋BOS、その他は内容L。末尾切り捨てと採点対象の差も記録。BOS以外の要因を完全に除去した比較ではない。PPLの集計は全採点トークンで重み付けし、batch別PPLを平均しない。

## 資源・保存

- GPUの自動確保・実行はしない。FP32・2048・batch2を含む全探索は40GB以上のGPUを推奨。L4 24GBでは全条件の成功を保証しない。OOMでbatchや精度を自動変更しない。
- 48設定は旧設定を含むため大きい。最初の2評価後に計画を判断できる。試行を減らすなら軸リストを明示的に編集し、その範囲のみの結果とする。
- 出力：`MyDrive/TUS/hashiguchi/data/ibis2026_de_lra/gemma-qk-ppl-settings-grid-v1/`
- HFキーは非表示入力。W&Bへの送信は行わない。Driveには入力、条件別損失・PPL、環境、ロード記録、変更重みハッシュ、SVDキャッシュ、集計CSV・PNGを保存。
- 完了評価はチェックサム確認後スキップ。全ジョブ完了のローダーではモデル読込・SVDもスキップ。失敗・中断した評価は再試行。部分ブロックからの再開ではなくPPL評価1条件単位。
- 入力は08_2の保存トークンと照合。データの不一致を黙って受け入れない。
- バージョン・GPU・実装の変更は別結果。異なる環境を同一のグリッドとして混ぜない。

## 読み方

`analysis/nishikawa_baseline_comparison.csv`で報告値とのPPL差。`paired_ppl.csv`の負の変化率は同設定の未圧縮より改善。`one_factor_contrasts.csv`は他の設定をそろえた1因子の影響。`completion.csv`と`failures.csv`で未完了・OOMを確認。

Q/K・DE別の全10群ヒートマップと、未圧縮の位置別PPLを生成する。色が飽和してもセル数値は実値。欠損を0として描かない。位置別PPLは先頭の不利を調べる補助診断であり、同一トークン位置を追跡した因果検証ではない。

最良設定だけを採用してtest性能改善と主張せず、削除効果の符号が設定を変えても残るかを報告する。以前の絶対PPLと新設定の削除後PPLを直接引き算しない。

## 検証範囲

`tools/check_ppl_settings_grid.py` はネットワークを無効化し、乱数初期化した極小Llamaで評価・DE・全10群削除・保存再開・分析を検証する。本物のLlama/Gemma読み込み、HF認証、Colab GPUメモリ、Drive上の実行はユーザー実行時の確認が必要。
