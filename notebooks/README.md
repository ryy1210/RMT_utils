# 研究ノートブック一覧

実験コードはこのリポジトリで管理し、入力データ・スペクトル・チェックポイント・実験結果は従来のGoogle Drive / W&Bに保存します。GPTの作業フォルダにあった元ファイルは削除していません。今後の編集は本リポジトリの対応ファイルを正本としてください。

## 番号順の入口

番号は過去の実験名を維持しています。**02〜05は `bulk_production/` にあります。** 同じ番号のLlama/Gemma版はそれぞれ別モデル用です。

|番号|ノートブック|用途・主な入力|実行環境|
|---|---|---|---|
|00|[環境確認](00_environment_check.ipynb)|ライブラリ・実行環境の確認|ローカル/Colab|
|01|[LLM実験入口](01_llm_experiment.ipynb)|共通ランナーを使う実験|GPU推奨|
|00/01（bulk準備）|[接続確認](bulk_production/00_preflight.ipynb) / [Colab準備](bulk_production/01_colab_setup.ipynb)|Drive接続・既存メタデータ確認|Colab|
|02|[Llama試行](bulk_production/02_llama_production.ipynb) / [Gemma試行](bulk_production/02_gemma_production.ipynb)|モデル重みからSVD・DE・bulkKSを試行|Colab GPU|
|03|[Llama全行列](bulk_production/03_llama_full.ipynb) / [Gemma全行列](bulk_production/03_gemma_full.ipynb)|全対象行列のSVD・DE・bulkKS、Drive/W&B保存|Colab GPU|
|04|[両端bulkKS](bulk_production/04_two_sided_bulk_ks.ipynb)|保存済みスペクトルで上下端を除外したKSを計算|CPU|
|05|[スペクトル可視化](bulk_production/05_spectral_histogram_diagnostics.ipynb)|保存済みスペクトル・KSのヒストグラム診断|CPU|
|06|[小さいKSのLRAとLoRA](06_low_ks_lra_finetune.ipynb)|行列選択・rank下限・DE有無の比較|GPU|
|07|[異分散性と下側外れ値の乱数実験](07_de_synthetic_lower_edge.ipynb)|分離可能ノイズ、低ランク信号、相関、ランク欠損の合成モデル|CPU可、大規模設定は高負荷|
|08|[DE/LRAの統制実験](08_de_lra_controlled_experiments.ipynb)|削減量を統制したモジュール別・単一行列・順序実験|GPU|
|08_2|[全行列の単独LRAとbulkKS順序](08_2_de_lra_all_single_bulk_order.ipynb)|全行列スクリーニングと両端bulkKSによる順序|GPU|
|09|[上下端の特異値を残すLRA](09_de_two_edge_lra.ipynb)|Hのみ / LとH、同rank対照・追加実験。既存08の評価設定を参照|GPU|
|09_2|[モジュール全層の上下端LRA](09_2_de_two_edge_module_all.ipynb)|UP/DOWN/GATE/Q/K/V/O別の全層処理|GPU|
|09_3|[Llama MAIN分析](09_3_llama_main_analysis.ipynb)|09の保存済みLlama結果|CPU|
|10|[KS変化とPPLの関連](10_bulkks_ppl_association.ipynb)|保存済みbulkKSと単一行列LRAの結果を結合|CPU|
|11|[10群削除の実験](11_de_decile_module_pruning.ipynb)|モジュール別に特異値を10群に分け、DE前後で削除。Llama/Gemma対応|GPU|
|11_1|[Llamaの10群分析](experiments/11_1_llama_decile_analysis.ipynb)|11の保存済みLlama結果|CPU|

## その他

- [bulkKS共通診断](bulk_ks_diagnostics.ipynb)
- [bulkKSの合成データ検証](validation/bulk_ks_synthetic_validation.ipynb)：小さい乱数行列による実装チェック。07の異分散モデル実験とは別。
- [ポスター用Llama ESD](analysis/llama_esd_normalized.ipynb)：保存済みスペクトルを使用。ローカルDriveとポスター出力先の設定を確認する。
- [従来のLlama/Gemma/Dyson実験](experiments/)・[旧版アーカイブ](archive/)
- [従来ノートブックの実行ガイド](../docs/notebook_execution.md)

## 入力・実行上の注意

- 主な入力は `MyDrive/TUS/hashiguchi/data/` 以下。ノートブック内の `DATA_ROOT`、`SOURCE_TAG`、モデルrevision、結果保存先を確認してください。データ本体やモデル重みはGitに追加していません。
- コードだけで再実行できる合成実験と、過去の保存結果が必要な分析を区別してください。例えば03→04→05、09→09_3、11→11_1の順に結果を受け渡します。10はbulkKSと単一行列LRAの両方の保存結果が必要です。
- 認証情報は環境変数または非表示入力で渡します。キーをコードに書き込んだ状態でコミットしないでください。
- 移管では既存の実験条件・実行フラグを維持しています。「すべて実行」でGPU計算、結果保存、W&B更新が始まるものがあります。03には実行後のColab切断セルもあります。
- ローカルではリポジトリを作業ディレクトリにすると既存のパス設定を利用しやすくなります。Colab用セルはローカル環境ではそのまま実行できない場合があります。

## 2026-11-05の移管

GPT作業領域から未登録の15冊を追加しました。移管元・ファイルハッシュ・軽微な修正は [移管記録](codex_migration_manifest.json) に記録しています。セル出力・実行カウント・UIメタデータを除去し、元の実行済みファイルは元の場所に保持しました。実験結果の再検証やモデルの再実行は今回行っていません。

06はリポジトリ側に後続の不具合修正があるため旧版で上書きせず、07・08・08_2も既存の正本を維持しています。07のGPT版との主要な違いは実行フラグです。旧版・リポジトリの複製・以前移管済みの実験は重複追加していません。

検証範囲：追加した15冊のnbformat形式、IPython構文変換後の構文チェック、実行出力の除去、認証情報パターンの検査。実行時のDrive参照・モデル取得・GPU計算・保存再開は未検証です。再実行時は必要なDriveデータ・認証・GPUカーネルを準備し、設定セルを確認してから実行してください。
