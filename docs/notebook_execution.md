# VS Code・Colab・Codexでの実験

## 構成と移行範囲（2026-09-24）

- `notebooks/archive/`: Llama 3.2 3B / Instruct / Gemma 3 4B / Dysonの原コード。セルのコード・順番は維持し、出力・実行番号・Colabメタデータだけを除去。
- `notebooks/experiments/`: 上記4冊の認証、保存先、clone/pipセルを調整した版。**過去の実験ログであり、Run Allで完走する構成ではない**。
- `notebooks/00_environment_check.ipynb`: モデル取得なしでCPU上のDE/BEMA・小型Llama/Gemma LoRA逆伝播を検証。
- `notebooks/01_llm_experiment.ipynb`: 新規実験の設定・起動画面。既定は設定確認のみ。
- `tools/run_llm_experiment.py`: 新しいプロセスからESD / 基準PPL / LRAを実行する入口。実モデルではCUDAが必要。過去のLoRA実験全体の自動化は含まない。
- `notebooks/migration_manifest.json`: 移行元名、原本SHA-256、登録先の対応。Drive検索が原本IDを返さなかったためIDは未確認。

過去の実験出力はDrive原本と、Git管理外の`.migration-backup/`に保存した。Drive原本は削除していない。今後のコード編集はGitHub側を正本にする。その他の授業ノート、ViT/AlexNetノートは今回の対象外。

## ローカルの準備

このMacはApple Silicon・メモリ8GB。ローカルは小規模テスト・分析用、実モデルのPPL/LoRAはColab GPU用とする。CUDA指定をMPSに置換するだけで同条件の実験になるとは限らない。

```sh
cd /Users/ryoyazushi/GitHub/RMT_utils
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-local-lock.txt
.venv/bin/python -m ipykernel install --sys-prefix --name python3 --display-name 'RMT_utils (.venv)'
.venv/bin/python -m pytest -q
.venv/bin/python -m nbconvert --execute --to notebook notebooks/00_environment_check.ipynb --output-dir outputs/validation
```

VS Codeでリポジトリを開き、ノートブック右上のカーネルで`.venv/bin/python`を選ぶ。Python / Jupyter / Google Colab拡張は導入済み。仮想環境はGitHubには登録されない。

## Colab有料プランをVS Codeで使う

[Google公式拡張](https://github.com/googlecolab/colab-vscode)と[公式ユーザーガイド](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)で確認。

1. ノートブックの「カーネルの選択」→「別のカーネルを選択」→「Colab」→「New Colab Server」。
2. 有料プランを契約しているGoogleアカウントでサインインし、GPUを選ぶ。Gemmaのbfloat16にはL4/A100等を選択する。
3. ローカルのリポジトリはリモートへ自動同期されない。初回は空のノートブックで以下を実行するか、VS Codeのリポジトリフォルダを右クリック→`Upload to Colab`で転送する。

```python
!git clone https://github.com/ryy1210/RMT_utils.git /content/RMT_utils
%pip install -r /content/RMT_utils/requirements-notebooks.txt
```

既にclone済みなら再cloneせず、未コミット変更を確認した上で更新する。依存パッケージを更新したらカーネルを再起動する。Colab既存のCUDA版torchを優先し、Mac用の固定一覧をColabへそのまま適用しない。

4. `00_environment_check.ipynb`で実行確認。HFの認証が必要な実験では、実行カーネル内で次を行う（トークンをコードに直書きしない）。モデル利用規約への同意・アクセス権も必要。

```python
from huggingface_hub import login
login()
```

5. 既存データを使う場合は、公式拡張の`Colab: Mount Google Drive to Server...`でDriveをマウントし、先頭の設定セルより前に保存先を設定する。

```python
import os
os.environ['RMT_DATA_DIR'] = '/content/drive/MyDrive/TUS/hashiguchi/data'
os.environ['RMT_OUTPUT_DIR'] = '/content/drive/MyDrive/TUS/hashiguchi/experiment_outputs'
```

ローカルでは`RMT_DATA_DIR`を同期済みの`TUS/hashiguchi/data`の絶対パスに設定する。未指定ならリポジトリの`outputs/data`。WandBは移行版で既定をofflineにした。既存の`wandb.Api()`による過去run読取には別途WandB認証が必要。

Colab計算ユニットの消費はVS Codeステータスバーで確認できる。2026-09-24のVS Code画面でColab Pro認識と計算ユニット表示を確認した。GPU割当・接続成功は未検証。終了時は結果をDriveへ保存またはダウンロードし、`Colab: Remove Server`でサーバーを終了する。リモート`/content`だけの保存は永続化にならない。

## 新規実験の実行

Colabカーネルのセルで、現在のPythonを使って起動する。下記はLlamaの例。Gemmaは`--model gemma`、Instructは`--model llama-instruct`。

```python
import subprocess, sys
subprocess.run([
    sys.executable, '/content/RMT_utils/tools/run_llm_experiment.py',
    '--model', 'llama', '--stage', 'ppl',
    '--output', '/content/drive/MyDrive/TUS/hashiguchi/experiment_outputs/llama_baseline_001',
], check=True)
```

ESDは`--stage esd`で実行し、`esd_metrics.pkl`を保存する。LRAは`--stage lra --metrics <このrunnerで作成した同じモデルのesd_metrics.pkl> --max-layers 50`。既定はalpha昇順・DEあり。KS降順は`--sort-by KS_postDE_1 --descending`、DEなしは`--no-de`。出力先は新規ディレクトリに限り、既存結果の上書きを防ぐ。各実行に条件・Git commit・パッケージ版・GPU・終了状態を記録する。pickleは信頼できる自分のファイルのみ読む。

モデル全体のESDはCPUで大きなSVDを行うため、GPUを選んでもCPU RAMと時間を使う。LRAの削減率はパラメータ数に基づく値で、実測メモリ・速度ではない。LoRAは移行済み実験ノートの対象セクションから実行し、未圧縮+LoRA対照とseedを記録する。

## 既存ノートブックの制約

- Gemma: 原本セル3で`model, lm_model`を削除した後、セル7で参照する。目的に応じて削除セルを飛ばすかモデルを再ロードする。
- Llama: ESD読込・計算などの代替手順と複数回のLoRA実験が同居する。原本セル22の`result`など、別セルで作られる変数が必要。
- Instruct/Dyson: 大きなSVD、途中段階の実装、代替実験が含まれる。
- CUDA直指定・fp16/bf16指定は元の実験条件として残している。ローカルCPU/MPSで全実験をそのまま動かす変更はしていない。
- `!git pull`や実験途中のパッケージ更新・`runtime.unassign()`は移行版から無効化した。
- 元の出力と現在のRMT_utilsの実装時点は異なる。今回の移行は過去の数値の再現を証明しない。

## Codexからの実行範囲

このワークスペースのCodexはローカル`.venv`を使ってコード編集・テスト・ノートブック実行ができる。VS CodeでColabカーネルを選んでも、Codexのローカルターミナルが自動でそのGPUにつながるわけではない。リモート実行にはVS Codeのセル実行を使い、変更をUploadまたはGitで反映する。Googleログイン済み・Colab Pro認識は確認済み。実GPU上の接続・実モデル実行は今回未検証。無人の長時間実験を保証する構成ではない。

## 今回の検証結果

- Python 3.11 / torch 2.8.0 / Transformers 4.57.6 / PEFT 0.17.1で、既存テスト29件成功。
- `00_environment_check.ipynb`はCPUで全セル実行成功。保存済みの出力を確認。
- 10冊のノートブック形式、移行前後の原コード一致、SHA-256、認証情報パターン、Git差分を検査。
- `requirements-local-lock.txt`はこのMacの検証済みパッケージ一覧。Colabは`requirements-notebooks.txt`を使用。
- フルサイズのモデル取得・GPU上のPPL/LRA/LoRA、移行済み実験ログ4冊の全セル実行は未実施。
