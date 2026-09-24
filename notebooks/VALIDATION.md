# バルクKS実装の検証（2026-09-24）

- CPUで40 tests passed（新規11件、既存29件）。
- Notebookを人工行列モードで全セル実行し、2つの比較図を目視確認。
- CPU版TorchによるDE前後のスペクトルが既存NumPy実装と一致することを、正方・縦長・横長・奇数次元で確認。
- 空バルク、閾値境界、不正入力、途中再開、異なる設定の混入拒否、既存KS保持を確認。
- W&B保存はモックでTable/Artifactと新規runの呼出しを確認。実際の送信は未実施。
- Driveの保存済みESD pickleを読み込み、Llama 196行・Gemma 238行と必要列を確認。
- 本番モデルの取得、Colab接続、CUDA実行、実データSVD、Driveへの本番結果保存は未実施。
- W&Bの2 runとモデルの対応、およびモデルrevisionは本番前にNotebookで確認する。

検証環境: Python 3.12、torch 2.14.0、numpy 2.3.5、scipy 1.18.1、
pandas 2.2.3、transformers 4.57.6、peft 0.17.1。
Transformers 5.17.0では既存LoRAのgroup_by_length引数が非互換だったため、
今回のNotebook用依存設定は4.57.6へ固定。既存LoRAコードには変更を加えていない。

GPU方針: GPUを自動取得しない。人工行列はCPUのみ。本番許可後にT4/CPUで
少数行列を計測し、必要な場合のみ資源を増強する。モデルはCPUに保持するため
GPU容量だけでなくホストRAMも確認する。float64 SVDの速度はprobeで測る。
