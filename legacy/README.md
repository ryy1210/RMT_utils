# 旧PEFT参照コード

修正: ルートの`peft/`をここへ移動し、pip版PEFTへの名前の衝突を解消した。
元のファイルは保存しているが、utils等が欠けた0.3.0.dev0の部分コピーであり実行用ではない。
mapping/peft_modelはadapterの組込み・タスク別ラッパー、tuners以下はLoRA/AdaLoRA/各prompt方式、import_utilsは任意依存の検出。
学習には`pip install peft`で導入した正規パッケージを使う。過去pickle内の独自PEFTクラスの復元互換性は保証しない。
