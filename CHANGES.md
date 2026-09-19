# 変更記録（2026-09-19）

基点：2959f275bd21a2e0e631a12b693ccb5f15c75f78。

| 対象 | 問題 | 変更 | 数値への影響 |
|---|---|---|---|
| DE | 横長行列でn×nのVを生成 | thin SVD。引数・戻り値維持 | 正常行列では参照計算と一致 |
| ESD | pre/postの分解を重複実行 | pre SVDを再利用、postも一度だけ | float64化による微小差、境界付近のrankに注意 |
| KS | 経験CDFの右側しか比較しない | 左右両側を比較、MP積分を再利用 | 指標と行列の順序が変わり得る |
| α | CUDA histogram、少数/零固有値、累積和i=0の扱い | NumPy化、推定不能はNaN、tail和の修正 | エッジケースとgoodness-of-fitで変化 |
| LRA | SVD処理の重複、過大q、全dtypeのfp16 clip | 共通化、rankチェック、dtypeに応じた異常検出 | 異常入力を黙って補正せず停止 |
| 実験履歴 | PPLcalc=Falseの最終PPLを捨てる | 最終行へ保存 | 既存列のNoneが実測PPLになる |
| α pruning | q_proj等の末尾一致で他blockのαを使用 | 完全名/一意の全pathで照合 | 過去のα-pruning結果の再実験が必要 |
| sparsity | 行列個数平均、clip後の予算不一致 | パラメータ数加重・上限付き予算調整 | 割当てとPPLが変化 |
| mask | 同点を一括削除、CUDA固定 | kthvalueと同点内の指定個数選択 | 実際の削除数が修正される |
| PPL | NaNバッチ除外、全loss保持、配置の破壊 | 非有限値は失敗、chunk和、device_map尊重 | 無効なPPLを出さない。低精度との差あり |
| large PPL | 学習済みnormではなく新規norm | 実モデルforwardへ統一 | 旧値と一致しない。手動offloadは廃止 |
| データ | WikiText train上書き/test split誤指定、校正末尾欠落 | split修正、token列から正確な個数を切出し | 校正・旧get_wikitext2の内容が変わる |
| データcache | tokenizerを識別しないdisk cache | 校正disk cacheを廃止、評価CPU cacheはtokenizer別 | 過去の別モデルtoken混入を防止 |
| WrappedLayer | batch平均を単純加算、未定義DEBUG/inp1 | token加重平均、入力平均でbias補正 | Wanda統計がbatch分割に依存しなくなる |
| ViT | Linear数4固定、argsのpathを破壊、勾配保持 | 実数を取得、local path、no_grad、予算再配分 | 可変構造を扱える |
| LoRA | 古いPEFT/TrainingArguments API、CPU fp16、state_dict復元漏れ | 現行API、旧/新引数判定、finallyで復元 | APIを保ち例外時も状態を復元 |
| Prompter | 空文字、区切りなし、未知templateで例外 | 空文字対応、1回だけsplit、JSON読込 | 不正/特殊入力での動作修正 |
| peft同梱コード | 不完全コピーがpip版を隠す | legacy/peftへ移動、元ファイルを保持 | pip版が使われる。旧pickleは未検証 |

変更箇所のコメントで目的を確認できます。古い実装を大量にコメントアウトして残す代わりに、履歴の基点とこの表から比較できるようにしています。
