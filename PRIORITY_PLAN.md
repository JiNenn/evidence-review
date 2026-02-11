# Priority Plan

Evidence-first UX を業務運用に近づけるための改修優先度を整理する。

## 実装状況（最新）

- 完了: P0-1 `selection` 明示返却（互換で `citation.span.selection` 併存）
- 完了: P0-2 選定再現性の smoke 検証（同一入力の別Run比較）
- 完了: P0-3 監査専用 API `/runs/{run_id}/audit/issues` 追加（互換維持）
- 完了: P0-4 `blocked_evidence` の failure_detail に summary を追加
- 完了: P1-1 vector fallback を char n-gram cosine 検索で実体化
- 完了: P1-2 issue fingerprint の重複抑止（char n-gram Jaccard統合 + v3 fingerprint）

## P0: 直近で実装すべき

1. Evidence 選定の説明データを API 契約として固定する  
`citation.span.selection` 依存をやめ、`GET /issues/{issue_id}` のレスポンスに `selection` を明示フィールドとして定義する。
補足: Issue クリック時に右パネルで evidence を表示する前提で、通常表示は「該当文」と「どの資料か」を優先し、行番号表示は必須にしない。

2. Evidence 選定ロジックの再現性テストを追加する  
同一入力で `attach_evidence_to_issues` の選定順が安定することを、固定フィクスチャで自動検証する。

3. `include_hidden=true` の監査用途 API を分離する  
通常 UI と監査 UI の責務を分け、監査専用エンドポイントで hidden の理由と件数を返す。
補足: 主画面は Issue 一覧を主役にし、raw diff は `debug` 導線に分離する。監査用途はこの `debug` 導線側で扱う。

4. Stage 失敗の運用導線を強化する  
`blocked_evidence` 時に「どの retry search が使われ、なぜ不足したか」を `run_stages.failure_detail` から一目で読める整形を追加する。
補足: `blocked_evidence` 時は再実行ボタンを置かず、推測理由を説明する表示を優先する。必要に応じて `confidence` 低下の扱いを合わせて定義する。

## P1: 次スプリントで着手

1. 検索品質の向上（vector fallback の実体化）  
現状の疑似 vector モードを本実装に置き換え、長文・言い換えに強い evidence 回収へ寄せる。

2. Issue fingerprint の重複抑止を強化する  
語尾違い・表記揺れで重複 issue が増える問題を、正規化規則と閾値で抑制する。

3. Stage 実行履歴の明細化  
`attempt` だけでなく、必要なら `run_stage_attempts` を導入して失敗履歴と復旧判断を残す。

## P2: 中期改善

1. 権限制御の本実装  
`AUTH_DEV_USER` 依存を脱却し、role ベースで `include_hidden` と監査機能を制御する。

2. 観測性の標準化  
選定スコア分布、hidden 率、retry 成功率をメトリクス化して継続監視できるようにする。

3. UI のレビュー効率改善  
Issue 一覧で evidence 強度、影響度、未解決状態で並び替え/フィルタできる操作を追加する。
補足: 一覧の基本データは `title / summary / severity / status` を基準とする。`severity` はギャップの大きさに基づく `high / medium / low` を採用する。0件時は「内容がほとんど一致しています」を表示する。

## UI決定事項（確定）

1. `severity` は API/DB の正本を `int` のまま維持し、UI で `high / medium / low` へ閾値マッピングする。

2. 0件時メッセージは `run.status` で分岐する。  
`success` のみ「内容がほとんど一致しています」を表示し、`blocked_evidence` は根拠不足、`running/pending` は処理中メッセージにする。

3. 監査導線は UI を通常画面と分離し、API は互換維持 + 監査専用 endpoint を追加する。  
`GET /runs/{run_id}/issues?include_hidden=true` は当面残し、`GET /runs/{run_id}/audit/issues` を追加する。

4. `selection` は互換期間中は二重返却する。  
新: `GET /issues/{issue_id}` の明示 `selection`  
旧: `citation.span.selection`

5. `blocked_evidence` の復旧操作は Issue単位ではなく Run単位の `debug` 画面に限定する。  
主画面には再実行ボタンを置かない。
