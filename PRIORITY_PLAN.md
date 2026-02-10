# Priority Plan

Evidence-first UX を業務運用に近づけるための改修優先度を整理する。

## P0: 直近で実装すべき

1. Evidence 選定の説明データを API 契約として固定する  
`citation.span.selection` 依存をやめ、`GET /issues/{issue_id}` のレスポンスに `selection` を明示フィールドとして定義する。

2. Evidence 選定ロジックの再現性テストを追加する  
同一入力で `attach_evidence_to_issues` の選定順が安定することを、固定フィクスチャで自動検証する。

3. `include_hidden=true` の監査用途 API を分離する  
通常 UI と監査 UI の責務を分け、監査専用エンドポイントで hidden の理由と件数を返す。

4. Stage 失敗の運用導線を強化する  
`blocked_evidence` 時に「どの retry search が使われ、なぜ不足したか」を `run_stages.failure_detail` から一目で読める整形を追加する。

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
