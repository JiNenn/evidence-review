# UI Review Sheet (Recovery Flow)

Home + Run 一覧 + Run 詳細 + Debug の復旧導線を、共通認証セッション前提で検証する。

## 1. 画面遷移

1. `/` は Home + Run 一覧であること  
期待: 既存 Run の公開メタ情報のみを表示し、主要導線は「Run 詳細へ」。

2. `/runs/{run_id}` への遷移時に未認証ならログイン導線へ遷移すること  
期待: `401` 時のみ `/login?next=/runs/{run_id}` に遷移する。

3. ログイン成功後に元の Run 詳細へ復帰すること  
期待: `next`（または同等の `return_to`）でクリック元に戻る。

4. `/debug` は通常導線と分離されること  
期待: 監査/復旧用途として別導線でアクセスする。

## 2. Home + Run 一覧 `/`

1. 一覧は公開メタ情報のみを表示すること  
期待: タイトル、作成日時、状態、件数サマリのみを表示する。

2. 一覧の主目的が「詳細へ入る」操作になっていること  
期待: 各行に `詳細` 操作があり、Run 詳細へ遷移できる。

3. 初手でトークン取得を要求しないこと  
期待: Home に `Token取得` の明示操作がない。

## 3. Run 詳細 `/runs/{run_id}`

1. Issue 一覧は根拠付きのみを表示すること  
期待: `evidences>=1` のみ表示し、`hidden` は通常表示しない。

2. 0件時メッセージが `run.status` で分岐すること  
期待:
- `success`: 「内容がほとんど一致しています。」
- `blocked_evidence`: 根拠不足メッセージ
- `running/pending`: 処理中メッセージ

3. `blocked_evidence` 時に `Debugで復旧` ボタンが表示されること  
期待: `run_id` と `return_to` を付与して `/debug` へ遷移する。

4. Issue 一覧の sort/filter が有効であること  
期待:
- 並び順（影響度/根拠強度/confidence/更新時刻/未解決優先）で順序が変わる
- 状態/影響度/根拠強度フィルタで件数が変わる
- 条件不一致時は「フィルタ条件に一致する論点はありません。」を表示

5. Issue 詳細の evidence が根拠強度順に並ぶこと  
期待: `combined_score` 降順、同点時は `search_rank` 昇順。

## 4. Debug `/debug`

1. Debug は復旧専用タスク画面であること  
期待: Auth 入力セクションがなく、共通セッション前提で動作する。

2. 復旧実行後に `run.status` を監視すること  
期待: `success/success_partial` になるまで監視し、進捗が見える。

3. 復旧成功時に自動復帰導線を表示すること  
期待: 「3秒後に戻ります（今すぐ戻る/このままDebugに残る）」を表示する。

4. 復旧失敗時は Debug に留まること  
期待: `failure_detail.summary` と次アクションを表示する。

5. hidden を含む監査情報が取得できること  
期待: `/runs/{run_id}/audit/issues`, stage attempts, run metrics を確認できる。

## 5. 権限制御（AUTH_ENABLED=true）

1. `admin`/`audit` は監査導線にアクセスできること  
期待: `/runs/{run_id}/issues?include_hidden=true` と `/runs/{run_id}/audit/issues` が成功。

2. `viewer` は監査導線にアクセスできないこと  
期待: 上記2 API が `403`。

3. 権限不足時の UI 挙動が一貫していること  
期待: `401` はログイン遷移、`403` は権限不足説明を表示する。
