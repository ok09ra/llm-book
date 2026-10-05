# 第1章 輪読会スライド

Slidev のスライド上で、本のデモコードをその場で編集・実行できます。

## 準備（初回のみ）

```bash
# スライド（Node 22 以上。nvm なら `nvm use`）
npm install

# デモ実行用の Python 環境（uv プロジェクト。transformers は本書に合わせて 4.40 系）
cd server && uv sync
```

## Windows の場合

macOS で動作確認しています。Windows では未確認ですが、使っているパッケージ（torch・fugashi・sentencepiece など）はすべて Windows 用のビルドがあるので、同じ手順で動く想定です。違いは次のとおりです。

- **Node.js：** 公式インストーラか nvm-windows で 22 以上を入れてください（nvm-windows は `.nvmrc` を読まないので `nvm install 22` / `nvm use 22`）。
- **uv：** PowerShell で `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` を実行して入れます。
- **`&&`：** Windows PowerShell 5.1 では使えません。`cd server` と `uv sync` を1行ずつ実行するか、PowerShell 7 を使ってください。
- **メモリ：** 実行サーバは起動時に全モデル（GPT-2 large を含む）を読み込むので、空きメモリが 8GB 以上あると安心です。

## 発表時

ターミナルを2つ使います。

```bash
# 1. 実行サーバ（起動時に全モデルを読み込むので、ready と出るまで待つ）
cd server && uv run python run_server.py

# 2. スライド
npm run dev
```

- コードブロック右下の ▶ で実行します。コードは書き換えられます。
- カーネルは1つを使い回すので、前のスライドで定義した変数を後のスライドでも使えます。デモは上から順に実行してください（「実装を覗く」スライドは直前のデモの変数を使います）。
- 発表者モードは http://localhost:3030/presenter です。本の出力（Out[n]）を発表者ノートに書いてあります。
