import { defineCodeRunnersSetup } from '@slidev/types'

const escape = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

// 改行や空白を保ったまま表示する
const pre = (s: string, color = 'inherit') =>
  ({ html: `<pre style="margin:0;white-space:pre-wrap;max-height:220px;overflow:auto;color:${color}">${escape(s)}</pre>` })

// ```py {monaco-run} のコードを server/run_server.py の常駐カーネルで実行する
async function runPython(code: string) {
  try {
    const res = await fetch('http://127.0.0.1:8765/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    })
    const { output, error } = await res.json()
    if (error)
      return pre(output ? `${output}\n${error}` : error, '#dc2626')
    return pre(output || '(出力なし)')
  }
  catch {
    return pre('実行サーバに接続できません。server/ で `uv run python run_server.py` を起動してください。', '#dc2626')
  }
}

export default defineCodeRunnersSetup(() => ({
  python: runPython,
  py: runPython,
}))
