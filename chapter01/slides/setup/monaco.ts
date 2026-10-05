import { defineMonacoSetup } from '@slidev/types'

// 2列レイアウトでも長い行が切れないよう、折り返して表示する
export default defineMonacoSetup(() => ({
  editorOptions: {
    wordWrap: 'on',
  },
}))
