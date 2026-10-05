<!-- 全スライドの右下にページ番号を表示する（表紙は除く） -->
<template>
  <div
    v-if="$nav.currentPage > 1"
    class="absolute bottom-3 right-5 text-sm opacity-60"
  >
    {{ $nav.currentPage }} / {{ $nav.total }}
  </div>
</template>
