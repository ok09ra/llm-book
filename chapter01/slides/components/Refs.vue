<!-- スライド左下に出典を小さく表示する -->
<template>
  <div class="absolute bottom-2 left-14 right-24 text-[10px] leading-tight opacity-60 refs">
    出典：<slot />
  </div>
</template>

<style scoped>
.refs :deep(a) {
  color: inherit;
  text-decoration: underline;
}
</style>
