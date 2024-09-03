<template>
  <div class="modalHeadingContainer p-4 bg-blue-950 text-gray-100">
    <h2>
      <slot />
    </h2>
  </div>
</template>

<style scoped lang="css">
.modalHeadingContainer {
  border-top-left-radius: inherit;
  border-top-right-radius: inherit;
}
</style>