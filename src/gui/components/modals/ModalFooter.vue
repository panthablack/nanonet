<template>
  <div class="modalFooterContainer p-4 bg-gray-100 justify-end flex text-gray-100">
    <h2>
      <slot />
    </h2>
  </div>
</template>

<style scoped lang="css">
.modalFooterContainer {
  border-bottom-left-radius: inherit;
  border-bottom-right-radius: inherit;
}
</style>