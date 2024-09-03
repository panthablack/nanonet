<template>
  <div
    class="modalBodyContainer bg-gray-50 p-8 overflow-hidden"
    :class="getClasses"
  >
    <slot />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  hasFooter?: boolean
  hasHeader?: boolean
}>()
const getClasses = computed(() => ({
  hasFooter: !!props.hasFooter,
  hasHeader: !!props.hasHeader
}))
</script>

<style scoped lang="css">
.modalBodyContainer {
  border-radius: inherit;
}

.modalBodyContainer.hasHeader {
  border-top-left-radius: 0;
  border-top-right-radius: 0;
}

.modalBodyContainer.hasFooter {
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
}
</style>