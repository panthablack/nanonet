<template>
  <div class="modalWrapper">
    <div
      @mousedown="onOverlayClicked"
      class="modalOverlay absolute top-0 left-0 z-50 h-full w-full bg-opacity-70 bg-slate-700"
      v-if="props.modelValue"
    >
      <div class="modalContainer flex justify-center items-center h-full w-full">
        <div
          class="modalContentContainer rounded-md min-w-96 bg-gray-50"
          @mousedown="onModalClicked"
        >
          <slot />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  modelValue: boolean
  persistent?: boolean
}>()

const emit = defineEmits(['update:modelValue'])

const onModalClicked = (e: Event) => {
  e.preventDefault()
  e.stopPropagation()
}

const onOverlayClicked = (e: Event) => {
  e.preventDefault()
  e.stopPropagation()
  if (props.persistent) return
  else emit('update:modelValue', false)
}
</script>