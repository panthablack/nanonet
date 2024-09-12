<template>
  <div class="textInputContainer">
    <label
      v-if="label"
      class="block text-sm font-medium leading-6 text-gray-900"
      :for="inputId"
    >{{ label }}</label>
    <div class="mt-2">
      <input
        :id="inputId"
        :value="modelValue"
        class="textInput block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 px-2"
        :placeholder="placeholder"
        @click="onInputClicked"
        @input="onInput"
        @keyup.enter="onSubmit"
        :type="type"
      >
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  inputId?: string
  label?: string
  placeholder?: string
  modelValue?: string
  type?: string
}>()

const emit = defineEmits(['submit', 'update:modelValue'])

const onInput = (e: Event) => emit('update:modelValue', (e.target as HTMLInputElement).value)
const onSubmit = (e: Event) => emit('submit', e)

// TODO: This is basically just to deal with the fact that the modal is currently preventing inputs from being clicked, so eventually need to fix that bug
const onInputClicked = (e: Event) => (e?.target as HTMLInputElement)?.focus()
</script>