<template>
  <div class="appContainer">
    <component :is="layout" />
  </div>
</template>

<script setup lang="ts">
import DefaultLayout from '@/dev/layouts/DefaultLayout.vue'
import { markRaw, ref } from 'vue'

const layout = ref(markRaw(DefaultLayout))
</script>