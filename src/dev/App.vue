<template>
  <div class="appContainer">
    <component :is="layout" />
  </div>
</template>

<script setup lang="ts">
import DefaultLayout from '@/dev/layouts/DefaultLayout.vue'
import { ref } from 'vue'

const layout = ref(DefaultLayout)
</script>