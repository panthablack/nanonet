<template>
  <div class="dashboardViewContainer">
    <h1 class="text-xl font-bold mb-4">Dashboard</h1>
    <PrimaryButton
      @click="nanonetStore.startNanonet"
      v-if="!nanonetStore.started"
    >
      Start NanoNet
    </PrimaryButton>
  </div>
</template>

<script setup lang="ts">
import PrimaryButton from '@/dev/components/buttons/PrimaryButton.vue'
import { useNanonetStore } from '@/dev/stores/nanonetStore'

const nanonetStore = useNanonetStore()
</script>