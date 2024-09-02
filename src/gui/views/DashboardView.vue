<template>
  <div class="dashboardViewContainer">
    <h1 class="text-xl font-bold mb-4">Dashboard</h1>
    <PrimaryButton
      @click="nanonetStore.createNanonet"
      v-if="!nanonetStore.creating"
    >
      Create New NanoNet
    </PrimaryButton>
  </div>
</template>

<script setup lang="ts">
import PrimaryButton from '@/gui/components/buttons/PrimaryButton.vue'
import { useNanonetStore } from '@/gui/stores/nanonetStore'

const nanonetStore = useNanonetStore()
</script>