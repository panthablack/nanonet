<template>
  <div class="dashboardViewContainer">
    <h1 class="text-xl font-bold mb-4">Dashboard</h1>
    <PrimaryButton
      @click="startNanonet"
      v-if="!started"
    >
      Start NanoNet
    </PrimaryButton>
  </div>
</template>

<script setup lang="ts">
import PrimaryButton from '@/dev/components/buttons/PrimaryButton.vue'
import { useNanonetStore } from '@/dev/stores/nanonetStore'

const { started, startNanonet } = useNanonetStore()
</script>