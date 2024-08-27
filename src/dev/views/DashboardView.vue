<template>
  <div class="dashboardViewContainer">
    <h1 class="clickable">Dashboard</h1>
    <Button
      @click="startNanonet"
      v-if="!started"
    >
      Start NanoNet
    </Button>
  </div>
</template>

<script setup lang="ts">
import Button from '@/dev/components/PrimaryButton.vue'
import { useNanonetStore } from '@/dev/stores/nanonetStore'

const { started, startNanonet } = useNanonetStore()
</script>