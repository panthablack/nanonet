<template>
  <div class="createModelFormContainer">
    <PrimaryButton
      @click="nanonetStore.createNanonet"
      :disabled="nanonetStore.creating"
      :text="createModelButtonText"
    />
  </div>
</template>

<script setup lang="ts">
import PrimaryButton from '@/gui/components/buttons/PrimaryButton.vue'
import { useNanonetStore } from '@/gui/stores/nanonetStore'
import { computed } from 'vue'

const nanonetStore = useNanonetStore()

const createModelButtonText = computed(() => nanonetStore.creating ? 'Creating...' : 'Create New Model')

</script>