<template>
  <div class="createModelFormContainer">
    <PrimaryButton
      @click="nanonetStore.createNanonet"
      :disabled="nanonetStore.creating"
      :text="createModelButtonText"
    />
    <Modal
      :modelValue="nanonetStore.creating"
      @update:modelValue="nanonetStore.abortCreation"
    >
      <p>Hello Joe!</p>
    </Modal>
    <Modal v-model="showTestModal">
      <p>Hello Joe! This is the test Modal!</p>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import PrimaryButton from '@/gui/components/buttons/PrimaryButton.vue'
import Modal from '@/gui/components/modals/Modal.vue'
import { useNanonetStore } from '@/gui/stores/nanonetStore'
import { computed, ref } from 'vue'

const nanonetStore = useNanonetStore()
const showTestModal = ref(true)

const createModelButtonText = computed(() => nanonetStore.creating ? 'Creating...' : 'Create New Model')

</script>