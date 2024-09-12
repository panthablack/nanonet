<template>
  <div class="PerceptronViewContainer">
    <PageHeading>Perceptrons</PageHeading>
    <div class="actions flex gap-4 py-2">
      <CreatePerceptronForm />
      <PrimaryButton
        v-if="perceptron"
        @click="perceptron.randomise()"
      >Randomise</PrimaryButton>
      <TestPerceptronForm
        :perceptron="perceptron"
        v-if="perceptron"
      />
    </div>
    <div
      class="perceptronVisualisationContainer"
      v-if="!!perceptron"
    >
      <div class="perceptronInfo p-8 bg-blue-600 rounded-md text-white">
        {{ perceptron }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import PrimaryButton from '@/gui/components/buttons/PrimaryButton.vue'
import PageHeading from '@/gui/components/pages/PageHeading.vue'
import CreatePerceptronForm from '@/gui/components/perceptrons/CreatePerceptronForm.vue'
import TestPerceptronForm from '@/gui/components/perceptrons/TestPerceptronForm.vue'
import { usePerceptronStoreStore } from '@/gui/stores/perceptronStore'
import { computed } from 'vue'

const perceptronStore = usePerceptronStoreStore()

// show latest perceptron
const perceptron = computed(() => perceptronStore.latest)
</script>

<style scoped lang="css">
/* ... */
</style>