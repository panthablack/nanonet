<template>
  <div class="createPerceptronFormContainer">
    <PrimaryButton
      @click="onCreateClicked"
      :disabled="creating"
      :text="createPerceptronButtonText"
    />
    <Modal
      :modelValue="creating"
      @update:modelValue="abortCreation"
    >
      <ModalHeading>Create New Perceptron</ModalHeading>
      <ModalBody
        hasFooter
        hasHeader
      >
        <TextInput
          v-model="generate"
          label="Number of Inputs"
          type="number"
          @submit="onSubmit"
        />
      </ModalBody>
      <ModalFooter>
        <PrimaryButton
          type="submit"
          @click="onSubmit"
        >Create Perceptron</PrimaryButton>
      </ModalFooter>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import PrimaryButton from '@/gui/components/buttons/PrimaryButton.vue'
import TextInput from '@/gui/components/forms/TextInput.vue'
import Modal from '@/gui/components/modals/Modal.vue'
import ModalBody from '@/gui/components/modals/ModalBody.vue'
import ModalFooter from '@/gui/components/modals/ModalFooter.vue'
import ModalHeading from '@/gui/components/modals/ModalHeader.vue'
import { usePerceptronStoreStore } from '@/gui/stores/perceptronStore'
import { computed, ref } from 'vue'

const perceptronStore = usePerceptronStoreStore()

const creating = ref(false)

const generate = ref('2')

const formValues = computed(() => ({
  generate: parseInt(generate.value),
  randomise: true
}))

const onCreateClicked = () => {
  creating.value = true
}

const onSubmit = () => {
  perceptronStore.createPerceptron(formValues.value)
  resetForm()
}

const abortCreation = () => resetForm()

const resetForm = () => {
  creating.value = false
}

const createPerceptronButtonText = computed(() => creating.value ? 'Creating...' : 'Create New Perceptron')

</script>