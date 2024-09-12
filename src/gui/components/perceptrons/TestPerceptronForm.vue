<template>
  <div class="testPerceptronFormContainer">
    <PrimaryButton
      @click="onCreateClicked"
      :disabled="creating"
      :text="createPerceptronButtonText"
    />
    <Modal
      :modelValue="creating"
      @update:modelValue="abortCreation"
    >
      <ModalHeading>Test Values</ModalHeading>
      <ModalBody
        hasFooter
        hasHeader
      >
        <TextInput
          v-for="i in perceptron.inputs.length"
          :key="`perceptronInput${i}`"
          :modelValue="String(testValues[i - 1])"
          @update:modelValue="testValues[i - 1] = Number($event)"
          :label="`Input${i}`"
          type="number"
          @submit="onSubmit"
        />
      </ModalBody>
      <ModalFooter>
        <PrimaryButton
          type="submit"
          @click="onSubmit"
        >Test Perceptron</PrimaryButton>
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
import { Perceptron } from '@/nanonet/classes/Perceptron'
import { computed, ref, type Ref } from 'vue'

const props = defineProps<{
  perceptron: Perceptron
}>()

const creating = ref(false)

const testValues: Ref<number[]> = ref([])

const onCreateClicked = () => {
  creating.value = true
  for (let i = 0; i < props.perceptron.inputs.length; i++) testValues.value[i] = i
}

const onSubmit = () => {
  props.perceptron.run(testValues.value)
  resetForm()
}

const abortCreation = () => resetForm()

const resetForm = () => {
  creating.value = false
}

const createPerceptronButtonText = computed(() => creating.value ? 'Testing...' : 'Test Perceptron')

</script>