<template>
  <div class="createModelFormContainer">
    <PrimaryButton
      @click="onCreateClicked"
      :disabled="creating"
      :text="createModelButtonText"
    />
    <Modal
      :modelValue="creating"
      @update:modelValue="abortCreation"
    >
      <ModalHeading>Create New Model</ModalHeading>
      <ModalBody
        hasFooter
        hasHeader
      >
        <p>
          (For now this just creates a default Model, but will have a customisable shape later.)
        </p>
      </ModalBody>
      <ModalFooter>
        <PrimaryButton
          type="submit"
          @click="onSubmit"
        >Create Model</PrimaryButton>
      </ModalFooter>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import PrimaryButton from '@/gui/components/buttons/PrimaryButton.vue'
import Modal from '@/gui/components/modals/Modal.vue'
import ModalBody from '@/gui/components/modals/ModalBody.vue'
import ModalFooter from '@/gui/components/modals/ModalFooter.vue'
import ModalHeading from '@/gui/components/modals/ModalHeader.vue'
import { useNanonetStore } from '@/gui/stores/nanonetStore'
import { computed, reactive, ref } from 'vue'

const nanonetStore = useNanonetStore()

const creating = ref(false)

const createFormValues = reactive({

})

const onCreateClicked = () => {
  creating.value = true
}

const onSubmit = () => {
  nanonetStore.createModel(createFormValues)
  resetForm()
}

const abortCreation = () => resetForm()

const resetForm = () => {
  creating.value = false
}

const createModelButtonText = computed(() => creating.value ? 'Creating...' : 'Create New Model')

</script>