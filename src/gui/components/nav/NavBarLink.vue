<script setup lang="ts">
import {
  RouterLink,
  type RouteLocationAsPathGeneric,
  type RouteLocationAsRelativeGeneric
} from 'vue-router'

const { to } = defineProps<{
  to: string | RouteLocationAsRelativeGeneric | RouteLocationAsPathGeneric
}>()
</script>

<template>
  <RouterLink
    :to="to"
    active-class="linkActive"
    class="navBarLink "
  >
    <slot />
  </RouterLink>
</template>

<style scoped lang="css">
.navBarLink {
  @apply text-gray-50 font-semibold hover:text-blue-300 active:text-blue-500;
}

.linkActive.navBarLink {
  @apply text-blue-300 font-semibold hover:text-blue-400 active:text-blue-500;
}
</style>