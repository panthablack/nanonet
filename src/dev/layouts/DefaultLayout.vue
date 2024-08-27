<script setup lang="ts">
import NavBar from '@/dev/components/nav/NavBar.vue'
import { RouterView } from 'vue-router'
</script>

<template>
  <div class="defaultLayoutContainer">
    <NavBar />
    <div class="routerViewContainer">
      <RouterView />
    </div>
  </div>
</template>