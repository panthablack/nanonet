<script setup lang="ts">
import NavBar from '@/gui/components/nav/NavBar.vue'
import { RouterView } from 'vue-router'
</script>

<template>
  <div class="defaultLayoutContainer">
    <header class="navBarContainer">
      <NavBar />
    </header>
    <div class="routerViewContainer p-4">
      <RouterView />
    </div>
  </div>
</template>