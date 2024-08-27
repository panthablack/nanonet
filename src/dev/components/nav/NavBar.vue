<script setup lang="ts">
import { RouterLink } from 'vue-router'
</script>

<template>
  <header class="navBarContainer">
    <nav>
      <RouterLink to="/">Dashboard</RouterLink>
      <RouterLink to="/models/test">Model Testing</RouterLink>
    </nav>
  </header>
</template>