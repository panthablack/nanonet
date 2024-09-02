<script setup lang="ts">
import NavBarLink from '@/gui/components/nav/NavBarLink.vue'
</script>

<template>
  <nav class="navBar p-4 flex flex-no-wrap gap-y-4 gap-x-6 bg-gray-700">
    <NavBarLink to="/">Dashboard</NavBarLink>
    <NavBarLink to="/models">Models</NavBarLink>
    <NavBarLink to="/models/train">Training</NavBarLink>
    <NavBarLink to="/models/test">Testing</NavBarLink>
  </nav>
</template>