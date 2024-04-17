<template>
  <q-layout view="lHh Lpr lFf">
    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script setup lang="ts">

defineOptions({
  name: 'MainLayout',
})

</script>

<style lang="sass">
.q-layout
  background-color: #71b6ef
</style>
