<template>
  <q-page class="flex flex-center">
    <div class="app-logo fade">
      <q-img
        src="~assets/logo.png"
        no-spinner
      />
    </div>
    <div>
      <Camera class="full-width" />
    </div>
  </q-page>
</template>

<script lang="ts" setup>
import Camera from 'src/components/Camera.vue'
</script>
<style lang="sass" scoped>
.app-logo
  width: 90%
.fade
  animation: fade 2s ease-in-out

@keyframes fade
  0%
    opacity: 0
    transform: opaci(0.5)

  100%
    transform: scale(1)
    opacity: 1

</style>
