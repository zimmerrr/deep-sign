<template>
  <router-view />
</template>

<script setup lang="ts">
import type { PermissionState } from '@capacitor/core'

defineOptions({
  name: 'App',
})

export interface PermissionStatus {
  // TODO: change 'location' to the actual name of your alias!
  camera: PermissionState;
}
</script>
