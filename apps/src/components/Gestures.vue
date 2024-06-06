<template>
  <div class="container q-mx-auto">
    <q-scroll-area
      class="fit"
      :thumb-style="{'opacity': '0'}"
    >
      <div
        class="gesture-container q-mx-auto"
      >
        <q-card
          v-for="(gesture, idx) in GESTURES"
          :key="idx"
          @click="zoom(gesture)"
        >
          <q-img
            :src="gesture.src"
            class="image"
            ratio="1"
            flat
          />
          <div class="text-black absolute-bottom-right image-label q-mb-md q-mr-md">
            {{ gesture.name }}
          </div>
        </q-card>
      </div>
    </q-scroll-area>
  </div>

  <q-dialog v-model="showDialog">
    <q-card class="popup">
      <q-img
        :src="activeGesture.src"
        class="image"
        ratio="1"
      />
      <div class="text-black absolute-bottom-right image-label">
        {{ activeGesture.name }}
      </div>
    </q-card>
  </q-dialog>
</template>
<script setup lang="ts">
import { ref } from 'vue'
import { GESTURES } from './gestures'

const showDialog = ref(false)
const activeGesture = ref()
function zoom(gesture: any) {
  showDialog.value = true
  activeGesture.value = gesture
}
</script>
<style lang="sass" scoped>
.container
  width: 100%
  height: 75vh !important
.gesture-container
  width: 90%
  display: grid
  grid-template-rows: auto
  grid-template-columns: repeat(3, 1fr)
  column-gap: 5%
  row-gap: 2%
  filter: drop-shadow(0px 5px 8px rgba(black, 0.5))

  .image
    object-fit: cover

  .image-label
    font-size: 70%
    font-weight: 500

.popup
  width: 400px
  max-width: 90%

  .image-label
    font-size: 500%
    font-weight: 500
    margin-right: 10%
</style>
