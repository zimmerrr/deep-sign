<template>
  <div class="header col-12 font-opensans text-weight-bold">
    Settings
  </div>
  <div class="row q-col-gutter-y-md q-mt-sm">
    <div class="col-7 label text-left">
      Model Version
    </div>
    <div class="col-5">
      <q-select
        v-model="modelVersion"
        filled
        dense
        square
        bg-color="white"
        :options="modelVersionOptions"
        @blur=" emit('updateModel', modelVersion)"
      />
    </div>
    <div class="col-7 label text-left">
      Mediapipe Complexity
    </div>
    <div class="col-5">
      <q-select
        v-model="complexity"
        filled
        dense
        square
        bg-color="white"
        :options="complexityOptions"
        @blur=" emit('updateComplexity', complexity)"
      />
    </div>
    <div class="col-7 label text-left">
      Minimum Detection Confidence
    </div>
    <div class="col-5">
      <q-input
        v-model="minDetectionConfidence"
        filled
        dense
        square
        bg-color="white"
        :options="complexityOptions"
        @blur=" emit('updateDetConf', minDetectionConfidence)"
      />
    </div>
    <div class="col-7 label text-left">
      Minimum Tracking Confidence
    </div>
    <div class="col-5">
      <q-input
        v-model="minTrackingConfidence"
        filled
        dense
        square
        bg-color="white"
        :options="complexityOptions"
        @blur=" emit('updateTrackConf', minTrackingConfidence)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  (event: 'updateModel', data: string): void
  (event: 'updateComplexity', data: string): void
  (event: 'updateDetConf', data: number): void
  (event: 'updateTrackConf', data: number): void
}>()

const props = defineProps({
  settings: { type: Object, default: null },
})

const modelVersionOptions = [
  'DeepSign v6', 'DeepSign v3', 'DeepSign v2',
]
const complexityOptions = [
  'Lite', 'Full', 'Heavy',
]

const modelVersion = ref(props.settings.modelVersion)
const complexity = ref(props.settings.modelComplexity)
const minDetectionConfidence = ref(props.settings.minDetectionConfidence)
const minTrackingConfidence = ref(props.settings.minTrackingConfidence)

</script>
<style lang="sass" scoped>
.header
  text-align: left
  font-size: 1.5rem
  border-bottom: 2px solid white

.label
  font-size: 50%
</style>
