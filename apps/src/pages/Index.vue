<template>
  <q-page
    :padding="$q.screen.width > $q.screen.height"
    class="flex flex-center items-stretch"
  >
    <q-card class="page-card">
      <q-card-section class="header">
        <Header
          :header="header"
          :subheader="subheader"
        />
      </q-card-section>

      <q-card-section>
        <div class="container q-py-md full-height">
          <EventLogo :with-hash-tag="true" />
          <div
            v-if="menu"
            class="q-my-md text-center text-white text-h4"
          >
            <gestures v-if="mode === Mode.gestures" />
            <text-to-gesture v-if="mode === Mode.text_to_gesture" />
            <gesture-to-text v-if="mode === Mode.gesture_to_text" />
            <Upload v-if="mode === Mode.upload" />
            <Settings v-if="mode === Mode.settings" />
          </div>
        </div>
      </q-card-section>
      <Footer @click="onUpdateMenu" />
    </q-card>
  </q-page>
</template>

<script lang="ts" setup>
import Header from 'src/components/Header.vue'
import Footer from 'src/components/Footer.vue'
import Gestures from 'src/components/Gestures.vue'
import TextToGesture from 'src/components/TextToGesture.vue'
import GestureToText from 'src/components/GestureToText.vue'
import Upload from 'src/components/Upload.vue'
import Settings from 'src/components/Settings.vue'
import { MENU } from 'src/components/constants'
import { ref } from 'vue'

enum Mode {
  loading,
  gestures,
  text_to_gesture,
  gesture_to_text,
  upload,
  settings,
}

const hasModel = ref(true)
const mode = ref<Mode>(hasModel.value ? Mode.gestures : Mode.loading)
const lastScannerMode = ref(0)

const menu = ref({})
const header = ref('Gesture')
const subheader = ref('Gesture')

function onUpdateMenu(option: any) {
  header.value = option.header
  subheader.value = option.subheader
  menu.value = option
  setMode(Mode[option.key as keyof typeof Mode])
}

function setMode(newMode: Mode) {
  lastScannerMode.value = newMode
  mode.value = newMode
}
</script>

<style scoped lang="sass">
.page-card
  background: rgb(12,104,243)
  background: linear-gradient(180deg, rgba(12,104,243,1) 15%, rgba(76,213,189,1) 91%)
  max-width: 550px
  width: 100%

.header
  height: 8vh

@media (max-width: 1024px)
  .page-card
    max-width: 1024px
    width: 100%

:deep(.q-field--outlined .q-field__control:before)
  border: 2px solid $primary
  transition: border-color 0.36s cubic-bezier(0.4, 0, 0.2, 1)
</style>
