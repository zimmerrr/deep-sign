<template>
  <q-page
    :padding="$q.screen.width > $q.screen.height"
    class="flex flex-center items-stretch"
  >
    <q-card class="content-center page-card">
      <Header
        :header="header"
        :subheader="subheader"
      />
      <q-card-section>
        <div class="content-center container q-py-md full-height">
          <EventLogo :with-hash-tag="true" />
          <div
            v-if="menu"
            class="q-my-md text-center text-white text-h4"
          >
            {{ header }}
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
import { MENU } from 'src/components/constants'
import { ref } from 'vue'

const menu = ref({})
const header = ref('Gesture')
const subheader = ref('Gesture')

function onUpdateMenu(option: any) {
  header.value = option.header
  subheader.value = option.subheader
  menu.value = option
}
</script>

<style scoped lang="sass">
.page-card
  max-width: 550px
  width: 100%

@media (max-width: 1024px)
  .page-card
    max-width: 1024px
    width: 100%

:deep(.q-field--outlined .q-field__control:before)
  border: 2px solid $primary
  transition: border-color 0.36s cubic-bezier(0.4, 0, 0.2, 1)
</style>

<style scoped lang="sass">
.page-card
  background: rgb(12,104,243)
  background: linear-gradient(180deg, rgba(12,104,243,1) 15%, rgba(76,213,189,1) 91%)
  max-width: 550px
  width: 100%

@media (max-width: 1024px)
  .page-card
    max-width: 1024px
    width: 100%

:deep(.q-field--outlined .q-field__control:before)
  border: 2px solid $primary
  transition: border-color 0.36s cubic-bezier(0.4, 0, 0.2, 1)
</style>
