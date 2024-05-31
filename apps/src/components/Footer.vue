<template>
  <div class="nav-container full-width absolute-bottom ">
    <div class="row">
      <div
        v-for="(item, idx) in MENU"
        :key="idx"
        class="col-grow text-center nav clickable"
        @click="updateActivePage(item)"
      >
        <q-icon
          :name="item.icon"
          class="nav-icon"
        />
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import { MENU } from './constants'

const emit = defineEmits<{
  (event: 'click', data: string): void
}>()

function updateActivePage(option: any) {
  emit('click', option)
}

</script>

<style lang="sass" scoped>
.nav-container
  background-color: white
  height: auto

.nav
  margin-top: auto
  margin-bottom: auto

.nav-icon
  font-size: 150%
  margin-top: 2vh
  margin-bottom: 2vh
</style>
