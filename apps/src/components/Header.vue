<template>
  <div class="header-container full-width absolute-top">
    <div class="row text-white">
      <div class="col-9">
        <div class="text-h5 text-weight-bold q-ma-md">
          {{ props.header }}
        </div>
      </div>
      <div class="col-3">
        <div class="q-pa-sm">
          <q-img
            src="~assets/logo.png"
            contain
            basic
            no-spinner
            class="logo"
          />
        </div>
      </div>
      <div class="col-12">
        <div class="q-mx-md">
          {{ props.subheader }}
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>

const props = defineProps({
  header: { type: String, default: null },
  subheader: { type: String, default: null },
})
</script>

<style lang="sass" scoped>
.header-container
  height: 10vh

.logo
  width: 100px !important
</style>
