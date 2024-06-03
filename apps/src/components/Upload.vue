<template>
  <q-form>
    <div class="row text-left">
      <!-- NAME -->
      <div class="col-12">
        <div class="text-subtitle2 q-mb-sm">
          Name (Optional)
        </div>
        <div>
          <q-input
            v-model="form.name"
            label="Enter your name"
            bg-color="white"
            color="primary"
            dense
            filled
            outlined
            bottom-slots
          />
        </div>
      </div>

      <!-- EMAIL -->
      <div class="col-12">
        <div class="text-subtitle2 q-mb-sm">
          Email
        </div>
        <div>
          <q-input
            v-model="form.name"
            label="Enter your email"
            bg-color="white"
            color="primary"
            dense
            filled
            outlined
            bottom-slots
          />
        </div>
      </div>

      <!-- GESTURE NAME -->
      <div class="col-12">
        <div class="text-subtitle2 q-mb-sm">
          Gesture Name
        </div>
        <div>
          <q-input
            v-model="form.name"
            label="Enter the name of the gesture"
            bg-color="white"
            color="primary"
            dense
            filled
            outlined
            bottom-slots
          />
        </div>
      </div>

      <!-- CATEGORY -->
      <div class="col-12 q-mt-sm">
        <div class="row">
          <div class="col-2 text-subtitle2  q-pt-sm">
            Category
          </div>
          <div class="col-10">
            <q-select
              v-model="form.name"
              label="Enter the name of the gesture"
              bg-color="white"
              color="primary"
              dense
              filled
              outlined
              bottom-slots
            />
          </div>
        </div>
      </div>
    </div>
  </q-form>
</template>

<script setup lang="ts">

const form = {
  name: '',
  email: '',
  gestureName: '',
  category: '',
  file: '',
}

</script>

<style lang="sass" scoped>
:deep(.q-field__control)
  border-radius: 12px
</style>
