<template>
  <div class="container q-mx-auto">
    <div class="video-container">
      <q-responsive :ratio="WIDTH/HEIGHT">
        <canvas
          ref="canvasRef"
          class="canvas absolute-full"
          style="z-index: 1;"
        />
        <video
          ref="videoElementRef"
          autoplay
          class="absolute-full"
        />
      </q-responsive>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, onMounted, onBeforeMount, withCtx } from 'vue'
import { FilesetResolver, HolisticLandmarker, DrawingUtils, NormalizedLandmark } from '@mediapipe/tasks-vision'
import { FilesetResolver, HolisticLandmarker, DrawingUtils, NormalizedLandmark } from '@mediapipe/holistic'
import { waitVideoMetadata } from 'src/components/utils'

const WIDTH = 500
const HEIGHT = 500

const devices = ref<MediaDeviceInfo[]>([])
const error = ref('')
const running = ref(true)

const videoElementRef = ref<HTMLVideoElement>(null as any)
const canvasRef = ref<HTMLCanvasElement>(null as any)

let canvasCtx: CanvasRenderingContext2D
let holistic: HolisticLandmarker
let drawingUtils: DrawingUtils

const config = {
  locateFile: (file) => {
    return 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/pose_landmark_full.tflite'
  },
}

// Function to initialize camera
async function initialize() {
  console.log('Initializing camera...')
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      throw new Error('MediaDevices not available')
    }

    const ctx = canvasRef.value.getContext('2d')
    if (!ctx) throw new Error('Unable to get canvas context')
    canvasCtx = ctx
    canvasRef.value.width = WIDTH
    canvasRef.value.height = HEIGHT
    drawingUtils = new DrawingUtils(ctx)

    devices.value = await navigator.mediaDevices.enumerateDevices()
    devices.value = devices.value
      .filter(device => device.kind === 'videoinput')

    const camera = devices.value[0]
    if (camera) {
      console.log(`Using camera: ${camera.label} (${camera.deviceId})`)
      await selectCamera(camera.deviceId)
    }
    console.log('Camera initialized successfully')
    running.value = true
    // IMPLEMENT MEDIAPIPE MODE
    const mediapipe = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
    )

    holistic = await HolisticLandmarker.createFromOptions(mediapipe, {
      baseOptions: {
        modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task',
      },
      runningMode: 'VIDEO',
    })

    // requestAnimationFrame(render)
    requestAnimationFrame(renderHolistic)
  } catch (err) {
    console.error(err)
    error.value = 'Unable to initialize camera'
  } finally {
    console.log('Initialized Successfully')
  }
}

async function selectCamera(cameraId: string) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: cameraId,
      width: 500,
      height: 500,
    },
  })
  console.log(stream)
  videoElementRef.value.srcObject = stream
  await waitVideoMetadata(videoElementRef.value)
}

function render() {
  if (!running.value) return
  canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)

  canvasCtx.save()
  canvasCtx.restore()

  setTimeout(() => {
    requestAnimationFrame(render)
  }, 1000 / 30)
}

function renderHolistic() {
  if (!running.value) return
  const startTime = performance.now()
  holistic.detectForVideo(videoElementRef.value, performance.now(), (results: any) => {
    console.log((performance.now() - startTime))
    if (results) {
      // POSE
      if (results.poseLandmarks.length > 0) {
        canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)
        drawingUtils.drawLandmarks(results.poseLandmarks[0])
      }

      // RIGHT HAND
      if (results.rightHandLandmarks.length > 0) {
        canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)
        drawingUtils.drawLandmarks(results.rightHandLandmarks[0])
      }

      // RIGHT HAND
      if (results.leftHandLandmarks.length > 0) {
        canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)
        drawingUtils.drawLandmarks(results.leftHandLandmarks[0])
      }

      // IMPLEMENT DRAW LANDMARKS
      // for (const result in results) {
      //   const normalizedLandmark = results[result]
      //   console.log(normalizedLandmark)

      //   // Loop through each array
      //   for (let i = 0; i < normalizedLandmark.length; i++) {
      //     console.log(normalizedLandmark[i])
      //     // drawingUtils.drawConnectors(normalizedLandmark, normalizedLandmark[i], { color: '#ffffff' })
      //   }
      // }
    }
    requestAnimationFrame(renderHolistic)
  })
}

onMounted(async () => {
  initialize()
})

onBeforeMount(async () => {
  running.value = false
})

// startCamera()
</script>
<style lang="sass" scoped>
.container
  width: 100%
  height: 70vh !important

.video-container
  width: 100%
  // height: 100%

</style>
