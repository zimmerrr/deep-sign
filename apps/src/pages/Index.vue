<template>
  <div>
    <video
      ref="videoElementRef"
      autoplay
    />

    <canvas
      ref="canvasRef"
      class="canvas"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { FilesetResolver, HolisticLandmarker } from '@mediapipe/tasks-vision'
import { waitVideoMetadata } from 'src/components/utils'

const WIDTH = 1200
const HEIGHT = 1200

const devices = ref<MediaDeviceInfo[]>([])
const error = ref('')
const running = ref(true)

const videoElementRef = ref<HTMLVideoElement>(null as any)
const canvasRef = ref<HTMLCanvasElement>(null as any)

let canvasCtx: CanvasRenderingContext2D
let holistic: HolisticLandmarker

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

    devices.value = await navigator.mediaDevices.enumerateDevices()
    devices.value = devices.value
      .filter(device => device.kind === 'videoinput')

    const camera = devices.value[0]
    if (camera) {
      console.log(`Using camera: ${camera.label} (${camera.deviceId})`)
      await selectCamera(camera.deviceId)
    }
    console.log('Camera initialized successfully')

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

    requestAnimationFrame(render)
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
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      deviceId: cameraId,
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

  holistic.detectForVideo(videoElementRef.value, performance.now(), (results: any) => {
    if (results) {
      console.log(results)
      // IMPLEMENT DRAW LANDMARKS
    }
    requestAnimationFrame(renderHolistic)
  })
}

onMounted(async () => {
  initialize()
})

// startCamera()
</script>

<style>

</style>
