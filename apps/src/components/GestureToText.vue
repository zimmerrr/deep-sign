<template>
  <div
    :class="{'hidden': !inference}"
    class="text-subtitle2 absolute-top-right q-mr-md"
    style="z-index: 2;"
  >
    Inference: {{ inference.toFixed(2) }}ms
  </div>
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
import * as Holistic from '@mediapipe/holistic'
import { drawConnectors, drawLandmarks, lerp, Data } from '@mediapipe/drawing_utils'
import { waitVideoMetadata } from 'src/components/utils'
import * as ort from 'onnxruntime-web'

const WIDTH = 500
const HEIGHT = 500

const devices = ref<MediaDeviceInfo[]>([])
const error = ref('')
const running = ref(true)

const videoElementRef = ref<HTMLVideoElement>(null as any)
const canvasRef = ref<HTMLCanvasElement>(null as any)

const debugMode = ref(true)
let startTime = performance.now()
const inference = ref(0)

let canvasCtx: CanvasRenderingContext2D
let holistic: Holistic.Holistic

const config: Holistic.HolisticConfig = {
  locateFile: (file: any) => {
    console.log('https://cdn.jsdelivr.net/npm/@mediapipe/holistic@' +
         `${Holistic.VERSION}/${file}`)
    return 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic@' +
         `${Holistic.VERSION}/${file}`
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

    holistic = new Holistic.Holistic(config)
    holistic.setOptions({ modelComplexity: 0 })
    holistic.onResults(onResults)
    requestAnimationFrame(render)
    // requestAnimationFrame(renderHolistic)
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

async function render() {
  if (!running.value) return
  canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)
  canvasCtx.save()
  canvasCtx.restore()
  startTime = performance.now()

  await holistic.send({ image: videoElementRef.value })
  setTimeout(() => {
    requestAnimationFrame(render)
  }, 1000 / 30)
}

async function useModel() {
  try {
    const session = await ort.InferenceSession.create('./model.onnx')

    // prepare inputs. a tensor need its corresponding TypedArray as data
    const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    const tensorA = new ort.Tensor('float32', dataA, [3, 4])
    const tensorB = new ort.Tensor('float32', dataB, [4, 3])

    // prepare feeds. use model input names as keys.
    const feeds = { a: tensorA, b: tensorB }

    // feed inputs and run
    const results = await session.run(feeds)

    // read from results
    const dataC = results.c.data
    document.write(`data of result tensor 'c': ${dataC}`)
  } catch (e) {
    document.write(`failed to inference ONNX model: ${e}.`)
  }
}

// function renderHolistic() {
//   if (!running.value) return
//   const startTime = performance.now()
//   holistic.detectForVideo(videoElementRef.value, performance.now(), (results: any) => {
//     console.log((performance.now() - startTime))
//     if (results) {
//       // POSE
//       if (results.poseLandmarks.length > 0) {
//         canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)
//         drawingUtils.drawLandmarks(results.poseLandmarks[0])
//       }

//       // RIGHT HAND
//       if (results.rightHandLandmarks.length > 0) {
//         canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)
//         drawingUtils.drawLandmarks(results.rightHandLandmarks[0])
//       }

//       // RIGHT HAND
//       if (results.leftHandLandmarks.length > 0) {
//         canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)
//         drawingUtils.drawLandmarks(results.leftHandLandmarks[0])
//       }

//       // IMPLEMENT DRAW LANDMARKS
//       // for (const result in results) {
//       //   const normalizedLandmark = results[result]
//       //   console.log(normalizedLandmark)

//       //   // Loop through each array
//       //   for (let i = 0; i < normalizedLandmark.length; i++) {
//       //     console.log(normalizedLandmark[i])
//       //     // drawingUtils.drawConnectors(normalizedLandmark, normalizedLandmark[i], { color: '#ffffff' })
//       //   }
//       // }
//     }
//     requestAnimationFrame(renderHolistic)
//   })
// }

function onResults(results: Holistic.Results): void {
  inference.value = performance.now() - startTime
  canvasCtx.save()
  canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)

  if (results.segmentationMask) {
    canvasCtx.drawImage(
      results.segmentationMask, 0, 0, WIDTH,
      HEIGHT)

    canvasCtx.fillStyle = '#00FF007F'
    canvasCtx.fillRect(0, 0, WIDTH, HEIGHT)

    // Only overwrite missing pixels.
    canvasCtx.globalCompositeOperation = 'destination-atop'
    canvasCtx.drawImage(
      results.image, 0, 0, WIDTH, HEIGHT)

    canvasCtx.globalCompositeOperation = 'source-over'
  } else {
    canvasCtx.drawImage(
      results.image, 0, 0, WIDTH, HEIGHT)
  }

  if (debugMode.value) {
  // POSE
    drawConnectors(
      canvasCtx, results.poseLandmarks, Holistic.POSE_CONNECTIONS,
      { color: 'white' })
    drawLandmarks(
      canvasCtx,
      Object.values(Holistic.POSE_LANDMARKS_LEFT)
        .map(index => results.poseLandmarks[index]),
      { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)' })
    drawLandmarks(
      canvasCtx,
      Object.values(Holistic.POSE_LANDMARKS_RIGHT)
        .map(index => results.poseLandmarks[index]),
      { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)' })

    // HANDS
    drawConnectors(
      canvasCtx, results.rightHandLandmarks, Holistic.HAND_CONNECTIONS,
      { color: 'white' })
    drawLandmarks(canvasCtx, results.rightHandLandmarks, {
      color: 'white',
      fillColor: 'rgb(0,217,231)',
      lineWidth: 2,
      radius: (data: Data) => {
        return lerp(data.from!.z!, -0.15, 0.1, 10, 1)
      },
    })
    drawConnectors(
      canvasCtx, results.leftHandLandmarks, Holistic.HAND_CONNECTIONS,
      { color: 'white' })
    drawLandmarks(canvasCtx, results.leftHandLandmarks, {
      color: 'white',
      fillColor: 'rgb(255,138,0)',
      lineWidth: 2,
      radius: (data: Data) => {
        return lerp(data.from!.z!, -0.15, 0.1, 10, 1)
      },
    })

    // FACE
    drawConnectors(
      canvasCtx, results.faceLandmarks, Holistic.FACEMESH_TESSELATION,
      { color: '#C0C0C070', lineWidth: 1 })
    drawConnectors(
      canvasCtx, results.faceLandmarks, Holistic.FACEMESH_RIGHT_EYE,
      { color: 'rgb(0,217,231)' })
    drawConnectors(
      canvasCtx, results.faceLandmarks, Holistic.FACEMESH_RIGHT_EYEBROW,
      { color: 'rgb(0,217,231)' })
    drawConnectors(
      canvasCtx, results.faceLandmarks, Holistic.FACEMESH_LEFT_EYE,
      { color: 'rgb(255,138,0)' })
    drawConnectors(
      canvasCtx, results.faceLandmarks, Holistic.FACEMESH_LEFT_EYEBROW,
      { color: 'rgb(255,138,0)' })
    drawConnectors(
      canvasCtx, results.faceLandmarks, Holistic.FACEMESH_FACE_OVAL,
      { color: '#E0E0E0', lineWidth: 5 })
    drawConnectors(
      canvasCtx, results.faceLandmarks, Holistic.FACEMESH_LIPS,
      { color: '#E0E0E0', lineWidth: 5 })

    canvasCtx.restore()
  }
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
