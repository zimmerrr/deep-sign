<template>
  <div
    :class="{ 'hidden': !inferenceTime }"
    class="inference absolute-top-right q-mr-md"
    style="z-index: 2;"
  >
    <div>
      Inference: {{ inferenceTime.toFixed(2) }}ms
    </div>
    <div>
      DeepSign: {{ deepSignTime.toFixed(2) }}ms
    </div>
  </div>
  <div
    :class="{ 'hidden': inferenceTime }"
    class="text-subtitle2 absolute-top-right q-mr-md"
    style="z-index: 2;"
  >
    Model loading, please wait
    <q-spinner-dots
      color="white"
      size="lg"
    />
  </div>
  <div class="container q-mx-auto">
    <div class="video-container">
      <q-responsive :ratio="WIDTH / HEIGHT">
        <canvas
          ref="canvasRef"
          class="canvas absolute-full"
          style="z-index: 1; object-fit: cover;"
        />
        <video
          ref="videoElementRef"
          autoplay
          muted
          playsinline
          class="absolute-full"
          style="object-fit: cover;"
        />
      </q-responsive>
    </div>
    <div
      v-if="predictions.length > 0"
      class="text-uppercase"
    >
      <div class="predictions">
        {{ predictions.map((el) => el[0].label).join(' ') }}
      </div>
      <div
        v-for="(gesture, idx) in predictions[predictions.length - 1]"
        :key="idx"
        class="text-left"
      >
        <div class="gesture-label">
          {{ gesture.label }}:
        </div>
        <div class>
          <q-linear-progress
            stripe
            rounded
            size="20px"
            :value="gesture.probability"
            color="white"
            class="q-mt-sm"
          />
        </div>
      </div>
    </div>
    <div v-else>
      <div class="text-uppercase gesture-label">
        Start signing to begin
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, onMounted, onBeforeMount, withCtx, Ref } from 'vue'
import { useRoute } from 'vue-router'
import * as mpHolistic from '@mediapipe/holistic'
import { drawConnectors, drawLandmarks, lerp, Data } from '@mediapipe/drawing_utils'
import { waitVideoMetadata } from 'src/components/utils'
import { extractKeypointsV3 } from './utils/keypoints'
import { DeepSignV6, DeepSignV2, loadModelV6, topk, loadModelV2, loadModelV3, DeepSignV3 } from './utils/model'

const WIDTH = 640
const HEIGHT = 360
const NUM_FRAMES_TO_IDLE = 5
const TARGET_FPS = 20
const INPUT_SEQ_LEN = 60

const props = defineProps({
  settings: { type: Object, default: null },
})

const devices = ref<MediaDeviceInfo[]>([])
const error = ref('')
const running = ref(true)

const videoElementRef = ref<HTMLVideoElement>(null as any)
const canvasRef = ref<HTMLCanvasElement>(null as any)

const route = useRoute()
const debugMode = !!route.query.debug
const complexity = route.query.complexity ? parseInt('0' || route.query.complexity as string) : 0
let startTime = performance.now()
const inferenceTime = ref(0)
const deepSignTime = ref(0)

let canvasCtx: CanvasRenderingContext2D
let holistic: mpHolistic.Holistic
let deepsign: DeepSignV6 | DeepSignV3 | DeepSignV2

const predictions: Ref<any[]> = ref([])

const config: mpHolistic.HolisticConfig = {
  locateFile: (file: any) => {
    console.log('/holistic@' +
      `${mpHolistic.VERSION}/${file}`)
    return '/holistic@' +
      `${mpHolistic.VERSION}/${file}`
  },
}

const complexityMapping: Record<string, number> = {
  Lite: 0,
  Full: 1,
  Heavy: 2,
}

// Function to initialize camera
async function initialize() {
  if (props.settings.modelVersion === 'DeepSign v3') {
    deepsign = await loadModelV3()
  } else if (props.settings.modelVersion === 'DeepSign v2') {
    deepsign = await loadModelV2()
  } else {
    deepsign = await loadModelV6()
  }

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
    holistic = new mpHolistic.Holistic(config)
    holistic.setOptions({
      minDetectionConfidence: 0.65,
      minTrackingConfidence: 0.5,
      modelComplexity: getMappedComplexity(props.settings.modelComplexity) as any,
    })
    holistic.onResults(onResults)
    requestAnimationFrame(render)
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
      width: WIDTH,
      height: HEIGHT,
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
  requestAnimationFrame(render)
}

// async function useModel() {
//   try {
//     const session = await ort.InferenceSession.create('./model.onnx')

//     // prepare inputs. a tensor need its corresponding TypedArray as data
//     const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
//     const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
//     const tensorA = new ort.Tensor('float32', dataA, [3, 4])
//     const tensorB = new ort.Tensor('float32', dataB, [4, 3])

//     // prepare feeds. use model input names as keys.
//     const feeds = { a: tensorA, b: tensorB }

//     // feed inputs and run
//     const results = await session.run(feeds)

//     // read from results
//     const dataC = results.c.data
//     document.write(`data of result tensor 'c': ${dataC}`)
//   } catch (e) {
//     document.write(`failed to inference ONNX model: ${e}.`)
//   }
// }

const sequence: Record<string, any>[] = []
const sentence: string[] = []
let recording = false
let numFramesNoHand = 0
let lastFrameTime = 0

function pushSequence(results: mpHolistic.Results) {
  sequence.push(extractKeypointsV3(results))
  lastFrameTime = performance.now()
}

async function onResults(results: mpHolistic.Results): Promise<void> {
  const hasHands = Boolean(results.leftHandLandmarks || results.rightHandLandmarks)
  if (!recording && hasHands) {
    recording = true
    sequence.splice(0, sequence.length)
    numFramesNoHand = 0
    pushSequence(results)
  } else if (recording && !hasHands && numFramesNoHand > NUM_FRAMES_TO_IDLE) {
    recording = false

    if (sequence.length) {
      const deepSignStartTime = performance.now()
      const result = await deepsign.runInference(sequence)
      const preds = topk(deepsign, result, 5)
      predictions.value.push(preds)
      if (predictions.value.length > 5) {
        predictions.value.shift()
      }
      deepSignTime.value = performance.now() - deepSignStartTime
    }
  } else if (recording) {
    console.log('recording')
    if (!hasHands) numFramesNoHand++

    const keypoints = extractKeypointsV3(results)
    sequence.push(keypoints)
    if (sequence.length > INPUT_SEQ_LEN) {
      sequence.shift()
    }
  }

  inferenceTime.value = performance.now() - startTime
  canvasCtx.save()
  canvasCtx.clearRect(0, 0, WIDTH, HEIGHT)

  if (debugMode) {
    // POSE
    drawConnectors(
      canvasCtx, results.poseLandmarks, mpHolistic.POSE_CONNECTIONS,
      { color: 'white' })
    drawLandmarks(
      canvasCtx,
      Object.values(mpHolistic.POSE_LANDMARKS_LEFT)
        .map(index => results.poseLandmarks[index]),
      { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)' })
    drawLandmarks(
      canvasCtx,
      Object.values(mpHolistic.POSE_LANDMARKS_RIGHT)
        .map(index => results.poseLandmarks[index]),
      { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)' })

    // HANDS
    drawConnectors(
      canvasCtx, results.rightHandLandmarks, mpHolistic.HAND_CONNECTIONS,
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
      canvasCtx, results.leftHandLandmarks, mpHolistic.HAND_CONNECTIONS,
      { color: 'white' })
    drawLandmarks(canvasCtx, results.leftHandLandmarks, {
      color: 'white',
      fillColor: 'rgb(255,138,0)',
      lineWidth: 2,
      radius: (data: Data) => {
        return lerp(data.from!.z!, -0.15, 0.1, 10, 1)
      },
    })

    canvasCtx.restore()
  }
}

function getMappedComplexity(complexity: string) {
  const mappedValue = complexityMapping[complexity as any]
  console.log(mappedValue)
  if (mappedValue === undefined) {
    return 0
  }

  return mappedValue
}

onMounted(async () => {
  initialize()
  // runModel()
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

.predictions
  font-size: 60%

.inference
  font-size: 40%
  line-height: 1.25
.gesture-label
  font-size: 60%
</style>
