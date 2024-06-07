<template>
  asd
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import testFrame from './TestFrame'
import _ from 'lodash'
import { NormalizedLandmark } from '@mediapipe/holistic'
import {
  normalizeKeypoints,
  normalizeKeypoint,
  getAngles,
  getDirections,
  extractKeypointsV3,
} from 'components/utils/keypoints'

function isClose(a: number, b: number, epsilon = 0.0001) {
  return Math.abs(a - b) < epsilon
}

function isKeypointClose(
  a: NormalizedLandmark,
  b: NormalizedLandmark,
  epsilon = 0.0001) {
  if (a.visibility !== undefined && b.visibility !== undefined) {
    return (
      isClose(a.x, b.x, epsilon) &&
      isClose(a.y, b.y, epsilon) &&
      isClose(a.z, b.z, epsilon) &&
      isClose(a.visibility, b.visibility, epsilon)
    )
  } else {
    return (
      isClose(a.x, b.x, epsilon) &&
      isClose(a.y, b.y, epsilon) &&
      isClose(a.z, b.z, epsilon)
    )
  }
}

function testNormalizeKeypoints() {
  const input = [
    { x: 1, y: 3, z: 9, visibility: 13 },
    { x: 27, y: 51, z: 34, visibility: 21 },
  ]
  const expected = [
    { x: -13, y: -24, z: -12.5, visibility: -4 },
    { x: 13, y: 24, z: 12.5, visibility: 4 },
  ]
  const result = normalizeKeypoints(input)
  if (!_.isEqual(result.keypoints, expected)) {
    console.log('normalizeKeypoints', { result: result.keypoints, expected })
    throw new Error('normalizeKeypoints is not as expected')
  }
}
function testNormalizeKeypoint() {
  const input = [
    { x: 1, y: 3, z: 9, visibility: 13 },
    { x: 27, y: 51, z: 34, visibility: 21 },
  ]
  const expected = [
    { x: 0.06201737, y: 0.18605210, z: 0.55815630, visibility: 0.80622580 },
    { x: 0.38465598, y: 0.72657245, z: 0.48438162, visibility: 0.29917687 },
  ]

  const result = input.map(normalizeKeypoint)

  if (result.some((r, i) => !isKeypointClose(r, expected[i]))) {
    console.log('normalizeKeypoint', { result, expected })
    throw new Error('normalizeKeypoint is not as expected')
  }
}

function testGetAngles() {
  const input = [
    { x: 1, y: 3, z: 9 },
    { x: 6, y: 7, z: 3 },
    { x: 4, y: 9, z: 6 },
  ]
  const expected = [123.558624]
  const result = getAngles(input, [0, 1], [1, 2], [0], [1])
  if (result.some((r, i) => !isClose(r, expected[i]))) {
    console.log('getAngles', { result, expected })
    throw new Error('getAngles is not as expected')
  }
}

function testGetDirections() {
  const input = [
    { x: 1, y: 3, z: 9 },
    { x: 6, y: 7, z: 3 },
    { x: 4, y: 9, z: 6 },
  ]
  const expected = [
    { x: 0.663348, y: -0.08291849, z: 0.497511 },
  ]
  const result = getDirections(input, [0, 1], [1, 2], [0], [1])
  if (result.some((r, i) => !isKeypointClose(r, expected[i]))) {
    console.log('getDirections', { result, expected })
    throw new Error('getDirections is not as expected')
  }
}

export default defineComponent({
  setup() {
    testNormalizeKeypoints()
    testNormalizeKeypoint()
    testGetAngles()
    testGetDirections()

    const result = extractKeypointsV3(testFrame as any)
    console.log(result)
    return {}
  },
})
</script>

<style lang="sass" scoped>

</style>
