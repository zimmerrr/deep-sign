import { Tensor } from 'onnxruntime-web'
import { Landmark, Results } from '@mediapipe/holistic'

interface Keypoint {
  x: number
  y: number
  z: number
  visibility?: number
}

function flattenKeypoints(keypoints: any[]) {
  return keypoints.flatMap(el => {
    return 'visibility' in el
      ? [el.x, el.y, el.z, el.visibility]
      : [el.x, el.y, el.z]
  })
}

function getAngles(data: Keypoint, interleave: number, parentIndices: any, childIndices: any, vecA: any, vecB: any) {
  //convert python code that uses numpy to ts

  // return angles

function getHandDirections(data: Keypoint, interleave: number, parentIndices: any, childIndices: any, vecA: any, vecB: any) {
  //convert python code that uses numpy to ts

  // return direction.flatten
}

function getPoseAngles(data: any) {
  let parent: any = [
    [0, 1, 2, 3, 0, 4, 5, 6], // nose, eyes
    [9], // mouth
    [11, 13, 15, 15, 15, 11, 23, 25, 27, 27], // left body
    [12, 14, 16, 16, 16, 12, 24, 26, 28, 28], // right body
    [11, 23], // body
    // [17, 18],  // fingers
    // [29, 30],  // foot
  ]
  let child: any = [
    [1, 2, 3, 7, 4, 5, 6, 8], // nose, eyes
    [10], // mouth
    [13, 15, 17, 19, 21, 23, 25, 27, 29, 31], // left body
    [14, 16, 18, 20, 22, 24, 26, 28, 30, 32], // right body
    [12, 24], // body
    // [19, 20],  // fingers
    // [31, 32],  // foot
  ]
  let vecA: any = [
    [0, 1, 2, 4, 5, 6], // nose, eyes
    [9, 10, 10, 10, 14, 15, 16, 16, 9], // left body
    [19, 20, 20, 20, 24, 25, 26, 26, 19], // right body
    [29, 29, 30, 30], // body
  ]
  let vecB: any = [
    [1, 2, 3, 5, 6, 7], // nose, eyes
    [10, 11, 12, 13, 15, 16, 17, 18, 14], // left body
    [20, 21, 22, 23, 25, 26, 27, 28, 24], // right body
    [14, 24, 15, 25], // body
  ]

  // Flatten the list
  parent = [...parent.flat()]
  child = [...child.flat()]
  vecA = [...vecA.flat()]
  vecB = [...vecB.flat()]

  return getAngles(data, 4, parent, child, vecA, vecB)
}

function getHandAngles(data: any) {
  const parent = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
  const child = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
  const vecA = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
  const vecB = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

  return getAngles(data, 3, parent, child, vecA, vecB)
}

function preprocessPoseLandmark(keypoints: Keypoint[]) {
  const angles = getPoseAngles(keypoints)
  return flattenKeypoints(keypoints).concat(angles)
}

function preprocessHandLandmark(keypoints: Keypoint[]) {
  const angles = getHandAngles(keypoints)
  const directions = getHandDirections(keypoints)
  return flattenKeypoints(keypoints).concat(angles).concat(directions)
}

function preprocessLandmarks(results: { poseLandmarks: Keypoint[]; leftHand: Keypoint[]; rightHand: Keypoint[] }) {
  let output: any[] = []

  if (results.poseLandmarks) {
    output = output.concat(preprocessPoseLandmark(results.poseLandmarks))
  } else {
    output = output.concat(Array(33 * 4 + 28).fill(0))
  }

  if (results.leftHand) {
    output = output.concat(preprocessHandLandmark(results.leftHand))
  } else {
    output = output.concat(Array(21 * 3 + 15 + 6 * 3).fill(0))
  }

  if (results.rightHand) {
    output = output.concat(preprocessHandLandmark(results.rightHand))
  } else {
    output = output.concat(Array(21 * 3 + 15 + 6 * 3).fill(0))
  }
}

export function KeypointsToTensor(keypoints: any): Tensor {
  const landmark = preprocessLandmarks(keypoints)

  const inputTensor = new Tensor('float32', keypoints)

  return inputTensor
}
