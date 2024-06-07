import { Landmark, NormalizedLandmark, NormalizedLandmarkList, Results } from '@mediapipe/holistic'
import * as math from 'mathjs'

export function flattenKeypoints(keypoints: NormalizedLandmarkList) {
  return keypoints.flatMap(el => {
    return el.visibility !== undefined
      ? [el.x, el.y, el.z, el.visibility]
      : [el.x, el.y, el.z]
  })
}

export function normalizeKeypoint(keypoint: NormalizedLandmark) {
  if (keypoint.visibility !== undefined) {
    const norm = math.norm([
      keypoint.x,
      keypoint.y,
      keypoint.z,
      keypoint.visibility,
    ]) as number
    return {
      x: keypoint.x / norm,
      y: keypoint.y / norm,
      z: keypoint.z / norm,
      visibility: keypoint.visibility / norm,
    }
  } else {
    const norm = math.norm([keypoint.x, keypoint.y, keypoint.z]) as number
    return {
      x: keypoint.x / norm,
      y: keypoint.y / norm,
      z: keypoint.z / norm,
    }
  }
}

export function normalizeKeypoints(keypoints: NormalizedLandmarkList) {
  const mean = keypoints.reduce(
    (acc, el) => {
      acc.x += el.x
      acc.y += el.y
      acc.z += el.z
      if (el.visibility !== undefined && acc.visibility !== undefined) {
        acc.visibility += el.visibility
      }
      return acc
    },
    {
      x: 0,
      y: 0,
      z: 0,
      visibility: keypoints[0].visibility !== undefined ? 0 : undefined,
    },
  )

  mean.x /= keypoints.length
  mean.y /= keypoints.length
  mean.z /= keypoints.length
  if (mean.visibility !== undefined) {
    mean.visibility /= keypoints.length
  }

  keypoints = keypoints.map(el => {
    return {
      x: el.x - mean.x,
      y: el.y - mean.y,
      z: el.z - mean.z,
      visibility: el.visibility !== undefined && mean.visibility !== undefined
        ? el.visibility - mean.visibility : undefined,
    }
  })

  return { keypoints, mean }
}

export function getAngles(
  data: NormalizedLandmarkList,
  parentIndices: number[],
  childIndices: number[],
  vecA: number[],
  vecB: number[],
) {
  const v1 = parentIndices.map(idx => data[idx])
  const v2 = childIndices.map(idx => data[idx])
  const v = v2
    .map((el, idx) => {
      return {
        x: el.x - v1[idx].x,
        y: el.y - v1[idx].y,
        z: el.z - v1[idx].z,
      }
    })
    .map(normalizeKeypoint)

  const angles = vecA.map((el, idx) => {
    const a = v[vecA[idx]]
    const b = v[vecB[idx]]
    const dot = math.dot([a.x, a.y, a.z], [b.x, b.y, b.z])
    const angleRad = math.acos(Math.max(-1, Math.min(1, dot))) as number
    return angleRad * (180 / Math.PI)
  })

  return angles
}

export function getDirections(
  data: NormalizedLandmarkList,
  parentIndices: number[],
  childIndices: number[],
  vecA: number[],
  vecB: number[],
) {
  const v1 = parentIndices.map(idx => data[idx])
  const v2 = childIndices.map(idx => data[idx])
  const v = v2
    .map((el, idx) => {
      return {
        x: el.x - v1[idx].x,
        y: el.y - v1[idx].y,
        z: el.z - v1[idx].z,
      }
    })
    .map(normalizeKeypoint)

  const directions = vecA.map((el, idx) => {
    const a = v[vecA[idx]]
    const b = v[vecB[idx]]
    const [x, y, z] = math.cross([a.x, a.y, a.z], [b.x, b.y, b.z]) as number[]
    return { x, y, z }
  })

  return directions
}

export function getPoseAngles(data: NormalizedLandmarkList) {
  const parent = [
    [0, 1, 2, 3, 0, 4, 5, 6], // nose, eyes
    [9], // mouth
    [11, 13, 15, 15, 15, 11, 23, 25, 27, 27], // left body
    [12, 14, 16, 16, 16, 12, 24, 26, 28, 28], // right body
    [11, 23], // body
    // [17, 18],  // fingers
    // [29, 30],  // foot
  ].flat()
  const child = [
    [1, 2, 3, 7, 4, 5, 6, 8], // nose, eyes
    [10], // mouth
    [13, 15, 17, 19, 21, 23, 25, 27, 29, 31], // left body
    [14, 16, 18, 20, 22, 24, 26, 28, 30, 32], // right body
    [12, 24], // body
    // [19, 20],  // fingers
    // [31, 32],  // foot
  ].flat()
  const vecA = [
    [0, 1, 2, 4, 5, 6], // nose, eyes
    [9, 10, 10, 10, 14, 15, 16, 16, 9], // left body
    [19, 20, 20, 20, 24, 25, 26, 26, 19], // right body
    [29, 29, 30, 30], // body
  ].flat()
  const vecB = [
    [1, 2, 3, 5, 6, 7], // nose, eyes
    [10, 11, 12, 13, 15, 16, 17, 18, 14], // left body
    [20, 21, 22, 23, 25, 26, 27, 28, 24], // right body
    [14, 24, 15, 25], // body
  ].flat()

  return getAngles(data, parent, child, vecA, vecB)
}

export function getHandAngles(data: NormalizedLandmarkList) {
  const parent = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
  const child = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
  const vecA = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
  const vecB = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
  return getAngles(data, parent, child, vecA, vecB)
}

export function getHandDirections(data: NormalizedLandmarkList) {
  const parent = [0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20]
  const child = [4, 20, 0, 2, 5, 9, 9, 13, 13, 17, 17, 13]
  const vecA = [0, 1, 3, 5, 7, 9]
  const vecB = [1, 2, 4, 6, 8, 10]
  return getDirections(data, parent, child, vecA, vecB)
}

export function preprocessPoseLandmark(inputKeypoints: NormalizedLandmarkList) {
  const { keypoints, mean } = normalizeKeypoints(inputKeypoints)
  const angles = getPoseAngles(keypoints)
  return { keypoints, mean, angles }
}

export function preprocessHandLandmark(inputKeypoints: NormalizedLandmarkList) {
  const { keypoints, mean } = normalizeKeypoints(inputKeypoints)
  const angles = getHandAngles(keypoints)
  const directions = getHandDirections(keypoints)
  return { keypoints, mean, angles, directions }
}

export function extractKeypointsV3(results: Results) {
  const output: Record<string, any> = {}

  if (results.poseLandmarks) {
    const { keypoints, mean, angles } = preprocessPoseLandmark(results.poseLandmarks)
    output.pose = keypoints
    output.poseMean = mean
    output.poseAngles = angles
  }

  if (results.leftHandLandmarks) {
    const { keypoints, mean, angles, directions } = preprocessHandLandmark(results.leftHandLandmarks)
    output.lh = keypoints
    output.lhMean = mean
    output.lhAngles = angles
    output.lhDir = directions
  }

  if (results.rightHandLandmarks) {
    const { keypoints, mean, angles, directions } = preprocessHandLandmark(results.rightHandLandmarks)
    output.rh = keypoints
    output.rhMean = mean
    output.rhAngles = angles
    output.rhDir = directions
  }

  return output
}
