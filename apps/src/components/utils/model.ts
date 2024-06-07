/* eslint-disable camelcase */
import * as ort from 'onnxruntime-web'
import _ from 'lodash'
import { extractKeypointsV1, extractKeypointsV3, flattenKeypoints } from './keypoints'
import { Results } from '@mediapipe/holistic'

export interface DeepSignV6Metadata {
  labels: string[]
  input_size: number,
  hidden_feature_size: number[],
  hidden_label_size: number[],
}

export interface DeepSignV3Metadata {
  labels: string[]
  input_size: number,
  hn_size: number[],
}

export interface DeepSignV2Metadata {
  labels: string[]
  input_size: number,
  hn_size: number[],
  cn_size: number[],
}

export interface DeepSignV6 {
  session: ort.InferenceSession
  metadata: DeepSignV6Metadata
  extractKeypoints(results: Results): Record<string, any>
  runInference(keypoints: Record<string, any>[]): Promise<Float32Array>
}

export interface DeepSignV3 {
  session: ort.InferenceSession
  metadata: DeepSignV3Metadata
  extractKeypoints(results: Results): Record<string, any>
  runInference(keypoints: Record<string, any>[]): Promise<Float32Array>
}

export interface DeepSignV2 {
  session: ort.InferenceSession
  metadata: DeepSignV2Metadata
  extractKeypoints(results: Results): Record<string, any>
  runInference(keypoints: Record<string, any>[]): Promise<Float32Array>
}

export async function loadModelV6(): Promise<DeepSignV6> {
  const session = await ort.InferenceSession.create('/model/earnest-cloud-155/deepsign.onnx', {
    // executionProviders: ['webgl'],
    graphOptimizationLevel: 'all',
  })
  const metadataResp = await fetch('/model/earnest-cloud-155/metadata.json')
  const metadata = await metadataResp.json()

  function keypointsToTensor(keypoints: any[]): ort.Tensor {
    const inputData = keypoints.map(keypoint => [
      ...keypoint.pose ? flattenKeypoints(keypoint.pose) : zeroArray(33 * 4),
      ...keypoint.poseAngles ? keypoint.poseAngles : zeroArray(28),
      ...keypoint.lh ? flattenKeypoints(keypoint.lh) : zeroArray(21 * 3),
      ...keypoint.lhAngles ? keypoint.lhAngles : zeroArray(15),
      ...keypoint.lhDir ? flattenKeypoints(keypoint.lhDir) : zeroArray(6 * 3),
      ...keypoint.rh ? flattenKeypoints(keypoint.rh) : zeroArray(21 * 3),
      ...keypoint.rhAngles ? keypoint.rhAngles : zeroArray(15),
      ...keypoint.rhDir ? flattenKeypoints(keypoint.rhDir) : zeroArray(6 * 3),
    ])
    return new ort.Tensor(
      'float32',
      Float32Array.from(inputData.flat()),
      [1, keypoints.length, inputData[0].length],
    )
  }

  async function runInference(keypoints: Record<string, any>[]) {
    const input = keypointsToTensor(keypoints)
    const hidden_features = new ort.Tensor(
      'float32',
      new Float32Array(dimToSize(metadata.hidden_feature_size)),
      metadata.hidden_feature_size,
    )
    const hidden_label = new ort.Tensor(
      'float32',
      new Float32Array(dimToSize(metadata.hidden_label_size)),
      metadata.hidden_label_size,
    )
    const result = await session.run({ input, hidden_features, hidden_label })
    return softmax(getFloat32Array(result.output))
  }

  return {
    session,
    metadata,
    runInference,
    extractKeypoints: extractKeypointsV3,
  }
}

export async function loadModelV3(): Promise<DeepSignV3> {
  const session = await ort.InferenceSession.create('/model/summer-thunder-101/deepsign.onnx', {
    graphOptimizationLevel: 'all',
  })
  const metadataResp = await fetch('/model/summer-thunder-101/metadata.json')
  const metadata = await metadataResp.json() as DeepSignV3Metadata

  function keypointsToTensor(keypoints: any[]): ort.Tensor {
    const inputData = keypoints.map(keypoint => [
      ...keypoint.pose ? flattenKeypoints(keypoint.pose) : zeroArray(33 * 4),
      ...keypoint.poseAngles ? keypoint.poseAngles : zeroArray(28),
      ...keypoint.lh ? flattenKeypoints(keypoint.lh) : zeroArray(21 * 3),
      ...keypoint.lhAngles ? keypoint.lhAngles : zeroArray(15),
      ...keypoint.rh ? flattenKeypoints(keypoint.rh) : zeroArray(21 * 3),
      ...keypoint.rhAngles ? keypoint.rhAngles : zeroArray(15),
    ])
    return new ort.Tensor(
      'float32',
      Float32Array.from(inputData.flat()),
      [1, keypoints.length, inputData[0].length],
    )
  }

  async function runInference(keypoints: Record<string, any>[]) {
    const input = keypointsToTensor(keypoints)
    const hn = new ort.Tensor(
      'float32',
      new Float32Array(dimToSize(metadata.hn_size)),
      metadata.hn_size,
    )
    const result = await session.run({ input, hn })
    return softmax(getFloat32Array(result.output))
  }

  return {
    session,
    metadata,
    runInference,
    extractKeypoints: extractKeypointsV3,
  }
}

export async function loadModelV2(): Promise<DeepSignV2> {
  const session = await ort.InferenceSession.create('/model/azure-sound-58/deepsign.onnx', {
    graphOptimizationLevel: 'all',
  })
  const metadataResp = await fetch('/model/azure-sound-58/metadata.json')
  const metadata = await metadataResp.json() as DeepSignV2Metadata

  function keypointsToTensor(keypoints: any[]): ort.Tensor {
    const inputData = keypoints.map(keypoint => [
      ...keypoint.pose ? flattenKeypoints(keypoint.pose) : zeroArray(33 * 4),
      ...keypoint.face ? flattenKeypoints(keypoint.face) : zeroArray(468 * 3),
      ...keypoint.lh ? flattenKeypoints(keypoint.lh) : zeroArray(21 * 3),
      ...keypoint.rh ? flattenKeypoints(keypoint.rh) : zeroArray(21 * 3),
    ])
    return new ort.Tensor(
      'float32',
      Float32Array.from(inputData.flat()),
      [1, keypoints.length, inputData[0].length],
    )
  }

  async function runInference(keypoints: Record<string, any>[]) {
    const input = keypointsToTensor(keypoints)
    const hn = new ort.Tensor(
      'float32',
      new Float32Array(dimToSize(metadata.hn_size)),
      metadata.hn_size,
    )
    const cn = new ort.Tensor(
      'float32',
      new Float32Array(dimToSize(metadata.cn_size)),
      metadata.cn_size,
    )
    const result = await session.run({ input, hn, cn })
    return softmax(getFloat32Array(result.output))
  }

  return {
    session,
    metadata,
    runInference,
    extractKeypoints: extractKeypointsV1,
  }
}

function zeroArray(length: number): number[] {
  return new Array(length).fill(0)
}

function dimToSize(dim: number[]) {
  return dim.reduce((acc, val) => acc * val, 1)
}

function getFloat32Array(tensor: ort.Tensor): Float32Array {
  if (tensor.type !== 'float32') {
    throw new Error('Expected tensor to be of type float32')
  }
  return tensor.data as Float32Array
}

function softmax(probabilities: Float32Array) {
  const largestNumber = Math.max(...probabilities)
  const sumOfExp = probabilities
    .map((resultItem) => Math.exp(resultItem - largestNumber))
    .reduce((prevNumber, currentNumber) => prevNumber + currentNumber)

  return probabilities.map((resultValue, index) => {
    return Math.exp(resultValue - largestNumber) / sumOfExp
  })
}

export function topk(deepsign: DeepSignV6 | DeepSignV3 | DeepSignV2, probabilities: Float32Array, k = 1) {
  const _probabilities = Array.from(probabilities)
  const sorted = _.reverse(_.sortBy(_probabilities.map((prob, index) => [prob, index]), (probIndex) => probIndex[0]))
  const topk = _.take(sorted, k).map((probIndex) => {
    return {
      index: probIndex[1],
      probability: probIndex[0],
      label: deepsign.metadata.labels[probIndex[1]],
    }
  })

  return topk
}
