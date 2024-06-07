/* eslint-disable camelcase */
import * as ort from 'onnxruntime-web'
import _ from 'lodash'
import { flattenKeypoints } from './keypoints'

export interface DeepSignMetadata {
  labels: string[]
  input_size: number,
  hidden_feature_size: number[],
  hidden_label_size: number[],
}

export interface DeepSign {
  session: ort.InferenceSession
  metadata: DeepSignMetadata
}

export async function loadModel() {
  const session = await ort.InferenceSession.create('/model/deepsign.onnx', {
    // executionProviders: ['webgl'],
    graphOptimizationLevel: 'all',
  })
  const metadataResp = await fetch('/model/metadata.json')
  const metadata = await metadataResp.json()
  return { session, metadata }
}

function zeroArray(length: number): number[] {
  return new Array(length).fill(0)
}

export function keypointsToTensor(keypoints: any[]): ort.Tensor {
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

function dimToSize(dim: number[]) {
  return dim.reduce((acc, val) => acc * val, 1)
}

export async function runInference(deepsign: DeepSign, keypoints: Record<string, any>[]) {
  const input = keypointsToTensor(keypoints)
  const hidden_features = new ort.Tensor(
    'float32',
    new Float32Array(dimToSize(deepsign.metadata.hidden_feature_size)),
    deepsign.metadata.hidden_feature_size,
  )
  const hidden_label = new ort.Tensor(
    'float32',
    new Float32Array(dimToSize(deepsign.metadata.hidden_label_size)),
    deepsign.metadata.hidden_label_size,
  )
  const result = await deepsign.session.run({ input, hidden_features, hidden_label })
  return softmax(getFloat32Array(result.output))
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

export function topk(deepsign: DeepSign, probabilities: Float32Array, k = 1) {
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
