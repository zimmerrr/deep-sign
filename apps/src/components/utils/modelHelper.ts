import * as ort from 'onnxruntime-web'
import _ from 'lodash'
import { InputMap } from '@mediapipe/holistic'
import labels from 'src/components/model/labels'

// CREATE SESSION
export async function rundDeepSignModel(preprocessedData: InputMap): Promise<[any, number]> {
  const session = await ort.InferenceSession
    .create('./_next/static/chunks/pages/deepsign.onnx',
      { executionProviders: ['webgl'], graphOptimizationLevel: 'all' })
  console.log('Inference session created')

  const [results, inferenceTime] = await runInference(session, preprocessedData)
  return [results, inferenceTime] as any

  async function runInference(session: ort.InferenceSession, preprocessedData: InputMap): Promise<[any, number]> {
    const start = new Date()
    const feeds: Record<string, ort.Tensor> = {}
    feeds[session.inputNames[0]] = preprocessedData

    const outputData = await session.run(feeds)
    const end = new Date()
    const inferenceTime = (end.getTime() - start.getTime()) / 1000
    const output = outputData[session.outputNames[0]]
    const outputSoftmax = softmax(Array.prototype.slice.call(output.data))

    const results = imagenetClassesTopK(outputSoftmax, 5)
    console.log('results: ', results)
    return [results, inferenceTime]
  }

  function softmax(resultArray: number[]): any {
    const largestNumber = Math.max(...resultArray)
    const sumOfExp = resultArray.map((resultItem) => Math.exp(resultItem - largestNumber)).reduce((prevNumber, currentNumber) => prevNumber + currentNumber)
    return resultArray.map((resultValue, index) => {
      return Math.exp(resultValue - largestNumber) / sumOfExp
    })
  }
}

// FIND THE HIGHEST PREDITION ON CLASS
export function imagenetClassesTopK(classProbabilities: any, k = 5) {
  const probs =
      _.isTypedArray(classProbabilities) ? Array.prototype.slice.call(classProbabilities) : classProbabilities

  const sorted = _.reverse(_.sortBy(probs.map((prob: any, index: number) => [prob, index]), (probIndex: Array<number>) => probIndex[0]))

  const topK = _.take(sorted, k).map((probIndex: Array<number>) => {
    const iClass = labels[probIndex[1]]
    return {
      id: iClass[0],
      index: parseInt(probIndex[1].toString(), 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0],
    }
  })
  return topK
}
