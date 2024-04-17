import * as tf from '@tensorflow/tfjs'

let sequence: any[][] = []
let sentence = []
let prevWord = ''
const actions = ['Hello', 'Thanks', 'Goodbye', 'Please', 'Yes', 'No']

export const onResults = (
  results: { poseLandmarks: any; faceLandmarks: any; leftHandLandmarks: any; rightHandLandmarks: any },
  model: { predict: (arg0: any) => any } | null,
  speechSynthesisUtterance: SpeechSynthesisUtterance,
  textAreaRef: { current: { innerText: string } },
) => {
  if (model !== null) {
    try {
      let pose = new Array(33 * 4).fill(0),
        face = new Array(468 * 3).fill(0),
        lh = new Array(21 * 3).fill(0),
        rh = new Array(21 * 3).fill(0)
      console.log('getting frame')
      if (results.poseLandmarks) {
        const arr = []
        for (const res of results.poseLandmarks) {
          arr.push(...[res.x, res.y, res.z, res.visibility])
        }
        pose = arr
      }
      if (results.faceLandmarks) {
        const arr = []
        for (const res of results.faceLandmarks) {
          arr.push(...[res.x, res.y, res.z])
        }
        face = arr
      }
      if (results.leftHandLandmarks) {
        const arr = []
        for (const res of results.leftHandLandmarks) {
          arr.push(...[res.x, res.y, res.z])
        }
        lh = arr
      }
      if (results.rightHandLandmarks) {
        const arr = []
        for (const res of results.rightHandLandmarks) {
          arr.push(...[res.x, res.y, res.z])
        }
        rh = arr
      }
      sequence.push([...pose, ...face, ...lh, ...rh])
      if (sequence.length === 20) {
        const newTensor = tf.tensor2d(sequence)
        const modelResult = model.predict(tf.expandDims(newTensor, 0))
        modelResult.array().then((res: any[]) => {
          const prediction = actions[res[0].indexOf(Math.max(...res[0]))]
          if (prediction !== prevWord) {
            speechSynthesisUtterance.text = prediction
            window.speechSynthesis.speak(speechSynthesisUtterance)
            textAreaRef.current.innerText = prediction
            sentence.push(prediction)
          }
          prevWord = prediction
        })
        sequence = []
      }
    } catch (err) {
      sequence = []
      console.log(err)
    }
  }
}

export const resetSentence = () => (sentence = [])
