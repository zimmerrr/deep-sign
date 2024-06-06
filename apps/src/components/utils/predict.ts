// Language: typescript
// Path: react-next\utils\predict.ts
import { KeypointsToTensor } from './videoHelper'
import { rundDeepSignModel } from './modelHelper'

export async function inferenceDeepSign(keypoints: any): Promise<[any, number]> {
  // CONVERT KEYPOINTS TO TENSOR
  const tensors = await KeypointsToTensor(keypoints)

  // RUN MODEL
  const [predictions, inferenceTime] = await rundDeepSignModel(tensors)

  // RETURN PREDICTION AND INFERENCE
  return [predictions, inferenceTime]
}
