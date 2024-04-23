export function waitVideoMetadata(video: HTMLVideoElement) {
  return new Promise((resolve, reject) => {
    video.addEventListener('loadedmetadata', resolve)
    video.addEventListener('error', reject)
  })
}
