const form = document.querySelector('#find-form')
const videoUrl = document.querySelector('#video-url')
const dialogue = document.querySelector('#dialogue')
const dialogueCount = document.querySelector('#dialogue-count')
const submitButton = document.querySelector('#submit-button')
const processing = document.querySelector('#processing')
const processingTitle = document.querySelector('#processing-title')
const errorBox = document.querySelector('#error')
const errorTitle = document.querySelector('#error-title')
const errorMessage = document.querySelector('#error-message')
const resultBox = document.querySelector('#result')

let elapsedTimer
let requestInFlight = false

function titleCase(value) {
  return String(value).replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase())
}

function formatElapsed(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return minutes ? `${minutes}m ${String(seconds).padStart(2, '0')}s` : `${seconds}s`
}

function setText(selector, value) {
  document.querySelector(selector).textContent = value
}

function validate() {
  if (!videoUrl.value.trim()) return 'Enter a video URL.'
  try {
    const parsed = new URL(videoUrl.value.trim())
    if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('invalid protocol')
  } catch {
    return 'Enter a valid public HTTP or HTTPS video URL.'
  }
  if (!dialogue.value.trim()) return 'Enter the dialogue you want to find.'
  return null
}

function startProgress() {
  const started = performance.now()
  processing.hidden = false
  processingTitle.textContent = 'Processing the video · 0s elapsed'
  elapsedTimer = window.setInterval(() => {
    const seconds = Math.floor((performance.now() - started) / 1000)
    processingTitle.textContent = `Processing the video · ${formatElapsed(seconds)} elapsed`
  }, 1000)
}

function stopProgress() {
  window.clearInterval(elapsedTimer)
  processing.hidden = true
}

function showResult(result) {
  const confidence = result.match.confidence.toLowerCase()
  setText('#result-start', result.match.start_hms)
  setText('#result-end', result.match.end_hms)
  setText('#result-confidence', `${result.match.confidence} confidence`)
  setText('#confidence-summary', result.match.confidence_reason)
  document.querySelector('#result-confidence').className = `confidence confidence-${confidence}`
  const frame = document.querySelector('#result-frame')
  frame.src = result.frame.url
  frame.alt = `Extracted video frame at ${result.frame.timestamp_hms}`
  setText('#result-caption', `Frame ${result.frame.index.toLocaleString()} at ${result.frame.timestamp_hms} · ${result.evidence.frame_match_type.replaceAll('_', ' ')}`)
  setText('#result-dialogue', `“${result.match.text}”`)
  setText('#result-score', result.match.score.toFixed(1))
  setText('#result-localized', titleCase(result.evidence.localization_source))
  setText('#result-verified', titleCase(result.evidence.verification_source))
  setText('#result-elapsed', `${result.processing.elapsed_seconds.toFixed(1)}s`)
  setText('#detail-interval', `${result.match.start_hms} – ${result.match.end_hms}`)
  setText('#detail-match-type', titleCase(result.match.type))
  setText('#detail-occurrences', result.match.occurrence_count)
  setText('#detail-pts', `${result.frame.pts} (${result.frame.time_base})`)
  resultBox.hidden = false
}

dialogue.addEventListener('input', () => {
  dialogueCount.textContent = `${dialogue.value.length} / 2000`
})

form.addEventListener('submit', async event => {
  event.preventDefault()
  if (requestInFlight) return
  const validationError = validate()
  if (validationError) {
    errorMessage.textContent = validationError
    errorBox.hidden = false
    return
  }

  errorBox.hidden = true
  resultBox.hidden = true
  submitButton.disabled = true
  requestInFlight = true
  submitButton.textContent = 'Finding frame…'
  startProgress()
  try {
    const response = await fetch('/api/find', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_url: videoUrl.value.trim(), dialogue: dialogue.value.trim() }),
    })
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) {
      const error = new Error(payload.detail?.message || 'The request could not be completed.')
      error.code = payload.detail?.code
      throw error
    }
    showResult(payload)
  } catch (requestError) {
    if (requestError instanceof TypeError) {
      errorTitle.textContent = 'Backend connection interrupted'
      errorMessage.textContent = 'The backend connection was interrupted. Please retry.'
    } else {
      errorTitle.textContent = requestError?.code === 'NO_MATCH' ? 'Couldn’t find that frame' : 'Couldn’t complete the request'
      errorMessage.textContent = requestError instanceof Error ? requestError.message : 'The request could not be completed.'
    }
    errorBox.hidden = false
  } finally {
    stopProgress()
    submitButton.disabled = false
    requestInFlight = false
    submitButton.replaceChildren('Find frame ', Object.assign(document.createElement('span'), { textContent: '→' }))
  }
})
