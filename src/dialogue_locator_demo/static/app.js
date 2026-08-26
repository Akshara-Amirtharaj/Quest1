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

const errorStates = {
  VIDEO_URL_REQUIRED: ['Video URL required', 'Enter a video URL to continue.'],
  INVALID_VIDEO_URL: ['Invalid video URL', 'Enter a valid public video URL.'],
  DIALOGUE_REQUIRED: ['Dialogue required', 'Enter the dialogue you want to find.'],
  VIDEO_UNAVAILABLE: ['Video unavailable', "We couldn't access this video. Check the URL and make sure the video is publicly available."],
  DIALOGUE_NOT_FOUND: ['Dialogue not found', "We couldn't find this dialogue in the video. Try a slightly different phrase."],
  PROCESSING_FAILED: ['Processing failed', 'Something went wrong while processing the video. Please try again.'],
}

const errorAliases = {
  NO_MATCH: 'DIALOGUE_NOT_FOUND',
  MEDIA_UNAVAILABLE: 'VIDEO_UNAVAILABLE',
  NO_AUDIO: 'VIDEO_UNAVAILABLE',
  NO_VIDEO: 'VIDEO_UNAVAILABLE',
  INVALID_INPUT: 'INVALID_VIDEO_URL',
  PIPELINE_ERROR: 'PROCESSING_FAILED',
  STORAGE_ERROR: 'PROCESSING_FAILED',
  INTERNAL_ERROR: 'PROCESSING_FAILED',
}

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
  if (!videoUrl.value.trim()) return 'VIDEO_URL_REQUIRED'
  try {
    const parsed = new URL(videoUrl.value.trim())
    if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('invalid protocol')
  } catch {
    return 'INVALID_VIDEO_URL'
  }
  if (!dialogue.value.trim()) return 'DIALOGUE_REQUIRED'
  return null
}

function clearFieldError(input, feedback) {
  input.removeAttribute('aria-invalid')
  feedback.hidden = true
  feedback.textContent = ''
}

function clearError() {
  errorBox.hidden = true
  errorTitle.textContent = ''
  errorMessage.textContent = ''
}

function showError(code, { inline = false } = {}) {
  const publicCode = errorAliases[code] || code
  const [title, message] = errorStates[publicCode] || errorStates.PROCESSING_FAILED
  errorTitle.textContent = title
  errorMessage.textContent = message
  errorBox.hidden = false

  if (!inline) return
  if (publicCode === 'VIDEO_URL_REQUIRED' || publicCode === 'INVALID_VIDEO_URL') {
    const feedback = document.querySelector('#video-url-error')
    videoUrl.setAttribute('aria-invalid', 'true')
    feedback.textContent = message
    feedback.hidden = false
  } else if (publicCode === 'DIALOGUE_REQUIRED') {
    const feedback = document.querySelector('#dialogue-error')
    dialogue.setAttribute('aria-invalid', 'true')
    feedback.textContent = message
    feedback.hidden = false
  }
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
  clearFieldError(dialogue, document.querySelector('#dialogue-error'))
  clearError()
})

videoUrl.addEventListener('input', () => {
  clearFieldError(videoUrl, document.querySelector('#video-url-error'))
  clearError()
})

form.addEventListener('submit', async event => {
  event.preventDefault()
  if (requestInFlight) return
  const validationError = validate()
  if (validationError) {
    showError(validationError, { inline: true })
    return
  }

  clearError()
  clearFieldError(videoUrl, document.querySelector('#video-url-error'))
  clearFieldError(dialogue, document.querySelector('#dialogue-error'))
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
      const error = new Error('Request failed')
      error.code = payload.detail?.code
      throw error
    }
    showResult(payload)
  } catch (requestError) {
    showError(requestError instanceof TypeError ? 'PROCESSING_FAILED' : requestError?.code)
  } finally {
    stopProgress()
    submitButton.disabled = false
    requestInFlight = false
    submitButton.replaceChildren('Find frame ', Object.assign(document.createElement('span'), { textContent: '→' }))
  }
})
