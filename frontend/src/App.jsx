import React from 'react'
import UploadForm from './UploadForm'
import ModelMetrics from './ModelMetrics'

export default function App() {
  const [showMetrics, setShowMetrics] = React.useState(false)

  return (
    <div className="container">
      <header className="page-header">
        <div className="brand-mark" aria-hidden="true">♥</div>
        <div>
          <p className="eyebrow">Clinical screening workspace</p>
          <h1>Autonomous Pediatric Cardiac Screening</h1>
          <p className="muted">Upload a WAV and/or ultrasound / X-ray images. AI will analyze all available modalities.</p>
        </div>
      </header>
      {/* PHASE 1 (50%): Currently showing only audio + ultrasound models. Phase 2 will add X-ray + fusion. */}
      
      <div className="toolbar">
        <button
          className="metrics-toggle"
          onClick={() => setShowMetrics(!showMetrics)}
        >
          {showMetrics ? '👉 Hide' : '📊 Show'} Model Performance Proof
        </button>
      </div>

      {showMetrics && <ModelMetrics />}
      
      <hr />
      
      <UploadForm />
    </div>
  )
}
