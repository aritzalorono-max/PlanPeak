import React, { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import Phase1Output from './components/Phase1Output'
import Phase2Output from './components/Phase2Output'
import Phase3Editor from './components/Phase3Editor'
import Phase4Output from './components/Phase4Output'

const PHASE_LABELS = {
  1: 'Upload & AI Cleaning',
  2: 'Vectorization',
  3: 'Calibration Editor',
  4: 'DXF Export',
  5: '3D Preview',
}

export default function App() {
  const [currentPhase, setCurrentPhase] = useState(1)
  const [sessionData, setSessionData] = useState(null)
  // sessionData shape:
  // {
  //   sessionId: string,
  //   fileType: 'pdf' | 'image',
  //   pageCount: number,
  //   previewB64: string,
  //   selectedPageB64: string | null,
  //   phase1: { cleanedImageB64: string | null, metadata: object, processingTimeMs: number } | null,
  // }

  function handleUploadComplete(data) {
    setSessionData({
      sessionId: data.session_id,
      fileType: data.file_type,
      pageCount: data.page_count,
      previewB64: data.preview_page_b64,
      selectedPageB64: data.file_type === 'image' ? data.preview_page_b64 : null,
      phase1: null,
    })
  }

  function handlePageSelected(pageB64) {
    setSessionData(prev => ({ ...prev, selectedPageB64: pageB64 }))
  }

  function handlePhase1Complete(result) {
    setSessionData(prev => ({
      ...prev,
      phase1: {
        cleanedImageB64: result.cleaned_image_b64,
        metadata: result.metadata,
        processingTimeMs: result.processing_time_ms,
      },
    }))
    setCurrentPhase(2)
  }

  function goToPhase(n) {
    setCurrentPhase(n)
  }

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>PlanPeak</h1>
        <p style={styles.subtitle}>Floor Plan → DXF Converter</p>
      </header>

      {/* Phase stepper */}
      <nav style={styles.stepper}>
        {Object.entries(PHASE_LABELS).map(([phase, label]) => {
          const p = Number(phase)
          const active = p === currentPhase
          const done = p < currentPhase
          return (
            <div
              key={phase}
              style={{
                ...styles.step,
                ...(active ? styles.stepActive : {}),
                ...(done ? styles.stepDone : {}),
              }}
              onClick={() => done && goToPhase(p)}
              title={done ? `Go back to ${label}` : undefined}
            >
              <span style={styles.stepNum}>{done ? '✓' : phase}</span>
              <span style={styles.stepLabel}>{label}</span>
            </div>
          )
        })}
      </nav>

      <main style={styles.main}>
        {currentPhase === 1 && (
          <UploadPanel
            sessionData={sessionData}
            onUploadComplete={handleUploadComplete}
            onPageSelected={handlePageSelected}
            onPhase1Complete={handlePhase1Complete}
          />
        )}
        {currentPhase === 2 && (
          <Phase1Output
            sessionData={sessionData}
            onProceed={() => goToPhase(3)}
          />
        )}
        {currentPhase === 3 && (
          <Phase2Output onProceed={() => goToPhase(4)} />
        )}
        {currentPhase === 4 && (
          <Phase3Editor onProceed={() => goToPhase(5)} />
        )}
        {currentPhase === 5 && (
          <Phase4Output />
        )}
      </main>
    </div>
  )
}

const styles = {
  app: {
    fontFamily: "'Inter', 'Segoe UI', sans-serif",
    minHeight: '100vh',
    background: '#f5f5f5',
    color: '#1a1a1a',
  },
  header: {
    background: '#1e3a5f',
    color: '#fff',
    padding: '1.5rem 2rem',
    display: 'flex',
    alignItems: 'baseline',
    gap: '1rem',
  },
  title: {
    margin: 0,
    fontSize: '2rem',
    fontWeight: 700,
    letterSpacing: '-0.5px',
  },
  subtitle: {
    margin: 0,
    fontSize: '1rem',
    opacity: 0.75,
  },
  stepper: {
    display: 'flex',
    background: '#fff',
    borderBottom: '1px solid #e0e0e0',
    padding: '0 2rem',
    overflowX: 'auto',
  },
  step: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '1rem 1.25rem',
    color: '#888',
    fontSize: '0.875rem',
    borderBottom: '3px solid transparent',
    whiteSpace: 'nowrap',
  },
  stepActive: {
    color: '#1e3a5f',
    fontWeight: 600,
    borderBottomColor: '#1e3a5f',
  },
  stepDone: {
    color: '#2e7d32',
    cursor: 'pointer',
  },
  stepNum: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '1.5rem',
    height: '1.5rem',
    borderRadius: '50%',
    background: 'currentColor',
    color: '#fff',
    fontSize: '0.75rem',
    fontWeight: 700,
    flexShrink: 0,
  },
  stepLabel: {
    color: 'inherit',
  },
  main: {
    maxWidth: '1200px',
    margin: '2rem auto',
    padding: '0 2rem',
  },
}
