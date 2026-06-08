import React, { useRef, useState } from 'react'
import axios from 'axios'

export default function UploadPanel({
  sessionData,
  onUploadComplete,
  onPageSelected,
  onPhase1Complete,
}) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [selectedPage, setSelectedPage] = useState(1)
  const fileInputRef = useRef(null)

  async function handleFile(file) {
    setError(null)
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const { data } = await axios.post('/api/upload', formData)
      onUploadComplete(data)
      setSelectedPage(1)
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  function onInputChange(e) {
    const file = e.target.files[0]
    if (file) handleFile(file)
  }

  async function handleSelectPage(pageNum) {
    setSelectedPage(pageNum)
    setError(null)
    try {
      const { data } = await axios.post('/api/upload/select-page', {
        session_id: sessionData.sessionId,
        page_number: pageNum,
      })
      onPageSelected(data.page_image_b64)
    } catch (err) {
      setError(err.response?.data?.detail || 'Page selection failed.')
    }
  }

  async function handleProcess() {
    if (!sessionData?.sessionId) return
    setError(null)
    setProcessing(true)
    try {
      const { data } = await axios.post('/api/phase1/process', {
        session_id: sessionData.sessionId,
      })
      onPhase1Complete(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'AI processing failed.')
    } finally {
      setProcessing(false)
    }
  }

  const hasImage = !!(sessionData?.selectedPageB64 || sessionData?.previewB64)
  const previewSrc = sessionData?.selectedPageB64
    ? `data:image/png;base64,${sessionData.selectedPageB64}`
    : sessionData?.previewB64
    ? `data:image/png;base64,${sessionData.previewB64}`
    : null

  return (
    <div style={styles.container}>
      <h2 style={styles.heading}>Upload Floor Plan</h2>

      {/* Drop zone */}
      <div
        style={{
          ...styles.dropzone,
          ...(dragging ? styles.dropzoneDragging : {}),
          ...(hasImage ? styles.dropzoneSmall : {}),
        }}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.pdf"
          style={{ display: 'none' }}
          onChange={onInputChange}
        />
        {uploading ? (
          <p style={styles.hint}>Uploading…</p>
        ) : hasImage ? (
          <p style={styles.hint}>Click or drop to replace</p>
        ) : (
          <>
            <div style={styles.uploadIcon}>⬆</div>
            <p style={styles.hint}>Drag & drop a PNG, JPG, or PDF</p>
            <p style={styles.hintSmall}>or click to browse</p>
          </>
        )}
      </div>

      {error && <p style={styles.error}>{error}</p>}

      {/* PDF page selector */}
      {sessionData?.fileType === 'pdf' && sessionData.pageCount > 1 && (
        <div style={styles.pageSelector}>
          <p style={styles.pageSelectorLabel}>
            Select page ({sessionData.pageCount} pages found):
          </p>
          <div style={styles.pageList}>
            {Array.from({ length: sessionData.pageCount }, (_, i) => i + 1).map(p => (
              <button
                key={p}
                style={{
                  ...styles.pageBtn,
                  ...(p === selectedPage ? styles.pageBtnActive : {}),
                }}
                onClick={() => handleSelectPage(p)}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Preview + process */}
      {previewSrc && (
        <div style={styles.previewSection}>
          <img src={previewSrc} alt="Floor plan preview" style={styles.preview} />

          <button
            style={{
              ...styles.processBtn,
              ...(processing ? styles.processBtnDisabled : {}),
            }}
            disabled={processing}
            onClick={handleProcess}
          >
            {processing ? (
              <span>
                <span style={styles.spinner}>⏳</span> Gemini is analyzing your floor plan…
              </span>
            ) : (
              'Process with AI →'
            )}
          </button>
        </div>
      )}
    </div>
  )
}

const styles = {
  container: { maxWidth: 900, margin: '0 auto' },
  heading: { marginTop: 0, fontSize: '1.5rem', fontWeight: 700 },
  dropzone: {
    border: '2px dashed #aaa',
    borderRadius: 12,
    padding: '3rem 2rem',
    textAlign: 'center',
    cursor: 'pointer',
    background: '#fafafa',
    transition: 'all 0.2s',
  },
  dropzoneDragging: { borderColor: '#1e3a5f', background: '#e8f0fe' },
  dropzoneSmall: { padding: '1rem 2rem' },
  uploadIcon: { fontSize: '2.5rem', marginBottom: '0.5rem' },
  hint: { margin: '0.25rem 0', color: '#555', fontSize: '1rem' },
  hintSmall: { margin: '0.25rem 0', color: '#999', fontSize: '0.85rem' },
  error: { color: '#c62828', background: '#ffebee', padding: '0.75rem 1rem', borderRadius: 8, marginTop: '1rem' },
  pageSelector: { marginTop: '1.5rem' },
  pageSelectorLabel: { fontWeight: 600, marginBottom: '0.5rem' },
  pageList: { display: 'flex', flexWrap: 'wrap', gap: '0.5rem' },
  pageBtn: {
    padding: '0.4rem 0.8rem',
    border: '1px solid #ccc',
    borderRadius: 6,
    cursor: 'pointer',
    background: '#fff',
    fontSize: '0.875rem',
  },
  pageBtnActive: { background: '#1e3a5f', color: '#fff', borderColor: '#1e3a5f' },
  previewSection: { marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' },
  preview: { maxWidth: '100%', maxHeight: 480, objectFit: 'contain', borderRadius: 8, border: '1px solid #ddd' },
  processBtn: {
    padding: '0.85rem 2rem',
    background: '#1e3a5f',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
    alignSelf: 'flex-start',
    transition: 'opacity 0.2s',
  },
  processBtnDisabled: { opacity: 0.65, cursor: 'not-allowed' },
  spinner: { marginRight: '0.5rem' },
}
