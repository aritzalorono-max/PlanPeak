import React, { useState } from 'react'

const ROOM_COLORS = {
  kitchen: '#e65100',
  bathroom: '#1565c0',
  bedroom: '#6a1b9a',
  living_room: '#2e7d32',
  hallway: '#827717',
  dining_room: '#ad1457',
  unknown: '#546e7a',
}

function RoomBadge({ type, label }) {
  const color = ROOM_COLORS[type] || ROOM_COLORS.unknown
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '0.25rem 0.6rem',
        borderRadius: 20,
        background: color,
        color: '#fff',
        fontSize: '0.8rem',
        fontWeight: 600,
        margin: '0.2rem',
        textTransform: 'capitalize',
      }}
    >
      {label || type.replace('_', ' ')}
    </span>
  )
}

function JsonViewer({ data }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ marginTop: '1rem' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          background: 'none',
          border: '1px solid #ccc',
          borderRadius: 6,
          padding: '0.4rem 0.8rem',
          cursor: 'pointer',
          fontSize: '0.875rem',
        }}
      >
        {open ? '▼ Hide raw JSON' : '▶ Show raw JSON'}
      </button>
      {open && (
        <pre
          style={{
            background: '#1e1e1e',
            color: '#d4d4d4',
            padding: '1rem',
            borderRadius: 8,
            overflow: 'auto',
            maxHeight: 400,
            fontSize: '0.78rem',
            marginTop: '0.5rem',
          }}
        >
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}

export default function Phase1Output({ sessionData, onProceed }) {
  if (!sessionData?.phase1) {
    return (
      <div style={styles.container}>
        <p>No Phase 1 results yet. Please upload and process a floor plan first.</p>
      </div>
    )
  }

  const { cleanedImageB64, structuralImageB64, metadata, processingTimeMs } = sessionData.phase1
  const originalSrc = sessionData.selectedPageB64
    ? `data:image/png;base64,${sessionData.selectedPageB64}`
    : sessionData.previewB64
    ? `data:image/png;base64,${sessionData.previewB64}`
    : null
  const cleanedSrc = cleanedImageB64 ? `data:image/png;base64,${cleanedImageB64}` : null
  const structuralSrc = structuralImageB64 ? `data:image/png;base64,${structuralImageB64}` : null

  const rooms = metadata?.rooms || []
  const openings = metadata?.openings || []
  const scaleRefs = metadata?.scale_references || []
  const doors = openings.filter(o => o.type === 'door' || o.type === 'sliding_door')
  const windows = openings.filter(o => o.type === 'window')

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <h2 style={styles.heading}>Phase 1 — AI Semantic Cleaning</h2>
        <span style={styles.timing}>Processed in {(processingTimeMs / 1000).toFixed(1)}s</span>
      </div>

      {metadata?.error && (
        <div style={styles.warning}>
          Metadata extraction warning: {metadata.error}
        </div>
      )}

      {/* Three-way image view */}
      <div style={styles.imageRow}>
        <div style={styles.imagePane}>
          <p style={styles.imageLabel}>Original</p>
          {originalSrc ? (
            <img src={originalSrc} alt="Original floor plan" style={styles.image} />
          ) : (
            <div style={styles.imagePlaceholder}>No original image</div>
          )}
        </div>
        <div style={styles.imagePane}>
          <p style={styles.imageLabel}>Cleaned (OpenCV)</p>
          {cleanedSrc ? (
            <img src={cleanedSrc} alt="Cleaned floor plan" style={styles.image} />
          ) : (
            <div style={styles.imagePlaceholder}>Cleaned image unavailable</div>
          )}
        </div>
        <div style={styles.imagePane}>
          <p style={styles.imageLabel}>Structural</p>
          <div style={styles.imageLegend}>
            <span style={{color:'#000', fontWeight:700}}>■</span> Walls &nbsp;
            <span style={{color:'red', fontWeight:700}}>■</span> Doors &nbsp;
            <span style={{color:'blue', fontWeight:700}}>■</span> Windows
          </div>
          {structuralSrc ? (
            <img src={structuralSrc} alt="Structural floor plan" style={styles.image} />
          ) : (
            <div style={styles.imagePlaceholder}>
              {metadata?.walls?.length === 0 && metadata?.openings?.length === 0
                ? 'No walls or openings detected yet'
                : 'Structural render unavailable'}
            </div>
          )}
        </div>
      </div>

      {/* Metadata summary */}
      <div style={styles.summaryGrid}>
        {/* Rooms */}
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Rooms detected ({rooms.length})</h3>
          {rooms.length > 0 ? (
            rooms.map((r, i) => (
              <RoomBadge key={i} type={r.type} label={r.label} />
            ))
          ) : (
            <p style={styles.empty}>No rooms detected</p>
          )}
        </div>

        {/* Openings */}
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Openings detected ({openings.length})</h3>
          <p style={styles.stat}>
            <strong>{doors.length}</strong> door{doors.length !== 1 ? 's' : ''}
            {' / '}
            <strong>{windows.length}</strong> window{windows.length !== 1 ? 's' : ''}
          </p>
        </div>

        {/* Scale references */}
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Scale references ({scaleRefs.length})</h3>
          {scaleRefs.length > 0 ? (
            <ul style={styles.list}>
              {scaleRefs.map((s, i) => (
                <li key={i}>
                  {s.value} {s.unit}
                </li>
              ))}
            </ul>
          ) : (
            <p style={styles.empty}>None detected</p>
          )}
          {metadata?.estimated_scale && (
            <p style={styles.scale}>Estimated scale: <strong>{metadata.estimated_scale}</strong></p>
          )}
        </div>

        {/* Image dimensions */}
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Image info</h3>
          {metadata?.image_dimensions ? (
            <p>{metadata.image_dimensions.width} × {metadata.image_dimensions.height} px</p>
          ) : (
            <p style={styles.empty}>—</p>
          )}
          {metadata?.north_arrow?.detected && (
            <p>North arrow: detected</p>
          )}
        </div>
      </div>

      {/* Raw JSON viewer */}
      <JsonViewer data={metadata} />

      <button style={styles.proceedBtn} onClick={onProceed}>
        Proceed to Vectorization →
      </button>
    </div>
  )
}

const styles = {
  container: { maxWidth: 1100, margin: '0 auto' },
  headerRow: { display: 'flex', alignItems: 'baseline', gap: '1rem', marginBottom: '1rem' },
  heading: { margin: 0, fontSize: '1.5rem', fontWeight: 700 },
  timing: { color: '#666', fontSize: '0.875rem' },
  warning: {
    background: '#fff3e0', border: '1px solid #ff9800', borderRadius: 8,
    padding: '0.75rem 1rem', marginBottom: '1rem', color: '#e65100',
  },
  imageRow: { display: 'flex', gap: '1.5rem', marginBottom: '1.5rem', flexWrap: 'wrap' },
  imagePane: { flex: '1 1 300px', minWidth: 0 },
  imageLegend: { fontSize: '0.78rem', color: '#555', marginBottom: '0.4rem' },
  imageLabel: { fontWeight: 600, margin: '0 0 0.5rem' },
  image: { width: '100%', borderRadius: 8, border: '1px solid #ddd', display: 'block' },
  imagePlaceholder: {
    height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: '#f5f5f5', border: '1px dashed #ccc', borderRadius: 8, color: '#888',
    textAlign: 'center', padding: '1rem',
  },
  summaryGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1rem' },
  card: { background: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: '1rem' },
  cardTitle: { margin: '0 0 0.75rem', fontSize: '1rem', fontWeight: 700 },
  empty: { color: '#999', fontSize: '0.875rem', margin: 0 },
  stat: { margin: 0, fontSize: '0.95rem' },
  list: { margin: '0 0 0.5rem', paddingLeft: '1.25rem', fontSize: '0.875rem' },
  scale: { margin: 0, fontSize: '0.875rem' },
  proceedBtn: {
    marginTop: '1.5rem',
    padding: '0.85rem 2rem',
    background: '#1e3a5f',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
  },
}
