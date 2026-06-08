import React from 'react'

/**
 * Phase 2 stub — OpenCV/Shapely vectorization output.
 * Will be fully implemented in Phase 2.
 */
export default function Phase2Output({ onProceed }) {
  return (
    <div style={styles.container}>
      <h2 style={styles.heading}>Phase 2 — Vectorization</h2>
      <div style={styles.stub}>
        <div style={styles.icon}>🔲</div>
        <h3>Coming in Phase 2</h3>
        <p>
          OpenCV contour detection and Shapely geometry building will be
          implemented here. The cleaned image from Phase 1 will be converted
          into vector primitives (walls, doors, windows) ready for DXF export.
        </p>
        <button style={styles.btn} onClick={onProceed}>
          Skip to Calibration →
        </button>
      </div>
    </div>
  )
}

const styles = {
  container: { maxWidth: 800, margin: '0 auto' },
  heading: { fontSize: '1.5rem', fontWeight: 700, marginTop: 0 },
  stub: {
    background: '#fff',
    border: '1px solid #e0e0e0',
    borderRadius: 12,
    padding: '3rem 2rem',
    textAlign: 'center',
    color: '#555',
  },
  icon: { fontSize: '3rem', marginBottom: '0.5rem' },
  btn: {
    marginTop: '1.5rem',
    padding: '0.75rem 1.75rem',
    background: '#546e7a',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
  },
}
