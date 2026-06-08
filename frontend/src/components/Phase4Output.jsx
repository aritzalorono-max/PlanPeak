import React from 'react'

/**
 * Phase 4 stub — DXF generation and 3D visualization output.
 * Will be fully implemented in Phase 4 (DXF) and Phase 5 (Three.js preview).
 */
export default function Phase4Output() {
  return (
    <div style={styles.container}>
      <h2 style={styles.heading}>Phase 4 — DXF Export & 3D Preview</h2>
      <div style={styles.stub}>
        <div style={styles.icon}>📐</div>
        <h3>Coming in Phases 4 & 5</h3>
        <p>
          <strong>Phase 4</strong> will generate a standards-compliant DXF file
          using ezdxf, with named layers for load-bearing walls, partition walls,
          doors, windows, and room annotations.
        </p>
        <p>
          <strong>Phase 5</strong> will render an interactive 3D preview of the
          extruded floor plan using Three.js directly in the browser.
        </p>
        <p style={styles.note}>
          Your floor plan has been processed through Phase 1. When the remaining
          phases are complete, your DXF file will be ready to download here.
        </p>
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
  note: {
    marginTop: '1.5rem',
    padding: '0.75rem 1rem',
    background: '#e8f5e9',
    borderRadius: 8,
    color: '#2e7d32',
    fontSize: '0.9rem',
  },
}
