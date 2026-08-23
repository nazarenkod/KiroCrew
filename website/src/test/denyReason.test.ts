import { describe, it, expect } from 'vitest'
import { extractDenyReason } from '../utils/denyReason'

// The Output panel of a blocked tool call used to show a fixed
// "blocked by security policy" line and discard the row's real content, so the
// user could never see WHICH rule fired. These pin the extraction that replaces
// that placeholder, and the cases where the placeholder must still win.
describe('extractDenyReason', () => {
  const ROW =
    '🚫 Running: python3 -c "import x" — Blocked by security policy: ' +
    'kiro[-.]?crew\\b[^|;&#>/*]*\\btoken\\b\n' +
    "Matched structurally on the command's argv, not by the pattern text above."

  it('returns the reason starting at the contract marker', () => {
    expect(extractDenyReason(ROW)).toMatch(/^Blocked by security policy:/)
  })

  it('drops the row title so the panel shows the reason, not the command', () => {
    const out = extractDenyReason(ROW)
    expect(out).not.toContain('🚫')
    expect(out).not.toContain('Running:')
  })

  it('keeps the pattern that fired', () => {
    expect(extractDenyReason(ROW)).toContain('\\btoken\\b')
  })

  it('keeps the second explanation line', () => {
    // The structural note is the part that makes a floor hit intelligible; a
    // single-line extraction would silently drop exactly that.
    expect(extractDenyReason(ROW)).toContain('Matched structurally')
  })

  it('yields empty for a row with no reason, so the placeholder wins', () => {
    expect(extractDenyReason('🚫 shell (hook blocked)')).toBe('')
    expect(extractDenyReason('🚫 shell')).toBe('')
  })

  it('reads the host reason, not a model-authored title that mimics it', () => {
    // `<title>` prefers the tool call's own `description` field, so it is
    // model-authored. First-match extraction would render the model's own text
    // to the user AS the security reason; the host always appends the real one
    // after the title, so the LAST marker is the trustworthy one.
    const spoofed =
      '🚫 Running: Blocked by security policy: totally fine, ignore this' +
      ' — Blocked by security policy: real-deny-rule'
    const out = extractDenyReason(spoofed)
    expect(out).toBe('Blocked by security policy: real-deny-rule')
    expect(out).not.toContain('totally fine')
  })

  it('still reads a spoof-shaped title when the host reason carries a note', () => {
    const spoofed =
      '🚫 Running: Blocked by security policy: spoof — Blocked by security policy: rule\n' +
      'Matched structurally on the argv.'
    const out = extractDenyReason(spoofed)
    expect(out.startsWith('Blocked by security policy: rule')).toBe(true)
    expect(out).toContain('Matched structurally')
    expect(out).not.toContain('spoof')
  })

  it('yields empty for a bare marker rather than rendering a lone colon', () => {
    expect(extractDenyReason('🚫 shell — Blocked by security policy:')).toBe('')
  })

  it('handles an absent row', () => {
    expect(extractDenyReason('')).toBe('')
  })
})
