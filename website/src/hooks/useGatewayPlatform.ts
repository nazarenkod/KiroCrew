import { useContext } from 'react'
import { QueryClient, QueryClientContext, useQuery, skipToken } from '@tanstack/react-query'
import type { KiroPrerequisiteStatus } from '../api/client'

/** What the gateway host is, for copy that names an OS feature by its real name. */
export type GatewayPlatform = 'darwin' | 'windows' | 'other'

/**
 * Classify a raw `process.platform`-shaped string into the three arms our copy has.
 *
 * Exported because not every reveal action runs on the gateway: Mochi's reveal is
 * an IPC send its Electron main process performs, so that surface reads the SHELL's
 * platform and must classify it by the same rule rather than a second one that could
 * disagree about what counts as Windows.
 *
 * Everything unrecognised collapses to `'other'`, deliberately. The gateway endpoint
 * reports the sentinel `'gateway'` to a non-owner dashboard user and to a probe that
 * could not run, an absent Electron bridge reports nothing at all, and Linux has no
 * single file manager to name — all three want the same generic wording, because
 * naming an application we are not sure exists is the failure mode worth designing
 * out.
 */
export function classifyPlatform(raw: string | undefined | null): GatewayPlatform {
  const platform = raw ?? ''
  if (platform === 'darwin') return 'darwin'
  // The gateway sends a human DISPLAY label, not `sys.platform`: the prerequisite
  // snapshot reports `"Windows"` (see `_platform_label` in kiro_prerequisite.py),
  // while Mochi's shell hands us the raw `process.platform` value `"win32"`.
  // Case-fold the raw value so both `"Windows"` and `"win32"` classify as Windows;
  // a lowercase-only `startsWith('win')` would collapse the display label to
  // `'other'` and mis-gate every Windows-only affordance.
  if (platform.toLowerCase().startsWith('win')) return 'windows'
  return 'other'
}

/**
 * Read-only stand-in for a tree that has no `QueryClientProvider`.
 *
 * Mochi's Electron windows and the popout frames mount with a bare `createRoot`,
 * and `useQuery` throws "No QueryClient set" there — which would turn a component
 * that merely wants to word a label correctly into one that cannot render at all.
 * This cache stays empty and the query below never fetches, so a caller in such a
 * tree resolves to `'other'`: the same generic wording an unreadable platform gets.
 */
const ORPHAN_TREE_CACHE = new QueryClient()

/**
 * The platform of the GATEWAY host, not of the browser.
 *
 * The browser's OS is the wrong signal for anything the gateway executes:
 * `/api/reveal` shells out on the gateway, so a dashboard opened from a Mac
 * against a Linux gateway must not name Finder. The install command has the same
 * property and is resolved server-side for the same reason.
 *
 * A pure reader — `skipToken` means this hook never fetches. The prerequisite
 * gate wraps the whole dashboard and owns that query, so the value is cached
 * before any page mounts, and this subscription re-renders when the gate
 * refreshes it.
 *
 * See `classifyPlatform` for why anything unrecognised is generic wording.
 */
export function useGatewayPlatform(): GatewayPlatform {
  const provided = useContext(QueryClientContext)
  const { data } = useQuery<KiroPrerequisiteStatus>(
    { queryKey: ['kiro-prerequisite'], queryFn: skipToken },
    provided ?? ORPHAN_TREE_CACHE,
  )
  return classifyPlatform(data?.platform)
}
