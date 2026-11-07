/**
 * NEO mission control — the Desktop half of the `neo` unified package.
 *
 * One ESM file, loaded uncompiled by Hermes Desktop. It imports the plugin SDK,
 * React and the JSX runtime only, and renders with `jsx()` calls because JSX
 * syntax cannot parse here.
 *
 * What it contributes:
 *   - a `/neo` page (mission control) and its sidebar row,
 *   - a right status-bar chip carrying the active-task count,
 *   - a palette command that opens the page,
 *   - a tool-media renderer for NEO's own tools, so a `MEDIA:` line they
 *     produce is drawn by NEO instead of the generic player.
 *
 * What it never does: touch app DOM, dispatch synthetic events, call `fetch`,
 * or hardcode a colour. Every backend read and write goes through `ctx.rest`
 * against `/api/plugins/neo`; every timer and listener is owned by a hook with
 * a disposer or by `ctx` itself.
 *
 * The state vocabulary is the seven user-facing buckets from the approved
 * requirements, mapped from NEO's richer internal states. Unknown internal
 * states land in `blocked`, never in `completed`: the UI never claims success
 * it cannot evidence.
 */

import * as sdk from '@hermes/plugin-sdk'
import { useCallback, useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

// <pure-helpers>
/*
 * The pure region is delimited so the static contract suite can import and
 * exercise it in Node without resolving the SDK. Keep it free of imports,
 * React, and host access.
 */

export const STATE_ORDER = [
  'queued',
  'working',
  'needs-you',
  'blocked',
  'reviewing',
  'completed',
  'failed'
]

export const STATE_LABELS = {
  queued: 'Queued',
  working: 'Working',
  'needs-you': 'Needs you',
  blocked: 'Blocked',
  reviewing: 'Reviewing',
  completed: 'Completed',
  failed: 'Failed'
}

/** Calm tone names shared by StatusDot and PanelPill. */
export const STATE_TONES = {
  queued: 'muted',
  working: 'good',
  'needs-you': 'warn',
  blocked: 'warn',
  reviewing: 'muted',
  completed: 'good',
  failed: 'bad'
}

const RAW_STATE_LABELS = {
  queued: 'Queued',
  starting: 'Starting',
  running: 'Running',
  waiting: 'Waiting on an external condition',
  needs_approval: 'Needs approval',
  needs_input: 'Needs input',
  cancelling: 'Cancelling',
  unknown: 'Unknown — no current evidence',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled'
}

const RAW_TO_USER_STATE = {
  queued: 'queued',
  starting: 'working',
  running: 'working',
  waiting: 'blocked',
  needs_approval: 'needs-you',
  needs_input: 'needs-you',
  cancelling: 'working',
  unknown: 'blocked',
  completed: 'completed',
  failed: 'failed',
  cancelled: 'failed'
}

/** Map one internal state onto the seven user-facing buckets. */
export function normalizeState(rawState) {
  if (typeof rawState !== 'string') return 'blocked'
  const key = rawState.trim().toLowerCase()
  return RAW_TO_USER_STATE[key] || 'blocked'
}

/**
 * A task's user-facing bucket. `waiting` with a recorded finish is NEO
 * verifying the executor's result — shown as `reviewing`; any other wait is
 * `blocked` because something outside NEO must change first.
 */
export function userStateForTask(task) {
  const rawState = task && typeof task === 'object' ? task.state : task
  if (
    rawState === 'waiting' &&
    task &&
    typeof task === 'object' &&
    task.finished_at !== null &&
    task.finished_at !== undefined
  ) {
    return 'reviewing'
  }
  return normalizeState(rawState)
}

export function rawStateLabel(rawState) {
  if (typeof rawState !== 'string') return RAW_STATE_LABELS.unknown
  const key = rawState.trim().toLowerCase()
  return RAW_STATE_LABELS[key] || `Unrecognised state: ${rawState.trim() || 'empty'}`
}

export function isTerminalTask(task) {
  const bucket = userStateForTask(task)
  return bucket === 'completed' || bucket === 'failed'
}

export function activeTaskCount(tasks) {
  if (!Array.isArray(tasks)) return 0
  return tasks.filter(task => task && typeof task === 'object' && !isTerminalTask(task)).length
}

function newestFirst(left, right) {
  const leftTime = Number(left && left.created_at)
  const rightTime = Number(right && right.created_at)
  const leftValue = Number.isFinite(leftTime) ? leftTime : 0
  const rightValue = Number.isFinite(rightTime) ? rightTime : 0
  return rightValue - leftValue
}

/** Group tasks into the seven buckets, newest first inside each bucket. */
export function groupTasks(tasks) {
  const groups = STATE_ORDER.map(state => ({
    state,
    label: STATE_LABELS[state],
    tasks: []
  }))
  const byState = new Map(groups.map(group => [group.state, group]))
  const rows = Array.isArray(tasks) ? tasks : []
  for (const task of rows) {
    if (!task || typeof task !== 'object') continue
    const group = byState.get(userStateForTask(task))
    if (group) group.tasks.push(task)
  }
  for (const group of groups) group.tasks.sort(newestFirst)
  return groups
}

export function kindsOf(tasks) {
  const seen = new Set()
  const rows = Array.isArray(tasks) ? tasks : []
  for (const task of rows) {
    if (task && typeof task === 'object' && typeof task.kind === 'string' && task.kind.trim()) {
      seen.add(task.kind.trim())
    }
  }
  return Array.from(seen).sort()
}

export function filterTasks(tasks, filters) {
  const rows = Array.isArray(tasks) ? tasks : []
  const state = filters && typeof filters.state === 'string' ? filters.state : 'all'
  const kind = filters && typeof filters.kind === 'string' ? filters.kind : 'all'
  return rows.filter(task => {
    if (!task || typeof task !== 'object') return false
    if (state !== 'all' && userStateForTask(task) !== state) return false
    if (kind !== 'all' && task.kind !== kind) return false
    return true
  })
}

const PREVIEW_MIME_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif']
const BASE64_RE = /^[A-Za-z0-9+/=\s]+$/
const TASK_ID_RE = /^[A-Za-z0-9_-]{1,64}$/

/** Build a data URI from an available preview; refuse anything else. */
export function buildPreviewUri(preview) {
  if (!preview || typeof preview !== 'object') return null
  if (preview.available !== true) return null
  const mime = typeof preview.mime === 'string' ? preview.mime.trim().toLowerCase() : ''
  if (PREVIEW_MIME_TYPES.indexOf(mime) === -1) return null
  const payload = typeof preview.data_base64 === 'string' ? preview.data_base64.trim() : ''
  if (!payload || !BASE64_RE.test(payload)) return null
  return `data:${mime};base64,${payload.replace(/\s+/g, '')}`
}

/** Build a data URI from one artifact's returned bytes. */
export function buildArtifactUri(artifact) {
  if (!artifact || typeof artifact !== 'object') return null
  const payload = typeof artifact.data_base64 === 'string' ? artifact.data_base64.trim() : ''
  if (!payload || !BASE64_RE.test(payload)) return null
  const rawMime = typeof artifact.mime === 'string' ? artifact.mime.trim() : ''
  const mime = /^[A-Za-z0-9.+-]+\/[A-Za-z0-9.+-]+$/.test(rawMime) ? rawMime : 'application/octet-stream'
  return `data:${mime};base64,${payload.replace(/\s+/g, '')}`
}

export function isImageMime(mime) {
  return typeof mime === 'string' && mime.trim().toLowerCase().indexOf('image/') === 0
}

/** An honest placeholder for a preview that cannot be drawn. */
export function previewPlaceholder(preview) {
  const reason =
    preview && typeof preview === 'object' && typeof preview.reason === 'string'
      ? preview.reason.trim()
      : ''
  if (reason) return `Preview unavailable — ${reason}.`
  return 'Preview unavailable — no revision has been published for this task.'
}

/** True when two preview payloads would paint the same frame. */
export function samePreview(left, right) {
  if (left === right) return true
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false
  return (
    left.available === right.available &&
    left.rev === right.rev &&
    left.sha256 === right.sha256 &&
    left.reason === right.reason
  )
}

/** Extract `?task=<id>` from a hash route, refusing anything NEO cannot issue. */
export function taskIdFromRoute(route) {
  if (typeof route !== 'string') return null
  const queryStart = route.indexOf('?')
  if (queryStart === -1) return null
  const query = route.slice(queryStart + 1)
  if (!query) return null
  for (const pair of query.split('&')) {
    const separator = pair.indexOf('=')
    if (separator === -1) continue
    if (pair.slice(0, separator) !== 'task') continue
    const rawValue = pair.slice(separator + 1)
    if (!rawValue) return null
    let value = rawValue
    try {
      value = decodeURIComponent(rawValue.replace(/\+/g, ' '))
    } catch (error) {
      return null
    }
    return TASK_ID_RE.test(value) ? value : null
  }
  return null
}

/**
 * A stable identity for one completed task version, or null when the task is
 * not a completed version NEO could present. This is the key the proactive
 * hook records, so one version presents once even across a page reload.
 */
export function completionKey(task) {
  if (!task || typeof task !== 'object') return null
  if (task.state !== 'completed') return null
  const id = typeof task.id === 'string' ? task.id.trim() : ''
  if (!id || !TASK_ID_RE.test(id)) return null
  const version = task.version
  if (typeof version !== 'number' || !Number.isFinite(version) || version < 0) return null
  return `${id}:${version}`
}

/** Completed tasks whose version has not been observed yet, in list order. */
export function newlyCompleted(tasks, seen) {
  const known = seen instanceof Set ? seen : new Set(Array.isArray(seen) ? seen : [])
  const rows = Array.isArray(tasks) ? tasks : []
  const fresh = []
  for (const task of rows) {
    const key = completionKey(task)
    if (!key || known.has(key)) continue
    fresh.push(task)
  }
  return fresh
}

/** Interpret a persisted quiet setting without guessing at its encoding. */
export function isQuietValue(value) {
  if (value === true) return true
  if (typeof value === 'number') return value === 1
  if (typeof value === 'string') {
    const normal = value.trim().toLowerCase()
    return normal === '1' || normal === 'true' || normal === 'quiet' || normal === 'on'
  }
  return false
}

/** Read a REST failure the way the SDK reports it: `"409: {\"detail\": …}"`. */
export function parseRestError(error) {
  const message =
    error && typeof error.message === 'string'
      ? error.message
      : typeof error === 'string'
        ? error
        : ''
  const statusMatch = /(?:^|[\s:])([45]\d{2})\b/.exec(message)
  const status = statusMatch ? Number(statusMatch[1]) : null
  let detail = ''
  const brace = message.indexOf('{')
  if (brace !== -1) {
    try {
      const body = JSON.parse(message.slice(brace))
      if (body && typeof body.detail === 'string') detail = body.detail.trim()
    } catch (error) {
      detail = ''
    }
  }
  return { status, detail, message: message.trim() }
}

/** User-facing text for a failed REST call; 404/409/422 keep their meaning. */
export function describeRestError(error) {
  const parsed = parseRestError(error)
  if (parsed.status === 404) {
    const detail = parsed.detail ? ` (${parsed.detail})` : ''
    return `Not found${detail}. The task or route is no longer available.`
  }
  if (parsed.status === 409) {
    return parsed.detail
      ? `NEO refused this change: ${parsed.detail}.`
      : 'NEO refused this change in the task’s current state.'
  }
  if (parsed.status === 422) {
    return parsed.detail
      ? `The backend rejected this request: ${parsed.detail}.`
      : 'The backend rejected this request as invalid.'
  }
  if (parsed.detail) return parsed.detail
  if (parsed.message) return parsed.message
  return 'The request failed.'
}

/** The native executor session, or an honest note that it is not exposed. */
export function nativeSessionId(task) {
  if (!task || typeof task !== 'object') return 'No evidence from the backend'
  const keys = ['executor_session_id', 'native_session_id', 'session_id']
  for (const key of keys) {
    const value = task[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return 'Not reported by this backend — no evidence yet'
}

export function progressLabel(task) {
  if (!task || typeof task !== 'object') return 'Not reported'
  if (typeof task.progress === 'string' && task.progress.trim()) return task.progress.trim()
  if (typeof task.progress === 'number' && Number.isFinite(task.progress)) {
    return `${Math.round(task.progress)}%`
  }
  if (task.progress && typeof task.progress === 'object') {
    const percent = Number(task.progress.percent)
    if (Number.isFinite(percent)) return `${Math.round(percent)}%`
    if (typeof task.progress.detail === 'string' && task.progress.detail.trim()) {
      return task.progress.detail.trim()
    }
  }
  if (task.state === 'queued') return 'Waiting to start'
  return 'Not reported'
}

export function approvalLabel(task) {
  if (!task || typeof task !== 'object') return 'No approval requested'
  const approval = task.approval && typeof task.approval === 'object' ? task.approval : null
  if (approval && typeof approval.decision === 'string' && approval.decision) {
    const when = approval.at ? ` at ${String(approval.at)}` : ''
    return `${approval.decision} recorded${when}`
  }
  if (task.state === 'needs_approval') return 'Waiting for your approval decision'
  if (task.state === 'needs_input') return 'Waiting for your input'
  return 'No approval requested'
}

export function artifactNames(task) {
  const rows = task && Array.isArray(task.artifacts) ? task.artifacts : []
  return rows
    .filter(name => typeof name === 'string' && name.trim())
    .map(name => name.trim())
}

export function expectedNames(task) {
  const rows = task && Array.isArray(task.expected) ? task.expected : []
  return rows
    .filter(name => typeof name === 'string' && name.trim())
    .map(name => name.trim())
}

export function formatClock(value) {
  const seconds = Number(value)
  if (!Number.isFinite(seconds) || seconds <= 0) return '—'
  try {
    return new Date(seconds * 1000).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch (error) {
    return '—'
  }
}

// </pure-helpers>

// -- host surface ------------------------------------------------------------

const ROUTE_PATH = '/neo'
const ROUTES_AREA = sdk.ROUTES_AREA || 'routes'
const SIDEBAR_NAV_AREA = sdk.SIDEBAR_NAV_AREA || 'sidebar.nav'
const STATUS_RIGHT_AREA = (sdk.STATUSBAR_AREAS && sdk.STATUSBAR_AREAS.right) || 'statusBar.right'
const PALETTE_AREA = sdk.PALETTE_AREA || 'palette'
const TOOL_MEDIA_AREA = sdk.TOOL_MEDIA_RENDERER_AREA || 'transcript.toolMedia'

const PREVIEW_POLL_MS = 3000
const POLL_MS = 5000
const CHIP_POLL_MS = 5000
const LOG_TAIL = 80
const STORAGE_QUIET = 'neo.quiet'
const STORAGE_SEEN = 'neo.presented'
const SEEN_LIMIT = 200
const TASK_EVENT_PREFIX = 'plugin.neo.task.'
const CAPABILITY_EVENT = 'plugin.neo.capability.install_changed'

const NEO_MEDIA_TOOLS = ['neo_preview_publish', 'neo_task_complete']

// SDK components are resolved through the namespace import (no named-import
// load failures on a host that predates one of them).
const {
  Badge,
  Button,
  LogView,
  PanelDetail,
  PanelEmpty,
  PanelHeader,
  PanelList,
  PanelListRow,
  PanelMeta,
  PanelPill,
  PanelSectionLabel,
  SearchField,
  StatusDot
} = sdk

const CANDIDATE_FIELDS = [
  'name',
  'repo',
  'sha',
  'tier',
  'description',
  'version',
  'subdir',
  'dependencies'
]

function candidatePayload(candidate) {
  const body = {}
  for (const field of CANDIDATE_FIELDS) {
    const value = candidate && typeof candidate === 'object' ? candidate[field] : undefined
    if (field === 'dependencies') {
      body.dependencies = Array.isArray(value) ? value.filter(item => typeof item === 'string') : []
    } else if (typeof value === 'string') {
      body[field] = value
    } else {
      body[field] = ''
    }
  }
  return body
}

// -- shared hooks ------------------------------------------------------------

/**
 * Run `work` now and on an interval owned by the plugin context.
 * The interval is stopped on unmount and when the plugin is disabled.
 */
function usePolling(ctx, work, ms) {
  useEffect(() => {
    let stopped = false
    const tick = () => {
      if (!stopped) void work()
    }
    tick()
    const stop = ctx.setInterval(tick, ms)
    return () => {
      stopped = true
      if (typeof stop === 'function') stop()
    }
  }, [ctx, work, ms])
}

/** The app's route, when a host exposes one; the hash route otherwise. */
function currentRoute() {
  const atom =
    sdk.host && sdk.host.state ? sdk.host.state.route || sdk.host.state.currentRoute : null
  if (atom && typeof atom.get === 'function') {
    const value = atom.get()
    if (typeof value === 'string') return value
  }
  try {
    if (typeof window !== 'undefined' && window.location && typeof window.location.hash === 'string') {
      return window.location.hash
    }
  } catch (error) {
    return ''
  }
  return ''
}

function notify(message) {
  if (sdk.host && typeof sdk.host.notify === 'function') {
    sdk.host.notify({ kind: 'warning', message })
  }
}

function notifyInfo(message) {
  if (sdk.host && typeof sdk.host.notify === 'function') {
    sdk.host.notify({ kind: 'info', message })
  }
}

function navigateToNeo() {
  if (sdk.host && typeof sdk.host.navigate === 'function') sdk.host.navigate('/neo')
}

/** Navigate straight to one task; the page reads `?task=` on its next look. */
function navigateToTask(taskId) {
  if (
    typeof taskId !== 'string' ||
    !TASK_ID_RE.test(taskId) ||
    !sdk.host ||
    typeof sdk.host.navigate !== 'function'
  ) {
    return
  }
  sdk.host.navigate(`/neo?task=${encodeURIComponent(taskId)}`)
}

/**
 * Read one value from the host's plugin storage. Storage shapes differ across
 * hosts, so both an async `get` and a `getItem` door are tried; absence is an
 * honest null rather than an exception.
 */
async function storageRead(ctx, key) {
  const store = ctx && ctx.storage && typeof ctx.storage === 'object' ? ctx.storage : null
  if (!store) return null
  try {
    if (typeof store.get === 'function') return await store.get(key)
    if (typeof store.getItem === 'function') return await store.getItem(key)
  } catch (error) {
    return null
  }
  return null
}

/** Write one value to plugin storage; a host without storage is not an error. */
async function storageWrite(ctx, key, value) {
  const store = ctx && ctx.storage && typeof ctx.storage === 'object' ? ctx.storage : null
  if (!store) return false
  try {
    if (typeof store.set === 'function') {
      await store.set(key, value)
      return true
    }
    if (typeof store.setItem === 'function') {
      await store.setItem(key, value)
      return true
    }
  } catch (error) {
    return false
  }
  return false
}

/** Keep a persisted seen list bounded without dropping the newest keys. */
function trimSeen(keys) {
  const list = Array.isArray(keys) ? keys.filter(key => typeof key === 'string' && key) : []
  return list.slice(Math.max(0, list.length - SEEN_LIMIT))
}

/** Present one newly completed task exactly once from this surface. */
async function presentCompletion(ctx, task) {
  try {
    return await ctx.rest(`/tasks/${encodeURIComponent(task.id)}/present`, { method: 'POST' })
  } catch (error) {
    return { presented: false, skipped: true, reason: describeRestError(error) }
  }
}

// -- pieces ------------------------------------------------------------------

function FilterRow({ label, options, value, onChange }) {
  return jsxs('div', {
    className: 'flex flex-col gap-1',
    children: [
      jsx(PanelSectionLabel, { children: label }),
      jsx('div', {
        className: 'flex flex-wrap gap-1',
        children: options.map(option =>
          jsx(
            Button,
            {
              className: 'h-6 px-2 text-[0.65rem]',
              onClick: () => onChange(option.value),
              size: 'sm',
              variant: value === option.value ? 'secondary' : 'ghost',
              children: option.label
            },
            option.value
          )
        )
      })
    ]
  })
}

function ArtifactView({ artifact }) {
  if (!artifact) return null
  if (isImageMime(artifact.mime)) {
    return jsx('img', {
      alt: artifact.name,
      className: 'mt-2 max-h-80 w-auto max-w-full rounded border border-(--ui-stroke-tertiary) object-contain',
      decoding: 'async',
      loading: 'lazy',
      src: artifact.uri
    })
  }
  return jsxs('p', {
    className: 'mt-2 flex flex-wrap items-center gap-2 text-xs',
    children: [
      jsx('span', { className: 'text-muted-foreground', children: artifact.mime || 'application/octet-stream' }),
      jsx('a', {
        className: 'text-primary underline underline-offset-4',
        download: artifact.name,
        href: artifact.uri,
        children: `Download ${artifact.name}`
      })
    ]
  })
}

function CandidateRow({ candidate, pending, onInstall }) {
  const installable = candidate && candidate.installable !== false
  return jsxs('div', {
    className: 'flex items-start justify-between gap-3 rounded border border-(--ui-stroke-tertiary) px-2 py-1.5',
    children: [
      jsxs('div', {
        className: 'min-w-0',
        children: [
          jsx('p', { className: 'text-xs font-medium text-foreground', children: candidate.name || 'Unnamed candidate' }),
          jsx('p', {
            className: 'mt-0.5 break-all text-[0.65rem] text-muted-foreground',
            children: [candidate.description, candidate.sha ? ` · ${String(candidate.sha).slice(0, 10)}` : '']
              .filter(Boolean)
              .join('') || 'No description supplied.'
          })
        ]
      }),
      jsx(Button, {
        disabled: pending !== '' || !installable,
        onClick: () => onInstall(candidate),
        size: 'sm',
        variant: 'secondary',
        children: installable ? 'Install' : 'Not installable'
      })
    ]
  })
}

function NeoDetail({
  task,
  detailError,
  preview,
  log,
  capabilities,
  artifact,
  catalog,
  catalogQuery,
  notice,
  pending,
  onCatalogQuery,
  onSearchCatalog,
  onInstall,
  onCancel,
  onRetry,
  onApprove,
  onDeny,
  onOpenArtifact
}) {
  if (detailError) {
    return jsx(PanelEmpty, {
      description: detailError,
      icon: 'warning',
      title: 'This task could not be read'
    })
  }
  if (!task) {
    return jsx(PanelEmpty, {
      description: 'Select a task from the list to see its intent, executor, approvals, log and preview.',
      icon: 'rocket',
      title: 'No task selected'
    })
  }

  const bucket = userStateForTask(task)
  const artifacts = artifactNames(task)
  const expected = expectedNames(task)
  const installed = capabilities && Array.isArray(capabilities.installed) ? capabilities.installed : []
  const unavailable = capabilities && Array.isArray(capabilities.unavailable) ? capabilities.unavailable : []
  const canCancel = !isTerminalTask(task)
  const canRetry = isTerminalTask(task)
  const canApprove = task.state === 'needs_approval'
  const previewUri = buildPreviewUri(preview)

  const metaRows = [
    { label: 'Intent', value: task.intent || 'Not recorded' },
    { label: 'Capability', value: task.capability_id || 'Not recorded' },
    { label: 'Executor', value: task.executor || 'Not recorded' },
    { label: 'Native session', value: nativeSessionId(task) },
    { label: 'Workspace', value: task.workdir || 'Not recorded' },
    { label: 'Progress', value: progressLabel(task) },
    { label: 'Approval', value: approvalLabel(task) },
    { label: 'Created', value: formatClock(task.created_at) },
    { label: 'Updated', value: formatClock(task.updated_at) },
    { label: 'Finished', value: formatClock(task.finished_at) },
    { label: 'Version', value: task.version === null || task.version === undefined ? 'Not reported' : String(task.version) },
    { label: 'Task id', value: task.id || 'Not recorded' },
    { label: 'Parent', value: task.parent_id || 'None' }
  ]

  const logLines = log && Array.isArray(log.lines) ? log.lines : []

  return jsxs(PanelDetail, {
    children: [
      jsxs('div', {
        className: 'flex flex-wrap items-start justify-between gap-2',
        children: [
          jsxs('div', {
            className: 'min-w-0',
            children: [
              jsx('h2', {
                className: 'text-sm font-semibold text-foreground',
                children: task.intent || task.id || 'Untitled task'
              }),
              jsxs('div', {
                className: 'mt-1 flex flex-wrap items-center gap-1.5 text-[0.68rem] text-muted-foreground',
                children: [
                  jsx(PanelPill, { tone: STATE_TONES[bucket], children: STATE_LABELS[bucket] }),
                  jsx('span', { children: rawStateLabel(task.state) }),
                  task.kind
                    ? jsx(Badge, { size: 'xs', variant: 'outline', children: task.kind })
                    : null,
                  jsx('span', { children: task.id })
                ]
              })
            ]
          }),
          jsxs('div', {
            className: 'flex flex-wrap items-center gap-1',
            children: [
              canCancel
                ? jsx(Button, {
                    'aria-busy': pending === 'cancel' || undefined,
                    disabled: pending !== '',
                    onClick: onCancel,
                    size: 'sm',
                    variant: 'outline',
                    children: 'Cancel'
                  })
                : null,
              canRetry
                ? jsx(Button, {
                    'aria-busy': pending === 'retry' || undefined,
                    disabled: pending !== '',
                    onClick: onRetry,
                    size: 'sm',
                    variant: 'secondary',
                    children: 'Retry'
                  })
                : null,
              artifacts.length
                ? jsx(Button, {
                    'aria-busy': pending === 'open' || undefined,
                    disabled: pending !== '',
                    onClick: () => onOpenArtifact(artifacts[0]),
                    size: 'sm',
                    variant: 'secondary',
                    children: 'Open result'
                  })
                : null
            ]
          })
        ]
      }),

      notice
        ? jsx('p', {
            className:
              'rounded border border-(--ui-stroke-tertiary) px-2 py-1 text-xs ' +
              (notice.kind === 'error' ? 'text-destructive' : 'text-muted-foreground'),
            children: notice.message
          })
        : null,

      jsxs('section', {
        children: [
          jsx(PanelSectionLabel, { children: 'Task' }),
          jsx(PanelMeta, { rows: metaRows })
        ]
      }),

      canApprove
        ? jsxs('section', {
            children: [
              jsx(PanelSectionLabel, { children: 'Approval' }),
              jsx('p', {
                className: 'text-xs text-muted-foreground',
                children: 'External actions stay stopped until you decide. Approving authorises only this request.'
              }),
              jsxs('div', {
                className: 'mt-1 flex items-center gap-2',
                children: [
                  jsx(Button, {
                    'aria-busy': pending === 'approve' || undefined,
                    disabled: pending !== '',
                    onClick: onApprove,
                    size: 'sm',
                    variant: 'default',
                    children: 'Approve'
                  }),
                  jsx(Button, {
                    'aria-busy': pending === 'deny' || undefined,
                    disabled: pending !== '',
                    onClick: onDeny,
                    size: 'sm',
                    variant: 'outline',
                    children: 'Deny'
                  })
                ]
              })
            ]
          })
        : null,

      task.result_summary || task.error_detail
        ? jsxs('section', {
            children: [
              jsx(PanelSectionLabel, { children: 'Result' }),
              task.result_summary
                ? jsx('p', { className: 'text-xs text-foreground/85', children: task.result_summary })
                : null,
              task.error_detail
                ? jsx('p', { className: 'text-xs text-destructive', children: task.error_detail })
                : null,
              task.error_code
                ? jsx('p', { className: 'text-[0.65rem] text-muted-foreground', children: `Code: ${task.error_code}` })
                : null
            ]
          })
        : null,

      jsxs('section', {
        children: [
          jsx(PanelSectionLabel, { children: 'Preview' }),
          previewUri
            ? jsx('img', {
                alt: `Latest preview for ${task.id}`,
                className: 'mt-1 max-h-80 w-auto max-w-full rounded border border-(--ui-stroke-tertiary) object-contain',
                decoding: 'async',
                key:
                  preview && preview.rev !== null && preview.rev !== undefined
                    ? `${task.id}:${preview.rev}`
                    : task.id,
                loading: 'lazy',
                src: previewUri
              })
            : jsx('p', { className: 'text-xs text-muted-foreground', children: previewPlaceholder(preview) }),
          preview && preview.rev !== null && preview.rev !== undefined
            ? jsx('p', {
                className: 'mt-1 text-[0.65rem] text-muted-foreground',
                children: `Revision ${preview.rev}`
              })
            : null
        ]
      }),

      jsxs('section', {
        children: [
          jsx(PanelSectionLabel, { children: 'Log tail' }),
          logLines.length
            ? jsx(LogView, { className: 'mt-1 max-h-48', children: logLines.join('\n') })
            : jsx('p', {
                className: 'text-xs text-muted-foreground',
                children: 'No log tail is available for this task.'
              })
        ]
      }),

      jsxs('section', {
        children: [
          jsx(PanelSectionLabel, { children: 'Artifacts' }),
          expected.length
            ? jsx('p', {
                className: 'text-[0.65rem] text-muted-foreground',
                children: `Expected: ${expected.join(', ')}`
              })
            : null,
          artifacts.length
            ? jsxs('div', {
                className: 'mt-1 flex flex-col gap-1',
                children: artifacts.map(name =>
                  jsxs(
                    'div',
                    {
                      className: 'flex items-center justify-between gap-2 text-xs',
                      children: [
                        jsx('span', { className: 'min-w-0 break-all text-foreground/85', children: name }),
                        jsx(Button, {
                          disabled: pending !== '',
                          onClick: () => onOpenArtifact(name),
                          size: 'sm',
                          variant: 'ghost',
                          children: 'Open'
                        })
                      ]
                    },
                    name
                  )
                )
              })
            : jsx('p', {
                className: 'text-xs text-muted-foreground',
                children: 'No artifacts are recorded for this task.'
              }),
          jsx(ArtifactView, { artifact })
        ]
      }),

      jsxs('section', {
        children: [
          jsx(PanelSectionLabel, { children: 'Capability' }),
          installed.length
            ? jsx('p', {
                className: 'text-xs text-muted-foreground',
                children:
                  'Installed: ' +
                  installed
                    .map(item => `${item.name}${item.enabled === false ? ' (disabled)' : ''}`)
                    .join(', ')
              })
            : jsx('p', {
                className: 'text-xs text-muted-foreground',
                children: 'No capability report is available for this task.'
              }),
          unavailable.length
            ? jsx('p', {
                className: 'text-xs text-muted-foreground',
                children: `Unavailable: ${unavailable.join(', ')}`
              })
            : null,
          jsxs('div', {
            className: 'mt-1 flex items-center gap-2',
            children: [
              jsx(SearchField, {
                'aria-label': 'Search the capability catalog',
                containerClassName: 'w-56',
                onChange: onCatalogQuery,
                placeholder: 'Search the capability catalog',
                value: catalogQuery
              }),
              jsx(Button, {
                'aria-busy': pending === 'catalog' || undefined,
                disabled: pending !== '',
                onClick: onSearchCatalog,
                size: 'sm',
                variant: 'secondary',
                children: 'Search'
              })
            ]
          }),
          catalog === null
            ? null
            : catalog.length
              ? jsxs('div', {
                  className: 'mt-1 flex flex-col gap-1',
                  children: catalog.map(candidate => jsx(CandidateRow, {
                    candidate,
                    onInstall,
                    pending
                  }, candidate.name || candidate.sha))
                })
              : jsx('p', {
                  className: 'mt-1 text-xs text-muted-foreground',
                  children: 'No catalog candidates matched that query.'
                })
        ]
      })
    ]
  })
}

// -- mission control page ----------------------------------------------------

function NeoPage({ ctx }) {
  const [tasks, setTasks] = useState([])
  const [health, setHealth] = useState(null)
  const [listError, setListError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState(() => taskIdFromRoute(currentRoute()))
  const [filters, setFilters] = useState({ state: 'all', kind: 'all' })
  const [detail, setDetail] = useState(null)
  const [detailError, setDetailError] = useState(null)
  const [preview, setPreview] = useState(null)
  const [log, setLog] = useState(null)
  const [capabilities, setCapabilities] = useState(null)
  const [artifact, setArtifact] = useState(null)
  const [catalog, setCatalog] = useState(null)
  const [catalogQuery, setCatalogQuery] = useState('')
  const [notice, setNotice] = useState(null)
  const [completionNotice, setCompletionNotice] = useState(null)
  const [pending, setPending] = useState('')
  const [quiet, setQuiet] = useState(false)

  const selectedIdRef = useRef(selectedId)
  const pendingRef = useRef('')
  const previewToken = useRef(0)
  const refreshRef = useRef(null)
  const completionsSeen = useRef(null)

  const loadList = useCallback(async () => {
    const results = await Promise.allSettled([ctx.rest(`/tasks`), ctx.rest(`/health`)])
    const listResult = results[0]
    const healthResult = results[1]
    if (listResult.status === 'fulfilled') {
      const body = listResult.value
      setTasks(body && Array.isArray(body.tasks) ? body.tasks : [])
      setListError(null)
    } else {
      setListError(describeRestError(listResult.reason))
    }
    if (healthResult.status === 'fulfilled' && healthResult.value && typeof healthResult.value === 'object') {
      setHealth(healthResult.value)
    } else {
      setHealth(null)
    }
    setLoading(false)
  }, [ctx])

  const loadDetail = useCallback(
    async id => {
      if (!id) {
        setDetail(null)
        setDetailError(null)
        setPreview(null)
        setLog(null)
        setCapabilities(null)
        return
      }
      const token = previewToken.current + 1
      previewToken.current = token
      const results = await Promise.allSettled([
        ctx.rest(`/tasks/${encodeURIComponent(id)}`),
        ctx.rest(`/tasks/${encodeURIComponent(id)}/preview`),
        ctx.rest(`/tasks/${encodeURIComponent(id)}/log?tail=${LOG_TAIL}`),
        ctx.rest(`/tasks/${encodeURIComponent(id)}/capabilities`)
      ])
      if (previewToken.current !== token || selectedIdRef.current !== id) return

      const taskResult = results[0]
      const previewResult = results[1]
      const logResult = results[2]
      const capabilityResult = results[3]

      if (taskResult.status === 'fulfilled' && taskResult.value && typeof taskResult.value === 'object') {
        setDetail(taskResult.value)
        setDetailError(null)
      } else {
        setDetail(null)
        setDetailError(
          taskResult.status === 'rejected'
            ? describeRestError(taskResult.reason)
            : 'The task is no longer available.'
        )
      }

      if (previewResult.status === 'fulfilled' && previewResult.value && typeof previewResult.value === 'object') {
        const next = previewResult.value
        setPreview(current => (samePreview(current, next) ? current : next))
      } else {
        setPreview({
          available: false,
          reason:
            previewResult.status === 'rejected'
              ? describeRestError(previewResult.reason)
              : 'preview is unavailable'
        })
      }

      if (logResult.status === 'fulfilled' && logResult.value && Array.isArray(logResult.value.lines)) {
        setLog(logResult.value)
      } else {
        setLog(null)
      }

      if (
        capabilityResult.status === 'fulfilled' &&
        capabilityResult.value &&
        typeof capabilityResult.value === 'object'
      ) {
        setCapabilities(capabilityResult.value)
      } else {
        setCapabilities(null)
      }
    },
    [ctx]
  )

  const pollPreview = useCallback(async () => {
    const id = selectedIdRef.current
    if (!id) return
    const token = previewToken.current
    let next = null
    try {
      next = await ctx.rest(`/tasks/${encodeURIComponent(id)}/preview`)
    } catch (error) {
      next = { available: false, reason: describeRestError(error) }
    }
    if (previewToken.current !== token || selectedIdRef.current !== id) return
    if (!next || typeof next !== 'object') return
    // The data URI changes with the bytes, so a revision paints immediately
    // and an unchanged revision leaves the mounted image alone.
    setPreview(current => (samePreview(current, next) ? current : next))
  }, [ctx])

  const refresh = useCallback(async () => {
    await loadList()
    const id = selectedIdRef.current
    if (id) await loadDetail(id)
  }, [loadList, loadDetail])

  usePolling(ctx, refresh, POLL_MS)
  usePolling(ctx, pollPreview, PREVIEW_POLL_MS)

  useEffect(() => {
    let live = true
    storageRead(ctx, STORAGE_QUIET).then(
      value => {
        if (live) setQuiet(isQuietValue(value))
      },
      () => {
        if (live) setQuiet(false)
      }
    )
    return () => {
      live = false
    }
  }, [ctx])

  const toggleQuiet = () => {
    setQuiet(current => {
      const next = !current
      void storageWrite(ctx, STORAGE_QUIET, next ? '1' : '0')
      return next
    })
  }

  useEffect(() => {
    selectedIdRef.current = selectedId
    setArtifact(null)
    setCatalog(null)
    setNotice(null)
    // Never paint one task's preview under another task's heading, even for
    // the moment before the new selection's preview arrives.
    setPreview(null)
    if (selectedId) void loadDetail(selectedId)
  }, [selectedId, loadDetail])

  // Auto-select from `?task=<id>` when the host route carries one, and fall
  // back to the newest task when the selection is gone.
  useEffect(() => {
    const rows = Array.isArray(tasks) ? tasks.filter(task => task && typeof task.id === 'string') : []
    const routeId = taskIdFromRoute(currentRoute())
    if (routeId && routeId !== selectedId && rows.some(task => task.id === routeId)) {
      setSelectedId(routeId)
      return
    }
    if (selectedId && rows.some(task => task.id === selectedId)) return
    if (rows.length && rows[0].id !== selectedId) setSelectedId(rows[0].id)
  }, [tasks, selectedId])

  // A completion that appears while the page is open is announced visibly and
  // selected. The initial list is context, not news, so it primes the set.
  useEffect(() => {
    const rows = Array.isArray(tasks)
      ? tasks.filter(task => task && typeof task.id === 'string')
      : []
    const completed = rows.filter(task => completionKey(task))
    if (completionsSeen.current === null) {
      completionsSeen.current = new Set(completed.map(task => completionKey(task)))
      return
    }
    const fresh = newlyCompleted(rows, completionsSeen.current)
    if (!fresh.length) return
    for (const task of fresh) completionsSeen.current.add(completionKey(task))
    const newest = fresh[fresh.length - 1]
    setCompletionNotice(
      `Task completed: ${newest.intent || newest.id}. The verified result is ready to open.`
    )
    if (typeof newest.id === 'string' && newest.id) setSelectedId(newest.id)
  }, [tasks])

  useEffect(() => {
    refreshRef.current = () => {
      void refresh()
    }
  }, [refresh])

  useEffect(() => {
    if (typeof ctx.onEvent !== 'function') return undefined
    const stop = ctx.onEvent('*', event => {
      const type = event && typeof event.type === 'string' ? event.type : ''
      if (type !== CAPABILITY_EVENT && type.indexOf(TASK_EVENT_PREFIX) !== 0) return
      const run = refreshRef.current
      if (typeof run === 'function') run()
    })
    return () => {
      if (typeof stop === 'function') stop()
    }
  }, [ctx])

  const runAction = useCallback(
    async (key, work, after) => {
      if (pendingRef.current) return
      pendingRef.current = key
      setPending(key)
      setNotice(null)
      try {
        const result = await work()
        if (typeof after === 'function') after(result)
        await refresh()
      } catch (error) {
        const message = describeRestError(error)
        setNotice({ kind: 'error', message })
        notify(message)
      } finally {
        pendingRef.current = ''
        setPending('')
      }
    },
    [ctx, refresh]
  )

  const withId = work => {
    const id = selectedIdRef.current
    if (!id) {
      setNotice({ kind: 'error', message: 'Select a task first.' })
      return
    }
    void work(id)
  }

  const cancelTask = () =>
    withId(id =>
      runAction('cancel', () => ctx.rest(`/tasks/${encodeURIComponent(id)}/cancel`, { method: 'POST' }), () =>
        setNotice({ kind: 'info', message: 'Cancellation requested. The state will confirm when the process has stopped.' })
      )
    )

  const retryTask = () =>
    withId(id =>
      runAction(
        'retry',
        () => ctx.rest(`/tasks/${encodeURIComponent(id)}/retry`, { method: 'POST' }),
        child => {
          if (child && typeof child.id === 'string' && child.id) {
            setSelectedId(child.id)
            setNotice({ kind: 'info', message: `New attempt created: ${child.id}` })
          } else {
            setNotice({ kind: 'info', message: 'Retry requested.' })
          }
        }
      )
    )

  const approveOrDeny = (body, key) =>
    withId(id =>
      runAction(
        key,
        () => ctx.rest(`/tasks/${encodeURIComponent(id)}/approve`, { method: 'POST', body }),
        () => setNotice({ kind: 'info', message: 'Decision recorded.' })
      )
    )

  const approveTask = () => approveOrDeny({ decision: 'approve' }, 'approve')
  const denyTask = () => approveOrDeny({ decision: 'deny' }, 'deny')

  const openArtifact = name => {
    const id = selectedIdRef.current
    if (!id || typeof name !== 'string' || !name) return
    void runAction(
      'open',
      () =>
        ctx.rest(`/tasks/${encodeURIComponent(id)}/artifact?name=${encodeURIComponent(name)}`),
      result => {
        const uri = buildArtifactUri(result)
        if (!uri) {
          setNotice({ kind: 'error', message: 'The backend returned no readable bytes for that artifact.' })
          return
        }
        setArtifact({
          name: typeof result.name === 'string' && result.name ? result.name : name,
          mime: typeof result.mime === 'string' ? result.mime : '',
          uri
        })
        setNotice({ kind: 'info', message: 'Artifact loaded below.' })
      }
    )
  }

  const installCandidate = candidate => {
    if (!candidate || typeof candidate.name !== 'string') return
    void runAction(
      `install:${candidate.name}`,
      async () => {
        const body = candidatePayload(candidate)
        await ctx.rest(`/capabilities/consent`, { method: 'POST', body })
        return ctx.rest(`/capabilities/install`, { method: 'POST', body })
      },
      result => {
        if (result && result.ok) {
          setNotice({ kind: 'info', message: `Installed ${candidate.name}.` })
        } else {
          const reason =
            result && typeof result.reason === 'string' && result.reason
              ? result.reason
              : 'the installer did not confirm the pinned revision'
          setNotice({ kind: 'error', message: `Install of ${candidate.name} did not complete: ${reason}.` })
        }
      }
    )
  }

  const searchCatalog = () => {
    const query = catalogQuery.trim()
    if (!query) {
      setNotice({ kind: 'error', message: 'Enter a capability name before searching the catalog.' })
      return
    }
    void runAction(
      'catalog',
      () => ctx.rest(`/capabilities?q=${encodeURIComponent(query)}`),
      result => {
        const found = result && Array.isArray(result.catalog) ? result.catalog : []
        setCatalog(found)
        setNotice({
          kind: found.length ? 'info' : 'error',
          message: found.length
            ? `${found.length} catalog candidate${found.length === 1 ? '' : 's'} found for “${query}”.`
            : `No catalog candidates matched “${query}”.`
        })
      }
    )
  }

  const rows = Array.isArray(tasks) ? tasks.filter(task => task && typeof task === 'object') : []
  const visible = filterTasks(rows, filters)
  const groups = groupTasks(visible)
  const kinds = kindsOf(rows)
  const activeCount = activeTaskCount(rows)
  const needsYouCount = rows.filter(task => userStateForTask(task) === 'needs-you').length
  const failedCount = rows.filter(task => userStateForTask(task) === 'failed').length
  const selectedTask =
    detail && detail.id === selectedId
      ? detail
      : rows.find(task => task.id === selectedId) || null

  const stateOptions = [{ value: 'all', label: 'All' }].concat(
    STATE_ORDER.map(state => ({ value: state, label: STATE_LABELS[state] }))
  )
  const kindOptions = [{ value: 'all', label: 'All kinds' }].concat(
    kinds.map(kind => ({ value: kind, label: kind }))
  )

  const healthOk = health && health.status === 'ok'
  const healthLabel = healthOk
    ? `Backend healthy · v${health.version || '0.1.0'}`
    : health
      ? 'Backend reported an unexpected status'
      : 'Backend unavailable'

  const listSections = groups
    .filter(group => group.tasks.length)
    .map(group =>
      jsxs(
        'div',
        {
          className: 'flex flex-col',
          children: [
            jsx(PanelSectionLabel, {
              className: 'px-2 pt-2',
              children: `${group.label} · ${group.tasks.length}`
            }),
            jsx('div', {
              className: 'flex flex-col',
              children: group.tasks.map(task =>
                jsx(
                  PanelListRow,
                  {
                    active: task.id === selectedId,
                    lead: jsx(StatusDot, { tone: STATE_TONES[group.state] }),
                    meta: rawStateLabel(task.state),
                    onSelect: () => setSelectedId(task.id),
                    rowKey: task.id,
                    title: task.intent || task.id || 'Untitled task'
                  },
                  task.id || group.state
                )
              )
            })
          ]
        },
        group.state
      )
    )

  return jsxs('div', {
    className: 'flex h-full min-h-0 flex-col px-3 pb-3 pt-2',
    children: [
      jsx(PanelHeader, {
        actions: jsxs('div', {
          className: 'flex flex-wrap items-center gap-2',
          children: [
            jsxs('div', {
              className: 'flex items-center gap-1.5 text-[0.68rem] text-muted-foreground',
              children: [
                jsx(StatusDot, { tone: healthOk ? 'good' : 'muted' }),
                jsx('span', { children: healthLabel })
              ]
            }),
            jsx(Badge, { variant: 'muted', children: `${activeCount} active` }),
            needsYouCount
              ? jsx(Badge, { variant: 'warn', children: `${needsYouCount} need you` })
              : null,
            failedCount
              ? jsx(Badge, { variant: 'destructive', children: `${failedCount} failed` })
              : null,
            jsx(Button, {
              'aria-pressed': quiet,
              onClick: toggleQuiet,
              size: 'sm',
              variant: quiet ? 'secondary' : 'ghost',
              children: quiet ? 'Quiet on' : 'Quiet'
            }),
            jsx(Button, {
              disabled: loading,
              onClick: () => {
                void refresh()
              },
              size: 'sm',
              variant: 'ghost',
              children: loading ? 'Loading…' : 'Refresh'
            })
          ]
        }),
        subtitle: 'Durable work across Codex, Claude, Blender and the browser.',
        title: 'Mission control'
      }),

      listError
        ? jsx('p', {
            className: 'mb-2 rounded border border-(--ui-stroke-tertiary) px-2 py-1 text-xs text-destructive',
            children: listError
          })
        : null,

      completionNotice
        ? jsx('p', {
            className:
              'mb-2 rounded border border-(--ui-stroke-tertiary) px-2 py-1 text-xs text-muted-foreground',
            children: completionNotice
          })
        : null,

      jsxs('div', {
        className: 'flex min-h-0 flex-1 flex-col gap-4 overflow-hidden min-[47.5rem]:flex-row min-[47.5rem]:gap-5',
        children: [
          jsxs(PanelList, {
            className: 'min-[47.5rem]:w-72',
            children: [
              jsx('div', {
                className: 'flex flex-col gap-2 px-2 pb-2',
                children: [
                  jsx(FilterRow, {
                    label: 'State',
                    onChange: value => setFilters(current => ({ ...current, state: value })),
                    options: stateOptions,
                    value: filters.state
                  }),
                  jsx(FilterRow, {
                    label: 'Kind',
                    onChange: value => setFilters(current => ({ ...current, kind: value })),
                    options: kindOptions,
                    value: filters.kind
                  })
                ]
              }),
              loading && !rows.length
                ? jsx('p', { className: 'px-2 text-xs text-muted-foreground', children: 'Loading tasks…' })
                : rows.length
                  ? visible.length
                    ? listSections
                    : jsx(PanelEmpty, {
                        description: 'Change the state or kind filter to see other work.',
                        icon: 'filter',
                        title: 'No tasks match these filters'
                      })
                  : jsx(PanelEmpty, {
                      description: 'Tasks NEO accepts appear here, grouped by their user-facing state.',
                      icon: 'rocket',
                      title: 'No tasks yet'
                    })
            ]
          }),

          jsx(NeoDetail, {
            artifact,
            capabilities,
            catalog,
            catalogQuery,
            detailError,
            log,
            notice,
            onApprove: approveTask,
            onCancel: cancelTask,
            onCatalogQuery: setCatalogQuery,
            onDeny: denyTask,
            onInstall: installCandidate,
            onOpenArtifact: openArtifact,
            onRetry: retryTask,
            onSearchCatalog: searchCatalog,
            pending,
            preview,
            task: selectedTask
          })
        ]
      })
    ]
  })
}

// -- status bar chip ---------------------------------------------------------

function NeoStatusChip({ ctx }) {
  const [summary, setSummary] = useState({ active: 0, needsYou: 0, loaded: false })

  const seenRef = useRef(new Set())
  const primedRef = useRef(false)
  const presentingRef = useRef(new Set())
  const quietRef = useRef(false)
  const hydratedRef = useRef(false)

  const persistSeen = useCallback(async () => {
    await storageWrite(ctx, STORAGE_SEEN, trimSeen(Array.from(seenRef.current)))
  }, [ctx])

  /**
   * Announce and present completions observed after the chip mounted. The
   * first list is context, never news: priming it prevents a backlog from
   * opening every finished result at startup. The backend ledger is the
   * second line of defence if two surfaces see the same completion.
   */
  const observeCompletions = useCallback(
    async rows => {
      const completed = rows.filter(task => completionKey(task))
      if (!primedRef.current) {
        for (const task of completed) seenRef.current.add(completionKey(task))
        primedRef.current = true
        await persistSeen()
        return
      }
      const fresh = newlyCompleted(rows, seenRef.current)
      for (const task of fresh) {
        const key = completionKey(task)
        if (!key || presentingRef.current.has(key)) continue
        presentingRef.current.add(key)
        seenRef.current.add(key)
        await presentCompletion(ctx, task)
        navigateToTask(task.id)
        if (!quietRef.current) {
          notifyInfo(`Task completed: ${task.intent || task.id}.`)
        }
      }
      if (fresh.length) await persistSeen()
    },
    [ctx, persistSeen]
  )

  const load = useCallback(async () => {
    quietRef.current = isQuietValue(await storageRead(ctx, STORAGE_QUIET))
    if (!hydratedRef.current) {
      const stored = await storageRead(ctx, STORAGE_SEEN)
      if (Array.isArray(stored)) {
        for (const key of stored) {
          if (typeof key === 'string' && key) seenRef.current.add(key)
        }
      }
      hydratedRef.current = true
    }
    try {
      const body = await ctx.rest(`/tasks`)
      const rows = body && Array.isArray(body.tasks) ? body.tasks : []
      setSummary({
        active: activeTaskCount(rows),
        needsYou: rows.filter(task => userStateForTask(task) === 'needs-you').length,
        loaded: true
      })
      await observeCompletions(rows)
    } catch (error) {
      setSummary({ active: 0, needsYou: 0, loaded: false })
    }
  }, [ctx, observeCompletions])

  usePolling(ctx, load, CHIP_POLL_MS)

  const tone = summary.needsYou ? 'warn' : summary.active ? 'good' : 'muted'
  const label = summary.loaded ? `NEO ${summary.active}` : 'NEO'
  const title = !summary.loaded
    ? 'NEO mission control — backend unavailable'
    : summary.needsYou
      ? `${summary.needsYou} NEO task${summary.needsYou === 1 ? '' : 's'} need you`
      : `${summary.active} active NEO task${summary.active === 1 ? '' : 's'}`

  return jsxs('button', {
    className: 'flex items-center gap-1 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) hover:text-(--ui-text-primary)',
    onClick: navigateToNeo,
    title,
    type: 'button',
    children: [jsx(StatusDot, { tone }), jsx('span', { children: label })]
  })
}

// -- tool media --------------------------------------------------------------

function NeoMediaCard({ args, fallback, file, result }) {
  const [src, setSrc] = useState(null)

  useEffect(() => {
    let live = true
    if (!file || typeof file.resolveUrl !== 'function') {
      setSrc('')
      return () => {
        live = false
      }
    }
    Promise.resolve(file.resolveUrl()).then(
      url => {
        if (live) setSrc(typeof url === 'string' ? url : '')
      },
      () => {
        if (live) setSrc('')
      }
    )
    return () => {
      live = false
    }
  }, [file])

  if (!file || (file.kind !== 'image' && file.kind !== 'video')) return fallback || null
  if (src === '') return fallback || null
  if (!src) {
    return jsx('p', { className: 'text-xs text-muted-foreground', children: 'Loading NEO media…' })
  }

  const caption =
    result && typeof result.summary === 'string'
      ? result.summary
      : args && typeof args.intent === 'string'
        ? args.intent
        : file.name

  if (file.kind === 'video') {
    return jsx('video', { className: 'max-h-80 max-w-full rounded', controls: true, src })
  }

  return jsxs('figure', {
    className: 'flex flex-col gap-1',
    children: [
      jsx('img', {
        alt: file.name || 'NEO result',
        className: 'max-h-80 max-w-full rounded border border-(--ui-stroke-tertiary) object-contain',
        decoding: 'async',
        loading: 'lazy',
        src
      }),
      caption
        ? jsx('figcaption', { className: 'text-[0.65rem] text-muted-foreground', children: caption })
        : null
    ]
  })
}

// -- registration ------------------------------------------------------------

function register(ctx) {
  ctx.registerMany([
    {
      id: 'page',
      area: ROUTES_AREA,
      data: { path: '/neo' },
      render: () => jsx(NeoPage, { ctx })
    },
    {
      id: 'nav',
      area: SIDEBAR_NAV_AREA,
      data: { path: '/neo', label: 'NEO', codicon: 'rocket' }
    },
    {
      id: 'chip',
      area: STATUS_RIGHT_AREA,
      order: 140,
      render: () => jsx(NeoStatusChip, { ctx })
    },
    {
      id: 'open',
      area: PALETTE_AREA,
      data: {
        id: 'neo.open',
        label: 'Open NEO mission control',
        keywords: ['neo', 'tasks', 'mission', 'control'],
        run: navigateToNeo
      }
    }
  ])

  if (TOOL_MEDIA_AREA) {
    ctx.register({
      id: 'media',
      area: TOOL_MEDIA_AREA,
      data: {
        tools: NEO_MEDIA_TOOLS,
        render: props => jsx(NeoMediaCard, props)
      }
    })
  }
}

export default {
  id: 'neo',
  name: 'NEO',
  register
}
