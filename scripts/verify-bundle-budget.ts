import { readFile, readdir } from 'node:fs/promises'
import { basename, relative, resolve, sep } from 'node:path'
import { gzipSync } from 'node:zlib'

const root = resolve(import.meta.dirname, '..')
const distRoot = resolve(root, 'dist')

const budgets = {
  entryJavascript: {
    rawBytes: 1_350_000,
    gzipBytes: 370_000,
  },
} as const

const indexHtml = await readFile(resolve(distRoot, 'index.html'), 'utf8').catch((error: unknown) => {
  throw new Error(`Production bundle is missing. Run npm run build first. ${messageOf(error)}`)
})
const entrySource = indexHtml.match(/<script\b[^>]*\btype="module"[^>]*\bsrc="([^"]+)"/)?.[1]
if (!entrySource) throw new Error('Could not locate the production module entry in dist/index.html.')

const entryPath = resolveDistAsset(entrySource)
const entryContents = await readFile(entryPath)
const entry = measure(entryContents)

const assetPaths = await listFiles(resolve(distRoot, 'assets'))
const onnxWasmPaths = assetPaths
  .filter(path => basename(path).startsWith('ort-') && path.endsWith('.wasm'))
  .sort((left, right) => left.localeCompare(right))

const report = {
  entryJavascript: reportMeasurement(relative(root, entryPath), entry, budgets.entryJavascript),
  retiredOnnxAssets: onnxWasmPaths.map(path => relative(root, path).split(sep).join('/')),
}

console.log(JSON.stringify(report, null, 2))

const failures = [
  budgetFailure('entry JavaScript (raw)', entry.rawBytes, budgets.entryJavascript.rawBytes),
  budgetFailure('entry JavaScript (gzip)', entry.gzipBytes, budgets.entryJavascript.gzipBytes),
  onnxWasmPaths.length > 0
    ? `retired ONNX Runtime assets are still shipped: ${onnxWasmPaths.map(path => basename(path)).join(', ')}`
    : null,
].filter((failure): failure is string => failure !== null)

if (failures.length > 0) {
  throw new Error(`Production bundle budget exceeded:\n${failures.map(failure => `- ${failure}`).join('\n')}`)
}

console.log('Production bundle budget verification passed.')

function resolveDistAsset(source: string) {
  const withoutQuery = source.split(/[?#]/, 1)[0]?.replace(/^\.?\//, '')
  if (!withoutQuery) throw new Error(`Invalid production entry source: ${source}`)
  const path = resolve(distRoot, withoutQuery)
  const relativePath = relative(distRoot, path)
  if (relativePath.startsWith('..') || resolve(distRoot, relativePath) !== path) {
    throw new Error(`Production entry escaped dist/: ${source}`)
  }
  return path
}

async function listFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true })
  const nested = await Promise.all(
    entries.map(entry => {
      const path = resolve(directory, entry.name)
      return entry.isDirectory() ? listFiles(path) : Promise.resolve([path])
    }),
  )
  return nested.flat()
}

function measure(contents: Uint8Array) {
  return {
    rawBytes: contents.byteLength,
    gzipBytes: gzipSync(contents, { level: 9 }).byteLength,
  }
}

function reportMeasurement(
  file: string,
  actual: { rawBytes: number; gzipBytes: number },
  budget: { rawBytes: number; gzipBytes: number },
) {
  return {
    file: file.split(sep).join('/'),
    raw: reportSize(actual.rawBytes, budget.rawBytes),
    gzip: reportSize(actual.gzipBytes, budget.gzipBytes),
  }
}

function reportSize(bytes: number, budgetBytes: number) {
  return {
    bytes,
    kibibytes: round(bytes / 1024),
    budgetBytes,
    budgetKibibytes: round(budgetBytes / 1024),
    usagePercent: round((bytes / budgetBytes) * 100),
    headroomBytes: budgetBytes - bytes,
  }
}

function budgetFailure(label: string, actual: number, budget: number) {
  if (actual <= budget) return null
  return `${label} is ${formatBytes(actual)}; budget is ${formatBytes(budget)} (${formatBytes(actual - budget)} over).`
}

function formatBytes(bytes: number) {
  return `${bytes.toLocaleString('en-US')} B / ${round(bytes / 1024).toLocaleString('en-US')} KiB`
}

function round(value: number) {
  return Math.round(value * 100) / 100
}

function messageOf(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}
