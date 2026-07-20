import { readFile, readdir } from 'node:fs/promises'
import { basename, relative, resolve, sep } from 'node:path'
import { brotliDecompressSync, gunzipSync, gzipSync } from 'node:zlib'

const root = resolve(import.meta.dirname, '..')
const distRoot = resolve(root, 'dist')

const budgets = {
  entryJavascript: {
    rawBytes: 1_350_000,
    gzipBytes: 370_000,
  },
} as const

verifyRetiredOnnxDetector()

const indexHtml = await readFile(resolve(distRoot, 'index.html'), 'utf8').catch((error: unknown) => {
  throw new Error(`Production bundle is missing. Run npm run build first. ${messageOf(error)}`)
})
const entrySource = indexHtml.match(/<script\b[^>]*\btype="module"[^>]*\bsrc="([^"]+)"/)?.[1]
if (!entrySource) throw new Error('Could not locate the production module entry in dist/index.html.')

const entryPath = resolveDistAsset(entrySource)
const entryContents = await readFile(entryPath)
const entry = measure(entryContents)

const bundlePaths = await listFiles(distRoot)
const retiredOnnxPaths = (
  await Promise.all(
    bundlePaths.map(async path => {
      if (hasRetiredOnnxArtifactName(path)) return path
      if (!isJavaScriptArtifact(path)) return null
      const source = await readJavaScriptArtifact(path)
      return hasOnnxRuntimeSourceSignature(source) ? path : null
    }),
  )
)
  .filter((path): path is string => path !== null)
  .sort((left, right) => left.localeCompare(right))

const report = {
  entryJavascript: reportMeasurement(relative(root, entryPath), entry, budgets.entryJavascript),
  retiredOnnxAssets: retiredOnnxPaths.map(path => relative(root, path).split(sep).join('/')),
}

console.log(JSON.stringify(report, null, 2))

const failures = [
  budgetFailure('entry JavaScript (raw)', entry.rawBytes, budgets.entryJavascript.rawBytes),
  budgetFailure('entry JavaScript (gzip)', entry.gzipBytes, budgets.entryJavascript.gzipBytes),
  retiredOnnxPaths.length > 0
    ? `retired ONNX artifacts are still shipped: ${retiredOnnxPaths
        .map(path => relative(distRoot, path).split(sep).join('/'))
        .join(', ')}`
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

function hasRetiredOnnxArtifactName(path: string) {
  const normalizedPath = path.toLowerCase().replaceAll('\\', '/')
  const pathParts = normalizedPath.split('/').filter(Boolean)
  const compressedName = basename(normalizedPath)
  const name = compressedName.replace(/\.(?:br|gz)$/i, '')
  const retiredRuntimeDirectory = pathParts
    .slice(0, -1)
    .some(part => /^(?:onnxruntime(?:-web)?|ort-wasm)(?:[._-]|$)/.test(part))
  if (retiredRuntimeDirectory) return true
  if (/\.(?:onnx|ort)(?:[._-]|$)/.test(name)) return true
  if (/(?:^|[._-])onnx(?:[._-]|$)/.test(name)) return true
  if (/(?:^|[._-])onnxruntime(?:-web)?(?:[._-]|$)/.test(name)) return true
  if (/(?:^|[._-])ort(?:[._-]|$)/.test(name) && /\.(?:wasm|[cm]?js|json|data|bin)(?:\.map)?$/.test(name)) {
    return true
  }
  return name === 'wurmkickflip_policy.meta.json'
}

function isJavaScriptArtifact(path: string) {
  return /\.(?:cjs|js|mjs)(?:\.map)?(?:\.(?:br|gz))?$/i.test(path)
}

async function readJavaScriptArtifact(path: string) {
  const contents = await readFile(path)
  if (/\.br$/i.test(path)) return brotliDecompressSync(contents).toString('utf8')
  if (/\.gz$/i.test(path)) return gunzipSync(contents).toString('utf8')
  return contents.toString('utf8')
}

function hasOnnxRuntimeSourceSignature(source: string) {
  return [
    /\bonnxruntime(?:-web|\/web|\.wasm)\b/i,
    /\bort-wasm(?:[._-](?:simd|threaded|jsep)|\.wasm)/i,
    /\bort\.(?:env|InferenceSession|Tensor)\b/,
    /\bInferenceSession\.create\s*\(/,
  ].some(signature => signature.test(source))
}

function verifyRetiredOnnxDetector() {
  const namedCases = [
    ['models/wurmkickflip_policy.onnx', true],
    ['assets/ort-wasm-simd-threaded.wasm', true],
    ['assets/ort-wasm-simd-threaded.wasm.br', true],
    ['assets/onnxruntime-web.min.js', true],
    ['assets/onnxruntime/model.wasm', true],
    ['models/wurmkickflip_policy.meta.json', true],
    ['assets/important-support.js', false],
    ['assets/orthography/model.wasm', false],
    ['models/wurmkickflip_locomotion_policy.json', false],
  ] as const
  for (const [path, expected] of namedCases) {
    const actual = hasRetiredOnnxArtifactName(path)
    if (actual !== expected) {
      throw new Error(`Retired ONNX filename detector misclassified ${path}.`)
    }
  }

  const javascriptCases = [
    ['assets/index.js', true],
    ['assets/index.js.br', true],
    ['assets/chunk.mjs.map.gz', true],
    ['assets/model.wasm.br', false],
  ] as const
  for (const [path, expected] of javascriptCases) {
    if (isJavaScriptArtifact(path) !== expected) {
      throw new Error(`JavaScript bundle artifact detector misclassified ${path}.`)
    }
  }

  const sourceCases = [
    ['const runtime = await import("onnxruntime-web")', true],
    ['await ort.InferenceSession.create(model)', true],
    ['const support = Math.max(0, contactRatio)', false],
  ] as const
  for (const [source, expected] of sourceCases) {
    const actual = hasOnnxRuntimeSourceSignature(source)
    if (actual !== expected) {
      throw new Error('Retired ONNX JavaScript signature detector failed its self-check.')
    }
  }
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
