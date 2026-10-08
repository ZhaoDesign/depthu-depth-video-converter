const TRANSFORMERS_MODULE = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm";
const FFMPEG_MODULE = "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/+esm";
const FFMPEG_UTIL_MODULE = "https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.2/+esm";
const FFMPEG_CORE_BASE = "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm";
const FFMPEG_WORKER_MODULE = "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/esm/worker.js";
const FFMPEG_CONST_MODULE = "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/esm/const.js";
const FFMPEG_ERRORS_MODULE = "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/esm/errors.js";
const LOCAL_AGENT_ORIGIN = "http://127.0.0.1:9876";
const MAX_OUTPUT_FPS = 60;
const SPEED_ANALYSIS_FPS = 12;
const SPEED_OUTPUT_FPS = 24;
const SPEED_OUTPUT_HEIGHT = 720;
const MODEL_CONFIGS = {
  small: { name: "Small", id: "onnx-community/depth-anything-v2-small", size: "约 99 MB" },
  base: { name: "Base", id: "onnx-community/depth-anything-v2-base", size: "约 1.16 GB" },
  large: { name: "Large", id: "onnx-community/depth-anything-v2-large", size: "约 3.9 GB" },
};
const MODEL_STORAGE_DB = "depthu-local-storage";
const MODEL_STORAGE_STORE = "settings";
const MODEL_STORAGE_KEY = "model-directory-handle";

const state = {
  file: null,
  objectUrl: null,
  resolution: "original",
  processingMode: "quality",
  converting: false,
  estimatorPromise: null,
  backend: null,
  depthReady: false,
  outputBlob: null,
  outputMime: "image/png",
  outputCodec: "",
  outputFps: 0,
  sourceFps: 0,
  outputUrl: null,
  outputFilename: "",
  outputKind: null,
  localOutputPath: "",
  outputWriteError: "",
  outputDirectoryHandle: null,
  outputFileHandle: null,
  modelDirectoryHandle: null,
  modelCache: null,
  modelPrepared: {},
  estimatorModel: "",
  modelDtype: "",
  modelPrepareTaskId: null,
  browserVideoEngine: null,
  browserVideoEnginePromise: null,
  browserVideoEngineReady: false,
  browserVideoSourceFps: 0,
  localAgentAvailable: false,
  localDevice: "",
  localModels: {},
  localMode: false,
  ffmpegAvailable: false,
  lastLog: "",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
const chooseButton = $("#chooseButton");
const sourcePreview = $("#sourcePreview");
const sourceVideo = $("#sourceVideo");
const sourceImage = $("#sourceImage");
const sourceName = $("#sourceName");
const sourceMeta = $("#sourceMeta");
const sourceType = $("#sourceType");
const fileState = $("#fileState");
const convertButton = $("#convertButton");
const convertLabel = $("#convertLabel");
const resultEmpty = $("#resultEmpty");
const resultReady = $("#resultReady");
const resultState = $("#resultState");
const resultName = $("#resultName");
const resultMeta = $("#resultMeta");
const exportButton = $("#exportButton");
const exportLabel = $("#exportLabel");
const depthPreview = $("#depthPreview");
const depthCanvas = $("#depthCanvas");
const outputVideo = $("#outputVideo");
const progressWrap = $("#progressWrap");
const progressBar = $("#progressBar");
const progressLabel = $("#progressLabel");
const progressPercent = $("#progressPercent");
const conversionLog = $("#conversionLog");
const conversionLogBox = $("#conversionLogBox");
const clearLogButton = $("#clearLogButton");
const modelRuntimeState = $("#modelRuntimeState");
const modelRuntimeText = modelRuntimeState.querySelector("span:last-child");
const modelDownload = $("#modelDownload");
const modelDownloadBar = $("#modelDownloadBar");
const modelDownloadLabel = $("#modelDownloadLabel");
const modelDownloadPercent = $("#modelDownloadPercent");
const prepareModelButton = $("#prepareModelButton");
const resolutionHint = $("#resolutionHint");
const processingModeControl = $("#processingModeControl");
const processingModeHint = $("#processingModeHint");
const sourceFpsValue = $("#sourceFpsValue");
const sourceFpsHint = $("#sourceFpsHint");
const smoothRange = $("#smoothRange");
const smoothValue = $("#smoothValue");
const modelSelect = $("#modelSelect");
const modelSelectWrap = $(".model-select-wrap");
const modelSelectTrigger = $("#modelSelectTrigger");
const modelSelectTriggerName = $("#modelSelectTriggerName");
const modelSelectTriggerMeta = $("#modelSelectTriggerMeta");
const modelSelectMenu = $("#modelSelectMenu");
const modelSelectOptions = () => $$(".model-select-option");
const audioToggle = $("#audioToggle");
const audioToggleHint = $("#audioToggleHint");
const outputLocationLabel = $("#outputLocationLabel");
const chooseOutputButton = $("#chooseOutputButton");
const modelStorage = $("#modelStorage");
const modelStorageMode = $("#modelStorageMode");
const modelStorageLabel = $("#modelStorageLabel");
const modelStorageHint = $("#modelStorageHint");
const chooseModelDirectoryButton = $("#chooseModelDirectoryButton");
const toast = $("#toast");

function selectedModelConfig(value = modelSelect?.value) {
  return MODEL_CONFIGS[value] || MODEL_CONFIGS.small;
}

function splitModelLabel(label) {
  const [name, ...rest] = String(label || "").split(" — ");
  return {
    name: name || "Small",
    meta: rest.join(" — ") || "浏览器本地",
  };
}

function setModelPickerOpen(open) {
  if (!modelSelectWrap || !modelSelectTrigger || !modelSelectMenu) return;
  modelSelectWrap.classList.toggle("is-open", open);
  modelSelectMenu.classList.toggle("hidden", !open);
  modelSelectTrigger.setAttribute("aria-expanded", String(open));
}

function syncModelPicker() {
  if (!modelSelect || !modelSelectTrigger) return;
  const option = modelSelect.options[modelSelect.selectedIndex];
  const label = splitModelLabel(option?.textContent);
  if (modelSelectTriggerName) modelSelectTriggerName.textContent = label.name;
  if (modelSelectTriggerMeta) modelSelectTriggerMeta.textContent = label.meta;
  modelSelectOptions().forEach((optionButton) => {
    const selected = optionButton.dataset.modelValue === modelSelect.value;
    optionButton.classList.toggle("is-selected", selected);
    optionButton.setAttribute("aria-selected", String(selected));
  });
}

function chooseModelValue(value) {
  if (!modelSelect || !["small", "base", "large"].includes(value)) return;
  modelSelect.value = value;
  modelSelect.dispatchEvent(new Event("change", { bubbles: true }));
  setModelPickerOpen(false);
  modelSelectTrigger?.focus();
}

function parseRateToken(value) {
  const [numerator, denominator] = String(value || "").split("/").map(Number);
  if (!Number.isFinite(numerator) || numerator <= 0) return 0;
  return Number.isFinite(denominator) && denominator > 0 ? numerator / denominator : numerator;
}

function parseVideoFpsFromLog(message) {
  const text = String(message || "");
  // FFmpeg progress lines contain values such as `frame=1 fps=1`.
  // They describe the current processing speed, not the source stream rate.
  if (/\bframe\s*=\s*\d+/i.test(text)) return 0;
  const match = text.match(/(?:,|\s)(\d+(?:\.\d+)?(?:\/\d+(?:\.\d+)?)?)\s+(?:fps|tbr)\b/i);
  return match ? parseRateToken(match[1]) : 0;
}

async function createSameOriginFfmpegWorkerURL() {
  const [workerResponse, constResponse, errorsResponse] = await Promise.all([
    fetch(FFMPEG_WORKER_MODULE),
    fetch(FFMPEG_CONST_MODULE),
    fetch(FFMPEG_ERRORS_MODULE),
  ]);
  if (!workerResponse.ok || !constResponse.ok || !errorsResponse.ok) {
    throw new Error("浏览器视频引擎 Worker 模块下载失败");
  }
  const [workerSource, constSource, errorsSource] = await Promise.all([
    workerResponse.text(),
    constResponse.text(),
    errorsResponse.text(),
  ]);
  const workerWithoutRelativeImports = workerSource.replace(
    /import\s+\{[^}]+\}\s+from\s+["']\.\/(?:const|errors)\.js["'];?\s*/g,
    "",
  );
  return URL.createObjectURL(new Blob([
    `${constSource}\n${errorsSource}\n${workerWithoutRelativeImports}`,
  ], { type: "text/javascript" }));
}

async function loadBrowserVideoEngine() {
  if (state.browserVideoEngine) return state.browserVideoEngine;
  if (state.browserVideoEnginePromise) return state.browserVideoEnginePromise;

  state.browserVideoEnginePromise = (async () => {
    setEngineState("正在下载浏览器视频引擎 · FFmpeg WASM");
    const [{ FFmpeg }, { toBlobURL }] = await Promise.all([
      import(FFMPEG_MODULE),
      import(FFMPEG_UTIL_MODULE),
    ]);
    if (!FFmpeg || !toBlobURL) throw new Error("浏览器视频引擎模块加载失败");
    const ffmpeg = new FFmpeg();
    ffmpeg.on("log", ({ message }) => {
      const fps = parseVideoFpsFromLog(message);
      if (fps > 0) state.browserVideoSourceFps = fps;
    });
    const coreURL = await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.js`, "text/javascript");
    const wasmURL = await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.wasm`, "application/wasm");
    const classWorkerURL = await createSameOriginFfmpegWorkerURL();
    try {
      await ffmpeg.load({ coreURL, wasmURL, classWorkerURL });
    } catch (error) {
      URL.revokeObjectURL(classWorkerURL);
      throw error;
    }
    state.browserVideoEngine = ffmpeg;
    state.browserVideoEngineReady = true;
    setEngineState("浏览器视频模式 · FFmpeg WASM 已就绪");
    updateConversionAvailability();
    return ffmpeg;
  })().catch((error) => {
    state.browserVideoEnginePromise = null;
    state.browserVideoEngine = null;
    state.browserVideoEngineReady = false;
    setEngineState("浏览器本地模式 · 视频引擎尚未准备");
    throw error;
  });
  return state.browserVideoEnginePromise;
}

function waitForVideoMetadata() {
  if (sourceVideo.readyState >= 1 && sourceVideo.videoWidth > 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const onLoaded = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error("视频元数据读取失败")); };
    const cleanup = () => {
      sourceVideo.removeEventListener("loadedmetadata", onLoaded);
      sourceVideo.removeEventListener("error", onError);
    };
    sourceVideo.addEventListener("loadedmetadata", onLoaded, { once: true });
    sourceVideo.addEventListener("error", onError, { once: true });
  });
}

async function canvasPngBytes(canvas) {
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("深度帧编码失败");
  return new Uint8Array(await blob.arrayBuffer());
}

async function listVideoFrames(ffmpeg, prefix) {
  const entries = await ffmpeg.listDir("/");
  return entries
    .map((entry) => entry.name)
    .filter((name) => new RegExp(`^${prefix}_[0-9]+\\.png$`).test(name))
    .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
}

async function deleteFfmpegFile(ffmpeg, filename) {
  try { await ffmpeg.deleteFile(filename); } catch {}
}

function modelDirectorySupported() {
  return typeof window.showDirectoryPicker === "function";
}

function requestKey(request) {
  if (typeof request === "string") return request;
  if (request instanceof URL) return request.toString();
  return request?.url || String(request || "");
}

async function stableKey(value) {
  if (window.crypto?.subtle) {
    const bytes = await window.crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
    return [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0).toString(16);
}

class FileSystemModelCache {
  constructor(directoryHandle) {
    this.directoryHandle = directoryHandle;
    this.indexFilename = ".depthu-model-cache.json";
    this.index = null;
    this.writeQueue = Promise.resolve();
  }

  async readIndex() {
    if (this.index) return this.index;
    try {
      const handle = await this.directoryHandle.getFileHandle(this.indexFilename);
      const file = await handle.getFile();
      const parsed = JSON.parse(await file.text());
      this.index = parsed && parsed.entries ? parsed : { version: 1, entries: {} };
    } catch {
      this.index = { version: 1, entries: {} };
    }
    return this.index;
  }

  async writeIndex() {
    const handle = await this.directoryHandle.getFileHandle(this.indexFilename, { create: true });
    const writable = await handle.createWritable();
    await writable.write(JSON.stringify(this.index));
    await writable.close();
  }

  async match(request) {
    const key = requestKey(request);
    if (!key) return undefined;
    const index = await this.readIndex();
    const entry = index.entries[key];
    if (!entry) return undefined;
    try {
      const fileHandle = await this.directoryHandle.getFileHandle(entry.filename);
      const file = await fileHandle.getFile();
      return new Response(file, {
        status: entry.status || 200,
        headers: entry.headers || {},
      });
    } catch {
      delete index.entries[key];
      await this.writeIndex().catch(() => {});
      return undefined;
    }
  }

  async put(request, response) {
    const key = requestKey(request);
    if (!key || !response) return;
    const body = await response.clone().arrayBuffer();
    const filename = `depthu-cache-${await stableKey(key)}.bin`;
    const writeJob = this.writeQueue.catch(() => {}).then(async () => {
      const index = await this.readIndex();
      const fileHandle = await this.directoryHandle.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(body);
      await writable.close();
      index.entries[key] = {
        filename,
        status: response.status || 200,
        headers: Object.fromEntries(response.headers.entries()),
      };
      await this.writeIndex();
    });
    this.writeQueue = writeJob;
    await writeJob;
  }
}

function openModelStorageDb() {
  if (!window.indexedDB) return Promise.resolve(null);
  return new Promise((resolve) => {
    const request = window.indexedDB.open(MODEL_STORAGE_DB, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(MODEL_STORAGE_STORE)) request.result.createObjectStore(MODEL_STORAGE_STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
  });
}

async function readStoredModelDirectory() {
  const db = await openModelStorageDb();
  if (!db) return null;
  return new Promise((resolve) => {
    const transaction = db.transaction(MODEL_STORAGE_STORE, "readonly");
    const request = transaction.objectStore(MODEL_STORAGE_STORE).get(MODEL_STORAGE_KEY);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => resolve(null);
  });
}

async function storeModelDirectory(handle) {
  const db = await openModelStorageDb();
  if (!db) return false;
  return new Promise((resolve) => {
    const transaction = db.transaction(MODEL_STORAGE_STORE, "readwrite");
    transaction.objectStore(MODEL_STORAGE_STORE).put(handle, MODEL_STORAGE_KEY);
    transaction.oncomplete = () => resolve(true);
    transaction.onerror = () => resolve(false);
  });
}

async function hasModelDirectoryPermission(handle, request = false) {
  if (!handle) return false;
  try {
    let permission = await handle.queryPermission?.({ mode: "readwrite" });
    if (permission !== "granted" && request && handle.requestPermission) {
      permission = await handle.requestPermission({ mode: "readwrite" });
    }
    return permission === "granted";
  } catch {
    return false;
  }
}

function updateModelStorageUI() {
  if (!modelStorage || !modelStorageMode || !modelStorageLabel || !chooseModelDirectoryButton) return;
  const hasFolder = Boolean(state.modelDirectoryHandle);
  const localOnly = state.localMode;
  modelStorage.classList.toggle("is-disabled", localOnly);
  chooseModelDirectoryButton.disabled = localOnly;
  if (localOnly) {
    modelStorageMode.textContent = "本机服务";
    modelStorageLabel.textContent = "由本机专业服务管理模型文件";
    chooseModelDirectoryButton.textContent = "不可用";
    if (modelStorageHint) modelStorageHint.textContent = "浏览器选择的文件夹只用于浏览器端模型，不影响本机服务。";
    return;
  }
  modelStorageMode.textContent = hasFolder ? "本地文件夹" : "浏览器缓存";
  modelStorageLabel.textContent = hasFolder
    ? `已选择文件夹 · ${state.modelDirectoryHandle.name}`
    : "由浏览器管理缓存位置";
  chooseModelDirectoryButton.textContent = hasFolder ? "更换文件夹" : "选择文件夹";
  if (modelStorageHint) {
    modelStorageHint.textContent = !modelDirectorySupported()
      ? "当前浏览器不支持选择文件夹，将使用浏览器缓存。"
      : (hasFolder ? "模型文件会保存到这个文件夹；下次打开可能需要重新授权。" : "Chrome / Edge 桌面版可选择 D 盘或其他本地文件夹。");
  }
}

async function restoreModelDirectory() {
  const handle = await readStoredModelDirectory();
  if (handle && await hasModelDirectoryPermission(handle)) state.modelDirectoryHandle = handle;
  updateModelStorageUI();
}

function formatSize(bytes) {
  if (!Number.isFinite(bytes)) return "未知大小";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function sanitizeFilenamePart(value, fallback = "depth-output") {
  const sanitized = String(value || fallback)
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, "_")
    .replace(/^\.+/g, "")
    .replace(/[. ]+$/g, "")
    .trim();
  return sanitized || fallback;
}

function sanitizeOutputFilename(value) {
  const match = String(value || "").match(/^(.*?)(\.[^.]*)?$/);
  const stem = sanitizeFilenamePart(match?.[1] || "depth-output");
  const extension = (match?.[2] || "").replace(/[<>:"/\\|?*\u0000-\u001F]/g, "");
  return `${stem}${extension}`;
}

function sourceFilenameStem() {
  return sanitizeFilenamePart((state.file?.name || "depth-output").replace(/\.[^/.]+$/, ""));
}

function formatType(file) {
  if (isImageFile(file)) return "IMAGE";
  return "VIDEO";
}

function isImageFile(file) {
  return Boolean(file?.type?.startsWith("image/") || /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file?.name || ""));
}

function notify(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  window.clearTimeout(notify.timer);
  notify.timer = window.setTimeout(() => toast.classList.remove("visible"), 3200);
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[character]));
}

function setEngineState(text) {
  const engineState = $(".engine-state");
  if (engineState) engineState.innerHTML = `<span class="status-dot"></span>${escapeHtml(text)}`;
}

function setModelState(text, status = "idle") {
  modelRuntimeText.textContent = text;
  modelRuntimeState.classList.toggle("is-ready", status === "ready");
  modelRuntimeState.classList.toggle("is-error", status === "error");
  if (status === "ready") {
    const label = state.localMode
      ? `本机专业模式 · ${state.localDevice || "本机服务已连接"}`
      : `${state.backend === "webgpu" ? "WebGPU" : "WASM"} 模型已就绪${state.modelDtype ? ` · ${state.modelDtype}` : ""}`;
    setEngineState(label);
  }
}

function updateModelProgress(info = {}) {
  modelDownload.classList.remove("hidden");
  const value = Number.isFinite(info.progress) ? Math.max(0, Math.min(100, Math.round(info.progress))) : null;
  const file = typeof info.file === "string" ? info.file.split("/").pop() : "模型文件";
  modelDownload.classList.toggle("is-loading", info.status === "progress" || info.status === "initiate");
  if (value !== null) {
    modelDownloadBar.style.width = `${value}%`;
    modelDownloadPercent.textContent = `${value}%`;
  }
  if (info.status === "done") {
    modelDownloadLabel.textContent = "模型文件已缓存";
    prepareModelButton.textContent = "模型已准备，可提前重新检查";
    prepareModelButton.disabled = false;
  }
  else if (info.status === "error") {
    modelDownloadLabel.textContent = "模型准备失败";
    prepareModelButton.textContent = "重试准备模型";
    prepareModelButton.disabled = false;
  }
  else if (info.status === "initiate") modelDownloadLabel.textContent = `准备 ${file}`;
  else if (info.status === "progress" || value !== null) modelDownloadLabel.textContent = `下载 ${file}`;
  else modelDownloadLabel.textContent = "转换时先下载所选模型";
}

async function localFetch(path, options = {}) {
  const { timeoutMs = 1600, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${LOCAL_AGENT_ORIGIN}${path}`, {
      ...fetchOptions,
      cache: "no-store",
      signal: controller.signal,
    });
  } catch (error) {
    error.isLocalTransport = true;
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function localModelEntry(value) {
  const prefix = String(value || "").toLowerCase();
  return Object.entries(state.localModels).find(([label]) => label.toLowerCase().startsWith(prefix));
}

function resolveLocalModelLabel(value) {
  const entry = localModelEntry(value);
  if (entry) return entry[0];
  return {
    small: "Small (fastest, ~99 MB)",
    base: "Base (balanced, ~392 MB)",
    large: "Large (best quality, ~1.3 GB)",
  }[value] || "Small (fastest, ~99 MB)";
}

function setSourceFpsInfo(fps = 0, outputFps = 0) {
  state.sourceFps = Number.isFinite(fps) && fps > 0 ? fps : 0;
  state.outputFps = Number.isFinite(outputFps) && outputFps > 0 ? outputFps : 0;
  if (!sourceFpsValue || !sourceFpsHint) return;
  if (state.sourceFps > 0) {
    const sourceText = `${state.sourceFps.toFixed(state.sourceFps % 1 ? 2 : 0)} FPS`;
    if (state.processingMode === "speed" && state.file && !isImageFile(state.file)) {
      const analyzedFps = Math.min(SPEED_ANALYSIS_FPS, state.sourceFps);
      const outputText = state.outputFps > 0
        ? `速度优先 · ${state.outputFps.toFixed(state.outputFps % 1 ? 2 : 0)} FPS 输出 · ${analyzedFps.toFixed(analyzedFps % 1 ? 2 : 0)} FPS 分析`
        : `速度优先 · 最高 ${SPEED_OUTPUT_FPS} FPS 输出 · ${analyzedFps.toFixed(analyzedFps % 1 ? 2 : 0)} FPS 分析`;
      sourceFpsValue.textContent = sourceText;
      sourceFpsHint.querySelector("span:last-child").textContent = outputText;
      return;
    }
    const outputText = state.outputFps > 0 && Math.abs(state.outputFps - state.sourceFps) > 0.01
      ? `输出限制为 ${state.outputFps.toFixed(state.outputFps % 1 ? 2 : 0)} FPS`
      : "输出保持原视频帧率";
    sourceFpsValue.textContent = sourceText;
    sourceFpsHint.querySelector("span:last-child").textContent = `${outputText} · 最高 ${MAX_OUTPUT_FPS} FPS`;
  } else {
    sourceFpsValue.textContent = "跟随原视频帧率";
    sourceFpsHint.querySelector("span:last-child").textContent = state.processingMode === "speed" && state.file && !isImageFile(state.file)
      ? `速度优先 · 最多 ${SPEED_ANALYSIS_FPS} FPS 分析 · 输出按时长补齐`
      : `最高 ${MAX_OUTPUT_FPS} FPS · 不插入额外帧`;
  }
}

function updateProcessingModeUI() {
  const isVideo = Boolean(state.file && !isImageFile(state.file));
  const isSpeed = state.processingMode === "speed";
  processingModeControl?.querySelectorAll("[data-processing-mode]").forEach((button) => {
    const active = button.dataset.processingMode === state.processingMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = state.converting;
  });
  if (processingModeHint) {
    const hint = isSpeed
      ? (isVideo ? `速度优先 · ${SPEED_OUTPUT_HEIGHT}p · ${SPEED_ANALYSIS_FPS} FPS 分析 · 输出补齐到 ${SPEED_OUTPUT_FPS} FPS · 声音按开关选择` : "速度优先仅应用于视频，图片仍按原参数生成")
      : "原始分辨率 · 原视频帧率 · 声音按开关选择";
    processingModeHint.querySelector("span:last-child").textContent = hint;
  }
  if (resolutionHint) {
    const selectedLabel = state.resolution === "original" ? "原始分辨率" : `${state.resolution}p`;
    resolutionHint.querySelector("span:last-child").textContent = isSpeed && isVideo
      ? `速度优先固定输出：${SPEED_OUTPUT_HEIGHT}p`
      : `当前输出：${selectedLabel}`;
  }
  setSourceFpsInfo(state.sourceFps, state.outputFps);
}

function setProcessingMode(mode) {
  if (!["quality", "speed"].includes(mode)) return;
  state.processingMode = mode;
  if (!state.converting) resetResult();
  updateProcessingModeUI();
  updateConversionAvailability();
  notify(mode === "speed" ? "已切换速度优先：视频将按 720p / 12 FPS 分析" : "已切换质量优先：视频保持原分辨率与帧率");
}

function updateOutputLocationLabel() {
  if (!outputLocationLabel) return;
  outputLocationLabel.textContent = state.outputFileHandle
    ? `已选择文件 · ${state.outputFileHandle.name}`
    : state.outputDirectoryHandle
    ? `已选择文件夹 · ${state.outputDirectoryHandle.name}`
    : (state.localMode ? "选择后保存到指定文件" : "选择后保存到浏览器文件");
}

function updateModelOptionLabels() {
  [...modelSelect.options].forEach((option) => {
    const config = MODEL_CONFIGS[option.value];
    if (!config) return;
    const entry = localModelEntry(option.value);
    const cached = entry?.[1]?.downloaded;
    const prepared = Boolean(state.modelPrepared[option.value]);
    option.textContent = state.localMode
      ? `${config.name} — 本机专业模式${cached ? " · 已缓存" : " · 按需下载"}`
      : `${config.name} — 浏览器本地 · ${config.size}${prepared ? " · 已准备" : ""}`;
  });
  modelSelectOptions().forEach((optionButton) => {
    const nativeOption = modelSelect.options.namedItem(optionButton.dataset.modelValue)
      || [...modelSelect.options].find((option) => option.value === optionButton.dataset.modelValue);
    const config = MODEL_CONFIGS[optionButton.dataset.modelValue];
    const label = splitModelLabel(nativeOption?.textContent);
    const name = optionButton.querySelector(".model-option-name");
    const meta = optionButton.querySelector(".model-option-meta");
    if (name) name.textContent = config?.name || label.name;
    if (meta) meta.textContent = label.meta;
  });
  syncModelPicker();
}

function updateConversionAvailability() {
  if (!convertButton) return;
  const hasFile = Boolean(state.file);
  const isImage = hasFile && isImageFile(state.file);
  // A detected local agent must not disable the browser fallback when it is
  // missing FFmpeg. This was the reason the button stayed disabled after a
  // video was selected on machines with a partial local-service install.
  const browserFallbackAllowed = !state.localMode || !state.ffmpegAvailable;
  const browserImageAllowed = browserFallbackAllowed && isImage;
  const browserVideoAllowed = browserFallbackAllowed && hasFile && !isImage;
  const professionalAllowed = state.localMode && state.localAgentAvailable && state.ffmpegAvailable;
  const allowed = hasFile && (professionalAllowed || browserImageAllowed || browserVideoAllowed) && !state.converting;
  convertButton.disabled = !allowed;
  if (state.converting) return;
  if (!hasFile) convertLabel.textContent = "等待导入素材";
  else if (state.localMode && state.localAgentAvailable && !state.ffmpegAvailable) convertLabel.textContent = isImage ? "浏览器模式生成图片深度图" : "浏览器模式转换视频";
  else if (professionalAllowed) convertLabel.textContent = "开始转换";
  else if (!state.localMode && isImage) convertLabel.textContent = "下载模型并生成图片深度图";
  else if (!state.localMode && state.browserVideoEngineReady) convertLabel.textContent = "开始转换视频";
  else if (!state.localMode) convertLabel.textContent = "下载模型并转换视频";
  else convertLabel.textContent = "视频需本机专业服务后转换";
}

function setMediaDependentControls() {
  const hasVideo = Boolean(state.file && !isImageFile(state.file));
  const browserVideo = hasVideo && !state.localMode;
  audioToggle.disabled = !hasVideo;
  if (!hasVideo) audioToggle.checked = false;
  if (audioToggleHint) {
    audioToggleHint.textContent = browserVideo
      ? "默认静音，可按开关保留原始 AAC 音频"
      : (hasVideo ? "默认静音，可按开关保留原始 AAC 音频" : "仅视频素材可保留原声");
  }
  audioToggle.title = "";
  audioToggle.closest(".toggle-row")?.classList.toggle("is-unavailable", audioToggle.disabled);
  if (hasVideo) setSourceFpsInfo();
  updateProcessingModeUI();
}

function applyProfessionalMode(capabilities) {
  state.localMode = true;
  state.localAgentAvailable = true;
  state.localDevice = capabilities.device || capabilities.device_type || "本机服务";

  const selectedValue = ["small", "base", "large"].includes(modelSelect.value) ? modelSelect.value : "small";
  state.ffmpegAvailable = capabilities.ffmpeg !== false;
  if (!state.ffmpegAvailable) {
    // Keep the site usable when the local service is present but cannot
    // encode video. Browser-local inference can still handle both images and
    // videos, including the audio toggle in the browser FFmpeg path.
    applyBrowserMode();
    setEngineState("浏览器本地模式 · 本机 FFmpeg 未就绪，已自动回退");
    setModelState("浏览器本地模型 · 本机服务缺少 FFmpeg", "idle");
    return;
  }
  updateModelOptionLabels();
  modelSelect.value = selectedValue;
  syncModelPicker();
  updateModelStorageUI();

  modelDownload.classList.remove("hidden");
  const selected = localModelEntry(modelSelect.value);
  const selectedCached = Boolean(selected?.[1]?.downloaded);
  modelDownloadLabel.textContent = selectedCached ? "所选模型已缓存" : "转换时先下载所选模型";
  modelDownloadPercent.textContent = selectedCached ? "100%" : "按需";
  modelDownloadBar.style.width = selectedCached ? "100%" : "0%";
  prepareModelButton.textContent = selectedCached ? "检查 / 准备所选模型" : "提前准备所选模型";
  prepareModelButton.disabled = false;

  setMediaDependentControls();
  updateOutputLocationLabel();
  setModelState(`本机服务已连接 · ${state.localDevice} · 所选模型按需下载`, "ready");
  setEngineState(`本机专业模式 · ${state.localDevice}${state.ffmpegAvailable ? " · FFmpeg 已就绪" : " · FFmpeg 未就绪"}`);
  updateConversionAvailability();
}

function applyBrowserMode() {
  state.localMode = false;
  state.ffmpegAvailable = false;
  updateModelOptionLabels();
  modelDownload.classList.remove("hidden");
  const config = selectedModelConfig();
  const prepared = Boolean(state.modelPrepared[modelSelect.value]);
  modelDownloadLabel.textContent = prepared ? `${config.name} 浏览器模型已准备` : "转换时下载所选模型";
  modelDownloadPercent.textContent = prepared ? "100%" : "按需";
  modelDownloadBar.style.width = prepared ? "100%" : "0%";
  prepareModelButton.disabled = false;
  prepareModelButton.textContent = prepared ? `检查 / 准备 ${config.name} 模型` : `提前准备 ${config.name} 模型`;
  setMediaDependentControls();
  updateOutputLocationLabel();
  updateModelStorageUI();
  setModelState(prepared ? `${config.name} 浏览器模型已准备` : "浏览器本地模型 · 选择后自动下载", prepared ? "ready" : "idle");
  setEngineState("浏览器本地模式 · 素材不会上传");
  updateConversionAvailability();
}

async function detectLocalAgent() {
  try {
    const [capabilitiesResponse, modelsResponse] = await Promise.all([
      localFetch("/api/capabilities"),
      localFetch("/api/models"),
    ]);
    if (!capabilitiesResponse.ok || !modelsResponse.ok) throw new Error("本机服务未就绪");
    const capabilities = await capabilitiesResponse.json();
    const modelsPayload = await modelsResponse.json();
    state.localModels = modelsPayload.models || {};
    applyProfessionalMode(capabilities);
  } catch {
    state.localAgentAvailable = false;
    applyBrowserMode();
  }
}

async function loadDepthEstimator(modelValue = modelSelect.value) {
  const config = selectedModelConfig(modelValue);
  if (state.estimatorPromise && state.estimatorModel === modelValue) return state.estimatorPromise;

  prepareModelButton.disabled = true;
  prepareModelButton.textContent = `正在准备 ${config.name} 模型…`;
  setModelState(`准备 ${config.name} 浏览器模型 · 首次运行下载`, "idle");
  updateModelProgress({ status: "initiate", file: `Depth Anything V2 ${config.name}` });
  state.estimatorPromise = (async () => {
    const { env, pipeline } = await import(TRANSFORMERS_MODULE);
    env.allowRemoteModels = true;
    env.allowLocalModels = false;
    const hasCustomDirectory = Boolean(state.modelDirectoryHandle);
    if (hasCustomDirectory && !await hasModelDirectoryPermission(state.modelDirectoryHandle, true)) {
      throw new Error("模型文件夹授权已失效，请重新选择文件夹");
    }
    state.modelCache = hasCustomDirectory ? new FileSystemModelCache(state.modelDirectoryHandle) : null;
    env.useBrowserCache = !hasCustomDirectory;
    env.useFSCache = false;
    env.useCustomCache = hasCustomDirectory;
    env.customCache = state.modelCache || undefined;
    env.useWasmCache = true;
    env.backends = env.backends || {};
    env.backends.onnx = env.backends.onnx || {};
    env.backends.onnx.wasm = env.backends.onnx.wasm || {};
    env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(8, navigator.hardwareConcurrency || 1));

    const preferredBackend = navigator.gpu ? "webgpu" : "wasm";
    const progress_callback = (info) => updateModelProgress(info);
    const createEstimator = (device, dtype = "") => pipeline("depth-estimation", config.id, {
      device,
      ...(dtype ? { dtype } : {}),
      progress_callback,
    });
    const preferredDtype = preferredBackend === "webgpu" ? "fp16" : "q8";
    let estimator;
    try {
      setModelState(`${preferredBackend === "webgpu" ? "WebGPU" : "WASM"} ${preferredDtype} 正在准备 ${config.name} 模型`, "idle");
      estimator = await createEstimator(preferredBackend, preferredDtype);
      state.backend = preferredBackend;
      state.modelDtype = preferredDtype;
      state.estimatorModel = modelValue;
      state.modelPrepared[modelValue] = true;
      updateModelOptionLabels();
      setModelState(`${preferredBackend === "webgpu" ? "WebGPU" : "WASM"} ${preferredDtype} 本地推理 · ${config.name} 已准备`, "ready");
      updateModelProgress({ status: "done", progress: 100, file: `Depth Anything V2 ${config.name}` });
      updateConversionAvailability();
      return estimator;
    } catch (preferredError) {
      if (preferredBackend === "webgpu") {
        setModelState("WebGPU fp16 不可用，尝试兼容模式", "idle");
        try {
          estimator = await createEstimator("webgpu", "fp32");
          state.backend = "webgpu";
          state.modelDtype = "fp32";
        } catch {
          setModelState("WebGPU 不可用，切换 WASM q8", "idle");
          estimator = await createEstimator("wasm", "q8");
          state.backend = "wasm";
          state.modelDtype = "q8";
        }
      } else {
        setModelState("WASM q8 不可用，切换兼容模式", "idle");
        estimator = await createEstimator("wasm");
        state.backend = "wasm";
        state.modelDtype = "默认";
      }
      state.estimatorModel = modelValue;
      state.modelPrepared[modelValue] = true;
      updateModelOptionLabels();
      setModelState(`${state.backend === "webgpu" ? "WebGPU" : "WASM"} ${state.modelDtype} 本地推理 · ${config.name} 已准备`, "ready");
      updateModelProgress({ status: "done", progress: 100, file: `Depth Anything V2 ${config.name}` });
      updateConversionAvailability();
      return estimator;
    }
  })().catch((error) => {
    state.estimatorPromise = null;
    state.estimatorModel = "";
    state.modelDtype = "";
    state.modelCache = null;
    const message = error?.message || "模型加载失败";
    setModelState(`模型加载失败 · ${message.slice(0, 45)}`, "error");
    modelDownloadLabel.textContent = "模型加载失败";
    prepareModelButton.disabled = false;
    prepareModelButton.textContent = `重新下载 / 准备 ${config.name}`;
    updateConversionAvailability();
    notify("模型加载失败，请检查网络或更换 Chrome / Edge 重试");
    throw error;
  });
  return state.estimatorPromise;
}

async function pollLocalTask(taskId, progressStart = 0, progressSpan = 100, modelFile = "") {
  const encodedId = encodeURIComponent(taskId);
  for (let attempt = 0; attempt < 7200; attempt += 1) {
    const response = await localFetch(`/api/progress/${encodedId}`, { timeoutMs: 5000 });
    if (!response.ok) throw new Error(`本机任务状态读取失败（${response.status}）`);
    const payload = await response.json();
    const fraction = Math.max(0, Math.min(1, Number(payload.progress) || 0));
    const description = payload.description || "本机专业模式处理中";
    setProgress(progressStart + fraction * progressSpan, description);
    if (modelFile) updateModelProgress({ status: fraction >= 1 ? "done" : "progress", progress: fraction * 100, file: modelFile });
    if (payload.state === "error" || payload.status === "error") {
      throw new Error(payload.error || description || "本机任务失败");
    }
    if (payload.state === "done" || payload.status === "done" || payload.done === true) return payload;
    await wait(450);
  }
  throw new Error("本机任务等待超时，请检查本机服务");
}

async function prepareProfessionalModel({ silent = false } = {}) {
  if (!state.localMode || !state.localAgentAvailable) throw new Error("请先启动本机专业服务");
  const modelLabel = resolveLocalModelLabel(modelSelect.value);
  const shortName = modelSelect.options[modelSelect.selectedIndex]?.textContent?.split(" — ")[0] || "所选";
  const existing = localModelEntry(modelSelect.value);
  if (existing?.[1]?.downloaded) {
    updateModelProgress({ status: "done", progress: 100, file: shortName });
    setModelState(`本机服务已连接 · ${shortName} 模型已缓存`, "ready");
    return { cached: true };
  }

  modelDownload.classList.add("is-loading");
  prepareModelButton.disabled = true;
  prepareModelButton.textContent = `正在准备 ${shortName} 模型…`;
  updateModelProgress({ status: "initiate", progress: 0, file: shortName });
  setModelState(`正在下载 ${shortName} 模型 · 下载完成后开始转换`, "idle");

  const formData = new FormData();
  formData.append("model_size_label", modelLabel);
  const response = await localFetch("/api/models/prepare", {
    method: "POST",
    body: formData,
    timeoutMs: 10000,
  });
  if (response.status === 404) {
    // Older local agents download inside /api/process. Keep compatibility;
    // the current server exposes the explicit prepare task.
    prepareModelButton.disabled = false;
    if (!silent) notify("当前本机服务会在转换开始时下载模型");
    return { legacy: true };
  }
  if (!response.ok) {
    let detail = `模型准备失败（${response.status}）`;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  const payload = await response.json();
  if (payload.task_id) await pollLocalTask(payload.task_id, 0, 24, shortName);
  state.localModels[modelLabel] = { ...(state.localModels[modelLabel] || {}), downloaded: true };
  updateModelOptionLabels();
  updateModelProgress({ status: "done", progress: 100, file: shortName });
  setModelState(`本机服务已连接 · ${shortName} 模型已缓存`, "ready");
  if (!silent) notify(`${shortName} 模型已下载，下一步可开始转换`);
  return { prepared: true };
}

function resetResult() {
  if (state.outputUrl) URL.revokeObjectURL(state.outputUrl);
  state.outputUrl = null;
  state.localOutputPath = "";
  state.outputFilename = "";
  state.outputKind = null;
  state.converting = false;
  state.depthReady = false;
  state.outputBlob = null;
  state.outputMime = "image/png";
  state.outputCodec = "";
  state.outputFps = 0;
  progressWrap.classList.add("hidden");
  progressBar.style.width = "0%";
  progressPercent.textContent = "0%";
  progressLabel.textContent = "准备中";
  resultReady.classList.add("hidden");
  resultEmpty.classList.remove("hidden");
  resultState.textContent = "等待转换";
  resultState.classList.add("muted-state");
  resultState.style.color = "";
  depthPreview.classList.remove("has-real-depth");
  depthCanvas.hidden = true;
  outputVideo.pause();
  outputVideo.removeAttribute("src");
  outputVideo.load();
  outputVideo.hidden = true;
  updateExportButton();
  clearConversionLog();
  setSourceFpsInfo();
  updateConversionAvailability();
}

function setMediaVisibility(mediaType) {
  const showVideo = mediaType === "video";
  const showImage = mediaType === "image";
  sourceVideo.hidden = !showVideo;
  sourceVideo.classList.toggle("hidden", !showVideo);
  sourceImage.hidden = !showImage;
  sourceImage.classList.toggle("hidden", !showImage);
}

function clearFile() {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = null;
  state.objectUrl = null;
  state.outputDirectoryHandle = null;
  state.outputFileHandle = null;
  fileInput.value = "";
  sourceVideo.pause();
  sourceVideo.removeAttribute("src");
  sourceImage.removeAttribute("src");
  setMediaVisibility("");
  sourcePreview.classList.add("hidden");
  dropzone.classList.remove("hidden");
  fileState.textContent = "未选择";
  fileState.style.color = "";
  sourceName.removeAttribute("title");
  updateOutputLocationLabel();
  setSourceFpsInfo();
  setMediaDependentControls();
  updateConversionAvailability();
  resetResult();
  notify("当前素材已关闭，可以上传其他视频");
}

function loadFile(file) {
  if (!file) return;
  const isAccepted = file.type.startsWith("video/") || file.type.startsWith("image/") || /\.(mkv|webm|m4v|avi)$/i.test(file.name);
  if (!isAccepted) {
    notify("请选择视频或图片文件");
    return;
  }
  if (file.size > 2 * 1024 * 1024 * 1024) {
    notify("文件大小不能超过 2 GB");
    return;
  }

  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.objectUrl = URL.createObjectURL(file);
  const isImage = isImageFile(file);
  const expectedOutputExtension = isImage ? ".png" : ".mp4";
  if (state.outputFileHandle && !state.outputFileHandle.name.toLowerCase().endsWith(expectedOutputExtension)) {
    state.outputFileHandle = null;
    updateOutputLocationLabel();
  }
  sourceType.textContent = formatType(file);
  sourceName.textContent = file.name;
  sourceName.title = file.name;
  sourceMeta.textContent = `${formatSize(file.size)} · ${isImage ? "图片素材" : "视频素材"}`;
  setMediaVisibility(isImage ? "image" : "video");
  if (isImage) sourceImage.src = state.objectUrl;
  else {
    sourceVideo.src = state.objectUrl;
    sourceVideo.load();
  }
  dropzone.classList.add("hidden");
  sourcePreview.classList.remove("hidden");
  fileState.textContent = "已选择";
  fileState.style.color = "var(--cyan)";
  // Start every newly imported media item with audio preservation disabled;
  // users can enable the switch explicitly before converting a video.
  audioToggle.checked = false;
  setMediaDependentControls();
  resetResult();
  if (!isImage) {
    sourceVideo.addEventListener("loadedmetadata", () => {
      const guessedFps = Number(sourceVideo.getAttribute("data-fps")) || 0;
      setSourceFpsInfo(guessedFps);
    }, { once: true });
  }
  updateConversionAvailability();
  if (!isImage && !state.localMode) {
    notify("视频已载入；首次转换会在浏览器中下载视频引擎与所选模型");
  } else if (!isImage && !state.localAgentAvailable) {
    notify("视频转换需要先启动本机服务：python -m server.main");
  } else {
    notify("素材已载入，可以开始本机推理");
  }
}

function updateSlider() {
  const value = Number(smoothRange.value);
  smoothValue.textContent = `${value}%`;
  smoothRange.style.background = `linear-gradient(to right, var(--cyan) 0 ${value}%, #263449 ${value}% 100%)`;
}

function setResolution(value) {
  state.resolution = value;
  const label = value === "original" ? "原始分辨率" : `${value}p`;
  $$(".segment").forEach((button) => {
    const active = button.dataset.resolution === value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (resolutionHint) {
    const isVideo = Boolean(state.file && !isImageFile(state.file));
    resolutionHint.querySelector("span:last-child").textContent = state.processingMode === "speed" && isVideo
      ? `速度优先固定输出：${SPEED_OUTPUT_HEIGHT}p`
      : `当前输出：${label}`;
  }
}

function setProgress(value, label) {
  progressBar.style.width = `${value}%`;
  progressPercent.textContent = `${Math.round(value)}%`;
  progressLabel.textContent = label;
  if (label && label !== state.lastLog && conversionLogBox) {
    state.lastLog = label;
    conversionLog.classList.remove("hidden");
    const line = document.createElement("div");
    const stamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    line.textContent = `[${stamp}] ${label}`;
    conversionLogBox.appendChild(line);
    while (conversionLogBox.children.length > 80) conversionLogBox.firstElementChild.remove();
    conversionLogBox.scrollTop = conversionLogBox.scrollHeight;
  }
}

function clearConversionLog() {
  state.lastLog = "";
  if (conversionLogBox) conversionLogBox.replaceChildren();
  conversionLog?.classList.add("hidden");
}

async function prepareInferenceInput() {
  return state.objectUrl;
}

function normalizeChannelData(raw) {
  const width = raw.width || raw.dims?.[raw.dims.length - 1];
  const height = raw.height || raw.dims?.[raw.dims.length - 2];
  const channels = raw.channels || 1;
  const data = raw.data;
  if (!width || !height || !data) throw new Error("模型没有返回可显示的深度数据");
  const sampleCount = width * height;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < sampleCount; i += Math.max(1, Math.floor(sampleCount / 4096))) {
    const value = Number(data[i * channels]);
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return { width, height, channels, data, min, max };
}

function renderDepthMap(output) {
  const { width, height } = getOutputSize(sourceImage);
  const gray = depthOutputToGray(output, width, height);
  drawGrayFrame(depthCanvas, gray, width, height);
  depthCanvas.hidden = false;
  outputVideo.hidden = true;
  depthPreview.classList.add("has-real-depth");
  state.depthReady = true;
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function getOutputSize(media) {
  const sourceWidth = media.videoWidth || media.naturalWidth || 1280;
  const sourceHeight = media.videoHeight || media.naturalHeight || 720;
  const isVideo = Boolean(state.file && !isImageFile(state.file));
  const requestedHeight = state.processingMode === "speed" && isVideo
    ? SPEED_OUTPUT_HEIGHT
    : (state.resolution === "original" ? sourceHeight : Number(state.resolution));
  const scale = Math.min(1, requestedHeight / sourceHeight);
  return {
    width: Math.max(2, Math.round((sourceWidth * scale) / 2) * 2),
    height: Math.max(2, Math.round((sourceHeight * scale) / 2) * 2),
  };
}

function getBrowserVideoPlan(sourceFps = 30) {
  const safeSourceFps = Number.isFinite(sourceFps) && sourceFps > 0 ? sourceFps : 30;
  if (state.processingMode === "speed") {
    const analysisFps = Math.min(SPEED_ANALYSIS_FPS, safeSourceFps);
    const outputFps = safeSourceFps > SPEED_ANALYSIS_FPS ? SPEED_OUTPUT_FPS : safeSourceFps;
    return {
      analysisFps,
      outputFps,
      duplicateFrames: safeSourceFps > SPEED_ANALYSIS_FPS ? 2 : 1,
    };
  }
  const outputFps = Math.min(MAX_OUTPUT_FPS, safeSourceFps);
  return { analysisFps: outputFps, outputFps, duplicateFrames: 1 };
}

function depthOutputToGray(output, targetWidth, targetHeight) {
  const raw = output?.depth || output?.predicted_depth;
  const { width, height, channels, data, min, max } = normalizeChannelData(raw);
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const sourceContext = sourceCanvas.getContext("2d", { alpha: false });
  const sourcePixels = sourceContext.createImageData(width, height);
  const range = max - min || 1;
  const shouldInvert = $("#invertToggle").checked;
  for (let i = 0; i < width * height; i += 1) {
    const rawValue = Number(data[i * channels]);
    let value = data instanceof Uint8Array || data instanceof Uint8ClampedArray ? rawValue : ((rawValue - min) / range) * 255;
    value = Math.max(0, Math.min(255, shouldInvert ? 255 - value : value));
    const offset = i * 4;
    sourcePixels.data[offset] = value;
    sourcePixels.data[offset + 1] = value;
    sourcePixels.data[offset + 2] = value;
    sourcePixels.data[offset + 3] = 255;
  }
  sourceContext.putImageData(sourcePixels, 0, 0);
  const targetCanvas = document.createElement("canvas");
  targetCanvas.width = targetWidth;
  targetCanvas.height = targetHeight;
  const targetContext = targetCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
  targetContext.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight);
  const targetPixels = targetContext.getImageData(0, 0, targetWidth, targetHeight).data;
  const gray = new Uint8ClampedArray(targetWidth * targetHeight);
  for (let i = 0; i < gray.length; i += 1) gray[i] = targetPixels[i * 4];
  return gray;
}

function drawGrayFrame(canvas, gray, width, height) {
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  const pixels = context.createImageData(width, height);
  for (let i = 0; i < gray.length; i += 1) {
    const offset = i * 4;
    pixels.data[offset] = gray[i];
    pixels.data[offset + 1] = gray[i];
    pixels.data[offset + 2] = gray[i];
    pixels.data[offset + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
}

function professionalOutputName() {
  return `${sourceFilenameStem()}_depth.${isImageFile(state.file) ? "png" : "mp4"}`;
}

async function renderProfessionalImage(blobUrl) {
  const image = new Image();
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("专业模式返回的 PNG 无法预览"));
    image.src = blobUrl;
  });
  depthCanvas.width = image.naturalWidth || image.width;
  depthCanvas.height = image.naturalHeight || image.height;
  depthCanvas.getContext("2d", { alpha: false }).drawImage(image, 0, 0);
  depthCanvas.hidden = false;
  outputVideo.hidden = true;
  depthPreview.classList.add("has-real-depth");
}

function finishBrowserVideoConversion(sourceFps, outputFps) {
  state.converting = false;
  resultEmpty.classList.add("hidden");
  resultReady.classList.remove("hidden");
  resultState.textContent = "已完成";
  resultState.classList.remove("muted-state");
  resultState.style.color = "var(--green)";
  convertLabel.textContent = "再次转换视频";
  convertButton.disabled = false;
  updateExportButton();
  resultName.textContent = state.outputFilename;
  $("#modalOutputName").textContent = state.outputFilename;
  const fpsText = outputFps > 0
    ? `${outputFps.toFixed(outputFps % 1 ? 2 : 0)} FPS`
    : "原视频帧率（最高 60 FPS）";
  const codecText = state.outputCodec ? `${state.outputCodec} MP4` : "MP4";
  const modeText = state.processingMode === "speed" ? `速度优先 · ${SPEED_OUTPUT_HEIGHT}p · ` : "";
  resultMeta.textContent = `浏览器本地 · ${modeText}${codecText} · ${fpsText} · ${audioToggle.checked ? "保留原声" : "静音"}`;
  setSourceFpsInfo(sourceFps, outputFps);
  updateConversionAvailability();
  notify(`浏览器本地 ${codecText} 深度视频已生成`);
}

async function startBrowserVideoConversion() {
  let ffmpeg;
  let inputName = "";
  let outputName = "";
  let sourceFrames = [];
  let depthFrames = [];
  try {
    await waitForVideoMetadata();
    const selectedName = modelSelect.options[modelSelect.selectedIndex]?.textContent?.split(" — ")[0] || "所选";
    setProgress(5, `准备 ${selectedName} 浏览器模型`);
    const estimator = await loadDepthEstimator();
    setProgress(22, state.browserVideoEngine
      ? "使用已就绪的浏览器视频引擎 · FFmpeg WASM"
      : "首次加载浏览器视频引擎 · FFmpeg WASM");
    ffmpeg = await loadBrowserVideoEngine();

    const fileExtension = (state.file.name.match(/\.[a-z0-9]+$/i)?.[0] || ".mp4").toLowerCase();
    const jobId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    inputName = `depthu-input-${jobId}${fileExtension}`;
    outputName = `depthu-output-${jobId}.mp4`;
    await ffmpeg.writeFile(inputName, new Uint8Array(await state.file.arrayBuffer()));

    state.browserVideoSourceFps = 0;
    try {
      await ffmpeg.exec([
        "-hide_banner", "-i", inputName, "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-",
      ]);
    } catch {}
    const sourceFps = state.browserVideoSourceFps || state.sourceFps || 30;
    const videoPlan = getBrowserVideoPlan(sourceFps);
    const outputFps = videoPlan.outputFps;
    setSourceFpsInfo(sourceFps, outputFps);
    setProgress(29, state.processingMode === "speed"
      ? `速度优先 · ${videoPlan.analysisFps.toFixed(videoPlan.analysisFps % 1 ? 2 : 0)} FPS 分析，${outputFps.toFixed(outputFps % 1 ? 2 : 0)} FPS 输出`
      : (sourceFps > MAX_OUTPUT_FPS
        ? `读取完成 · 源视频 ${sourceFps.toFixed(2)} FPS，输出封顶 ${MAX_OUTPUT_FPS} FPS`
        : `读取完成 · 保持源视频 ${sourceFps.toFixed(sourceFps % 1 ? 2 : 0)} FPS`));

    const extractArgs = ["-i", inputName, "-map", "0:v:0"];
    if (sourceFps > videoPlan.analysisFps) extractArgs.push("-vf", `fps=${videoPlan.analysisFps}`);
    extractArgs.push("-vsync", "0", "-start_number", "1", "source_%06d.png");
    await ffmpeg.exec(extractArgs);
    sourceFrames = await listVideoFrames(ffmpeg, "source");
    if (!sourceFrames.length) throw new Error("浏览器视频引擎没有提取到视频帧");

    const outputSize = getOutputSize(sourceVideo);
    let depthOutputIndex = 1;
    for (let index = 0; index < sourceFrames.length; index += 1) {
      const frameName = sourceFrames[index];
      const frameBytes = await ffmpeg.readFile(frameName);
      const frameBlob = new Blob([frameBytes], { type: "image/png" });
      const frameUrl = URL.createObjectURL(frameBlob);
      let depthOutput;
      try {
        depthOutput = await estimator(frameUrl);
      } finally {
        URL.revokeObjectURL(frameUrl);
      }
      const gray = depthOutputToGray(depthOutput, outputSize.width, outputSize.height);
      drawGrayFrame(depthCanvas, gray, outputSize.width, outputSize.height);
      const depthBytes = await canvasPngBytes(depthCanvas);
      const duplicateBytes = videoPlan.duplicateFrames > 1 ? new Uint8Array(depthBytes) : null;
      const depthName = `depth_${String(depthOutputIndex).padStart(6, "0")}.png`;
      await ffmpeg.writeFile(depthName, depthBytes);
      depthFrames.push(depthName);
      depthOutputIndex += 1;
      if (videoPlan.duplicateFrames > 1) {
        const duplicateName = `depth_${String(depthOutputIndex).padStart(6, "0")}.png`;
        await ffmpeg.writeFile(duplicateName, duplicateBytes);
        depthFrames.push(duplicateName);
        depthOutputIndex += 1;
      }
      await deleteFfmpegFile(ffmpeg, frameName);
      const progress = 34 + ((index + 1) / sourceFrames.length) * 50;
      setProgress(progress, state.processingMode === "speed"
        ? `速度优先 · 生成深度图 · ${index + 1} / ${sourceFrames.length}`
        : `逐帧生成深度图 · ${index + 1} / ${sourceFrames.length}`);
    }

    const encodePrefix = [
      "-framerate", String(outputFps),
      "-start_number", "1",
      "-i", "depth_%06d.png",
      "-i", inputName,
      "-map", "0:v:0",
    ];
    const audioArgs = audioToggle.checked
      ? ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
      : ["-an"];
    const outputSuffix = ["-shortest", "-movflags", "+faststart", outputName];
    const h264Args = [
      ...encodePrefix,
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-pix_fmt", "yuv420p",
      ...audioArgs,
      ...outputSuffix,
    ];
    setProgress(87, audioToggle.checked ? "编码 H.264 MP4 并合并原始音频" : "编码 H.264 MP4（静音输出）");
    try {
      await ffmpeg.exec(h264Args);
      state.outputCodec = "H.264";
    } catch (encodeError) {
      await deleteFfmpegFile(ffmpeg, outputName);
      const compatibleArgs = [
        ...encodePrefix,
        "-c:v", "mpeg4",
        "-q:v", "3",
        "-pix_fmt", "yuv420p",
        ...audioArgs,
        ...outputSuffix,
      ];
      setProgress(87, audioToggle.checked ? "H.264 不可用，切换兼容 MP4 并保留原声" : "H.264 不可用，切换兼容 MP4（静音输出）");
      try {
        await ffmpeg.exec(compatibleArgs);
      } catch (fallbackError) {
        const firstMessage = String(encodeError?.message || encodeError || "");
        const fallbackMessage = String(fallbackError?.message || fallbackError || "");
        throw new Error(`视频编码失败：${fallbackMessage || firstMessage}`);
      }
      state.outputCodec = "兼容编码";
    }
    const outputBytes = await ffmpeg.readFile(outputName);
    if (!outputBytes?.length) throw new Error("浏览器视频引擎没有返回 MP4 文件");
    const blob = new Blob([outputBytes], { type: "video/mp4" });
    if (state.outputUrl) URL.revokeObjectURL(state.outputUrl);
    state.outputBlob = blob;
    state.outputMime = "video/mp4";
    state.outputKind = "video";
    state.outputFps = outputFps;
    state.outputFilename = professionalOutputName();
    state.outputUrl = URL.createObjectURL(blob);
    state.localOutputPath = "";
    depthCanvas.hidden = true;
    outputVideo.src = state.outputUrl;
    outputVideo.hidden = false;
    depthPreview.classList.add("has-real-depth");
    state.depthReady = true;
    setProgress(100, `${state.outputCodec || "MP4"} 深度视频已生成`);
    finishBrowserVideoConversion(sourceFps, outputFps);
  } catch (error) {
    state.converting = false;
    resultState.textContent = "转换失败";
    resultState.style.color = "#ff9aa6";
    convertLabel.textContent = "重试浏览器视频转换";
    convertButton.disabled = false;
    setProgress(0, "浏览器视频转换失败");
    const message = typeof error === "string"
      ? error
      : error?.message || "浏览器视频转换失败";
    notify(message.includes("libx264")
      ? "当前浏览器视频引擎不包含 H.264 编码器，请改用本机专业模式"
      : `浏览器视频转换失败：${message.slice(0, 100)}`);
  } finally {
    if (ffmpeg) {
      for (const filename of [...sourceFrames, ...depthFrames, inputName, outputName]) {
        if (filename) await deleteFfmpegFile(ffmpeg, filename);
      }
    }
  }
}

async function startProfessionalConversion() {
  const imageInput = isImageFile(state.file);
  const formData = new FormData();
  formData.append("input_media", state.file, state.file.name);
  formData.append("model_size_label", resolveLocalModelLabel(modelSelect.value));
  formData.append("resolution_choice", state.resolution === "original" ? "Original" : `${state.resolution}p`);
  formData.append("invert_bw", String($("#invertToggle").checked));
  formData.append("smoothing_strength", String(Number(smoothRange.value)));
  formData.append("preserve_audio", String(!imageInput && audioToggle.checked));

  try {
    const selectedName = modelSelect.options[modelSelect.selectedIndex]?.textContent?.split(" — ")[0] || "所选";
    convertLabel.textContent = `准备 ${selectedName} 模型…`;
    setProgress(5, `准备 ${selectedName} 模型`);
    await prepareProfessionalModel({ silent: true });
    convertLabel.textContent = "本机专业处理中…";
    setProgress(27, "模型已准备，开始读取素材");
    formData.append("return_task", "true");
    const response = await localFetch("/api/process", {
      method: "POST",
      body: formData,
      timeoutMs: 12000,
    });
    if (!response.ok) {
      let detail = `本机专业服务返回错误（${response.status}）`;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch {}
      throw new Error(detail);
    }

    let resultResponse = response;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const task = await response.json();
      if (!task.task_id) throw new Error("本机服务没有返回转换任务");
      const completed = await pollLocalTask(task.task_id, 27, 69);
      resultResponse = await localFetch(`/api/result/${encodeURIComponent(task.task_id)}`, {
        timeoutMs: 30000,
      });
      if (!resultResponse.ok) throw new Error("本机服务无法读取转换结果");
      state.outputFps = Number(completed.output_fps || 0);
    }

    const blob = await resultResponse.blob();
    if (!blob.size) throw new Error("本机专业服务没有返回输出文件");
    if (!imageInput && (resultResponse.headers.get("content-type") || "").includes("webm")) {
      throw new Error("本机服务返回了 WebM，未生成桌面端要求的 MP4");
    }
    if (state.outputUrl) URL.revokeObjectURL(state.outputUrl);
    state.outputBlob = blob;
    state.outputMime = blob.type || (imageInput ? "image/png" : "video/mp4");
    state.outputKind = imageInput ? "image" : "video";
    state.outputFilename = professionalOutputName();
    state.outputUrl = URL.createObjectURL(blob);
    state.localOutputPath = resultResponse.headers.get("X-Depth-Output-Path") || "";
    if (!state.outputFps) state.outputFps = Number(resultResponse.headers.get("X-Depth-Output-FPS") || 0);

    setProgress(96, "准备输出预览");
    if (imageInput) {
      await renderProfessionalImage(state.outputUrl);
    } else {
      depthCanvas.hidden = true;
      outputVideo.src = state.outputUrl;
      outputVideo.hidden = false;
      depthPreview.classList.add("has-real-depth");
    }
    state.depthReady = true;
    setProgress(100, imageInput ? "深度 PNG 已生成" : "H.264 MP4 已生成");
    finishProfessionalConversion();
  } catch (error) {
    state.converting = false;
    resultState.textContent = "转换失败";
    resultState.style.color = "#ff9aa6";
    convertLabel.textContent = "重试本机专业模式";
    convertButton.disabled = false;
    setProgress(0, "专业模式失败");
    notify((error?.message || "本机专业模式转换失败").slice(0, 120));
  }
}

function finishProfessionalConversion() {
  const imageInput = state.outputKind === "image";
  state.converting = false;
  resultEmpty.classList.add("hidden");
  resultReady.classList.remove("hidden");
  resultState.textContent = "已完成";
  resultState.classList.remove("muted-state");
  resultState.style.color = "var(--green)";
  convertLabel.textContent = imageInput ? "再次转换图片" : "再次转换视频";
  convertButton.disabled = false;
  updateExportButton();
  resultName.textContent = state.outputFilename;
  $("#modalOutputName").textContent = state.outputFilename;
  const fpsText = state.outputFps > 0
    ? `${state.outputFps.toFixed(state.outputFps % 1 ? 2 : 0)} FPS`
    : "原视频帧率（最高 60 FPS）";
  resultMeta.textContent = imageInput
    ? `桌面端专业模式 · PNG · ${state.localDevice}`
    : `桌面端专业模式 · H.264 MP4 · ${fpsText} · ${audioToggle.checked ? "保留原声" : "静音"} · ${state.localDevice}`;
  updateConversionAvailability();
  notify(imageInput ? "专业模式深度 PNG 已生成" : "专业模式 H.264 深度视频已生成");
}

async function startConversion() {
  if (!state.file || state.converting) return;
  if (state.outputUrl) URL.revokeObjectURL(state.outputUrl);
  state.outputUrl = null;
  state.localOutputPath = "";
  state.outputFilename = "";
  state.outputKind = null;
  state.outputWriteError = "";
  state.outputBlob = null;
  state.outputMime = "image/png";
  state.outputCodec = "";
  state.depthReady = false;
  outputVideo.pause();
  outputVideo.removeAttribute("src");
  outputVideo.load();
  outputVideo.hidden = true;
  depthCanvas.hidden = true;
  state.converting = true;
  convertButton.disabled = true;
  convertLabel.textContent = "本机推理中…";
  resultReady.classList.add("hidden");
  resultEmpty.classList.remove("hidden");
  resultState.textContent = "处理中";
  resultState.classList.remove("muted-state");
  resultState.style.color = "";
  progressWrap.classList.remove("hidden");
  setProgress(3, "加载本地模型");

  if (state.localMode && state.localAgentAvailable && state.ffmpegAvailable) {
    await startProfessionalConversion();
    return;
  }

  if (!isImageFile(state.file)) {
    if (!state.localMode || !state.ffmpegAvailable) await startBrowserVideoConversion();
    else {
      state.converting = false;
      resultState.textContent = "需要本机专业服务";
      resultState.style.color = "#e8c878";
      progressWrap.classList.add("hidden");
      updateConversionAvailability();
      notify("视频必须通过本机专业服务转换，才能保持原视频帧率并输出 H.264 MP4");
    }
    return;
  }

  try {
    const estimator = await loadDepthEstimator();
    setProgress(28, "读取图片素材");
    const input = await prepareInferenceInput();
    setProgress(42, "计算深度图");
    const output = await estimator(input);
    setProgress(82, "渲染真实深度预览");
    renderDepthMap(output);
    setProgress(100, "深度图已生成");
    finishImageConversion();
  } catch (error) {
    state.converting = false;
    resultState.textContent = "转换失败";
    resultState.style.color = "#ff9aa6";
    convertLabel.textContent = "重试本机推理";
    updateConversionAvailability();
    const message = error?.message || "本地模型推理失败";
    setProgress(0, "推理失败");
    notify(message.slice(0, 100));
  }
}

function finishImageConversion() {
  state.converting = false;
  resultEmpty.classList.add("hidden");
  resultReady.classList.remove("hidden");
  resultState.textContent = "已完成";
  resultState.classList.remove("muted-state");
  resultState.style.color = "var(--green)";
  convertLabel.textContent = "再次分析首帧";
  updateConversionAvailability();
  const outputName = `${sourceFilenameStem()}_depth.png`;
  state.outputKind = "image";
  state.outputFilename = outputName;
  updateExportButton();
  resultName.textContent = outputName;
  $("#modalOutputName").textContent = outputName;
  resultMeta.textContent = isImageFile(state.file)
    ? `真实模型推理 · ${state.backend === "webgpu" ? "WebGPU" : "WASM"} · 灰度深度图`
    : "视频请使用本机专业模式输出 MP4";
  notify("真实深度图已生成");
}

function updateExportButton() {
  if (!exportButton || !exportLabel) return;
  const label = state.outputKind === "video" ? "导出深度视频" : "导出深度 PNG";
  exportLabel.textContent = label;
  exportButton.setAttribute("aria-label", label);
}

function resetControls() {
  modelSelect.value = "small";
  syncModelPicker();
  state.processingMode = "quality";
  setResolution("original");
  setMediaDependentControls();
  audioToggle.checked = false;
  smoothRange.value = "60";
  updateSlider();
  $("#invertToggle").checked = false;
  updateModelOptionLabels();
  updateProcessingModeUI();
  updateConversionAvailability();
  notify("转换参数已重置");
}

function openFullscreen() {
  if (!state.depthReady) {
    notify("请先完成一次本机推理");
    return;
  }
  const modalPreview = $(".modal-depth-preview");
  modalPreview.innerHTML = "";
  if (state.outputKind === "video" && state.outputUrl) {
    const clone = document.createElement("video");
    clone.className = "modal-output-video";
    clone.controls = true;
    clone.playsInline = true;
    clone.src = state.outputUrl;
    modalPreview.appendChild(clone);
    $("#fullscreenModal").classList.remove("hidden");
    return;
  }
  const clone = document.createElement("canvas");
  clone.width = depthCanvas.width;
  clone.height = depthCanvas.height;
  clone.className = "modal-depth-canvas";
  clone.getContext("2d").drawImage(depthCanvas, 0, 0);
  modalPreview.appendChild(clone);
  $("#fullscreenModal").classList.remove("hidden");
}

function closeFullscreen() { $("#fullscreenModal").classList.add("hidden"); }

async function chooseOutputDirectory() {
  // In Chrome, choose the final file directly. This avoids a second
  // directory-permission round trip that can be rejected for Desktop and
  // Downloads even after the user has selected them.
  if (typeof window.showSaveFilePicker === "function") {
    try {
      const isVideo = !state.file || !isImageFile(state.file);
      const handle = await window.showSaveFilePicker({
        suggestedName: `${sourceFilenameStem()}_depth.${isVideo ? "mp4" : "png"}`,
        types: [{
          description: isVideo ? "MP4 深度视频" : "PNG 深度图",
          accept: { [isVideo ? "video/mp4" : "image/png"]: [isVideo ? ".mp4" : ".png"] },
        }],
      });
      state.outputFileHandle = handle;
      state.outputDirectoryHandle = null;
      updateOutputLocationLabel();
      notify(`输出文件已选择：${handle.name}`);
    } catch (error) {
      if (error?.name !== "AbortError") notify("无法选择输出文件，请重新选择桌面或下载文件夹");
    }
    return;
  }

  if (typeof window.showDirectoryPicker !== "function") {
    notify("当前浏览器不支持选择保存位置，导出时会保存到默认下载目录");
    return;
  }
  try {
    const handle = await window.showDirectoryPicker({ mode: "readwrite" });
    if (!await hasModelDirectoryPermission(handle, true)) {
      notify("没有获得桌面文件夹的写入权限，请重新选择并允许访问");
      return;
    }
    state.outputDirectoryHandle = handle;
    state.outputFileHandle = null;
    updateOutputLocationLabel();
    notify(`输出文件夹已选择：${state.outputDirectoryHandle.name}`);
  } catch (error) {
    if (error?.name !== "AbortError") notify("无法选择输出文件夹");
  }
}

async function chooseModelDirectory() {
  if (state.localMode) {
    notify("当前为本机专业模式，模型目录由本机服务管理");
    return;
  }
  if (!modelDirectorySupported()) {
    notify("当前浏览器不支持选择模型文件夹，将继续使用浏览器缓存");
    return;
  }
  try {
    const handle = await window.showDirectoryPicker({ mode: "readwrite", id: "depthu-models" });
    if (!await hasModelDirectoryPermission(handle, true)) {
      notify("没有获得模型文件夹的读写权限");
      return;
    }
    state.modelDirectoryHandle = handle;
    state.modelCache = null;
    state.estimatorPromise = null;
    state.estimatorModel = "";
    state.modelPrepared = {};
    await storeModelDirectory(handle);
    updateModelStorageUI();
    updateModelOptionLabels();
    applyBrowserMode();
    notify(`模型将保存到：${handle.name}`);
  } catch (error) {
    if (error?.name !== "AbortError") notify("无法选择模型文件夹，请重试");
  }
}

async function writeBlobToSelectedDirectory(blob, filename) {
  state.outputWriteError = "";
  if (!state.outputDirectoryHandle && !state.outputFileHandle) return false;
  try {
    if (state.outputFileHandle) {
      const writable = await state.outputFileHandle.createWritable();
      await writable.write(blob);
      await writable.close();
      return true;
    }
    if (!await hasModelDirectoryPermission(state.outputDirectoryHandle, true)) {
      state.outputWriteError = "没有获得文件夹写入权限";
      return false;
    }
    const safeFilename = sanitizeOutputFilename(filename);
    const fileHandle = await state.outputDirectoryHandle.getFileHandle(safeFilename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
    return true;
  } catch (error) {
    state.outputWriteError = "浏览器拒绝了写入，请重新选择桌面文件夹";
    return false;
  }
}

async function writeBlobWithSavePicker(blob, filename) {
  if (typeof window.showSaveFilePicker !== "function") return false;
  try {
    const isVideo = state.outputKind === "video";
    const handle = await window.showSaveFilePicker({
      suggestedName: filename,
      types: [{
        description: isVideo ? "MP4 深度视频" : "PNG 深度图",
        accept: { [isVideo ? "video/mp4" : "image/png"]: [isVideo ? ".mp4" : ".png"] },
      }],
    });
    const writable = await handle.createWritable();
    await writable.write(blob);
    await writable.close();
    state.outputFileHandle = handle;
    state.outputDirectoryHandle = null;
    updateOutputLocationLabel();
    return true;
  } catch (error) {
    if (error?.name === "AbortError") return null;
    state.outputWriteError = "浏览器拒绝了文件写入";
    return false;
  }
}

async function exportDepthResult() {
  if (!state.depthReady) {
    notify("请先完成一次本机推理");
    return;
  }
  const filename = sanitizeOutputFilename(
    state.outputFilename || `${sourceFilenameStem()}_depth.${state.outputKind === "video" ? "mp4" : "png"}`,
  );
  const blob = state.outputBlob || await new Promise((resolve) => depthCanvas.toBlob(resolve, "image/png"));
  if (blob && await writeBlobToSelectedDirectory(blob, filename)) {
    notify(`${filename} 已保存到指定文件夹`);
    return;
  }
  if (blob && (state.outputDirectoryHandle || state.outputFileHandle) && state.outputWriteError) {
    const savedWithPicker = await writeBlobWithSavePicker(blob, filename);
    if (savedWithPicker === true) {
      notify(`${filename} 已保存`);
      return;
    }
  }
  const link = document.createElement("a");
  if (blob) {
    link.download = filename;
    link.href = URL.createObjectURL(blob);
  } else {
    link.download = filename;
    link.href = depthCanvas.toDataURL("image/png");
  }
  link.click();
  if (blob) window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  notify(state.outputWriteError
    ? `${filename} 无法写入所选文件夹，已下载到浏览器默认目录`
    : (state.outputKind === "video" ? "深度视频已导出" : "深度 PNG 已导出"));
}

function openOutputResult() {
  if (!state.depthReady) {
    notify("请先完成一次本机推理");
    return;
  }
  const target = state.outputUrl || (state.outputKind === "image" ? depthCanvas.toDataURL("image/png") : "");
  if (!target) {
    notify("请先导出结果");
    return;
  }
  const opened = window.open(target, "_blank", "noopener,noreferrer");
  if (!opened) notify("浏览器阻止了新窗口，请允许打开结果预览");
}

chooseButton.addEventListener("click", (event) => { event.stopPropagation(); fileInput.click(); });
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", (event) => loadFile(event.target.files[0]));
$("#removeFile").addEventListener("click", clearFile);
convertButton.addEventListener("click", () => state.converting ? null : startConversion());
$("#resetButton").addEventListener("click", resetControls);
smoothRange.addEventListener("input", updateSlider);
clearLogButton?.addEventListener("click", clearConversionLog);
prepareModelButton.addEventListener("click", async () => {
  try {
    if (state.localMode) {
      await prepareProfessionalModel();
    } else {
      const config = selectedModelConfig();
      await loadDepthEstimator(modelSelect.value);
      notify(state.modelDirectoryHandle
        ? `${config.name} 模型已下载到所选文件夹`
        : `${config.name} 模型已下载并缓存到当前浏览器`);
      updateConversionAvailability();
    }
  } catch (error) {
    notify((error?.message || "模型准备失败").slice(0, 120));
  }
});

modelSelectTrigger?.addEventListener("click", (event) => {
  event.stopPropagation();
  setModelPickerOpen(modelSelectMenu?.classList.contains("hidden"));
});
modelSelectTrigger?.addEventListener("keydown", (event) => {
  if (["Enter", " ", "ArrowDown"].includes(event.key)) {
    event.preventDefault();
    setModelPickerOpen(true);
    modelSelectOptions()[modelSelect.selectedIndex]?.focus();
  }
  if (event.key === "Escape") setModelPickerOpen(false);
});
modelSelectMenu?.addEventListener("click", (event) => {
  const option = event.target.closest(".model-select-option");
  if (option) chooseModelValue(option.dataset.modelValue);
});
modelSelectOptions().forEach((option, index, options) => {
  option.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const nextIndex = event.key === "ArrowDown"
        ? (index + 1) % options.length
        : (index - 1 + options.length) % options.length;
      options[nextIndex]?.focus();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setModelPickerOpen(false);
      modelSelectTrigger?.focus();
    }
    if (event.key === "Tab") setModelPickerOpen(false);
  });
});

modelSelect.addEventListener("change", () => {
  if (!state.localMode && state.estimatorModel !== modelSelect.value) {
    state.estimatorPromise = null;
    state.estimatorModel = "";
    state.modelCache = null;
  }
  const entry = localModelEntry(modelSelect.value);
  const cached = Boolean(entry?.[1]?.downloaded);
  const config = selectedModelConfig();
  const prepared = state.localMode ? cached : Boolean(state.modelPrepared[modelSelect.value]);
  modelDownloadLabel.textContent = prepared ? `${config.name} 模型已准备` : "转换时先下载所选模型";
  modelDownloadPercent.textContent = prepared ? "100%" : "按需";
  modelDownloadBar.style.width = prepared ? "100%" : "0%";
  prepareModelButton.textContent = prepared ? `检查 / 准备 ${config.name} 模型` : `提前准备 ${config.name} 模型`;
  if (!state.localMode) setModelState(prepared ? `${config.name} 浏览器模型已准备` : `浏览器本地模型 · ${config.name} 尚未准备`, prepared ? "ready" : "idle");
  syncModelPicker();
  updateConversionAvailability();
});

chooseOutputButton?.addEventListener("click", chooseOutputDirectory);
chooseModelDirectoryButton?.addEventListener("click", chooseModelDirectory);

$("#resolutionControl").addEventListener("click", (event) => {
  const button = event.target.closest(".segment");
  if (button) setResolution(button.dataset.resolution);
});

processingModeControl?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-processing-mode]");
  if (button && !button.disabled) setProcessingMode(button.dataset.processingMode);
});

["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add("is-dragging");
}));
["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove("is-dragging");
}));
dropzone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));

document.addEventListener("click", (event) => {
  if (modelSelectWrap && !event.target.closest(".model-select-wrap")) setModelPickerOpen(false);
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "fullscreen") openFullscreen();
  if (action === "open-result") openOutputResult();
  if (action === "close-fullscreen") closeFullscreen();
  if (action === "export") exportDepthResult();
  if (action === "settings") notify(state.localMode ? `本机专业模式已连接：${state.localDevice}` : "浏览器模式已启用：Small 模型通过 WebGPU / WASM 本地运行");
  if (action === "profile") notify(state.localMode ? "当前为本机专业工作区" : "当前为浏览器本地工作区");
  if (action === "output") notify(state.localMode ? "导出文件名与桌面端一致：<原文件名>_depth.mp4 / PNG" : "图片导出为 PNG，并保存到浏览器默认下载目录");
});
$("#fullscreenModal").addEventListener("click", (event) => { if (event.target.id === "fullscreenModal") closeFullscreen(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeFullscreen(); });

updateSlider();
setResolution("original");
updateProcessingModeUI();
setSourceFpsInfo();
syncModelPicker();
updateOutputLocationLabel();
updateModelStorageUI();
updateConversionAvailability();
(async () => {
  await restoreModelDirectory();
  detectLocalAgent();
})();
