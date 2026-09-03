import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

import {
  BRUSH_FUNCTIONS,
  BRUSH_NAMES,
  FIELD_NAMES,
  MATH_PROPERTIES,
  P5_FUNCTIONS,
  P5_VALUES,
  apiDescription,
  diagnostic,
  instrumentProgram,
  seedFor,
  validateProgram,
  validateRequest
} from "./gate.js";

const TIMEOUT_MS = 3_000;
const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = resolve(SCRIPT_DIR, "..");
const P5_PATH = resolve(ROOT_DIR, "node_modules/p5/lib/p5.min.js");
const BRUSH_PATH = resolve(ROOT_DIR, "node_modules/p5.brush/dist/p5.brush.js");
const PAGE_HTML = "<!doctype html><html><head><meta charset=\"utf-8\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; script-src 'unsafe-inline'; img-src data: blob:; style-src 'unsafe-inline'\"></head><body></body></html>";

let runtimeSources;

async function getRuntimeSources() {
  if (!runtimeSources) {
    const [p5, brush] = await Promise.all([readFile(P5_PATH, "utf8"), readFile(BRUSH_PATH, "utf8")]);
    runtimeSources = { p5, brush };
  }
  return runtimeSources;
}

async function createPage(browser) {
  const page = await browser.newPage();
  await page.setRequestInterception(true);
  page.on("request", request => request.abort());
  const sources = await getRuntimeSources();
  await page.setContent(PAGE_HTML, { waitUntil: "domcontentloaded" });
  await page.addStyleTag({ content: "html,body{margin:0;padding:0;overflow:hidden;background:#fff}canvas{display:block}" });
  await page.addScriptTag({ content: sources.p5 });
  return page;
}

async function preparePage(page, width, height) {
  const sources = await getRuntimeSources();
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  await page.evaluate(sources.brush);
}

async function withTimeout(promise, milliseconds) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const error = new Error(`render exceeded ${milliseconds} ms`);
      error.code = "RENDER_TIMEOUT";
      reject(error);
    }, milliseconds);
    timer.unref?.();
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

async function closeBrowser(browser) {
  try {
    await withTimeout(browser.close(), 2_000);
  } catch {
    browser.process()?.kill("SIGKILL");
  }
}

async function terminatePage(page) {
  try {
    await withTimeout(page.close({ runBeforeUnload: false }), 2_000);
  } catch {}
}

async function executeProgram(page, request, seed) {
  return page.evaluate(async payload => {
    window.__gateInstance?.remove();
    window.__gateFailure = null;
    window.__gateBudgetExceeded = false;
    window.onerror = (message, _source, _line, _column, error) => {
      window.__gateFailure ??= error?.message ?? message ?? "page error";
    };
    window.onunhandledrejection = event => {
      window.__gateFailure ??= event.reason?.message ?? String(event.reason);
    };
    const brushes = new Set(payload.brushNames);
    const fields = new Set(payload.fieldNames);
    const allowedBrushFunctions = new Set(payload.brushFunctions);
    const checkedBrushMethods = new Set(["set", "pick", "hatchStyle", "mass"]);
    const brushProxy = new Proxy(window.brush, {
      get(target, property) {
        if (typeof property !== "string" || !allowedBrushFunctions.has(property)) throw new Error(`brush.${String(property)} is not allowed`);
        const value = target[property];
        if (typeof value !== "function") throw new Error(`brush.${property} is unavailable`);
        return (...args) => {
          if (checkedBrushMethods.has(property) && !brushes.has(args[0])) throw new Error(`unknown brush ${String(args[0])}`);
          if (property === "field" && !fields.has(args[0])) throw new Error(`unknown field ${String(args[0])}`);
          return value.apply(target, args);
        };
      },
      set() {
        throw new Error("brush cannot be modified");
      }
    });
    let instance;
    await new Promise(resolve => {
      const sketch = p => {
        window.brush.instance(p);
        p.setup = () => {
          try {
            p.pixelDensity(1);
            p.createCanvas(payload.width, payload.height, p.WEBGL);
            window.brush.load();
            p.randomSeed(payload.seed);
            p.noiseSeed(payload.seed);
            p.background(255);
            const safeMath = {};
            for (const property of payload.mathProperties) safeMath[property] = property === "random" ? () => p.random() : Math[property];
            Object.freeze(safeMath);
            const names = [...payload.p5Functions, ...payload.p5Values, "brush", "Math"];
            let ticks = 0;
            const deadline = performance.now() + payload.timeoutMs - 500;
            const tick = () => {
              ticks += 1;
              if (ticks > 5_000_000 || (ticks % 1_024 === 0 && performance.now() > deadline)) {
                window.__gateBudgetExceeded = true;
                throw new Error("execution budget exceeded");
              }
            };
            const values = [
              ...payload.p5Functions.map(name => {
                if (typeof p[name] !== "function") throw new Error(`p5.${name} is unavailable`);
                return p[name].bind(p);
              }),
              ...payload.p5Values.map(name => name === "width" ? payload.width : name === "height" ? payload.height : p[name]),
              brushProxy,
              safeMath,
              tick
            ];
            names.push("__gateTick");
            const run = new Function(...names, `"use strict";\n${payload.code}`);
            p.push();
            p.translate(-payload.width / 2, -payload.height / 2);
            run(...values);
            p.pop();
            p.noLoop();
          } catch (error) {
            window.__gateFailure = error instanceof Error ? error.message : String(error);
          } finally {
            resolve();
          }
        };
      };
      instance = window.__gateInstance = new window.p5(sketch);
    });
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const gl = instance?._renderer?.drawingContext;
    if (gl && typeof gl.finish === "function") gl.finish();
    return { failure: window.__gateFailure, budgetExceeded: window.__gateBudgetExceeded };
  }, {
    code: request.instrumentedCode,
    width: request.width,
    height: request.height,
    seed,
    p5Functions: P5_FUNCTIONS,
    p5Values: P5_VALUES,
    brushFunctions: BRUSH_FUNCTIONS,
    brushNames: BRUSH_NAMES,
    fieldNames: FIELD_NAMES,
    mathProperties: [...MATH_PROPERTIES],
    timeoutMs: TIMEOUT_MS
  });
}

export async function launchBrowser(options = {}) {
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      "--use-angle=metal",
      "--disable-background-networking",
      "--disable-component-update",
      "--disable-default-apps",
      "--disable-sync",
      "--no-first-run"
    ],
    ...options
  });
  let page;
  try {
    page = await browser.newPage();
    const webgl = await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("webgl2");
      if (!context) return null;
      const extension = context.getExtension("WEBGL_debug_renderer_info");
      return {
        version: context.getParameter(context.VERSION),
        renderer: extension ? context.getParameter(extension.UNMASKED_RENDERER_WEBGL) : context.getParameter(context.RENDERER)
      };
    });
    if (!webgl) throw new Error("WebGL2 is unavailable");
    await page.close();
    return browser;
  } catch (error) {
    if (page) await page.close().catch(() => {});
    await browser.close();
    throw error;
  }
}

export async function createGateWorker(browser, artifactDir = join(tmpdir(), "divinci-gate")) {
  if (!browser) throw new TypeError("browser is required");
  const outputDir = resolve(artifactDir);
  await mkdir(outputDir, { recursive: true });
  let page = null;
  let closed = false;
  async function ensurePage() {
    if (!page || page.isClosed()) page = await createPage(browser);
    return page;
  }
  async function replacePage() {
    if (page) await terminatePage(page);
    page = null;
  }
  return {
    artifactDir: outputDir,
    async render(request) {
      const started = performance.now();
      const id = request?.id ?? null;
      const key = request?.key ?? null;
      if (closed) return { id, key, valid: false, image: null, render_ms: performance.now() - started, diagnostics: [diagnostic("input", "WORKER_CLOSED", "worker is closed")] };
      const inputDiagnostics = validateRequest(request);
      if (inputDiagnostics.length) return { id, key, valid: false, image: null, render_ms: performance.now() - started, diagnostics: inputDiagnostics };
      const validation = validateProgram(request.code);
      if (!validation.valid) return { id, key, valid: false, image: null, render_ms: performance.now() - started, diagnostics: validation.diagnostics };
      const seed = seedFor(request.key);
      const instrumentedCode = instrumentProgram(request.code, validation.ast);
      let activePage;
      try {
        activePage = await ensurePage();
        await preparePage(activePage, request.width, request.height);
        const execution = await withTimeout(executeProgram(activePage, { ...request, instrumentedCode }, seed), TIMEOUT_MS);
        if (execution.failure) {
          return {
            id,
            key,
            valid: false,
            image: null,
            render_ms: performance.now() - started,
            diagnostics: [diagnostic(execution.budgetExceeded ? "timeout" : "runtime", execution.budgetExceeded ? "EXECUTION_BUDGET" : "PROGRAM_ERROR", execution.failure)]
          };
        }
        const canvas = await activePage.$("canvas");
        if (!canvas) throw new Error("render did not create a canvas");
        const png = await canvas.screenshot({ type: "png", omitBackground: false });
        const sha256 = createHash("sha256").update(png).digest("hex");
        const path = join(outputDir, `${randomUUID()}.png`);
        await writeFile(path, png);
        return {
          id,
          key,
          valid: true,
          image: { path, width: request.width, height: request.height, sha256 },
          size: validation.size,
          seed,
          render_ms: performance.now() - started,
          diagnostics: []
        };
      } catch (error) {
        const timedOut = error?.code === "RENDER_TIMEOUT";
        await replacePage();
        return {
          id,
          key,
          valid: false,
          image: null,
          render_ms: performance.now() - started,
          diagnostics: [diagnostic(timedOut ? "timeout" : "capture", timedOut ? "RENDER_TIMEOUT" : "RENDER_FAILURE", error instanceof Error ? error.message : String(error))]
        };
      }
    },
    async close() {
      closed = true;
      await replacePage();
    },
    get closed() {
      return closed;
    }
  };
}

export async function createGatePool(size = 4, artifactDir = join(tmpdir(), "divinci-gate"), options = {}) {
  const count = Math.max(1, Math.floor(size));
  const entries = [];
  for (let index = 0; index < count; index += 1) {
    const browser = await launchBrowser(options);
    entries.push({ browser, worker: await createGateWorker(browser, artifactDir) });
  }
  const free = [...entries];
  const waiting = [];
  let closed = false;
  function acquire() {
    if (free.length) return Promise.resolve(free.pop());
    return new Promise(resolve => waiting.push(resolve));
  }
  function release(entry) {
    const next = waiting.shift();
    if (next) next(entry);
    else free.push(entry);
  }
  return {
    size: count,
    artifactDir: entries[0].worker.artifactDir,
    async render(request) {
      if (closed) return { id: request?.id ?? null, key: request?.key ?? null, valid: false, image: null, render_ms: 0, diagnostics: [diagnostic("input", "POOL_CLOSED", "pool is closed")] };
      const entry = await acquire();
      try {
        return await entry.worker.render(request);
      } finally {
        release(entry);
      }
    },
    async close() {
      closed = true;
      for (const entry of entries) {
        await entry.worker.close();
        await closeBrowser(entry.browser);
      }
    }
  };
}

async function serve(poolSize, artifactDir) {
  const pool = await createGatePool(poolSize, artifactDir);
  process.stdout.write(`${JSON.stringify({ ready: true, pool: pool.size, artifact_dir: pool.artifactDir })}\n`);
  const pending = new Set();
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of input) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let request;
    try {
      request = JSON.parse(trimmed);
    } catch (error) {
      process.stdout.write(`${JSON.stringify({ id: null, key: null, valid: false, image: null, render_ms: 0, diagnostics: [diagnostic("input", "MALFORMED_REQUEST", error.message)] })}\n`);
      continue;
    }
    const task = pool.render(request).then(result => {
      process.stdout.write(`${JSON.stringify(result)}\n`);
    });
    pending.add(task);
    task.finally(() => pending.delete(task));
  }
  await Promise.all(pending);
  await pool.close();
}

function parseCli(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--api") result.api = true;
    else if (argument === "--serve") result.serve = true;
    else if (argument.startsWith("--")) {
      result[argument.slice(2)] = argv[index + 1];
      index += 1;
    }
  }
  return result;
}

async function main() {
  const args = parseCli(process.argv.slice(2));
  if (args.api) {
    process.stdout.write(`${JSON.stringify(apiDescription(), null, 2)}\n`);
    return;
  }
  if (args.serve) {
    await serve(Number(args.pool ?? 4), args.out ?? join(tmpdir(), "divinci-gate"));
    return;
  }
  if (!args.code || !args.key || !args.width || !args.height) {
    process.stderr.write("usage: node scripts/gate_server.js --code FILE --key KEY --width N --height N [--out DIR]\n   or: node scripts/gate_server.js --serve [--pool N] [--out DIR]\n");
    process.exitCode = 2;
    return;
  }
  const browser = await launchBrowser();
  const worker = await createGateWorker(browser, args.out ?? join(tmpdir(), "divinci-gate"));
  try {
    const result = await worker.render({
      id: basename(args.code),
      key: args.key,
      code: await readFile(resolve(args.code), "utf8"),
      width: Number(args.width),
      height: Number(args.height)
    });
    process.stdout.write(`${JSON.stringify(result)}\n`);
    if (!result.valid) process.exitCode = 1;
  } finally {
    await worker.close();
    await closeBrowser(browser);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
