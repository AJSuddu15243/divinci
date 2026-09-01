import { parse } from "acorn";
import * as walk from "acorn-walk";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const SOURCE_LIMIT = 65_536;
const AST_LIMIT = 20_000;
const EDGE_LIMIT = 4_096;
const PIXEL_LIMIT = 1_048_576;
const TIMEOUT_MS = 10_000;

export const BRUSH_NAMES = Object.freeze([
  "2B",
  "HB",
  "2H",
  "cpencil",
  "pen",
  "rotring",
  "spray",
  "marker",
  "marker2",
  "charcoal",
  "hatch_brush"
]);

export const FIELD_NAMES = Object.freeze([
  "hand",
  "curved",
  "zigzag",
  "waves",
  "seabed",
  "spiral",
  "columns"
]);

export const P5_FUNCTIONS = Object.freeze([
  "abs",
  "acos",
  "alpha",
  "angleMode",
  "arc",
  "asin",
  "atan",
  "atan2",
  "background",
  "bezier",
  "bezierPoint",
  "bezierTangent",
  "bezierVertex",
  "beginShape",
  "blue",
  "brightness",
  "ceil",
  "circle",
  "color",
  "colorMode",
  "constrain",
  "cos",
  "degrees",
  "dist",
  "ellipse",
  "ellipseMode",
  "endShape",
  "exp",
  "fill",
  "floor",
  "green",
  "hue",
  "lerp",
  "lerpColor",
  "lightness",
  "line",
  "log",
  "mag",
  "map",
  "max",
  "min",
  "noFill",
  "noise",
  "noiseDetail",
  "norm",
  "noStroke",
  "point",
  "pop",
  "pow",
  "push",
  "quad",
  "radians",
  "random",
  "randomGaussian",
  "rect",
  "rectMode",
  "red",
  "rotate",
  "rotateX",
  "rotateY",
  "rotateZ",
  "round",
  "saturation",
  "scale",
  "shearX",
  "shearY",
  "sin",
  "sq",
  "sqrt",
  "square",
  "stroke",
  "strokeCap",
  "strokeJoin",
  "strokeWeight",
  "tan",
  "translate",
  "triangle",
  "vertex"
]);

export const P5_VALUES = Object.freeze([
  "width",
  "height",
  "PI",
  "TWO_PI",
  "HALF_PI",
  "QUARTER_PI",
  "TAU",
  "DEGREES",
  "RADIANS",
  "RGB",
  "HSB",
  "HSL",
  "CENTER",
  "CORNER",
  "CORNERS",
  "RADIUS",
  "OPEN",
  "CHORD",
  "PIE",
  "CLOSE",
  "ROUND",
  "SQUARE",
  "PROJECT",
  "MITER",
  "BEVEL"
]);

export const BRUSH_FUNCTIONS = Object.freeze([
  "scaleBrushes",
  "field",
  "noField",
  "refreshField",
  "wiggle",
  "clip",
  "noClip",
  "set",
  "pick",
  "stroke",
  "noStroke",
  "strokeWeight",
  "fill",
  "noFill",
  "wash",
  "noWash",
  "fillBleed",
  "fillTexture",
  "hatch",
  "noHatch",
  "hatchStyle",
  "mass",
  "noMass",
  "line",
  "flowLine",
  "beginStroke",
  "move",
  "endStroke",
  "spline",
  "rect",
  "circle",
  "arc",
  "beginShape",
  "vertex",
  "endShape",
  "polygon"
]);

const MATH_PROPERTIES = new Set([
  "E",
  "LN2",
  "LN10",
  "LOG2E",
  "LOG10E",
  "PI",
  "SQRT1_2",
  "SQRT2",
  "abs",
  "acos",
  "acosh",
  "asin",
  "asinh",
  "atan",
  "atanh",
  "atan2",
  "cbrt",
  "ceil",
  "clz32",
  "cos",
  "cosh",
  "exp",
  "expm1",
  "floor",
  "fround",
  "hypot",
  "imul",
  "log",
  "log1p",
  "log2",
  "log10",
  "max",
  "min",
  "pow",
  "random",
  "round",
  "sign",
  "sin",
  "sinh",
  "sqrt",
  "tan",
  "tanh",
  "trunc"
]);

const SAFE_METHODS = new Set([
  "at",
  "charAt",
  "charCodeAt",
  "concat",
  "endsWith",
  "entries",
  "every",
  "filter",
  "find",
  "findIndex",
  "flat",
  "flatMap",
  "forEach",
  "includes",
  "indexOf",
  "join",
  "keys",
  "lastIndexOf",
  "map",
  "padEnd",
  "padStart",
  "pop",
  "push",
  "reduce",
  "reduceRight",
  "repeat",
  "replace",
  "replaceAll",
  "reverse",
  "shift",
  "slice",
  "some",
  "sort",
  "splice",
  "split",
  "startsWith",
  "substring",
  "substr",
  "toExponential",
  "toFixed",
  "toLowerCase",
  "toPrecision",
  "toUpperCase",
  "trim",
  "trimEnd",
  "trimStart",
  "unshift",
  "values"
]);

const SAFE_GLOBALS = new Set([
  "Math",
  "Number",
  "String",
  "Boolean",
  "parseInt",
  "parseFloat",
  "isFinite",
  "isNaN",
  "Infinity",
  "NaN",
  "undefined"
]);

const FORBIDDEN_IDENTIFIERS = new Set([
  "window",
  "document",
  "globalThis",
  "self",
  "parent",
  "top",
  "frames",
  "navigator",
  "location",
  "history",
  "fetch",
  "XMLHttpRequest",
  "WebSocket",
  "EventSource",
  "Worker",
  "SharedWorker",
  "ServiceWorker",
  "eval",
  "Function",
  "AsyncFunction",
  "GeneratorFunction",
  "Date",
  "performance",
  "crypto",
  "setTimeout",
  "setInterval",
  "requestAnimationFrame",
  "queueMicrotask",
  "localStorage",
  "sessionStorage",
  "indexedDB",
  "caches",
  "console",
  "arguments",
  "__gateTick"
]);

const FORBIDDEN_PROPERTIES = new Set([
  "constructor",
  "prototype",
  "__proto__",
  "caller",
  "callee",
  "call",
  "apply",
  "bind"
]);

const FORBIDDEN_NODES = new Map([
  ["AwaitExpression", "await"],
  ["ClassDeclaration", "class"],
  ["ClassExpression", "class"],
  ["DebuggerStatement", "debugger"],
  ["ImportExpression", "import"],
  ["MetaProperty", "meta property"],
  ["NewExpression", "constructor call"],
  ["Super", "super"],
  ["TaggedTemplateExpression", "tagged template"],
  ["ThisExpression", "this"],
  ["WithStatement", "with"],
  ["YieldExpression", "yield"]
]);

const BRUSH_SET = new Set(BRUSH_FUNCTIONS);
const P5_FUNCTION_SET = new Set(P5_FUNCTIONS);
const P5_VALUE_SET = new Set(P5_VALUES);
const CAPABILITY_NAMES = new Set([...P5_FUNCTIONS, ...P5_VALUES, "brush", ...SAFE_GLOBALS]);
const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = resolve(SCRIPT_DIR, "..");
const P5_PATH = resolve(ROOT_DIR, "node_modules/p5/lib/p5.min.js");
const BRUSH_PATH = resolve(ROOT_DIR, "node_modules/p5.brush/dist/p5.brush.js");
const PAGE_HTML = "<!doctype html><html><head><meta charset=\"utf-8\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; script-src 'unsafe-inline'; img-src data: blob:; style-src 'unsafe-inline'\"></head><body></body></html>";

let runtimeSources;

function diagnostic(stage, code, message, node = null) {
  return {
    stage,
    code,
    message,
    line: node?.loc?.start?.line ?? null,
    column: node?.loc?.start?.column ?? null
  };
}

function patternNames(node, names) {
  if (!node) return;
  if (node.type === "Identifier") names.add(node.name);
  if (node.type === "RestElement") patternNames(node.argument, names);
  if (node.type === "AssignmentPattern") patternNames(node.left, names);
  if (node.type === "ArrayPattern") {
    for (const element of node.elements) patternNames(element, names);
  }
  if (node.type === "ObjectPattern") {
    for (const property of node.properties) patternNames(property.value ?? property.argument, names);
  }
}

function isReferenceIdentifier(node, parent) {
  if (!parent) return true;
  if (parent.type === "VariableDeclarator" && parent.id === node) return false;
  if ((parent.type === "FunctionDeclaration" || parent.type === "FunctionExpression" || parent.type === "ArrowFunctionExpression") && (parent.id === node || parent.params.includes(node))) return false;
  if (parent.type === "CatchClause" && parent.param === node) return false;
  if (parent.type === "RestElement" && parent.argument === node) return false;
  if (parent.type === "AssignmentPattern" && parent.left === node) return false;
  if (parent.type === "Property" && parent.key === node && !parent.computed && !parent.shorthand) return false;
  if (parent.type === "MemberExpression" && parent.property === node && !parent.computed) return false;
  if (parent.type === "MethodDefinition" && parent.key === node && !parent.computed) return false;
  if ((parent.type === "LabeledStatement" || parent.type === "BreakStatement" || parent.type === "ContinueStatement") && parent.label === node) return false;
  return true;
}

function memberName(node) {
  if (!node.computed && node.property.type === "Identifier") return node.property.name;
  if (node.computed && node.property.type === "Literal" && typeof node.property.value === "string") return node.property.value;
  return null;
}

function addPolicyDiagnostic(diagnostics, code, message, node) {
  if (diagnostics.length < 32) diagnostics.push(diagnostic("policy", code, message, node));
}

export function validateProgram(code) {
  if (typeof code !== "string") {
    return { valid: false, diagnostics: [diagnostic("input", "CODE_TYPE", "code must be a string")] };
  }
  if (Buffer.byteLength(code, "utf8") > SOURCE_LIMIT) {
    return { valid: false, diagnostics: [diagnostic("input", "SOURCE_LIMIT", `code exceeds ${SOURCE_LIMIT} bytes`)] };
  }
  let ast;
  try {
    ast = parse(code, { ecmaVersion: 2022, sourceType: "script", locations: true });
  } catch (error) {
    return {
      valid: false,
      diagnostics: [{
        stage: "parse",
        code: "MALFORMED_JAVASCRIPT",
        message: error.message,
        line: error.loc?.line ?? null,
        column: error.loc?.column ?? null
      }]
    };
  }
  let nodeCount = 0;
  walk.full(ast, () => {
    nodeCount += 1;
  });
  if (nodeCount > AST_LIMIT) {
    return { valid: false, diagnostics: [diagnostic("input", "AST_LIMIT", `program exceeds ${AST_LIMIT} syntax nodes`)] };
  }
  const declared = new Set();
  walk.full(ast, node => {
    if (node.type === "VariableDeclarator") patternNames(node.id, declared);
    if (node.type === "FunctionDeclaration" || node.type === "FunctionExpression" || node.type === "ArrowFunctionExpression") {
      if (node.id) declared.add(node.id.name);
      for (const parameter of node.params) patternNames(parameter, declared);
    }
    if (node.type === "CatchClause") patternNames(node.param, declared);
  });
  const diagnostics = [];
  for (const name of declared) {
    if (CAPABILITY_NAMES.has(name) || FORBIDDEN_IDENTIFIERS.has(name)) {
      addPolicyDiagnostic(diagnostics, "CAPABILITY_SHADOW", `declaration cannot shadow ${name}`, ast);
    }
  }
  walk.fullAncestor(ast, (node, _state, ancestors) => {
    const parent = ancestors.at(-2) ?? null;
    if (FORBIDDEN_NODES.has(node.type)) {
      addPolicyDiagnostic(diagnostics, "FORBIDDEN_SYNTAX", `${FORBIDDEN_NODES.get(node.type)} is not allowed`, node);
    }
    if ((node.type === "FunctionDeclaration" || node.type === "FunctionExpression" || node.type === "ArrowFunctionExpression") && (node.async || node.generator)) {
      addPolicyDiagnostic(diagnostics, "FORBIDDEN_FUNCTION", "async and generator functions are not allowed", node);
    }
    if (node.type === "Identifier") {
      if (FORBIDDEN_IDENTIFIERS.has(node.name)) {
        addPolicyDiagnostic(diagnostics, "FORBIDDEN_IDENTIFIER", `${node.name} is not allowed`, node);
      } else if (isReferenceIdentifier(node, parent) && !declared.has(node.name) && !CAPABILITY_NAMES.has(node.name)) {
        addPolicyDiagnostic(diagnostics, "UNKNOWN_IDENTIFIER", `unknown identifier ${node.name}`, node);
      }
    }
    if (node.type === "Property") {
      const name = !node.computed && node.key.type === "Identifier" ? node.key.name : node.key.type === "Literal" ? node.key.value : null;
      if (FORBIDDEN_PROPERTIES.has(name)) addPolicyDiagnostic(diagnostics, "FORBIDDEN_PROPERTY", `${name} is not allowed`, node);
    }
    if (node.type === "MemberExpression") {
      const name = memberName(node);
      if (FORBIDDEN_PROPERTIES.has(name)) addPolicyDiagnostic(diagnostics, "FORBIDDEN_PROPERTY", `${name} is not allowed`, node);
      if (node.object.type === "Identifier" && node.object.name === "brush") {
        if (node.computed) addPolicyDiagnostic(diagnostics, "DYNAMIC_BRUSH_ACCESS", "computed brush access is not allowed", node);
        else if (!BRUSH_SET.has(name)) addPolicyDiagnostic(diagnostics, "UNKNOWN_BRUSH_API", `unknown brush API ${name}`, node);
      }
      if (node.object.type === "Identifier" && node.object.name === "Math") {
        if (node.computed) addPolicyDiagnostic(diagnostics, "DYNAMIC_MATH_ACCESS", "computed Math access is not allowed", node);
        else if (!MATH_PROPERTIES.has(name)) addPolicyDiagnostic(diagnostics, "UNKNOWN_MATH_API", `unknown Math API ${name}`, node);
      }
    }
    if (node.type === "CallExpression" && node.callee.type === "MemberExpression") {
      const name = memberName(node.callee);
      const object = node.callee.object;
      if (!(object.type === "Identifier" && (object.name === "brush" || object.name === "Math")) && !SAFE_METHODS.has(name)) {
        addPolicyDiagnostic(diagnostics, "UNKNOWN_METHOD", `method ${name ?? "<computed>"} is not allowed`, node.callee);
      }
    }
    if (node.type === "AssignmentExpression" && node.left.type === "MemberExpression") {
      if (node.left.object.type === "Identifier" && (node.left.object.name === "brush" || node.left.object.name === "Math")) {
        addPolicyDiagnostic(diagnostics, "CAPABILITY_MUTATION", `${node.left.object.name} cannot be modified`, node.left);
      }
    }
  });
  const unique = [];
  const seen = new Set();
  for (const item of diagnostics) {
    const key = `${item.code}:${item.line}:${item.column}:${item.message}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(item);
    }
  }
  return { valid: unique.length === 0, diagnostics: unique };
}

function instrumentProgram(code) {
  const ast = parse(code, { ecmaVersion: 2022, sourceType: "script" });
  const insertions = new Map();
  const add = (position, text) => {
    const values = insertions.get(position) ?? [];
    values.push(text);
    insertions.set(position, values);
  };
  const instrumentBlock = body => {
    if (body.type === "BlockStatement") add(body.start + 1, "__gateTick();");
    else {
      add(body.start, "{__gateTick();");
      add(body.end, "}");
    }
  };
  walk.full(ast, node => {
    if (["ForStatement", "ForInStatement", "ForOfStatement", "WhileStatement", "DoWhileStatement"].includes(node.type)) instrumentBlock(node.body);
    if (["FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression"].includes(node.type)) {
      if (node.body.type === "BlockStatement") add(node.body.start + 1, "__gateTick();");
      else {
        add(node.body.start, "(__gateTick(),");
        add(node.body.end, ")");
      }
    }
  });
  let result = code;
  for (const position of [...insertions.keys()].sort((a, b) => b - a)) result = `${result.slice(0, position)}${insertions.get(position).join("")}${result.slice(position)}`;
  return result;
}

function validateRequest(request) {
  const diagnostics = [];
  if (!request || typeof request !== "object" || Array.isArray(request)) diagnostics.push(diagnostic("input", "REQUEST_TYPE", "request must be an object"));
  if (request && !["string", "number"].includes(typeof request.id)) diagnostics.push(diagnostic("input", "ID_TYPE", "id must be a string or number"));
  if (request && (typeof request.key !== "string" || request.key.length === 0)) diagnostics.push(diagnostic("input", "KEY_TYPE", "key must be a non-empty string"));
  for (const name of ["width", "height"]) {
    const value = request?.[name];
    if (!Number.isInteger(value) || value < 2 || value > EDGE_LIMIT) diagnostics.push(diagnostic("input", "DIMENSION_LIMIT", `${name} must be an integer from 2 to ${EDGE_LIMIT}`));
  }
  if (Number.isInteger(request?.width) && Number.isInteger(request?.height) && request.width * request.height > PIXEL_LIMIT) {
    diagnostics.push(diagnostic("input", "PIXEL_LIMIT", `canvas exceeds ${PIXEL_LIMIT} pixels`));
  }
  return diagnostics;
}

function seedFor(key) {
  return createHash("sha256").update(key).digest().readUInt32BE(0);
}

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
  return page;
}

async function preparePage(page, width, height) {
  const sources = await getRuntimeSources();
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  await page.setContent(PAGE_HTML, { waitUntil: "domcontentloaded" });
  await page.addStyleTag({ content: "html,body{margin:0;padding:0;overflow:hidden;background:#fff}canvas{display:block}" });
  await page.addScriptTag({ content: sources.p5 });
  await page.addScriptTag({ content: sources.brush });
}

async function terminatePage(page) {
  try {
    await withTimeout(page.close({ runBeforeUnload: false }), 2_000);
  } catch {}
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

async function executeProgram(page, request, seed) {
  return page.evaluate(async payload => {
    window.__gateFailure = null;
    window.__gateDone = false;
    window.__gateBudgetExceeded = false;
    window.addEventListener("error", event => {
      window.__gateFailure ??= event.error?.message ?? event.message ?? "page error";
    });
    window.addEventListener("unhandledrejection", event => {
      window.__gateFailure ??= event.reason?.message ?? String(event.reason);
    });
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
            window.__gateDone = true;
            resolve();
          }
        };
      };
      instance = new window.p5(sketch);
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
      const instrumentedCode = instrumentProgram(request.code);
      let activePage;
      try {
        activePage = await ensurePage();
        await preparePage(activePage, request.width, request.height);
        const execution = await withTimeout(executeProgram(activePage, { ...request, instrumentedCode }, seed), TIMEOUT_MS);
        if (execution.failure) {
          await replacePage();
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

export function apiDescription() {
  return {
    contract: "JavaScript draw body",
    canvas: "width and height are provided; origin is top-left; background starts opaque white",
    p5_functions: P5_FUNCTIONS,
    p5_values: P5_VALUES,
    brush_functions: BRUSH_FUNCTIONS,
    brush_names: BRUSH_NAMES,
    field_names: FIELD_NAMES
  };
}

function parseCli(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--api") result.api = true;
    else if (argument.startsWith("--")) {
      const name = argument.slice(2);
      result[name] = argv[index + 1];
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
  if (!args.code || !args.key || !args.width || !args.height) {
    process.stderr.write("usage: node scripts/gate.js --code FILE --key KEY --width N --height N [--out DIR]\n");
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
    await browser.close();
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
