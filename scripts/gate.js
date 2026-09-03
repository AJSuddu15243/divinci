import { parse } from "acorn";
import * as walk from "acorn-walk";
import { createHash } from "node:crypto";

const SOURCE_LIMIT = 65_536;
const AST_LIMIT = 20_000;
const EDGE_LIMIT = 4_096;
const PIXEL_LIMIT = 1_048_576;

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

export const MATH_PROPERTIES = new Set([
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
const CAPABILITY_NAMES = new Set([...P5_FUNCTIONS, ...P5_VALUES, "brush", ...SAFE_GLOBALS]);

export function diagnostic(stage, code, message, node = null) {
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
  const declared = new Set();
  walk.full(ast, node => {
    nodeCount += 1;
    if (node.type === "VariableDeclarator") patternNames(node.id, declared);
    if (node.type === "FunctionDeclaration" || node.type === "FunctionExpression" || node.type === "ArrowFunctionExpression") {
      if (node.id) declared.add(node.id.name);
      for (const parameter of node.params) patternNames(parameter, declared);
    }
    if (node.type === "CatchClause") patternNames(node.param, declared);
  });
  if (nodeCount > AST_LIMIT) {
    return { valid: false, diagnostics: [diagnostic("input", "AST_LIMIT", `program exceeds ${AST_LIMIT} syntax nodes`)] };
  }
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
  return { valid: unique.length === 0, diagnostics: unique, ast, size: { bytes: Buffer.byteLength(code, "utf8"), nodes: nodeCount, byte_limit: SOURCE_LIMIT, node_limit: AST_LIMIT } };
}

export function instrumentProgram(code, ast) {
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

export function validateRequest(request) {
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

export function seedFor(key) {
  return createHash("sha256").update(key).digest().readUInt32BE(0);
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
