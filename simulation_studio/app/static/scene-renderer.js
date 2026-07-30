(function (global) {
  "use strict";

  var WEATHER_INDEX = {
    clear: 0,
    cloudy: 1,
    snow: 2,
    ice_storm: 3,
    blizzard: 4,
  };

  var WEATHER_LABELS = {
    clear: "Clear Air",
    cloudy: "Cloud Cover",
    snow: "Snowfall",
    ice_storm: "Ice Storm",
    blizzard: "Blizzard Front",
  };

  var WORLD_WEATHER_LABELS = {
    clear: "Clear Skies",
    clear_skies: "Clear Skies",
    cloudy: "Cloud Cover",
    overcast: "Overcast",
    snow: "Snowfall",
    light_snow: "Light Snow",
    heavy_snow: "Heavy Snow",
    ice_storm: "Ice Storm",
    freezing_rain: "Freezing Rain",
    blizzard: "Blizzard",
    heat_wave: "Heat Wave",
    spring_thaw: "Spring Thaw",
  };

  var CENTER_STYLE = {
    hospital: {
      fill: "#ff3347",
      glow: "rgba(255, 51, 71, 0.48)",
      edge: "rgba(255, 196, 203, 0.96)",
      glyph: "#fff6f4",
      label: "Hospital",
    },
    blood_bank: {
      fill: "#1677ff",
      glow: "rgba(22, 119, 255, 0.5)",
      edge: "rgba(190, 219, 255, 0.95)",
      glyph: "#f4f9ff",
      label: "Blood Bank",
    },
    mobile: {
      fill: "#10b981",
      glow: "rgba(16, 185, 129, 0.46)",
      edge: "rgba(187, 247, 208, 0.95)",
      glyph: "#f0fdf4",
      label: "Mobile Unit",
    },
    default: {
      fill: "#a855f7",
      glow: "rgba(168, 85, 247, 0.42)",
      edge: "rgba(233, 213, 255, 0.9)",
      glyph: "#faf5ff",
      label: "Center",
    },
  };

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatNumber(value) {
    if (value == null || isNaN(value)) return "0";
    var numeric = Number(value);
    if (Math.abs(numeric) >= 1000) return numeric.toLocaleString("en-US");
    return numeric.toLocaleString("en-US", {
      minimumFractionDigits: Math.abs(numeric) >= 100 ? 0 : 1,
      maximumFractionDigits: 1,
    });
  }

  function formatInteger(value) {
    if (value == null || isNaN(value)) return "0";
    return Math.round(Number(value)).toLocaleString("en-US");
  }

  function formatPercent(value) {
    if (value == null || isNaN(value)) return "0.0%";
    return Number(value).toFixed(1) + "%";
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function titleCase(text) {
    return String(text || "")
      .split(/[_\s-]+/)
      .filter(Boolean)
      .map(function (part) {
        return part.charAt(0).toUpperCase() + part.slice(1);
      })
      .join(" ");
  }

  function normalizeWeatherState(weather) {
    var value = String(weather || "").toLowerCase();
    if (!value) return "clear";
    if (value === "clear_skies") return "clear";
    if (value === "overcast") return "cloudy";
    if (value === "light_snow" || value === "heavy_snow") return "snow";
    if (value === "freezing_rain") return "ice_storm";
    if (value === "heat_wave") return "clear";
    if (value === "spring_thaw") return "cloudy";
    if (WEATHER_LABELS[value]) return value;
    return "clear";
  }

  function formatWorldWeather(weather, temperature) {
    var label = WORLD_WEATHER_LABELS[weather] || titleCase(weather || "clear");
    if (temperature == null || temperature === "") return label;
    return label + " • " + temperature + "°C";
  }

  function shortCenterName(name) {
    var value = String(name || "Center")
      .replace("Hema-Quebec ", "HQ ")
      .replace("Hema-Quebec", "HQ")
      .replace("Héma-Québec ", "HQ ")
      .replace("Héma-Québec", "HQ")
      .replace("CHU de Quebec - ", "")
      .replace("CHU de Québec - ", "");
    if (value.length > 20) return value.slice(0, 18) + "...";
    return value;
  }

  function latLonToPosition(lat, lon) {
    if (typeof global.latLonToPosition === "function") {
      return global.latLonToPosition(lat, lon);
    }
    var latMin = 46.76;
    var latMax = 46.86;
    var lonMin = -71.35;
    var lonMax = -71.15;
    var x = ((lon - lonMin) / (lonMax - lonMin)) * 80 + 10;
    var y = ((latMax - lat) / (latMax - latMin)) * 80 + 10;
    return { x: clamp(x, 5, 95), y: clamp(y, 5, 95) };
  }

  function distanceSquared(ax, ay, bx, by) {
    var dx = ax - bx;
    var dy = ay - by;
    return dx * dx + dy * dy;
  }

  function quadraticPoint(fromX, fromY, controlX, controlY, toX, toY, t) {
    var omt = 1 - t;
    return {
      x: omt * omt * fromX + 2 * omt * t * controlX + t * t * toX,
      y: omt * omt * fromY + 2 * omt * t * controlY + t * t * toY,
    };
  }

  function createShader(gl, type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      var info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(info || "Shader compilation failed");
    }
    return shader;
  }

  function createProgram(gl, vertexSource, fragmentSource) {
    var vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
    var fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    var program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      var info = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(info || "Program linking failed");
    }
    return program;
  }

  function drawRoundedRectPath(ctx, x, y, width, height, radius) {
    var r = Math.min(radius, width * 0.5, height * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function SimulationSceneRenderer(board) {
    this.board = typeof board === "string" ? document.getElementById(board) : board;
    if (!this.board) throw new Error("Map board element not found");

    this.shell = null;
    this.gpuCanvas = null;
    this.canvas = null;
    this.ctx = null;
    this.staticCanvas = null;
    this.staticCtx = null;
    this.staticDirty = true;
    this.refs = {};
    this.resizeObserver = null;
    this.boundResize = this.resize.bind(this);
    this.boundFrame = this.frame.bind(this);
    this.boundPointerMove = this.handlePointerMove.bind(this);
    this.boundPointerLeave = this.handlePointerLeave.bind(this);

    this.width = 1;
    this.height = 1;
    this.cssWidth = 1;
    this.cssHeight = 1;
    this.dpr = 1;
    this.frameId = 0;
    this.active = false;
    this.lastTime = 0;
    this.elapsed = 0;

    this.gpu = null;
    this.gpuInitPromise = null;
    this.centers = [];
    this.baseLinks = [];
    this.hoverIndex = -1;
    this.weather = "clear";
    this.hour = 0;
    this.dayFactor = 1;
    this.dayFactorTarget = 1;
    this.alertLevel = 0;
    this.alertTarget = 0;
    this.frameData = {
      score: 0,
      shortage_rate: 0,
      budget_remaining: 0,
      total_donated: 0,
      total_transfused: 0,
      total_expired: 0,
      total_hours: 0,
      progress: 0,
    };
    this.worldContext = {
      date: "",
      season: "",
      weatherLabel: "",
    };
    this.meta = {
      scenarioName: "Quebec Blood Network",
      strategyName: "Simulation Control",
      mode: "Simulation",
    };

    this.ambientParticles = [];
    this.weatherParticles = [];
    this.transferEffects = [];
    this.donationBursts = [];
    this.pulses = [];

    this.initDom();
    this.initGpu();
    this.seedAmbientParticles();
    this.syncWeatherParticles();
    this.setMissionContext(null, null, false);
    this.setOperationalFrame({});
    this.refreshInspector();
    this.resize();
  }

  SimulationSceneRenderer.prototype.initDom = function () {
    this.board.innerHTML = "";
    this.board.classList.add("scene-board-ready");

    this.shell = document.createElement("div");
    this.shell.className = "scene-shell";
    this.shell.innerHTML =
      '<canvas class="scene-gpu-canvas" aria-hidden="true"></canvas>' +
      '<canvas class="scene-canvas" aria-hidden="true"></canvas>' +
      '<div class="scene-topbar">' +
      '<div class="scene-copy">' +
      '<div class="scene-kicker">Realtime Logistics Graph</div>' +
      '<div class="scene-context-row">' +
      '<span class="scene-context-chip"><span>Date</span><strong data-scene-date>Simulation Day 1</strong></span>' +
      '<span class="scene-context-chip"><span>Season</span><strong data-scene-season>Live Network</strong></span>' +
      '<span class="scene-context-chip"><span>Weather</span><strong data-scene-world-weather>Clear Skies</strong></span>' +
      "</div>" +
      '<div class="scene-heading-row">' +
      '<h3 class="scene-title" data-scene-title>Quebec Blood Network</h3>' +
      '<span class="scene-mode" data-scene-mode>Simulation</span>' +
      "</div>" +
      '<p class="scene-subtitle" data-scene-subtitle>GPU scene renderer armed for live inventory flow.</p>' +
      "</div>" +
      '<div class="scene-status-grid">' +
      '<div class="scene-status-card"><span>Clock</span><strong data-scene-clock>Hour 0</strong></div>' +
      '<div class="scene-status-card"><span>Weather</span><strong data-scene-weather>Clear Air</strong></div>' +
      '<div class="scene-status-card"><span>Pressure</span><strong data-scene-pressure>0.0%</strong></div>' +
      "</div>" +
      "</div>" +
      '<div class="scene-inspector" data-scene-inspector></div>' +
      '<div class="scene-footer">' +
      '<div class="scene-legend">' +
      '<span class="scene-legend-title">Inventory Mix</span>' +
      '<span class="scene-legend-chip chip-rbc">RBC</span>' +
      '<span class="scene-legend-chip chip-plt">Platelets</span>' +
      '<span class="scene-legend-chip chip-pls">Plasma</span>' +
      '<span class="scene-legend-chip chip-flow">Flow</span>' +
      "</div>" +
      '<div class="scene-mini-metrics" data-scene-mini-metrics></div>' +
      "</div>";
    this.board.appendChild(this.shell);

    this.gpuCanvas = this.shell.querySelector(".scene-gpu-canvas");
    this.canvas = this.shell.querySelector(".scene-canvas");
    this.ctx = this.canvas.getContext("2d", { alpha: true, desynchronized: true });
    if (this.ctx) this.ctx.imageSmoothingEnabled = true;
    this.staticCanvas = document.createElement("canvas");
    this.staticCtx = this.staticCanvas.getContext("2d", {
      alpha: true,
      desynchronized: true,
    });
    if (this.staticCtx) this.staticCtx.imageSmoothingEnabled = true;

    this.refs = {
      title: this.shell.querySelector("[data-scene-title]"),
      mode: this.shell.querySelector("[data-scene-mode]"),
      subtitle: this.shell.querySelector("[data-scene-subtitle]"),
      date: this.shell.querySelector("[data-scene-date]"),
      season: this.shell.querySelector("[data-scene-season]"),
      worldWeather: this.shell.querySelector("[data-scene-world-weather]"),
      clock: this.shell.querySelector("[data-scene-clock]"),
      weather: this.shell.querySelector("[data-scene-weather]"),
      pressure: this.shell.querySelector("[data-scene-pressure]"),
      inspector: this.shell.querySelector("[data-scene-inspector]"),
      miniMetrics: this.shell.querySelector("[data-scene-mini-metrics]"),
    };

    this.canvas.addEventListener("mousemove", this.boundPointerMove);
    this.canvas.addEventListener("mouseleave", this.boundPointerLeave);
    global.addEventListener("resize", this.boundResize);

    if (typeof global.ResizeObserver === "function") {
      this.resizeObserver = new global.ResizeObserver(this.boundResize);
      this.resizeObserver.observe(this.board);
    }
  };

  SimulationSceneRenderer.prototype.initGpu = function () {
    var renderer = this;
    if (global.navigator && global.navigator.gpu) {
      this.gpuInitPromise = this.initWebGpu().catch(function (error) {
        console.warn("Scene renderer WebGPU init failed:", error);
        renderer.initWebGl();
      });
      return;
    }
    this.initWebGl();
  };

  SimulationSceneRenderer.prototype.initWebGpu = async function () {
    var adapter = await global.navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("No WebGPU adapter available");
    }

    var device = await adapter.requestDevice();
    var context = this.gpuCanvas.getContext("webgpu");
    if (!context) {
      throw new Error("WebGPU canvas context unavailable");
    }

    var format = global.navigator.gpu.getPreferredCanvasFormat
      ? global.navigator.gpu.getPreferredCanvasFormat()
      : "bgra8unorm";
    var shaderCode =
      "struct Uniforms {" +
      "  resolution: vec2f," +
      "  time: f32," +
      "  weather: f32," +
      "  day: f32," +
      "  alert: f32," +
      "  padding: f32," +
      "};" +
      "@group(0) @binding(0) var<uniform> uniforms: Uniforms;" +
      "struct VertexOut {" +
      "  @builtin(position) position: vec4f," +
      "  @location(0) uv: vec2f," +
      "};" +
      "@vertex fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {" +
      "  var positions = array<vec2f, 6>(" +
      "    vec2f(-1.0, -1.0)," +
      "    vec2f( 1.0, -1.0)," +
      "    vec2f(-1.0,  1.0)," +
      "    vec2f(-1.0,  1.0)," +
      "    vec2f( 1.0, -1.0)," +
      "    vec2f( 1.0,  1.0)" +
      "  );" +
      "  var out: VertexOut;" +
      "  let pos = positions[vertexIndex];" +
      "  out.position = vec4f(pos, 0.0, 1.0);" +
      "  out.uv = (pos + vec2f(1.0, 1.0)) * 0.5;" +
      "  return out;" +
      "}" +
      "fn gridLine(uv: vec2f, scale: f32, thickness: f32) -> f32 {" +
      "  let grid = abs(fract(uv * scale) - vec2f(0.5, 0.5));" +
      "  let line = min(grid.x, grid.y);" +
      "  return smoothstep(thickness, 0.0, line);" +
      "}" +
      "@fragment fn fsMain(in: VertexOut) -> @location(0) vec4f {" +
      "  let uv = in.uv;" +
      "  let p = uv * 2.0 - vec2f(1.0, 1.0);" +
      "  let vignette = smoothstep(1.28, 0.18, dot(p, p));" +
      "  let gridFine = gridLine(uv + vec2f(uniforms.time * 0.006, 0.0), 18.0, 0.028);" +
      "  let gridCoarse = gridLine(uv, 6.0, 0.05);" +
      "  let sweep = exp(-abs(fract(uniforms.time * 0.05 + uv.y * 0.85) - 0.5) * 22.0);" +
      "  let riverY = 0.60 + sin(uv.x * 8.0 + uniforms.time * 0.15) * 0.02 + sin(uv.x * 18.0) * 0.008;" +
      "  let river = exp(-pow((uv.y - riverY) / 0.038, 2.0));" +
      "  let contour = 0.5 + 0.5 * sin(uv.x * 24.0 + uv.y * 18.0 + uniforms.time * 0.12);" +
      "  let dayTop = vec3f(0.018, 0.045, 0.07);" +
      "  let dayBottom = vec3f(0.010, 0.016, 0.022);" +
      "  let nightTop = vec3f(0.013, 0.018, 0.026);" +
      "  let nightBottom = vec3f(0.026, 0.018, 0.022);" +
      "  var col = mix(mix(nightTop, nightBottom, uv.y), mix(dayTop, dayBottom, uv.y), clamp(uniforms.day, 0.0, 1.0));" +
      "  col += gridCoarse * mix(vec3f(0.05, 0.08, 0.09), vec3f(0.15, 0.06, 0.03), uniforms.alert) * 0.28;" +
      "  col += gridFine * vec3f(0.10, 0.15, 0.18) * 0.14;" +
      "  col += river * vec3f(0.04, 0.18, 0.26);" +
      "  col += sweep * mix(vec3f(0.0, 0.08, 0.10), vec3f(0.22, 0.08, 0.04), uniforms.alert) * 0.08;" +
      "  col += contour * 0.008;" +
      "  if (uniforms.weather > 0.5 && uniforms.weather < 1.5) {" +
      "    col += vec3f(0.026, 0.028, 0.03) * 0.4;" +
      "  } else if (uniforms.weather >= 1.5 && uniforms.weather < 2.5) {" +
      "    col += vec3f(0.03, 0.05, 0.08) * 0.55;" +
      "  } else if (uniforms.weather >= 2.5 && uniforms.weather < 3.5) {" +
      "    col += vec3f(0.05, 0.09, 0.16) * 0.62;" +
      "  } else if (uniforms.weather >= 3.5) {" +
      "    col += vec3f(0.07, 0.08, 0.11) * 0.75;" +
      "  }" +
      "  col *= vignette;" +
      "  return vec4f(col, 1.0);" +
      "}";

    var uniformBuffer = device.createBuffer({
      size: 32,
      usage: global.GPUBufferUsage.UNIFORM | global.GPUBufferUsage.COPY_DST,
    });
    var shaderModule = device.createShaderModule({ code: shaderCode });
    var pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vsMain",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fsMain",
        targets: [{ format: format }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });
    var bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
      ],
    });

    this.gpu = {
      backend: "webgpu",
      device: device,
      context: context,
      format: format,
      pipeline: pipeline,
      bindGroup: bindGroup,
      uniformBuffer: uniformBuffer,
      uniformData: new Float32Array(8),
    };
    this.configureWebGpuSurface();

    var renderer = this;
    device.lost.then(function () {
      renderer.gpu = null;
      renderer.initWebGl();
    });
  };

  SimulationSceneRenderer.prototype.configureWebGpuSurface = function () {
    if (!this.gpu || this.gpu.backend !== "webgpu") return;
    this.gpu.context.configure({
      device: this.gpu.device,
      format: this.gpu.format,
      alphaMode: "opaque",
    });
  };

  SimulationSceneRenderer.prototype.initWebGl = function () {
    var gl =
      this.gpuCanvas.getContext("webgl", {
        alpha: false,
        antialias: false,
        desynchronized: true,
        powerPreference: "high-performance",
      }) ||
      this.gpuCanvas.getContext("experimental-webgl", {
        alpha: false,
        antialias: false,
        desynchronized: true,
        powerPreference: "high-performance",
      });

    if (!gl) {
      this.gpu = null;
      return;
    }

    var vertexSource =
      "attribute vec2 a_position;" +
      "varying vec2 v_uv;" +
      "void main() {" +
      "  v_uv = (a_position + 1.0) * 0.5;" +
      "  gl_Position = vec4(a_position, 0.0, 1.0);" +
      "}";

    var fragmentSource =
      "precision mediump float;" +
      "varying vec2 v_uv;" +
      "uniform vec2 u_resolution;" +
      "uniform float u_time;" +
      "uniform float u_weather;" +
      "uniform float u_day;" +
      "uniform float u_alert;" +
      "float gridLine(vec2 uv, float scale, float thickness) {" +
      "  vec2 grid = abs(fract(uv * scale) - 0.5);" +
      "  float line = min(grid.x, grid.y);" +
      "  return smoothstep(thickness, 0.0, line);" +
      "}" +
      "void main() {" +
      "  vec2 uv = v_uv;" +
      "  vec2 p = uv * 2.0 - 1.0;" +
      "  float vignette = smoothstep(1.28, 0.18, dot(p, p));" +
      "  float gridFine = gridLine(uv + vec2(u_time * 0.006, 0.0), 18.0, 0.028);" +
      "  float gridCoarse = gridLine(uv, 6.0, 0.05);" +
      "  float sweep = exp(-abs(fract(u_time * 0.05 + uv.y * 0.85) - 0.5) * 22.0);" +
      "  float riverY = 0.60 + sin(uv.x * 8.0 + u_time * 0.15) * 0.02 + sin(uv.x * 18.0) * 0.008;" +
      "  float river = exp(-pow((uv.y - riverY) / 0.038, 2.0));" +
      "  float contour = 0.5 + 0.5 * sin(uv.x * 24.0 + uv.y * 18.0 + u_time * 0.12);" +
      "  vec3 dayTop = vec3(0.018, 0.045, 0.07);" +
      "  vec3 dayBottom = vec3(0.010, 0.016, 0.022);" +
      "  vec3 nightTop = vec3(0.013, 0.018, 0.026);" +
      "  vec3 nightBottom = vec3(0.026, 0.018, 0.022);" +
      "  vec3 col = mix(mix(nightTop, nightBottom, uv.y), mix(dayTop, dayBottom, uv.y), clamp(u_day, 0.0, 1.0));" +
      "  col += gridCoarse * mix(vec3(0.05, 0.08, 0.09), vec3(0.15, 0.06, 0.03), u_alert) * 0.28;" +
      "  col += gridFine * vec3(0.10, 0.15, 0.18) * 0.14;" +
      "  col += river * vec3(0.04, 0.18, 0.26);" +
      "  col += sweep * mix(vec3(0.0, 0.08, 0.10), vec3(0.22, 0.08, 0.04), u_alert) * 0.08;" +
      "  col += contour * 0.008;" +
      "  if (u_weather > 0.5 && u_weather < 1.5) {" +
      "    col += vec3(0.026, 0.028, 0.03) * 0.4;" +
      "  } else if (u_weather >= 1.5 && u_weather < 2.5) {" +
      "    col += vec3(0.03, 0.05, 0.08) * 0.55;" +
      "  } else if (u_weather >= 2.5 && u_weather < 3.5) {" +
      "    col += vec3(0.05, 0.09, 0.16) * 0.62;" +
      "  } else if (u_weather >= 3.5) {" +
      "    col += vec3(0.07, 0.08, 0.11) * 0.75;" +
      "  }" +
      "  col *= vignette;" +
      "  gl_FragColor = vec4(col, 1.0);" +
      "}";

    try {
      var program = createProgram(gl, vertexSource, fragmentSource);
      var buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([
          -1, -1,
          1, -1,
          -1, 1,
          -1, 1,
          1, -1,
          1, 1,
        ]),
        gl.STATIC_DRAW,
      );
      this.gpu = {
        backend: "webgl",
        gl: gl,
        program: program,
        buffer: buffer,
        locations: {
          position: gl.getAttribLocation(program, "a_position"),
          resolution: gl.getUniformLocation(program, "u_resolution"),
          time: gl.getUniformLocation(program, "u_time"),
          weather: gl.getUniformLocation(program, "u_weather"),
          day: gl.getUniformLocation(program, "u_day"),
          alert: gl.getUniformLocation(program, "u_alert"),
        },
      };
    } catch (error) {
      console.warn("Scene renderer GPU init failed:", error);
      this.gpu = null;
    }
  };

  SimulationSceneRenderer.prototype.seedAmbientParticles = function () {
    this.ambientParticles = [];
    for (var index = 0; index < 52; index += 1) {
      this.ambientParticles.push({
        x: Math.random(),
        y: Math.random(),
        vx: (Math.random() - 0.5) * 0.006,
        vy: 0.004 + Math.random() * 0.01,
        alpha: 0.08 + Math.random() * 0.22,
        size: 0.6 + Math.random() * 2.2,
      });
    }
  };

  SimulationSceneRenderer.prototype.syncWeatherParticles = function () {
    var count = 0;
    if (this.weather === "snow") count = 28;
    if (this.weather === "ice_storm") count = 34;
    if (this.weather === "blizzard") count = 54;

    this.weatherParticles = [];
    for (var index = 0; index < count; index += 1) {
      var heavy = this.weather === "blizzard";
      this.weatherParticles.push({
        x: Math.random(),
        y: Math.random(),
        vx:
          this.weather === "ice_storm"
            ? -0.22 - Math.random() * 0.12
            : heavy
              ? -0.08 - Math.random() * 0.08
              : -0.03 - Math.random() * 0.03,
        vy:
          this.weather === "ice_storm"
            ? 0.26 + Math.random() * 0.16
            : heavy
              ? 0.13 + Math.random() * 0.09
              : 0.05 + Math.random() * 0.07,
        size:
          this.weather === "ice_storm"
            ? 1.4 + Math.random() * 1.8
            : heavy
              ? 1.0 + Math.random() * 2.0
              : 0.8 + Math.random() * 1.4,
        alpha:
          this.weather === "ice_storm"
            ? 0.28 + Math.random() * 0.34
            : 0.26 + Math.random() * 0.42,
      });
    }
  };

  SimulationSceneRenderer.prototype.setMissionContext = function (scenario, strategy, twinMode) {
    var scenarioName =
      scenario && typeof scenario === "object"
        ? scenario.name || scenario.key || "Quebec Blood Network"
        : "Quebec Blood Network";
    var strategyName =
      strategy && typeof strategy === "object"
        ? strategy.name || strategy.key || "Simulation Control"
        : strategy || "Simulation Control";
    this.meta = {
      scenarioName: scenarioName,
      strategyName: strategyName,
      mode: twinMode ? "Digital Twin" : "Simulation",
    };

    if (this.refs.title) this.refs.title.textContent = scenarioName || "Quebec Blood Network";
    if (this.refs.mode) this.refs.mode.textContent = this.meta.mode;
    if (this.refs.subtitle) {
      this.refs.subtitle.textContent =
        strategyName +
        " controller engaged across the live Quebec logistics field.";
    }
  };

  SimulationSceneRenderer.prototype.setOperationalFrame = function (frame) {
    this.frameData = {
      score: Number(frame.score || 0),
      shortage_rate: Number(frame.shortage_rate || 0),
      budget_remaining: Number(frame.budget_remaining || frame.budget_total || 0),
      total_donated: Number(frame.total_donated || 0),
      total_transfused: Number(frame.total_transfused || 0),
      total_expired: Number(frame.total_expired || 0),
      total_hours: Number(frame.total_hours || 0),
      progress: Number(frame.progress || 0),
    };

    if (frame.weather) this.setWeather(frame.weather);
    if (frame.hour != null) this.setHour(frame.hour);

    this.alertTarget = clamp(this.frameData.shortage_rate / 12, 0, 1);

    if (this.refs.clock) {
      this.refs.clock.textContent =
        "Hour " +
        Math.round(Number(frame.hour || 0)) +
        (this.frameData.total_hours ? " / " + Math.round(this.frameData.total_hours) : "");
    }
    if (this.refs.weather) {
      this.refs.weather.textContent =
        WEATHER_LABELS[this.weather] || titleCase(this.weather);
    }
    if (this.refs.pressure) {
      this.refs.pressure.textContent = formatPercent(this.frameData.shortage_rate);
    }

    this.refreshContextRow();
    this.refreshMiniMetrics();
    this.refreshInspector();
  };

  SimulationSceneRenderer.prototype.setWeather = function (weather) {
    var next = normalizeWeatherState(weather);
    if (next === this.weather) return;
    this.weather = next;
    this.syncWeatherParticles();
    if (this.refs.weather) {
      this.refs.weather.textContent = WEATHER_LABELS[this.weather] || titleCase(this.weather);
    }
    this.refreshContextRow();
  };

  SimulationSceneRenderer.prototype.setHour = function (hour) {
    this.hour = Number(hour || 0);
    var hourOfDay = ((this.hour % 24) + 24) % 24;
    var daylight = hourOfDay >= 6 && hourOfDay < 20 ? 1 : 0.32;
    this.dayFactorTarget = daylight;
  };

  SimulationSceneRenderer.prototype.setWorldContext = function (world) {
    world = world || {};
    this.worldContext = {
      date: world.date_display || world.date || "",
      season: world.season ? titleCase(world.season) : "",
      weatherLabel: formatWorldWeather(world.weather_pattern || this.weather, world.temperature_c),
    };
    if (world.weather_pattern) this.setWeather(world.weather_pattern);
    this.refreshContextRow();
  };

  SimulationSceneRenderer.prototype.refreshContextRow = function () {
    var simulatedDay = "Simulation Day " + (Math.floor(this.hour / 24) + 1);
    var dateText = this.worldContext.date || simulatedDay;
    var seasonText = this.worldContext.season || this.meta.mode || "Simulation";
    var weatherText =
      this.worldContext.weatherLabel ||
      (WEATHER_LABELS[this.weather] || titleCase(this.weather));

    if (this.refs.date) this.refs.date.textContent = dateText;
    if (this.refs.season) this.refs.season.textContent = seasonText;
    if (this.refs.worldWeather) this.refs.worldWeather.textContent = weatherText;
  };

  function centerRole(center) {
    return center.role || center.type || "default";
  }

  SimulationSceneRenderer.prototype.setCenters = function (centers, options) {
    options = options || {};
    var donorCountMap = options.donorCountMap || {};
    var maxInventory = 1;
    var nextCenters = [];
    var index;

    for (index = 0; index < centers.length; index += 1) {
      var inventory = centers[index].inventory || {};
      maxInventory = Math.max(
        maxInventory,
        inventory.RBC || 0,
        inventory.PLATELETS || 0,
        inventory.PLASMA || 0,
      );
    }

    for (index = 0; index < centers.length; index += 1) {
      var center = centers[index];
      var position = latLonToPosition(center.lat || 46.8, center.lon || -71.25);
      var inv = center.inventory || {};
      var stats = center.stats || {};
      var rbc = Number(inv.RBC || 0);
      var platelets = Number(inv.PLATELETS || 0);
      var plasma = Number(inv.PLASMA || 0);
      var total = Number(center.total_units != null ? center.total_units : rbc + platelets + plasma);
      var donorInfo = donorCountMap[center.name] || {};
      var queue = Number(donorInfo.queue_length || 0) + Number(donorInfo.active_donors || 0);
      var health = clamp(total / (maxInventory * 3), 0, 1);
      var role = centerRole(center);
      nextCenters.push({
        raw: center,
        index: index,
        name: center.name || "Center",
        shortName: shortCenterName(center.name),
        type: center.type || "default",
        role: role,
        xNorm: position.x / 100,
        yNorm: position.y / 100,
        x: position.x / 100 * this.cssWidth,
        y: position.y / 100 * this.cssHeight,
        radius:
          role === "hospital"
            ? 18
            : role === "mobile"
              ? 14
              : 16,
        inventory: {
          rbc: rbc,
          platelets: platelets,
          plasma: plasma,
        },
        stats: {
          donated: Number(stats.donated || 0),
          transfused: Number(stats.transfused || 0),
          expired: Number(stats.expired || 0),
          rejected: Number(stats.rejected || 0),
          no_show: Number(stats.no_show || 0),
        },
        total: total,
        queue: queue,
        health: health,
      });
    }

    this.centers = nextCenters;
    this.baseLinks = this.buildBaseLinks();
    this.staticDirty = true;

    if (options.donatedDiff > 0) this.spawnDonationEffects(options.donatedDiff);
    if (options.transfusedDiff > 0) this.spawnTransferEffects(options.transfusedDiff);
    this.refreshMiniMetrics();
    this.refreshInspector();
  };

  SimulationSceneRenderer.prototype.buildBaseLinks = function () {
    var links = [];
    var centers = this.centers.slice();
    var banks = centers.filter(function (center) {
      return center.role === "blood_bank" || center.role === "mobile";
    });
    var hospitals = centers.filter(function (center) {
      return center.role === "hospital";
    });

    for (var index = 0; index < hospitals.length; index += 1) {
      var best = null;
      var bestDistance = Infinity;
      for (var sourceIndex = 0; sourceIndex < banks.length; sourceIndex += 1) {
        var candidate = banks[sourceIndex];
        var distance = distanceSquared(
          hospitals[index].x,
          hospitals[index].y,
          candidate.x,
          candidate.y,
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          best = candidate;
        }
      }
      if (best) {
        links.push({
          from: best,
          to: hospitals[index],
        });
      }
    }

    return links;
  };

  SimulationSceneRenderer.prototype.spawnPulse = function (x, y, color, strength) {
    this.pulses.push({
      x: x,
      y: y,
      color: color,
      radius: 14 + Math.random() * 18,
      life: 0.9,
      maxLife: 0.9,
      strength: strength || 1,
    });
  };

  SimulationSceneRenderer.prototype.spawnDonationEffects = function (donatedDiff) {
    var candidates = this.centers.filter(function (center) {
      return center.role === "blood_bank" || center.role === "mobile";
    });
    if (!candidates.length) return;

    var count = clamp(Math.round(donatedDiff / 8) || 1, 1, 10);
    for (var index = 0; index < count; index += 1) {
      var center = candidates[index % candidates.length];
      var angle = Math.random() * Math.PI * 2;
      var speed = 12 + Math.random() * 18;
      this.donationBursts.push({
        x: center.x,
        y: center.y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 22,
        size: 2 + Math.random() * 4,
        life: 1.0 + Math.random() * 0.4,
        maxLife: 1.4,
        color: "rgba(125, 242, 208, 0.92)",
      });
      this.spawnPulse(center.x, center.y, "rgba(125, 242, 208, 0.32)", 1);
    }
  };

  SimulationSceneRenderer.prototype.spawnTransferEffects = function (transfusedDiff) {
    var banks = this.centers.filter(function (center) {
      return center.role === "blood_bank" || center.role === "mobile";
    });
    var hospitals = this.centers.filter(function (center) {
      return center.role === "hospital";
    });
    if (!banks.length || !hospitals.length) return;

    var count = clamp(Math.round(transfusedDiff / 10) || 1, 1, 8);
    for (var index = 0; index < count; index += 1) {
      var from = banks[index % banks.length];
      var to = hospitals[(index + Math.floor(Math.random() * hospitals.length)) % hospitals.length];
      var curveLift = 40 + Math.random() * 80;
      this.transferEffects.push({
        fromX: from.x,
        fromY: from.y,
        toX: to.x,
        toY: to.y,
        controlX: (from.x + to.x) * 0.5 + (Math.random() - 0.5) * 70,
        controlY: Math.min(from.y, to.y) - curveLift,
        progress: Math.random() * 0.25,
        speed: 0.48 + Math.random() * 0.46,
        color: "rgba(255, 192, 125, 0.95)",
      });
      this.spawnPulse(to.x, to.y, "rgba(255, 192, 125, 0.24)", 1.2);
    }
  };

  SimulationSceneRenderer.prototype.handlePointerMove = function (event) {
    if (!this.canvas) return;
    var rect = this.canvas.getBoundingClientRect();
    var pointerX = event.clientX - rect.left;
    var pointerY = event.clientY - rect.top;
    var nextHover = -1;
    var bestDistance = Infinity;

    for (var index = 0; index < this.centers.length; index += 1) {
      var center = this.centers[index];
      var hitRadius = center.radius + 16;
      var distance = distanceSquared(pointerX, pointerY, center.x, center.y);
      if (distance <= hitRadius * hitRadius && distance < bestDistance) {
        bestDistance = distance;
        nextHover = index;
      }
    }

    if (nextHover !== this.hoverIndex) {
      this.hoverIndex = nextHover;
      this.refreshInspector();
    }
  };

  SimulationSceneRenderer.prototype.handlePointerLeave = function () {
    if (this.hoverIndex !== -1) {
      this.hoverIndex = -1;
      this.refreshInspector();
    }
  };

  SimulationSceneRenderer.prototype.refreshMiniMetrics = function () {
    if (!this.refs.miniMetrics) return;
    var totalUnits = 0;
    var critical = 0;
    for (var index = 0; index < this.centers.length; index += 1) {
      totalUnits += this.centers[index].total;
      if (this.centers[index].health < 0.18) critical += 1;
    }

    this.refs.miniMetrics.innerHTML =
      '<div class="scene-mini-card"><span>Live Units</span><strong>' +
      escapeHtml(formatInteger(totalUnits)) +
      "</strong></div>" +
      '<div class="scene-mini-card"><span>Donated</span><strong>' +
      escapeHtml(formatInteger(this.frameData.total_donated)) +
      "</strong></div>" +
      '<div class="scene-mini-card"><span>Transfused</span><strong>' +
      escapeHtml(formatInteger(this.frameData.total_transfused)) +
      "</strong></div>" +
      '<div class="scene-mini-card"><span>Critical Sites</span><strong>' +
      escapeHtml(formatInteger(critical)) +
      "</strong></div>" +
      '<div class="scene-mini-card"><span>Budget</span><strong>$' +
      escapeHtml(formatNumber(this.frameData.budget_remaining)) +
      "</strong></div>";
  };

  SimulationSceneRenderer.prototype.refreshInspector = function () {
    if (!this.refs.inspector) return;
    if (this.shell) {
      this.shell.classList.toggle("scene-node-focus", this.hoverIndex >= 0);
    }

    var selected = null;
    var heading = "Critical Watch";

    if (this.hoverIndex >= 0 && this.centers[this.hoverIndex]) {
      selected = this.centers[this.hoverIndex];
      heading = "Hovered Site";
    } else if (this.centers.length) {
      selected = this.centers[0];
      for (var index = 1; index < this.centers.length; index += 1) {
        if (this.centers[index].health < selected.health) {
          selected = this.centers[index];
        }
      }
    }

    if (!selected) {
      this.refs.inspector.innerHTML =
        '<div class="scene-inspector-empty">Simulation feed idle. Launch a mission to populate the network graph.</div>';
      return;
    }

    var typeStyle = CENTER_STYLE[selected.type] || CENTER_STYLE.default;
    var inventoryTotal =
      selected.inventory.rbc + selected.inventory.platelets + selected.inventory.plasma;
    var healthLabel =
      selected.health < 0.18
        ? "Critical reserve"
        : selected.health < 0.42
          ? "Tight reserve"
          : "Stable reserve";

    this.refs.inspector.innerHTML =
      '<div class="scene-inspector-head">' +
      '<span class="scene-inspector-kicker">' +
      escapeHtml(heading) +
      "</span>" +
      '<strong class="scene-inspector-title">' +
      escapeHtml(selected.name) +
      "</strong>" +
      '<span class="scene-inspector-type" style="color:' +
      escapeHtml(typeStyle.edge) +
      '">' +
      escapeHtml(typeStyle.label) +
      "</span>" +
      "</div>" +
      '<div class="scene-inspector-grid">' +
      '<div class="scene-inspector-metric"><span>Reserve</span><strong>' +
      escapeHtml(healthLabel) +
      "</strong></div>" +
      '<div class="scene-inspector-metric"><span>Total Units</span><strong>' +
      escapeHtml(formatInteger(inventoryTotal)) +
      "</strong></div>" +
      '<div class="scene-inspector-metric"><span>Queue</span><strong>' +
      escapeHtml(formatInteger(selected.queue)) +
      "</strong></div>" +
      '<div class="scene-inspector-metric"><span>Transfused</span><strong>' +
      escapeHtml(formatInteger(selected.stats.transfused)) +
      "</strong></div>" +
      "</div>" +
      '<div class="scene-inspector-mix">' +
      '<div class="scene-mix-row"><span>RBC</span><div class="scene-mix-bar"><i style="width:' +
      clamp(selected.inventory.rbc / Math.max(inventoryTotal, 1), 0, 1) * 100 +
      '%"></i></div><strong>' +
      escapeHtml(formatInteger(selected.inventory.rbc)) +
      "</strong></div>" +
      '<div class="scene-mix-row"><span>PLT</span><div class="scene-mix-bar is-platelets"><i style="width:' +
      clamp(selected.inventory.platelets / Math.max(inventoryTotal, 1), 0, 1) * 100 +
      '%"></i></div><strong>' +
      escapeHtml(formatInteger(selected.inventory.platelets)) +
      "</strong></div>" +
      '<div class="scene-mix-row"><span>PLS</span><div class="scene-mix-bar is-plasma"><i style="width:' +
      clamp(selected.inventory.plasma / Math.max(inventoryTotal, 1), 0, 1) * 100 +
      '%"></i></div><strong>' +
      escapeHtml(formatInteger(selected.inventory.plasma)) +
      "</strong></div>" +
      "</div>" +
      '<div class="scene-inspector-foot">Rejected ' +
      escapeHtml(formatInteger(selected.stats.rejected)) +
      " · No-show " +
      escapeHtml(formatInteger(selected.stats.no_show)) +
      " · Expired " +
      escapeHtml(formatInteger(selected.stats.expired)) +
      "</div>";
  };

  SimulationSceneRenderer.prototype.clearTransientEffects = function () {
    this.transferEffects = [];
    this.donationBursts = [];
    this.pulses = [];
  };

  SimulationSceneRenderer.prototype.reset = function () {
    this.centers = [];
    this.baseLinks = [];
    this.hoverIndex = -1;
    this.weather = "clear";
    this.hour = 0;
    this.dayFactor = 1;
    this.dayFactorTarget = 1;
    this.alertLevel = 0;
    this.alertTarget = 0;
    this.frameData = {
      score: 0,
      shortage_rate: 0,
      budget_remaining: 0,
      total_donated: 0,
      total_transfused: 0,
      total_expired: 0,
      total_hours: 0,
      progress: 0,
    };
    this.clearTransientEffects();
    this.syncWeatherParticles();
    this.staticDirty = true;
    if (this.refs.clock) this.refs.clock.textContent = "Hour 0";
    if (this.refs.weather) this.refs.weather.textContent = WEATHER_LABELS.clear;
    if (this.refs.pressure) this.refs.pressure.textContent = "0.0%";
    this.refreshMiniMetrics();
    this.refreshInspector();
    if (this.ctx) {
      this.ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);
    }
  };

  SimulationSceneRenderer.prototype.resize = function () {
    if (!this.board || !this.shell) return;
    var rect = this.board.getBoundingClientRect();
    var width = Math.max(1, Math.floor(rect.width));
    var height = Math.max(1, Math.floor(rect.height));
    var area = width * height;
    var renderScaleCap = 1.5;
    if (area > 1200000) {
      renderScaleCap = 1.15;
    } else if (area > 850000) {
      renderScaleCap = 1.25;
    } else if (area > 550000) {
      renderScaleCap = 1.35;
    }
    this.cssWidth = width;
    this.cssHeight = height;
    this.dpr = clamp(global.devicePixelRatio || 1, 1, renderScaleCap);
    this.width = Math.floor(width * this.dpr);
    this.height = Math.floor(height * this.dpr);

    this.gpuCanvas.width = this.width;
    this.gpuCanvas.height = this.height;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.staticCanvas.width = this.width;
    this.staticCanvas.height = this.height;
    if (this.gpu && this.gpu.backend === "webgpu") {
      this.configureWebGpuSurface();
    }

    this.gpuCanvas.style.width = width + "px";
    this.gpuCanvas.style.height = height + "px";
    this.canvas.style.width = width + "px";
    this.canvas.style.height = height + "px";

    if (this.ctx) {
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }
    if (this.staticCtx) {
      this.staticCtx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }

    for (var index = 0; index < this.centers.length; index += 1) {
      this.centers[index].x = this.centers[index].xNorm * this.cssWidth;
      this.centers[index].y = this.centers[index].yNorm * this.cssHeight;
    }
    this.baseLinks = this.buildBaseLinks();
    this.staticDirty = true;

    if (this.active) {
      this.scheduleFrame();
    } else {
      this.draw();
    }
  };

  SimulationSceneRenderer.prototype.setActive = function (active) {
    this.active = !!active;
    if (!this.active) {
      if (this.frameId) {
        global.cancelAnimationFrame(this.frameId);
        this.frameId = 0;
      }
      return;
    }
    this.resize();
    this.scheduleFrame();
  };

  SimulationSceneRenderer.prototype.scheduleFrame = function () {
    if (this.frameId) return;
    this.frameId = global.requestAnimationFrame(this.boundFrame);
  };

  SimulationSceneRenderer.prototype.frame = function (timestamp) {
    this.frameId = 0;
    var now = timestamp || 0;
    if (!this.lastTime) this.lastTime = now;
    var delta = clamp((now - this.lastTime) / 1000, 0.001, 0.033);
    this.lastTime = now;
    this.elapsed += delta;

    this.dayFactor += (this.dayFactorTarget - this.dayFactor) * Math.min(1, delta * 2.6);
    this.alertLevel += (this.alertTarget - this.alertLevel) * Math.min(1, delta * 2.4);
    this.advanceEffects(delta);
    this.draw();

    if (this.active) this.scheduleFrame();
  };

  SimulationSceneRenderer.prototype.advanceEffects = function (delta) {
    var index;

    for (index = 0; index < this.ambientParticles.length; index += 1) {
      var ambient = this.ambientParticles[index];
      ambient.x += ambient.vx * delta;
      ambient.y += ambient.vy * delta;
      if (ambient.x < -0.05) ambient.x = 1.05;
      if (ambient.x > 1.05) ambient.x = -0.05;
      if (ambient.y > 1.08) ambient.y = -0.08;
    }

    for (index = this.weatherParticles.length - 1; index >= 0; index -= 1) {
      var particle = this.weatherParticles[index];
      particle.x += particle.vx * delta;
      particle.y += particle.vy * delta;
      if (particle.x < -0.15) particle.x = 1.05;
      if (particle.y > 1.15) {
        particle.x = Math.random();
        particle.y = -0.08;
      }
    }

    for (index = this.transferEffects.length - 1; index >= 0; index -= 1) {
      this.transferEffects[index].progress += this.transferEffects[index].speed * delta;
      if (this.transferEffects[index].progress >= 1.08) {
        this.transferEffects.splice(index, 1);
      }
    }

    for (index = this.donationBursts.length - 1; index >= 0; index -= 1) {
      var burst = this.donationBursts[index];
      burst.life -= delta;
      burst.x += burst.vx * delta;
      burst.y += burst.vy * delta;
      burst.vy += 30 * delta;
      if (burst.life <= 0) this.donationBursts.splice(index, 1);
    }

    for (index = this.pulses.length - 1; index >= 0; index -= 1) {
      this.pulses[index].life -= delta;
      if (this.pulses[index].life <= 0) this.pulses.splice(index, 1);
    }
  };

  SimulationSceneRenderer.prototype.draw = function () {
    if (!this.ctx) return;
    var hasGpuBackground = this.renderGpuBackground();
    var ctx = this.ctx;
    ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);
    if (!hasGpuBackground) this.drawFallbackBackground(ctx);
    this.drawAmbientParticles(ctx);
    if (this.staticDirty) this.renderStaticLayer();
    ctx.drawImage(this.staticCanvas, 0, 0, this.cssWidth, this.cssHeight);
    this.drawTransferEffects(ctx);
    this.drawPulses(ctx);
    this.drawDonationBursts(ctx);
    this.drawWeatherParticles(ctx);
    this.drawHoverOverlay(ctx);
  };

  SimulationSceneRenderer.prototype.renderStaticLayer = function () {
    if (!this.staticCtx) return;
    var ctx = this.staticCtx;
    ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);
    this.drawBaseLinks(ctx, false);
    this.drawCenterNodes(ctx, -1);
    this.staticDirty = false;
  };

  SimulationSceneRenderer.prototype.renderGpuBackground = function () {
    if (!this.gpu) return false;

    if (this.gpu.backend === "webgpu") {
      var data = this.gpu.uniformData;
      data[0] = this.width;
      data[1] = this.height;
      data[2] = this.elapsed;
      data[3] = WEATHER_INDEX[this.weather] || 0;
      data[4] = this.dayFactor;
      data[5] = this.alertLevel;
      data[6] = 0;
      data[7] = 0;
      this.gpu.device.queue.writeBuffer(this.gpu.uniformBuffer, 0, data);

      var encoder = this.gpu.device.createCommandEncoder();
      var view = this.gpu.context.getCurrentTexture().createView();
      var pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(this.gpu.pipeline);
      pass.setBindGroup(0, this.gpu.bindGroup);
      pass.draw(6, 1, 0, 0);
      pass.end();
      this.gpu.device.queue.submit([encoder.finish()]);
      return true;
    }

    if (this.gpu.backend === "webgl") {
      var gl = this.gpu.gl;
      var program = this.gpu.program;
      gl.viewport(0, 0, this.width, this.height);
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gpu.buffer);
      gl.enableVertexAttribArray(this.gpu.locations.position);
      gl.vertexAttribPointer(this.gpu.locations.position, 2, gl.FLOAT, false, 0, 0);
      gl.uniform2f(this.gpu.locations.resolution, this.width, this.height);
      gl.uniform1f(this.gpu.locations.time, this.elapsed);
      gl.uniform1f(this.gpu.locations.weather, WEATHER_INDEX[this.weather] || 0);
      gl.uniform1f(this.gpu.locations.day, this.dayFactor);
      gl.uniform1f(this.gpu.locations.alert, this.alertLevel);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      return true;
    }

    return false;
  };

  SimulationSceneRenderer.prototype.drawFallbackBackground = function (ctx) {
    var gradient = ctx.createLinearGradient(0, 0, 0, this.cssHeight);
    gradient.addColorStop(0, "#062550");
    gradient.addColorStop(0.58, "#04142d");
    gradient.addColorStop(1, "#020817");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, this.cssWidth, this.cssHeight);

    ctx.save();
    ctx.strokeStyle = "rgba(96, 165, 250, 0.08)";
    ctx.lineWidth = 1;
    for (var x = 0; x <= this.cssWidth; x += 56) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.cssHeight);
      ctx.stroke();
    }
    for (var y = 0; y <= this.cssHeight; y += 56) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.cssWidth, y);
      ctx.stroke();
    }
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawAmbientParticles = function (ctx) {
    ctx.save();
    for (var index = 0; index < this.ambientParticles.length; index += 1) {
      var particle = this.ambientParticles[index];
      ctx.fillStyle = "rgba(145, 244, 255," + particle.alpha.toFixed(3) + ")";
      ctx.beginPath();
      ctx.arc(
        particle.x * this.cssWidth,
        particle.y * this.cssHeight,
        particle.size,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawBaseLinks = function (ctx, animated) {
    if (!this.baseLinks.length) return;
    ctx.save();
    ctx.setLineDash([7, 12]);
    ctx.lineDashOffset = animated ? -this.elapsed * 28 : 0;
    ctx.lineCap = "round";
    for (var index = 0; index < this.baseLinks.length; index += 1) {
      var link = this.baseLinks[index];
      var midX = (link.from.x + link.to.x) * 0.5;
      var midY = (link.from.y + link.to.y) * 0.5;
      var dx = link.to.x - link.from.x;
      var dy = link.to.y - link.from.y;
      var length = Math.sqrt(dx * dx + dy * dy) || 1;
      var curve = 34 + (index % 3) * 12;
      var controlX = midX - (dy / length) * curve;
      var controlY = midY + (dx / length) * curve;

      ctx.strokeStyle = "rgba(255, 255, 255, 0.26)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(link.from.x, link.from.y);
      ctx.quadraticCurveTo(controlX, controlY, link.to.x, link.to.y);
      ctx.stroke();

      ctx.strokeStyle = "rgba(46, 160, 255, 0.62)";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(link.from.x, link.from.y);
      ctx.quadraticCurveTo(controlX, controlY, link.to.x, link.to.y);
      ctx.stroke();

      var arrowT = 0.58;
      var arrowPoint = quadraticPoint(
        link.from.x,
        link.from.y,
        controlX,
        controlY,
        link.to.x,
        link.to.y,
        arrowT,
      );
      var tangentX =
        2 * (1 - arrowT) * (controlX - link.from.x) +
        2 * arrowT * (link.to.x - controlX);
      var tangentY =
        2 * (1 - arrowT) * (controlY - link.from.y) +
        2 * arrowT * (link.to.y - controlY);
      var arrowAngle = Math.atan2(tangentY, tangentX);
      ctx.save();
      ctx.translate(arrowPoint.x, arrowPoint.y);
      ctx.rotate(arrowAngle);
      ctx.fillStyle = "rgba(145, 212, 255, 0.92)";
      ctx.shadowBlur = 12;
      ctx.shadowColor = "rgba(46, 160, 255, 0.74)";
      ctx.beginPath();
      ctx.moveTo(8, 0);
      ctx.lineTo(-5, -4);
      ctx.lineTo(-2, 0);
      ctx.lineTo(-5, 4);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawTransferEffects = function (ctx) {
    if (!this.transferEffects.length) return;
    ctx.save();
    for (var index = 0; index < this.transferEffects.length; index += 1) {
      var effect = this.transferEffects[index];
      ctx.setLineDash([8, 10]);
      ctx.lineDashOffset = -(this.elapsed * 42 + effect.progress * 28);
      ctx.lineWidth = 1.1;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.24)";
      ctx.beginPath();
      ctx.moveTo(effect.fromX, effect.fromY);
      ctx.quadraticCurveTo(
        effect.controlX,
        effect.controlY,
        effect.toX,
        effect.toY,
      );
      ctx.stroke();

      ctx.setLineDash([3, 13]);
      ctx.lineDashOffset = -(this.elapsed * 56 + effect.progress * 34);
      ctx.lineWidth = 2.6;
      ctx.strokeStyle = "rgba(255, 152, 40, 0.82)";
      ctx.stroke();

      var marker = quadraticPoint(
        effect.fromX,
        effect.fromY,
        effect.controlX,
        effect.controlY,
        effect.toX,
        effect.toY,
        clamp(effect.progress, 0, 1),
      );
      ctx.setLineDash([]);
      ctx.shadowBlur = 18;
      ctx.shadowColor = "rgba(255, 152, 40, 0.72)";
      ctx.fillStyle = "rgba(255, 152, 40, 0.96)";
      ctx.beginPath();
      ctx.arc(marker.x, marker.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      var markerT = clamp(effect.progress, 0, 1);
      var markerDx =
        2 * (1 - markerT) * (effect.controlX - effect.fromX) +
        2 * markerT * (effect.toX - effect.controlX);
      var markerDy =
        2 * (1 - markerT) * (effect.controlY - effect.fromY) +
        2 * markerT * (effect.toY - effect.controlY);
      ctx.save();
      ctx.translate(marker.x, marker.y);
      ctx.rotate(Math.atan2(markerDy, markerDx));
      ctx.shadowBlur = 14;
      ctx.shadowColor = "rgba(255, 152, 40, 0.8)";
      ctx.fillStyle = "#ff9f1c";
      drawRoundedRectPath(ctx, -9, -6, 13, 9, 2);
      ctx.fill();
      drawRoundedRectPath(ctx, 2, -4, 8, 7, 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#fff7ed";
      ctx.fillRect(-5, -3, 4, 3);
      ctx.fillStyle = "#111827";
      ctx.beginPath();
      ctx.arc(-5, 5, 2, 0, Math.PI * 2);
      ctx.arc(6, 5, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.setLineDash([]);
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawPulses = function (ctx) {
    if (!this.pulses.length) return;
    ctx.save();
    for (var index = 0; index < this.pulses.length; index += 1) {
      var pulse = this.pulses[index];
      var progress = 1 - pulse.life / pulse.maxLife;
      var radius = pulse.radius + progress * 42 * pulse.strength;
      ctx.strokeStyle = pulse.color.replace(/0?\.\d+\)/, (0.28 * (1 - progress)).toFixed(3) + ")");
      ctx.lineWidth = 1.2 + (1 - progress) * 2.5;
      ctx.beginPath();
      ctx.arc(pulse.x, pulse.y, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawCenterNodes = function (ctx, hoveredIndex) {
    if (hoveredIndex == null) hoveredIndex = this.hoverIndex;
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (var index = 0; index < this.centers.length; index += 1) {
      var center = this.centers[index];
      var style = CENTER_STYLE[center.role] || CENTER_STYLE.default;
      var hover = index === hoveredIndex;
      var alertStrength = 1 - center.health;
      var radius = center.radius + (hover ? 2 : 0);

      var halo = ctx.createRadialGradient(center.x, center.y, radius * 0.4, center.x, center.y, radius * 3.2);
      halo.addColorStop(0, style.glow);
      halo.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(center.x, center.y, radius * 3.2, 0, Math.PI * 2);
      ctx.fill();

      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.shadowBlur = 24;
      ctx.shadowColor = style.glow;

      var nodeFill = ctx.createRadialGradient(0, 0, radius * 0.18, 0, 0, radius);
      nodeFill.addColorStop(0, style.fill);
      nodeFill.addColorStop(1, "rgba(11, 16, 22, 0.98)");
      ctx.fillStyle = nodeFill;
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = hover ? 3 : 2;
      ctx.strokeStyle = style.edge;
      ctx.stroke();

      ctx.strokeStyle = style.glow;
      ctx.lineWidth = 7;
      ctx.beginPath();
      ctx.arc(0, 0, radius + 9, 0, Math.PI * 2);
      ctx.stroke();

      if (alertStrength > 0.68) {
        ctx.strokeStyle = "rgba(255, 106, 77, 0.65)";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 5]);
        ctx.lineDashOffset = -this.elapsed * 14;
        ctx.beginPath();
        ctx.arc(0, 0, radius + 7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      this.drawCenterGlyph(ctx, center.role, radius, style.glyph);
      this.drawInventoryStrips(ctx, center, radius);

      if (center.queue > 0) {
        ctx.fillStyle = "rgba(255, 192, 125, 0.96)";
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(radius * 0.86, -radius * 0.84, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#081018";
        ctx.font = "700 11px 'IBM Plex Mono', monospace";
        ctx.fillText(formatInteger(center.queue), radius * 0.86, -radius * 0.84 + 0.5);
      }

      if (hover) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(-radius - 11, -radius - 6);
        ctx.lineTo(-radius - 11, -radius - 18);
        ctx.lineTo(-radius + 1, -radius - 18);
        ctx.moveTo(radius + 11, -radius - 6);
        ctx.lineTo(radius + 11, -radius - 18);
        ctx.lineTo(radius - 1, -radius - 18);
        ctx.moveTo(-radius - 11, radius + 6);
        ctx.lineTo(-radius - 11, radius + 18);
        ctx.lineTo(-radius + 1, radius + 18);
        ctx.moveTo(radius + 11, radius + 6);
        ctx.lineTo(radius + 11, radius + 18);
        ctx.lineTo(radius - 1, radius + 18);
        ctx.stroke();
      }

      ctx.restore();

      ctx.font = "600 12px 'IBM Plex Mono', monospace";
      ctx.fillStyle = hover ? "#ffffff" : "rgba(241, 245, 249, 0.82)";
      ctx.shadowBlur = 16;
      ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
      ctx.fillText(center.shortName, center.x, center.y + radius + 18);
    }

    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawHoverOverlay = function (ctx) {
    if (this.hoverIndex < 0 || !this.centers[this.hoverIndex]) return;
    var center = this.centers[this.hoverIndex];
    var radius = center.radius + 2;

    ctx.save();
    ctx.translate(center.x, center.y);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-radius - 11, -radius - 6);
    ctx.lineTo(-radius - 11, -radius - 18);
    ctx.lineTo(-radius + 1, -radius - 18);
    ctx.moveTo(radius + 11, -radius - 6);
    ctx.lineTo(radius + 11, -radius - 18);
    ctx.lineTo(radius - 1, -radius - 18);
    ctx.moveTo(-radius - 11, radius + 6);
    ctx.lineTo(-radius - 11, radius + 18);
    ctx.lineTo(-radius + 1, radius + 18);
    ctx.moveTo(radius + 11, radius + 6);
    ctx.lineTo(radius + 11, radius + 18);
    ctx.lineTo(radius - 1, radius + 18);
    ctx.stroke();
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawCenterGlyph = function (ctx, type, radius, color) {
    ctx.save();
    ctx.translate(0, -radius * 0.1);
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.7, radius * 0.1);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.fillStyle = "rgba(5, 10, 16, 0.68)";
    ctx.beginPath();
    ctx.arc(0, 0, radius * 0.48, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.8, radius * 0.1);

    if (type === "hospital") {
      ctx.fillStyle = color;
      ctx.font = "900 " + Math.max(18, radius * 0.9).toFixed(0) + "px 'Segoe UI', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("H", 0, radius * 0.02);
      ctx.lineWidth = Math.max(2, radius * 0.09);
      ctx.strokeRect(
        -radius * 0.38,
        -radius * 0.38,
        radius * 0.76,
        radius * 0.76,
      );
      ctx.restore();
      return;
    }

    if (type === "blood_bank") {
      drawRoundedRectPath(
        ctx,
        -radius * 0.34,
        -radius * 0.22,
        radius * 0.68,
        radius * 0.5,
        radius * 0.06,
      );
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-radius * 0.42, -radius * 0.22);
      ctx.lineTo(0, -radius * 0.44);
      ctx.lineTo(radius * 0.42, -radius * 0.22);
      ctx.closePath();
      ctx.stroke();
      for (var column = -1; column <= 1; column += 1) {
        var columnX = column * radius * 0.18;
        ctx.beginPath();
        ctx.moveTo(columnX, -radius * 0.14);
        ctx.lineTo(columnX, radius * 0.18);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(-radius * 0.42, radius * 0.3);
      ctx.lineTo(radius * 0.42, radius * 0.3);
      ctx.stroke();
      ctx.restore();
      return;
    }

    if (type === "mobile") {
      drawRoundedRectPath(
        ctx,
        -radius * 0.31,
        -radius * 0.08,
        radius * 0.46,
        radius * 0.24,
        radius * 0.06,
      );
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(radius * 0.15, -radius * 0.08);
      ctx.lineTo(radius * 0.28, -radius * 0.08);
      ctx.lineTo(radius * 0.36, radius * 0.02);
      ctx.lineTo(radius * 0.36, radius * 0.16);
      ctx.lineTo(radius * 0.15, radius * 0.16);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-radius * 0.16, radius * 0.04);
      ctx.lineTo(-radius * 0.16, -radius * 0.04);
      ctx.moveTo(-radius * 0.22, 0);
      ctx.lineTo(-radius * 0.1, 0);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(-radius * 0.18, radius * 0.2, radius * 0.07, 0, Math.PI * 2);
      ctx.arc(radius * 0.2, radius * 0.2, radius * 0.07, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    drawRoundedRectPath(
      ctx,
      -radius * 0.22,
      -radius * 0.24,
      radius * 0.44,
      radius * 0.56,
      radius * 0.08,
    );
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-radius * 0.06, -radius * 0.34);
    ctx.lineTo(-radius * 0.06, -radius * 0.24);
    ctx.lineTo(radius * 0.06, -radius * 0.24);
    ctx.lineTo(radius * 0.06, -radius * 0.34);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, -radius * 0.08);
    ctx.bezierCurveTo(radius * 0.11, -radius * 0.18, radius * 0.17, 0, 0, radius * 0.14);
    ctx.bezierCurveTo(-radius * 0.17, 0, -radius * 0.11, -radius * 0.18, 0, -radius * 0.08);
    ctx.fill();
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawInventoryStrips = function (ctx, center, radius) {
    var total = Math.max(
      center.inventory.rbc,
      center.inventory.platelets,
      center.inventory.plasma,
      1,
    );
    var bars = [
      {
        ratio: center.inventory.rbc / total,
        color: "#ff7360",
        x: -radius * 0.28,
      },
      {
        ratio: center.inventory.platelets / total,
        color: "#ffd36f",
        x: 0,
      },
      {
        ratio: center.inventory.plasma / total,
        color: "#7be6ff",
        x: radius * 0.28,
      },
    ];

    ctx.save();
    for (var index = 0; index < bars.length; index += 1) {
      var bar = bars[index];
      var barHeight = 5 + bar.ratio * 12;
      ctx.fillStyle = "rgba(7, 12, 18, 0.76)";
      drawRoundedRectPath(ctx, bar.x - 4, radius * 0.62 - 14, 8, 16, 2);
      ctx.fill();
      ctx.fillStyle = bar.color;
      drawRoundedRectPath(ctx, bar.x - 3, radius * 0.62 + 1 - barHeight, 6, barHeight, 2);
      ctx.fill();
    }
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawDonationBursts = function (ctx) {
    if (!this.donationBursts.length) return;
    ctx.save();
    ctx.shadowBlur = 16;
    ctx.shadowColor = "rgba(125, 242, 208, 0.48)";
    for (var index = 0; index < this.donationBursts.length; index += 1) {
      var burst = this.donationBursts[index];
      var alpha = clamp(burst.life / burst.maxLife, 0, 1);
      ctx.fillStyle = "rgba(125, 242, 208, " + alpha.toFixed(3) + ")";
      ctx.beginPath();
      ctx.arc(burst.x, burst.y, burst.size, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.drawWeatherParticles = function (ctx) {
    if (!this.weatherParticles.length) return;
    ctx.save();
    for (var index = 0; index < this.weatherParticles.length; index += 1) {
      var particle = this.weatherParticles[index];
      var px = particle.x * this.cssWidth;
      var py = particle.y * this.cssHeight;
      if (this.weather === "ice_storm") {
        ctx.strokeStyle = "rgba(145, 244, 255," + particle.alpha.toFixed(3) + ")";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px - particle.size * 2.4, py + particle.size * 3.4);
        ctx.stroke();
      } else {
        ctx.fillStyle = "rgba(240, 247, 255," + particle.alpha.toFixed(3) + ")";
        ctx.beginPath();
        ctx.arc(px, py, particle.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  };

  SimulationSceneRenderer.prototype.destroy = function () {
    this.setActive(false);
    global.removeEventListener("resize", this.boundResize);
    if (this.resizeObserver) this.resizeObserver.disconnect();
    if (this.canvas) {
      this.canvas.removeEventListener("mousemove", this.boundPointerMove);
      this.canvas.removeEventListener("mouseleave", this.boundPointerLeave);
    }
  };

  global.SceneRenderer = {
    create: function (board) {
      return new SimulationSceneRenderer(board);
    },
  };
})(window);
