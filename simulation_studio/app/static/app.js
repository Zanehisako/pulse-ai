/* ============================================================
   PIOS Blood Supply Simulation — Game-Like Dashboard
   Plain vanilla JS (no modules, no frameworks)
   ============================================================ */

var state = {
  ws: null,
  scenarios: [],
  strategies: [],
  dreamerv3Runs: [],
  defaultDreamerv3RunKey: "",
  currentStep: null,
  centers: [],
  paused: false,
  speed: 1,
  initData: null,
  lastStepData: null,
  report: null,
  previousHud: {},
  previousDonated: 0,
  previousTransfused: 0,
  snowflakes: [],
  snowActive: false,
  transferTimers: [],
  maxInventoryGlobal: 80,
  donorCountMap: {},
  // Digital Twin state
  twinMode: false,
  twinConfig: null,
  twinRunId: null,
  twinSnapshotId: null,
  twinConfigVersion: null,
  twinSourceStatus: null,
  twinLastActionId: null,
  twinEvents: [],
  twinForecastJobId: null,
  twinCycleNumber: 1,
  twinBudgetHistory: [],
  // Dynamic World state
  dynamicWorldEnabled: false,
  worldState: null,
  studioSetup: null,
  labPolicyCatalog: [],
  labPolicies: [],
  labPolicyMin: 2,
  labPolicyMax: 2,
  studioResult: null,
  studioAssistantResult: null,
  customScenarioEditor: null,
  customScenarioDirty: false,
  actionCatalog: {},
  previousActiveActionsByKey: {},
  sceneRenderer: null,
  runtimeMapZoom: 1,
};

var RUNTIME_UI_CONFIG = {
  title: "Blood Supply Digital Twin",
  defaultVersionLabel: "PIOS",
  riskBands: [
    { maxShortageRate: 3, label: "Low" },
    { maxShortageRate: 10, label: "Medium" },
    { maxShortageRate: Infinity, label: "High" },
  ],
  sourceLabels: {
    fresh_live: "Live",
    stale_live: "Stale live",
    synthetic: "Synthetic",
    degraded: "Degraded",
  },
  centerTypeLabels: {
    hospital: "Hospital",
    blood_bank: "Blood Bank",
    mobile: "DC / Center",
    default: "Center",
  },
  metricDefaults: {
    totalInventory: 0,
    activeDeliveries: 0,
    criticalSites: 0,
    shortageRate: 0,
    serviceRate: null,
    networkEfficiency: null,
  },
  operator: {
    initials: "OPS",
    name: "Operations Console",
    role: "Live data status",
  },
  zoom: {
    min: 0.85,
    max: 1.45,
    step: 0.08,
  },
};

/* ----------------------------------------------------------
   Helpers
   ---------------------------------------------------------- */

function formatNumber(n) {
  if (n == null || isNaN(n)) return "—";
  var num = Number(n);
  if (Math.abs(num) >= 1e6) return num.toLocaleString("en-US");
  if (Number.isInteger(num)) return num.toLocaleString("en-US");
  return num.toLocaleString("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

function formatPercent(n) {
  if (n == null || isNaN(n)) return "—";
  return Number(n).toFixed(1) + "%";
}

function formatInteger(n) {
  if (n == null || isNaN(n)) return "—";
  return Math.round(Number(n)).toLocaleString("en-US");
}

function formatCompactUnits(n) {
  if (n == null || isNaN(n)) return "—";
  var num = Number(n);
  if (Math.abs(num) >= 1e6) {
    return (num / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
  }
  if (Math.abs(num) >= 1e3) {
    return (num / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
  }
  return formatInteger(num);
}

function escapeHtml(s) {
  if (s == null) return "";
  var str = String(s);
  var div = document.createElement("div");
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function $(id) {
  return document.getElementById(id);
}

function ensureSceneRenderer() {
  if (state.sceneRenderer) return state.sceneRenderer;
  if (!window.SceneRenderer) return null;
  var board = $("map-board");
  if (!board) return null;
  state.sceneRenderer = window.SceneRenderer.create(board);
  return state.sceneRenderer;
}

function syncSceneOperationalFrame(snapshot) {
  var renderer = state.sceneRenderer || ensureSceneRenderer();
  if (!renderer) return;
  renderer.setMissionContext(
    state.initData && state.initData.scenario,
    state.initData && state.initData.strategy,
    !!state.twinMode,
  );
  renderer.setOperationalFrame(snapshot || {});
}

function lookupRuntimeScenarioName(key) {
  var scenarioKey = String(key || "");
  for (var i = 0; i < state.scenarios.length; i++) {
    if (String(state.scenarios[i].key) === scenarioKey) {
      return state.scenarios[i].name || state.scenarios[i].key;
    }
  }
  return scenarioKey || "Scenario";
}

function currentRuntimeScenarioKey() {
  return (
    (state.initData &&
      state.initData.scenario &&
      (state.initData.scenario.key || state.initData.scenario.name)) ||
    (($("scenario-select") || {}).value || "")
  );
}

function currentRuntimeStrategyName() {
  var strategy = state.initData && state.initData.strategy;
  if (!strategy) return (($("strategy-select") || {}).value || "Strategy");
  return typeof strategy === "string"
    ? strategy
    : strategy.name || strategy.key || "Strategy";
}

function totalInventoryFromCenters(centers) {
  var total = 0;
  for (var i = 0; i < centers.length; i++) {
    if (centers[i].total_units != null) {
      total += Number(centers[i].total_units || 0);
      continue;
    }
    var inventory = centers[i].inventory || {};
    total +=
      Number(inventory.RBC || 0) +
      Number(inventory.PLATELETS || 0) +
      Number(inventory.PLASMA || 0);
  }
  return total;
}

function criticalSiteCount(centers) {
  if (!centers.length) return 0;
  var maxInventory = 1;
  for (var i = 0; i < centers.length; i++) {
    maxInventory = Math.max(
      maxInventory,
      Number(centers[i].total_units || 0),
    );
  }
  var critical = 0;
  for (var j = 0; j < centers.length; j++) {
    var total = Number(centers[j].total_units || 0);
    if (total / maxInventory < 0.18) critical += 1;
  }
  return critical;
}

function runtimeRiskBand(shortageRate) {
  var rate = Number(shortageRate || 0);
  for (var i = 0; i < RUNTIME_UI_CONFIG.riskBands.length; i++) {
    if (rate <= RUNTIME_UI_CONFIG.riskBands[i].maxShortageRate) {
      return RUNTIME_UI_CONFIG.riskBands[i].label;
    }
  }
  return RUNTIME_UI_CONFIG.riskBands[RUNTIME_UI_CONFIG.riskBands.length - 1].label;
}

function setText(id, value) {
  var el = $(id);
  if (el) el.textContent = String(value);
}

function setRuntimeSelectValue(key) {
  var select = $("runtime-scenario-select");
  if (!select) return;
  var scenarioKey = String(key || "");
  if (!select.options.length && state.scenarios.length) {
    for (var i = 0; i < state.scenarios.length; i++) {
      var option = document.createElement("option");
      option.value = state.scenarios[i].key;
      option.textContent = state.scenarios[i].name || state.scenarios[i].key;
      select.appendChild(option);
    }
  }
  if (scenarioKey) {
    select.value = scenarioKey;
  }
}

function updateRuntimeSourceStatus(status) {
  var statusKey = (status && status.status) || (state.twinMode ? "degraded" : "synthetic");
  var label =
    RUNTIME_UI_CONFIG.sourceLabels[statusKey] ||
    twinStatusLabel(statusKey || "synthetic");
  setText("runtime-source-label", label);

  var badge = $("runtime-source-badge");
  if (badge) {
    badge.className = "dt-source-badge source-" + statusKey;
  }
}

function updateRuntimeDashboard(snapshot) {
  snapshot = snapshot || {};
  var centers = snapshot.centers || state.centers || [];
  var totalInventory = totalInventoryFromCenters(centers);
  var activeActions = snapshot.active_actions || [];
  var activeEvents = snapshot.active_events || state.twinEvents || [];
  var recentDonations = Number(snapshot.recent_donations || 0);
  var recentShortages = Number(snapshot.recent_shortages || 0);
  var totalTransfused = Number(snapshot.total_transfused || 0);
  var totalShortage = Number(snapshot.total_shortage || 0);
  var shortageRate = Number(snapshot.shortage_rate || 0);
  var serviceRate =
    totalTransfused + totalShortage > 0
      ? (totalTransfused / (totalTransfused + totalShortage)) * 100
      : RUNTIME_UI_CONFIG.metricDefaults.serviceRate;
  var networkEfficiency = Math.max(0, Math.min(100, 100 - shortageRate));
  var activeDeliveries = Math.max(
    activeActions.length,
    recentDonations,
    recentShortages,
  );
  var criticalSites = Math.max(criticalSiteCount(centers), recentShortages > 0 ? 1 : 0);
  var routeCount = centers.length ? Math.max(centers.length - 1, 0) : 0;
  var scenarioKey = currentRuntimeScenarioKey();
  var sourceStatus = snapshot.source_status || state.twinSourceStatus || {};
  var versionLabel =
    state.twinConfigVersion ||
    (state.twinConfig && state.twinConfig.version) ||
    RUNTIME_UI_CONFIG.defaultVersionLabel;

  setText("runtime-title", RUNTIME_UI_CONFIG.title);
  setText("runtime-version", versionLabel);
  setText("runtime-mode-label", state.twinMode ? "Digital Twin" : "Simulation");
  setText("runtime-strategy-label", currentRuntimeStrategyName());
  setRuntimeSelectValue(scenarioKey);
  updateRuntimeSourceStatus(sourceStatus);

  var inventoryDisplay =
    totalInventory > 0 ? formatCompactUnits(totalInventory) : "—";
  var inventoryDelta = totalInventory > 0 ? "Live simulator stock" : "No live inventory";

  setText("metric-total-inventory", inventoryDisplay);
  setText("metric-inventory-delta", inventoryDelta);
  setText("metric-active-deliveries", formatInteger(activeDeliveries));
  setText("metric-delivery-caption", "Live tracking");
  setText("metric-critical-shortages", formatInteger(criticalSites));
  setText("metric-shortage-caption", "Requires attention");
  setText("metric-service-level", formatPercent(serviceRate));
  setText("metric-service-caption", "On-time delivery");
  setText("metric-network-efficiency", formatPercent(networkEfficiency));
  setText(
    "metric-efficiency-delta",
    "Derived from current shortage rate",
  );
  setText("metric-risk-label", runtimeRiskBand(shortageRate));
  setText("metric-risk-score", "Risk Score: " + Math.round(shortageRate * 10) + " / 100");
  setText("bottom-active-routes", formatInteger(routeCount));
  setText("bottom-in-transit", formatInteger(activeDeliveries));
  setText("bottom-service-rate", formatPercent(serviceRate));
  setText("bottom-efficiency", formatPercent(networkEfficiency));
  setText("dt-delivery-total", formatInteger(activeDeliveries));
  setText("runtime-event-count", formatInteger(activeEvents.length));
  setText("runtime-alert-count", formatInteger(activeEvents.length));
  setText("dt-alert-badge", formatInteger(activeEvents.length));
  setText("runtime-budget-value", "$" + formatNumber(snapshot.budget_remaining || 0));
  setText("runtime-operator-initials", RUNTIME_UI_CONFIG.operator.initials);
  setText("runtime-operator-name", RUNTIME_UI_CONFIG.operator.name);
  setText("runtime-operator-role", RUNTIME_UI_CONFIG.operator.role);

  if (window.DashboardUI && typeof window.DashboardUI.mergeRuntimeFrame === "function") {
    window.DashboardUI.mergeRuntimeFrame(snapshot);
  }
}

function setRuntimeMapZoom(nextZoom) {
  var cfg = RUNTIME_UI_CONFIG.zoom;
  state.runtimeMapZoom = Math.max(cfg.min, Math.min(cfg.max, nextZoom));
  if (window.DashboardUI && typeof window.DashboardUI.setMapZoom === "function") {
    window.DashboardUI.setMapZoom(state.runtimeMapZoom);
    return;
  }
  document.documentElement.style.setProperty(
    "--runtime-map-scale",
    state.runtimeMapZoom.toFixed(2),
  );
}

function clearGameChromeHoverState() {
  var body = document.body;
  if (!body) return;
  body.classList.remove(
    "game-hover-top",
    "game-hover-bottom",
    "game-hover-left",
    "game-hover-right",
  );
}

function updateHudOverview(screenId) {
  var textEl = $("hud-overview-text");
  if (!textEl) return;
  if (screenId === "game-screen") {
    textEl.textContent = state.twinMode ? "Digital Twin" : "Simulation Live";
    return;
  }
  if (screenId === "report-screen") {
    textEl.textContent = "Mission Report";
    return;
  }
  textEl.textContent = "Mission Setup";
}

function updateGameChromeHoverState(event) {
  var body = document.body;
  if (!body || !body.classList.contains("screen-game")) {
    clearGameChromeHoverState();
    return;
  }

  var screen = $("game-screen");
  if (!screen) {
    clearGameChromeHoverState();
    return;
  }

  var rect = screen.getBoundingClientRect();
  var x = event.clientX - rect.left;
  var y = event.clientY - rect.top;
  if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
    clearGameChromeHoverState();
    return;
  }

  var target = event.target;
  var topHot = y <= Math.min(96, rect.height * 0.16);
  var bottomHot = y >= rect.height - Math.min(112, rect.height * 0.2);
  var leftHot = x <= Math.min(96, rect.width * 0.12);
  var rightHot = x >= rect.width - Math.min(108, rect.width * 0.14);

  if (target && target.closest) {
    topHot = topHot || !!target.closest("#hud") || !!target.closest(".scene-topbar");
    bottomHot =
      bottomHot ||
      !!target.closest(".controls-area") ||
      !!target.closest(".scene-footer");
    leftHot = leftHot || !!target.closest(".sidebar-left");
    rightHot =
      rightHot ||
      !!target.closest(".scene-inspector") ||
      !!target.closest(".world-detail-dropdown");
  }

  body.classList.toggle("game-hover-top", topHot);
  body.classList.toggle("game-hover-bottom", bottomHot);
  body.classList.toggle("game-hover-left", leftHot);
  body.classList.toggle("game-hover-right", rightHot);
}

function fetchJson(url, options) {
  return fetch(url, options).then(function (res) {
    if (!res.ok) {
      return res
        .json()
        .catch(function () {
          return { detail: "Request failed with status " + res.status };
        })
        .then(function (payload) {
          var detail =
            payload && payload.detail ? payload.detail : "Request failed";
          throw new Error(detail);
        });
    }
    return res.json();
  });
}

function monthsLabel(months) {
  var value = parseInt(months, 10) || 0;
  if (value === 0) return "Current mission horizon";
  if (value === 1) return "1 month ahead";
  return value + " months ahead";
}

var LAB_METRIC_LABELS = {
  active_action_cost: "Active Action Cost",
  active_action_count: "Active Actions",
  active_last_180d_share: "Active In 180 Days",
  average_dropout_risk: "Average Dropout Risk",
  avg_travel_min: "Average Travel Time",
  avg_wait_min: "Average Wait Time",
  base_service_rate: "Base Service Rate",
  base_shortage_rate: "Base Shortage Rate",
  budget_exhausted_hours: "Budget Exhausted Hours",
  budget_remaining: "Budget Remaining",
  budget_spent: "Budget Spent",
  budget_total: "Budget Total",
  compatible_substitution_rate: "Compatible Substitution Rate",
  conservation_rate: "Conservation Rate",
  controller_decisions: "Controller Decisions",
  donor_attempts: "Donor Attempts",
  eligible_rate: "Eligible Rate",
  episode_score: "Episode Score",
  exact_match_rate: "Exact Match Rate",
  forecast_pressure: "Forecast Pressure",
  incompatible_fulfillment_rate: "Incompatible Fulfillment Rate",
  priority_weighted_requested: "Priority-Weighted Demand",
  priority_weighted_shortage: "Priority-Weighted Shortage",
  recent_priority_pressure: "Recent Priority Pressure",
  regular_donor_share: "Regular Donor Share",
  rejection_rate: "Rejection Rate",
  reward_total: "Reward Total",
  service_rate: "Service Rate",
  shortage_plasma: "Plasma Shortage",
  shortage_platelets: "Platelet Shortage",
  shortage_rate: "Shortage Rate",
  shortage_rbc: "RBC Shortage",
  total_base_requested: "Base Requested",
  total_collected: "Total Collected",
  total_compatible_substitution_units: "Compatible Substitutions",
  total_conserved: "Total Conserved",
  total_donated: "Total Donated",
  total_exact_match_units: "Exact Match Units",
  total_expired: "Expired Units",
  total_external_units: "External Units",
  total_incompatible_fulfillment_units: "Incompatible Fulfillments",
  total_lab_rejected: "Lab Rejected",
  total_net_requested: "Net Requested",
  total_no_show: "No Shows",
  total_rejected: "Total Rejected",
  total_released: "Total Released",
  total_shortage: "Total Shortage",
  total_donors: "Donors",
  total_transfused: "Total Transfused",
  transfer_units: "Transfer Units",
  transport_congestion: "Transport Congestion",
  available_model_count: "Available Models",
  critical_event_count: "Critical Events",
};

var LAB_METRIC_PRIORITY = {
  shortage_rate: 0,
  total_shortage: 1,
  shortage_rbc: 2,
  shortage_platelets: 3,
  shortage_plasma: 4,
  service_rate: 5,
  episode_score: 6,
  reward_total: 7,
  budget_spent: 8,
  budget_remaining: 9,
  budget_total: 10,
  total_transfused: 11,
  total_donated: 12,
  total_expired: 13,
  exact_match_rate: 14,
  compatible_substitution_rate: 15,
};

function titleCaseMetricLabel(key) {
  return String(key || "")
    .split("_")
    .filter(Boolean)
    .map(function (part) {
      if (part === "rbc") return "RBC";
      if (part === "ppo") return "PPO";
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

function labMetricBaseKey(key) {
  return String(key || "").replace(/_(mean|low|high|std)$/, "");
}

function labMetricLabel(baseKey) {
  return LAB_METRIC_LABELS[baseKey] || titleCaseMetricLabel(baseKey);
}

function isIntegerish(value) {
  var numeric = Number(value);
  if (isNaN(numeric)) return false;
  return Math.abs(numeric - Math.round(numeric)) < 0.05;
}

function formatAggregateMetricValue(baseKey, value) {
  if (value == null || isNaN(value)) return "—";
  if (/(^|_)(.*rate|share)$/.test(baseKey)) {
    return formatPercent(value);
  }
  if (/_min$/.test(baseKey)) {
    return formatNumber(value) + " min";
  }
  if (/^budget_exhausted_hours$/.test(baseKey)) {
    return formatNumber(value) + " h";
  }
  if (
    /^(episode_score|reward_total|budget_|active_action_cost|forecast_pressure|transport_congestion)$/.test(
      baseKey,
    )
  ) {
    return formatNumber(value);
  }
  if (isIntegerish(value)) {
    return formatInteger(value);
  }
  return formatNumber(value);
}

function formatSnapshotMetricValue(baseKey, value) {
  if (value == null || isNaN(value)) return "—";
  if (
    /^(eligible_rate|regular_donor_share|average_dropout_risk|active_last_180d_share)$/.test(
      baseKey,
    )
  ) {
    return formatPercent(Number(value) * 100);
  }
  if (isIntegerish(value)) {
    return formatInteger(value);
  }
  return formatNumber(value);
}

function collectAggregateMetrics(summary) {
  var metrics = [];
  var keys = Object.keys(summary || {});
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    if (!/_mean$/.test(key)) continue;
    var baseKey = labMetricBaseKey(key);
    var lowKey = baseKey + "_low";
    var highKey = baseKey + "_high";
    var stdKey = baseKey + "_std";
    metrics.push({
      baseKey: baseKey,
      label: labMetricLabel(baseKey),
      value: summary[key],
      low: summary[lowKey],
      high: summary[highKey],
      std: summary[stdKey],
      valueText: formatAggregateMetricValue(baseKey, summary[key]),
      bandText:
        summary[lowKey] != null && summary[highKey] != null
          ? "Band " +
            formatAggregateMetricValue(baseKey, summary[lowKey]) +
            " to " +
            formatAggregateMetricValue(baseKey, summary[highKey])
          : "",
      stdText:
        summary[stdKey] != null ? "Std " + formatNumber(summary[stdKey]) : "",
      priority:
        LAB_METRIC_PRIORITY[baseKey] != null ? LAB_METRIC_PRIORITY[baseKey] : 100,
    });
  }
  metrics.sort(function (a, b) {
    if (a.priority !== b.priority) return a.priority - b.priority;
    return a.label.localeCompare(b.label);
  });
  return metrics;
}

function collectSnapshotMetrics(summary) {
  var priorities = {
    total_donors: 0,
    eligible_rate: 1,
    regular_donor_share: 2,
    average_dropout_risk: 3,
    active_last_180d_share: 4,
    critical_event_count: 5,
    available_model_count: 6,
  };
  var metrics = [];
  var keys = Object.keys(summary || {});
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    if (typeof summary[key] !== "number") continue;
    metrics.push({
      baseKey: key,
      label: labMetricLabel(key),
      valueText: formatSnapshotMetricValue(key, summary[key]),
      priority: priorities[key] != null ? priorities[key] : 100,
    });
  }
  metrics.sort(function (a, b) {
    if (a.priority !== b.priority) return a.priority - b.priority;
    return a.label.localeCompare(b.label);
  });
  return metrics;
}

function findMetric(metrics, baseKey) {
  for (var i = 0; i < metrics.length; i++) {
    if (metrics[i].baseKey === baseKey) {
      return metrics[i];
    }
  }
  return null;
}

function renderLabMetric(metric, extraClass) {
  var classes = ["lab-metric"];
  if (extraClass) classes.push(extraClass);
  var meta = [];
  if (metric.bandText) meta.push(metric.bandText);
  if (metric.stdText) meta.push(metric.stdText);
  return (
    '<div class="' +
    classes.join(" ") +
    '">' +
    '<span class="lab-metric-label">' +
    escapeHtml(metric.label) +
    "</span>" +
    '<span class="lab-metric-value">' +
    escapeHtml(metric.valueText) +
    "</span>" +
    (meta.length
      ? '<span class="lab-metric-range">' + escapeHtml(meta.join(" · ")) + "</span>"
      : "") +
    "</div>"
  );
}

function renderBloodMix(distribution) {
  var keys = Object.keys(distribution || {});
  if (!keys.length) return "";
  var items = "";
  for (var i = 0; i < keys.length; i++) {
    items +=
      '<span class="lab-blood-pill"><strong>' +
      escapeHtml(keys[i]) +
      "</strong>" +
      escapeHtml(formatInteger(distribution[keys[i]])) +
      "</span>";
  }
  return (
    '<div class="lab-blood-mix">' +
    '<span class="lab-blood-mix-label">Blood Mix</span>' +
    '<div class="lab-blood-pill-row">' +
    items +
    "</div>" +
    "</div>"
  );
}

function renderUniverseMetricsBoard(universe, index) {
  var strategyName =
    universe.strategy && universe.strategy.name
      ? universe.strategy.name
      : universe.strategy_name || "Unknown Strategy";
  var metrics = collectAggregateMetrics(universe.summary || {});
  var shortageMetric = findMetric(metrics, "shortage_rate");
  var heroKeys = [
    "total_shortage",
    "service_rate",
    "episode_score",
    "reward_total",
    "budget_spent",
    "budget_remaining",
    "budget_total",
  ];
  var heroMetrics = [];
  var hiddenKeys = { shortage_rate: true };
  for (var i = 0; i < heroKeys.length; i++) {
    var heroMetric = findMetric(metrics, heroKeys[i]);
    if (heroMetric) {
      heroMetrics.push(heroMetric);
      hiddenKeys[heroKeys[i]] = true;
    }
  }
  var detailMetrics = [];
  for (var j = 0; j < metrics.length; j++) {
    if (!hiddenKeys[metrics[j].baseKey]) {
      detailMetrics.push(metrics[j]);
    }
  }

  return (
    '<div class="lab-branch' +
    (index === 0 ? " is-best" : "") +
    '">' +
    '<div class="lab-branch-header">' +
    '<div class="lab-branch-main"><strong>' +
    escapeHtml(universe.label) +
    "</strong><span>" +
    escapeHtml(strategyName) +
    "</span></div>" +
    '<div class="lab-primary-shortage">' +
    '<span class="lab-primary-shortage-label">Primary Metric</span>' +
    '<strong>' +
    escapeHtml(shortageMetric ? shortageMetric.valueText : "—") +
    "</strong>" +
    '<span class="lab-primary-shortage-caption">Shortage Rate</span>' +
    "</div>" +
    "</div>" +
    '<div class="lab-branch-meta">' +
    heroMetrics.map(function (metric) {
      return renderLabMetric(metric, "lab-metric-compact");
    }).join("") +
    "</div>" +
    '<div class="lab-branch-metrics">' +
    detailMetrics.map(function (metric) {
      return renderLabMetric(metric, "");
    }).join("") +
    "</div>" +
    "</div>"
  );
}

/* ----------------------------------------------------------
   Toast Notifications
   ---------------------------------------------------------- */
function showToast(opts) {
  var title = opts.title || "";
  var description = opts.description || "";
  var severity = opts.severity != null ? opts.severity : 0;
  var icon = opts.icon || "fa-circle-info";
  var type = opts.type || "info";
  var duration = opts.duration || 6000;

  var container = document.getElementById("toast-container");
  if (!container) return;

  // Limit to ~5 visible toasts — remove oldest if exceeded
  var existing = container.querySelectorAll(".toast:not(.toast-exit)");
  if (existing.length >= 5) {
    var oldest = existing[0];
    oldest.classList.add("toast-exit");
    setTimeout(function () {
      if (oldest.parentNode) oldest.parentNode.removeChild(oldest);
    }, 300);
  }

  // Determine severity class
  var sevClass =
    severity >= 0.6
      ? "toast-severity-high"
      : severity >= 0.3
        ? "toast-severity-medium"
        : "toast-severity-low";

  var sevLabel = severity >= 0.6 ? "High" : severity >= 0.3 ? "Medium" : "Low";

  // Build toast element
  var toast = document.createElement("div");
  toast.className = "toast toast-" + type;
  toast.style.position = "relative";
  toast.style.overflow = "hidden";

  // Icon
  var iconEl = document.createElement("div");
  iconEl.className = "toast-icon";
  var iconI = document.createElement("i");
  iconI.className = "fa-solid " + icon;
  iconEl.appendChild(iconI);

  // Body
  var body = document.createElement("div");
  body.className = "toast-body";

  // Title row (title text + severity badge)
  var titleEl = document.createElement("div");
  titleEl.className = "toast-title";
  var titleSpan = document.createElement("span");
  titleSpan.textContent = title;
  titleEl.appendChild(titleSpan);

  var sevBadge = document.createElement("span");
  sevBadge.className = "toast-severity " + sevClass;
  sevBadge.textContent = sevLabel;
  titleEl.appendChild(sevBadge);

  // Description
  var descEl = document.createElement("div");
  descEl.className = "toast-desc";
  descEl.textContent = description;

  body.appendChild(titleEl);
  body.appendChild(descEl);

  // Close button
  var closeBtn = document.createElement("div");
  closeBtn.className = "toast-close";
  var closeIcon = document.createElement("i");
  closeIcon.className = "fa-solid fa-xmark";
  closeBtn.appendChild(closeIcon);

  // Progress bar
  var progress = document.createElement("div");
  progress.className = "toast-progress";
  progress.style.setProperty("--toast-duration", duration + "ms");

  // Assemble
  toast.appendChild(iconEl);
  toast.appendChild(body);
  toast.appendChild(closeBtn);
  toast.appendChild(progress);
  container.appendChild(toast);

  // Auto-remove after duration
  var autoTimer = setTimeout(function () {
    dismissToast(toast);
  }, duration);

  // Manual dismiss via close button
  closeBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    clearTimeout(autoTimer);
    dismissToast(toast);
  });

  // Also dismiss on toast click
  toast.addEventListener("click", function () {
    clearTimeout(autoTimer);
    dismissToast(toast);
  });
}

function dismissToast(toast) {
  if (toast.classList.contains("toast-exit")) return;
  toast.classList.add("toast-exit");
  setTimeout(function () {
    if (toast.parentNode) toast.parentNode.removeChild(toast);
  }, 300);
}

function normalizeTwinSeverity(value) {
  if (value && typeof value === "object") {
    if (value.value != null) {
      return clamp(parseFloat(value.value) || 0.5, 0, 1);
    }
    if (value.severity != null) {
      return clamp(parseFloat(value.severity) || 0.5, 0, 1);
    }
  }
  return clamp(parseFloat(value) || 0.5, 0, 1);
}

function resetProgressMode() {
  var track = document.querySelector(".progress-track");
  var bar = $("progress-bar");
  var infiniteBar = $("progress-infinite");

  if (infiniteBar && infiniteBar.parentNode) {
    infiniteBar.parentNode.removeChild(infiniteBar);
  }
  if (track) {
    track.classList.remove("infinite");
  }
  if (bar) {
    bar.style.display = "";
    bar.style.width = "0%";
  }
}

function enableInfiniteProgressMode() {
  var track = document.querySelector(".progress-track");
  var bar = $("progress-bar");
  if (!track) return;

  track.classList.add("infinite");
  if (bar) bar.style.display = "none";
  if (!$("progress-infinite")) {
    var infiniteBar = document.createElement("div");
    infiniteBar.className = "progress-fill-infinite";
    infiniteBar.id = "progress-infinite";
    track.appendChild(infiniteBar);
  }
}

function populateTwinSetup(config) {
  if (!config) return;
  state.twinConfig = config;

  var stepHoursInput = $("twin-hours-input");
  if (stepHoursInput && config.step_hours && config.step_hours.default != null) {
    stepHoursInput.value = String(config.step_hours.default);
  }

  var tickInput = $("tick-interval-input");
  if (tickInput && config.tick_interval_s) {
    if (config.tick_interval_s.default != null) {
      tickInput.value = String(config.tick_interval_s.default);
    }
    if (config.tick_interval_s.min != null) {
      tickInput.min = String(config.tick_interval_s.min);
    }
    if (config.tick_interval_s.max != null) {
      tickInput.max = String(config.tick_interval_s.max);
    }
  }

  var budgetSelect = $("budget-cycle-select");
  if (budgetSelect && config.budget_cycle_options) {
    budgetSelect.innerHTML = "";
    for (var i = 0; i < config.budget_cycle_options.length; i++) {
      var option = config.budget_cycle_options[i];
      var opt = document.createElement("option");
      opt.value = option.hours;
      opt.textContent = option.label + " (" + option.hours + "h)";
      if (option.hours === config.default_budget_cycle_hours) {
        opt.selected = true;
      }
      budgetSelect.appendChild(opt);
    }
  }

  var injectSelect = $("inject-event-select");
  if (injectSelect && config.injectable_events) {
    injectSelect.innerHTML = "";
    for (var j = 0; j < config.injectable_events.length; j++) {
      var eventOption = config.injectable_events[j];
      var injectOpt = document.createElement("option");
      injectOpt.value = eventOption.key;
      injectOpt.textContent = eventOption.label || eventOption.key;
      injectSelect.appendChild(injectOpt);
    }
  }

  var actionSelect = $("twin-action-select");
  if (actionSelect && config.actions) {
    actionSelect.innerHTML = "";
    for (var k = 0; k < config.actions.length; k++) {
      var actionOption = config.actions[k];
      var actionOpt = document.createElement("option");
      actionOpt.value = actionOption.key;
      actionOpt.textContent = actionOption.label || actionOption.key;
      actionSelect.appendChild(actionOpt);
    }
  }

  var branchSelect = $("twin-branch-policy-select");
  var policies =
    config.branches && config.branches.allowed_policies
      ? config.branches.allowed_policies
      : [];
  if (branchSelect) {
    branchSelect.innerHTML = "";
    for (var p = 0; p < policies.length; p++) {
      var policyOpt = document.createElement("option");
      policyOpt.value = policies[p];
      policyOpt.textContent = String(policies[p])
        .replace(/_/g, " ")
        .replace(/\b\w/g, function (c) {
          return c.toUpperCase();
        });
      branchSelect.appendChild(policyOpt);
    }
  }

  syncInjectSeverityFromSelection();
}

function syncInjectSeverityFromSelection() {
  var injectSelect = $("inject-event-select");
  var severityInput = $("inject-severity");
  var severityLabel = $("inject-severity-label");
  var events =
    state.twinConfig && state.twinConfig.injectable_events
      ? state.twinConfig.injectable_events
      : [];

  if (!injectSelect || !severityInput || events.length === 0) return;

  for (var i = 0; i < events.length; i++) {
    if (events[i].key === injectSelect.value) {
      if (events[i].default_severity != null) {
        severityInput.value = String(events[i].default_severity);
        if (severityLabel) {
          severityLabel.textContent = parseFloat(events[i].default_severity).toFixed(
            2,
          );
        }
      }
      return;
    }
  }
}

function showScreen(screenId) {
  var screenDisplay = {
    "setup-screen": "flex",
    "game-screen": "flex",
    "report-screen": "block",
  };
  var screens = ["setup-screen", "game-screen", "report-screen"];
  for (var i = 0; i < screens.length; i++) {
    var el = $(screens[i]);
    if (el)
      el.style.display =
        screens[i] === screenId ? screenDisplay[screens[i]] : "none";
  }

  // Toggle game-mode class on app-root to lock viewport during game
  var appRoot = document.querySelector(".app-root");
  if (appRoot) {
    if (screenId === "game-screen") {
      appRoot.classList.add("game-mode");
    } else {
      appRoot.classList.remove("game-mode");
    }
  }

  var body = document.body;
  if (body) {
    body.classList.remove("screen-setup", "screen-game", "screen-report");
    body.classList.add("screen-" + screenId.replace("-screen", ""));
  }
  updateHudOverview(screenId);
  if (screenId !== "game-screen") {
    clearGameChromeHoverState();
  }

  var renderer = state.sceneRenderer;
  if (!renderer && screenId === "game-screen") {
    renderer = ensureSceneRenderer();
  }
  if (renderer) {
    renderer.setActive(screenId === "game-screen");
  }
  if (screenId === "game-screen" && window.DashboardUI && typeof window.DashboardUI.onGameScreenShown === "function") {
    window.DashboardUI.onGameScreenShown();
  }
}

function setConnectionStatus(text, connected) {
  var el = $("connection-status");
  if (!el) return;
  var dot = el.querySelector(".status-dot");
  var label = el.querySelector(".status-label");
  if (label) label.textContent = text;
  el.classList.toggle("connected", !!connected);
  el.classList.toggle("disconnected", !connected);
}

function pulseElement(el) {
  if (!el) return;
  el.classList.remove("pulse");
  // Re-add on the next frame so the CSS animation restarts WITHOUT a synchronous
  // forced reflow. The old `void el.offsetWidth` flushed layout on every call,
  // and setHudValue fires it ~10x per simulation step (a major jank source).
  requestAnimationFrame(function () {
    el.classList.add("pulse");
  });
}

function setHudValue(id, value) {
  var el = $(id);
  if (!el) return;
  var text = String(value);
  if (el.textContent !== text) {
    el.textContent = text;
    pulseElement(el);
  }
}

/* ----------------------------------------------------------
   Lat/Lon to map position
   ---------------------------------------------------------- */

function latLonToPosition(lat, lon) {
  var LAT_MIN = 46.76,
    LAT_MAX = 46.86;
  var LON_MIN = -71.35,
    LON_MAX = -71.15;
  var x = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * 80 + 10;
  var y = ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * 80 + 10;
  return { x: clamp(x, 5, 95), y: clamp(y, 5, 95) };
}

/* ----------------------------------------------------------
   Weather helpers
   ---------------------------------------------------------- */

var weatherIcons = {
  clear: "fa-sun",
  cloudy: "fa-cloud",
  snow: "fa-snowflake",
  ice_storm: "fa-icicles",
  blizzard: "fa-cloud-showers-heavy",
};

var weatherLabels = {
  clear: "Clear",
  cloudy: "Cloudy",
  snow: "Snow",
  ice_storm: "Ice Storm",
  blizzard: "Blizzard",
};

function getWeatherIconClass(weather) {
  return "fa-solid " + (weatherIcons[weather] || "fa-cloud-sun");
}

function updateWeatherOverlay(weather) {
  var renderer = state.sceneRenderer || ensureSceneRenderer();
  if (renderer) {
    renderer.setWeather(weather || "clear");
  }

  var overlay = $("weather-overlay");
  if (!overlay) return;

  // Remove all weather classes
  overlay.className = "weather-overlay";

  if (weather) {
    overlay.classList.add("weather-" + weather);
  }

  // Handle snowflakes
  var needSnow = weather === "snow" || weather === "blizzard";
  if (needSnow && !state.snowActive) {
    startSnow(weather === "blizzard");
    state.snowActive = true;
  } else if (!needSnow && state.snowActive) {
    stopSnow();
    state.snowActive = false;
  } else if (needSnow && state.snowActive) {
    // Update intensity if switching between snow and blizzard
    stopSnow();
    startSnow(weather === "blizzard");
  }
}

function startSnow(heavy) {
  var container = $("snow-container");
  if (!container) return;
  container.innerHTML = "";

  var count = heavy ? 40 : 18;
  var flakes = ["❄", "❅", "❆", "•"];

  for (var i = 0; i < count; i++) {
    var flake = document.createElement("span");
    flake.className = "snowflake";
    flake.textContent = flakes[Math.floor(Math.random() * flakes.length)];
    var left = Math.random() * 100;
    var duration = heavy ? 3 + Math.random() * 4 : 6 + Math.random() * 8;
    var delay = Math.random() * duration;
    var size = heavy ? 0.5 + Math.random() * 1.0 : 0.4 + Math.random() * 0.7;
    flake.style.left = left + "%";
    flake.style.fontSize = size + "rem";
    flake.style.animationName = heavy ? "snowfall-wind" : "snowfall";
    flake.style.animationDuration = duration + "s";
    flake.style.animationDelay = delay + "s";
    container.appendChild(flake);
  }
}

function stopSnow() {
  var container = $("snow-container");
  if (container) container.innerHTML = "";
}

/* ----------------------------------------------------------
   Day/Night helper
   ---------------------------------------------------------- */

function updateDayNight(hour) {
  var hourOfDay = ((Number(hour) % 24) + 24) % 24;
  var dayNum = Math.floor(hour / 24) + 1;
  var isDay = hourOfDay >= 6 && hourOfDay < 20;

  // Drive the map day/night crossfade independently of the HUD elements
  // so it still works even when the HUD indicator isn't mounted.
  if (window.DashboardUI && typeof DashboardUI.setMapDayFactorFromHour === "function") {
    DashboardUI.setMapDayFactorFromHour(hour);
  } else if (window.DashboardUI && typeof DashboardUI.setMapNightMode === "function") {
    DashboardUI.setMapNightMode(!isDay);
  }

  var dayEl = $("day-night");
  var dnText = $("hud-daynight");
  if (!dayEl || !dnText) return;

  dayEl.classList.toggle("is-day", isDay);
  dayEl.classList.toggle("is-night", !isDay);

  var icon = dayEl.querySelector(".dn-icon");
  if (icon) {
    icon.className = "fa-solid dn-icon " + (isDay ? "fa-sun" : "fa-moon");
  }

  dnText.textContent = "Day " + dayNum;
}

/* ----------------------------------------------------------
   Shortage pulse
   ---------------------------------------------------------- */

function updateShortagePulse(rate) {
  var el = $("hud-shortage-container");
  if (!el) return;
  if (rate > 5) {
    el.classList.add("shortage-high");
  } else {
    el.classList.remove("shortage-high");
  }
}

/* ----------------------------------------------------------
   Populate setup dropdowns
   ---------------------------------------------------------- */

function populateSetup(data) {
  state.scenarios = data.scenarios || [];
  state.strategies = data.strategies || [];
  state.dreamerv3Runs = data.dreamerv3_runs || [];
  state.defaultDreamerv3RunKey = data.default_dreamerv3_run_key || "";
  state.customScenarioEditor = data.custom_scenario_editor || null;
  sortScenariosInPlace();

  populateScenarioSelect("scenario-select", "baseline");
  populateScenarioSelect("lab-scenario-select", "baseline");
  populateStrategySelect("strategy-select", "baseline");
  populateStrategySelect("twin-strategy-select", "baseline");
  populateStrategySelect("custom-scenario-strategy-select", "baseline");
  populateCustomScenarioBaseSelect(currentMissionScenarioKey());
  loadCustomScenarioBase(currentMissionScenarioKey());

  updateScenarioDesc();
  populateDreamerRuns(
    "dreamerv3-run-select",
    state.dreamerv3Runs,
    state.defaultDreamerv3RunKey,
  );
  populateDreamerRuns(
    "twin-dreamerv3-run-select",
    state.dreamerv3Runs,
    state.defaultDreamerv3RunKey,
  );
  setRuntimeSelectValue(currentMissionScenarioKey());
  updateDreamerRunVisibility("strategy-select", "dreamerv3-run-field");
  updateDreamerRunVisibility("twin-strategy-select", "twin-dreamerv3-run-field");
}

function populateDreamerRuns(selectId, runs, defaultKey) {
  var runSelect = $(selectId);
  if (!runSelect) return;

  runSelect.innerHTML = "";
  if (!runs || runs.length === 0) {
    var opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "— No runs available —";
    runSelect.appendChild(opt);
    return;
  }

  for (var i = 0; i < runs.length; i++) {
    var run = runs[i];
    var opt = document.createElement("option");
    opt.value = run.key;
    opt.textContent = run.label || run.key;
    if (run.key === defaultKey) {
      opt.selected = true;
    }
    runSelect.appendChild(opt);
  }
}

function updateDreamerRunVisibility(strategySelectId, dreamerv3RunFieldId) {
  var strategySelect = $(strategySelectId);
  var dreamerv3RunField = $(dreamerv3RunFieldId);
  if (!strategySelect || !dreamerv3RunField) return;

  var key = strategySelect.value;
  if (key === "dreamerv3_official") {
    dreamerv3RunField.classList.add("visible");
  } else {
    dreamerv3RunField.classList.remove("visible");
  }
}

function updateScenarioDesc() {
  var scenarioSelect = $("scenario-select");
  var descEl = $("scenario-desc");
  if (!scenarioSelect || !descEl) return;

  var key = scenarioSelect.value;
  var found = scenarioByKey(key);
  if (found) {
    descEl.textContent = found.description || found.narrative || "";
  } else {
    descEl.textContent = "";
  }
}

function scenarioByKey(key) {
  for (var i = 0; i < state.scenarios.length; i++) {
    if (state.scenarios[i].key === key) {
      return state.scenarios[i];
    }
  }
  return null;
}

function sortScenariosInPlace() {
  state.scenarios.sort(function (left, right) {
    var leftSource = left && left.source ? left.source : "built_in";
    var rightSource = right && right.source ? right.source : "built_in";
    if (leftSource !== rightSource) {
      return leftSource === "built_in" ? -1 : 1;
    }
    if (leftSource === "custom") {
      return String(left.created_at || "").localeCompare(
        String(right.created_at || ""),
      );
    }
    return String(left.name || left.key || "").localeCompare(
      String(right.name || right.key || ""),
    );
  });
}

function upsertScenario(scenario) {
  if (!scenario || !scenario.key) return;
  var replaced = false;
  for (var i = 0; i < state.scenarios.length; i++) {
    if (state.scenarios[i].key === scenario.key) {
      state.scenarios[i] = scenario;
      replaced = true;
      break;
    }
  }
  if (!replaced) {
    state.scenarios.push(scenario);
  }
  sortScenariosInPlace();
}

function currentMissionScenarioKey() {
  return (($("scenario-select") || {}).value || "baseline");
}

function populateCustomScenarioBaseSelect(selectedValue) {
  var select = $("custom-scenario-base-select");
  if (!select) return;
  var currentValue = selectedValue || select.value || currentMissionScenarioKey();
  select.innerHTML = "";
  for (var i = 0; i < state.scenarios.length; i++) {
    var scenario = state.scenarios[i];
    var option = document.createElement("option");
    option.value = scenario.key;
    option.textContent =
      (scenario.name || scenario.key) +
      (scenario.source === "custom" ? " (Custom)" : "");
    if (scenario.key === currentValue) {
      option.selected = true;
    }
    select.appendChild(option);
  }
}

function suggestCustomScenarioName(baseScenario) {
  var baseName = (baseScenario && baseScenario.name) || "Custom Scenario";
  if (/custom|variant|copy/i.test(baseName)) {
    return baseName + " Copy";
  }
  return baseName + " Custom";
}

function formatCustomScenarioValue(spec, value) {
  if (value == null || value === "") {
    if (spec && spec.input === "select") {
      return "Dynamic winter mix";
    }
    return "—";
  }
  if (spec && spec.input === "select") {
    var options = spec.options || [];
    for (var i = 0; i < options.length; i++) {
      if (String(options[i].value) === String(value)) {
        return options[i].label || options[i].value;
      }
    }
    return String(value);
  }
  var numeric = Number(value);
  if (isNaN(numeric)) return String(value);
  var text = numeric.toLocaleString("en-US", {
    minimumFractionDigits:
      spec && spec.input === "integer"
        ? 0
        : Number.isInteger(numeric)
          ? 0
          : Math.abs(numeric) < 1
            ? 2
            : 1,
    maximumFractionDigits: spec && spec.input === "integer" ? 0 : 3,
  });
  return spec && spec.unit ? text + " " + spec.unit : text;
}

function formatCustomScenarioRange(spec) {
  if (!spec) return "";
  if (spec.input === "select") {
    var labels = [];
    if (spec.nullable) labels.push("Dynamic");
    var options = spec.options || [];
    for (var i = 0; i < options.length; i++) {
      labels.push(options[i].label || options[i].value);
    }
    return labels.join(" / ");
  }
  var parts = [];
  if (spec.min != null) {
    parts.push("Min " + spec.min + (spec.unit ? " " + spec.unit : ""));
  }
  if (spec.max != null) {
    parts.push("Max " + spec.max + (spec.unit ? " " + spec.unit : ""));
  }
  if (spec.step != null) {
    parts.push("Step " + spec.step);
  }
  return parts.join(" · ");
}

function renderCustomScenarioField(spec, value) {
  var inputId = "custom-scenario-field-" + spec.key;
  var baseValue = formatCustomScenarioValue(spec, value);
  var rangeLabel = formatCustomScenarioRange(spec);
  var controlHtml = "";

  if (spec.input === "select") {
    controlHtml +=
      '<select id="' +
      inputId +
      '" data-custom-scenario-field="' +
      escapeHtml(spec.key) +
      '">';
    if (spec.nullable) {
      controlHtml +=
        '<option value=""' +
        (value == null || value === "" ? " selected" : "") +
        ">Dynamic winter mix</option>";
    }
    var options = spec.options || [];
    for (var i = 0; i < options.length; i++) {
      var option = options[i];
      controlHtml +=
        '<option value="' +
        escapeHtml(option.value) +
        '"' +
        (String(option.value) === String(value) ? " selected" : "") +
        ">" +
        escapeHtml(option.label || option.value) +
        "</option>";
    }
    controlHtml += "</select>";
  } else {
    controlHtml +=
      '<input type="number" id="' +
      inputId +
      '" data-custom-scenario-field="' +
      escapeHtml(spec.key) +
      '"';
    if (spec.input === "integer") {
      controlHtml += ' inputmode="numeric"';
    } else {
      controlHtml += ' inputmode="decimal"';
    }
    if (spec.min != null) controlHtml += ' min="' + escapeHtml(spec.min) + '"';
    if (spec.max != null) controlHtml += ' max="' + escapeHtml(spec.max) + '"';
    if (spec.step != null) controlHtml += ' step="' + escapeHtml(spec.step) + '"';
    if (value != null && value !== "") {
      controlHtml += ' value="' + escapeHtml(value) + '"';
    }
    controlHtml += " />";
  }

  return (
    '<div class="setup-field custom-param-card">' +
    '<label for="' +
    inputId +
    '">' +
    escapeHtml(spec.label) +
    "</label>" +
    '<p class="custom-param-description">' +
    escapeHtml(spec.description || "") +
    "</p>" +
    '<div class="custom-param-meta">' +
    '<span class="custom-param-chip">Base ' +
    escapeHtml(baseValue) +
    "</span>" +
    (rangeLabel
      ? '<span class="custom-param-chip">' + escapeHtml(rangeLabel) + "</span>"
      : "") +
    "</div>" +
    controlHtml +
    "</div>"
  );
}

function renderCustomScenarioFields(baseScenario) {
  var container = $("custom-scenario-fields");
  if (!container) return;

  var editor = state.customScenarioEditor || {};
  var groups = editor.groups || [];
  var fields = editor.fields || [];
  if (!fields.length) {
    container.innerHTML =
      '<div class="custom-scenario-empty">Custom scenario controls are unavailable.</div>';
    return;
  }

  var groupedFields = {};
  for (var i = 0; i < fields.length; i++) {
    var field = fields[i];
    if (!groupedFields[field.group]) {
      groupedFields[field.group] = [];
    }
    groupedFields[field.group].push(field);
  }

  var html = "";
  var params = (baseScenario && baseScenario.params) || {};
  for (var g = 0; g < groups.length; g++) {
    var group = groups[g];
    var groupFields = groupedFields[group.key] || [];
    if (!groupFields.length) continue;
    html +=
      '<section class="custom-scenario-group">' +
      '<div class="custom-scenario-group-header">' +
      "<h5>" +
      escapeHtml(group.label || group.key) +
      "</h5>" +
      '<p>' +
      escapeHtml(group.description || "") +
      "</p>" +
      "</div>" +
      '<div class="custom-scenario-grid">';
    for (var f = 0; f < groupFields.length; f++) {
      var spec = groupFields[f];
      html += renderCustomScenarioField(spec, params[spec.key]);
    }
    html += "</div></section>";
  }
  container.innerHTML = html;
}

function setCustomScenarioStatus(message, kind) {
  var el = $("custom-scenario-status");
  if (!el) return;
  el.className = "custom-scenario-status";
  if (kind) {
    el.classList.add("is-" + kind);
  }
  el.textContent = message;
}

function loadCustomScenarioBase(baseScenarioKey) {
  var baseScenario =
    scenarioByKey(baseScenarioKey) ||
    scenarioByKey(currentMissionScenarioKey()) ||
    scenarioByKey("baseline") ||
    state.scenarios[0] ||
    null;
  if (!baseScenario) return;

  populateCustomScenarioBaseSelect(baseScenario.key);
  populateStrategySelect(
    "custom-scenario-strategy-select",
    baseScenario.default_strategy_key || "baseline",
  );

  var nameInput = $("custom-scenario-name-input");
  if (nameInput) {
    nameInput.value = suggestCustomScenarioName(baseScenario);
  }

  var descInput = $("custom-scenario-description-input");
  if (descInput) {
    descInput.value = baseScenario.description || "";
  }

  var narrativeInput = $("custom-scenario-narrative-input");
  if (narrativeInput) {
    narrativeInput.value = baseScenario.narrative || "";
  }

  renderCustomScenarioFields(baseScenario);
  state.customScenarioDirty = false;
  setCustomScenarioStatus(
    'Loaded "' +
      (baseScenario.name || baseScenario.key) +
      '" as the base scenario. Edit any values below and save when ready.',
    "info",
  );
}

function toggleCustomScenarioBuilder(forceOpen) {
  var shell = $("custom-scenario-shell");
  var button = $("custom-scenario-toggle-btn");
  if (!shell || !button) return;

  var shouldOpen =
    typeof forceOpen === "boolean"
      ? forceOpen
      : !shell.classList.contains("is-open");
  shell.classList.toggle("is-open", shouldOpen);
  button.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
  button.innerHTML = shouldOpen
    ? '<i class="fa-solid fa-xmark" aria-hidden="true"></i> Close Builder'
    : '<i class="fa-solid fa-sliders" aria-hidden="true"></i> Open Builder';
}

function buildCustomScenarioPayload() {
  var nameInput = $("custom-scenario-name-input");
  var descriptionInput = $("custom-scenario-description-input");
  var narrativeInput = $("custom-scenario-narrative-input");
  var baseSelect = $("custom-scenario-base-select");
  var strategySelect = $("custom-scenario-strategy-select");

  var name = nameInput ? String(nameInput.value || "").trim() : "";
  if (name.length < 3) {
    throw new Error("Custom scenario name must be at least 3 characters.");
  }

  var payload = {
    name: name,
    description: descriptionInput ? String(descriptionInput.value || "").trim() : "",
    narrative: narrativeInput ? String(narrativeInput.value || "").trim() : "",
    base_scenario_key: (baseSelect && baseSelect.value) || currentMissionScenarioKey(),
    recommended_strategy_key:
      (strategySelect && strategySelect.value) || "baseline",
  };

  var fields = (state.customScenarioEditor && state.customScenarioEditor.fields) || [];
  for (var i = 0; i < fields.length; i++) {
    var spec = fields[i];
    var input = $("custom-scenario-field-" + spec.key);
    if (!input) continue;
    if (spec.input === "select") {
      payload[spec.key] = input.value ? input.value : null;
      continue;
    }
    var numeric =
      spec.input === "integer"
        ? parseInt(input.value, 10)
        : parseFloat(input.value);
    if (isNaN(numeric)) {
      throw new Error("Invalid value for " + (spec.label || spec.key) + ".");
    }
    payload[spec.key] = numeric;
  }

  return payload;
}

function setCustomScenarioButtonBusy(busy) {
  var button = $("custom-scenario-create-btn");
  if (!button) return;
  button.disabled = busy;
  button.innerHTML = busy
    ? '<i class="fa-solid fa-spinner fa-spin" aria-hidden="true"></i> Saving…'
    : '<i class="fa-solid fa-plus" aria-hidden="true"></i> Save Scenario';
}

function createCustomScenario() {
  var payload;
  try {
    payload = buildCustomScenarioPayload();
  } catch (err) {
    setCustomScenarioStatus(err.message || "Custom scenario form is invalid.", "error");
    return Promise.resolve();
  }

  setCustomScenarioButtonBusy(true);
  setCustomScenarioStatus("Saving custom scenario to the studio store…", "info");

  return fetchJson("/api/custom-scenarios", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(function (result) {
      var scenario = result && result.scenario ? result.scenario : null;
      if (!scenario) {
        throw new Error("Server did not return the saved scenario.");
      }

      upsertScenario(scenario);
      populateScenarioSelect("scenario-select", scenario.key);
      populateScenarioSelect("lab-scenario-select", scenario.key);
      updateScenarioDesc();

      var strategySelect = $("strategy-select");
      if (strategySelect && scenario.default_strategy_key) {
        strategySelect.value = scenario.default_strategy_key;
      }
      updateDreamerRunVisibility(
        "strategy-select",
        "dreamerv3-run-field",
      );

      loadCustomScenarioBase(scenario.key);
      setLabStatus(
        'Custom scenario "' +
          (scenario.name || scenario.key) +
          '" is now available in the Simulation Lab.',
        "success",
      );
      setCustomScenarioStatus(
        'Saved "' +
          (scenario.name || scenario.key) +
          '". It is now selected in the scenario menus.',
        "success",
      );
      toggleCustomScenarioBuilder(true);
    })
    .catch(function (err) {
      console.error("Custom scenario create error:", err);
      setCustomScenarioStatus(
        err.message || "Failed to create the custom scenario.",
        "error",
      );
    })
    .finally(function () {
      setCustomScenarioButtonBusy(false);
    });
}

function setLabStatus(message, kind) {
  var el = $("lab-status");
  if (!el) return;
  el.className = "studio-lab-status";
  if (kind) {
    el.classList.add("is-" + kind);
  }
  el.textContent = message;
}

function updateTimelineLabel() {
  var slider = $("lab-timeline-slider");
  var label = $("lab-timeline-value");
  if (!slider || !label) return;
  label.textContent = monthsLabel(slider.value);
}

function showMenuView(viewId) {
  var views = ["menu-home-view", "mission-detail", "twin-detail", "lab-detail"];
  for (var i = 0; i < views.length; i++) {
    var el = $(views[i]);
    if (!el) continue;
    el.classList.toggle("active", views[i] === viewId);
  }
  var setupScreen = $("setup-screen");
  if (setupScreen) {
    setupScreen.scrollTop = 0;
  }
}

function openSetupDetail(viewId) {
  showMenuView(viewId);
  if (viewId === "twin-detail" && state.twinConfig) {
    populateTwinSetup(state.twinConfig);
  }
}

function populateScenarioSelect(selectId, selectedValue) {
  var scenarioSelect = $(selectId);
  if (!scenarioSelect) return;
  var currentValue = selectedValue || scenarioSelect.value || "baseline";
  scenarioSelect.innerHTML = "";
  for (var i = 0; i < state.scenarios.length; i++) {
    var sc = state.scenarios[i];
    var option = document.createElement("option");
    option.value = sc.key;
    option.textContent = sc.name || sc.key;
    if (sc.key === currentValue) {
      option.selected = true;
    }
    scenarioSelect.appendChild(option);
  }
}

function populateStrategySelect(selectId, selectedValue) {
  var strategySelect = $(selectId);
  if (!strategySelect) return;
  var currentValue = selectedValue || strategySelect.value || "baseline";
  strategySelect.innerHTML = "";
  for (var i = 0; i < state.strategies.length; i++) {
    var strategy = state.strategies[i];
    var option = document.createElement("option");
    option.value = strategy.key;
    option.textContent = strategy.name || strategy.key;
    if (strategy.key === currentValue) {
      option.selected = true;
    }
    strategySelect.appendChild(option);
  }
}

function labPolicyCatalog() {
  var catalog =
    state.labPolicyCatalog && state.labPolicyCatalog.length
      ? state.labPolicyCatalog
      : state.strategies;
  var available = [];
  for (var i = 0; i < catalog.length; i++) {
    if (catalog[i] && catalog[i].available !== false) {
      available.push(catalog[i]);
    }
  }
  return available;
}

function labPolicyByKey(strategyKey) {
  var catalog = labPolicyCatalog();
  for (var i = 0; i < catalog.length; i++) {
    if (catalog[i].key === strategyKey) {
      return catalog[i];
    }
  }
  return null;
}

function labPolicySlotLabel(index) {
  var alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  if (index >= 0 && index < alphabet.length) {
    return "Policy " + alphabet.charAt(index);
  }
  return "Policy " + formatInteger(index + 1);
}

function nextUnusedLabStrategyKey(usedKeys) {
  var catalog = labPolicyCatalog();
  var selectedKeys = usedKeys || [];
  for (var i = 0; i < catalog.length; i++) {
    if (selectedKeys.indexOf(catalog[i].key) === -1) {
      return catalog[i].key;
    }
  }
  return catalog.length ? catalog[0].key : "baseline";
}

function nextUnusedLabPolicyKey(usedKeys) {
  var takenKeys = usedKeys || [];
  var index = 1;
  var candidate = "policy-" + String(index);
  while (takenKeys.indexOf(candidate) !== -1) {
    index += 1;
    candidate = "policy-" + String(index);
  }
  return candidate;
}

function normalizeLabPolicyDraft(rawPolicy, index, usedStrategyKeys, usedPolicyKeys) {
  var source = rawPolicy || {};
  var strategyKey = source.strategy_key || "";
  if (!strategyKey || !labPolicyByKey(strategyKey)) {
    strategyKey = nextUnusedLabStrategyKey(usedStrategyKeys);
  }
  var strategy = labPolicyByKey(strategyKey);
  var policyKey =
    source.key && (usedPolicyKeys || []).indexOf(source.key) === -1
      ? source.key
      : nextUnusedLabPolicyKey(usedPolicyKeys);
  return {
    key: policyKey,
    label:
      source.label ||
      (strategy && strategy.name ? strategy.name : labPolicySlotLabel(index)),
    strategy_key: strategyKey,
    description:
      source.description || (strategy && strategy.description) || "",
  };
}

function syncLabPolicyControls() {
  var countEl = $("lab-policy-count");
  var helperEl = $("lab-policy-helper");
  var addBtn = $("lab-add-policy-btn");
  var useAllBtn = $("lab-all-policies-btn");
  var resetBtn = $("lab-reset-policies-btn");
  var catalogSize = labPolicyCatalog().length;
  var selectedCount = state.labPolicies.length;
  var maxPolicies = state.labPolicyMax || Math.max(2, catalogSize || 2);

  if (countEl) {
    countEl.textContent =
      formatInteger(selectedCount) +
      " of " +
      formatInteger(maxPolicies) +
      " policies selected";
  }
  if (helperEl) {
    helperEl.textContent =
      "Compare between " +
      formatInteger(state.labPolicyMin || 2) +
      " and " +
      formatInteger(maxPolicies) +
      " available policies in one lab run.";
  }
  if (addBtn) {
    addBtn.disabled = selectedCount >= maxPolicies;
  }
  if (useAllBtn) {
    useAllBtn.disabled = !catalogSize || selectedCount >= maxPolicies;
  }
  if (resetBtn) {
    resetBtn.disabled = !state.studioSetup;
  }
}

function renderLabPolicyComposer() {
  var container = $("lab-policy-list");
  if (!container) return;
  var policies = state.labPolicies || [];
  var catalog = labPolicyCatalog();
  container.innerHTML = policies
    .map(function (policy, index) {
      var canRemove = policies.length > (state.labPolicyMin || 2);
      var strategy = labPolicyByKey(policy.strategy_key);
      var description =
        (strategy && strategy.description) || policy.description || "";
      var optionsHtml = catalog
        .map(function (entry) {
          return (
            '<option value="' +
            escapeHtml(entry.key) +
            '"' +
            (entry.key === policy.strategy_key ? " selected" : "") +
            ">" +
            escapeHtml(entry.name || entry.key) +
            "</option>"
          );
        })
        .join("");
      return (
        '<div class="studio-policy-card">' +
        '<div class="studio-policy-card-header">' +
        '<div class="studio-policy-card-title"><strong>' +
        escapeHtml(labPolicySlotLabel(index)) +
        "</strong><span>Branch " +
        escapeHtml(String(index + 1)) +
        "</span></div>" +
        '<button type="button" class="studio-policy-remove" data-policy-action="remove" data-policy-index="' +
        escapeHtml(String(index)) +
        '"' +
        (canRemove ? "" : " disabled") +
        '>Remove</button>' +
        "</div>" +
        '<div class="studio-policy-card-grid">' +
        '<div class="setup-field">' +
        '<label for="lab-policy-label-' +
        escapeHtml(String(index)) +
        '"><i class="fa-solid fa-tag" aria-hidden="true"></i> Branch Label</label>' +
        '<input type="text" id="lab-policy-label-' +
        escapeHtml(String(index)) +
        '" value="' +
        escapeHtml(policy.label || "") +
        '" data-policy-action="label" data-policy-index="' +
        escapeHtml(String(index)) +
        '" />' +
        "</div>" +
        '<div class="setup-field">' +
        '<label for="lab-policy-select-' +
        escapeHtml(String(index)) +
        '"><i class="fa-solid fa-code-branch" aria-hidden="true"></i> Strategy</label>' +
        '<select id="lab-policy-select-' +
        escapeHtml(String(index)) +
        '" data-policy-action="strategy" data-policy-index="' +
        escapeHtml(String(index)) +
        '">' +
        optionsHtml +
        "</select>" +
        "</div>" +
        "</div>" +
        '<p class="studio-policy-description">' +
        escapeHtml(description) +
        "</p>" +
        "</div>"
      );
    })
    .join("");
  syncLabPolicyControls();
}

function setLabPolicies(rawPolicies) {
  var catalog = labPolicyCatalog();
  var maxPolicies = state.labPolicyMax || Math.max(2, catalog.length || 2);
  var minPolicies = state.labPolicyMin || 2;
  var source = rawPolicies && rawPolicies.length ? rawPolicies.slice(0, maxPolicies) : [];
  var normalized = [];
  var usedStrategyKeys = [];
  var usedPolicyKeys = [];
  for (var i = 0; i < source.length; i++) {
    var policy = normalizeLabPolicyDraft(
      source[i],
      i,
      usedStrategyKeys,
      usedPolicyKeys,
    );
    normalized.push(policy);
    usedStrategyKeys.push(policy.strategy_key);
    usedPolicyKeys.push(policy.key);
  }
  while (normalized.length < minPolicies) {
    var fallback = normalizeLabPolicyDraft(
      {},
      normalized.length,
      usedStrategyKeys,
      usedPolicyKeys,
    );
    normalized.push(fallback);
    usedStrategyKeys.push(fallback.strategy_key);
    usedPolicyKeys.push(fallback.key);
  }
  state.labPolicies = normalized.slice(0, maxPolicies);
  renderLabPolicyComposer();
}

function addLabPolicy() {
  var catalog = labPolicyCatalog();
  var maxPolicies = state.labPolicyMax || Math.max(2, catalog.length || 2);
  if (state.labPolicies.length >= maxPolicies) {
    setLabStatus("All available policies are already selected.", "warning");
    return;
  }
  var usedStrategyKeys = state.labPolicies.map(function (policy) {
    return policy.strategy_key;
  });
  var usedPolicyKeys = state.labPolicies.map(function (policy) {
    return policy.key;
  });
  var nextPolicy = normalizeLabPolicyDraft(
    {},
    state.labPolicies.length,
    usedStrategyKeys,
    usedPolicyKeys,
  );
  state.labPolicies.push(nextPolicy);
  renderLabPolicyComposer();
}

function resetLabPolicies() {
  var defaults =
    state.studioSetup && state.studioSetup.default_universes
      ? state.studioSetup.default_universes
      : [];
  setLabPolicies(defaults);
}

function selectAllLabPolicies() {
  var catalog = labPolicyCatalog();
  var allPolicies = [];
  for (var i = 0; i < catalog.length; i++) {
    allPolicies.push({
      key: "policy-" + String(i + 1),
      label: catalog[i].name || labPolicySlotLabel(i),
      strategy_key: catalog[i].key,
      description: catalog[i].description || "",
    });
  }
  setLabPolicies(allPolicies);
}

function updateLabPolicyDraft(index, field, value) {
  if (index < 0 || index >= state.labPolicies.length) return;
  var draft = state.labPolicies[index];
  if (field === "label") {
    draft.label = value || "";
  } else if (field === "strategy") {
    var previousStrategy = labPolicyByKey(draft.strategy_key);
    var nextStrategy = labPolicyByKey(value);
    var currentLabel = (draft.label || "").trim();
    draft.strategy_key = value;
    draft.description = nextStrategy && nextStrategy.description
      ? nextStrategy.description
      : "";
    if (
      !currentLabel ||
      (previousStrategy && currentLabel === previousStrategy.name) ||
      currentLabel === labPolicySlotLabel(index)
    ) {
      draft.label = nextStrategy && nextStrategy.name
        ? nextStrategy.name
        : labPolicySlotLabel(index);
    }
  }
  renderLabPolicyComposer();
}

function removeLabPolicy(index) {
  if (state.labPolicies.length <= (state.labPolicyMin || 2)) {
    setLabStatus("Simulation Lab requires at least two policies.", "warning");
    return;
  }
  state.labPolicies.splice(index, 1);
  renderLabPolicyComposer();
}

function handleLabPolicyComposerInput(event) {
  var target = event.target;
  if (!target) return;
  var action = target.getAttribute("data-policy-action");
  var index = parseInt(target.getAttribute("data-policy-index"), 10);
  if (isNaN(index)) return;
  if (action === "label" && index >= 0 && index < state.labPolicies.length) {
    state.labPolicies[index].label = target.value || "";
  }
}

function handleLabPolicyComposerChange(event) {
  var target = event.target;
  if (!target) return;
  var action = target.getAttribute("data-policy-action");
  var index = parseInt(target.getAttribute("data-policy-index"), 10);
  if (isNaN(index)) return;
  if (action === "strategy") {
    updateLabPolicyDraft(index, "strategy", target.value);
  }
}

function handleLabPolicyComposerClick(event) {
  var target = event.target;
  if (!target) return;
  var button = target.closest("[data-policy-action='remove']");
  if (!button) return;
  var index = parseInt(button.getAttribute("data-policy-index"), 10);
  if (isNaN(index)) return;
  removeLabPolicy(index);
}

function buildLabUniversePayloads() {
  var usedPolicyKeys = [];
  return (state.labPolicies || []).map(function (policy, index) {
    var strategy = labPolicyByKey(policy.strategy_key);
    var key =
      policy.key && usedPolicyKeys.indexOf(policy.key) === -1
        ? policy.key
        : nextUnusedLabPolicyKey(usedPolicyKeys);
    usedPolicyKeys.push(key);
    return {
      key: key,
      label:
        (policy.label || "").trim() ||
        ((strategy && strategy.name) || labPolicySlotLabel(index)),
      strategy_key: policy.strategy_key || "baseline",
      description: policy.description || ((strategy && strategy.description) || ""),
    };
  });
}

function populateStudioSetup(data) {
  state.studioSetup = data;
  state.labPolicyCatalog = data.policy_catalog || [];
  state.labPolicyMin =
    data.policy_selection && data.policy_selection.min_policies
      ? data.policy_selection.min_policies
      : 2;
  state.labPolicyMax =
    data.policy_selection && data.policy_selection.max_policies
      ? data.policy_selection.max_policies
      : Math.max(state.labPolicyMin, labPolicyCatalog().length || 2);

  var backendBadge = $("lab-backend-badge");
  if (backendBadge && data.snapshot_backend) {
    backendBadge.textContent = data.snapshot_backend.label || "Snapshot adapter";
    backendBadge.className = "studio-backend-badge";
    if (data.snapshot_backend.adapter === "sql_table") {
      backendBadge.classList.add("is-sql");
    } else {
      backendBadge.classList.add("is-model");
    }
  }

  var sourceSelect = $("lab-snapshot-source");
  if (sourceSelect && data.snapshot_sources) {
    sourceSelect.innerHTML = "";
    for (var i = 0; i < data.snapshot_sources.length; i++) {
      var source = data.snapshot_sources[i];
      var option = document.createElement("option");
      option.value = source;
      option.textContent = source === "django" ? "Django / Postgres" : "Synthetic";
      if (source === "django") option.selected = true;
      sourceSelect.appendChild(option);
    }
  }

  var slider = $("lab-timeline-slider");
  if (slider && data.timeline_slider) {
    slider.min = String(data.timeline_slider.min_months || 0);
    slider.max = String(data.timeline_slider.max_months || 24);
    slider.value = String(
      data.timeline_slider.default_months != null
        ? data.timeline_slider.default_months
        : 0,
    );
  }
  updateTimelineLabel();
  setLabPolicies(data.default_universes || []);

  var results = $("lab-results");
  if (results) {
    results.innerHTML =
      '<div class="lab-result-card"><h4>Ready</h4><div class="lab-branch-list"><div class="lab-branch"><div class="lab-branch-main"><strong>Snapshot layer online</strong><span>Use the controls above to compare multiple policies or ask the strategy assistant.</span></div></div></div></div>';
  }

  setLabStatus(
    data.snapshot_backend && data.snapshot_backend.adapter === "sql_table"
      ? "Connected to the real donor table through Postgres."
      : "Studio is ready. The donor snapshot will use the configured Django adapter.",
    "success",
  );
}

function currentScenarioKey() {
  var scenarioSelect = $("lab-scenario-select") || $("scenario-select");
  return (scenarioSelect && scenarioSelect.value) || "baseline";
}

function currentSeed() {
  return parseInt(($("seed-input") || {}).value, 10) || 100;
}

function currentHoursOverride() {
  return parseInt(($("hours-input") || {}).value, 10) || 168;
}

function currentDreamerV3RunKey() {
  var runSelect = $("dreamerv3-run-select");
  if (runSelect && runSelect.value) {
    return runSelect.value;
  }
  return state.defaultDreamerv3RunKey || "";
}

function buildStudioSnapshotPayload() {
  var sourceSelect = $("lab-snapshot-source");
  return {
    source: (sourceSelect && sourceSelect.value) || "django",
    include_raw_donors: false,
    include_alert_events: true,
    include_model_registry: true,
  };
}

function buildStudioExperimentPayload() {
  var replications = parseInt(($("lab-replications-input") || {}).value, 10) || 2;
  var timelineValue = parseInt(($("lab-timeline-slider") || {}).value, 10);
  var timelineMonths = isNaN(timelineValue) ? 0 : timelineValue;
  var replayEnabled = $("lab-replay-toggle")
    ? $("lab-replay-toggle").checked
    : false;
  var agentEnabled = $("lab-agent-toggle")
    ? $("lab-agent-toggle").checked
    : false;

  return {
    scenario_key: currentScenarioKey(),
    seed: currentSeed(),
    replications: replications,
    hours_override: currentHoursOverride(),
    step_hours: 6,
    timeline_months_ahead: timelineMonths,
    include_timeline: false,
    dreamerv3_run_key: currentDreamerV3RunKey(),
    snapshot: buildStudioSnapshotPayload(),
    universes: buildLabUniversePayloads(),
    replay: {
      mode: replayEnabled ? "alerts" : "none",
    },
    agent_mode: {
      enabled: agentEnabled,
    },
  };
}

function buildAssistantPayload() {
  var replications = parseInt(($("lab-replications-input") || {}).value, 10) || 2;
  var timelineValue = parseInt(($("lab-timeline-slider") || {}).value, 10);
  var timelineMonths = isNaN(timelineValue) ? 0 : timelineValue;
  var dropoutRise = parseFloat(($("lab-dropout-input") || {}).value) || 12;

  return {
    scenario_key: currentScenarioKey(),
    seed: currentSeed(),
    replications: replications,
    hours_override: currentHoursOverride(),
    timeline_months_ahead: timelineMonths,
    dropout_rise_pct: dropoutRise,
    dreamerv3_run_key: currentDreamerV3RunKey(),
    snapshot: buildStudioSnapshotPayload(),
    candidate_universes: buildLabUniversePayloads(),
  };
}

function setButtonBusy(buttonId, busy, busyText, idleHtml) {
  var button = $(buttonId);
  if (!button) return;
  button.disabled = busy;
  button.innerHTML = busy ? busyText : idleHtml;
}

function renderSnapshotSummary(snapshot) {
  var summary = snapshot.donor_summary || {};
  var metrics = collectSnapshotMetrics(summary);
  return (
    '<div class="lab-result-card">' +
    "<h4>Snapshot Summary</h4>" +
    '<div class="lab-result-grid">' +
    metrics
      .map(function (metric) {
        return renderLabMetric(metric, "");
      })
      .join("") +
    "</div>" +
    renderBloodMix(summary.blood_type_distribution || {}) +
    "</div>"
  );
}

function renderExperimentResult(payload) {
  var results = $("lab-results");
  if (!results) return;
  var universes = payload.universes || [];
  var branchHtml = "";
  for (var i = 0; i < universes.length; i++) {
    branchHtml += renderUniverseMetricsBoard(universes[i], i);
  }

  var best = universes.length ? universes[0] : null;
  var bestSummary = best ? best.summary || {} : {};
  results.innerHTML =
    renderSnapshotSummary(payload.snapshot || {}) +
    '<div class="lab-result-card"><h4>Mission Control</h4><div class="lab-result-grid">' +
    '<div class="lab-metric lab-metric-primary"><span class="lab-metric-label">Best Shortage</span><span class="lab-metric-value">' +
    formatPercent(bestSummary.shortage_rate_mean || 0) +
    "</span><span class=\"lab-metric-range\">Lowest projected shortage rate across the compared branches.</span></div>" +
    '<div class="lab-metric"><span class="lab-metric-label">Horizon</span><span class="lab-metric-value">' +
    monthsLabel(payload.timeline_months_ahead) +
    "</span></div>" +
    '<div class="lab-metric"><span class="lab-metric-label">Best Branch</span><span class="lab-metric-value">' +
    escapeHtml(best ? best.label : "—") +
    "</span></div>" +
    '<div class="lab-metric"><span class="lab-metric-label">Episode Score</span><span class="lab-metric-value">' +
    formatNumber(bestSummary.episode_score_mean || 0) +
    "</span></div>" +
    '<div class="lab-metric"><span class="lab-metric-label">Budget Spent</span><span class="lab-metric-value">' +
    formatNumber(bestSummary.budget_spent_mean || 0) +
    "</span></div>" +
    '<div class="lab-metric"><span class="lab-metric-label">Confidence Bands</span><span class="lab-metric-value">' +
    formatInteger(
      best && best.confidence_bands ? best.confidence_bands.length : 0,
    ) +
    "</span></div>" +
    "</div></div>" +
    '<div class="lab-result-card"><h4>Parallel Universes</h4><p class="lab-result-copy">All aggregate metrics are shown below. Branches are ranked by lowest shortage rate first, then by episode score, reward total, and service rate.</p><div class="lab-branch-list">' +
    branchHtml +
    "</div></div>";
}

function renderAssistantResult(payload) {
  var results = $("lab-results");
  if (!results) return;
  var rankings = payload.rankings || [];
  var rankingHtml = "";
  for (var i = 0; i < rankings.length; i++) {
    rankingHtml += renderUniverseMetricsBoard(
      {
        label: rankings[i].label,
        strategy_name: rankings[i].strategy_name,
        summary: rankings[i].summary || {},
      },
      i,
    );
  }
  var best = payload.recommendation && payload.recommendation.best_universe
    ? payload.recommendation.best_universe
    : null;
  var bestSummary = best && best.summary ? best.summary : {};

  results.innerHTML =
    '<div class="lab-result-card"><h4>Strategy Assistant</h4><div class="lab-result-grid">' +
    '<div class="lab-metric lab-metric-primary"><span class="lab-metric-label">Recommended Shortage</span><span class="lab-metric-value">' +
    formatPercent(bestSummary.shortage_rate_mean || 0) +
    '</span><span class="lab-metric-range">The assistant now prioritizes shortage containment.</span></div>' +
    '<div class="lab-metric"><span class="lab-metric-label">Best Branch</span><span class="lab-metric-value">' +
    escapeHtml(
      best && best.label
        ? best.label
        : "No recommendation",
    ) +
    "</span></div>" +
    '<div class="lab-metric"><span class="lab-metric-label">Episode Score</span><span class="lab-metric-value">' +
    formatNumber(bestSummary.episode_score_mean || 0) +
    "</span></div>" +
    '<div class="lab-metric"><span class="lab-metric-label">Budget Spent</span><span class="lab-metric-value">' +
    formatNumber(bestSummary.budget_spent_mean || 0) +
    "</span></div>" +
    "</div>" +
    '<p class="lab-result-copy">' +
    escapeHtml(
      payload.recommendation && payload.recommendation.reasoning
        ? payload.recommendation.reasoning
        : "",
    ) +
    "</p></div>" +
    '<div class="lab-result-card"><h4>Rankings</h4><p class="lab-result-copy">All aggregate metrics for each candidate branch are shown below.</p><div class="lab-branch-list">' +
    rankingHtml +
    "</div></div>";
}

function runStudioExperiment() {
  setButtonBusy(
    "lab-run-btn",
    true,
    '<i class="fa-solid fa-spinner fa-spin"></i> Running…',
    '<i class="fa-solid fa-flask-vial" aria-hidden="true"></i> Run Lab',
  );
  setLabStatus("Running snapshot experiment across the selected policy branches…", null);

  return fetchJson("/api/studio/experiments/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildStudioExperimentPayload()),
  })
    .then(function (payload) {
      state.studioResult = payload;
      renderExperimentResult(payload);
      setLabStatus("Simulation Lab run completed.", "success");
    })
    .catch(function (err) {
      console.error("Studio experiment error:", err);
      setLabStatus(err.message || "Studio experiment failed.", "error");
    })
    .finally(function () {
      setButtonBusy(
        "lab-run-btn",
        false,
        "",
        '<i class="fa-solid fa-flask-vial" aria-hidden="true"></i> Run Lab',
      );
    });
}

function runStrategyAssistant() {
  setButtonBusy(
    "lab-assistant-btn",
    true,
    '<i class="fa-solid fa-spinner fa-spin"></i> Thinking…',
    '<i class="fa-solid fa-brain" aria-hidden="true"></i> Ask Strategy Assistant',
  );
  setLabStatus("Evaluating dropout shock responses across the selected policy branches…", null);

  return fetchJson("/api/studio/assistant/strategy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildAssistantPayload()),
  })
    .then(function (payload) {
      state.studioAssistantResult = payload;
      renderAssistantResult(payload);
      setLabStatus("Strategy assistant finished ranking the candidate branches.", "success");
    })
    .catch(function (err) {
      console.error("Strategy assistant error:", err);
      setLabStatus(err.message || "Strategy assistant failed.", "error");
    })
    .finally(function () {
      setButtonBusy(
        "lab-assistant-btn",
        false,
        "",
        '<i class="fa-solid fa-brain" aria-hidden="true"></i> Ask Strategy Assistant',
      );
    });
}

/* ----------------------------------------------------------
   WebSocket management
   ---------------------------------------------------------- */

function openWebSocket() {
  var protocol = location.protocol === "https:" ? "wss:" : "ws:";
  var url = protocol + "//" + location.host + "/api/sim/ws";

  var ws = new WebSocket(url);

  ws.onopen = function () {
    setConnectionStatus("Connected", true);
    sendStartCommand(ws);
  };

  ws.onmessage = function (event) {
    var msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      console.error("Bad WS message", e);
      return;
    }
    handleMessage(msg);
  };

  ws.onerror = function () {
    setConnectionStatus("Connection error", false);
  };

  ws.onclose = function () {
    setConnectionStatus("Disconnected", false);
    state.ws = null;
  };

  state.ws = ws;
}

function sendStartCommand(ws) {
  var scenarioKey = ($("scenario-select") || {}).value || "";
  var strategyKey = ($("strategy-select") || {}).value || "";
  var seed = parseInt(($("seed-input") || {}).value, 10) || 100;
  var hours = parseInt(($("hours-input") || {}).value, 10) || 168;
  var speed = parseFloat(($("speed-input") || {}).value) || 1;

  state.speed = speed;

  var cmd = {
    command: "start",
    scenario_key: scenarioKey,
    strategy_key: strategyKey,
    seed: seed,
    hours_override: hours,
    step_hours: 6,
    speed: speed,
    use_operational_seed: true,
  };

  // Include DreamerV3 run key if applicable
  if (strategyKey === "dreamerv3_official") {
    var runSelect = $("dreamerv3-run-select");
    if (runSelect && runSelect.value) {
      cmd.dreamerv3_run_key = runSelect.value;
    }
  }

  var forecastSelect = $("dt-forecast-model");
  if (forecastSelect && forecastSelect.value) {
    cmd.forecast_job_id = forecastSelect.value;
    state.twinForecastJobId = forecastSelect.value;
  } else if (state.twinForecastJobId) {
    cmd.forecast_job_id = state.twinForecastJobId;
  }

  ws.send(JSON.stringify(cmd));
}

function sendCommand(cmd) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(cmd));
  }
}

function closeWebSocket() {
  if (state.ws) {
    try {
      state.ws.close();
    } catch (e) {
      /* ignore */
    }
    state.ws = null;
  }
}

/* ----------------------------------------------------------
   Digital Twin – WebSocket
   ---------------------------------------------------------- */

function openTwinWebSocket() {
  var protocol = location.protocol === "https:" ? "wss:" : "ws:";
  var url = protocol + "//" + location.host + "/api/twin/ws";
  var ws = new WebSocket(url);

  ws.onopen = function () {
    setConnectionStatus("Connected (Twin)", true);
    sendTwinStartCommand(ws);
  };
  ws.onmessage = function (event) {
    var msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      return;
    }
    handleTwinMessage(msg);
  };
  ws.onerror = function () {
    setConnectionStatus("Connection error", false);
  };
  ws.onclose = function () {
    setConnectionStatus("Disconnected", false);
    state.ws = null;
  };
  state.ws = ws;
}

function sendTwinStartCommand(ws) {
  var scenarioKey = ($("scenario-select") || {}).value || "baseline";
  var strategyKey = ($("twin-strategy-select") || {}).value || "baseline";
  var seed = parseInt(($("twin-seed-input") || {}).value, 10) || 100;
  var speed = parseFloat(($("twin-speed-input") || {}).value) || 1;
  var tickInterval = parseFloat(($("tick-interval-input") || {}).value) || 2;
  var budgetCycle = parseInt(($("budget-cycle-select") || {}).value, 10) || 168;
  var liveData = $("live-data-toggle") ? $("live-data-toggle").checked : false;
  var stepHours = parseFloat(($("twin-hours-input") || {}).value) || 6;

  state.speed = speed;

  var cmd = {
    command: "start",
    scenario_key: state.dynamicWorldEnabled ? "baseline" : scenarioKey,
    strategy_key: strategyKey,
    seed: seed,
    tick_interval_s: tickInterval,
    step_hours: stepHours,
    enable_live_data: liveData,
    enable_dynamic_world: state.dynamicWorldEnabled,
    budget_cycle_hours: budgetCycle,
    speed: speed,
    random_event_rate: parseFloat(
      (document.getElementById("random-event-rate") || {}).value || "0.5",
    ),
  };

  if (strategyKey === "dreamerv3_official") {
    var runSelect = $("twin-dreamerv3-run-select");
    if (runSelect && runSelect.value) {
      cmd.dreamerv3_run_key = runSelect.value;
    }
  }

  var forecastSelect = $("dt-forecast-model");
  if (forecastSelect && forecastSelect.value) {
    cmd.forecast_job_id = forecastSelect.value;
    state.twinForecastJobId = forecastSelect.value;
  } else if (state.twinForecastJobId) {
    cmd.forecast_job_id = state.twinForecastJobId;
  }

  ws.send(JSON.stringify(cmd));
}

/* ----------------------------------------------------------
   Message dispatch
   ---------------------------------------------------------- */

function handleMessage(msg) {
  switch (msg.type) {
    case "init":
      handleInit(msg);
      break;
    case "step":
      handleStep(msg);
      break;
    case "complete":
      handleComplete(msg);
      break;
    case "error":
      handleError(msg);
      break;
    case "paused":
      handlePaused();
      break;
    case "resumed":
      handleResumed();
      break;
    case "stopped":
      handleStopped();
      break;
    case "forecast_panel_updated":
      handleForecastPanelUpdated(msg);
      break;
    default:
      console.warn("Unknown message type:", msg.type);
  }
}

/* ----------------------------------------------------------
   INIT handler
   ---------------------------------------------------------- */

function handleInit(msg) {
  state.initData = msg;
  state.centers = msg.centers || [];
  state.paused = false;
  state.previousHud = {};
  setSimRunning(true);
  state.previousDonated = 0;
  state.previousTransfused = 0;
  state.previousActiveActionsByKey = {};
  // Reset coalesced-render state so the first step of every run renders fresh.
  state.lastActionsSig = null;
  if (state.stepRenderHandle) {
    cancelAnimationFrame(state.stepRenderHandle);
    state.stepRenderHandle = null;
  }
  state.donorCountMap = {};
  state.twinRunId = null;
  state.twinSnapshotId = null;
  state.twinConfigVersion = null;
  state.twinSourceStatus = null;
  state.twinLastActionId = null;
  state.twinEvents = [];

  var scenarioInfo = $("scenario-info");
  if (scenarioInfo && msg.scenario) {
    scenarioInfo.innerHTML =
      "<strong>" +
      escapeHtml(msg.scenario.name || msg.scenario.key || "") +
      "</strong>" +
      (msg.scenario.description
        ? "<p>" + escapeHtml(msg.scenario.description) + "</p>"
        : "");
  }

  var strategyInfo = $("strategy-info");
  if (strategyInfo && msg.strategy) {
    var sName =
      typeof msg.strategy === "string"
        ? msg.strategy
        : msg.strategy.name || msg.strategy.key || "";
    var sDesc =
      typeof msg.strategy === "object" ? msg.strategy.description || "" : "";
    strategyInfo.innerHTML =
      "<strong>" +
      escapeHtml(sName) +
      "</strong>" +
      (sDesc ? "<p>" + escapeHtml(sDesc) + "</p>" : "");

    updateControllerPanelVisibility(msg);
  }

  updateControllerPanelVisibility(msg);
  renderControllerDecision(null);
  renderControllerActiveActions([]);

  // Build the map
  syncSceneOperationalFrame(msg);
  buildMapBoard(state.centers);

  setHudValue("hud-score", "0.00");
  setHudValue("hud-time", "Hour 0/" + (msg.total_hours || "?"));
  setHudValue("hud-shortage", "0%");
  setHudValue("hud-budget", "$" + formatNumber(msg.budget_total || 0));
  setHudValue(
    "hud-weather",
    weatherLabels[msg.weather] || msg.weather || "Clear",
  );
  setHudValue("hud-donated", "0");
  setHudValue("hud-transfused", "0");
  setHudValue("hud-expired", "0");

  updateWeatherOverlay(msg.weather || "clear");
  updateDayNight(0);
  updateProgress(0, 0, msg.total_hours || 0);
  updateRuntimeClock(0, msg.total_hours || 0);
  updateRuntimeDashboard(msg);
  clearEventsFeed();
  clearActionsList();
  updatePauseButton(false);
  highlightSpeedButton(state.speed);
  highlightRuntimeSpeed(state.speed);

  // Update weather icon in HUD
  updateWeatherIcon(msg.weather || "clear");

  showTwinPanels(false);
  if (window.DashboardUI && window.DashboardUI.resetForecastTickSeries) {
    window.DashboardUI.resetForecastTickSeries();
  }
  initTwinForecastPanel(msg.forecast_panel || {});
  if (msg.forecast_job_id) {
    state.twinForecastJobId = msg.forecast_job_id;
  }
  if (msg.forecast_live) {
    applyTwinForecastPanel(msg.forecast_live, { resetSeries: true, appendChart: true });
  }
}

/* ----------------------------------------------------------
   STEP handler
   ---------------------------------------------------------- */

function handleStep(msg) {
  state.lastStepData = msg;
  state.currentStep = msg.step;

  // Accumulating work that must run for EVERY message — these append to the
  // event log and the forecast chart series, so coalescing would drop entries.
  if (msg.forecast_panel) {
    applyTwinForecastPanel(msg.forecast_panel, { appendChart: true });
  }
  addControllerEventEntry(msg);
  addEventEntry(msg);

  // Idempotent display work (HUD, overlays, map, action lists) only needs to
  // reflect the LATEST step, so coalesce it to one render per animation frame.
  // At 5x/10x playback steps arrive faster than 60fps; this drops the per-step
  // DOM churn from N-per-frame to 1-per-frame with no visible difference.
  scheduleStepRender();
}

function scheduleStepRender() {
  if (state.stepRenderHandle) return;
  state.stepRenderHandle = requestAnimationFrame(function () {
    state.stepRenderHandle = null;
    var msg = state.lastStepData;
    if (msg) renderStepDisplay(msg);
  });
}

function actionsSignature(actions) {
  if (!actions || !actions.length) return "";
  var parts = [];
  for (var i = 0; i < actions.length; i++) {
    var a = actions[i];
    var key = a.key || a.action_key || a.name || "action-" + i;
    parts.push(key + ":" + Math.round((a.intensity || 0) * 100));
  }
  return parts.join("|");
}

function renderStepDisplay(msg) {
  setHudValue("hud-score", formatNumber(msg.score));
  setHudValue(
    "hud-time",
    "Hour " + Math.round(msg.hour) + "/" + msg.total_hours,
  );
  updateRuntimeClock(msg.hour || 0, msg.total_hours);
  setHudValue("hud-shortage", formatPercent(msg.shortage_rate));
  setHudValue("hud-budget", "$" + formatNumber(msg.budget_remaining));
  setHudValue(
    "hud-weather",
    weatherLabels[msg.weather] || msg.weather || "Clear",
  );
  setHudValue("hud-donated", formatInteger(msg.total_donated));
  setHudValue("hud-transfused", formatInteger(msg.total_transfused));
  setHudValue("hud-expired", formatInteger(msg.total_expired));

  updateWeatherOverlay(msg.weather || "clear");
  updateWeatherIcon(msg.weather || "clear");
  updateDayNight(msg.hour || 0);
  updateShortagePulse(msg.shortage_rate || 0);
  updateProgress(msg.progress, msg.hour, msg.total_hours);
  syncSceneOperationalFrame(msg);
  updateRuntimeDashboard(msg);

  if (msg.active_actions) {
    // Skip rebuilding the action lists when the active set is unchanged since
    // the last rendered frame (S3) — avoids redundant innerHTML churn.
    var sig = actionsSignature(msg.active_actions);
    if (sig !== state.lastActionsSig) {
      state.lastActionsSig = sig;
      renderActionsList(msg.active_actions);
      renderControllerActiveActions(msg.active_actions);
    }
  }

  if (msg.controller_decision) {
    renderControllerDecision(msg.controller_decision);
  } else if (strategyExpectsRlController(state.initData)) {
    renderControllerDecision({
      headline: "Held previous controls",
      controller:
        (state.initData && state.initData.runtime_controller) || "controller",
      hour: msg.hour,
      top_levels: [],
      has_changes: false,
    });
  }

  // Update donor count map
  if (msg.donor_count_by_center) {
    state.donorCountMap = {};
    for (var d = 0; d < msg.donor_count_by_center.length; d++) {
      var dc = msg.donor_count_by_center[d];
      state.donorCountMap[dc.name] = dc;
    }
  }

  if (msg.centers && msg.centers.length > 0) {
    // Check donation difference for floater animation
    var newDonated = msg.total_donated || 0;
    var donatedDiff = newDonated - state.previousDonated;
    state.previousDonated = newDonated;

    // Detect transfusions for transfer lines
    var newTransfused = msg.total_transfused || 0;
    var transfusedDiff = newTransfused - state.previousTransfused;
    state.previousTransfused = newTransfused;

    state.centers = msg.centers;
    updateMapBoard(msg.centers, donatedDiff, transfusedDiff);
  }
}

/* ----------------------------------------------------------
   COMPLETE handler
   ---------------------------------------------------------- */

function handleComplete(msg) {
  state.report = msg.report || msg;
  setSimRunning(false);
  showScreen("report-screen");
  renderReport(state.report);
  stopSnow();
  state.snowActive = false;
  closeWebSocket();
}

/* ----------------------------------------------------------
   ERROR handler
   ---------------------------------------------------------- */

function handleError(msg) {
  var gameScreen = $("game-screen");
  if (!gameScreen) return;

  var existing = gameScreen.querySelector(".error-banner");
  if (existing) existing.remove();

  var banner = document.createElement("div");
  banner.className = "error-banner";
  banner.innerHTML =
    "<strong>Error:</strong> " + escapeHtml(msg.message || "Unknown error");
  banner.onclick = function () {
    banner.remove();
  };
  gameScreen.insertBefore(banner, gameScreen.firstChild);
}

/* ----------------------------------------------------------
   PAUSED / RESUMED / STOPPED
   ---------------------------------------------------------- */

function handlePaused() {
  state.paused = true;
  updatePauseButton(true);
}

function handleResumed() {
  state.paused = false;
  updatePauseButton(false);
}

function handleStopped() {
  state.paused = false;
  setSimRunning(false);
  stopSnow();
  state.snowActive = false;
  if (state.lastStepData || state.report) {
    showScreen("report-screen");
    if (state.report) {
      renderReport(state.report);
    } else {
      renderPartialReport();
    }
  } else {
    showScreen("setup-screen");
  }
  closeWebSocket();
}

/* ----------------------------------------------------------
   Digital Twin – message dispatch
   ---------------------------------------------------------- */

function handleTwinMessage(msg) {
  switch (msg.type) {
    case "twin_init":
      handleTwinInit(msg);
      break;
    case "twin_tick":
      handleTwinTick(msg);
      break;
    case "forecast_panel_updated":
      handleForecastPanelUpdated(msg);
      break;
    case "twin_event":
      handleTwinEvent(msg);
      break;
    case "budget_refresh":
      handleBudgetRefresh(msg);
      break;
    case "action_result":
      handleActionResult(msg);
      break;
    case "branch_result":
      handleBranchResult(msg);
      break;
    case "recommendation_result":
      handleRecommendationResult(msg);
      break;
    case "promotion_result":
      handlePromotionResult(msg);
      break;
    case "data_status":
      handleDataStatus(msg);
      break;
    case "paused":
      handlePaused();
      break;
    case "resumed":
      handleResumed();
      break;
    case "stopped":
      handleStopped();
      break;
    case "error":
      handleError(msg);
      break;
    case "info":
      if (msg.message) {
        showToast({
          title: "Info",
          description: msg.message,
          severity: 0.1,
          icon: "fa-circle-info",
          type: "info",
          duration: 3000,
        });
      }
      break;
    default:
      console.warn("Unknown twin message type:", msg.type);
  }
}

function applyTwinForecastPanel(panel, options) {
  if (!window.DashboardUI || typeof window.DashboardUI.mergeForecastTick !== "function") {
    return;
  }
  window.DashboardUI.mergeForecastTick(panel || {}, options || {});
}

function initTwinForecastPanel(meta) {
  meta = meta || {};
  if (!window.DashboardUI) return;
  if (typeof window.DashboardUI.populateForecastModelSelect === "function") {
    var jobId = window.DashboardUI.populateForecastModelSelect(
      meta,
      state.twinForecastJobId,
    );
    if (jobId) state.twinForecastJobId = jobId;
  }
  if (typeof window.DashboardUI.bindForecastModelSelect === "function") {
    window.DashboardUI.bindForecastModelSelect(function (selectedJobId) {
      state.twinForecastJobId = selectedJobId;
      if (window.DashboardUI.resetForecastTickSeries) {
        window.DashboardUI.resetForecastTickSeries();
      }
      sendCommand({ command: "set_forecast_model", job_id: selectedJobId });
    });
  }
}

function handleForecastPanelUpdated(msg) {
  applyTwinForecastPanel(msg.forecast_panel, { resetSeries: true, appendChart: true });
}

/* ----------------------------------------------------------
   Digital Twin – init handler
   ---------------------------------------------------------- */

function handleTwinInit(msg) {
  handleInit(msg);
  showTwinPanels(true);
  state.twinRunId = msg.run_id || null;
  state.twinSnapshotId =
    msg.snapshot && msg.snapshot.snapshot_id ? msg.snapshot.snapshot_id : null;
  state.twinConfigVersion = msg.config_version || null;
  state.twinSourceStatus = msg.source_status || null;
  renderTwinSourceStatus(msg.source_status || {});
  renderTwinSignals(msg.source_status || {});
  renderTwinPredictionsPanel(msg.snapshot, msg.source_status || {});
  renderTwinDivergence({});
  updateRuntimeDashboard(msg);
  if (window.DashboardUI && window.DashboardUI.resetForecastTickSeries) {
    window.DashboardUI.resetForecastTickSeries();
  }
  initTwinForecastPanel(msg.forecast_panel || {});
  if (msg.forecast_job_id) {
    state.twinForecastJobId = msg.forecast_job_id;
  }
  if (msg.forecast_live) {
    applyTwinForecastPanel(msg.forecast_live, { resetSeries: true, appendChart: true });
  }

  enableInfiniteProgressMode();
  var progressText = $("progress-text");
  if (progressText) progressText.textContent = "∞ Continuous";

  // Initialize world state
  if (msg.world) {
    state.worldState = msg.world;
    updateWorldStatePanel(msg.world);
  }

  // Sync in-game rate slider
  var ingameRate = document.getElementById("ingame-random-rate");
  var ingameRateVal = document.getElementById("ingame-rate-value");
  var setupRate = document.getElementById("random-event-rate");
  if (ingameRate && setupRate) {
    ingameRate.value = setupRate.value;
    if (ingameRateVal)
      ingameRateVal.textContent = parseFloat(setupRate.value).toFixed(2);
  }
}

/* ----------------------------------------------------------
   Digital Twin – tick handler
   ---------------------------------------------------------- */

function handleTwinTick(msg) {
  handleStep(msg);
  state.twinRunId = msg.run_id || state.twinRunId;
  if (msg.source_status) {
    state.twinSourceStatus = msg.source_status;
    state.twinSnapshotId = msg.source_status.snapshot_id || state.twinSnapshotId;
    renderTwinSourceStatus(msg.source_status);
    renderTwinSignals(msg.source_status);
    updateRuntimeSourceStatus(msg.source_status);
  }
  renderTwinPredictionsPanel(msg.snapshot, msg.source_status);
  renderTwinDivergence(msg.divergence || {});
  updateRuntimeDashboard(msg);
  if (msg.forecast_panel) {
    applyTwinForecastPanel(msg.forecast_panel, { appendChart: true });
  }

  // Cycle info
  var cycleNum = $("twin-cycle-number");
  if (cycleNum) cycleNum.textContent = "#" + (msg.cycle_number || 1);

  var cycleHour = $("twin-cycle-hour");
  var budgetCycleHours = 168;
  if (state.initData && state.initData.budget_cycle_hours) {
    budgetCycleHours = state.initData.budget_cycle_hours;
  }
  if (cycleHour)
    cycleHour.textContent =
      Math.round(msg.cycle_hour || 0) + " / " + budgetCycleHours;

  var cycleBar = $("twin-cycle-bar");
  if (cycleBar) cycleBar.style.width = (msg.budget_cycle_progress || 0) + "%";

  // Active events
  if (msg.active_events && msg.active_events.length > 0) {
    renderTwinEvents(msg.active_events);
  } else if (!state.twinEvents.length) {
    renderTwinEvents([]);
  }

  // Data sources
  if (msg.data_sources_status) {
    updateDataSources(msg.data_sources_status);
  }

  // Override progress text for twin
  var progressText = $("progress-text");
  if (progressText) {
    progressText.textContent =
      "Cycle #" + (msg.cycle_number || 1) + " · Hour " + Math.round(msg.hour);
  }

  // World state
  if (msg.world) {
    state.worldState = msg.world;
    updateWorldStatePanel(msg.world);
  }
  // Narratives
  if (msg.narrative_updates) {
    renderNarratives(msg.narrative_updates);
  }
}

/* ----------------------------------------------------------
   Digital Twin – event / budget / action / data handlers
   ---------------------------------------------------------- */

function handleTwinEvent(msg) {
  var ev = msg.event || {};
  var severity = normalizeTwinSeverity(ev.severity);
  var eventKey = ev.event_key || "unknown";
  var source = ev.source || "scheduled";
  var firedAt = ev.fired_at_hour != null ? ev.fired_at_hour : msg.hour || 0;

  var title =
    ev.name ||
    eventKey.replace(/_/g, " ").replace(/\b\w/g, function (c) {
      return c.toUpperCase();
    });
  var description =
    ev.description ||
    source +
      " event at hour " +
      Math.round(firedAt) +
      " (severity " +
      Math.round(severity * 100) +
      "%)";

  var icon =
    severity >= 0.6
      ? "fa-bolt"
      : severity >= 0.3
        ? "fa-exclamation-triangle"
        : "fa-info-circle";
  var type = severity >= 0.6 ? "danger" : severity >= 0.3 ? "warning" : "info";

  showToast({
    title: title,
    description: description,
    severity: severity,
    icon: icon,
    type: type,
  });

  // Store in twin events list
  state.twinEvents.unshift({
    key: eventKey,
    name: ev.name || title,
    description: ev.description || description,
    severity: severity,
    source: source,
    hour: firedAt,
    title: title,
  });

  // Keep max 50 events
  if (state.twinEvents.length > 50) {
    state.twinEvents = state.twinEvents.slice(0, 50);
  }
  renderTwinEvents(state.twinEvents);
}

function handleBudgetRefresh(msg) {
  var budget = msg.budget || msg;
  var newBudget = budget.new_budget || budget.budget_remaining;
  var cycle = budget.cycle_number || state.twinCycleNumber;

  state.twinCycleNumber = cycle;

  if (newBudget != null) {
    setHudValue("hud-budget", "$" + formatNumber(newBudget));
  }

  showToast({
    title: "Budget Refresh",
    description:
      "Cycle #" +
      cycle +
      " — Budget replenished to $" +
      formatNumber(newBudget || 0),
    severity: 0.2,
    icon: "fa-coins",
    type: "success",
    duration: 4000,
  });
}

function handleActionResult(msg) {
  var result = msg.result || {};
  var actionKey = result.action_key || "unknown";
  var success = result.success !== false;
  var cost = result.cost || 0;
  var message =
    result.message || (success ? "Action executed" : "Action failed");

  var title = actionKey.replace(/_/g, " ").replace(/\b\w/g, function (c) {
    return c.toUpperCase();
  });

  showToast({
    title: title,
    description:
      message + (cost > 0 ? " (cost: $" + formatNumber(cost) + ")" : ""),
    severity: success ? 0.2 : 0.6,
    icon: success ? "fa-check-circle" : "fa-times-circle",
    type: success ? "success" : "danger",
    duration: 4000,
  });

  if (success && result.action_id) {
    state.twinLastActionId = result.action_id;
  }
  renderTwinActionResult(result);
}

function handleBranchResult(msg) {
  var branch = msg.branch || {};
  var panel = $("twin-recommendation-panel");
  if (!panel) return;
  if (!branch.branch_id) {
    panel.textContent = "Branch could not be created.";
    return;
  }
  var metrics = branch.summary_metrics || {};
  panel.innerHTML =
    "<strong>Branch saved</strong><br>" +
    escapeHtml(branch.branch_key || branch.branch_id) +
    "<br>Projected pressure: " +
    escapeHtml(metrics.projected_shortage_pressure != null ? metrics.projected_shortage_pressure : "—");
}

function handleRecommendationResult(msg) {
  var rec = msg.recommendation || {};
  renderTwinRecommendation(rec);
}

function handlePromotionResult(msg) {
  var action = msg.action || {};
  var panel = $("twin-action-result");
  if (!panel) return;
  panel.innerHTML =
    "<strong>Promotion</strong><br>" +
    escapeHtml(action.action_key || action.action_id || "Action") +
    "<br>Status: " +
    escapeHtml(action.promotion_status || "unknown");
}

function handleDataStatus(msg) {
  var status = msg.data_sources_status || {
    weather: msg.weather,
    events: msg.events,
  };
  state.twinSourceStatus = status;
  if (msg.snapshot && msg.snapshot.snapshot_id) {
    state.twinSnapshotId = msg.snapshot.snapshot_id;
  }
  renderTwinSourceStatus(status);
  renderTwinSignals(status);
  renderTwinPredictionsPanel(msg.snapshot, status);
  updateRuntimeSourceStatus(status);
  updateDataSources(status);
}

function twinStatusLabel(status) {
  return String(status || "degraded").replace(/_/g, " ");
}

function renderTwinSourceStatus(status) {
  var badge = $("twin-source-badge");
  var source = status || {};
  var sourceStatus = source.status || "degraded";
  if (badge) {
    badge.className = "twin-source-badge " + sourceStatus;
    badge.textContent = twinStatusLabel(sourceStatus);
  }

  var list = $("twin-run-status");
  if (!list) return;
  list.innerHTML =
    '<div class="twin-mini-row"><span>Run</span><strong>' +
    escapeHtml(state.twinRunId || "—") +
    "</strong></div>" +
    '<div class="twin-mini-row"><span>Snapshot</span><strong>' +
    escapeHtml(source.snapshot_id || state.twinSnapshotId || "—") +
    "</strong></div>" +
    '<div class="twin-mini-row"><span>Config</span><strong>' +
    escapeHtml(state.twinConfigVersion || (state.twinConfig && state.twinConfig.version) || "—") +
    "</strong></div>" +
    '<div class="twin-mini-row"><span>Source</span><strong>' +
    escapeHtml(source.source_mode || "—") +
    "</strong></div>";
}

function renderTwinPredictionsPanel(snapshot, sourceStatus) {
  var emptyEl = $("twin-predictions-empty");
  var modelList = $("twin-model-config-list");
  var tableWrap = $("twin-predictions-table-wrap");
  var tbody = $("twin-predictions-tbody");
  var warningsEl = $("twin-prediction-warnings");
  if (!emptyEl || !tbody) return;

  var payload = (snapshot && snapshot.payload) || {};
  var predictions = payload.predictions || [];
  var modelConfigs = payload.model_configs || [];
  var warnings = (snapshot && snapshot.warnings) || (sourceStatus && sourceStatus.warnings) || [];

  if (!predictions.length && !modelConfigs.length) {
    emptyEl.hidden = false;
    emptyEl.textContent =
      warnings.length > 0
        ? "ML predictions unavailable. See warning below."
        : "No predictions yet. Use Digital Twin mode and Refresh snapshot, or check PIOS_ML_CONFIG_ROOT / model artifacts.";
    if (modelList) modelList.hidden = true;
    if (tableWrap) tableWrap.hidden = true;
  } else {
    emptyEl.hidden = true;
    if (tableWrap) tableWrap.hidden = false;
    var rowsHtml = "";
    var limit = Math.min(predictions.length, 12);
    for (var i = 0; i < limit; i++) {
      var row = predictions[i] || {};
      rowsHtml +=
        "<tr><td>" +
        escapeHtml(row.model_name || "—") +
        "</td><td>" +
        escapeHtml(row.entity_id || row.hospital_id || "—") +
        "</td><td>" +
        escapeHtml(row.predicted_value != null ? row.predicted_value : "—") +
        "</td></tr>";
    }
    tbody.innerHTML = rowsHtml;
    if (modelList) {
      modelList.hidden = false;
      var cfgHtml = "";
      var cfgLimit = Math.min(modelConfigs.length, 8);
      for (var j = 0; j < cfgLimit; j++) {
        var cfg = modelConfigs[j] || {};
        cfgHtml +=
          '<div class="twin-mini-row"><span>' +
          escapeHtml(cfg.model_id || "model") +
          "</span><strong>" +
          escapeHtml(cfg.is_active ? "active" : "inactive") +
          "</strong></div>";
      }
      if (modelConfigs.length > cfgLimit) {
        cfgHtml +=
          '<div class="twin-mini-row"><span>…</span><strong>' +
          (modelConfigs.length - cfgLimit) +
          " more</strong></div>";
      }
      modelList.innerHTML = cfgHtml;
    }
  }

  if (warningsEl) {
    if (warnings.length) {
      warningsEl.hidden = false;
      warningsEl.innerHTML =
        '<div class="twin-prediction-warn-title">ML status</div><ul class="twin-prediction-warn-list"><li>' +
        warnings.map(function (w) {
          return escapeHtml(String(w));
        }).join("</li><li>") +
        "</li></ul>";
    } else {
      warningsEl.hidden = true;
      warningsEl.innerHTML = "";
    }
  }
}

function renderTwinSignals(status) {
  var list = $("twin-signal-list");
  if (!list) return;
  var signals = (status && status.real_signals) || {};
  var rows = [
    ["Alerts", signals.alerts],
    ["Predictions", signals.predictions],
    ["Drift Reports", signals.drift_reports],
    ["Model Configs", signals.model_configs],
    ["Inventory Rows", signals.blood_supplies],
    ["Donors", signals.donors],
  ];
  var html = "";
  for (var i = 0; i < rows.length; i++) {
    html +=
      '<div class="twin-mini-row"><span>' +
      escapeHtml(rows[i][0]) +
      "</span><strong>" +
      escapeHtml(rows[i][1] != null ? rows[i][1] : "—") +
      "</strong></div>";
  }
  list.innerHTML = html;
}

function renderTwinDivergence(divergence) {
  var list = $("twin-divergence-list");
  if (!list) return;
  if (!divergence || Object.keys(divergence).length === 0) {
    list.innerHTML = '<div class="empty-state">Waiting for first frame…</div>';
    return;
  }
  var rows = [
    ["Status", divergence.status],
    ["Inventory Δ", divergence.inventory_delta_units],
    ["Inventory Δ %", divergence.inventory_delta_pct],
    ["Donor Δ", divergence.donor_count_delta],
    ["Alert Δ", divergence.alert_count_delta],
  ];
  var html = "";
  for (var i = 0; i < rows.length; i++) {
    html +=
      '<div class="twin-mini-row"><span>' +
      escapeHtml(rows[i][0]) +
      "</span><strong>" +
      escapeHtml(rows[i][1] != null ? rows[i][1] : "—") +
      "</strong></div>";
  }
  list.innerHTML = html;
}

function renderTwinActionResult(result) {
  var panel = $("twin-action-result");
  if (!panel) return;
  var html =
    "<strong>" +
    escapeHtml(result.name || result.action_key || "Action") +
    "</strong><br>" +
    escapeHtml(result.message || "Simulated") +
    "<br>Status: " +
    escapeHtml(result.promotion_status || "simulated");
  if (result.action_id && result.promotion_status === "pending") {
    html +=
      '<br><button id="twin-promote-btn" type="button" data-action-id="' +
      escapeHtml(result.action_id) +
      '">Promote</button>';
  }
  panel.innerHTML = html;
}

function renderTwinRecommendation(rec) {
  var panel = $("twin-recommendation-panel");
  if (!panel) return;
  if (!rec || !rec.recommendation_id) {
    panel.textContent = "No recommendation returned.";
    return;
  }
  panel.innerHTML =
    "<strong>" +
    escapeHtml(rec.recommendation_key || "Recommendation") +
    "</strong><br>" +
    escapeHtml(rec.rationale || "") +
    '<br><button id="twin-simulate-recommendation-btn" type="button" data-action-key="' +
    escapeHtml(rec.recommendation_key || "") +
    '">Simulate</button>';
}

/* ----------------------------------------------------------
   Twin UI helpers
   ---------------------------------------------------------- */

function showTwinPanels(visible) {
  var panels = document.querySelectorAll(".twin-panel");
  for (var i = 0; i < panels.length; i++) {
    if (visible) {
      panels[i].classList.add("visible");
    } else {
      panels[i].classList.remove("visible");
    }
  }
  document.body.classList.toggle("twin-active", !!visible);
  if (!visible) {
    document.body.classList.remove("twin-details-open");
  }
}

function updateWorldStatePanel(world) {
  if (!world) return;

  var seasonIcons = { winter: "❄️", spring: "🌸", summer: "☀️", fall: "🍂" };
  var renderer = state.sceneRenderer || ensureSceneRenderer();

  if (renderer && typeof renderer.setWorldContext === "function") {
    renderer.setWorldContext(world);
  }

  // ── Update top-bar compact world info ──────────────────────
  var seasonCapitalized =
    world.season.charAt(0).toUpperCase() + world.season.slice(1);
  var compactSeasonIcon = document.getElementById("hud-world-season-icon");
  var compactSeasonText = document.getElementById("hud-world-season-text");
  var compactDateText = document.getElementById("hud-world-date-text");
  var compactTempText = document.getElementById("hud-world-temp-text");
  var compactWeatherIcon = document.getElementById("hud-world-weather-icon");
  var compactWeatherText = document.getElementById("hud-world-weather-text");

  if (compactSeasonIcon)
    compactSeasonIcon.textContent = seasonIcons[world.season] || "🌍";
  if (compactSeasonText) compactSeasonText.textContent = seasonCapitalized;
  if (compactDateText) {
    var d = world.date_display || world.date || "";
    var shortDate = d;
    var dateParts = d.match(/^(\w+)\s+(\d+)/);
    if (dateParts) {
      shortDate = dateParts[1].substring(0, 3) + " " + dateParts[2];
    }
    compactDateText.textContent = shortDate;
  }
  if (compactTempText) compactTempText.textContent = world.temperature_c + "°C";

  var weatherIcons = {
    clear_skies: "☀️",
    overcast: "☁️",
    light_snow: "🌨️",
    heavy_snow: "❄️",
    freezing_rain: "🧊",
    blizzard: "🌪️",
    heat_wave: "🔥",
    spring_thaw: "💧",
  };

  if (compactWeatherIcon)
    compactWeatherIcon.textContent =
      weatherIcons[world.weather_pattern] || "🌤️";
  if (compactWeatherText) {
    var wpShort = (world.weather_pattern || "").replace(/_/g, " ");
    wpShort = wpShort.charAt(0).toUpperCase() + wpShort.slice(1);
    if (wpShort.length > 12) wpShort = wpShort.substring(0, 10) + "…";
    compactWeatherText.textContent = wpShort;
  }

  // ── Update dropdown detail panel ──────────────────────
  var wddSeasonIcon = document.getElementById("wdd-season-icon");
  var wddSeason = document.getElementById("wdd-season");
  var wddDate = document.getElementById("wdd-date");
  var wddTemp = document.getElementById("wdd-temp");
  var wddWeatherIcon = document.getElementById("wdd-weather-icon");
  var wddWeather = document.getElementById("wdd-weather");

  if (wddSeasonIcon)
    wddSeasonIcon.textContent = seasonIcons[world.season] || "🌍";
  if (wddSeason) wddSeason.textContent = seasonCapitalized;
  if (wddDate) wddDate.textContent = world.date_display || world.date;
  if (wddTemp) wddTemp.textContent = world.temperature_c + "°C";
  if (wddWeatherIcon)
    wddWeatherIcon.textContent = weatherIcons[world.weather_pattern] || "🌤️";
  if (wddWeather) {
    var wpFull = (world.weather_pattern || "").replace(/_/g, " ");
    wddWeather.textContent = wpFull.charAt(0).toUpperCase() + wpFull.slice(1);
  }

  // Dropdown meters
  var wddRoads = document.getElementById("wdd-roads-fill");
  if (wddRoads) wddRoads.style.width = world.road_conditions * 100 + "%";
  var wddMood = document.getElementById("wdd-mood-fill");
  if (wddMood) wddMood.style.width = world.population_mood * 100 + "%";
  var wddDiff = document.getElementById("wdd-difficulty-fill");
  if (wddDiff) wddDiff.style.width = world.difficulty * 100 + "%";

  // Dropdown crisis badge
  var wddCrisis = document.getElementById("wdd-crisis-badge");
  var wddCrisisLabel = document.getElementById("wdd-crisis-label");
  if (wddCrisis && world.crisis_label) {
    var crisisColors = {
      Calm: {
        bg: "rgba(34,197,94,0.1)",
        fg: "#86efac",
        border: "rgba(34,197,94,0.2)",
        icon: "fa-shield",
      },
      Elevated: {
        bg: "rgba(245,158,11,0.1)",
        fg: "#fcd34d",
        border: "rgba(245,158,11,0.2)",
        icon: "fa-exclamation-triangle",
      },
      Crisis: {
        bg: "rgba(239,68,68,0.1)",
        fg: "#fca5a5",
        border: "rgba(239,68,68,0.2)",
        icon: "fa-bolt",
      },
      Emergency: {
        bg: "rgba(220,38,38,0.15)",
        fg: "#ef4444",
        border: "rgba(220,38,38,0.3)",
        icon: "fa-skull-crossbones",
      },
    };
    var cc = crisisColors[world.crisis_label] || crisisColors.Calm;
    wddCrisis.style.background = cc.bg;
    wddCrisis.style.color = cc.fg;
    wddCrisis.style.borderColor = cc.border;
    var ci = wddCrisis.querySelector("i");
    if (ci) ci.className = "fa-solid " + cc.icon;
  }
  if (wddCrisisLabel) wddCrisisLabel.textContent = world.crisis_label || "Calm";

  // Dropdown holiday
  var wddHoliday = document.getElementById("wdd-holiday");
  var wddHolidayName = document.getElementById("wdd-holiday-name");
  if (wddHoliday) {
    if (world.is_holiday && world.holiday_name) {
      wddHoliday.style.display = "inline-flex";
      if (wddHolidayName) wddHolidayName.textContent = world.holiday_name;
    } else {
      wddHoliday.style.display = "none";
    }
  }
}

function renderNarratives(narratives) {
  var list = $("narrative-list");
  if (!list) return;

  if (!narratives || narratives.length === 0) {
    list.innerHTML =
      '<div class="narrative-empty">No active narratives yet...</div>';
    return;
  }

  var html = "";
  for (var i = 0; i < narratives.length; i++) {
    var n = narratives[i];
    var name = escapeHtml(n.name || n.arc_key || "Unknown");
    var stage = n.current_stage || 0;
    var totalStages = n.total_stages || 3;
    var severity = n.severity || "low";
    var desc = escapeHtml(n.description || "");

    html += '<div class="narrative-card">';
    html += '<div class="narrative-card-header">';
    html += '<span class="narrative-card-name">' + name + "</span>";
    html +=
      '<span class="narrative-card-severity severity-' +
      severity +
      '">' +
      severity +
      "</span>";
    html += "</div>";
    if (desc) {
      html += '<div class="narrative-card-desc">' + desc + "</div>";
    }
    html +=
      '<div class="narrative-card-stage">Stage ' +
      stage +
      " / " +
      totalStages +
      "</div>";
    html += '<div class="narrative-stage-dots">';
    for (var s = 1; s <= totalStages; s++) {
      var dotClass = "narrative-stage-dot";
      if (s < stage) dotClass += " completed";
      else if (s === stage) dotClass += " active";
      html += '<span class="' + dotClass + '"></span>';
    }
    html += "</div>";
    html += "</div>";
  }

  list.innerHTML = html;
}

function renderTwinEvents(events) {
  var list = $("twin-event-list");
  if (!list) return;

  if (!events || events.length === 0) {
    list.innerHTML = '<div class="empty-state">No active events</div>';
    return;
  }

  var html = "";
  for (var i = 0; i < events.length; i++) {
    var ev = events[i];
    var key = ev.event_key || ev.key || "unknown";
    var name =
      ev.name ||
      ev.title ||
      key.replace(/_/g, " ").replace(/\b\w/g, function (c) {
        return c.toUpperCase();
      });
    var severity = normalizeTwinSeverity(ev.severity);
    var source = ev.source || "scheduled";
    var hour = ev.fired_at_hour != null ? ev.fired_at_hour : 0;

    var priorityClass = "priority-low";
    if (severity >= 0.7) priorityClass = "priority-critical";
    else if (severity >= 0.5) priorityClass = "priority-high";
    else if (severity >= 0.3) priorityClass = "priority-medium";

    var icon =
      severity >= 0.6
        ? "fa-bolt"
        : severity >= 0.3
          ? "fa-exclamation-triangle"
          : "fa-info-circle";

    html += '<div class="twin-event-item">';
    html +=
      '<span class="tei-icon ' +
      priorityClass +
      '"><i class="fa-solid ' +
      icon +
      '"></i></span>';
    html += '<div class="tei-body">';
    html += '<div class="tei-name">' + escapeHtml(name) + "</div>";
    html +=
      '<div class="tei-meta">' +
      source +
      " · H" +
      Math.round(hour) +
      " · " +
      Math.round(severity * 100) +
      "%</div>";
    html += "</div>";
    html += "</div>";
  }

  list.innerHTML = html;
}

function updateDataSources(status) {
  if (!status) return;
  var weatherDot = $("ds-weather-dot");
  var eventsDot = $("ds-events-dot");
  var weatherConnected = !!(status.weather && status.weather.connected);
  var eventsConnected = !!(
    status.events &&
    !Array.isArray(status.events) &&
    status.events.connected
  );
  if (weatherDot) {
    weatherDot.classList.toggle("online", weatherConnected);
    weatherDot.classList.toggle("offline", !weatherConnected);
  }
  if (eventsDot) {
    eventsDot.classList.toggle("online", eventsConnected);
    eventsDot.classList.toggle("offline", !eventsConnected);
  }
}

/* ----------------------------------------------------------
   Pause button toggle
   ---------------------------------------------------------- */

function setPauseButtonState(btn, paused) {
  if (!btn) return;
  var icon = btn.querySelector("i");
  var label = btn.querySelector("span");
  if (paused) {
    if (icon) icon.className = "fa-solid fa-play";
    if (label) label.textContent = "Resume";
    btn.setAttribute("data-state", "paused");
  } else {
    if (icon) icon.className = "fa-solid fa-pause";
    if (label) label.textContent = "Pause";
    btn.setAttribute("data-state", "running");
  }
}

function updatePauseButton(paused) {
  setPauseButtonState($("pause-btn"), paused);
  setPauseButtonState($("runtime-pause-btn"), paused);
  setPauseButtonState($("toolbar-pause-btn"), paused);
}

/* ----------------------------------------------------------
   Dashboard runtime clock + speed (digital-twin header bar)
   ---------------------------------------------------------- */

function formatRuntimeClockText(hour, totalHours) {
  var h = Number(hour) || 0;
  var hourOfDay = Math.floor(((h % 24) + 24) % 24);
  var timeText =
    "Hour " + Math.round(h) +
    (totalHours ? "/" + Math.round(totalHours) : "") +
    "  ·  " + String(hourOfDay).padStart(2, "0") + ":00";
  var dayText = "Day " + (Math.floor(h / 24) + 1);
  return { timeText: timeText, dayText: dayText };
}

function updateRuntimeClock(hour, totalHours) {
  var parts = formatRuntimeClockText(hour, totalHours);
  var pairs = [
    ["runtime-clock-time", "runtime-clock-day"],
    ["toolbar-clock-time", "toolbar-clock-day"],
  ];
  for (var i = 0; i < pairs.length; i++) {
    var hourEl = $(pairs[i][0]);
    var dayEl = $(pairs[i][1]);
    if (hourEl) hourEl.textContent = parts.timeText;
    if (dayEl) dayEl.textContent = parts.dayText;
  }
}

function highlightRuntimeSpeed(speed) {
  var groups = [
    ["rt-speed-1x", "rt-speed-2x", "rt-speed-5x", "rt-speed-10x"],
    ["toolbar-speed-1x", "toolbar-speed-2x", "toolbar-speed-5x", "toolbar-speed-10x"],
  ];
  var vals = [1, 2, 5, 10];
  for (var g = 0; g < groups.length; g++) {
    for (var i = 0; i < groups[g].length; i++) {
      var btn = $(groups[g][i]);
      if (btn) btn.classList.toggle("active", vals[i] === speed);
    }
  }
}

function setSimRunning(running) {
  document.body.classList.toggle("sim-running", !!running);
}

function updateProgress(percent, hour, totalHours) {
  var bar = $("progress-bar");
  var text = $("progress-text");
  var runtimeBar = $("runtime-progress-bar");
  var runtimeText = $("runtime-progress-text");
  var pct = clamp(percent || 0, 0, 100);
  var label =
    "Hour " +
    Math.round(hour || 0) +
    " / " +
    (totalHours || "?") +
    "  (" +
    pct.toFixed(1) +
    "%)";
  if (bar) bar.style.width = pct + "%";
  if (text) text.textContent = label;
  if (runtimeBar) {
    runtimeBar.style.width = pct + "%";
    runtimeBar.setAttribute("aria-valuenow", String(Math.round(pct)));
  }
  if (runtimeText) runtimeText.textContent = label;
}

/* ----------------------------------------------------------
   Weather icon in HUD
   ---------------------------------------------------------- */

function updateWeatherIcon(weather) {
  var iconEl = $("hud-weather-icon");
  if (!iconEl) return;
  iconEl.className = getWeatherIconClass(weather) + " stat-icon";
}

/* ----------------------------------------------------------
   Actions list (power-up cards)
   ---------------------------------------------------------- */

function clearActionsList() {
  var el = $("actions-list");
  if (el) el.innerHTML = '<div class="empty-state">No active actions</div>';
}

function actionDisplayName(key, fallbackName) {
  if (fallbackName) return fallbackName;
  var catalog = state.actionCatalog[key];
  if (catalog && catalog.name) return catalog.name;
  return key || "Action";
}

function strategyExpectsRlController(msg) {
  var strategy = msg && msg.strategy;
  if (!strategy || typeof strategy !== "object") {
    strategy = state.initData && state.initData.strategy;
  }
  if (!strategy || typeof strategy !== "object") return false;
  return !!strategy.controller_key;
}

function updateControllerPanelVisibility(msg) {
  var expects = strategyExpectsRlController(msg);
  var runtimeController =
    (msg && msg.runtime_controller) ||
    (state.initData && state.initData.runtime_controller);
  var show = expects || !!runtimeController;

  var aiPanel = $("ai-panel");
  if (aiPanel) {
    aiPanel.style.display = show ? "" : "none";
  }

  var dtCard = $("dt-section-controller");
  if (dtCard) {
    if (show) {
      dtCard.removeAttribute("hidden");
    } else {
      dtCard.setAttribute("hidden", "hidden");
    }
  }

  if (show && expects && !runtimeController) {
    renderControllerDecision({
      headline:
        "RL strategy selected but controller did not start (checkpoint missing or load error).",
      controller: (msg && msg.strategy && msg.strategy.controller_key) || "",
      hour: null,
      top_levels: [],
      has_changes: false,
    });
  }
}

function renderControllerDecision(decision) {
  var headline = decision
    ? decision.headline || "Controller update"
    : "Waiting for first step…";
  var metaText = "";
  if (decision) {
    var controllerLabel = decision.controller || "controller";
    var hourLabel =
      decision.hour != null ? " · Hour " + decision.hour : "";
    var stepHours =
      state.initData && state.initData.controller_step_hours != null
        ? " · decides every " + state.initData.controller_step_hours + "h"
        : "";
    metaText = controllerLabel + hourLabel + stepHours;
  }

  var chipHtml = "";
  if (decision && decision.top_levels && decision.top_levels.length) {
    for (var t = 0; t < decision.top_levels.length; t++) {
      var row = decision.top_levels[t];
      var pct = clamp(Math.abs(row.level || 0) * 100, 0, 100);
      chipHtml +=
        '<span class="controller-chip">' +
        escapeHtml(row.name || row.key) +
        " " +
        pct.toFixed(0) +
        "%</span>";
    }
  }

  var statusEl = $("ai-status");
  var metaEl = $("controller-meta");
  var chipsEl = $("controller-top-levels");
  if (statusEl) statusEl.textContent = headline;
  if (metaEl) metaEl.textContent = metaText;
  if (chipsEl) chipsEl.innerHTML = chipHtml;

  var dtHeadline = $("dt-controller-headline");
  var dtMeta = $("dt-controller-meta");
  var dtChips = $("dt-controller-chips");
  if (dtHeadline) dtHeadline.textContent = headline;
  if (dtMeta) dtMeta.textContent = metaText;
  if (dtChips) dtChips.innerHTML = chipHtml;
}

function renderControllerActiveActions(actions) {
  var el = $("dt-controller-active-actions");
  if (!el) return;
  if (!actions || !actions.length) {
    el.innerHTML =
      '<div class="dt-forecast-empty">No active interventions right now</div>';
    return;
  }
  var html = "";
  for (var i = 0; i < Math.min(actions.length, 6); i++) {
    var a = actions[i];
    var key = a.key || a.action_key || "";
    var pct = clamp((a.intensity || 0) * 100, 0, 100);
    html +=
      '<div class="dt-controller-action-row"><span>' +
      escapeHtml(actionDisplayName(key, a.name)) +
      '</span><span>' +
      pct.toFixed(0) +
      "%</span></div>";
  }
  el.innerHTML = html;
}

function renderActionsList(actions) {
  var el = $("actions-list");
  if (!el) return;

  if (!actions || actions.length === 0) {
    el.innerHTML = '<div class="empty-state">No active actions</div>';
    state.previousActiveActionsByKey = {};
    return;
  }

  var prev = state.previousActiveActionsByKey || {};
  var nextMap = {};
  var html = "";
  for (var i = 0; i < actions.length; i++) {
    var a = actions[i];
    var key = a.key || a.action_key || a.name || "action-" + i;
    var intensity = clamp((a.intensity || 0) * 100, 0, 100);
    var prevIntensity = prev[key];
    var changed =
      prevIntensity == null ||
      Math.abs(prevIntensity - (a.intensity || 0)) > 0.001;
    nextMap[key] = a.intensity || 0;
    html +=
      '<div class="action-card' +
      (changed ? " action-card-changed" : "") +
      '">' +
      '<div class="action-name"><i class="fa-solid fa-bolt"></i> ' +
      escapeHtml(actionDisplayName(key, a.name)) +
      "</div>" +
      '<div class="action-bar-track">' +
      '<div class="action-bar-fill" style="width:' +
      intensity +
      '%"></div>' +
      "</div>" +
      '<span class="action-pct">' +
      intensity.toFixed(0) +
      "% intensity</span>" +
      "</div>";
  }
  state.previousActiveActionsByKey = nextMap;
  el.innerHTML = html;
}

function addControllerEventEntry(stepMsg) {
  var decision = stepMsg.controller_decision;
  if (!decision || !decision.has_changes) return;

  var el = $("events-feed");
  if (!el) return;

  var entry = document.createElement("div");
  entry.className = "event-item event-neutral";
  entry.innerHTML =
    '<span class="event-icon"><i class="fa-solid fa-robot"></i></span>' +
    '<span class="event-time">H' +
    Math.round(stepMsg.hour || decision.hour || 0) +
    "</span> " +
    '<span class="event-text">' +
    escapeHtml(decision.headline || "Controller update") +
    "</span>";

  el.insertBefore(entry, el.firstChild);
  while (el.children.length > 30) {
    el.removeChild(el.lastChild);
  }
  el.scrollTop = 0;
}

/* ----------------------------------------------------------
   Events feed
   ---------------------------------------------------------- */

function clearEventsFeed() {
  var el = $("events-feed");
  if (el) el.innerHTML = "";
}

function addEventEntry(stepMsg) {
  var el = $("events-feed");
  if (!el) return;

  var donations =
    stepMsg.recent_donations != null
      ? stepMsg.recent_donations
      : stepMsg.total_donated || 0;
  var shortages =
    stepMsg.recent_shortages != null
      ? stepMsg.recent_shortages
      : stepMsg.total_shortage || 0;

  // Determine event type for icon
  var eventClass = "event-neutral";
  var eventIcon = "fa-solid fa-circle-info";
  if (shortages > 0) {
    eventClass = "event-negative";
    eventIcon = "fa-solid fa-triangle-exclamation";
  } else if (donations > 0) {
    eventClass = "event-positive";
    eventIcon = "fa-solid fa-heart";
  }

  // Weather event
  var weatherNote = "";
  if (stepMsg.weather && stepMsg.weather !== "clear") {
    weatherNote =
      ' <span style="color:var(--accent-cyan)">[' +
      (weatherLabels[stepMsg.weather] || stepMsg.weather) +
      "]</span>";
  }

  var entry = document.createElement("div");
  entry.className = "event-item " + eventClass;
  entry.innerHTML =
    '<span class="event-icon"><i class="' +
    eventIcon +
    '"></i></span>' +
    '<span class="event-time">H' +
    Math.round(stepMsg.hour || 0) +
    "</span> " +
    '<span class="event-text">+' +
    donations +
    " donated, " +
    shortages +
    " shortages" +
    weatherNote +
    "</span>";

  el.insertBefore(entry, el.firstChild);

  while (el.children.length > 30) {
    el.removeChild(el.lastChild);
  }

  el.scrollTop = 0;
}

/* ----------------------------------------------------------
   MAP BOARD — Build center nodes
   ---------------------------------------------------------- */

function buildMapBoard(centers) {
  var renderer = ensureSceneRenderer();
  if (!renderer) return;
  renderer.setMissionContext(
    state.initData && state.initData.scenario,
    state.initData && state.initData.strategy,
    !!state.twinMode,
  );
  renderer.setOperationalFrame(state.lastStepData || state.initData || {});
  renderer.setCenters(centers || [], {
    donorCountMap: state.donorCountMap,
    donatedDiff: 0,
    transfusedDiff: 0,
  });
  renderer.resize();
}

function getCenterIconClass(type) {
  if (type === "hospital") return "icon-hospital";
  if (type === "mobile") return "icon-mobile";
  return "icon-blood_bank";
}

function getCenterIconFA(type) {
  if (type === "hospital") return "fa-hospital";
  if (type === "mobile") return "fa-truck-medical";
  return "fa-droplet";
}

function shortenName(name) {
  if (!name) return "Center";
  // Shorten common prefixes
  name = name.replace("Héma-Québec ", "HQ ");
  name = name.replace("CHU de Québec - ", "");
  if (name.length > 20) {
    return name.substring(0, 18) + "…";
  }
  return name;
}

function buildTooltipHTML(center) {
  var inv = center.inventory || {};
  var stats = center.stats || {};
  var typeLabel = escapeHtml(center.type || "");
  if (center.role && center.role !== center.type) {
    typeLabel += " / " + escapeHtml(center.role);
  }
  return (
    '<div class="center-tooltip">' +
    '<div class="tooltip-title">' +
    escapeHtml(center.name || "Center") +
    " (" +
    typeLabel +
    ")</div>" +
    '<div class="tooltip-row"><span>RBC</span><span class="tv" style="color:var(--rbc-color)">' +
    (inv.RBC || 0) +
    "</span></div>" +
    '<div class="tooltip-row"><span>Platelets</span><span class="tv" style="color:var(--plt-color)">' +
    (inv.PLATELETS || 0) +
    "</span></div>" +
    '<div class="tooltip-row"><span>Plasma</span><span class="tv" style="color:var(--pls-color)">' +
    (inv.PLASMA || 0) +
    "</span></div>" +
    '<div class="tooltip-row"><span>Total Units</span><span class="tv">' +
    (center.total_units || 0) +
    "</span></div>" +
    '<div class="tooltip-row"><span>Donated</span><span class="tv" style="color:var(--accent-emerald)">' +
    (stats.donated || 0) +
    "</span></div>" +
    '<div class="tooltip-row"><span>Transfused</span><span class="tv" style="color:var(--accent-purple)">' +
    (stats.transfused || 0) +
    "</span></div>" +
    '<div class="tooltip-row"><span>Expired</span><span class="tv" style="color:var(--text-muted)">' +
    (stats.expired || 0) +
    "</span></div>" +
    '<div class="tooltip-row"><span>Rejected</span><span class="tv">' +
    (stats.rejected || 0) +
    "</span></div>" +
    '<div class="tooltip-row"><span>No-Show</span><span class="tv">' +
    (stats.no_show || 0) +
    "</span></div>" +
    "</div>"
  );
}

/* ----------------------------------------------------------
   MAP BOARD — Update center nodes
   ---------------------------------------------------------- */

function updateMapBoard(centers, donatedDiff, transfusedDiff) {
  var renderer = ensureSceneRenderer();
  if (!renderer) return;
  renderer.setCenters(centers || [], {
    donorCountMap: state.donorCountMap,
    donatedDiff: donatedDiff || 0,
    transfusedDiff: transfusedDiff || 0,
  });
}

/* ----------------------------------------------------------
   Donor sprite animation
   ---------------------------------------------------------- */

function spawnDonorSprite(node, board) {
  return;
}

/* ----------------------------------------------------------
   Donation floater (+N)
   ---------------------------------------------------------- */

function spawnDonationFloater(node, board, amount) {
  return;
}

/* ----------------------------------------------------------
   Transfer lines between blood banks and hospitals
   ---------------------------------------------------------- */

function drawTransferLines(nodes, bankIndices, hospitalIndices, board) {
  return;
}

function clearTransferTimers() {
  for (var i = 0; i < state.transferTimers.length; i++) {
    clearTimeout(state.transferTimers[i]);
  }
  state.transferTimers = [];
  if (state.sceneRenderer) {
    state.sceneRenderer.clearTransientEffects();
  }
  var svg = $("transfer-svg");
  if (svg) svg.innerHTML = "";
}

/* ----------------------------------------------------------
   Speed buttons
   ---------------------------------------------------------- */

function highlightSpeedButton(speed) {
  var ids = ["speed-1x", "speed-2x", "speed-5x", "speed-10x"];
  var vals = [1, 2, 5, 10];
  for (var i = 0; i < ids.length; i++) {
    var btn = $(ids[i]);
    if (btn) {
      if (vals[i] === speed) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    }
  }
}

function setSpeed(speed) {
  state.speed = speed;
  sendCommand({ command: "set_speed", speed: speed });
  highlightSpeedButton(speed);
  highlightRuntimeSpeed(speed);
}

/* ----------------------------------------------------------
   Report rendering
   ---------------------------------------------------------- */

function renderReport(report) {
  var el = $("report-content");
  if (!el) return;

  var summary = report.summary || {};
  var rewardTerms = report.reward_terms || {};
  var centers = report.centers || [];
  var timeline = report.inventory_timeline || [];

  /* Update the hero score elements */
  var finalScoreEl = $("report-final-score");
  if (finalScoreEl) {
    finalScoreEl.textContent = formatNumber(summary.episode_score);
  }
  var shortageRateEl = $("report-shortage-rate");
  if (shortageRateEl) {
    shortageRateEl.textContent = formatPercent(summary.shortage_rate);
  }
  var fillRateEl = $("report-fill-rate");
  if (fillRateEl) {
    var sr = parseFloat(summary.shortage_rate);
    var fillRate = isNaN(sr) ? "—" : formatPercent(100 - sr);
    fillRateEl.textContent = fillRate;
  }

  var html = "";

  html += '<div class="report-hero">';
  html +=
    '<div class="report-score">' +
    formatNumber(summary.episode_score) +
    "</div>";
  html += '<div class="report-score-label">Episode Score</div>';
  if (report.generated_at) {
    html +=
      '<div class="report-timestamp">' +
      escapeHtml(report.generated_at) +
      "</div>";
  }
  if (report.hours) {
    html +=
      '<div class="report-hours">' + report.hours + " hours simulated</div>";
  }
  html += "</div>";

  html += "<h3>Key Metrics</h3>";
  html += '<div class="report-metrics-grid">';
  var metricDefs = [
    { key: "shortage_rate", label: "Shortage Rate", fmt: formatPercent },
    { key: "total_donated", label: "Total Donated", fmt: formatInteger },
    { key: "total_transfused", label: "Total Transfused", fmt: formatInteger },
    { key: "total_expired", label: "Total Expired", fmt: formatInteger },
    { key: "total_shortage", label: "Total Shortages", fmt: formatInteger },
    { key: "budget_spent", label: "Budget Spent", fmt: formatNumber },
    { key: "budget_remaining", label: "Budget Remaining", fmt: formatNumber },
    { key: "episode_score", label: "Episode Score", fmt: formatNumber },
    { key: "reward_total", label: "Total Reward", fmt: formatNumber },
    { key: "fill_rate", label: "Fill Rate", fmt: formatPercent },
  ];
  for (var m = 0; m < metricDefs.length; m++) {
    var md = metricDefs[m];
    var val = summary[md.key];
    if (val == null) continue;
    html +=
      '<div class="metric-card">' +
      '<div class="metric-value">' +
      md.fmt(val) +
      "</div>" +
      '<div class="metric-label">' +
      escapeHtml(md.label) +
      "</div>" +
      "</div>";
  }
  html += "</div>";

  var rewardKeys = Object.keys(rewardTerms);
  if (rewardKeys.length > 0) {
    html += "<h3>Reward Terms Breakdown</h3>";
    html += '<div class="reward-terms">';
    var maxReward = 0;
    for (var r = 0; r < rewardKeys.length; r++) {
      var rv = Math.abs(rewardTerms[rewardKeys[r]]);
      if (rv > maxReward) maxReward = rv;
    }
    if (maxReward === 0) maxReward = 1;
    for (var r2 = 0; r2 < rewardKeys.length; r2++) {
      var rk = rewardKeys[r2];
      var rVal = rewardTerms[rk];
      var barW = clamp((Math.abs(rVal) / maxReward) * 100, 0, 100);
      var isNeg = rVal < 0;
      html +=
        '<div class="reward-row">' +
        '<span class="reward-label">' +
        escapeHtml(rk) +
        "</span>" +
        '<div class="reward-bar-track">' +
        '<div class="reward-bar' +
        (isNeg ? " reward-bar-negative" : "") +
        '" style="width:' +
        barW +
        '%"></div>' +
        "</div>" +
        '<span class="reward-value">' +
        formatNumber(rVal) +
        "</span>" +
        "</div>";
    }
    html += "</div>";
  }

  if (centers.length > 0) {
    html += "<h3>Centers Final State</h3>";
    html +=
      '<table class="report-table"><thead><tr>' +
      "<th>Name</th><th>Type</th><th>RBC</th><th>PLT</th><th>PLS</th>" +
      "<th>Donated</th><th>Transfused</th><th>Expired</th><th>Total</th>" +
      "</tr></thead><tbody>";
    for (var c = 0; c < centers.length; c++) {
      var ct = centers[c];
      var ci = ct.inventory || {};
      var cs = ct.stats || {};
      html +=
        "<tr>" +
        "<td>" +
        escapeHtml(ct.name || "") +
        "</td>" +
        "<td>" +
        escapeHtml(ct.type || "") +
        "</td>" +
        "<td>" +
        (ci.RBC || 0) +
        "</td>" +
        "<td>" +
        (ci.PLATELETS || 0) +
        "</td>" +
        "<td>" +
        (ci.PLASMA || 0) +
        "</td>" +
        "<td>" +
        (cs.donated || 0) +
        "</td>" +
        "<td>" +
        (cs.transfused || 0) +
        "</td>" +
        "<td>" +
        (cs.expired || 0) +
        "</td>" +
        "<td>" +
        (ct.total_units || 0) +
        "</td>" +
        "</tr>";
    }
    html += "</tbody></table>";
  }

  if (timeline.length > 0) {
    html += "<h3>Inventory Timeline</h3>";
    html += renderTimelineSummary(timeline);
  }

  el.innerHTML = html;
}

function renderTimelineSummary(timeline) {
  if (!timeline || timeline.length === 0) return "";

  var html = '<div class="timeline-summary">';
  html +=
    '<table class="report-table"><thead><tr>' +
    "<th>Hour</th><th>RBC</th><th>PLT</th><th>PLS</th><th>Shortages</th>" +
    "</tr></thead><tbody>";

  var step = Math.max(1, Math.floor(timeline.length / 20));
  for (var i = 0; i < timeline.length; i += step) {
    var t = timeline[i];
    html +=
      "<tr>" +
      "<td>" +
      (t.hour != null ? t.hour : i) +
      "</td>" +
      "<td>" +
      (t.RBC || 0) +
      "</td>" +
      "<td>" +
      (t.PLATELETS || 0) +
      "</td>" +
      "<td>" +
      (t.PLASMA || 0) +
      "</td>" +
      "<td>" +
      (t.shortages || 0) +
      "</td>" +
      "</tr>";
  }

  var last = timeline[timeline.length - 1];
  if (timeline.length % step !== 1 && timeline.length > 1) {
    html +=
      "<tr>" +
      "<td>" +
      (last.hour != null ? last.hour : timeline.length - 1) +
      "</td>" +
      "<td>" +
      (last.RBC || 0) +
      "</td>" +
      "<td>" +
      (last.PLATELETS || 0) +
      "</td>" +
      "<td>" +
      (last.PLASMA || 0) +
      "</td>" +
      "<td>" +
      (last.shortages || 0) +
      "</td>" +
      "</tr>";
  }

  html += "</tbody></table>";
  html += renderTextSparkline(timeline);
  html += "</div>";
  return html;
}

function renderTextSparkline(timeline) {
  if (!timeline || timeline.length === 0) return "";

  var maxRBC = 1;
  for (var i = 0; i < timeline.length; i++) {
    var val = timeline[i].RBC || 0;
    if (val > maxRBC) maxRBC = val;
  }

  var blocks = [
    " ",
    "\u2581",
    "\u2582",
    "\u2583",
    "\u2584",
    "\u2585",
    "\u2586",
    "\u2587",
    "\u2588",
  ];
  var spark = "";
  var step = Math.max(1, Math.floor(timeline.length / 60));
  for (var j = 0; j < timeline.length; j += step) {
    var ratio = (timeline[j].RBC || 0) / maxRBC;
    var idx = Math.round(ratio * (blocks.length - 1));
    spark += blocks[clamp(idx, 0, blocks.length - 1)];
  }

  return (
    '<div class="timeline-spark"><span class="spark-label">RBC trend: </span><span class="spark-chars">' +
    spark +
    "</span></div>"
  );
}

function renderPartialReport() {
  var el = $("report-content");
  if (!el) return;
  var step = state.lastStepData;
  if (!step) {
    el.innerHTML = "<p>Mission stopped. No data to display.</p>";
    return;
  }

  var html = '<div class="report-hero">';
  html += '<div class="report-score">' + formatNumber(step.score) + "</div>";
  html +=
    '<div class="report-score-label">Score at Stop (Hour ' +
    Math.round(step.hour || 0) +
    ")</div>";
  html += "</div>";

  html += '<div class="report-metrics-grid">';
  html +=
    '<div class="metric-card"><div class="metric-value">' +
    formatPercent(step.shortage_rate) +
    '</div><div class="metric-label">Shortage Rate</div></div>';
  html +=
    '<div class="metric-card"><div class="metric-value">' +
    formatInteger(step.total_donated) +
    '</div><div class="metric-label">Total Donated</div></div>';
  html +=
    '<div class="metric-card"><div class="metric-value">' +
    formatInteger(step.total_transfused) +
    '</div><div class="metric-label">Total Transfused</div></div>';
  html +=
    '<div class="metric-card"><div class="metric-value">' +
    formatInteger(step.total_expired) +
    '</div><div class="metric-label">Total Expired</div></div>';
  html +=
    '<div class="metric-card"><div class="metric-value">' +
    formatNumber(step.budget_remaining) +
    '</div><div class="metric-label">Budget Remaining</div></div>';
  html += "</div>";

  html +=
    "<p style='text-align:center;color:var(--text-muted);margin-top:1rem;'><em>Mission was stopped before completion.</em></p>";
  el.innerHTML = html;
}

/* ----------------------------------------------------------
   Reset / Restart
   ---------------------------------------------------------- */

function resetState() {
  closeWebSocket();
  state.currentStep = null;
  state.centers = [];
  state.paused = false;
  state.speed = 1;
  setSimRunning(false);
  state.initData = null;
  state.lastStepData = null;
  state.report = null;
  state.previousHud = {};
  state.previousDonated = 0;
  state.previousTransfused = 0;
  state.donorCountMap = {};
  setRuntimeMapZoom(RUNTIME_UI_CONFIG.zoom.min);
  stopSnow();
  state.snowActive = false;
  clearTransferTimers();
  if (state.sceneRenderer) {
    state.sceneRenderer.reset();
  }
  showTwinPanels(false);
  resetProgressMode();
  renderTwinEvents([]);
  renderTwinSourceStatus({});
  renderTwinSignals({});
  renderTwinDivergence({});
  renderNarratives([]);
  updateRuntimeDashboard({
    centers: [],
    shortage_rate: RUNTIME_UI_CONFIG.metricDefaults.shortageRate,
    total_transfused: 0,
    total_shortage: 0,
    budget_remaining: 0,
  });
  updateDataSources({
    weather: { connected: false },
    events: { connected: false },
  });
}

function restart() {
  resetState();
  showScreen("setup-screen");
  showMenuView("menu-home-view");
  setConnectionStatus("Disconnected", false);

  // Clear map board center nodes
  var board = $("map-board");
  if (board) {
    var existing = board.querySelectorAll(
      ".center-node, .donor-sprite, .donation-floater",
    );
    for (var i = 0; i < existing.length; i++) {
      existing[i].remove();
    }
  }
  var svg = $("transfer-svg");
  if (svg) svg.innerHTML = "";

  clearEventsFeed();
  clearActionsList();
}

/* ----------------------------------------------------------
   Event binding
   ---------------------------------------------------------- */

function bindEvents() {
  document.addEventListener("mousemove", updateGameChromeHoverState);
  window.addEventListener("blur", clearGameChromeHoverState);

  var gameScreen = $("game-screen");
  if (gameScreen) {
    gameScreen.addEventListener("mouseleave", clearGameChromeHoverState);
  }

  var menuTargets = document.querySelectorAll("[data-menu-target]");
  for (var mt = 0; mt < menuTargets.length; mt++) {
    menuTargets[mt].addEventListener("click", function () {
      openSetupDetail(this.getAttribute("data-menu-target"));
    });
  }

  var menuBackButtons = document.querySelectorAll("[data-menu-back]");
  for (var mb = 0; mb < menuBackButtons.length; mb++) {
    menuBackButtons[mb].addEventListener("click", function () {
      showMenuView("menu-home-view");
    });
  }

  var scenarioSelect = $("scenario-select");
  if (scenarioSelect) {
    scenarioSelect.addEventListener("change", updateScenarioDesc);
    scenarioSelect.addEventListener("change", function () {
      if (!state.customScenarioDirty) {
        loadCustomScenarioBase(this.value);
      } else {
        populateCustomScenarioBaseSelect(
          ($("custom-scenario-base-select") || {}).value || this.value,
        );
      }
      setLabStatus(
        "Scenario changed. The Simulation Lab will use this scenario on the next run.",
        null,
      );
    });
  }

  var customScenarioToggleBtn = $("custom-scenario-toggle-btn");
  if (customScenarioToggleBtn) {
    customScenarioToggleBtn.addEventListener("click", function () {
      toggleCustomScenarioBuilder();
      if (
        $("custom-scenario-shell") &&
        $("custom-scenario-shell").classList.contains("is-open") &&
        !state.customScenarioDirty
      ) {
        loadCustomScenarioBase(
          ($("custom-scenario-base-select") || {}).value ||
            currentMissionScenarioKey(),
        );
      }
    });
  }

  var customScenarioBaseSelect = $("custom-scenario-base-select");
  if (customScenarioBaseSelect) {
    customScenarioBaseSelect.addEventListener("change", function () {
      loadCustomScenarioBase(this.value);
    });
  }

  var customScenarioResetBtn = $("custom-scenario-reset-btn");
  if (customScenarioResetBtn) {
    customScenarioResetBtn.addEventListener("click", function () {
      loadCustomScenarioBase(
        ($("custom-scenario-base-select") || {}).value ||
          currentMissionScenarioKey(),
      );
    });
  }

  var customScenarioCreateBtn = $("custom-scenario-create-btn");
  if (customScenarioCreateBtn) {
    customScenarioCreateBtn.addEventListener("click", function () {
      createCustomScenario();
    });
  }

  var customScenarioEditor = $("custom-scenario-editor");
  if (customScenarioEditor) {
    customScenarioEditor.addEventListener("input", function (event) {
      if (
        event.target &&
        (
          event.target.hasAttribute("data-custom-scenario-field") ||
          event.target.id === "custom-scenario-name-input" ||
          event.target.id === "custom-scenario-description-input" ||
          event.target.id === "custom-scenario-narrative-input"
        )
      ) {
        state.customScenarioDirty = true;
      }
    });
    customScenarioEditor.addEventListener("change", function (event) {
      if (
        event.target &&
        event.target.id === "custom-scenario-strategy-select"
      ) {
        state.customScenarioDirty = true;
      }
    });
  }

  var startBtn = $("start-btn");
  if (startBtn) {
    startBtn.addEventListener("click", function () {
      state.twinMode = false;
      state.dynamicWorldEnabled = false;
      showScreen("game-screen");
      openWebSocket();
    });
  }

  var startTwinBtn = $("start-twin-btn");
  if (startTwinBtn) {
    startTwinBtn.addEventListener("click", function () {
      state.twinMode = true;
      state.dynamicWorldEnabled = true;
      showScreen("game-screen");
      openTwinWebSocket();
    });
  }

  var pauseBtn = $("pause-btn");
  if (pauseBtn) {
    pauseBtn.addEventListener("click", function () {
      if (state.paused) {
        sendCommand({ command: "resume" });
      } else {
        sendCommand({ command: "pause" });
      }
    });
  }

  function bindPauseToggle(btn) {
    if (!btn) return;
    btn.addEventListener("click", function () {
      if (state.paused) {
        sendCommand({ command: "resume" });
      } else {
        sendCommand({ command: "pause" });
      }
    });
  }

  bindPauseToggle($("runtime-pause-btn"));
  bindPauseToggle($("toolbar-pause-btn"));

  var stopBtn = $("stop-btn");
  if (stopBtn) {
    stopBtn.addEventListener("click", function () {
      sendCommand({ command: "stop" });
    });
  }

  var runtimeResetBtn = $("runtime-reset-btn");
  if (runtimeResetBtn) {
    runtimeResetBtn.addEventListener("click", function () {
      sendCommand({ command: "stop" });
    });
  }

  var twinDetailsToggle = $("twin-details-toggle");
  if (twinDetailsToggle) {
    twinDetailsToggle.addEventListener("click", function () {
      document.body.classList.toggle("twin-details-open");
    });
  }

  var runtimeZoomIn = $("runtime-zoom-in");
  if (runtimeZoomIn) {
    runtimeZoomIn.addEventListener("click", function () {
      setRuntimeMapZoom(state.runtimeMapZoom + RUNTIME_UI_CONFIG.zoom.step);
    });
  }

  var runtimeZoomOut = $("runtime-zoom-out");
  if (runtimeZoomOut) {
    runtimeZoomOut.addEventListener("click", function () {
      setRuntimeMapZoom(state.runtimeMapZoom - RUNTIME_UI_CONFIG.zoom.step);
    });
  }

  var speed1 = $("speed-1x");
  var speed2 = $("speed-2x");
  var speed5 = $("speed-5x");
  var speed10 = $("speed-10x");
  if (speed1)
    speed1.addEventListener("click", function () {
      setSpeed(1);
    });
  if (speed2)
    speed2.addEventListener("click", function () {
      setSpeed(2);
    });
  if (speed5)
    speed5.addEventListener("click", function () {
      setSpeed(5);
    });
  if (speed10)
    speed10.addEventListener("click", function () {
      setSpeed(10);
    });

  // Dashboard runtime-bar + toolbar speed buttons
  var rtSpeeds = [
    ["rt-speed-1x", 1],
    ["rt-speed-2x", 2],
    ["rt-speed-5x", 5],
    ["rt-speed-10x", 10],
    ["toolbar-speed-1x", 1],
    ["toolbar-speed-2x", 2],
    ["toolbar-speed-5x", 5],
    ["toolbar-speed-10x", 10],
  ];
  for (var i = 0; i < rtSpeeds.length; i++) {
    var rtBtn = $(rtSpeeds[i][0]);
    if (rtBtn) {
      (function (val) {
        rtBtn.addEventListener("click", function () {
          setSpeed(val);
        });
      })(rtSpeeds[i][1]);
    }
  }

  var restartBtn = $("restart-btn");
  if (restartBtn) {
    restartBtn.addEventListener("click", function () {
      restart();
    });
  }

  // Mode toggle
  var simStrategySelect = $("strategy-select");
  if (simStrategySelect) {
    simStrategySelect.addEventListener("change", function () {
      updateDreamerRunVisibility("strategy-select", "dreamerv3-run-field");
    });
  }

  var twinStrategySelect = $("twin-strategy-select");
  if (twinStrategySelect) {
    twinStrategySelect.addEventListener("change", function () {
      updateDreamerRunVisibility(
        "twin-strategy-select",
        "twin-dreamerv3-run-field",
      );
    });
  }

  // Inject event controls
  var injectSeverity = $("inject-severity");
  if (injectSeverity) {
    injectSeverity.addEventListener("input", function () {
      var lbl = $("inject-severity-label");
      if (lbl) lbl.textContent = parseFloat(injectSeverity.value).toFixed(2);
    });
  }

  var injectSelect = $("inject-event-select");
  if (injectSelect) {
    injectSelect.addEventListener("change", syncInjectSeverityFromSelection);
  }

  var injectBtn = $("inject-btn");
  if (injectBtn) {
    injectBtn.addEventListener("click", function () {
      var evSelect = $("inject-event-select");
      var sevInput = $("inject-severity");
      if (evSelect && sevInput) {
        sendCommand({
          command: "inject_event",
          event_key: evSelect.value,
          severity: parseFloat(sevInput.value) || 0.5,
        });
      }
    });
  }

  var twinActionBtn = $("twin-action-btn");
  if (twinActionBtn) {
    twinActionBtn.addEventListener("click", function () {
      var actionSelect = $("twin-action-select");
      if (actionSelect && actionSelect.value) {
        sendCommand({
          command: "execute_action",
          action_key: actionSelect.value,
        });
      }
    });
  }

  var twinRefreshBtn = $("twin-refresh-btn");
  if (twinRefreshBtn) {
    twinRefreshBtn.addEventListener("click", function () {
      sendCommand({ command: "refresh_snapshot" });
    });
  }

  var twinBranchBtn = $("twin-branch-btn");
  if (twinBranchBtn) {
    twinBranchBtn.addEventListener("click", function () {
      var policySelect = $("twin-branch-policy-select");
      var policy = policySelect && policySelect.value ? policySelect.value : "";
      if (!policy) return;
      sendCommand({
        command: "run_branch",
        branch_key: "studio-" + policy + "-" + Date.now(),
        policy_overrides: { strategy_key: policy },
      });
    });
  }

  var twinRecommendBtn = $("twin-recommend-btn");
  if (twinRecommendBtn) {
    twinRecommendBtn.addEventListener("click", function () {
      sendCommand({ command: "request_recommendation" });
    });
  }

  var twinControlRoom = $("twin-control-room");
  if (twinControlRoom) {
    twinControlRoom.addEventListener("click", function (event) {
      var target = event.target;
      if (!target || !target.closest) return;
      var promoteBtn = target.closest("#twin-promote-btn");
      if (promoteBtn) {
        sendCommand({
          command: "promote_action",
          action_id: promoteBtn.getAttribute("data-action-id"),
        });
        return;
      }
      var simulateBtn = target.closest("#twin-simulate-recommendation-btn");
      if (simulateBtn) {
        sendCommand({
          command: "execute_action",
          action_key: simulateBtn.getAttribute("data-action-key"),
        });
      }
    });
  }

  // Random event rate slider (setup screen)
  var rateSlider = document.getElementById("random-event-rate");
  var rateLabel = document.getElementById("random-rate-label");
  if (rateSlider) {
    rateSlider.addEventListener("input", function () {
      if (rateLabel) rateLabel.textContent = parseFloat(this.value).toFixed(2);
    });
  }

  // In-game random event rate slider
  var ingameRate = document.getElementById("ingame-random-rate");
  var ingameRateVal = document.getElementById("ingame-rate-value");
  if (ingameRate) {
    ingameRate.addEventListener("input", function () {
      try {
        var val = parseFloat(this.value);
        if (ingameRateVal) ingameRateVal.textContent = val.toFixed(2);
        if (rateSlider) rateSlider.value = this.value;
        if (rateLabel) rateLabel.textContent = val.toFixed(2);
        sendCommand({ command: "set_random_event_rate", rate: val });
        if (!state.twinMode) {
          showToast({
            title: "Event Rate Updated",
            description:
              "Runtime event-rate changes are not supported in simulation mode. The rate will apply on the next run.",
            severity: 0.2,
            icon: "fa-circle-info",
            type: "info",
            duration: 3500,
          });
        }
      } catch (err) {
        console.error("Event rate slider error:", err);
      }
    });
  }

  // World detail dropdown toggle
  var worldCompact = document.getElementById("hud-world-compact");
  var worldDropdown = document.getElementById("world-detail-dropdown");
  if (worldCompact && worldDropdown) {
    worldCompact.addEventListener("click", function (e) {
      e.stopPropagation();
      worldDropdown.classList.toggle("open");
    });
    document.addEventListener("click", function (e) {
      if (
        worldDropdown.classList.contains("open") &&
        !worldDropdown.contains(e.target) &&
        !worldCompact.contains(e.target)
      ) {
        worldDropdown.classList.remove("open");
      }
    });
  }

  var labSlider = $("lab-timeline-slider");
  if (labSlider) {
    labSlider.addEventListener("input", updateTimelineLabel);
  }

  var labPolicyList = $("lab-policy-list");
  if (labPolicyList) {
    labPolicyList.addEventListener("input", handleLabPolicyComposerInput);
    labPolicyList.addEventListener("change", handleLabPolicyComposerChange);
    labPolicyList.addEventListener("click", handleLabPolicyComposerClick);
  }

  var labAddPolicyBtn = $("lab-add-policy-btn");
  if (labAddPolicyBtn) {
    labAddPolicyBtn.addEventListener("click", addLabPolicy);
  }

  var labAllPoliciesBtn = $("lab-all-policies-btn");
  if (labAllPoliciesBtn) {
    labAllPoliciesBtn.addEventListener("click", selectAllLabPolicies);
  }

  var labResetPoliciesBtn = $("lab-reset-policies-btn");
  if (labResetPoliciesBtn) {
    labResetPoliciesBtn.addEventListener("click", resetLabPolicies);
  }

  var labRunBtn = $("lab-run-btn");
  if (labRunBtn) {
    labRunBtn.addEventListener("click", runStudioExperiment);
  }

  var labAssistantBtn = $("lab-assistant-btn");
  if (labAssistantBtn) {
    labAssistantBtn.addEventListener("click", runStrategyAssistant);
  }
}

/* ----------------------------------------------------------
   Initialization
   ---------------------------------------------------------- */

function init() {
  showScreen("setup-screen");
  showMenuView("menu-home-view");
  setConnectionStatus("Disconnected", false);
  bindEvents();

  fetch("/api/meta")
    .then(function (res) {
      return res.ok ? res.json() : null;
    })
    .then(function (meta) {
      if (!meta || !meta.actions) return;
      var catalog = {};
      for (var i = 0; i < meta.actions.length; i++) {
        var action = meta.actions[i];
        if (action && action.key) {
          catalog[action.key] = action;
        }
      }
      state.actionCatalog = catalog;
    })
    .catch(function () {
      /* action catalog optional */
    });

  fetch("/api/sim/setup")
    .then(function (res) {
      if (!res.ok) throw new Error("Failed to fetch setup: " + res.status);
      return res.json();
    })
    .then(function (data) {
      populateSetup(data);
      if (data && data.forecast_panel) {
        initTwinForecastPanel(data.forecast_panel);
      }
    })
    .catch(function (err) {
      console.error("Setup fetch error:", err);
      var desc = $("scenario-desc");
      if (desc) desc.textContent = "Failed to load setup data. Please refresh.";
    });

  // Also fetch twin config silently
  fetch("/api/twin/setup")
    .then(function (res) {
      return res.ok ? res.json() : null;
    })
    .then(function (data) {
      if (data && data.twin_config) {
        populateTwinSetup(data.twin_config);
      }
      if (data && data.forecast_panel) {
        initTwinForecastPanel(data.forecast_panel);
      }
    })
    .catch(function () {
      /* twin setup not available */
    });

  fetch("/api/studio/setup")
    .then(function (res) {
      return res.ok ? res.json() : null;
    })
    .then(function (data) {
      if (data) {
        populateStudioSetup(data);
      }
    })
    .catch(function (err) {
      console.error("Studio setup error:", err);
      setLabStatus("Studio lab setup failed to load.", "error");
    });
}

document.addEventListener("DOMContentLoaded", init);
