/* Blood Supply Digital Twin — Dashboard UI Components */
(function (global) {
  "use strict";

  var DEFAULT_DASHBOARD = {
    source: { status: "loading", label: "Loading Django data", warnings: [] },
    weather: "Weather unavailable",
    kpis: {
      totalInventory: 0,
      inventoryDeltaPct: null,
      inventoryCaption: "Loading inventory",
      unitsInTransit: 0,
      deliveryCaption: "No transport records",
      criticalShortages: 0,
      shortageCaption: "No active alerts",
      serviceLevel: null,
      networkEfficiency: null,
      efficiencyDeltaPct: null,
    },
    facilities: [],
    mapCenters: [],
    routes: [],
    vehicles: [],
    forecast: { demand: [], supply: [], labels: [] },
    alerts: [],
    alertCount: 0,
    bloodTypes: [],
    deliveries: { total: 0, onTime: "0", delayed: "0", items: [], caption: "No transport records" },
    utilization: [],
  };
  var dashboardData = cloneDashboard(DEFAULT_DASHBOARD);
  var dashboardLoaded = false;
  var dashboardLoading = false;
  var forecastPanelRuntime = {
    jobId: null,
    maxPoints: 24,
    selectBound: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function cloneDashboard(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function asNumber(value, fallback) {
    var parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function cssToken(value, fallback) {
    var token = String(value == null ? "" : value).toLowerCase().replace(/[^a-z0-9_-]/g, "-");
    return token || fallback || "default";
  }

  function formatInteger(value) {
    var parsed = asNumber(value, null);
    if (parsed == null) return "—";
    return Math.round(parsed).toLocaleString("en-US");
  }

  function formatCompact(value) {
    var parsed = asNumber(value, null);
    if (parsed == null) return "—";
    var abs = Math.abs(parsed);
    if (abs >= 1000000) return (parsed / 1000000).toFixed(1).replace(/\.0$/, "") + "M";
    if (abs >= 1000) return (parsed / 1000).toFixed(1).replace(/\.0$/, "") + "K";
    return formatInteger(parsed);
  }

  function formatPercent(value) {
    var parsed = asNumber(value, null);
    return parsed == null ? "—" : parsed.toFixed(1) + "%";
  }

  function formatDelta(value, fallback) {
    var parsed = asNumber(value, null);
    if (parsed == null) return fallback || "No comparison data";
    return (parsed > 0 ? "+" : "") + parsed.toFixed(1) + "% vs latest snapshot";
  }

  function formatUnits(value) {
    var parsed = asNumber(value, null);
    return parsed == null ? "—" : formatCompact(parsed);
  }

  function dateLabel(value) {
    var raw = String(value || "");
    if (!raw) return "";
    var parsed = new Date(raw);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    }
    return raw.slice(5) || raw;
  }

  function timeAgo(value) {
    if (!value) return "";
    var parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    var seconds = Math.max(0, Math.round((Date.now() - parsed.getTime()) / 1000));
    if (seconds < 60) return "just now";
    var minutes = Math.floor(seconds / 60);
    if (minutes < 60) return minutes + " min ago";
    var hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + " hr ago";
    return Math.floor(hours / 24) + " d ago";
  }

  function normalizeDashboardData(payload) {
    var next = cloneDashboard(DEFAULT_DASHBOARD);
    payload = payload || {};
    next.source = Object.assign(next.source, payload.source || {});
    next.weather = payload.weather || next.weather;
    next.kpis = Object.assign(next.kpis, payload.kpis || {});
    next.facilities = asArray(payload.facilities);
    next.mapCenters = asArray(payload.mapCenters);
    next.routes = asArray(payload.routes);
    next.vehicles = asArray(payload.vehicles);
    next.forecast = Object.assign(next.forecast, payload.forecast || {});
    next.forecast.demand = asArray(next.forecast.demand);
    next.forecast.supply = asArray(next.forecast.supply);
    next.forecast.labels = asArray(next.forecast.labels);
    next.alerts = asArray(payload.alerts);
    next.alertCount = asNumber(payload.alertCount, next.alerts.length);
    next.bloodTypes = asArray(payload.bloodTypes);
    next.deliveries = Object.assign(next.deliveries, payload.deliveries || {});
    next.deliveries.items = asArray(next.deliveries.items);
    next.utilization = asArray(payload.utilization);
    return next;
  }

  function currentData() {
    return dashboardData || DEFAULT_DASHBOARD;
  }

  function setText(id, value) {
    var el = $(id);
    if (el) el.textContent = value;
  }

  function icon(name, size) {
    size = size || 18;
    return (
      '<svg xmlns="http://www.w3.org/2000/svg" width="' +
      size +
      '" height="' +
      size +
      '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><use href="#icon-' +
      name +
      '"/></svg>'
    );
  }

  function injectIconSprites() {
    if ($("dt-icon-sprites")) return;
    var ns = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("id", "dt-icon-sprites");
    svg.setAttribute("aria-hidden", "true");
    svg.style.cssText = "position:absolute;width:0;height:0;overflow:hidden";
    svg.innerHTML =
      '<symbol id="icon-layout-dashboard" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></symbol>' +
      '<symbol id="icon-map" viewBox="0 0 24 24"><polygon points="1 6 9 2 15 6 23 2 23 18 15 22 9 18 1 22"/><line x1="9" y1="2" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="22"/></symbol>' +
      '<symbol id="icon-package" viewBox="0 0 24 24"><path d="M16.5 9.4 7.55 4.24"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.29 7 12 12 20.71 7"/><line x1="12" y1="22" x2="12" y2="12"/></symbol>' +
      '<symbol id="icon-clipboard" viewBox="0 0 24 24"><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></symbol>' +
      '<symbol id="icon-truck" viewBox="0 0 24 24"><path d="M14 18V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v11a1 1 0 0 0 1 1h2"/><path d="M15 18H9"/><path d="M19 18h2a1 1 0 0 0 1-1v-3.65a1 1 0 0 0-.22-.624l-3.48-4.35A1 1 0 0 0 17.52 8H14"/><circle cx="17" cy="18" r="2"/><circle cx="7" cy="18" r="2"/></symbol>' +
      '<symbol id="icon-building" viewBox="0 0 24 24"><rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/></symbol>' +
      '<symbol id="icon-chart" viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></symbol>' +
      '<symbol id="icon-bell" viewBox="0 0 24 24"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></symbol>' +
      '<symbol id="icon-file" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></symbol>' +
      '<symbol id="icon-settings" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></symbol>' +
      '<symbol id="icon-droplet" viewBox="0 0 24 24"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></symbol>' +
      '<symbol id="icon-shield" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></symbol>' +
      '<symbol id="icon-alert-triangle" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></symbol>' +
      '<symbol id="icon-database" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/></symbol>' +
      '<symbol id="icon-layers" viewBox="0 0 24 24"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></symbol>' +
      '<symbol id="icon-crosshair" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></symbol>' +
      '<symbol id="icon-chevrons-left" viewBox="0 0 24 24"><polyline points="11 17 6 12 11 7"/><polyline points="18 17 13 12 18 7"/></symbol>' +
      '<symbol id="icon-cloud-rain" viewBox="0 0 24 24"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M16 14v6"/><path d="M8 14v6"/><path d="M12 16v6"/></symbol>';
    document.body.appendChild(svg);
  }

  function hasOperationalMapData(data) {
    return asArray(data && data.facilities).length > 0;
  }

  function mapSourceSummary(data) {
    data = data || {};
    var source = data.source || {};
    var warnings = asArray(source.warnings);
    if (warnings.length) return String(warnings[0]).split("\n")[0];
    if (source.generatedAt) return "Generated " + timeAgo(source.generatedAt);
    return source.label || "Waiting for operational rows";
  }

  var ISO_MAP_DAY_SRC = "assets/isometric_pastel_city_with_key_landmarks_day.png";
  var ISO_MAP_NIGHT_SRC = "assets/isometric_pastel_city_with_key_landmarks_night.png";
  var ISO_MAP_VIEWBOX = "0 0 1672 941";
  var mapNightMode = false;
  var ISO_MAP_WIDTH = 1672;
  var ISO_MAP_HEIGHT = 941;

  var ISO_MAP_HOSPITAL_SLOTS = [
    { x: 470, y: 255 },
    { x: 1210, y: 260 },
    { x: 265, y: 520 },
    { x: 1230, y: 720 },
  ];

  var ISO_MAP_BLOODBANK_SLOTS = [
    { x: 790, y: 215 },
    { x: 710, y: 690 },
  ];

  var ISO_MAP_STATIC_FACILITIES = [
    { id: "central-lab", name: "Central Lab", type: "lab", x: 825, y: 450 },
    { id: "donation-west", name: "Donation Center West", type: "donation", x: 310, y: 705 },
    { id: "donation-east", name: "Donation Center East", type: "donation", x: 1285, y: 500 },
  ];

  var ISO_MAP_FACILITIES = [
    { id: "northside", name: "Northside Hospital", type: "hospital", x: 470, y: 255 },
    { id: "city", name: "City Hospital", type: "hospital", x: 1210, y: 260 },
    { id: "westview", name: "Westview Hospital", type: "hospital", x: 265, y: 520 },
    { id: "southpoint", name: "Southpoint Hospital", type: "hospital", x: 1230, y: 720 },
    { id: "bloodbank-north", name: "Blood Bank", type: "bloodbank", x: 790, y: 215 },
    { id: "bloodbank-south", name: "Blood Bank", type: "bloodbank", x: 710, y: 690 },
  ].concat(ISO_MAP_STATIC_FACILITIES);

  var ISO_MAP_ROUTES = [
    {
      id: "route-red-north",
      type: "red",
      d: "M 470 255 C 610 260, 700 230, 790 215 C 940 210, 1080 225, 1210 260",
    },
    {
      id: "route-blue-main",
      type: "blue",
      d: "M 310 705 C 500 650, 650 555, 825 450 C 980 390, 1140 430, 1285 500",
    },
    {
      id: "route-green-east",
      type: "green",
      d: "M 1285 500 C 1110 520, 960 510, 825 450 C 670 390, 555 310, 470 255",
    },
    {
      id: "route-purple-lab",
      type: "purple",
      d: "M 710 690 C 750 590, 790 520, 825 450 C 820 350, 805 275, 790 215",
    },
    {
      id: "route-blue-south",
      type: "blue",
      d: "M 265 520 C 420 570, 560 630, 710 690 C 880 760, 1060 750, 1230 720",
    },
    {
      id: "route-red-east-hospital",
      type: "red",
      d: "M 1210 260 C 1280 350, 1310 430, 1285 500 C 1270 590, 1250 660, 1230 720",
    },
  ];

  var ISO_MAP_DELIVERY_GLOWS = [
    { x: 300, y: 185, r: 28, color: "#ef4444" },
    { x: 600, y: 300, r: 24, color: "#2563eb" },
    { x: 520, y: 348, r: 22, color: "#2563eb" },
    { x: 900, y: 320, r: 24, color: "#22c55e" },
  ];

  var FACILITY_STYLES = {
    hospital: {
      roof: "#fca5a5",
      left: "#f1f5f9",
      right: "#ffffff",
      accent: "#ef4444",
      badge: "H",
      badgeKind: "text",
      scale: 1.25,
      labelWidth: 132,
    },
    bloodbank: {
      roof: "#c4b5fd",
      left: "#ede9fe",
      right: "#ffffff",
      accent: "#8b5cf6",
      badge: "BB",
      badgeKind: "text",
      scale: 1.05,
      labelWidth: 108,
    },
    lab: {
      roof: "#93c5fd",
      left: "#dbeafe",
      right: "#ffffff",
      accent: "#2563eb",
      badge: "flask",
      badgeKind: "flask",
      scale: 1.05,
      labelWidth: 108,
    },
    donation: {
      roof: "#86efac",
      left: "#dcfce7",
      right: "#ffffff",
      accent: "#22c55e",
      badge: "droplet",
      badgeKind: "droplet",
      scale: 1.0,
      labelWidth: 142,
    },
  };

  var ISO_MAP_ROAD_PATHS = [
    { d: "M 60 395 C 190 335, 340 312, 520 348 S 780 438, 1060 318", type: "main" },
    { d: "M 70 225 C 210 252, 360 282, 520 348", type: "main" },
    { d: "M 520 348 C 585 398, 635 428, 710 472", type: "secondary" },
    { d: "M 340 178 C 455 198, 555 178, 655 208", type: "secondary" },
    { d: "M 875 168 C 915 248, 938 322, 898 438", type: "secondary" },
    { d: "M 185 358 C 295 388, 420 418, 558 442", type: "secondary" },
    { d: "M 1080 318 C 1000 360, 920 388, 850 430", type: "secondary" },
    { d: "M 250 248 C 360 288, 470 318, 580 338", type: "secondary" },
  ];

  var ISO_MAP_ROUNDABOUTS = [
    { x: 520, y: 348, r: 20 },
    { x: 600, y: 300, r: 16 },
    { x: 390, y: 318, r: 14 },
  ];

  var ISO_MAP_BRIDGES = [
    { x: 718, y: 332, rotation: -8, width: 52 },
    { x: 708, y: 468, rotation: 6, width: 44 },
    { x: 728, y: 198, rotation: -14, width: 46 },
  ];

  var ISO_MAP_RIVER = "M 720 -40 C 690 130, 790 230, 720 360 C 660 470, 720 560, 670 700";

  var ISO_MAP_BACKGROUND_BUILDINGS = [
    { x: 88, y: 142, w: 34, h: 22, depth: 18, opacity: 0.48 },
    { x: 132, y: 168, w: 28, h: 18, depth: 15, opacity: 0.44 },
    { x: 168, y: 128, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 214, y: 152, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 248, y: 118, w: 36, h: 22, depth: 18, opacity: 0.5 },
    { x: 118, y: 198, w: 30, h: 19, depth: 14, opacity: 0.42 },
    { x: 156, y: 214, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 202, y: 192, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 96, y: 248, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 142, y: 268, w: 30, h: 18, depth: 14, opacity: 0.44 },
    { x: 188, y: 252, w: 42, h: 25, depth: 21, opacity: 0.54 },
    { x: 236, y: 278, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 278, y: 238, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 322, y: 258, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 368, y: 228, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 412, y: 248, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 448, y: 212, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 492, y: 232, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 538, y: 198, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 582, y: 218, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 628, y: 188, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 672, y: 208, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 718, y: 178, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 762, y: 198, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 808, y: 168, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 952, y: 188, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 998, y: 208, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 1042, y: 178, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 1088, y: 198, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 1128, y: 228, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 78, y: 318, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 124, y: 338, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 172, y: 318, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 218, y: 348, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 262, y: 328, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 348, y: 338, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 398, y: 358, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 442, y: 328, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 488, y: 348, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 648, y: 338, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 692, y: 358, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 738, y: 328, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 782, y: 348, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 828, y: 318, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 872, y: 338, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 918, y: 358, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 962, y: 328, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 1008, y: 348, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 1052, y: 318, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 1098, y: 338, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 92, y: 408, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 138, y: 428, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 182, y: 398, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 228, y: 418, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 272, y: 438, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 358, y: 408, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 402, y: 428, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 448, y: 448, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 492, y: 418, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 538, y: 438, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 648, y: 418, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 692, y: 438, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 738, y: 458, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 782, y: 428, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 828, y: 448, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 872, y: 468, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 918, y: 438, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 962, y: 458, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 1008, y: 428, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 1052, y: 448, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 1098, y: 468, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 1128, y: 498, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 118, y: 488, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 168, y: 508, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 218, y: 488, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 268, y: 508, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 318, y: 528, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 368, y: 498, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 418, y: 518, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 468, y: 538, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 518, y: 508, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 568, y: 528, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 618, y: 498, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 668, y: 518, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 718, y: 538, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 768, y: 508, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 818, y: 528, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 868, y: 498, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 918, y: 518, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 968, y: 538, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 1018, y: 508, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 1068, y: 528, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 1118, y: 548, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 52, y: 168, w: 30, h: 19, depth: 14, opacity: 0.43 },
    { x: 72, y: 288, w: 34, h: 21, depth: 17, opacity: 0.47 },
    { x: 108, y: 368, w: 28, h: 18, depth: 14, opacity: 0.41 },
    { x: 148, y: 148, w: 36, h: 22, depth: 18, opacity: 0.49 },
    { x: 192, y: 168, w: 32, h: 20, depth: 16, opacity: 0.45 },
    { x: 288, y: 148, w: 38, h: 23, depth: 19, opacity: 0.51 },
    { x: 332, y: 178, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 378, y: 198, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 428, y: 168, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 468, y: 188, w: 40, h: 24, depth: 20, opacity: 0.52 },
    { x: 518, y: 168, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 598, y: 228, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 638, y: 248, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 688, y: 228, w: 26, h: 17, depth: 13, opacity: 0.4 },
    { x: 748, y: 248, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 888, y: 248, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 938, y: 268, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 1028, y: 248, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 1148, y: 168, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 48, y: 448, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 198, y: 468, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 298, y: 388, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 548, y: 468, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 628, y: 468, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 698, y: 488, w: 38, h: 23, depth: 19, opacity: 0.5 },
    { x: 848, y: 488, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 998, y: 488, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 1148, y: 418, w: 32, h: 20, depth: 16, opacity: 0.46 },
    { x: 158, y: 568, w: 28, h: 18, depth: 14, opacity: 0.42 },
    { x: 428, y: 568, w: 36, h: 22, depth: 18, opacity: 0.48 },
    { x: 758, y: 568, w: 30, h: 19, depth: 15, opacity: 0.44 },
    { x: 1028, y: 568, w: 34, h: 21, depth: 17, opacity: 0.48 },
    { x: 555, y: 275, w: 34, h: 22, depth: 18, opacity: 0.54 },
    { x: 890, y: 195, w: 36, h: 23, depth: 19, opacity: 0.56 },
    { x: 745, y: 310, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 165, y: 420, w: 38, h: 24, depth: 20, opacity: 0.52 },
    { x: 545, y: 365, w: 32, h: 20, depth: 16, opacity: 0.52 },
    { x: 430, y: 315, w: 34, h: 21, depth: 17, opacity: 0.51 },
    { x: 335, y: 205, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 545, y: 145, w: 32, h: 20, depth: 16, opacity: 0.52 },
    { x: 815, y: 215, w: 28, h: 18, depth: 14, opacity: 0.48 },
    { x: 475, y: 378, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 62, y: 118, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 98, y: 108, w: 32, h: 20, depth: 16, opacity: 0.52 },
    { x: 42, y: 198, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 78, y: 178, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 118, y: 118, w: 34, h: 21, depth: 17, opacity: 0.52 },
    { x: 158, y: 98, w: 28, h: 18, depth: 14, opacity: 0.48 },
    { x: 198, y: 118, w: 32, h: 20, depth: 16, opacity: 0.5 },
    { x: 238, y: 168, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 268, y: 198, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 318, y: 168, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 358, y: 148, w: 32, h: 20, depth: 16, opacity: 0.52 },
    { x: 252, y: 218, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 282, y: 168, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 62, y: 358, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 98, y: 388, w: 32, h: 20, depth: 16, opacity: 0.52 },
    { x: 128, y: 448, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 168, y: 388, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 208, y: 438, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 248, y: 468, w: 32, h: 20, depth: 16, opacity: 0.52 },
    { x: 188, y: 498, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 138, y: 528, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 88, y: 498, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 268, y: 348, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 328, y: 438, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 248, y: 398, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 352, y: 288, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 412, y: 288, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 452, y: 268, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 502, y: 288, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 542, y: 258, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 618, y: 258, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 658, y: 268, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 528, y: 318, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 568, y: 328, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 638, y: 288, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 678, y: 318, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 568, y: 388, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 608, y: 368, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 528, y: 408, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 698, y: 368, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 758, y: 358, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 698, y: 398, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 768, y: 378, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 708, y: 428, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 738, y: 398, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 698, y: 268, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 758, y: 288, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 788, y: 248, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 858, y: 268, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 868, y: 348, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 928, y: 288, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 948, y: 348, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 978, y: 368, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 1028, y: 368, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 1078, y: 358, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 1098, y: 288, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 1048, y: 398, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 1088, y: 418, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 1128, y: 358, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 1158, y: 298, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 718, y: 148, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 738, y: 228, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 708, y: 288, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 728, y: 348, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 748, y: 408, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 718, y: 468, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 168, y: 238, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 228, y: 318, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 178, y: 328, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 388, y: 368, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 458, y: 198, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 498, y: 168, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 618, y: 168, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 648, y: 198, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 828, y: 198, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 878, y: 218, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 928, y: 198, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 978, y: 228, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 1028, y: 198, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 318, y: 498, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 378, y: 448, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 428, y: 488, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 588, y: 448, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 648, y: 448, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 798, y: 448, w: 30, h: 19, depth: 15, opacity: 0.5 },
    { x: 948, y: 418, w: 28, h: 18, depth: 14, opacity: 0.5 },
    { x: 998, y: 438, w: 26, h: 17, depth: 13, opacity: 0.48 },
    { x: 1048, y: 468, w: 30, h: 19, depth: 15, opacity: 0.5 },
  ];

  var ISO_MAP_VEHICLES = [
    { routeId: "route-red-north", delay: "0s", duration: "9s" },
    { routeId: "route-blue-main", delay: "1s", duration: "10s" },
    { routeId: "route-green-east", delay: "2s", duration: "11s" },
    { routeId: "route-purple-lab", delay: "0.5s", duration: "8s" },
    { routeId: "route-blue-south", delay: "1.5s", duration: "12s" },
    { routeId: "route-red-east-hospital", delay: "2.5s", duration: "10s" },
  ];

  var ISO_MAP_CITY_BLOCKS = [
    [95, 255, 70, 38], [170, 220, 58, 32], [420, 195, 64, 34], [720, 210, 72, 36],
    [1020, 280, 60, 30], [130, 430, 66, 35], [380, 480, 74, 38], [680, 500, 68, 34],
    [860, 480, 62, 32], [500, 520, 80, 40], [240, 290, 52, 28], [800, 360, 56, 30],
    [350, 520, 64, 34], [960, 420, 58, 30], [1080, 340, 54, 28],
  ];

  var ISO_MAP_TREES = [
    [105, 275, 7], [225, 205, 6], [365, 265, 8], [535, 245, 6], [775, 285, 7],
    [965, 305, 6], [145, 455, 7], [315, 475, 6], [605, 485, 8], [875, 505, 6],
    [455, 385, 6], [685, 365, 7], [255, 335, 6], [815, 395, 7], [128, 318, 5],
    [488, 268, 6], [1018, 390, 5], [388, 538, 6], [748, 518, 7], [928, 268, 5],
    [178, 178, 6], [428, 198, 5], [558, 228, 6], [678, 248, 5], [828, 278, 6],
    [978, 298, 5], [108, 378, 6], [258, 398, 5], [508, 418, 6], [758, 438, 5],
    [958, 458, 6], [108, 528, 5], [358, 548, 6], [608, 558, 5], [858, 538, 6],
  ];

  function facilityVisualType(type) {
    var raw = cssToken(type || "hospital", "hospital");
    if (raw === "blood_bank" || raw === "bank" || raw === "bloodbank" || raw === "fixe") return "bloodbank";
    if (raw === "mobile" || raw === "donation_center" || raw === "donation") return "donation";
    if (raw === "lab" || raw === "testing_lab") return "lab";
    if (raw === "hospital" || raw === "hopital" || raw === "clinique" || raw === "clinic") return "hospital";
    return "lab";
  }

  function mapOperationalFacilityToSlot(facility, slot, index, kind) {
    return Object.assign({}, slot, {
      id: facility.id || (kind + "-" + (index + 1)),
      name: facility.name || facility.id || (kind === "bloodbank" ? "Blood Bank" : "Hospital"),
      type: facility.type || kind,
      stock: facility.stock,
      usage: facility.usage,
      critical: facility.critical,
      utilization: facility.utilization,
      inventory: facility.inventory,
      stats: facility.stats,
      total_units: facility.total_units,
      region: facility.region,
      centerType: facility.centerType || facility.type,
    });
  }

  function resolveMapFacilityTemplates(data) {
    var runtime = asArray(data && data.facilities);
    var hospitals = runtime.filter(function (facility) {
      return facilityVisualType(facility.type) === "hospital";
    });
    var bloodbanks = runtime.filter(function (facility) {
      return facilityVisualType(facility.type) === "bloodbank";
    });

    if (!hospitals.length && !bloodbanks.length) {
      return ISO_MAP_FACILITIES.slice();
    }

    var mapped = [];
    hospitals.forEach(function (facility, index) {
      var slot = ISO_MAP_HOSPITAL_SLOTS[index % ISO_MAP_HOSPITAL_SLOTS.length];
      mapped.push(mapOperationalFacilityToSlot(facility, slot, index, "hospital"));
    });
    bloodbanks.forEach(function (facility, index) {
      var slot = ISO_MAP_BLOODBANK_SLOTS[index % ISO_MAP_BLOODBANK_SLOTS.length];
      mapped.push(mapOperationalFacilityToSlot(facility, slot, index, "bloodbank"));
    });
    return mapped.concat(ISO_MAP_STATIC_FACILITIES);
  }

  function resolveIsoFacilities(data) {
    return resolveMapFacilityTemplates(data);
  }

  function normalizeCenterKey(value) {
    return String(value || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  }

  function findSimCenter(centers, key) {
    var rawKey = String(key || "").trim();
    if (!rawKey || !centers.length) return null;
    for (var i = 0; i < centers.length; i++) {
      if (String(centers[i].external_id || "") === rawKey) return centers[i];
    }
    var normalized = normalizeCenterKey(key);
    if (!normalized) return null;
    var partial = null;
    for (var j = 0; j < centers.length; j++) {
      var center = centers[j];
      var centerKey = normalizeCenterKey(center.name);
      if (!centerKey) continue;
      if (centerKey === normalized) return center;
      if (!partial && (centerKey.indexOf(normalized) >= 0 || normalized.indexOf(centerKey) >= 0)) {
        partial = center;
      }
    }
    return partial;
  }

  function centerSnapshotToFacilityFields(center, meta) {
    meta = meta || {};
    var total = runtimeTotalInventory([center]);
    return Object.assign({
      stock: total,
      usage: asNumber(center.stats && center.stats.transfused, 0),
      inventory: center.inventory,
      stats: center.stats,
      total_units: center.total_units != null ? center.total_units : total,
      centerType: center.type,
      simSourceName: center.name,
      external_id: center.external_id,
    }, meta);
  }

  function aggregateCentersByKind(centers, kind) {
    var matched = centers.filter(function (center) {
      return facilityVisualType(center.type) === kind;
    });
    if (!matched.length) return null;
    var inventory = { RBC: 0, PLATELETS: 0, PLASMA: 0 };
    var stats = {
      donated: 0,
      transfused: 0,
      expired: 0,
      rejected: 0,
      no_show: 0,
    };
    matched.forEach(function (center) {
      var inv = center.inventory || {};
      inventory.RBC += asNumber(inv.RBC, 0);
      inventory.PLATELETS += asNumber(inv.PLATELETS, 0);
      inventory.PLASMA += asNumber(inv.PLASMA, 0);
      var centerStats = center.stats || {};
      stats.donated += asNumber(centerStats.donated, 0);
      stats.transfused += asNumber(centerStats.transfused, 0);
      stats.expired += asNumber(centerStats.expired, 0);
      stats.rejected += asNumber(centerStats.rejected, 0);
      stats.no_show += asNumber(centerStats.no_show, 0);
    });
    var total = runtimeTotalInventory(matched);
    return {
      stock: total,
      usage: stats.transfused,
      inventory: inventory,
      stats: stats,
      total_units: total,
      simAggregate: true,
      simAggregateCount: matched.length,
      simSourceName: kind === "hospital" ? "Hospital network pool" : "Blood bank network pool",
    };
  }

  function resolveCenterForTemplate(template, centers, poolIndex) {
    var kind = facilityVisualType(template.type);
    if (template.id) {
      var byId = findSimCenter(centers, template.id);
      if (byId) return byId;
    }
    if (template.simCenter) {
      var mapped = findSimCenter(centers, template.simCenter);
      if (mapped) return mapped;
    }
    if (template.name) {
      var named = findSimCenter(centers, template.name);
      if (named) return named;
    }
    var pool = centers.filter(function (center) {
      return facilityVisualType(center.type) === kind;
    });
    if (!pool.length) return null;
    var index = poolIndex[kind] || 0;
    poolIndex[kind] = index + 1;
    return pool[index % pool.length];
  }

  function assignCentersToFacilities(templates, centers) {
    centers = asArray(centers);
    if (!centers.length) return templates;
    var poolIndex = { hospital: 0, bloodbank: 0 };
    return templates.map(function (template) {
      var kind = facilityVisualType(template.type);
      if (!facilityShowsDetailTooltip(kind)) return template;
      var center = resolveCenterForTemplate(template, centers, poolIndex);
      if (center) {
        return Object.assign({}, template, centerSnapshotToFacilityFields(center, {
          simSourceShared: !!template.simCenter,
        }));
      }
      var aggregate = aggregateCentersByKind(centers, kind);
      return aggregate ? Object.assign({}, template, aggregate) : template;
    });
  }

  function resolveIsoFacilitiesWithCenters(data) {
    var templates = resolveIsoFacilities(data);
    if (!asArray(data && data.mapCenters).length) return templates;
    return assignCentersToFacilities(templates, data.mapCenters);
  }

  function facilityShowsDetailTooltip(type) {
    var visual = facilityVisualType(type);
    return visual === "hospital" || visual === "bloodbank";
  }

  function facilityTypeLabel(type) {
    var visual = facilityVisualType(type);
    if (visual === "hospital") return "Hospital";
    if (visual === "bloodbank") return "Blood Bank";
    if (visual === "lab") return "Lab";
    if (visual === "donation") return "Donation Center";
    return "Facility";
  }

  function facilityAlertsFor(data, facility) {
    var alerts = asArray(data && data.alerts);
    var name = String(facility.name || "").trim().toLowerCase();
    var id = String(facility.id || "").trim();
    return alerts.filter(function (alert) {
      var text = String(alert.text || alert.message || alert.title || "").toLowerCase();
      if (alert.facilityId && String(alert.facilityId) === id) return true;
      if (alert.entity_id && String(alert.entity_id) === id) return true;
      if (name && name.length > 3 && text.indexOf(name) >= 0) return true;
      return false;
    });
  }

  function facilityInventoryRows(facility) {
    var inventory = facility && facility.inventory;
    if (!inventory || typeof inventory !== "object") return [];
    return [
      { label: "RBC", value: asNumber(inventory.RBC, 0), tone: "rbc" },
      { label: "Platelets", value: asNumber(inventory.PLATELETS, 0), tone: "plt" },
      { label: "Plasma", value: asNumber(inventory.PLASMA, 0), tone: "pls" },
    ];
  }

  function facilityHasComponentInventory(facility) {
    return !!(facility && facility.inventory && typeof facility.inventory === "object");
  }

  function buildFacilityTooltipHtml(facility, data) {
    if (!facility || !facilityShowsDetailTooltip(facility.type)) return "";
    var alerts = facilityAlertsFor(data, facility);
    var stats = facility.stats || {};
    var inventoryRows = facilityInventoryRows(facility);
    var hasInventory = facilityHasComponentInventory(facility);
    var totalUnits = asNumber(
      facility.total_units,
      hasInventory
        ? inventoryRows.reduce(function (sum, row) { return sum + row.value; }, 0)
        : asNumber(facility.stock, null),
    );
    var sections = "";

    sections +=
      '<div class="mft-header' + (facility.critical ? " is-critical" : "") + '">' +
      '<span class="mft-type-badge mft-type-' + facilityVisualType(facility.type) + '">' +
      escapeHtml(facilityTypeLabel(facility.type)) +
      "</span>" +
      '<h4 class="mft-title">' + escapeHtml(facility.name || "Facility") + "</h4>" +
      (facility.region ? '<p class="mft-region">' + escapeHtml(facility.region) + "</p>" : "") +
      (facility.simSourceName
        ? '<p class="mft-source">' +
          escapeHtml(
            facility.simAggregate
              ? "Live network pool (" + formatInteger(facility.simAggregateCount) + " simulator sites)"
              : "Live simulator site: " + facility.simSourceName,
          ) +
          "</p>"
        : "") +
      "</div>";

    if (hasInventory) {
      sections += '<div class="mft-section"><div class="mft-section-title">Blood component inventory</div>';
      inventoryRows.forEach(function (row) {
        sections +=
          '<div class="mft-row"><span>' + escapeHtml(row.label) + '</span><strong class="mft-val mft-' + row.tone + '">' +
          formatInteger(row.value) +
          "</strong></div>";
      });
      sections +=
        '<div class="mft-row mft-row-total"><span>Total units</span><strong class="mft-val">' +
        formatInteger(totalUnits) +
        "</strong></div></div>";
    } else if (facility.stock != null || facility.usage != null) {
      sections += '<div class="mft-section"><div class="mft-section-title">Inventory summary</div>';
      if (facility.stock != null) {
        sections +=
          '<div class="mft-row"><span>Current stock</span><strong class="mft-val">' +
          formatUnits(facility.stock) +
          "</strong></div>";
      }
      if (facility.usage != null) {
        sections +=
          '<div class="mft-row"><span>Usage today</span><strong class="mft-val">' +
          formatUnits(facility.usage) +
          "</strong></div>";
      }
      if (facility.utilization != null) {
        sections +=
          '<div class="mft-row"><span>Utilization</span><strong class="mft-val">' +
          formatPercent(facility.utilization) +
          "</strong></div>";
      }
      sections += "</div>";
    } else {
      var waitingForSim = asArray(data && data.mapCenters).length === 0;
      sections +=
        '<div class="mft-section mft-empty">' +
        '<p class="mft-empty-text">' +
        escapeHtml(
          waitingForSim
            ? "Start or resume the simulation to load live component inventory."
            : "No simulator center is linked to this map site yet.",
        ) +
        "</p></div>";
    }

    if (stats.donated != null || stats.transfused != null || stats.expired != null) {
      sections += '<div class="mft-section"><div class="mft-section-title">Operations</div>';
      if (stats.donated != null) {
        sections +=
          '<div class="mft-row"><span>Donated</span><strong class="mft-val mft-positive">' +
          formatInteger(stats.donated) +
          "</strong></div>";
      }
      if (stats.transfused != null) {
        sections +=
          '<div class="mft-row"><span>Transfused</span><strong class="mft-val mft-transfused">' +
          formatInteger(stats.transfused) +
          "</strong></div>";
      }
      if (stats.expired != null) {
        sections +=
          '<div class="mft-row"><span>Expired</span><strong class="mft-val mft-muted">' +
          formatInteger(stats.expired) +
          "</strong></div>";
      }
      if (stats.rejected != null) {
        sections +=
          '<div class="mft-row"><span>Rejected</span><strong class="mft-val">' +
          formatInteger(stats.rejected) +
          "</strong></div>";
      }
      if (stats.no_show != null) {
        sections +=
          '<div class="mft-row"><span>No-show</span><strong class="mft-val">' +
          formatInteger(stats.no_show) +
          "</strong></div>";
      }
      sections += "</div>";
    }

    if (alerts.length) {
      sections += '<div class="mft-section mft-alerts"><div class="mft-section-title">Shortages &amp; alerts</div>';
      alerts.slice(0, 4).forEach(function (alert) {
        var alertClass = alert.type === "critical" || facility.critical ? " is-critical" : "";
        sections +=
          '<div class="mft-alert' + alertClass + '">' +
          escapeHtml(alert.text || alert.title || "Operational alert") +
          "</div>";
      });
      sections += "</div>";
    } else if (facility.critical) {
      sections +=
        '<div class="mft-section mft-alerts">' +
        '<div class="mft-alert is-critical">Critical shortage flagged for this facility</div>' +
        "</div>";
    }

    return sections;
  }

  var mapFacilityTooltipState = {
    bound: false,
    activeId: null,
    hideTimer: null,
  };

  function hideMapFacilityTooltip() {
    var tooltip = document.getElementById("map-facility-tooltip");
    if (!tooltip) return;
    tooltip.classList.remove("is-visible");
    tooltip.hidden = true;
    tooltip.innerHTML = "";
    mapFacilityTooltipState.activeId = null;
  }

  function positionMapFacilityTooltip(anchorEl) {
    var tooltip = document.getElementById("map-facility-tooltip");
    if (!tooltip || !anchorEl) return;
    var rect = anchorEl.getBoundingClientRect();
    var top = Math.max(12, rect.top - 10);
    var left = rect.left + rect.width / 2;
    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }

  function showMapFacilityTooltip(facility, anchorEl) {
    var tooltip = document.getElementById("map-facility-tooltip");
    if (!tooltip || !facility || !anchorEl) return;
    clearTimeout(mapFacilityTooltipState.hideTimer);
    mapFacilityTooltipState.activeId = facility.id;
    tooltip.innerHTML = buildFacilityTooltipHtml(facility, currentData());
    tooltip.hidden = false;
    positionMapFacilityTooltip(anchorEl);
    tooltip.classList.add("is-visible");
  }

  function scheduleHideMapFacilityTooltip() {
    clearTimeout(mapFacilityTooltipState.hideTimer);
    mapFacilityTooltipState.hideTimer = setTimeout(hideMapFacilityTooltip, 120);
  }

  function bindMapFacilityTooltips() {
    var root = $("dt-iso-map");
    if (!root || mapFacilityTooltipState.bound) return;
    mapFacilityTooltipState.bound = true;

    root.addEventListener("mouseover", function (event) {
      var facilityEl = event.target.closest(".facility-overlay .facility");
      if (!facilityEl) return;
      if (
        !facilityEl.classList.contains("facility-hospital") &&
        !facilityEl.classList.contains("facility-bloodbank")
      ) {
        return;
      }
      var facilityId = facilityEl.getAttribute("data-facility-id");
      if (!facilityId || facilityId === mapFacilityTooltipState.activeId) return;
      var facilities = resolveIsoFacilitiesWithCenters(currentData());
      var facility = null;
      for (var i = 0; i < facilities.length; i++) {
        if (String(facilities[i].id) === facilityId) {
          facility = facilities[i];
          break;
        }
      }
      if (!facility) return;
      showMapFacilityTooltip(facility, facilityEl);
    });

    root.addEventListener("mouseout", function (event) {
      var facilityEl = event.target.closest(".facility-overlay .facility");
      if (!facilityEl) return;
      var related = event.relatedTarget;
      if (related && facilityEl.contains(related)) return;
      if (related && related.closest && related.closest(".facility-overlay .facility") === facilityEl) return;
      scheduleHideMapFacilityTooltip();
    });

    window.addEventListener("scroll", hideMapFacilityTooltip, true);
    window.addEventListener("resize", hideMapFacilityTooltip);
  }

  function isoFacilityGlyph(type) {
    var visual = facilityVisualType(type);
    if (visual === "hospital") return "H";
    if (visual === "bloodbank") return "B";
    if (visual === "lab") return "L";
    return "D";
  }

  function isoFacilityLabel(name) {
    var raw = String(name || "");
    if (raw.length > 24) return raw.slice(0, 22) + "…";
    return raw;
  }

  function createBackgroundBuilding(x, y, w, h, depth, opacity) {
    var hw = w / 2;
    var lift = depth || h * 0.85;
    return (
      '<g class="bg-building" transform="translate(' + x + " " + y + ')" opacity="' + (opacity == null ? 0.5 : opacity) + '">' +
      '<ellipse class="bg-building-shadow" cx="' + hw.toFixed(1) + '" cy="' + (h * 0.5).toFixed(1) + '" rx="' + (hw * 0.68).toFixed(1) + '" ry="' + (h * 0.12).toFixed(1) + '"/>' +
      '<polygon class="bg-building-roof" points="0,' + (-lift * 0.52).toFixed(1) + " " + hw.toFixed(1) + "," + (-lift * 0.84).toFixed(1) + " " + w.toFixed(1) + "," + (-lift * 0.52).toFixed(1) + " " + hw.toFixed(1) + "," + (-lift * 0.18).toFixed(1) + '"/>' +
      '<polygon class="bg-building-left" points="0,' + (-lift * 0.52).toFixed(1) + " " + hw.toFixed(1) + "," + (-lift * 0.18).toFixed(1) + " " + hw.toFixed(1) + "," + (h * 0.32).toFixed(1) + " 0," + (h * 0.18).toFixed(1) + '"/>' +
      '<polygon class="bg-building-right" points="' + hw.toFixed(1) + "," + (-lift * 0.18).toFixed(1) + " " + w.toFixed(1) + "," + (-lift * 0.52).toFixed(1) + " " + w.toFixed(1) + "," + (h * 0.18).toFixed(1) + " " + hw.toFixed(1) + "," + (h * 0.32).toFixed(1) + '"/>' +
      "</g>"
    );
  }

  function facilityBadgeMarkup(style) {
    if (style.badgeKind === "flask") {
      return '<path class="facility-badge-icon" d="M -3.5 -7 H 3.5 V -2.5 L 0 5 L -3.5 -2.5 Z"/>';
    }
    if (style.badgeKind === "droplet") {
      return '<path class="facility-badge-icon" d="M0 -7.5 C3.2 -2.5 5.5 1.5 0 8.5 C-5.5 1.5 -3.2 -2.5 0 -7.5Z"/>';
    }
    return '<text class="facility-badge-text" text-anchor="middle" y="4">' + escapeHtml(style.badge || "H") + "</text>";
  }

  function facilityWindowGrid(hw, h, cols, rows, startX, startY, cellW, cellH, gapX, gapY) {
    var windows = "";
    for (var row = 0; row < rows; row++) {
      for (var col = 0; col < cols; col++) {
        windows +=
          '<rect class="building-window" x="' + (startX + col * (cellW + gapX)).toFixed(1) + '" y="' + (startY + row * (cellH + gapY)).toFixed(1) + '" width="' + cellW + '" height="' + cellH + '" rx="1"/>';
      }
    }
    return windows;
  }

  function createFacilityBuilding(x, y, type, scale) {
    type = facilityVisualType(type);
    var style = FACILITY_STYLES[type] || FACILITY_STYLES.hospital;
    scale = scale || style.scale || 1;
    var w = 98 * scale;
    var h = 68 * scale;
    var hw = w / 2;
    var depth = h * 0.92;
    var extras =
      (type === "lab"
        ? '<rect class="building-lab-tower" x="' + (hw - 10).toFixed(1) + '" y="' + (-depth * 1.12).toFixed(1) + '" width="20" height="22" rx="4" style="fill:' + style.accent + '"/>' +
          '<rect class="building-lab-cap" x="' + (hw - 6).toFixed(1) + '" y="' + (-depth * 1.2).toFixed(1) + '" width="12" height="6" rx="2" style="fill:' + style.roof + '"/>'
        : "") +
      (type === "bloodbank"
        ? '<rect class="building-bank-sign" x="' + (hw - 18).toFixed(1) + '" y="' + (-depth * 0.82).toFixed(1) + '" width="36" height="10" rx="5" style="fill:' + style.accent + '"/>'
        : "") +
      (type === "donation"
        ? '<rect class="building-canopy" x="' + (hw - 22).toFixed(1) + '" y="' + (-depth * 0.72).toFixed(1) + '" width="44" height="8" rx="4" style="fill:' + style.accent + '" opacity="0.85"/>'
        : "");

    return (
      '<g class="facility-building" transform="translate(' + x + " " + y + ')">' +
      '<ellipse class="building-shadow" cx="' + hw.toFixed(1) + '" cy="' + (h * 0.58).toFixed(1) + '" rx="' + (hw * 0.95).toFixed(1) + '" ry="' + (h * 0.18).toFixed(1) + '"/>' +
      '<polygon class="building-roof" points="0,' + (-depth * 0.6).toFixed(1) + " " + hw.toFixed(1) + "," + (-depth * 0.98).toFixed(1) + " " + w.toFixed(1) + "," + (-depth * 0.6).toFixed(1) + " " + hw.toFixed(1) + "," + (-depth * 0.22).toFixed(1) + '" style="fill:' + style.roof + '"/>' +
      '<polygon class="building-left" points="0,' + (-depth * 0.6).toFixed(1) + " " + hw.toFixed(1) + "," + (-depth * 0.22).toFixed(1) + " " + hw.toFixed(1) + "," + (h * 0.46).toFixed(1) + " 0," + (h * 0.3).toFixed(1) + '" style="fill:' + style.left + '"/>' +
      '<polygon class="building-right" points="' + hw.toFixed(1) + "," + (-depth * 0.22).toFixed(1) + " " + w.toFixed(1) + "," + (-depth * 0.6).toFixed(1) + " " + w.toFixed(1) + "," + (h * 0.3).toFixed(1) + " " + hw.toFixed(1) + "," + (h * 0.46).toFixed(1) + '" style="fill:' + style.right + '"/>' +
      '<rect class="building-accent" x="8" y="' + (-depth * 0.5).toFixed(1) + '" width="8" height="' + (h * 0.68).toFixed(1) + '" rx="2" style="fill:' + style.accent + '"/>' +
      facilityWindowGrid(hw, h, 3, 3, hw + 8, -depth * 0.46, 8, 10, 6, 8) +
      '<rect class="building-entrance" x="' + (hw - 12).toFixed(1) + '" y="' + (h * 0.22).toFixed(1) + '" width="24" height="16" rx="3"/>' +
      extras +
      "</g>"
    );
  }

  function createRoadPath(d, type) {
    var roadClass = type === "secondary" ? "road-secondary" : "road-main";
    return '<path class="' + roadClass + '" d="' + d + '"/><path class="road-line" d="' + d + '"/>';
  }

  function createBridge(x, y, rotation, width) {
    width = width || 44;
    var half = width / 2;
    return (
      '<g class="river-bridge" transform="translate(' + x + " " + y + ") rotate(" + rotation + ')">' +
      '<rect x="' + (-half).toFixed(1) + '" y="-6" width="' + width + '" height="12" rx="3"/>' +
      '<path class="bridge-lane" d="M ' + (-half + 6).toFixed(1) + ' 0 L ' + (half - 6).toFixed(1) + ' 0"/>' +
      "</g>"
    );
  }

  function createVehicle(x, y, rotation, delay) {
    return (
      '<g class="vehicle" transform="translate(' + x + " " + y + ") rotate(" + rotation + ')" style="animation-delay:' + escapeHtml(delay || "0s") + '">' +
      '<ellipse class="vehicle-shadow" cx="0" cy="8" rx="12" ry="2.5"/>' +
      '<rect x="-12" y="-6" width="24" height="12" rx="3.5"/>' +
      '<rect x="2" y="-9" width="8" height="7" rx="2"/>' +
      '<circle cx="-6" cy="7" r="2"/>' +
      '<circle cx="7" cy="7" r="2"/>' +
      '<path d="M -2.5 -1.5 H 2.5 M 0 -4.5 V -0.5"/>' +
      "</g>"
    );
  }

  function createRouteVehicle(routeId, duration, delay) {
    routeId = escapeHtml(routeId);
    duration = escapeHtml(duration || "10s");
    delay = escapeHtml(delay || "0s");

    return (
      '<g class="vehicle route-vehicle">' +
      '<ellipse class="vehicle-shadow" cx="0" cy="9" rx="14" ry="5"/>' +
      '<rect x="-14" y="-8" width="28" height="16" rx="5"/>' +
      '<rect x="3" y="-13" width="10" height="8" rx="3"/>' +
      '<path d="M -4 0 H 4 M 0 -4 V 4"/>' +
      '<circle cx="-7" cy="9" r="2.5"/>' +
      '<circle cx="8" cy="9" r="2.5"/>' +
      '<animateMotion dur="' + duration + '" begin="' + delay + '" repeatCount="indefinite" rotate="auto">' +
      '<mpath href="#' + routeId + '"/>' +
      "</animateMotion>" +
      "</g>"
    );
  }

  function createFacilityStatusOverlay(facility) {
    var type = facilityVisualType(facility.type || "hospital");
    var name = escapeHtml(isoFacilityLabel(facility.name || facility.label || "Facility"));
    var facilityId = escapeHtml(facility.id || "");
    var x = asNumber(facility.x, 0);
    var y = asNumber(facility.y, 0);
    var critical = facility.critical ? " is-critical" : "";
    var labelWidth = Math.max(86, Math.min(170, name.length * 7 + 34));
    var detailClass = facilityShowsDetailTooltip(facility.type) ? " has-detail-tooltip" : "";

    return (
      '<g class="facility facility-' + type + critical + detailClass + '" data-facility-id="' + facilityId + '" transform="translate(' + x + " " + y + ')">' +
      '<circle class="facility-hit-area" r="30" fill="transparent"/>' +
      '<title>' + escapeHtml(facility.name || "Facility") + "</title>" +
      '<circle class="facility-status-glow" r="24"/>' +
      '<circle class="facility-status-dot" r="8"/>' +
      '<rect class="facility-label-bg" x="16" y="-15" width="' + labelWidth + '" height="30" rx="15"/>' +
      '<text class="facility-label-text" x="31" y="5">' + name + "</text>" +
      "</g>"
    );
  }

  function buildStaticMapImageBase() {
    return (
      '<div class="static-map-base" data-layer-group="buildings" aria-hidden="true">' +
      '<img class="day-layer" src="' + ISO_MAP_DAY_SRC + '" alt="" draggable="false" />' +
      '<img class="night-layer" src="' + ISO_MAP_NIGHT_SRC + '" alt="" draggable="false" />' +
      "</div>"
    );
  }

  // Continuous day/night factor for the isometric map. 1 = full day, 0 = full
  // night. We ease the displayed value toward the target each frame so the
  // transition is a smooth dawn/dusk blend instead of a hard swap.
  var mapDayFactor = 1;
  var mapDayFactorFrom = 1;
  var mapDayFactorTarget = 1;
  var mapDayFactorRaf = 0;
  var mapDayFactorAnimStart = 0;
  var mapDayFactorAnimDuration = 700;

  function applyMapDayFactorToDom() {
    var root = document.getElementById("dt-iso-map");
    if (root) {
      root.style.setProperty("--map-day-factor", mapDayFactor.toFixed(4));
    }
  }

  function easeMapDayFactor(now) {
    var start = mapDayFactorAnimStart || 0;
    var duration = Math.max(120, mapDayFactorAnimDuration);
    var elapsed = Math.max(0, (now || performance.now()) - start);
    var t = Math.min(1, elapsed / duration);
    // Smoothstep for a perceptually even dawn/dusk crossfade.
    var eased = t * t * (3 - 2 * t);
    mapDayFactor = mapDayFactorFrom + (mapDayFactorTarget - mapDayFactorFrom) * eased;
    applyMapDayFactorToDom();
    if (t < 1) {
      mapDayFactorRaf = requestAnimationFrame(easeMapDayFactor);
    } else {
      mapDayFactor = mapDayFactorTarget;
      mapDayFactorRaf = 0;
      applyMapDayFactorToDom();
    }
  }

  // Translate a simulation hour (elapsed hours) into a continuous daylight
  // factor in [0,1]. Day (06:00-20:00) = 1, night (22:00-04:00) = 0, with a
  // smooth dusk ramp 20:00->22:00 and dawn ramp 04:00->06:00.
  function hourToDayFactor(hour) {
    var h = ((Number(hour) % 24) + 24) % 24;
    var DAWN_START = 4, DAWN_END = 6, DUSK_START = 20, DUSK_END = 22;
    if (h >= DAWN_END && h < DUSK_START) return 1;            // full day
    if (h >= DUSK_END || h < DAWN_START) return 0;            // full night
    if (h >= DUSK_START) return 1 - (h - DUSK_START) / (DUSK_END - DUSK_START); // dusk
    return (h - DAWN_START) / (DAWN_END - DAWN_START);        // dawn
  }

  function setMapNightMode(isNight) {
    setMapDayFactor(isNight ? 0 : 1);
  }

  function setMapDayFactorFromHour(hour) {
    mapDayFactorFrom = mapDayFactor;
    mapDayFactorTarget = hourToDayFactor(hour);
    mapNightMode = mapDayFactorTarget < 0.5;
    mapDayFactorAnimStart = performance.now();
    if (mapDayFactorRaf) {
      cancelAnimationFrame(mapDayFactorRaf);
    }
    mapDayFactorRaf = requestAnimationFrame(easeMapDayFactor);
  }

  function setMapDayFactor(factor) {
    mapNightMode = factor < 0.5;
    mapDayFactorFrom = mapDayFactor;
    mapDayFactorTarget = Math.max(0, Math.min(1, factor));
    mapDayFactorAnimStart = performance.now();
    if (mapDayFactorRaf) {
      cancelAnimationFrame(mapDayFactorRaf);
    }
    mapDayFactorRaf = requestAnimationFrame(easeMapDayFactor);
  }

  function bindMapCoordinateLogger() {
    var map = document.querySelector(".route-overlay");
    if (!map) return;

    map.addEventListener("click", function (event) {
      var rect = map.getBoundingClientRect();
      var x = ((event.clientX - rect.left) / rect.width) * ISO_MAP_WIDTH;
      var y = ((event.clientY - rect.top) / rect.height) * ISO_MAP_HEIGHT;
      console.log("Map coordinate:", Math.round(x), Math.round(y));
    });
  }

  function createFacility(config) {
    config = config || {};
    var type = facilityVisualType(config.type || "hospital");
    var style = FACILITY_STYLES[type] || FACILITY_STYLES.hospital;
    var x = asNumber(config.x, 0);
    var y = asNumber(config.y, 0);
    var label = isoFacilityLabel(config.label || config.name || "Facility");
    var id = escapeHtml(config.id || label);
    var scale = style.scale || 1;
    var w = 98 * scale;
    var h = 68 * scale;
    var hw = w / 2;
    var depth = h * 0.92;
    var subtitle = config.stock != null ? "Stock: " + formatUnits(config.stock) : label;
    var labelWidth = Math.min(118, Math.max(68, label.length * 4.8 + 16));
    var criticalClass = config.critical ? " is-critical" : "";

    var building =
      '<g class="facility-site facility-site-' + type + criticalClass + '" transform="translate(' + x + " " + y + ')" data-facility="' + id + '">' +
      '<title>' + escapeHtml(subtitle) + "</title>" +
      createFacilityBuilding(0, 0, type, scale) +
      "</g>";

    var overlay =
      '<g class="facility facility-' + type + " dt-facility-marker" + criticalClass + '" transform="translate(' + x + " " + y + ')" data-facility="' + id + '">' +
      '<title>' + escapeHtml(subtitle) + "</title>" +
      '<ellipse class="facility-soft-glow" cx="' + hw.toFixed(1) + '" cy="' + (-depth * 0.78).toFixed(1) + '" rx="18" ry="6" style="fill:' + style.accent + '"/>' +
      '<g class="facility-badge" transform="translate(' + hw.toFixed(1) + " " + (-depth * 0.9).toFixed(1) + ') scale(0.72)">' +
      '<circle class="facility-badge-core" r="10" style="fill:' + style.accent + '"/>' +
      facilityBadgeMarkup(style) +
      "</g>" +
      '<g class="facility-label" transform="translate(' + (hw + 14).toFixed(1) + " " + (-depth * 0.94).toFixed(1) + ')">' +
      '<rect class="facility-label-bg" width="' + labelWidth.toFixed(1) + '" height="17" rx="8.5" style="fill:' + style.accent + '"/>' +
      '<text class="facility-label-text" x="8" y="12">' + escapeHtml(label) + "</text>" +
      "</g>" +
      "</g>";

    return { building: building, overlay: overlay };
  }

  function svgCityBlock(cx, cy, w, h) {
    var hw = w / 2;
    var hh = h / 2;
    return (
      '<polygon class="city-block" points="' +
      (cx - hw).toFixed(1) + "," + cy.toFixed(1) + " " +
      cx.toFixed(1) + "," + (cy - hh).toFixed(1) + " " +
      (cx + hw).toFixed(1) + "," + cy.toFixed(1) + " " +
      cx.toFixed(1) + "," + (cy + hh).toFixed(1) +
      '"/>'
    );
  }

  function svgGridLayer() {
    var lines = "";
    for (var i = -1; i <= 14; i++) {
      var x = i * 96;
      lines += '<line class="iso-grid-line" x1="' + x + '" y1="-40" x2="' + (x - 280) + '" y2="700"/>';
      lines += '<line class="iso-grid-line" x1="-40" y1="' + (i * 58) + '" x2="1240" y2="' + (i * 58 - 200) + '"/>';
    }
    return '<g class="layer-grid" data-layer-group="buildings">' + lines + "</g>";
  }

  function svgRiverLayer() {
    var bridges = ISO_MAP_BRIDGES.map(function (bridge) {
      return createBridge(bridge.x, bridge.y, bridge.rotation, bridge.width);
    }).join("");
    return (
      '<g class="layer-river" data-layer-group="buildings">' +
      '<path class="river-glow" d="' + ISO_MAP_RIVER + '"/>' +
      '<path class="river" d="' + ISO_MAP_RIVER + '"/>' +
      bridges +
      "</g>"
    );
  }

  function svgRoadLayer() {
    var paths = ISO_MAP_ROAD_PATHS.map(function (road) {
      return createRoadPath(road.d, road.type);
    }).join("");
    var roundabouts = ISO_MAP_ROUNDABOUTS.map(function (node) {
      return (
        '<circle class="road-roundabout" cx="' + node.x + '" cy="' + node.y + '" r="' + node.r + '"/>' +
        '<circle class="road-roundabout-core" cx="' + node.x + '" cy="' + node.y + '" r="' + (node.r * 0.42).toFixed(1) + '"/>'
      );
    }).join("");
    return '<g class="layer-roads" data-layer-group="buildings">' + paths + roundabouts + "</g>";
  }

  function svgCityBlocksLayer() {
    var blocks = ISO_MAP_CITY_BLOCKS.map(function (row) {
      return svgCityBlock(row[0], row[1], row[2], row[3]);
    }).join("");
    return '<g class="layer-blocks" data-layer-group="buildings">' + blocks + "</g>";
  }

  function svgCityBuildingsLayer() {
    var buildings = ISO_MAP_BACKGROUND_BUILDINGS.map(function (row) {
      return createBackgroundBuilding(row.x, row.y, row.w, row.h, row.depth, row.opacity);
    }).join("");
    var trees = ISO_MAP_TREES.map(function (tree) {
      return (
        '<g class="iso-tree-cluster" transform="translate(' + tree[0] + " " + tree[1] + ')">' +
        '<circle class="iso-tree" cx="0" cy="0" r="' + tree[2] + '"/>' +
        '<circle class="iso-tree iso-tree-accent" cx="-3" cy="-2" r="' + (tree[2] * 0.55).toFixed(1) + '"/>' +
        "</g>"
      );
    }).join("");
    return '<g class="layer-city-buildings" data-layer-group="buildings">' + buildings + trees + "</g>";
  }

  function svgStaticFacilityBuildingsLayer() {
    var buildingParts = "";
    ISO_MAP_FACILITIES.forEach(function (facility) {
      var parts = createFacility({
        x: facility.x,
        y: facility.y,
        type: facility.type,
        label: facility.name,
        name: facility.name,
        id: facility.id,
      });
      buildingParts += parts.building;
    });
    return '<g class="layer-facility-buildings">' + buildingParts + "</g>";
  }

  function svgFacilityOverlayLayer(facilities) {
    var overlayParts = facilities.map(function (facility) {
      return createFacilityStatusOverlay(facility);
    }).join("");

    return '<g class="layer-facility-overlays" data-layer-group="facilities">' + overlayParts + "</g>";
  }

  function svgRouteLayer(routes) {
    var paths = routes.map(function (route, index) {
      var type = cssToken(route.type || route.color || "blue", "blue");
      var d = escapeHtml(route.d || "");
      var routeId = escapeHtml(route.id || ("route-path-" + index));

      return (
        '<path class="route-glow route-' + type + '-glow" d="' + d + '"/>' +
        '<path id="' + routeId + '" class="route-line route-' + type + '" d="' + d + '"/>'
      );
    }).join("");

    return '<g class="layer-routes" data-layer-group="routes">' + paths + "</g>";
  }

  function svgVehicleLayer(vehicles) {
    return (
      '<g class="layer-vehicles" data-layer-group="vehicles">' +
      vehicles.map(function (vehicle, index) {
        return createRouteVehicle(vehicle.routeId, vehicle.duration || "10s", vehicle.delay || (index * 0.4) + "s");
      }).join("") +
      "</g>"
    );
  }

  function svgDeliveryGlowLayer() {
    var glows = ISO_MAP_DELIVERY_GLOWS.map(function (glow) {
      return '<circle class="delivery-glow" cx="' + glow.x + '" cy="' + glow.y + '" r="' + glow.r + '" style="stroke:' + glow.color + '"/>';
    }).join("");
    return '<g class="layer-delivery-glows" data-layer-group="routes">' + glows + "</g>";
  }

  function svgBaseDefs() {
    return (
      "<defs>" +
      '<pattern id="iso-map-grid" width="52" height="52" patternUnits="userSpaceOnUse" patternTransform="rotate(32)">' +
      '<path d="M 52 0 L 0 0 0 52" fill="none" stroke="rgba(148,163,184,0.08)" stroke-width="1"/>' +
      "</pattern>" +
      '<radialGradient id="map-center-glow" cx="50%" cy="44%" r="52%">' +
      '<stop offset="0%" stop-color="#ffffff" stop-opacity="0.42"/>' +
      '<stop offset="55%" stop-color="#f4f9ff" stop-opacity="0.12"/>' +
      '<stop offset="100%" stop-color="#eef6ff" stop-opacity="0"/>' +
      "</radialGradient>" +
      '<radialGradient id="map-vignette" cx="50%" cy="46%" r="72%">' +
      '<stop offset="0%" stop-color="#ffffff" stop-opacity="0"/>' +
      '<stop offset="68%" stop-color="#eef6ff" stop-opacity="0"/>' +
      '<stop offset="100%" stop-color="#b8d4ef" stop-opacity="0.34"/>' +
      "</radialGradient>" +
      "</defs>"
    );
  }

  function svgOverlayDefs() {
    return (
      "<defs>" +
      '<filter id="facility-glow-blur" x="-50%" y="-50%" width="200%" height="200%">' +
      '<feGaussianBlur stdDeviation="4" result="blur"/>' +
      "</filter>" +
      "</defs>"
    );
  }

  function buildIsoCityBase() {
    return (
      '<svg class="iso-city-base" viewBox="' + ISO_MAP_VIEWBOX + '" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" role="img" aria-hidden="true">' +
      svgBaseDefs() +
      '<rect class="iso-map-bg" width="1200" height="620"/>' +
      '<rect class="iso-map-grid" width="1200" height="620" fill="url(#iso-map-grid)"/>' +
      '<rect class="map-center-glow" width="1200" height="620" fill="url(#map-center-glow)" pointer-events="none"/>' +
      svgGridLayer() +
      svgRiverLayer() +
      svgRoadLayer() +
      svgCityBlocksLayer() +
      svgCityBuildingsLayer() +
      '<rect class="map-vignette" width="1200" height="620" fill="url(#map-vignette)" pointer-events="none"/>' +
      "</svg>"
    );
  }

  function buildIsoFacilityBuildingsOverlay() {
    return (
      '<svg class="facility-buildings-overlay" viewBox="' + ISO_MAP_VIEWBOX + '" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      svgStaticFacilityBuildingsLayer() +
      "</svg>"
    );
  }

  function buildIsoRouteOverlay() {
    return (
      '<svg class="route-overlay" viewBox="' + ISO_MAP_VIEWBOX + '" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      svgRouteLayer(ISO_MAP_ROUTES) +
      svgVehicleLayer(ISO_MAP_VEHICLES) +
      "</svg>"
    );
  }

  function buildIsoFacilityOverlay(data) {
    var facilities = resolveIsoFacilitiesWithCenters(data);
    return (
      '<div class="facility-overlay" aria-hidden="true">' +
      '<svg class="facility-overlay-svg" viewBox="' + ISO_MAP_VIEWBOX + '" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">' +
      svgOverlayDefs() +
      svgFacilityOverlayLayer(facilities) +
      "</svg>" +
      "</div>"
    );
  }

  function buildIsoMapSvg(data) {
    return (
      buildStaticMapImageBase() +
      buildIsoRouteOverlay() +
      buildIsoFacilityOverlay(data)
    );
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function routeEndpoints(route) {
    var numbers = String((route && route.d) || "").match(/-?\d+(?:\.\d+)?/g) || [];
    if (numbers.length < 4) return null;
    var x1 = asNumber(numbers[0], 0);
    var y1 = asNumber(numbers[1], 0);
    var x2 = asNumber(numbers[numbers.length - 2], x1);
    var y2 = asNumber(numbers[numbers.length - 1], y1);
    return { x1: x1, y1: y1, x2: x2, y2: y2 };
  }

  function routeVehiclePoint(route, index) {
    var midpoint = route && route.midpoint;
    var x = midpoint && midpoint.x != null ? asNumber(midpoint.x, 50) : null;
    var y = midpoint && midpoint.y != null ? asNumber(midpoint.y, 50) : null;
    var rotation = route && route.rotation;
    if (x == null || y == null || !rotation) {
      var endpoints = routeEndpoints(route);
      if (endpoints) {
        x = ((endpoints.x1 + endpoints.x2) / 2) / 10;
        y = ((endpoints.y1 + endpoints.y2) / 2) / 6;
        if (!rotation) {
          rotation = (Math.atan2(endpoints.y2 - endpoints.y1, endpoints.x2 - endpoints.x1) * 180 / Math.PI).toFixed(1) + "deg";
        }
      }
    }
    return {
      x: clamp(asNumber(x, 28 + index * 8) + ((index % 2) * 2 - 1), 8, 92),
      y: clamp(asNumber(y, 42 + index * 7) + ((index % 3) - 1), 10, 88),
      rotation: rotation || "0deg",
    };
  }

  function withBloodColor(row) {
    var colors = {
      "O+": "#ef4444",
      "O-": "#f59e0b",
      "A+": "#2563eb",
      "A-": "#38bdf8",
      "B+": "#22c55e",
      "B-": "#14b8a6",
      "AB+": "#8b5cf6",
      "AB-": "#a855f7",
    };
    var copy = Object.assign({}, row || {});
    copy.type = copy.type || "Unknown";
    copy.color = copy.color || colors[copy.type] || "#94a3b8";
    return copy;
  }

  function routePathBetween(a, b, offset) {
    var x1 = isoMapScaleX(a.x);
    var y1 = isoMapScaleY(a.y);
    var x2 = isoMapScaleX(b.x);
    var y2 = isoMapScaleY(b.y);
    var cx1 = x1 + (x2 - x1) * 0.28;
    var cy1 = y1 - 52 + offset;
    var cx2 = x1 + (x2 - x1) * 0.72;
    var cy2 = y2 + 42 - offset;
    return (
      "M " + x1.toFixed(1) + " " + y1.toFixed(1) +
      " C " + cx1.toFixed(1) + " " + cy1.toFixed(1) +
      " " + cx2.toFixed(1) + " " + cy2.toFixed(1) +
      " " + x2.toFixed(1) + " " + y2.toFixed(1)
    );
  }

  function routeMetaBetween(a, b) {
    var x1 = asNumber(a.x, 50);
    var y1 = asNumber(a.y, 50);
    var x2 = asNumber(b.x, 50);
    var y2 = asNumber(b.y, 50);
    var rotation = (Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI).toFixed(1) + "deg";
    return {
      sourceFacilityId: a.id,
      targetFacilityId: b.id,
      sourceName: a.name || a.id || "",
      targetName: b.name || b.id || "",
      midpoint: {
        x: Number(((x1 + x2) / 2).toFixed(1)),
        y: Number(((y1 + y2) / 2).toFixed(1)),
      },
      rotation: rotation,
      pressure: asNumber(b.utilization, 0),
    };
  }

  function derivedRoutes(facilities) {
    facilities = asArray(facilities);
    if (facilities.length < 2) return [];
    var hub = facilities.slice().sort(function (a, b) {
      return asNumber(b.stock, 0) - asNumber(a.stock, 0);
    })[0];
    return facilities
      .filter(function (facility) { return facility.id !== hub.id; })
      .slice(0, 8)
      .map(function (facility, index) {
        var palette = facility.critical ? "red" : ["blue", "green", "purple"][index % 3];
        return {
          sourceFacilityId: hub.id,
          targetFacilityId: facility.id,
          sourceName: hub.name || hub.id || "",
          targetName: facility.name || facility.id || "",
          color: palette,
          d: routePathBetween(hub, facility, (index % 3) * 22),
          animated: true,
          derivedFrom: "dashboard_facilities",
          midpoint: routeMetaBetween(hub, facility).midpoint,
          rotation: routeMetaBetween(hub, facility).rotation,
          pressure: routeMetaBetween(hub, facility).pressure,
        };
      });
  }

  function derivedVehicles(routes, deliveries) {
    var total = asNumber(deliveries && deliveries.total, 0);
    if (!total) return [];
    return routes.slice(0, Math.min(routes.length, total, 5)).map(function (_, index) {
      var point = routeVehiclePoint(routes[index], index);
      return {
        x: point.x,
        y: point.y,
        rotation: point.rotation,
        delay: (index * 0.8).toFixed(1) + "s",
        type: index % 2 ? "van" : "ambulance",
      };
    });
  }

  var mapState = {
    zoom: 1,
    mode3d: false,
    layers: {
      buildings: true,
      facilities: true,
      routes: true,
      vehicles: true,
      live: false,
    },
    controlsBound: false,
  };

  /* ── MapPanel ── */
  function renderMapPanel(container) {
    if (!container) return;
    var data = currentData();
    var weather = escapeHtml(data.weather || "14°C Light rain");

    container.innerHTML =
      '<div class="dt-iso-viewport" id="dt-iso-viewport">' +
      '<section class="iso-map-card" id="dt-iso-map-section" aria-label="Isometric blood network map">' +
      '<div class="map-top-controls">' +
      '<button class="map-chip map-chip-left" id="iso-map-layers-btn" type="button" aria-expanded="false" aria-controls="dt-layers-panel">' +
      icon("layers", 16) + " Layers</button>" +
      '<div class="map-chip weather-chip" id="iso-weather-chip">' + weather + "</div>" +
      "</div>" +
      buildStaticMapImageBase() +
      buildIsoRouteOverlay() +
      buildIsoFacilityOverlay(data) +
      '<div class="map-right-controls map-controls">' +
      '<button id="iso-map-3d-btn" type="button" aria-label="Toggle flat view">2D</button>' +
      '<button id="iso-map-locate-btn" type="button" aria-label="Locate">' + icon("crosshair", 16) + "</button>" +
      '<button id="iso-map-zoom-in" type="button" aria-label="Zoom in">+</button>' +
      '<button id="iso-map-zoom-out" type="button" aria-label="Zoom out">−</button>' +
      "</div>" +
      '<div class="map-facility-tooltip" id="map-facility-tooltip" role="tooltip" hidden></div>' +
      "</section>" +
      "</div>";

    applyMapState();
    // Re-apply the current day/night factor to the freshly-rendered map so a
    // re-render never briefly flashes the wrong time-of-day layer.
    var root = document.getElementById("dt-iso-map");
    if (root) root.style.setProperty("--map-day-factor", mapDayFactor.toFixed(3));
    bindMapCoordinateLogger();
    hideMapFacilityTooltip();
  }

  function applyMapState() {
    var panel = $("dt-map-panel");
    var viewport = $("dt-iso-viewport");
    if (panel) {
      panel.classList.remove("dt-map-3d");
    }
    if (viewport) {
      viewport.style.setProperty("--dt-map-zoom", mapState.zoom.toFixed(2));
    }
    document.documentElement.style.setProperty(
      "--runtime-map-scale",
      mapState.zoom.toFixed(2),
    );

    Object.keys(mapState.layers).forEach(function (key) {
      var visible = mapState.layers[key];
      if (key === "live") {
        document.body.classList.toggle("dt-map-live-mode", visible);
        return;
      }
      if (key === "facilities") {
        var facilityOverlay = document.querySelector(".facility-overlay");
        if (facilityOverlay) facilityOverlay.style.display = visible ? "" : "none";
      }
      if (key === "routes" || key === "vehicles") {
        var routeOverlay = document.querySelector(".route-overlay");
        if (routeOverlay && !mapState.layers.routes && !mapState.layers.vehicles) {
          routeOverlay.style.display = "none";
        } else if (routeOverlay) {
          routeOverlay.style.display = "";
        }
      }
      var nodes = document.querySelectorAll('[data-layer-group="' + key + '"]');
      for (var i = 0; i < nodes.length; i++) {
        nodes[i].style.display = visible ? "" : "none";
      }
    });

    ["dt-map-3d-btn", "iso-map-3d-btn"].forEach(function (id) {
      var btn3d = $(id);
      if (!btn3d) return;
      btn3d.classList.toggle("active", !mapState.mode3d);
      btn3d.setAttribute("aria-pressed", mapState.mode3d ? "false" : "true");
      btn3d.textContent = "2D";
    });

    var layerChecks = document.querySelectorAll("#dt-layers-panel [data-layer]");
    for (var j = 0; j < layerChecks.length; j++) {
      var layerKey = layerChecks[j].getAttribute("data-layer");
      layerChecks[j].checked = !!mapState.layers[layerKey];
    }
  }

  function setMapZoom(nextZoom) {
    mapState.zoom = Math.max(0.85, Math.min(1.45, nextZoom));
    applyMapState();
  }

  function resetMapView() {
    mapState.zoom = 1;
    applyMapState();
    notify("Map view reset to network center.");
  }

  function notify(message) {
    if (typeof global.showToast === "function") {
      global.showToast({ description: message, type: "info", duration: 2400 });
      return;
    }
    console.info(message);
  }

  function dashboardNavTarget(key) {
    return {
      overview: "#dt-section-overview",
      map: "#dt-map-panel",
      inventory: "#dt-section-inventory",
      orders: "#dt-section-forecast",
      transports: "#dt-section-deliveries",
      facilities: "#dt-section-facilities",
      analytics: "#dt-section-analytics",
      alerts: "#dt-section-alerts",
    }[key || ""];
  }

  function focusDashboardSection(key) {
    if (key === "reports") {
      if (global.state && global.state.report && typeof global.showScreen === "function") {
        global.showScreen("report-screen");
        return true;
      }
      var fallback = document.querySelector("#dt-section-deliveries");
      if (fallback) fallback.scrollIntoView({ behavior: "smooth", block: "center" });
      notify("Report opens after the current run is stopped.");
      return true;
    }

    if (key === "settings") {
      var layersBtn = $("dt-layers-btn");
      var layersPanel = $("dt-layers-panel");
      var mapPanel = $("dt-map-panel");
      if (mapPanel) mapPanel.scrollIntoView({ behavior: "smooth", block: "center" });
      if (layersPanel && layersBtn) {
        layersPanel.removeAttribute("hidden");
        layersBtn.setAttribute("aria-expanded", "true");
      }
      return true;
    }

    var selector = dashboardNavTarget(key);
    var target = selector ? document.querySelector(selector) : null;
    if (!target) return false;
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("dt-section-pulse");
    global.setTimeout(function () {
      target.classList.remove("dt-section-pulse");
    }, 900);
    return true;
  }

  function bindControls() {
    if (mapState.controlsBound) return;
    mapState.controlsBound = true;

    var collapseBtn = $("dt-sidebar-collapse");
    if (collapseBtn) {
      collapseBtn.addEventListener("click", function () {
        document.body.classList.toggle("dt-sidebar-collapsed");
        var collapsed = document.body.classList.contains("dt-sidebar-collapsed");
        collapseBtn.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
        notify(collapsed ? "Sidebar collapsed." : "Sidebar expanded.");
      });
    }

    var navItems = document.querySelectorAll(".dt-sidebar .dt-nav-item:not(.dt-nav-collapse)");
    for (var n = 0; n < navItems.length; n++) {
      navItems[n].addEventListener("click", function (event) {
        event.stopPropagation();
        for (var i = 0; i < navItems.length; i++) navItems[i].classList.remove("active");
        event.currentTarget.classList.add("active");
        var navKey = event.currentTarget.getAttribute("data-dashboard-nav") || "";
        if (!focusDashboardSection(navKey)) notify("Dashboard section is not available.");
      });
    }

    var layersBtn = $("dt-layers-btn");
    var layersPanel = $("dt-layers-panel");
    if (layersBtn && layersPanel) {
      layersBtn.addEventListener("click", function (event) {
        event.stopPropagation();
        var open = layersPanel.hasAttribute("hidden");
        if (open) {
          layersPanel.removeAttribute("hidden");
          layersBtn.setAttribute("aria-expanded", "true");
        } else {
          layersPanel.setAttribute("hidden", "");
          layersBtn.setAttribute("aria-expanded", "false");
        }
      });
      document.addEventListener("click", function (event) {
        var isoLayers = $("iso-map-layers-btn");
        var insideLayers =
          layersPanel.contains(event.target) ||
          event.target === layersBtn ||
          layersBtn.contains(event.target) ||
          (isoLayers && (event.target === isoLayers || isoLayers.contains(event.target)));
        if (!insideLayers) {
          layersPanel.setAttribute("hidden", "");
          layersBtn.setAttribute("aria-expanded", "false");
          if (isoLayers) isoLayers.setAttribute("aria-expanded", "false");
        }
      });
      layersPanel.addEventListener("change", function (event) {
        var target = event.target;
        if (!target || !target.getAttribute) return;
        var layer = target.getAttribute("data-layer");
        if (!layer) return;
        mapState.layers[layer] = !!target.checked;
        applyMapState();
        notify((target.checked ? "Showing " : "Hiding ") + layer.replace("_", " ") + " layer.");
      });
    }

    function bindMapButton(id, handler) {
      var node = $(id);
      if (node) node.addEventListener("click", handler);
    }

    function toggleMapMode() {
      mapState.mode3d = false;
      applyMapState();
      notify("Flat illustration view enabled.");
    }

    bindMapButton("dt-map-3d-btn", toggleMapMode);
    bindMapButton("iso-map-3d-btn", toggleMapMode);
    bindMapButton("dt-map-locate-btn", resetMapView);
    bindMapButton("iso-map-locate-btn", resetMapView);
    bindMapButton("iso-map-zoom-in", function () { setMapZoom(mapState.zoom + 0.1); });
    bindMapButton("iso-map-zoom-out", function () { setMapZoom(mapState.zoom - 0.1); });

    var isoLayersBtn = $("iso-map-layers-btn");
    if (isoLayersBtn && layersPanel) {
      isoLayersBtn.addEventListener("click", function (event) {
        event.stopPropagation();
        var open = layersPanel.hasAttribute("hidden");
        if (open) {
          layersPanel.removeAttribute("hidden");
          isoLayersBtn.setAttribute("aria-expanded", "true");
          if (layersBtn) layersBtn.setAttribute("aria-expanded", "true");
        } else {
          layersPanel.setAttribute("hidden", "");
          isoLayersBtn.setAttribute("aria-expanded", "false");
          if (layersBtn) layersBtn.setAttribute("aria-expanded", "false");
        }
      });
    }

    var settingsBtn = $("dt-settings-btn");
    if (settingsBtn) {
      settingsBtn.addEventListener("click", function () {
        if (document.body.classList.contains("twin-active")) {
          document.body.classList.toggle("twin-details-open");
          notify("Twin details panel toggled.");
          return;
        }
        notify("Dashboard settings panel is not open in this view.");
      });
    }

    var notificationsBtn = $("dt-notifications-btn");
    if (notificationsBtn) {
      notificationsBtn.addEventListener("click", function () {
        var alertsCard = document.querySelector(".dt-right-sidebar .dt-card:last-child");
        if (alertsCard) alertsCard.scrollIntoView({ behavior: "smooth", block: "center" });
        notify(formatInteger(currentData().alertCount || asArray(currentData().alerts).length) + " active alerts require attention.");
      });
    }

    document.addEventListener("click", function (event) {
      var marker = event.target.closest && event.target.closest("g.dt-facility-marker, g.facility, g.facility-site");
      if (!marker) return;
      var titleNode = marker.querySelector("title");
      var name = (titleNode && titleNode.textContent) || marker.getAttribute("data-facility") || "Facility";
      notify("Selected facility: " + name);
    });
  }

  function onGameScreenShown() {
    renderDashboard();
    if (!dashboardLoaded && !dashboardLoading) {
      loadDashboardData();
    }
  }

  function trimForecastSeries(forecast, maxPoints) {
    forecast = forecast || {};
    var demand = asArray(forecast.demand);
    var supply = asArray(forecast.supply);
    var labels = asArray(forecast.labels);
    var cap = Math.max(1, asNumber(maxPoints, 24));
    if (demand.length <= cap) {
      return { demand: demand, supply: supply, labels: labels };
    }
    var start = demand.length - cap;
    return {
      demand: demand.slice(start),
      supply: supply.slice(start),
      labels: labels.slice(start),
    };
  }

  function resetForecastTickSeries() {
    var next = cloneDashboard(currentData());
    next.forecast = { demand: [], supply: [], labels: [] };
    dashboardData = normalizeDashboardData(next);
    renderForecastChart($("dt-forecast-svg"));
    renderForecastPredictionsTable([]);
  }

  function forecastStatusLine(panel) {
    panel = panel || {};
    var parts = [];
    if (panel.job_id) parts.push("Job: " + panel.job_id);
    if (panel.tick_number != null) parts.push("Tick #" + panel.tick_number);
    if (panel.simulated_hour != null) parts.push("Hour " + Math.round(panel.simulated_hour));
    if (panel.prediction_count != null) parts.push(panel.prediction_count + " rows");
    var warnings = asArray(panel.warnings);
    if (warnings.length) parts.push(warnings[0]);
    return parts.join(" · ") || "Waiting for twin predictions…";
  }

  function renderForecastPredictionsTable(rows) {
    var tbody = $("dt-forecast-predictions-tbody");
    if (!tbody) return;
    rows = asArray(rows);
    if (!rows.length) {
      tbody.innerHTML =
        '<tr><td colspan="3" class="dt-forecast-empty">No predictions for this tick</td></tr>';
      return;
    }
    tbody.innerHTML = rows
      .map(function (row) {
        row = row || {};
        var hospital = row.hospital_id || row.entity_id || "—";
        var blood = row.blood_type || "—";
        var value = formatUnits(row.predicted_value);
        return (
          "<tr><td>" +
          escapeHtml(hospital) +
          "</td><td>" +
          escapeHtml(blood) +
          "</td><td>" +
          escapeHtml(value) +
          "</td></tr>"
        );
      })
      .join("");
  }

  function populateForecastModelSelect(meta, selectedJobId) {
    var select = $("dt-forecast-model");
    if (!select) return null;
    meta = meta || {};
    var options = asArray(meta.options);
    forecastPanelRuntime.maxPoints = asNumber(meta.chart_max_points, 24);
    if (!meta.enabled || !options.length) {
      select.innerHTML = '<option value="">Forecast models unavailable</option>';
      select.disabled = true;
      return null;
    }
    select.disabled = false;
    var chosen =
      selectedJobId ||
      forecastPanelRuntime.jobId ||
      meta.default_job_id ||
      (options[0] && options[0].job_id) ||
      "";
    select.innerHTML = options
      .map(function (opt) {
        opt = opt || {};
        var id = String(opt.job_id || "");
        var label = opt.label || opt.model_id || id;
        return (
          '<option value="' +
          escapeHtml(id) +
          '">' +
          escapeHtml(label) +
          "</option>"
        );
      })
      .join("");
    select.value = chosen;
    forecastPanelRuntime.jobId = select.value || null;
    var chosenOpt = options.filter(function (opt) {
      return String((opt || {}).job_id || "") === String(select.value || "");
    })[0];
    if (chosenOpt) updateForecastLegend(chosenOpt.predicted_label, chosenOpt.actual_label);
    return select.value || null;
  }

  function mergeForecastTick(panel, options) {
    options = options || {};
    panel = panel || {};
    if (!panel.job_id && !asArray(panel.predictions).length && !panel.chart_point) {
      return;
    }
    if (options.resetSeries) {
      resetForecastTickSeries();
    }
    if (panel.job_id) {
      forecastPanelRuntime.jobId = panel.job_id;
      var select = $("dt-forecast-model");
      if (select && panel.job_id && select.value !== panel.job_id) {
        select.value = panel.job_id;
      }
    }
    var next = cloneDashboard(currentData());
    var point = panel.chart_point || {};
    // Series arrays keep the historical demand/supply names but now hold the
    // generic predicted (solid) and actual (dashed) validation series.
    var predictedVal = point.predicted != null ? point.predicted : point.demand;
    var actualVal = point.actual != null ? point.actual : point.supply;
    if (options.appendChart !== false && point.label != null) {
      next.forecast.demand = asArray(next.forecast.demand).concat([asNumber(predictedVal, 0)]);
      next.forecast.supply = asArray(next.forecast.supply).concat([asNumber(actualVal, 0)]);
      next.forecast.labels = asArray(next.forecast.labels).concat([String(point.label)]);
      var trimmed = trimForecastSeries(next.forecast, forecastPanelRuntime.maxPoints);
      next.forecast.demand = trimmed.demand;
      next.forecast.supply = trimmed.supply;
      next.forecast.labels = trimmed.labels;
    }
    updateForecastLegend(point.predicted_label, point.actual_label);
    dashboardData = normalizeDashboardData(next);
    setText("dt-forecast-status", forecastStatusLine(panel));
    renderForecastPredictionsTable(panel.predictions);
    renderForecastChart($("dt-forecast-svg"));
  }

  function updateForecastLegend(predictedLabel, actualLabel) {
    if (predictedLabel) setText("dt-forecast-legend-predicted", predictedLabel);
    if (actualLabel) setText("dt-forecast-legend-actual", actualLabel);
  }

  function bindForecastModelSelect(onChange) {
    if (forecastPanelRuntime.selectBound) return;
    var select = $("dt-forecast-model");
    if (!select || typeof onChange !== "function") return;
    select.addEventListener("change", function () {
      var jobId = select.value || "";
      if (!jobId) return;
      forecastPanelRuntime.jobId = jobId;
      onChange(jobId);
    });
    forecastPanelRuntime.selectBound = true;
  }

  function renderDashboard() {
    renderMapPanel($("dt-iso-map"));
    bindControls();
    applyDashboardKpis();
    renderForecastChart($("dt-forecast-svg"));
    renderDonutChart($("dt-donut-svg"));
    renderDonutLegend($("dt-donut-legend"));
    renderAlerts($("dt-alert-list"));
    renderDeliveries($("dt-delivery-list"));
    renderUtilization($("dt-util-list"));
  }

  function loadDashboardData() {
    dashboardLoading = true;
    fetch("/api/studio/dashboard-data", { headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) throw new Error("Dashboard data request failed: " + response.status);
        return response.json();
      })
      .then(function (payload) {
        dashboardData = normalizeDashboardData(payload);
        dashboardLoaded = true;
        renderDashboard();
      })
      .catch(function (error) {
        dashboardData = normalizeDashboardData({
          source: {
            status: "degraded",
            label: "Django data unavailable",
            warnings: [error.message || String(error)],
          },
        });
        renderDashboard();
      })
      .finally(function () {
        dashboardLoading = false;
      });
  }

  function runtimeTotalInventory(centers) {
    return asArray(centers).reduce(function (sum, center) {
      var inventory = center && center.inventory;
      if (inventory && typeof inventory === "object") {
        return sum + Object.keys(inventory).reduce(function (inner, key) {
          return inner + Math.max(asNumber(inventory[key], 0), 0);
        }, 0);
      }
      return sum + Math.max(asNumber(center && center.total_units, 0), 0);
    }, 0);
  }

  function runtimeFacilityRows(centers, existingFacilities) {
    centers = asArray(centers);
    existingFacilities = asArray(existingFacilities);
    if (!centers.length) return existingFacilities;
    if (existingFacilities.length) {
      return resolveMapFacilityTemplates({ facilities: existingFacilities }).map(function (facility) {
        if (!facilityShowsDetailTooltip(facility.type)) return facility;
        var center = findSimCenter(centers, facility.id) || findSimCenter(centers, facility.name);
        if (!center) return facility;
        return Object.assign({}, facility, centerSnapshotToFacilityFields(center));
      });
    }
    var positions = [
      [22, 24], [58, 28], [78, 38], [28, 58], [52, 62],
      [75, 72], [15, 72], [42, 42], [86, 58], [34, 78],
    ];
    return centers.map(function (center, index) {
      var pos = positions[index % positions.length];
      var total = runtimeTotalInventory([center]);
      var stats = center.stats || {};
      return {
        id: center.name || "center-" + (index + 1),
        name: center.name || "Center " + (index + 1),
        type: center.type || "hospital",
        glyph: String(center.type || "").toLowerCase() === "donor" ? "D" : "H",
        x: pos[0],
        y: pos[1],
        stock: total,
        usage: asNumber(stats.transfused, 0),
        utilization: total ? clamp((asNumber(stats.transfused, 0) / Math.max(total + asNumber(stats.transfused, 0), 1)) * 100, 0, 100) : 0,
        critical: false,
        inventory: center.inventory,
        stats: stats,
        total_units: center.total_units != null ? center.total_units : total,
        centerType: center.type,
      };
    });
  }

  function runtimeSource(sourceStatus, fallback) {
    sourceStatus = sourceStatus || {};
    fallback = fallback || {};
    var status = sourceStatus.status || fallback.status || "degraded";
    var labels = {
      fresh_live: "Live Django data",
      stale_live: "Stale Django data",
      degraded: "Django data degraded",
      synthetic: "Synthetic simulation",
      fresh: "Django operational data",
    };
    return Object.assign({}, fallback, {
      status: status,
      label: labels[status] || fallback.label || "Runtime data",
      warnings: asArray(sourceStatus.warnings).length ? sourceStatus.warnings : asArray(fallback.warnings),
      generatedAt: (sourceStatus.freshness && sourceStatus.freshness.captured_at) || fallback.generatedAt,
    });
  }

  function runtimeWeatherLabel(value) {
    var labels = {
      clear: "Clear",
      rain: "Rain",
      storm: "Storm",
      snow: "Snow",
      heat: "Heat",
      cold: "Cold",
    };
    var key = cssToken(value || "", "");
    return labels[key] || String(value || "");
  }

  function runtimeDeliveryRows(snapshot, routes, facilities, previous) {
    previous = previous || DEFAULT_DASHBOARD.deliveries;
    var activeActions = asArray(snapshot.active_actions);
    var recentDonations = Math.max(asNumber(snapshot.recent_donations, 0), 0);
    var recentShortages = Math.max(asNumber(snapshot.recent_shortages, 0), 0);
    var total = Math.max(activeActions.length, recentDonations, recentShortages);
    if (!total && asArray(previous.items).length) {
      return previous;
    }
    if (!total) {
      return Object.assign({}, previous, {
        total: 0,
        onTime: "0",
        delayed: "0",
        items: [],
        caption: "No live transport actions in current twin tick",
      });
    }
    var facilityNames = {};
    asArray(facilities).forEach(function (facility) {
      facilityNames[facility.id] = facility.name || facility.id;
    });
    var rows = asArray(routes).slice(0, Math.min(asArray(routes).length, total, 5)).map(function (route, index) {
      var source = route.sourceName || facilityNames[route.sourceFacilityId] || "Network source";
      var target = route.targetName || facilityNames[route.targetFacilityId] || "Network target";
      var pressure = route.pressure != null ? asNumber(route.pressure, 0) : asNumber(snapshot.shortage_rate, 0);
      return {
        route: source + " → " + target,
        eta: "Pressure " + formatPercent(clamp(pressure, 0, 100)),
        progress: clamp(pressure || asNumber(snapshot.shortage_rate, 0) || 8 + index * 10, 6, 98),
      };
    });
    return {
      total: total,
      onTimeLabel: "Tracked",
      delayedLabel: "Shortage-linked",
      onTime: String(Math.max(total - recentShortages, 0)),
      delayed: String(recentShortages),
      items: rows,
      caption: rows.length ? "Derived from live twin actions and operational links" : "Active twin movement has no route detail",
    };
  }

  function runtimeAlerts(snapshot, previousAlerts) {
    var eventRows = asArray(snapshot.active_events).map(function (event) {
      return {
        type: "warning",
        icon: "alert-triangle",
        title: event.name || event.event_key || "Twin event",
        text: event.description || event.name || event.event_key || "Twin event",
        meta: event.source ? "Source: " + event.source : "",
        time: "",
      };
    });
    return eventRows.concat(asArray(previousAlerts)).slice(0, 5);
  }

  function mergeRuntimeFrame(snapshot) {
    snapshot = snapshot || {};
    var next = cloneDashboard(currentData());
    var centers = asArray(snapshot.centers);
    var facilities = runtimeFacilityRows(centers, next.facilities);
    var routes = asArray(next.routes).length ? asArray(next.routes) : derivedRoutes(facilities);
    var totalInventory = runtimeTotalInventory(centers);
    var activeAlerts = runtimeAlerts(snapshot, next.alerts);
    var totalTransfused = Math.max(asNumber(snapshot.total_transfused, 0), 0);
    var totalShortage = Math.max(asNumber(snapshot.total_shortage, 0), 0);
    var serviceLevel = totalTransfused + totalShortage > 0
      ? (totalTransfused / Math.max(totalTransfused + totalShortage, 1)) * 100
      : next.kpis.serviceLevel;
    var shortageRate = asNumber(snapshot.shortage_rate, null);
    next.source = runtimeSource(snapshot.source_status, next.source);
    next.facilities = facilities;
    next.mapCenters = centers;
    next.routes = routes;
    next.alerts = activeAlerts;
    next.alertCount = activeAlerts.length;
    next.kpis = Object.assign({}, next.kpis, {
      totalInventory: totalInventory > 0 ? totalInventory : next.kpis.totalInventory,
      unitsInTransit: Math.max(
        asArray(snapshot.active_actions).length,
        asNumber(snapshot.recent_donations, 0),
        asNumber(snapshot.recent_shortages, 0),
      ),
      criticalShortages: Math.max(asNumber(next.kpis.criticalShortages, 0), asNumber(snapshot.recent_shortages, 0)),
      serviceLevel: serviceLevel,
      networkEfficiency: shortageRate == null ? next.kpis.networkEfficiency : clamp(100 - shortageRate, 0, 100),
      deliveryCaption: "Live tracking",
      shortageCaption: "Requires attention",
      efficiencyDeltaPct: null,
    });
    next.deliveries = runtimeDeliveryRows(snapshot, routes, facilities, next.deliveries);
    next.vehicles = asArray(next.vehicles).length ? next.vehicles : derivedVehicles(routes, next.deliveries);
    if (snapshot.weather && (!next.weather || next.weather === "Weather unavailable")) {
      next.weather = runtimeWeatherLabel(snapshot.weather);
    }
    dashboardData = normalizeDashboardData(next);
    dashboardLoaded = true;
    renderDashboard();
  }

  /* ── ForecastCard chart ── */
  function renderForecastChart(svgEl) {
    if (!svgEl) return;
    var w = 280;
    var h = 140;
    var pad = { t: 12, r: 30, b: 28, l: 40 };
    var data = currentData().forecast || {};
    var demand = asArray(data.demand).map(function (v) { return asNumber(v, 0); });
    var supply = asArray(data.supply).map(function (v) { return asNumber(v, 0); });
    var labels = asArray(data.labels);
    var points = Math.max(demand.length, supply.length, labels.length);
    if (!points) {
      svgEl.setAttribute("viewBox", "0 0 " + w + " " + h);
      svgEl.innerHTML =
        '<text x="' + (w / 2) + '" y="' + (h / 2) + '" text-anchor="middle" fill="#64748b" font-size="11">No forecast rows</text>';
      return;
    }
    var rawMax = Math.max.apply(null, demand.concat(supply).concat([0]));
    // Adaptive Y scale: fractional series (e.g. stockout probability / rate live
    // in [0,1]) get fine-grained rounding; unit counts round up to tens.
    var maxY = rawMax <= 1 ? (rawMax <= 0 ? 1 : Math.ceil(rawMax * 10) / 10) : Math.ceil(rawMax / 10) * 10;
    function fmtY(v) {
      return v === 0 ? "0" : (maxY <= 1 ? v.toFixed(2) : formatCompact(v));
    }

    function scaleX(i) {
      return pad.l + (points === 1 ? 0.5 : i / (points - 1)) * (w - pad.l - pad.r);
    }
    function scaleY(v) {
      return pad.t + (1 - v / maxY) * (h - pad.t - pad.b);
    }

    function pathFrom(arr) {
      if (!arr.length) return "";
      if (arr.length === 1) return "M" + scaleX(0) + " " + scaleY(arr[0]) + " L" + (scaleX(0) + 1) + " " + scaleY(arr[0]);
      return arr
        .map(function (v, i) {
          return (i === 0 ? "M" : "L") + scaleX(i) + " " + scaleY(v);
        })
        .join(" ");
    }

    var yLabels = [0, maxY / 3, (maxY / 3) * 2, maxY];
    var gridLines = yLabels
      .map(function (v) {
        return '<line x1="' + pad.l + '" y1="' + scaleY(v) + '" x2="' + (w - pad.r) + '" y2="' + scaleY(v) + '" stroke="#e5eaf2" stroke-width="1"/>';
      })
      .join("");

    var yText = yLabels
      .map(function (v) {
        return '<text x="' + (pad.l - 6) + '" y="' + (scaleY(v) + 4) + '" text-anchor="end" fill="#64748b" font-size="9">' + fmtY(v) + "</text>";
      })
      .join("");

    var tickIndexes = points === 1 ? [0] : [0, Math.floor((points - 1) / 2), points - 1];
    var xText = tickIndexes
      .filter(function (value, index, list) { return list.indexOf(value) === index; })
      .map(function (tick) {
        return (
          '<text x="' +
          scaleX(tick) +
          '" y="' +
          (h - 6) +
          '" text-anchor="middle" fill="#64748b" font-size="9">' +
          escapeHtml(dateLabel(labels[tick] || "")) +
          "</text>"
        );
      })
      .join("");

    svgEl.setAttribute("viewBox", "0 0 " + w + " " + h);
    svgEl.innerHTML =
      gridLines +
      yText +
      xText +
      '<path d="' +
      pathFrom(supply) +
      '" fill="none" stroke="#22c55e" stroke-width="2" stroke-dasharray="5 4" opacity="0.8"/>' +
      '<path d="' +
      pathFrom(demand) +
      '" fill="none" stroke="#2563eb" stroke-width="2.5"/>';
  }

  /* ── DonutChart ── */
  function renderDonutChart(svgEl) {
    if (!svgEl) return;
    var cx = 60;
    var cy = 60;
    var r = 48;
    var inner = 32;
    var rows = asArray(currentData().bloodTypes).map(withBloodColor);
    var total = rows.reduce(function (s, b) { return s + asNumber(b.pct, 0); }, 0);
    if (!rows.length || !total) {
      svgEl.setAttribute("viewBox", "0 0 120 120");
      svgEl.innerHTML = '<circle cx="60" cy="60" r="46" fill="none" stroke="#e2e8f0" stroke-width="16"/>';
      return;
    }
    var angle = -90;

    function arc(start, end, outer, innerR) {
      var s = (start * Math.PI) / 180;
      var e = (end * Math.PI) / 180;
      var x1 = cx + outer * Math.cos(s);
      var y1 = cy + outer * Math.sin(s);
      var x2 = cx + outer * Math.cos(e);
      var y2 = cy + outer * Math.sin(e);
      var x3 = cx + innerR * Math.cos(e);
      var y3 = cy + innerR * Math.sin(e);
      var x4 = cx + innerR * Math.cos(s);
      var y4 = cy + innerR * Math.sin(s);
      var large = end - start > 180 ? 1 : 0;
      return (
        "M" + x1 + " " + y1 +
        " A" + outer + " " + outer + " 0 " + large + " 1 " + x2 + " " + y2 +
        " L" + x3 + " " + y3 +
        " A" + innerR + " " + innerR + " 0 " + large + " 0 " + x4 + " " + y4 + " Z"
      );
    }

    var paths = rows
      .map(function (b) {
        var sweep = (asNumber(b.pct, 0) / total) * 360;
        var start = angle;
        angle += sweep;
        return '<path d="' + arc(start, angle, r, inner) + '" fill="' + b.color + '"/>';
      })
      .join("");

    svgEl.setAttribute("viewBox", "0 0 120 120");
    svgEl.innerHTML = paths;
  }

  function renderDonutLegend(container) {
    if (!container) return;
    var rows = asArray(currentData().bloodTypes).map(withBloodColor);
    if (!rows.length) {
      container.innerHTML = '<div class="dt-empty-state">No inventory rows</div>';
      return;
    }
    container.innerHTML = rows
      .map(function (b) {
        return (
          '<div class="dt-donut-legend-item">' +
          '<span class="dot" style="background:' + escapeHtml(b.color) + '"></span>' +
          "<span>" + escapeHtml(b.type) + " " + formatPercent(b.pct) + " " + formatUnits(b.units) + "</span>" +
          "</div>"
        );
      })
      .join("");
  }

  /* ── AlertCard ── */
  function renderAlerts(container) {
    if (!container) return;
    var rows = asArray(currentData().alerts);
    if (!rows.length) {
      container.innerHTML = '<div class="dt-alert-row is-empty"><div><div class="dt-alert-text">No active Django alerts</div><div class="dt-alert-meta">Operational alert table returned no open rows</div></div></div>';
      return;
    }
    container.innerHTML = rows
      .map(function (a) {
        var meta = a.link
          ? '<a href="#">' + escapeHtml(a.link) + "</a>"
          : escapeHtml(a.meta || "");
        var ago = timeAgo(a.time);
        return (
          '<div class="dt-alert-row">' +
          '<div class="dt-alert-icon ' + cssToken(a.type || "warning", "warning") + '">' + icon(a.icon || "alert-triangle", 18) + "</div>" +
          '<div><div class="dt-alert-text">' + escapeHtml(a.text || a.title || "Operational alert") + '</div>' +
          '<div class="dt-alert-meta">' + meta + (meta && ago ? " · " : "") + escapeHtml(ago) + "</div></div>" +
          "</div>"
        );
      })
      .join("");
  }

  /* ── DeliveryList ── */
  function renderDeliveries(container) {
    if (!container) return;
    var d = currentData().deliveries || DEFAULT_DASHBOARD.deliveries;
    var total = asNumber(d.total, 0);
    setText("dt-delivery-total", formatInteger(total));
    var status = document.querySelector(".dt-delivery-status");
    if (status) {
      var onTime = status.querySelector(".on-time");
      var delayed = status.querySelector(".delayed");
      if (onTime) onTime.textContent = (d.onTimeLabel || "On-time") + ": " + (d.onTime || "0");
      if (delayed) delayed.textContent = (d.delayedLabel || "Delayed") + ": " + (d.delayed || "0");
    }
    if (!asArray(d.items).length) {
      container.innerHTML = '<div class="dt-empty-state">' + escapeHtml(d.caption || "No active transport records") + "</div>";
      return;
    }
    container.innerHTML = d.items
      .map(function (item) {
        return (
          '<div class="dt-delivery-item">' +
          '<div class="dt-delivery-item-header"><span>' + escapeHtml(item.route || "Delivery") + "</span><small>" + escapeHtml(item.eta || "") + "</small></div>" +
          '<div class="dt-progress-bar"><div class="fill" style="width:' + Math.max(0, Math.min(100, asNumber(item.progress, 0))) + '%"></div></div>' +
          "</div>"
        );
      })
      .join("");
  }

  /* ── UtilizationBars ── */
  function renderUtilization(container) {
    if (!container) return;
    var rows = asArray(currentData().utilization);
    if (!rows.length) {
      container.innerHTML = '<div class="dt-empty-state">No facility utilization rows</div>';
      return;
    }
    container.innerHTML = rows
      .map(function (u) {
        var pct = Math.max(0, Math.min(100, asNumber(u.pct, 0)));
        return (
          '<div class="dt-util-row">' +
          '<div class="dt-util-header"><span>' + escapeHtml(u.name || "Facility") + "</span><span>" + formatPercent(pct) + "</span></div>" +
          '<div class="dt-util-bar"><div class="fill" style="width:' + pct + '%"></div></div>' +
          "</div>"
        );
      })
      .join("");
  }

  /* ── Apply dashboard KPI data ── */
  function applyDashboardKpis() {
    var data = currentData();
    var k = data.kpis || DEFAULT_DASHBOARD.kpis;
    var totalInventory = formatCompact(k.totalInventory);
    var serviceLevel = formatPercent(k.serviceLevel);
    var efficiency = formatPercent(k.networkEfficiency);
    var alertCount = asNumber(data.alertCount, asArray(data.alerts).length);
    var unitsInTransit = asNumber(k.unitsInTransit, asNumber(data.deliveries && data.deliveries.total, 0));
    var fields = {
      "metric-total-inventory": totalInventory,
      "metric-inventory-delta": formatDelta(k.inventoryDeltaPct, k.inventoryCaption),
      "metric-active-deliveries": formatInteger(unitsInTransit),
      "metric-delivery-caption": k.deliveryCaption || "No transport records",
      "metric-critical-shortages": formatInteger(k.criticalShortages),
      "metric-shortage-caption": k.shortageCaption || "No active alerts",
      "metric-service-level": serviceLevel,
      "metric-service-caption": "Derived from active alert load",
      "metric-network-efficiency": efficiency,
      "metric-efficiency-delta": k.efficiencyDeltaPct == null ? "Derived from stock and usage" : formatDelta(k.efficiencyDeltaPct, ""),
      "bottom-in-transit": formatInteger(unitsInTransit),
      "bottom-service-rate": serviceLevel,
      "bottom-efficiency": efficiency,
      "runtime-alert-count": formatInteger(alertCount),
      "dt-alert-badge": formatInteger(alertCount),
      "dt-delivery-total": formatInteger(unitsInTransit),
      "dt-weather-chip": data.weather || "Weather unavailable",
      "iso-weather-chip": data.weather || "14°C Light rain",
    };
    Object.keys(fields).forEach(function (id) { setText(id, fields[id]); });
    var donutCenter = document.querySelector(".dt-donut-center strong");
    if (donutCenter) donutCenter.textContent = totalInventory;
    var source = data.source || {};
    var sourceLabel = source.label || (source.status === "fresh" ? "Django" : "Degraded");
    setText("runtime-source-label", sourceLabel);
    var badge = $("runtime-source-badge");
    if (badge) {
      badge.className = "dt-source-badge source-" + cssToken(source.status || "degraded", "degraded");
      badge.style.display = "inline-flex";
    }
  }

  function isGameScreenVisible() {
    var screen = $("game-screen");
    if (!screen) return false;
    if (screen.style.display === "none") return false;
    return window.getComputedStyle(screen).display !== "none";
  }

  function init() {
    injectIconSprites();
    bindControls();
    bindMapFacilityTooltips();
    if (isGameScreenVisible()) {
      onGameScreenShown();
    }
  }

  global.DashboardUI = {
    data: function () { return cloneDashboard(currentData()); },
    init: init,
    onGameScreenShown: onGameScreenShown,
    renderMapPanel: renderMapPanel,
    setMapZoom: setMapZoom,
    resetMapView: resetMapView,
    mergeRuntimeFrame: mergeRuntimeFrame,
    populateForecastModelSelect: populateForecastModelSelect,
    bindForecastModelSelect: bindForecastModelSelect,
    mergeForecastTick: mergeForecastTick,
    resetForecastTickSeries: resetForecastTickSeries,
    renderForecastPredictionsTable: renderForecastPredictionsTable,
    renderForecastChart: renderForecastChart,
    renderDonutChart: renderDonutChart,
    renderAlerts: renderAlerts,
    renderDeliveries: renderDeliveries,
    renderUtilization: renderUtilization,
    buildIsoMapSvg: buildIsoMapSvg,
    buildStaticMapImageBase: buildStaticMapImageBase,
    setMapNightMode: setMapNightMode,
    setMapDayFactorFromHour: setMapDayFactorFromHour,
    setMapDayFactor: setMapDayFactor,
    buildIsoCityBase: buildIsoCityBase,
    buildIsoRouteOverlay: buildIsoRouteOverlay,
    buildIsoFacilityBuildingsOverlay: buildIsoFacilityBuildingsOverlay,
    buildIsoFacilityOverlay: buildIsoFacilityOverlay,
    createFacility: createFacility,
    createBackgroundBuilding: createBackgroundBuilding,
    createFacilityBuilding: createFacilityBuilding,
    createRoadPath: createRoadPath,
    createVehicle: createVehicle,
    createBridge: createBridge,
    resolveIsoFacilities: resolveIsoFacilities,
    buildFacilityTooltipHtml: buildFacilityTooltipHtml,
    bindMapFacilityTooltips: bindMapFacilityTooltips,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(window);
