(function () {
  "use strict";

  var API_URL = "/api/studio/config";

  function $(id) {
    return document.getElementById(id);
  }

  function notify(message, type) {
    type = type || "info";
    if (typeof window.showToast === "function") {
      window.showToast({ description: message, type: type, duration: 3000 });
      return;
    }
    if (type === "error") {
      console.error(message);
    } else {
      console.info(message);
    }
  }

  function fetchConfig() {
    return fetch(API_URL, { method: "GET", headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      });
  }

  function saveConfig(payload) {
    return fetch(API_URL, {
      method: "PUT",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
    }).then(function (response) {
      if (!response.ok) {
        return response.json().then(function (body) {
          var detail = body.detail || JSON.stringify(body);
          throw new Error(detail);
        });
      }
      return response.json();
    });
  }

  function getNumber(id) {
    var el = $(id);
    if (!el) return null;
    var value = parseFloat(el.value);
    return isNaN(value) ? null : value;
  }

  function getInt(id) {
    var el = $(id);
    if (!el) return null;
    var value = parseInt(el.value, 10);
    return isNaN(value) ? null : value;
  }

  function getBool(id) {
    var el = $(id);
    return el ? el.checked : null;
  }

  function setValue(id, value) {
    var el = $(id);
    if (!el) return;
    if (typeof value === "boolean") {
      el.checked = value;
    } else if (value !== null && value !== undefined) {
      el.value = value;
    }
  }

  function populateForm(config) {
    setValue("cfg-enabled", config.enabled);
    setValue("cfg-max-hospitals", config.max_hospitals);
    setValue("cfg-min-hospitals", config.min_hospitals);
    setValue("cfg-include-blood-banks", config.include_blood_banks);
    setValue("cfg-include-mobile", config.include_mobile_from_strategy);
    setValue("cfg-capacity-nurses", config.hospital_capacity && config.hospital_capacity.nurses);
    setValue("cfg-capacity-lab", config.hospital_capacity && config.hospital_capacity.lab);
    setValue("cfg-capacity-processing", config.hospital_capacity && config.hospital_capacity.processing);
    setValue("cfg-hours-open", config.hospital_hours && config.hospital_hours[0]);
    setValue("cfg-hours-close", config.hospital_hours && config.hospital_hours[1]);
    setValue("cfg-geo-north", config.geo_bbox && config.geo_bbox.north);
    setValue("cfg-geo-south", config.geo_bbox && config.geo_bbox.south);
    setValue("cfg-geo-east", config.geo_bbox && config.geo_bbox.east);
    setValue("cfg-geo-west", config.geo_bbox && config.geo_bbox.west);
    setValue("cfg-split-rbc", config.component_split && config.component_split.RBC);
    setValue("cfg-split-platelets", config.component_split && config.component_split.PLATELETS);
    setValue("cfg-split-plasma", config.component_split && config.component_split.PLASMA);
    setValue("cfg-fallback-defaults", config.fallback_to_default_centers);
    setValue("cfg-demand-weight", config.demand_weight_from_usage);
    setValue("cfg-weight-floor", config.usage_demand_weight_floor);
    setValue("cfg-weight-cap", config.usage_demand_weight_cap);
    setValue("cfg-use-django-centers", config.use_django_centers);
    setValue("cfg-max-django-centers", config.max_django_centers);
  }

  function buildPayload() {
    var payload = {};

    var enabled = getBool("cfg-enabled");
    if (enabled !== null) payload.enabled = enabled;

    var maxHospitals = getInt("cfg-max-hospitals");
    if (maxHospitals !== null) payload.max_hospitals = maxHospitals;

    var minHospitals = getInt("cfg-min-hospitals");
    if (minHospitals !== null) payload.min_hospitals = minHospitals;

    var includeBloodBanks = getBool("cfg-include-blood-banks");
    if (includeBloodBanks !== null) payload.include_blood_banks = includeBloodBanks;

    var includeMobile = getBool("cfg-include-mobile");
    if (includeMobile !== null) payload.include_mobile_from_strategy = includeMobile;

    payload.hospital_capacity = {
      nurses: getInt("cfg-capacity-nurses"),
      lab: getInt("cfg-capacity-lab"),
      processing: getInt("cfg-capacity-processing"),
    };

    payload.hospital_hours = {
      open: getNumber("cfg-hours-open"),
      close: getNumber("cfg-hours-close"),
    };

    payload.geo_bbox = {
      north: getNumber("cfg-geo-north"),
      south: getNumber("cfg-geo-south"),
      east: getNumber("cfg-geo-east"),
      west: getNumber("cfg-geo-west"),
    };

    payload.component_split = {
      RBC: getNumber("cfg-split-rbc"),
      PLATELETS: getNumber("cfg-split-platelets"),
      PLASMA: getNumber("cfg-split-plasma"),
    };

    var fallback = getBool("cfg-fallback-defaults");
    if (fallback !== null) payload.fallback_to_default_centers = fallback;

    var demandWeight = getBool("cfg-demand-weight");
    if (demandWeight !== null) payload.demand_weight_from_usage = demandWeight;

    var weightFloor = getNumber("cfg-weight-floor");
    if (weightFloor !== null) payload.usage_demand_weight_floor = weightFloor;

    var weightCap = getNumber("cfg-weight-cap");
    if (weightCap !== null) payload.usage_demand_weight_cap = weightCap;

    var useDjango = getBool("cfg-use-django-centers");
    if (useDjango !== null) payload.use_django_centers = useDjango;

    var maxDjango = getInt("cfg-max-django-centers");
    if (maxDjango !== null) payload.max_django_centers = maxDjango;

    return payload;
  }

  function validatePayload(payload) {
    if (
      payload.hospital_hours &&
      payload.hospital_hours.close !== null &&
      payload.hospital_hours.open !== null &&
      payload.hospital_hours.close < payload.hospital_hours.open
    ) {
      throw new Error("Hospital close time must be after open time.");
    }

    if (
      payload.geo_bbox &&
      payload.geo_bbox.north !== null &&
      payload.geo_bbox.south !== null &&
      payload.geo_bbox.north < payload.geo_bbox.south
    ) {
      throw new Error("Geo north must be greater than or equal to south.");
    }

    if (
      payload.geo_bbox &&
      payload.geo_bbox.east !== null &&
      payload.geo_bbox.west !== null &&
      payload.geo_bbox.east < payload.geo_bbox.west
    ) {
      throw new Error("Geo east must be greater than or equal to west.");
    }

    if (payload.component_split) {
      var values = [
        payload.component_split.RBC,
        payload.component_split.PLATELETS,
        payload.component_split.PLASMA,
      ].filter(function (v) {
        return v !== null && v !== undefined;
      });
      if (values.length === 3) {
        var total = values.reduce(function (a, b) {
          return a + b;
        }, 0);
        if (total < 0.99 || total > 1.01) {
          throw new Error("Component split must sum to 1.0.");
        }
      }
    }

    if (
      payload.usage_demand_weight_floor !== null &&
      payload.usage_demand_weight_cap !== null &&
      payload.usage_demand_weight_floor > payload.usage_demand_weight_cap
    ) {
      throw new Error("Demand weight floor must be less than or equal to cap.");
    }

    if (
      payload.max_hospitals !== null &&
      payload.min_hospitals !== null &&
      payload.max_hospitals < payload.min_hospitals
    ) {
      throw new Error("Max hospitals must be greater than or equal to min hospitals.");
    }
  }

  function loadConfigIntoForm() {
    fetchConfig()
      .then(function (config) {
        populateForm(config);
        setStatus("Loaded operational simulation config.", "info");
      })
      .catch(function (error) {
        setStatus("Failed to load config: " + error.message, "error");
      });
  }

  function setStatus(message, type) {
    var el = $("cfg-status");
    if (!el) return;
    el.textContent = message;
    el.className = "cfg-status cfg-status-" + type;
  }

  function openModal() {
    var modal = $("cfg-modal");
    if (!modal) return;
    modal.removeAttribute("hidden");
    loadConfigIntoForm();
  }

  function closeModal() {
    var modal = $("cfg-modal");
    if (!modal) return;
    modal.setAttribute("hidden", "");
  }

  function onSubmit(event) {
    event.preventDefault();
    var payload;
    try {
      payload = buildPayload();
      validatePayload(payload);
    } catch (error) {
      setStatus(error.message, "error");
      notify(error.message, "error");
      return;
    }

    setStatus("Saving...", "info");
    saveConfig(payload)
      .then(function () {
        setStatus("Config saved.", "success");
        notify("Simulation config saved.", "success");
      })
      .catch(function (error) {
        setStatus("Save failed: " + error.message, "error");
        notify("Save failed: " + error.message, "error");
      });
  }

  function bindControls() {
    var openBtn = $("cfg-open-btn");
    if (openBtn) openBtn.addEventListener("click", openModal);

    var closeBtn = $("cfg-close-btn");
    if (closeBtn) closeBtn.addEventListener("click", closeModal);

    var cancelBtn = $("cfg-cancel-btn");
    if (cancelBtn) cancelBtn.addEventListener("click", closeModal);

    var form = $("cfg-form");
    if (form) form.addEventListener("submit", onSubmit);

    var modal = $("cfg-modal");
    if (modal) {
      modal.addEventListener("click", function (event) {
        if (event.target === modal) closeModal();
      });
    }

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && modal && !modal.hasAttribute("hidden")) {
        closeModal();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindControls);
  } else {
    bindControls();
  }
})();
