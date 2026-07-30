{% include "drf_spectacular/swagger_ui.js" %}

{% if docs_auto_auth_enabled %}
(function () {
  const authUrl = "{{ docs_auto_auth_url|escapejs }}";
  const authName = "{{ docs_auto_auth_security_scheme|escapejs }}";

  const authorize = async () => {
    if (!authUrl || !authName || typeof ui === "undefined") {
      return;
    }

    try {
      const response = await fetch(authUrl, {
        credentials: "same-origin",
        headers: { Accept: "application/json" },
      });

      if (!response.ok) {
        console.warn("Swagger auto-auth failed with HTTP " + response.status + ".");
        return;
      }

      const payload = await response.json();
      if (payload.access_token && ui.preauthorizeApiKey) {
        ui.preauthorizeApiKey(authName, payload.access_token);
      }
    } catch (error) {
      console.warn("Swagger auto-auth failed.", error);
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", authorize, { once: true });
  } else {
    window.setTimeout(authorize, 0);
  }
})();
{% endif %}
