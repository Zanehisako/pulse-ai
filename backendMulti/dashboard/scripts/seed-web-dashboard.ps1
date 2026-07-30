$regions = @("quebec", "montreal")
$periods = @("24h", "2d", "1w")
$alerts = @("low", "medium", "high")

function rand($min, $max) {
  Get-Random -Minimum $min -Maximum $max
}

foreach ($region in $regions) {
  foreach ($period in $periods) {
    foreach ($alert in $alerts) {

      # ---- IDEAL ----
      curl.exe -X POST "http://localhost:8000/dashboard/add-web/" `
        -H "Content-Type: application/json" `
        -d "{
          `"typeModel`": `"ideal`",
          `"region`": `"$region`",
          `"product`": `"blood`",
          `"period`": `"$period`",
          `"alert_level`": `"$alert`",
          `"jsonResponse`": {
            `"age`": `"$(rand 18 25)-$(rand 50 70) yrs`",
            `"bmi`": `"$(rand 18 22)-$(rand 24 30)`",
            `"weight_min_kg`": $(rand 45 65)
          }
        }"

      # ---- STOCK ----
      curl.exe -X POST "http://localhost:8000/dashboard/add-web/" `
        -H "Content-Type: application/json" `
        -d "{
          `"typeModel`": `"stock`",
          `"region`": `"$region`",
          `"product`": `"blood`",
          `"period`": `"$period`",
          `"alert_level`": `"$alert`",
          `"jsonResponse`": {
            `"A+`": $(rand 20 150),
            `"A-`": $(rand 10 80),
            `"B+`": $(rand 20 120),
            `"B-`": $(rand 5 60),
            `"O+`": $(rand 50 200),
            `"O-`": $(rand 10 70),
            `"AB+`": $(rand 5 100),
            `"AB-`": $(rand 1 50)
          }
        }"

      # ---- DONATIONS ----
      $base = rand 5 50
      curl.exe -X POST "http://localhost:8000/dashboard/add-web/" `
        -H "Content-Type: application/json" `
        -d "{
          `"typeModel`": `"donations`",
          `"region`": `"$region`",
          `"product`": `"blood`",
          `"period`": `"$period`",
          `"alert_level`": `"$alert`",
          `"jsonResponse`": {
            `"data`": [
              { `"date`": `"2026-01-01`", `"value`": $base },
              { `"date`": `"2026-01-02`", `"value`": $($base + (rand -10 20)) },
              { `"date`": `"2026-01-03`", `"value`": $($base + (rand -5 30)) }
            ]
          }
        }"

      # ---- METRICS ----
      $don = rand 5000 20000
      $active = rand 1000 8000
      $critical = rand 0 6
      curl.exe -X POST "http://localhost:8000/dashboard/add-web/" `
        -H "Content-Type: application/json" `
        -d "{
          `"typeModel`": `"metrics`",
          `"region`": `"$region`",
          `"product`": `"blood`",
          `"period`": `"$period`",
          `"alert_level`": `"$alert`",
          `"jsonResponse`": [
            { `"label`": `"Total Donations`", `"value`": $don, `"change`": $(rand -10 15), `"icon`": `"droplets`", `"color`": `"primary`" },
            { `"label`": `"Active Donors`", `"value`": $active, `"change`": $(rand -10 10), `"icon`": `"users`", `"color`": `"warning`" },
            { `"label`": `"Critical Stock Types`", `"value`": $critical, `"change`": $(rand -2 3), `"icon`": `"alertTriangle`", `"color`": `"danger`" }
          ]
        }"

    }
  }
}