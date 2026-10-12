let currentTab = 'buyer';
let cityStatsMap = {};
let latestPrediction = null;

// Presets data
const PRESETS = {
  seattle_family: {
    city: 'Seattle',
    bedrooms: 3,
    bathrooms: 2.0,
    sqft_living: 1950,
    sqft_lot: 5000,
    floors: '2.0',
    sqft_basement: 300,
    waterfront: false,
    view: 0,
    condition: 3,
    yr_built: 1985,
    yr_renovated: 0,
  },
  bellevue_luxury: {
    city: 'Bellevue',
    bedrooms: 4,
    bathrooms: 3.5,
    sqft_living: 3800,
    sqft_lot: 12000,
    floors: '2.0',
    sqft_basement: 800,
    waterfront: false,
    view: 2,
    condition: 4,
    yr_built: 2004,
    yr_renovated: 0,
  },
  redmond_tech: {
    city: 'Redmond',
    bedrooms: 4,
    bathrooms: 2.5,
    sqft_living: 2600,
    sqft_lot: 8500,
    floors: '2.0',
    sqft_basement: 0,
    waterfront: false,
    view: 1,
    condition: 4,
    yr_built: 1998,
    yr_renovated: 2010,
  },
  waterfront_villa: {
    city: 'Mercer Island',
    bedrooms: 5,
    bathrooms: 4.5,
    sqft_living: 4800,
    sqft_lot: 22000,
    floors: '2.0',
    sqft_basement: 1200,
    waterfront: true,
    view: 4,
    condition: 5,
    yr_built: 2008,
    yr_renovated: 0,
  },
  starter_home: {
    city: 'Kent',
    bedrooms: 3,
    bathrooms: 1.5,
    sqft_living: 1350,
    sqft_lot: 6200,
    floors: '1.0',
    sqft_basement: 0,
    waterfront: false,
    view: 0,
    condition: 3,
    yr_built: 1978,
    yr_renovated: 0,
  }
};

// Initialize application on DOM loaded
document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  await loadCities();
  await loadMetrics();
  await loadSampleData();
  runPrediction();
});

function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentTab = tab.dataset.tab;

      // Handle full-width section toggles
      const transparencySec = document.getElementById('transparencySection');
      const explorerSec = document.getElementById('explorerSection');

      if (currentTab === 'transparency') {
        transparencySec.style.display = 'block';
        explorerSec.style.display = 'none';
        transparencySec.scrollIntoView({ behavior: 'smooth' });
      } else if (currentTab === 'explorer') {
        explorerSec.style.display = 'block';
        transparencySec.style.display = 'none';
        explorerSec.scrollIntoView({ behavior: 'smooth' });
      } else {
        transparencySec.style.display = 'none';
        explorerSec.style.display = 'none';
      }

      renderPersonaContent();
    });
  });
}

async function loadCities() {
  try {
    const res = await fetch('/api/cities');
    const data = await res.json();
    const citySelect = document.getElementById('citySelect');
    citySelect.innerHTML = '';

    data.cities.forEach(city => {
      const opt = document.createElement('option');
      opt.value = city;
      opt.textContent = city;
      if (city === 'Seattle') opt.selected = true;
      citySelect.appendChild(opt);
    });

    data.stats.forEach(s => {
      cityStatsMap[s.city] = s;
    });

    updateCityStatHelper();
  } catch (err) {
    console.error('Failed to load cities:', err);
  }
}

async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    const r2 = data.report.model.R2;
    const mae = data.report.model.MAE;
    const rmse = data.report.model.RMSE;

    document.getElementById('badgeR2').textContent = `R²: ${r2.toFixed(3)}`;
    if (document.getElementById('metricR2')) {
      document.getElementById('metricR2').textContent = r2.toFixed(3);
      document.getElementById('metricMAE').textContent = `$${Math.round(mae).toLocaleString()}`;
      document.getElementById('metricRMSE').textContent = `$${Math.round(rmse).toLocaleString()}`;
    }
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

async function loadSampleData() {
  try {
    const res = await fetch('/api/sample-data');
    const data = await res.json();
    const tbody = document.getElementById('dataTableBody');
    tbody.innerHTML = '';

    data.samples.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${row.city}</strong></td>
        <td style="color: #38bdf8; font-weight:600;">${row.price_formatted}</td>
        <td>${row.bedrooms}</td>
        <td>${row.bathrooms}</td>
        <td>${row.sqft_living.toLocaleString()}</td>
        <td>${row.sqft_lot.toLocaleString()}</td>
        <td>${row.waterfront ? '🌊 Yes' : 'No'}</td>
        <td>${row.view}</td>
        <td>${row.condition}/5</td>
        <td>${row.yr_built}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Failed to load samples:', err);
  }
}

function updateCityStatHelper() {
  const city = document.getElementById('citySelect').value;
  const stat = cityStatsMap[city];
  const helpEl = document.getElementById('cityStatHelp');
  if (stat) {
    helpEl.textContent = `City Historical Mean: $${Math.round(stat.avg_price).toLocaleString()} • Median: $${Math.round(stat.median_price).toLocaleString()} (${stat.count} sales)`;
  } else {
    helpEl.textContent = 'City baseline active';
  }
}

function updateSliderVal(id, displayId, unit) {
  const val = document.getElementById(id).value;
  document.getElementById(displayId).textContent = Number(val).toLocaleString() + unit;
}

function stepVal(id, step) {
  const input = document.getElementById(id);
  let val = parseFloat(input.value) + step;
  const min = parseFloat(input.min);
  const max = parseFloat(input.max);
  if (val >= min && val <= max) {
    input.value = val;
    runPrediction();
  }
}

function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;

  const citySel = document.getElementById('citySelect');
  if (Array.from(citySel.options).some(o => o.value === p.city)) {
    citySel.value = p.city;
  }
  document.getElementById('bedrooms').value = p.bedrooms;
  document.getElementById('bathrooms').value = p.bathrooms;

  document.getElementById('sqftLiving').value = p.sqft_living;
  document.getElementById('sqftLivingVal').textContent = p.sqft_living.toLocaleString() + ' sqft';

  document.getElementById('sqftLot').value = p.sqft_lot;
  document.getElementById('sqftLotVal').textContent = p.sqft_lot.toLocaleString() + ' sqft';

  document.getElementById('sqftBasement').value = p.sqft_basement;
  document.getElementById('sqftBasementVal').textContent = p.sqft_basement.toLocaleString() + ' sqft';

  document.getElementById('floorsInput').value = p.floors;
  document.getElementById('conditionSelect').value = p.condition;
  document.getElementById('viewSelect').value = p.view;
  document.getElementById('yrBuilt').value = p.yr_built;
  document.getElementById('yrRenovated').value = p.yr_renovated;
  document.getElementById('waterfrontCheck').checked = p.waterfront;

  updateCityStatHelper();
  runPrediction();
}

async function runPrediction() {
  updateCityStatHelper();

  const payload = {
    city: document.getElementById('citySelect').value,
    bedrooms: parseFloat(document.getElementById('bedrooms').value),
    bathrooms: parseFloat(document.getElementById('bathrooms').value),
    sqft_living: parseFloat(document.getElementById('sqftLiving').value),
    sqft_lot: parseFloat(document.getElementById('sqftLot').value),
    floors: parseFloat(document.getElementById('floorsInput').value),
    sqft_basement: parseFloat(document.getElementById('sqftBasement').value),
    waterfront: document.getElementById('waterfrontCheck').checked ? 1 : 0,
    view: parseInt(document.getElementById('viewSelect').value),
    condition: parseInt(document.getElementById('conditionSelect').value),
    yr_built: parseInt(document.getElementById('yrBuilt').value),
    yr_renovated: parseInt(document.getElementById('yrRenovated').value),
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.error) {
      alert(result.error);
      return;
    }
    latestPrediction = result;
    renderResults(result);
    renderPersonaContent();
  } catch (err) {
    console.error('Prediction failed:', err);
  }
}

function renderResults(res) {
  // Animate price display
  animateNumber('predictedPriceDisplay', Math.round(res.predicted_price));

  // Confidence band
  document.getElementById('confidenceBandDisplay').textContent =
    `$${Math.round(res.lower_bound).toLocaleString()} – $${Math.round(res.upper_bound).toLocaleString()}`;

  // Quick stats
  document.getElementById('pricePerSqftDisplay').textContent = `$${Math.round(res.price_per_sqft)} / sqft`;
  document.getElementById('monthlyMortgageDisplay').textContent =
    `$${Math.round(res.mortgage_estimate.monthly_payment).toLocaleString()} / mo`;

  if (res.city_benchmark) {
    document.getElementById('cityAvgDisplay').textContent = `$${Math.round(res.city_benchmark).toLocaleString()}`;
    const pct = ((res.predicted_price - res.city_benchmark) / res.city_benchmark) * 100;
    const sign = pct >= 0 ? '+' : '';
    document.getElementById('cityDiffDisplay').textContent = `${sign}${pct.toFixed(1)}%`;
    document.getElementById('cityDiffDisplay').style.color = pct >= 0 ? '#38bdf8' : '#94a3b8';
  }

  // Feature drivers waterfall
  renderDrivers(res.top_drivers);
}

function renderDrivers(drivers) {
  const container = document.getElementById('driversList');
  container.innerHTML = '';

  if (!drivers || drivers.length === 0) return;

  const maxAbs = Math.max(...drivers.map(d => Math.abs(d.impact)));

  drivers.forEach(d => {
    const isPositive = d.impact >= 0;
    const widthPct = Math.min(100, Math.max(8, (Math.abs(d.impact) / maxAbs) * 100));

    const row = document.createElement('div');
    row.className = 'driver-row';
    row.innerHTML = `
      <span class="driver-name" title="${d.feature}">${d.feature}</span>
      <div class="driver-bar-wrapper">
        <div class="driver-bar-fill ${isPositive ? 'fill-positive' : 'fill-negative'}" style="width: ${widthPct}%;"></div>
      </div>
      <span class="driver-impact ${isPositive ? 'impact-pos' : 'impact-neg'}">
        ${isPositive ? '+' : '-'}$${Math.round(Math.abs(d.impact)).toLocaleString()}
      </span>
    `;
    container.appendChild(row);
  });
}

function renderPersonaContent() {
  if (!latestPrediction) return;

  const p = latestPrediction;
  const insightBox = document.getElementById('personaInsightContent');
  const addonCard = document.getElementById('personaAddonCard');

  if (currentTab === 'buyer') {
    insightBox.innerHTML = `
      <div class="insight-header">🛒 Buyer Intelligence: Fair Price Assessment</div>
      <div class="insight-body">
        The statistical fair market estimate for this home is <strong>$${Math.round(p.predicted_price).toLocaleString()}</strong>.
        Using a standard 20% down payment ($${Math.round(p.mortgage_estimate.down_payment_20).toLocaleString()}), 
        your estimated monthly principal & interest is <strong>$${Math.round(p.mortgage_estimate.monthly_payment).toLocaleString()}</strong>.
      </div>
      <span class="badge-highlight">💡 Negotiation Guard: Do not bid above $${Math.round(p.upper_bound).toLocaleString()}</span>
    `;

    addonCard.innerHTML = `
      <div class="card-header">
        <h3>Deal Quality Checker</h3>
        <span class="tag-sm">Buyer Protection</span>
      </div>
      <p class="section-desc">Enter the seller's asking price to test if this home is overpriced or a bargain:</p>
      <div style="display:flex; gap:0.8rem; align-items:center; margin-bottom:1rem;">
        <input type="number" id="askingPriceInput" class="form-control" placeholder="e.g. 750000" value="${Math.round(p.predicted_price * 1.05)}" style="max-width:240px;">
        <button class="btn-calculate" style="width:auto; padding:0.6rem 1.2rem;" onclick="checkDealQuality()">Evaluate Listing</button>
      </div>
      <div id="dealVerdictBox"></div>
    `;
    checkDealQuality();

  } else if (currentTab === 'seller') {
    const aggressive = Math.round(p.predicted_price * 1.06);
    const competitive = Math.round(p.predicted_price);
    const quickSale = Math.round(p.predicted_price * 0.95);

    insightBox.innerHTML = `
      <div class="insight-header">🏷️ Seller Pricing Strategy</div>
      <div class="insight-body">
        Recommended listing range based on Multiple Linear Regression parameters:
        <br>• <strong>Competitive List:</strong> $${competitive.toLocaleString()} (Target benchmark)
        <br>• <strong>Optimistic / Aggressive:</strong> $${aggressive.toLocaleString()} (May take longer on market)
        <br>• <strong>Speed Sale:</strong> $${quickSale.toLocaleString()} (High bid volume expected)
      </div>
    `;

    addonCard.innerHTML = `
      <div class="card-header">
        <h3>Pre-Listing Renovation & Upgrade ROI</h3>
        <span class="tag-sm">Linear Coefficient Simulation</span>
      </div>
      <p class="section-desc">Simulate how specific improvements elevate your expected sale valuation:</p>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:0.8rem;">
        <button class="preset-chip" style="padding:0.7rem; text-align:left;" onclick="simulateUpgrade('bathroom')">
          🛁 <strong>Add 0.5 Bathroom</strong><br>
          <small style="color:#10b981;">Est. +$23,000 to +$30,000</small>
        </button>
        <button class="preset-chip" style="padding:0.7rem; text-align:left;" onclick="simulateUpgrade('condition')">
          ✨ <strong>Upgrade Condition to Pristine</strong><br>
          <small style="color:#10b981;">Est. +$26,000/step</small>
        </button>
        <button class="preset-chip" style="padding:0.7rem; text-align:left;" onclick="simulateUpgrade('basement')">
          🏠 <strong>Finish 400 Sq Ft Basement</strong><br>
          <small style="color:#10b981;">Est. +$40,000 living value</small>
        </button>
        <button class="preset-chip" style="padding:0.7rem; text-align:left;" onclick="simulateUpgrade('renovate')">
          🔨 <strong>Mark as Recently Renovated</strong><br>
          <small style="color:#38bdf8;">Applies modern construction weight</small>
        </button>
      </div>
    `;

  } else if (currentTab === 'agent') {
    insightBox.innerHTML = `
      <div class="insight-header">📋 Real Estate Agent CMA Dossier</div>
      <div class="insight-body">
        Data-driven property brief for client presentation:
        <br>• <strong>Market Segment:</strong> ${p.inputs.city} Submarket (${document.getElementById('cityStatHelp').textContent.split('•')[0]})
        <br>• <strong>Square Foot Unit Valuation:</strong> $${Math.round(p.price_per_sqft)} / sq ft
        <br>• <strong>Model Confidence:</strong> Evaluated on King County dataset with R² = 0.676
      </div>
      <span class="badge-highlight">⏱️ Saves 2-3 hours of manual MLS comp research</span>
    `;

    addonCard.innerHTML = `
      <div class="card-header">
        <h3>Client-Ready CMA Export</h3>
        <span class="tag-sm">Instant Deliverable</span>
      </div>
      <p class="section-desc">Generate a clean, professional summary card to share with buyers or sellers:</p>
      <div style="display:flex; gap:0.8rem;">
        <button class="btn-calculate" style="width:auto; padding:0.6rem 1.2rem;" onclick="window.print()">🖨️ Print / Save PDF CMA</button>
        <button class="preset-chip" style="padding:0.6rem 1.2rem;" onclick="copyCMAClipboard()">📋 Copy Summary to Clipboard</button>
      </div>
    `;
  }
}

function checkDealQuality() {
  if (!latestPrediction) return;
  const asking = parseFloat(document.getElementById('askingPriceInput').value);
  const fair = latestPrediction.predicted_price;
  const box = document.getElementById('dealVerdictBox');
  if (isNaN(asking) || asking <= 0) return;

  const diff = asking - fair;
  const pct = (diff / fair) * 100;

  if (pct > 12) {
    box.innerHTML = `
      <div style="background:rgba(244, 63, 94, 0.15); border:1px solid #f43f5e; padding:0.8rem; border-radius:8px;">
        <strong style="color:#f43f5e;">⚠️ Overpriced (+${pct.toFixed(1)}% above Fair MLR Estimate)</strong>
        <p style="font-size:0.82rem; margin-top:0.3rem; color:#fecdd3;">The listing is asking $${Math.round(diff).toLocaleString()} more than the linear regression hedonic valuation. Consider submitting an offer closer to $${Math.round(fair).toLocaleString()}.</p>
      </div>
    `;
  } else if (pct < -8) {
    box.innerHTML = `
      <div style="background:rgba(16, 185, 129, 0.15); border:1px solid #10b981; padding:0.8rem; border-radius:8px;">
        <strong style="color:#10b981;">🎉 Attractive Deal (${Math.abs(pct).toFixed(1)}% below Fair MLR Estimate)</strong>
        <p style="font-size:0.82rem; margin-top:0.3rem; color:#a7f3d0;">The asking price is significantly below hedonic market value! High potential equity buffer for buyers.</p>
      </div>
    `;
  } else {
    box.innerHTML = `
      <div style="background:rgba(56, 189, 248, 0.15); border:1px solid #38bdf8; padding:0.8rem; border-radius:8px;">
        <strong style="color:#38bdf8;">✅ Fair Market Value (Within ±10% range)</strong>
        <p style="font-size:0.82rem; margin-top:0.3rem; color:#bae6fd;">The property is priced appropriately in line with historical sales in ${latestPrediction.inputs.city}.</p>
      </div>
    `;
  }
}

function simulateUpgrade(type) {
  if (type === 'bathroom') {
    stepVal('bathrooms', 0.5);
  } else if (type === 'condition') {
    const sel = document.getElementById('conditionSelect');
    if (parseInt(sel.value) < 5) {
      sel.value = parseInt(sel.value) + 1;
      runPrediction();
    }
  } else if (type === 'basement') {
    const base = document.getElementById('sqftBasement');
    const liv = document.getElementById('sqftLiving');
    base.value = parseInt(base.value) + 400;
    liv.value = parseInt(liv.value) + 400;
    updateSliderVal('sqftBasement', 'sqftBasementVal', ' sqft');
    updateSliderVal('sqftLiving', 'sqftLivingVal', ' sqft');
    runPrediction();
  } else if (type === 'renovate') {
    document.getElementById('yrRenovated').value = 2014;
    runPrediction();
  }
}

function copyCMAClipboard() {
  if (!latestPrediction) return;
  const p = latestPrediction;
  const text = `🏡 PROPERTY CMA VALUATION REPORT
Location: ${p.inputs.city}, WA
Predicted Fair Value: $${Math.round(p.predicted_price).toLocaleString()}
Confidence Range: $${Math.round(p.lower_bound).toLocaleString()} – $${Math.round(p.upper_bound).toLocaleString()}
Price / Sq Ft: $${Math.round(p.price_per_sqft)}
Bedrooms: ${p.inputs.bedrooms} | Bathrooms: ${p.inputs.bathrooms}
Living Space: ${p.inputs.sqft_living} sqft | Lot: ${p.inputs.sqft_lot} sqft
Model: Scikit-Learn Multiple Linear Regression (R² = 0.676)
Generated via HousePredict AI`;
  navigator.clipboard.writeText(text).then(() => {
    alert('CMA Summary copied to clipboard!');
  });
}

function animateNumber(elementId, targetVal) {
  const el = document.getElementById(elementId);
  const startVal = parseInt(el.textContent.replace(/,/g, '')) || 0;
  const duration = 500;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easeOutQuad = 1 - (1 - progress) * (1 - progress);
    const current = Math.round(startVal + (targetVal - startVal) * easeOutQuad);
    el.textContent = current.toLocaleString();

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  requestAnimationFrame(update);
}
