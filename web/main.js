/* ================================================================
   Radar de Valorización Urbana — Lógica Principal
   Módulo: main.js
   Descripción: Maneja la interacción con la API, renderizado del
   mapa Leaflet, pestañas, simulador, comparador y explicabilidad.
   ================================================================ */

'use strict';

/* ================================================================
   CONFIGURACIÓN
   ================================================================ */

/** URL base del backend API */
const API_BASE_URL = 'http://localhost:8000';

/** Estado global de la aplicación */
const AppState = {
    map: null,
    markersLayer: null,
    heatLayer: null,
    cells: [],
    selectedCells: [],
    currentHorizon: 12,
    apiOnline: false,
    usingDemoData: false,
    globalDrivers: null,
    dashboardLoaded: false
};

/* ================================================================
   DATOS DE DEMO (Fallback cuando la API no está disponible)
   ================================================================ */

/** Celdas de demostración alrededor de Quito y Guayaquil */
const DEMO_CELLS = [
    // Quito — norte y centro
    { cell_id: 'QUI-001', lat: -0.0925, lon: -78.4540, city: 'Quito',  avg_price: 1450.00 },
    { cell_id: 'QUI-002', lat: -0.1037, lon: -78.4702, city: 'Quito',  avg_price: 1320.50 },
    { cell_id: 'QUI-003', lat: -0.1759, lon: -78.4673, city: 'Quito',  avg_price: 1180.00 },
    { cell_id: 'QUI-004', lat: -0.2056, lon: -78.4913, city: 'Quito',  avg_price:  980.00 },
    { cell_id: 'QUI-005', lat: -0.1508, lon: -78.4900, city: 'Quito',  avg_price: 1240.00 },
    { cell_id: 'QUI-006', lat: -0.0674, lon: -78.4678, city: 'Quito',  avg_price: 1620.00 },
    { cell_id: 'QUI-007', lat: -0.2215, lon: -78.5123, city: 'Quito',  avg_price:  820.00 },
    { cell_id: 'QUI-008', lat: -0.1132, lon: -78.4935, city: 'Quito',  avg_price: 1390.00 },
    { cell_id: 'QUI-009', lat: -0.0820, lon: -78.5100, city: 'Quito',  avg_price: 1535.00 },
    { cell_id: 'QUI-010', lat: -0.1950, lon: -78.4300, city: 'Quito',  avg_price:  910.00 },
    // Guayaquil — norte, sur y centro
    { cell_id: 'GYE-001', lat: -2.1455, lon: -79.8745, city: 'Guayaquil', avg_price: 1280.00 },
    { cell_id: 'GYE-002', lat: -2.1703, lon: -79.9028, city: 'Guayaquil', avg_price: 1150.00 },
    { cell_id: 'GYE-003', lat: -2.0872, lon: -79.8585, city: 'Guayaquil', avg_price: 1640.00 },
    { cell_id: 'GYE-004', lat: -2.2010, lon: -79.9120, city: 'Guayaquil', avg_price:  870.00 },
    { cell_id: 'GYE-005', lat: -2.1560, lon: -79.9235, city: 'Guayaquil', avg_price: 1090.00 },
    { cell_id: 'GYE-006', lat: -2.0530, lon: -79.8650, city: 'Guayaquil', avg_price: 1820.00 },
    { cell_id: 'GYE-007', lat: -2.2290, lon: -79.9020, city: 'Guayaquil', avg_price:  760.00 },
    { cell_id: 'GYE-008', lat: -2.1340, lon: -79.8840, city: 'Guayaquil', avg_price: 1360.00 },
    { cell_id: 'GYE-009', lat: -2.0990, lon: -79.9400, city: 'Guayaquil', avg_price: 1490.00 },
    { cell_id: 'GYE-010', lat: -2.1810, lon: -79.8400, city: 'Guayaquil', avg_price: 1010.00 }
];

/** Datos SHAP de demostración */
const DEMO_SHAP = {
    cell_id: 'QUI-001',
    base_value: 0.085,
    shap_values: {
        'distancia_al_metro':      0.032,
        'proximidad_comercial':    0.018,
        'densidad_poblacional':    0.011,
        'tasa_interes_mensual':   -0.015,
        'crecimiento_pib':         0.009,
        'inversion_infraestructura': 0.007,
        'migracion_neta':          0.005,
        'antiguedad_construccion':-0.004,
        'area_verde_cercana':      0.003,
        'acceso_vias_principales': 0.002
    },
    top_drivers: [
        { feature: 'distancia_al_metro',        value:  0.032, direction: 'positivo' },
        { feature: 'proximidad_comercial',      value:  0.018, direction: 'positivo' },
        { feature: 'tasa_interes_mensual',      value: -0.015, direction: 'negativo' },
        { feature: 'densidad_poblacional',      value:  0.011, direction: 'positivo' },
        { feature: 'crecimiento_pib',           value:  0.009, direction: 'positivo' }
    ]
};

/** Predicciones de demostración por celda */
const DEMO_PREDICTIONS = {
    'QUI-001': { cell_id: 'QUI-001', predicted_valuation: 0.142, lower_bound: 0.098, upper_bound: 0.186, confidence: 0.87 },
    'QUI-002': { cell_id: 'QUI-002', predicted_valuation: 0.118, lower_bound: 0.082, upper_bound: 0.154, confidence: 0.84 },
    'QUI-003': { cell_id: 'QUI-003', predicted_valuation: 0.095, lower_bound: 0.065, upper_bound: 0.125, confidence: 0.81 },
    'QUI-004': { cell_id: 'QUI-004', predicted_valuation: 0.063, lower_bound: 0.038, upper_bound: 0.088, confidence: 0.76 },
    'QUI-005': { cell_id: 'QUI-005', predicted_valuation: 0.104, lower_bound: 0.072, upper_bound: 0.136, confidence: 0.82 },
    'QUI-006': { cell_id: 'QUI-006', predicted_valuation: 0.158, lower_bound: 0.112, upper_bound: 0.204, confidence: 0.89 },
    'QUI-007': { cell_id: 'QUI-007', predicted_valuation: 0.041, lower_bound: 0.022, upper_bound: 0.060, confidence: 0.71 },
    'QUI-008': { cell_id: 'QUI-008', predicted_valuation: 0.127, lower_bound: 0.090, upper_bound: 0.164, confidence: 0.85 },
    'QUI-009': { cell_id: 'QUI-009', predicted_valuation: 0.151, lower_bound: 0.107, upper_bound: 0.195, confidence: 0.88 },
    'QUI-010': { cell_id: 'QUI-010', predicted_valuation: 0.072, lower_bound: 0.046, upper_bound: 0.098, confidence: 0.77 },
    'GYE-001': { cell_id: 'GYE-001', predicted_valuation: 0.119, lower_bound: 0.084, upper_bound: 0.154, confidence: 0.83 },
    'GYE-002': { cell_id: 'GYE-002', predicted_valuation: 0.098, lower_bound: 0.068, upper_bound: 0.128, confidence: 0.80 },
    'GYE-003': { cell_id: 'GYE-003', predicted_valuation: 0.156, lower_bound: 0.110, upper_bound: 0.202, confidence: 0.88 },
    'GYE-004': { cell_id: 'GYE-004', predicted_valuation: 0.055, lower_bound: 0.032, upper_bound: 0.078, confidence: 0.74 },
    'GYE-005': { cell_id: 'GYE-005', predicted_valuation: 0.087, lower_bound: 0.058, upper_bound: 0.116, confidence: 0.79 },
    'GYE-006': { cell_id: 'GYE-006', predicted_valuation: 0.172, lower_bound: 0.124, upper_bound: 0.220, confidence: 0.90 },
    'GYE-007': { cell_id: 'GYE-007', predicted_valuation: 0.038, lower_bound: 0.019, upper_bound: 0.057, confidence: 0.70 },
    'GYE-008': { cell_id: 'GYE-008', predicted_valuation: 0.131, lower_bound: 0.094, upper_bound: 0.168, confidence: 0.85 },
    'GYE-009': { cell_id: 'GYE-009', predicted_valuation: 0.143, lower_bound: 0.101, upper_bound: 0.185, confidence: 0.86 },
    'GYE-010': { cell_id: 'GYE-010', predicted_valuation: 0.082, lower_bound: 0.055, upper_bound: 0.109, confidence: 0.78 }
};

/** Resultado de simulación de demostración */
const DEMO_SIMULATION = {
    adjusted_valuations: [
        { cell_id: 'QUI-001', base_valuation: 0.142, adjusted_valuation: 0.168, change: 0.026 },
        { cell_id: 'QUI-002', base_valuation: 0.118, adjusted_valuation: 0.139, change: 0.021 },
        { cell_id: 'QUI-003', base_valuation: 0.095, adjusted_valuation: 0.111, change: 0.016 },
        { cell_id: 'QUI-004', base_valuation: 0.063, adjusted_valuation: 0.072, change: 0.009 },
        { cell_id: 'QUI-005', base_valuation: 0.104, adjusted_valuation: 0.122, change: 0.018 },
        { cell_id: 'QUI-006', base_valuation: 0.158, adjusted_valuation: 0.185, change: 0.027 },
        { cell_id: 'QUI-007', base_valuation: 0.041, adjusted_valuation: 0.046, change: 0.005 },
        { cell_id: 'QUI-008', base_valuation: 0.127, adjusted_valuation: 0.149, change: 0.022 },
        { cell_id: 'QUI-009', base_valuation: 0.151, adjusted_valuation: 0.178, change: 0.027 },
        { cell_id: 'QUI-010', base_valuation: 0.072, adjusted_valuation: 0.083, change: 0.011 },
        { cell_id: 'GYE-001', base_valuation: 0.119, adjusted_valuation: 0.140, change: 0.021 },
        { cell_id: 'GYE-002', base_valuation: 0.098, adjusted_valuation: 0.115, change: 0.017 },
        { cell_id: 'GYE-003', base_valuation: 0.156, adjusted_valuation: 0.183, change: 0.027 },
        { cell_id: 'GYE-004', base_valuation: 0.055, adjusted_valuation: 0.063, change: 0.008 },
        { cell_id: 'GYE-005', base_valuation: 0.087, adjusted_valuation: 0.102, change: 0.015 },
        { cell_id: 'GYE-006', base_valuation: 0.172, adjusted_valuation: 0.201, change: 0.029 },
        { cell_id: 'GYE-007', base_valuation: 0.038, adjusted_valuation: 0.043, change: 0.005 },
        { cell_id: 'GYE-008', base_valuation: 0.131, adjusted_valuation: 0.154, change: 0.023 },
        { cell_id: 'GYE-009', base_valuation: 0.143, adjusted_valuation: 0.168, change: 0.025 },
        { cell_id: 'GYE-010', base_valuation: 0.082, adjusted_valuation: 0.096, change: 0.014 }
    ],
    impact_summary: {
        promedio_cambio: 0.018,
        celda_mayor_impacto: 'GYE-006',
        celda_menor_impacto: 'QUI-007',
        total_celdas_afectadas: 20,
        direccion_dominante: 'positiva'
    }
};

/* ================================================================
   UTILIDADES DE API
   ================================================================ */

/**
 * Construye una URL completa para un endpoint del API.
 * @param {string} path — Ruta relativa del endpoint (ej: '/cells')
 * @returns {string} URL completa
 */
function getApiUrl(path) {
    return `${API_BASE_URL}${path}`;
}

/**
 * Realiza un fetch al API con manejo de errores y timeout.
 * @param {string} path — Ruta del endpoint
 * @param {Object} [options] — Opciones de fetch
 * @returns {Promise<Object>} Respuesta JSON del API
 */
async function fetchApi(path, options = {}) {
    const url = getApiUrl(path);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
            headers: {
                'Content-Type': 'application/json',
                ...(options.headers || {})
            }
        });
        clearTimeout(timeout);

        if (!response.ok) {
            throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (err) {
        clearTimeout(timeout);
        throw err;
    }
}

/* ================================================================
   UTILIDADES DE UI
   ================================================================ */

/**
 * Muestra una notificación toast.
 * @param {string} message — Mensaje a mostrar
 * @param {('success'|'error'|'info')} [type='info'] — Tipo de toast
 * @param {number} [duration=3500] — Duración en milisegundos
 */
function showToast(message, type = 'info', duration = 3500) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        toast.style.transition = 'opacity 280ms ease, transform 280ms ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Muestra el overlay de carga global.
 */
function showGlobalLoading() {
    const el = document.getElementById('globalLoading');
    if (el) el.classList.remove('hidden');
}

/**
 * Oculta el overlay de carga global.
 */
function hideGlobalLoading() {
    const el = document.getElementById('globalLoading');
    if (el) el.classList.add('hidden');
}

/**
 * Muestra un elemento de carga.
 * @param {string} elementId — ID del elemento de loading
 */
function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.classList.remove('hidden');
}

/**
 * Oculta un elemento de carga.
 * @param {string} elementId — ID del elemento de loading
 */
function hideLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.classList.add('hidden');
}

/**
 * Muestra un mensaje de error en un contenedor.
 * @param {string} elementId — ID del elemento de error
 * @param {string} message — Mensaje de error
 */
function showError(elementId, message) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = message;
        el.classList.remove('hidden');
    }
}

/**
 * Oculta un elemento de error.
 * @param {string} elementId — ID del elemento de error
 */
function hideError(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.classList.add('hidden');
}

/**
 * Retorna un color según el valor de valorización.
 * @param {number} value — Valor de valorización (0 a 1+)
 * @returns {string} Color hex
 */
function getValuationColor(value) {
    if (value > 0.15) return '#4ade80'; // Verde — alta valorización
    if (value > 0.08) return '#fbbf24'; // Amarillo — valorización media
    if (value > 0.03) return '#f4823c'; // Naranja — valorización baja
    return '#ef4444';                   // Rojo — muy baja o negativa
}

/**
 * Formatea un valor numérico como porcentaje.
 * @param {number} value — Valor decimal (0.15 = 15%)
 * @param {number} [decimals=2] — Número de decimales
 * @returns {string} Porcentaje formateado
 */
function formatPercent(value, decimals = 2) {
    return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Formatea un precio en USD.
 * @param {number} value — Precio
 * @returns {string} Precio formateado
 */
function formatPrice(value) {
    return `$${value.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}`;
}

/* ================================================================
   SALUD DEL API
   ================================================================ */

/**
 * Verifica la salud del API y actualiza el indicador de estado.
 */
async function checkApiHealth() {
    const dot = document.getElementById('apiStatusDot');
    const text = document.getElementById('apiStatusText');

    if (!dot || !text) return;

    dot.className = 'status-dot status-dot--unknown';
    text.textContent = 'Conectando…';

    try {
        const data = await fetchApi('/health');
        if (data && (data.status === 'ok' || data.status === 'healthy' || data.status === 'UP')) {
            AppState.apiOnline = true;
            AppState.usingDemoData = false;
            dot.className = 'status-dot status-dot--online';
            text.textContent = 'API en línea';
        } else {
            throw new Error('Respuesta inesperada del API');
        }
    } catch (err) {
        AppState.apiOnline = false;
        AppState.usingDemoData = true;
        dot.className = 'status-dot status-dot--offline';
        text.textContent = 'Modo demo (API no disponible)';
        console.warn('[Radar] API no disponible, usando datos de demostración:', err.message);
    }
}

/* ================================================================
   MAPA — LEAFLET
   ================================================================ */

/**
 * Inicializa el mapa Leaflet centrado en Ecuador.
 */
function initMap() {
    const mapEl = document.getElementById('leafletMap');
    if (!mapEl) return;

    AppState.map = L.map('leafletMap', {
        zoomControl: true,
        attributionControl: true
    }).setView([-1.0, -78.5], 7);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap — Radar de Valorización Urbana',
        maxZoom: 18
    }).addTo(AppState.map);

    AppState.markersLayer = L.layerGroup().addTo(AppState.map);
    AppState.heatLayer = L.layerGroup().addTo(AppState.map);

    // Invalidar tamaño después de que el contenedor sea visible
    setTimeout(() => {
        if (AppState.map) AppState.map.invalidateSize();
    }, 200);
}

/**
 * Carga las celdas desde el API o usa datos de demostración.
 */
async function loadCells() {
    showGlobalLoading();

    try {
        let cells;

        if (AppState.apiOnline) {
            cells = await fetchApi('/cells');
        } else {
            cells = DEMO_CELLS;
        }

        if (!Array.isArray(cells) || cells.length === 0) {
            cells = DEMO_CELLS;
            AppState.usingDemoData = true;
        }

        AppState.cells = cells;
        renderCellMarkers(cells);
        renderHeatmap(cells);
        populateCellSelects(cells);

        if (AppState.usingDemoData) {
            showToast('Usando datos de demostración — API no disponible', 'info');
        } else {
            showToast(`${cells.length} celdas cargadas`, 'success');
        }
    } catch (err) {
        console.error('[Radar] Error al cargar celdas:', err);
        AppState.cells = DEMO_CELLS;
        AppState.usingDemoData = true;
        renderCellMarkers(DEMO_CELLS);
        renderHeatmap(DEMO_CELLS);
        populateCellSelects(DEMO_CELLS);
        showToast('Error al cargar celdas — usando datos de demostración', 'error');
    } finally {
        hideGlobalLoading();
    }
}


/* ================================================================
   DASHBOARD — KPIs, índice, drivers globales, estadísticas
   ================================================================ */

/**
 * Carga y renderiza el panel general (dashboard).
 */
async function loadDashboard() {
    hideError('dashboardError');

    try {
        let summary, indexData, driversData;

        if (AppState.apiOnline) {
            [summary, indexData, driversData] = await Promise.all([
                fetchApi('/summary'),
                fetchApi('/index'),
                fetchApi('/drivers'),
            ]);
        } else {
            summary = generateDemoSummary();
            indexData = generateDemoIndex();
            driversData = generateDemoDrivers();
        }

        // KPIs
        document.getElementById('kpiCells').textContent = summary.total_cells ?? '—';
        document.getElementById('kpiTx').textContent = (summary.total_transactions ?? 0).toLocaleString();
        document.getElementById('kpiValuation').textContent =
            ((summary.avg_valuation ?? 0) * 100).toFixed(2) + '%';
        document.getElementById('kpiIndex').textContent =
            (indexData.global_average ?? '—') + ' / 100';

        // Gráficos
        renderIndexDistribution('indexDistChart', summary.index_distribution || {});
        renderCityComparison('cityIndexChart', indexData.city_averages || {}, indexData.global_average || 0);
        renderTopCells('topCellsChart', summary.top_cells || []);

        const drivers = (driversData.drivers || []).slice(0, 12);
        renderFeatureImportance('globalDriversChart', {
            features: drivers.map(d => d.feature),
            importances: drivers.map(d => d.importance),
        });

        renderCityStatsTable('cityStatsTable', summary.city_stats || []);

        // Guardar drivers globales para la pestaña de explicabilidad
        AppState.globalDrivers = driversData;
        renderGlobalDrivers();
    } catch (err) {
        console.error('[Radar] Error al cargar dashboard:', err);
        showError('dashboardError', 'No se pudo cargar el panel general.');
    }
}

/**
 * Renderiza el ranking global de drivers en la pestaña de explicabilidad.
 */
function renderGlobalDrivers() {
    const drivers = (AppState.globalDrivers?.drivers || []).slice(0, 12);
    if (drivers.length === 0) return;

    renderFeatureImportance('globalDriversExplainChart', {
        features: drivers.map(d => d.feature),
        importances: drivers.map(d => d.importance),
    });
}

/**
 * Datos de demostración para el summary.
 */
function generateDemoSummary() {
    return {
        total_cells: DEMO_CELLS.length,
        total_transactions: 12450,
        avg_valuation: 0.058,
        avg_price: 128500,
        top_cells: DEMO_CELLS.slice(0, 10).map((c, i) => ({
            cell_id: c.cell_id,
            city: c.city,
            annualized_valuation: 0.16 - i * 0.012,
        })),
        city_stats: [
            { city: 'Quito', cells: 10, transactions: 6120, avg_price: 132000, avg_valuation: 0.062 },
            { city: 'Guayaquil', cells: 10, transactions: 6330, avg_price: 125000, avg_valuation: 0.054 },
        ],
        index_distribution: { 'bajo (<40)': 6, 'medio (40-60)': 10, 'alto (60-80)': 4, 'muy alto (>80)': 0 },
    };
}

/**
 * Datos de demostración para el índice.
 */
function generateDemoIndex() {
    const cityAverages = {};
    DEMO_CELLS.forEach(c => {
        cityAverages[c.city] = cityAverages[c.city] || { sum: 0, n: 0 };
        cityAverages[c.city].sum += c.index || 55;
        cityAverages[c.city].n += 1;
    });
    const averages = {};
    Object.entries(cityAverages).forEach(([city, v]) => {
        averages[city] = Math.round((v.sum / v.n) * 10) / 10;
    });
    const allVals = DEMO_CELLS.map(c => c.index || 55);
    const global = Math.round((allVals.reduce((a, b) => a + b, 0) / allVals.length) * 10) / 10;
    return { city_averages: averages, global_average: global, cells: [] };
}

/**
 * Datos de demostración para el ranking de drivers.
 */
function generateDemoDrivers() {
    return {
        method: 'demo',
        drivers: [
            { feature: 'price_trend', importance: 0.158 },
            { feature: 'avg_price', importance: 0.065 },
            { feature: 'banks_count', importance: 0.065 },
            { feature: 'transaction_count', importance: 0.056 },
            { feature: 'walkability_score', importance: 0.056 },
            { feature: 'connectivity_index', importance: 0.048 },
            { feature: 'ndvi_mean', importance: 0.042 },
            { feature: 'schools_count', importance: 0.040 },
            { feature: 'median_income_usd', importance: 0.038 },
            { feature: 'transit_stops_count', importance: 0.036 },
        ],
    };
}

/**
 * Renderiza marcadores circulares en el mapa, coloreados por valorización.
 * @param {Array<Object>} cells — Lista de celdas
 */
function renderCellMarkers(cells) {
    if (!AppState.markersLayer) return;
    AppState.markersLayer.clearLayers();

    const markersToggle = document.getElementById('markersToggle');
    if (markersToggle && !markersToggle.checked) return;

    cells.forEach(cell => {
        // Estimar valorización a partir del precio promedio
        const estimatedValuation = cell.avg_price ? Math.min(cell.avg_price / 10000, 0.25) : 0.05;
        const color = getValuationColor(estimatedValuation);

        const marker = L.circleMarker([cell.lat, cell.lon], {
            radius: 10,
            fillColor: color,
            color: '#1e2026',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        });

        marker.bindPopup(`
            <div style="min-width:180px">
                <strong style="color:#5cb8b0">Celda ${cell.cell_id}</strong><br>
                <span style="color:#a0a6b5">Ciudad:</span> ${cell.city}<br>
                <span style="color:#a0a6b5">Precio prom.:</span> ${formatPrice(cell.avg_price || 0)}<br>
                <span style="color:#a0a6b5">Coord.:</span> ${cell.lat.toFixed(4)}, ${cell.lon.toFixed(4)}
            </div>
        `);

        marker.on('click', () => onCellClick(cell));

        AppState.markersLayer.addLayer(marker);
    });
}

/**
 * Renderiza una capa de calor (heatmap) sobre el mapa.
 * @param {Array<Object>} cells — Lista de celdas
 */
function renderHeatmap(cells) {
    if (!AppState.heatLayer) return;
    AppState.heatLayer.clearLayers();

    const heatToggle = document.getElementById('heatmapToggle');
    if (heatToggle && !heatToggle.checked) return;

    if (typeof L.heat !== 'function') return;

    const points = cells.map(cell => {
        const intensity = cell.avg_price ? Math.min(cell.avg_price / 2000, 1.0) : 0.3;
        return [cell.lat, cell.lon, intensity];
    });

    const heatLayer = L.heatLayer(points, {
        radius: 35,
        blur: 20,
        maxZoom: 15,
        gradient: {
            0.0: '#1d6b62',
            0.3: '#2a9d8f',
            0.5: '#fbbf24',
            0.7: '#f4823c',
            1.0: '#ef4444'
        }
    });

    AppState.heatLayer.addLayer(heatLayer);
}

/**
 * Maneja el clic en una celda del mapa.
 * @param {Object} cell — Datos de la celda clickeada
 */
async function onCellClick(cell) {
    const horizon = AppState.currentHorizon;

    try {
        let prediction;

        if (AppState.apiOnline) {
            prediction = await fetchApi(`/predict?cell_id=${encodeURIComponent(cell.cell_id)}&horizon=${horizon}`);
        } else {
            prediction = DEMO_PREDICTIONS[cell.cell_id] || {
                cell_id: cell.cell_id,
                predicted_valuation: 0.05,
                lower_bound: 0.02,
                upper_bound: 0.08,
                confidence: 0.70
            };
        }

        showCellDetail(cell, prediction);
    } catch (err) {
        console.error('[Radar] Error al obtener predicción:', err);
        const fallback = DEMO_PREDICTIONS[cell.cell_id] || {
            cell_id: cell.cell_id,
            predicted_valuation: 0.05,
            lower_bound: 0.02,
            upper_bound: 0.08,
            confidence: 0.70
        };
        showCellDetail(cell, fallback);
        showToast('Error al obtener predicción — mostrando datos de demostración', 'error');
    }
}

/**
 * Muestra el panel de detalles de una celda.
 * @param {Object} cell — Datos de la celda
 * @param {Object} prediction — Datos de predicción
 */
function showCellDetail(cell, prediction) {
    const panel = document.getElementById('cellDetailPanel');
    const content = document.getElementById('cellDetailContent');
    if (!panel || !content) return;

    const valColor = getValuationColor(prediction.predicted_valuation);
    const confidencePercent = Math.round((prediction.confidence || 0) * 100);

    content.innerHTML = `
        <h3 style="margin:0 0 12px;color:#5cb8b0;font-size:1.05rem">Celda ${cell.cell_id}</h3>
        <div style="display:grid;gap:8px;font-size:.85rem">
            <div style="display:flex;justify-content:space-between">
                <span style="color:#7a8090">Ciudad</span>
                <span style="color:#e8eaed">${cell.city}</span>
            </div>
            <div style="display:flex;justify-content:space-between">
                <span style="color:#7a8090">Coordenadas</span>
                <span style="color:#e8eaed">${cell.lat.toFixed(4)}, ${cell.lon.toFixed(4)}</span>
            </div>
            <div style="display:flex;justify-content:space-between">
                <span style="color:#7a8090">Precio promedio</span>
                <span style="color:#e8eaed">${formatPrice(cell.avg_price || 0)}</span>
            </div>
            <hr style="border:none;border-top:1px solid #353944;margin:6px 0">
            <div style="display:flex;justify-content:space-between">
                <span style="color:#7a8090">Valorización predicha</span>
                <span style="color:${valColor};font-weight:700">${formatPercent(prediction.predicted_valuation)}</span>
            </div>
            <div style="display:flex;justify-content:space-between">
                <span style="color:#7a8090">Límite inferior</span>
                <span style="color:#ef4444">${formatPercent(prediction.lower_bound)}</span>
            </div>
            <div style="display:flex;justify-content:space-between">
                <span style="color:#7a8090">Límite superior</span>
                <span style="color:#4ade80">${formatPercent(prediction.upper_bound)}</span>
            </div>
            <div style="display:flex;justify-content:space-between">
                <span style="color:#7a8090">Confianza</span>
                <span style="color:#fbbf24">${confidencePercent}%</span>
            </div>
            <hr style="border:none;border-top:1px solid #353944;margin:6px 0">
            <div style="display:flex;justify-content:space-between">
                <span style="color:#7a8090">Horizonte</span>
                <span style="color:#e8eaed">${AppState.currentHorizon} meses</span>
            </div>
        </div>
        <div style="margin-top:14px;padding:10px;background:#0f3b36;border-radius:6px;font-size:.78rem;color:#b8e4e0">
            💡 Haga clic en la pestaña "Explicabilidad" para ver el análisis SHAP de esta celda.
        </div>
    `;

    panel.classList.remove('hidden');

    // Cargar la tendencia histórica de precios de la celda
    loadCellTrend(cell.cell_id);
}

/**
 * Carga y renderiza la tendencia de precios histórica de una celda.
 * @param {string} cellId — ID de la celda
 */
async function loadCellTrend(cellId) {
    const chartEl = document.getElementById('cellTrendChart');
    if (!chartEl) return;

    try {
        let trendData;
        if (AppState.apiOnline) {
            trendData = await fetchApi(`/trends?cell_id=${encodeURIComponent(cellId)}`);
        } else {
            // Tendencia de demostración: serie 2019-2024 con crecimiento del 6%
            const base = 90000 + (cellId.charCodeAt(0) % 8) * 10000;
            trendData = {
                cell_id: cellId,
                price_trend: 0.06,
                series: Array.from({ length: 6 }, (_, i) => ({
                    year: 2019 + i,
                    avg_price: Math.round(base * Math.pow(1.06, i)),
                    transactions: 12 + i * 3,
                })),
            };
        }
        renderPriceTrend('cellTrendChart', trendData);
    } catch (err) {
        console.error('[Radar] Error al cargar tendencia:', err);
        chartEl.innerHTML = '';
    }
}

/* ================================================================
   POBLACIÓN DE SELECTS
   ================================================================ */

/**
 * Puebla los selectores de celdas en todas las pestañas.
 * @param {Array<Object>} cells — Lista de celdas
 */
function populateCellSelects(cells) {
    const selects = [
        'explainCellSelect',
        'simulateCellSelect',
        'compareCellSelect'
    ];

    selects.forEach(selectId => {
        const select = document.getElementById(selectId);
        if (!select) return;

        // Preservar la opción placeholder
        const placeholder = select.querySelector('option[value=""]');
        select.innerHTML = '';
        if (placeholder) {
            select.appendChild(placeholder);
        } else {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = selectId === 'simulateCellSelect'
                ? '-- Todas las celdas --'
                : selectId === 'compareCellSelect'
                    ? '-- Añadir celda --'
                    : '-- Seleccione una celda --';
            select.appendChild(opt);
        }

        cells.forEach(cell => {
            const opt = document.createElement('option');
            opt.value = cell.cell_id;
            opt.textContent = `${cell.cell_id} — ${cell.city} (${formatPrice(cell.avg_price || 0)})`;
            select.appendChild(opt);
        });
    });
}

/* ================================================================
   PESTAÑAS
   ================================================================ */

/**
 * Cambia a la pestaña especificada.
 * @param {string} tabName — Nombre de la pestaña (mapa, explicabilidad, simulador, comparador)
 */
function switchTab(tabName) {
    const panels = document.querySelectorAll('.tab-panel');
    const buttons = document.querySelectorAll('.tab-btn');

    panels.forEach(panel => panel.classList.remove('tab-panel--active'));
    buttons.forEach(btn => btn.classList.remove('tab-btn--active'));

    const targetPanel = document.getElementById(`tab-${tabName}`);
    const targetButton = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);

    if (targetPanel) targetPanel.classList.add('tab-panel--active');
    if (targetButton) targetButton.classList.add('tab-btn--active');

    // Invalidar tamaño del mapa si se cambia a la pestaña del mapa
    if (tabName === 'mapa' && AppState.map) {
        setTimeout(() => AppState.map.invalidateSize(), 100);
    }
}

/* ================================================================
   EXPLICABILIDAD — SHAP
   ================================================================ */

/**
 * Carga el análisis de explicabilidad SHAP para una celda.
 * @param {string} cellId — ID de la celda a analizar
 */
async function loadExplain(cellId) {
    if (!cellId) {
        showToast('Seleccione una celda primero', 'error');
        return;
    }

    hideError('explainError');
    hideLoading('explainLoading');
    document.getElementById('explainContent')?.classList.add('hidden');

    showLoading('explainLoading');

    try {
        let data;

        if (AppState.apiOnline) {
            data = await fetchApi(`/explain?cell_id=${encodeURIComponent(cellId)}`);
        } else {
            // Generar datos SHAP de demostración para la celda seleccionada
            data = generateDemoShap(cellId);
        }

        // Preparar datos para renderShapWaterfall
        const shapEntries = Object.entries(data.shap_values || {});
        const features = shapEntries.map(([k]) => k);
        const values = shapEntries.map(([, v]) => v);
        const prediction = DEMO_PREDICTIONS[cellId]?.predicted_valuation || 0.10;

        const shapData = {
            features: features,
            values: values,
            base_value: data.base_value || 0.085,
            prediction: prediction
        };

        // Renderizar gráfico waterfall
        renderShapWaterfall('shapWaterfallChart', shapData);

        // Preparar datos de importancia
        const importanceData = {
            features: features,
            importances: values
        };
        renderFeatureImportance('featureImportanceChart', importanceData);

        // Renderizar tabla resumen
        renderShapSummaryTable(data, cellId);

        document.getElementById('explainContent')?.classList.remove('hidden');
    } catch (err) {
        console.error('[Radar] Error al cargar explicabilidad:', err);
        showError('explainError', `Error al cargar análisis SHAP: ${err.message}`);
    } finally {
        hideLoading('explainLoading');
    }
}

/**
 * Genera datos SHAP de demostración para una celda específica.
 * @param {string} cellId — ID de la celda
 * @returns {Object} Datos SHAP simulados
 */
function generateDemoShap(cellId) {
    const base = { ...DEMO_SHAP };
    base.cell_id = cellId;
    // Variar ligeramente los valores para cada celda
    const seed = cellId.charCodeAt(cellId.length - 1) || 1;
    const factor = 0.8 + (seed % 10) * 0.04;
    const shapValues = {};
    for (const [k, v] of Object.entries(DEMO_SHAP.shap_values)) {
        shapValues[k] = v * factor;
    }
    base.shap_values = shapValues;
    base.top_drivers = DEMO_SHAP.top_drivers.map(d => ({
        ...d,
        value: d.value * factor
    }));
    return base;
}

/**
 * Renderiza la tabla resumen de factores SHAP.
 * @param {Object} data — Datos SHAP del API
 * @param {string} cellId — ID de la celda analizada
 */
function renderShapSummaryTable(data, cellId) {
    const container = document.getElementById('shapSummaryTable');
    if (!container) return;

    const drivers = data.top_drivers || [];
    const shapValues = data.shap_values || {};
    const baseValue = data.base_value || 0;
    const prediction = DEMO_PREDICTIONS[cellId]?.predicted_valuation || 0.10;

    let html = `
        <table>
            <thead>
                <tr>
                    <th>Característica</th>
                    <th>Valor SHAP</th>
                    <th>Dirección</th>
                    <th>Contribución</th>
                </tr>
            </thead>
            <tbody>
    `;

    // Ordenar por valor absoluto descendente
    const sorted = Object.entries(shapValues)
        .map(([feature, value]) => ({ feature, value }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    const totalContribution = Math.abs(prediction - baseValue) || 1;

    sorted.forEach(({ feature, value }) => {
        const direction = value >= 0 ? 'positivo' : 'negativo';
        const contribution = ((Math.abs(value) / totalContribution) * 100).toFixed(1);
        const dirClass = value >= 0 ? 'shap-bar-pos' : 'shap-bar-neg';
        const dirIcon = value >= 0 ? '▲' : '▼';

        html += `
            <tr>
                <td>${feature}</td>
                <td class="${dirClass}">${value >= 0 ? '+' : ''}${value.toFixed(4)}</td>
                <td>${dirIcon} ${direction}</td>
                <td>${contribution}%</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

/* ================================================================
   SIMULADOR
   ================================================================ */

/**
 * Actualiza los valores mostrados junto a los sliders.
 */
function updateSliderValues() {
    const sliders = [
        { input: 'interestRate',     display: 'interestRateValue',   format: v => v.toFixed(1) },
        { input: 'gdpGrowth',        display: 'gdpGrowthValue',      format: v => v.toFixed(1) },
        { input: 'migrationRate',    display: 'migrationRateValue',   format: v => v.toFixed(1) },
        { input: 'infraInvestment',  display: 'infraInvestmentValue', format: v => Math.round(v).toString() }
    ];

    sliders.forEach(({ input, display, format }) => {
        const inputEl = document.getElementById(input);
        const displayEl = document.getElementById(display);
        if (inputEl && displayEl) {
            displayEl.textContent = format(parseFloat(inputEl.value));
        }
    });
}

/**
 * Ejecuta la simulación con los valores actuales de los sliders.
 */
async function runSimulation() {
    hideError('simError');
    hideLoading('simLoading');
    document.getElementById('simResults')?.classList.add('hidden');

    showLoading('simLoading');

    const interestRate = parseFloat(document.getElementById('interestRate')?.value || 5.5);
    const gdpGrowth = parseFloat(document.getElementById('gdpGrowth')?.value || 2.5);
    const migrationRate = parseFloat(document.getElementById('migrationRate')?.value || 1.0);
    const infraInvestment = parseFloat(document.getElementById('infraInvestment')?.value || 50);
    const cellId = document.getElementById('simulateCellSelect')?.value || '';

    const payload = {
        interest_rate: interestRate,
        gdp_growth: gdpGrowth,
        migration_rate: migrationRate,
        infrastructure_investment: infraInvestment
    };

    try {
        let data;

        if (AppState.apiOnline) {
            data = await fetchApi('/simulate', {
                method: 'POST',
                body: JSON.stringify(payload)
            });
        } else {
            // Generar simulación de demostración ajustada a los sliders
            data = generateDemoSimulation(payload);
        }

        // Filtrar por celda si se seleccionó una
        let adjustedValuations = data.adjusted_valuations || [];
        if (cellId) {
            adjustedValuations = adjustedValuations.filter(v => v.cell_id === cellId);
        }

        if (adjustedValuations.length === 0) {
            showError('simError', 'No se encontraron resultados para la simulación.');
            return;
        }

        // Preparar datos para el gráfico de escenarios
        const cellIds = adjustedValuations.map(v => v.cell_id);
        const baseValues = adjustedValuations.map(v => v.base_valuation);
        const simulatedValues = adjustedValuations.map(v => v.adjusted_valuation);

        renderScenarioChart('scenarioChart',
            { cell_ids: cellIds, values: baseValues },
            { cell_ids: cellIds, values: simulatedValues }
        );

        // Renderizar tabla de detalle
        renderSimDetailTable(adjustedValuations, data.impact_summary);

        document.getElementById('simResults')?.classList.remove('hidden');
        showToast('Simulación completada', 'success');
    } catch (err) {
        console.error('[Radar] Error en simulación:', err);
        showError('simError', `Error al ejecutar simulación: ${err.message}`);
    } finally {
        hideLoading('simLoading');
    }
}

/**
 * Genera datos de simulación de demostración basados en los parámetros.
 * @param {Object} params — Parámetros de simulación
 * @returns {Object} Resultado simulado
 */
function generateDemoSimulation(params) {
    const { interest_rate, gdp_growth, migration_rate, infrastructure_investment } = params;

    // Calcular factor de ajuste basado en parámetros
    const interestFactor = -(interest_rate - 5.5) * 0.008;
    const gdpFactor = (gdp_growth - 2.5) * 0.006;
    const migrationFactor = (migration_rate - 1.0) * 0.004;
    const infraFactor = (infrastructure_investment - 50) * 0.0003;

    const totalAdjustment = interestFactor + gdpFactor + migrationFactor + infraFactor;

    const adjustedValuations = DEMO_SIMULATION.adjusted_valuations.map(item => {
        const newAdjusted = Math.max(0, item.base_valuation + totalAdjustment + (Math.random() - 0.5) * 0.005);
        return {
            ...item,
            adjusted_valuation: newAdjusted,
            change: newAdjusted - item.base_valuation
        };
    });

    const changes = adjustedValuations.map(v => v.change);
    const avgChange = changes.reduce((a, b) => a + b, 0) / changes.length;
    const maxIdx = changes.indexOf(Math.max(...changes));
    const minIdx = changes.indexOf(Math.min(...changes));

    return {
        adjusted_valuations: adjustedValuations,
        impact_summary: {
            promedio_cambio: avgChange,
            celda_mayor_impacto: adjustedValuations[maxIdx]?.cell_id || '',
            celda_menor_impacto: adjustedValuations[minIdx]?.cell_id || '',
            total_celdas_afectadas: adjustedValuations.length,
            direccion_dominante: avgChange >= 0 ? 'positiva' : 'negativa'
        }
    };
}

/**
 * Renderiza la tabla de detalle de la simulación.
 * @param {Array<Object>} valuations — Lista de valorizaciones ajustadas
 * @param {Object} summary — Resumen de impacto
 */
function renderSimDetailTable(valuations, summary) {
    const container = document.getElementById('simDetailTable');
    if (!container) return;

    let html = `
        <table>
            <thead>
                <tr>
                    <th>Celda</th>
                    <th>Valorización Base</th>
                    <th>Valorización Ajustada</th>
                    <th>Cambio</th>
                </tr>
            </thead>
            <tbody>
    `;

    valuations.forEach(v => {
        const changeClass = v.change >= 0 ? 'shap-bar-pos' : 'shap-bar-neg';
        const changeIcon = v.change >= 0 ? '▲' : '▼';

        html += `
            <tr>
                <td>${v.cell_id}</td>
                <td>${formatPercent(v.base_valuation)}</td>
                <td>${formatPercent(v.adjusted_valuation)}</td>
                <td class="${changeClass}">${changeIcon} ${formatPercent(Math.abs(v.change))}</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    if (summary) {
        const dirText = summary.direccion_dominante === 'positiva' ? 'Positiva ▲' : 'Negativa ▼';
        const dirClass = summary.direccion_dominante === 'positiva' ? 'shap-bar-pos' : 'shap-bar-neg';

        html += `
            <div style="margin-top:16px;padding:14px;background:#262932;border-radius:10px;display:grid;grid-template-columns:repeat(2,1fr);gap:10px;font-size:.82rem">
                <div>
                    <span style="color:#7a8090">Cambio promedio:</span>
                    <span style="color:#e8eaed;font-weight:700">${formatPercent(summary.promedio_cambio)}</span>
                </div>
                <div>
                    <span style="color:#7a8090">Dirección dominante:</span>
                    <span class="${dirClass}" style="font-weight:700">${dirText}</span>
                </div>
                <div>
                    <span style="color:#7a8090">Mayor impacto:</span>
                    <span style="color:#5cb8b0">${summary.celda_mayor_impacto}</span>
                </div>
                <div>
                    <span style="color:#7a8090">Menor impacto:</span>
                    <span style="color:#f9b87a">${summary.celda_menor_impacto}</span>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

/* ================================================================
   COMPARADOR
   ================================================================ */

/**
 * Añade una celda a la lista de comparación.
 * @param {string} cellId — ID de la celda a añadir
 */
function addCompareChip(cellId) {
    if (!cellId) {
        showToast('Seleccione una celda para añadir', 'error');
        return;
    }

    if (AppState.selectedCells.includes(cellId)) {
        showToast('La celda ya está en la lista de comparación', 'info');
        return;
    }

    if (AppState.selectedCells.length >= 8) {
        showToast('Máximo 8 celdas para comparar', 'error');
        return;
    }

    AppState.selectedCells.push(cellId);
    renderCompareChips();

    // Resetear el select
    const select = document.getElementById('compareCellSelect');
    if (select) select.value = '';
}

/**
 * Elimina una celda de la lista de comparación.
 * @param {string} cellId — ID de la celda a remover
 */
function removeCompareChip(cellId) {
    AppState.selectedCells = AppState.selectedCells.filter(id => id !== cellId);
    renderCompareChips();
}

/**
 * Renderiza los chips de celdas seleccionadas para comparación.
 */
function renderCompareChips() {
    const container = document.getElementById('cellChips');
    if (!container) return;

    container.innerHTML = '';

    AppState.selectedCells.forEach(cellId => {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.innerHTML = `
            ${cellId}
            <span class="chip-remove" data-cell-id="${cellId}" title="Quitar celda">✕</span>
        `;

        chip.querySelector('.chip-remove').addEventListener('click', (e) => {
            e.stopPropagation();
            removeCompareChip(cellId);
        });

        container.appendChild(chip);
    });
}

/**
 * Ejecuta la comparación de celdas seleccionadas.
 */
async function loadCompare() {
    if (AppState.selectedCells.length < 2) {
        showToast('Seleccione al menos 2 celdas para comparar', 'error');
        return;
    }

    hideError('compareError');
    hideLoading('compareLoading');
    document.getElementById('compareResults')?.classList.add('hidden');

    showLoading('compareLoading');

    const horizon = document.getElementById('compareHorizonSelect')?.value || 12;
    const cellIdsParam = AppState.selectedCells.join(',');

    try {
        let data;

        if (AppState.apiOnline) {
            data = await fetchApi(`/compare?project_id=web&cell_ids=${encodeURIComponent(cellIdsParam)}`);
        } else {
            // Generar comparación de demostración
            data = generateDemoCompare(AppState.selectedCells, horizon);
        }

        // Renderizar gráfico de valorizaciones
        const predictions = data.predictions || [];
        const cellIds = predictions.map(p => p.cell_id);
        const values = predictions.map(p => p.predicted_valuation);

        renderValuationChart('valuationChart', {
            cell_ids: cellIds,
            values: values,
            horizon: parseInt(horizon)
        });

        // Renderizar tabla comparativa
        renderCompareTable(data);

        document.getElementById('compareResults')?.classList.remove('hidden');
        showToast('Comparación completada', 'success');
    } catch (err) {
        console.error('[Radar] Error en comparación:', err);
        showError('compareError', `Error al comparar celdas: ${err.message}`);
    } finally {
        hideLoading('compareLoading');
    }
}

/**
 * Genera datos de comparación de demostración.
 * @param {string[]} cellIds — IDs de celdas a comparar
 * @param {number} horizon — Horizonte en meses
 * @returns {Object} Datos de comparación simulados
 */
function generateDemoCompare(cellIds, horizon) {
    const horizonFactor = horizon / 12;

    const predictions = cellIds.map(cellId => {
        const base = DEMO_PREDICTIONS[cellId] || {
            cell_id: cellId,
            predicted_valuation: 0.05,
            lower_bound: 0.02,
            upper_bound: 0.08,
            confidence: 0.70
        };
        const adjusted = base.predicted_valuation * horizonFactor;
        return {
            ...base,
            predicted_valuation: adjusted,
            lower_bound: base.lower_bound * horizonFactor,
            upper_bound: base.upper_bound * horizonFactor
        };
    });

    const comparisonTable = predictions.map(p => {
        const cell = AppState.cells.find(c => c.cell_id === p.cell_id);
        return {
            cell_id: p.cell_id,
            city: cell?.city || 'N/D',
            avg_price: cell?.avg_price || 0,
            predicted_valuation: p.predicted_valuation,
            lower_bound: p.lower_bound,
            upper_bound: p.upper_bound,
            confidence: p.confidence
        };
    });

    return {
        project_id: 'web',
        cell_ids: cellIds,
        predictions: predictions,
        comparison_table: comparisonTable
    };
}

/**
 * Renderiza la tabla comparativa de celdas.
 * @param {Object} data — Datos de comparación del API
 */
function renderCompareTable(data) {
    const container = document.getElementById('compareTable');
    if (!container) return;

    const table = data.comparison_table || data.predictions || [];

    let html = `
        <table>
            <thead>
                <tr>
                    <th>Celda</th>
                    <th>Ciudad</th>
                    <th>Precio Prom.</th>
                    <th>Valorización</th>
                    <th>Límite Inf.</th>
                    <th>Límite Sup.</th>
                    <th>Confianza</th>
                </tr>
            </thead>
            <tbody>
    `;

    table.forEach(row => {
        const valColor = getValuationColor(row.predicted_valuation);
        const confidencePercent = Math.round((row.confidence || 0) * 100);

        html += `
            <tr>
                <td>${row.cell_id}</td>
                <td>${row.city || 'N/D'}</td>
                <td>${formatPrice(row.avg_price || 0)}</td>
                <td style="color:${valColor};font-weight:700">${formatPercent(row.predicted_valuation)}</td>
                <td style="color:#ef4444">${formatPercent(row.lower_bound)}</td>
                <td style="color:#4ade80">${formatPercent(row.upper_bound)}</td>
                <td style="color:#fbbf24">${confidencePercent}%</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

/* ================================================================
   INICIALIZACIÓN
   ================================================================ */

/**
 * Carga el índice de inteligencia urbana y colorea los marcadores por índice.
 */
async function loadIndexLayer() {
    try {
        let indexData;
        if (AppState.apiOnline) {
            indexData = await fetchApi('/index');
        } else {
            indexData = generateDemoIndex();
            // Construir cells con índice sintético
            indexData.cells = DEMO_CELLS.map((c, i) => ({
                cell_id: c.cell_id,
                city: c.city,
                lat: c.lat,
                lon: c.lon,
                index: 40 + ((i * 7) % 50),
            }));
        }

        if (!AppState.markersLayer) return;
        AppState.markersLayer.clearLayers();

        const markersToggle = document.getElementById('markersToggle');
        if (markersToggle && !markersToggle.checked) return;

        const cells = indexData.cells || [];
        cells.forEach(cell => {
            const idx = cell.index ?? 50;
            const color = idx >= 70 ? '#4ade80' : idx >= 50 ? '#2a9d8f' : idx >= 35 ? '#fbbf24' : '#ef4444';

            const marker = L.circleMarker([cell.lat, cell.lon], {
                radius: 10,
                fillColor: color,
                color: '#1e2026',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.85
            });

            marker.bindPopup(`
                <div style="min-width:180px">
                    <strong style="color:#5cb8b0">Celda ${cell.cell_id}</strong><br>
                    <span style="color:#a0a6b5">Ciudad:</span> ${cell.city}<br>
                    <span style="color:#a0a6b5">Índice Inteligencia Urbana:</span> <b style="color:#f9b87a">${idx}</b>/100
                </div>
            `);

            const demoCell = DEMO_CELLS.find(c => c.cell_id === cell.cell_id) || {
                cell_id: cell.cell_id, city: cell.city, lat: cell.lat, lon: cell.lon, avg_price: 100000
            };
            marker.on('click', () => onCellClick(demoCell));

            AppState.markersLayer.addLayer(marker);
        });

        updateLegend('index');
    } catch (err) {
        console.error('[Radar] Error al cargar capa de índice:', err);
        showToast('Error al cargar la capa de índice', 'error');
    }
}

/**
 * Actualiza la leyenda del mapa según la capa activa.
 * @param {string} mode — 'valuation' | 'index'
 */
function updateLegend(mode) {
    const legend = document.getElementById('mapLegend');
    if (!legend) return;

    const title = legend.querySelector('.legend-title');
    const items = legend.querySelectorAll('.legend-item');
    if (!title || items.length < 4) return;

    if (mode === 'index') {
        title.textContent = 'Índice Urbano';
        items[0].innerHTML = '<span class="legend-dot" style="background:#4ade80"></span> Muy alto (70+)';
        items[1].innerHTML = '<span class="legend-dot" style="background:#2a9d8f"></span> Alto (50-70)';
        items[2].innerHTML = '<span class="legend-dot" style="background:#fbbf24"></span> Medio (35-50)';
        items[3].innerHTML = '<span class="legend-dot" style="background:#ef4444"></span> Bajo (<35)';
    } else {
        title.textContent = 'Valorización';
        items[0].innerHTML = '<span class="legend-dot" style="background:#4ade80"></span> Alta';
        items[1].innerHTML = '<span class="legend-dot" style="background:#2a9d8f"></span> Media';
        items[2].innerHTML = '<span class="legend-dot" style="background:#fbbf24"></span> Baja';
        items[3].innerHTML = '<span class="legend-dot" style="background:#ef4444"></span> Negativa';
    }
}

/**
 * Configura todos los event listeners de la interfaz.
 */
function setupEventListeners() {
    // --- Pestañas ---
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            if (tabName) switchTab(tabName);
        });
    });

    // --- Mapa: toggles ---
    const heatmapToggle = document.getElementById('heatmapToggle');
    if (heatmapToggle) {
        heatmapToggle.addEventListener('change', () => {
            renderHeatmap(AppState.cells);
        });
    }

    const markersToggle = document.getElementById('markersToggle');
    if (markersToggle) {
        markersToggle.addEventListener('change', () => {
            renderCellMarkers(AppState.cells);
        });
    }

    // --- Mapa: horizonte ---
    const horizonSelect = document.getElementById('horizonSelect');
    if (horizonSelect) {
        horizonSelect.addEventListener('change', () => {
            AppState.currentHorizon = parseInt(horizonSelect.value) || 12;
            showToast(`Horizonte cambiado a ${AppState.currentHorizon} meses`, 'info');
        });
    }

    // --- Mapa: recargar ---
    const refreshBtn = document.getElementById('refreshCellsBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async () => {
            await checkApiHealth();
            await loadCells();
        });
    }

    // --- Mapa: colorear por índice/valorización ---
    const colorBySelect = document.getElementById('colorBySelect');
    if (colorBySelect) {
        colorBySelect.addEventListener('change', async () => {
            if (colorBySelect.value === 'index') {
                await loadIndexLayer();
            } else {
                renderCellMarkers(AppState.cells);
                updateLegend('valuation');
            }
        });
    }

    // --- Mapa: toggle de capa de índice ---
    const indexToggle = document.getElementById('indexToggle');
    if (indexToggle) {
        indexToggle.addEventListener('change', async () => {
            if (indexToggle.checked) {
                await loadIndexLayer();
            } else {
                renderCellMarkers(AppState.cells);
                updateLegend('valuation');
            }
        });
    }

    // --- Dashboard: cargar al abrir pestaña ---
    const dashboardTab = document.querySelector('.tab-btn[data-tab="dashboard"]');
    if (dashboardTab && !AppState.dashboardLoaded) {
        dashboardTab.addEventListener('click', () => {
            loadDashboard();
            AppState.dashboardLoaded = true;
        });
    }

    // --- Explicabilidad ---
    const explainBtn = document.getElementById('explainFetchBtn');
    if (explainBtn) {
        explainBtn.addEventListener('click', () => {
            const cellId = document.getElementById('explainCellSelect')?.value;
            loadExplain(cellId);
        });
    }

    const explainSelect = document.getElementById('explainCellSelect');
    if (explainSelect) {
        explainSelect.addEventListener('change', () => {
            const cellId = explainSelect.value;
            if (cellId) {
                loadExplain(cellId);
            }
        });
    }

    // --- Simulador: sliders ---
    const sliderIds = ['interestRate', 'gdpGrowth', 'migrationRate', 'infraInvestment'];
    sliderIds.forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            slider.addEventListener('input', updateSliderValues);
        }
    });

    // --- Simulador: botón ---
    const simulateBtn = document.getElementById('simulateBtn');
    if (simulateBtn) {
        simulateBtn.addEventListener('click', runSimulation);
    }

    // --- Comparador: añadir ---
    const addCompareBtn = document.getElementById('addCompareCellBtn');
    if (addCompareBtn) {
        addCompareBtn.addEventListener('click', () => {
            const cellId = document.getElementById('compareCellSelect')?.value;
            if (cellId) addCompareChip(cellId);
        });
    }

    // --- Comparador: comparar ---
    const compareBtn = document.getElementById('compareBtn');
    if (compareBtn) {
        compareBtn.addEventListener('click', loadCompare);
    }

    // --- Comparador: limpiar ---
    const clearCompareBtn = document.getElementById('clearCompareBtn');
    if (clearCompareBtn) {
        clearCompareBtn.addEventListener('click', () => {
            AppState.selectedCells = [];
            renderCompareChips();
            document.getElementById('compareResults')?.classList.add('hidden');
            showToast('Lista de comparación limpiada', 'info');
        });
    }

    // --- Comparador: Enter en select añade celda ---
    const compareSelect = document.getElementById('compareCellSelect');
    if (compareSelect) {
        compareSelect.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const cellId = compareSelect.value;
                if (cellId) addCompareChip(cellId);
            }
        });
    }
}

/**
 * Punto de entrada principal de la aplicación.
 * Inicializa el mapa, verifica el API y carga las celdas.
 */
async function init() {
    console.log('[Radar] Inicializando Radar de Valorización Urbana…');

    // Inicializar valores de sliders
    updateSliderValues();

    // Configurar event listeners
    setupEventListeners();

    // Inicializar mapa
    initMap();

    // Verificar salud del API
    await checkApiHealth();

    // Cargar celdas
    await loadCells();

    // Cargar dashboard (pestaña inicial)
    await loadDashboard();

    console.log('[Radar] Aplicación lista.');
}

/* ================================================================
   ARRANQUE — DOMContentLoaded
   ================================================================ */

document.addEventListener('DOMContentLoaded', init);
