/* ================================================================
   Radar de Valorización Urbana — Panel de Visualizaciones Plotly
   Módulo: plotly_panel.js
   ================================================================ */

const URBAN_COLORS = {
    teal:      '#2a9d8f',
    tealLight: '#5cb8b0',
    tealDark:  '#1d6b62',
    orange:    '#f4823c',
    orangeLight:'#f9b87a',
    orangeDark:'#c45f1a',
    gray:      '#4a4f5c',
    grayLight: '#7a8090',
    grayDark:  '#353944',
    success:   '#4ade80',
    danger:    '#ef4444',
    warning:   '#fbbf24',
    bg:        '#1e2026',
    text:      '#e8eaed',
    grid:      '#353944'
};

const PLOTLY_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
        color: URBAN_COLORS.text,
        family: 'Segoe UI, system-ui, sans-serif',
        size: 12
    },
    margin: { t: 40, r: 20, b: 50, l: 60 }
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false,
    displaylogo: false
};

/* ================================================================
   renderShapWaterfall — Gráfico Waterfall para valores SHAP
   ================================================================ */
function renderShapWaterfall(elementId, shapData) {
    const el = document.getElementById(elementId);
    if (!el || !shapData) return;

    const features = shapData.features || [];
    const values   = shapData.values || [];
    const baseValue = shapData.base_value || 0;
    const prediction = shapData.prediction || 0;

    if (features.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Sin datos SHAP disponibles</p>';
        return;
    }

    const cumulative = [];
    let running = baseValue;
    cumulative.push(baseValue);
    for (let i = 0; i < values.length; i++) {
        running += values[i];
        cumulative.push(running);
    }

    const traces = [];

    for (let i = 0; i < features.length; i++) {
        const isPositive = values[i] >= 0;
        const yStart = cumulative[i];
        const yEnd = cumulative[i + 1];

        traces.push({
            type: 'bar',
            x: [features[i]],
            y: [Math.abs(yEnd - yStart)],
            base: Math.min(yStart, yEnd),
            marker: {
                color: isPositive ? URBAN_COLORS.success : URBAN_COLORS.danger,
                opacity: 0.85
            },
            text: [`${isPositive ? '+' : ''}${values[i].toFixed(4)}`],
            textposition: 'outside',
            textfont: { size: 10, color: URBAN_COLORS.grayLight },
            name: features[i],
            showlegend: false,
            hovertemplate: `<b>${features[i]}</b><br>SHAP: ${values[i].toFixed(4)}<br>Contribución: ${((values[i] / Math.abs(prediction - baseValue || 1)) * 100).toFixed(1)}%<extra></extra>`
        });
    }

    traces.push({
        type: 'bar',
        x: ['Valor Base'],
        y: [baseValue],
        marker: { color: URBAN_COLORS.gray },
        text: [baseValue.toFixed(4)],
        textposition: 'outside',
        textfont: { size: 10, color: URBAN_COLORS.grayLight },
        name: 'Base',
        showlegend: false
    });

    traces.push({
        type: 'bar',
        x: ['Predicción'],
        y: [prediction],
        marker: { color: URBAN_COLORS.orange },
        text: [prediction.toFixed(4)],
        textposition: 'outside',
        textfont: { size: 10, color: URBAN_COLORS.orangeLight },
        name: 'Predicción',
        showlegend: false
    });

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'Contribución SHAP por Característica',
            font: { color: URBAN_COLORS.tealLight, size: 14 }
        },
        xaxis: {
            tickangle: -35,
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        yaxis: {
            title: 'Valor SHAP',
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        barmode: 'stack',
        bargap: 0.3
    };

    Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderValuationChart — Gráfico de barras comparando valorizaciones
   ================================================================ */
function renderValuationChart(elementId, data) {
    const el = document.getElementById(elementId);
    if (!el || !data) return;

    const cellIds = data.cell_ids || [];
    const values  = data.values || [];
    const horizon = data.horizon || 12;

    if (cellIds.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Seleccione celdas para comparar</p>';
        return;
    }

    const barColors = values.map(v => {
        if (v > 0.15) return URBAN_COLORS.success;
        if (v > 0.05) return URBAN_COLORS.teal;
        if (v > 0) return URBAN_COLORS.warning;
        return URBAN_COLORS.danger;
    });

    const trace = {
        type: 'bar',
        x: cellIds.map(c => `Celda ${c}`),
        y: values.map(v => (v * 100).toFixed(2)),
        marker: {
            color: barColors,
            line: { color: URBAN_COLORS.tealDark, width: 1 }
        },
        text: values.map(v => `${(v * 100).toFixed(2)}%`),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.grayLight, size: 11 },
        hovertemplate: '<b>Celda %{x}</b><br>Valorización: %{y}%<br>Horizonte: ' + horizon + ' meses<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: `Predicción de Valorización (Horizonte: ${horizon} meses)`,
            font: { color: URBAN_COLORS.tealLight, size: 14 }
        },
        xaxis: {
            title: 'Celda',
            gridcolor: URBAN_COLORS.grid,
            tickangle: cellIds.length > 6 ? -35 : 0
        },
        yaxis: {
            title: 'Valorización Predicha (%)',
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        bargap: 0.35
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderScenarioChart — Comparación de escenario base vs simulado
   ================================================================ */
function renderScenarioChart(elementId, baseline, scenario) {
    const el = document.getElementById(elementId);
    if (!el || !baseline || !scenario) return;

    const cellIds = baseline.cell_ids || [];
    const baseVals = baseline.values || [];
    const scenVals = scenario.values || [];

    if (cellIds.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Ejecute la simulación para ver resultados</p>';
        return;
    }

    const traceBase = {
        type: 'bar',
        name: 'Escenario Base',
        x: cellIds.map(c => `Celda ${c}`),
        y: baseVals.map(v => (v * 100).toFixed(2)),
        marker: { color: URBAN_COLORS.teal, opacity: 0.8 },
        text: baseVals.map(v => `${(v * 100).toFixed(1)}%`),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.tealLight, size: 9 },
        hovertemplate: '<b>Base — Celda %{x}</b><br>Valorización: %{y}%<extra></extra>'
    };

    const traceScenario = {
        type: 'bar',
        name: 'Escenario Simulado',
        x: cellIds.map(c => `Celda ${c}`),
        y: scenVals.map(v => (v * 100).toFixed(2)),
        marker: { color: URBAN_COLORS.orange, opacity: 0.85 },
        text: scenVals.map(v => `${(v * 100).toFixed(1)}%`),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.orangeLight, size: 9 },
        hovertemplate: '<b>Simulado — Celda %{x}</b><br>Valorización: %{y}%<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'Comparación: Escenario Base vs Simulado',
            font: { color: URBAN_COLORS.tealLight, size: 14 }
        },
        xaxis: {
            title: 'Celda',
            gridcolor: URBAN_COLORS.grid,
            tickangle: cellIds.length > 6 ? -35 : 0
        },
        yaxis: {
            title: 'Valorización Predicha (%)',
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        barmode: 'group',
        bargroupgap: 0.15,
        legend: {
            orientation: 'h',
            y: -0.3,
            font: { color: URBAN_COLORS.grayLight }
        }
    };

    Plotly.newPlot(el, [traceBase, traceScenario], layout, PLOTLY_CONFIG);
}

/* ================================================================
   renderFeatureImportance — Gráfico de barras horizontales
   ================================================================ */
function renderFeatureImportance(elementId, importanceData) {
    const el = document.getElementById(elementId);
    if (!el || !importanceData) return;

    let features = importanceData.features || [];
    let importances = importanceData.importances || [];

    if (features.length === 0) {
        el.innerHTML = '<p style="color:#7a8090;text-align:center;padding:40px">Sin datos de importancia disponibles</p>';
        return;
    }

    const paired = features.map((f, i) => ({ feature: f, importance: importances[i] }));
    paired.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));

    features = paired.map(p => p.feature);
    importances = paired.map(p => p.importance);

    const colors = importances.map(v => v >= 0 ? URBAN_COLORS.teal : URBAN_COLORS.orange);

    const trace = {
        type: 'bar',
        orientation: 'h',
        y: features,
        x: importances.map(v => Math.abs(v).toFixed(4)),
        marker: {
            color: colors,
            line: { color: URBAN_COLORS.grayDark, width: 1 }
        },
        text: importances.map(v => {
            const sign = v >= 0 ? '+' : '';
            return `${sign}${(v * 100).toFixed(2)}%`;
        }),
        textposition: 'outside',
        textfont: { color: URBAN_COLORS.grayLight, size: 10 },
        hovertemplate: '<b>%{y}</b><br>Importancia: %{x}<extra></extra>'
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        title: {
            text: 'Importancia de Características',
            font: { color: URBAN_COLORS.tealLight, size: 14 }
        },
        xaxis: {
            title: 'Importancia Relativa',
            gridcolor: URBAN_COLORS.grid,
            zerolinecolor: URBAN_COLORS.grid
        },
        yaxis: {
            gridcolor: URBAN_COLORS.grid,
            autorange: 'reversed'
        },
        bargap: 0.4,
        margin: { ...PLOTLY_LAYOUT.margin, l: 140 }
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}
