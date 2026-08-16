"""
Tests para los módulos ETL del Radar de Valorización Urbana.

Cobertura:
- Rejilla canónica H3 (grid.py)
- Generación de transacciones inmobiliarias
- Generación de datos de movilidad
- Generación de datos satelitales
- Generación de datos de servicios
- Generación de datos socioeconómicos
- Feature engineering

Todos los tests usan paths temporales (tmp_path) para no contaminar
los datos reales del proyecto.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def base_config():
    """Carga la configuración del proyecto desde config.yaml."""
    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def tmp_config(base_config, tmp_path):
    """Configuración con paths apuntando al directorio temporal."""
    config = copy.deepcopy(base_config)
    config["paths"] = {
        "processed_dir": str(tmp_path),
        "transactions_parquet": str(tmp_path / "transactions.parquet"),
        "transactions_csv": str(tmp_path / "transactions.csv"),
        "transactions_geojson": str(tmp_path / "transactions.geojson"),
        "mobility_csv": str(tmp_path / "mobility.csv"),
        "satellite_csv": str(tmp_path / "satellite_features.csv"),
        "services_csv": str(tmp_path / "services.csv"),
        "socioeconomic_csv": str(tmp_path / "socioeconomic.csv"),
        "features_csv": str(tmp_path / "features.csv"),
        "features_parquet": str(tmp_path / "features.parquet"),
    }
    return config


@pytest.fixture
def etl_transactions(tmp_config):
    """Ejecuta el ETL de transacciones y retorna el DataFrame resultante."""
    from src.etl.etl_transactions import run
    return run(tmp_config)


@pytest.fixture
def etl_mobility(tmp_config):
    """Ejecuta el ETL de movilidad y retorna el DataFrame resultante."""
    from src.etl.etl_mobility import run
    return run(tmp_config)


@pytest.fixture
def etl_satellite(tmp_config):
    """Ejecuta el ETL satelital y retorna el DataFrame resultante."""
    from src.etl.etl_satellite import run
    return run(tmp_config)


@pytest.fixture
def etl_services(tmp_config):
    """Ejecuta el ETL de servicios y retorna el DataFrame resultante."""
    from src.etl.etl_services import run
    return run(tmp_config)


@pytest.fixture
def etl_socioeconomic(tmp_config):
    """Ejecuta el ETL socioeconómico y retorna el DataFrame resultante."""
    from src.etl.etl_socioeconomic import run
    return run(tmp_config)


@pytest.fixture
def canonical_grid(tmp_config):
    """Genera la rejilla canónica."""
    from src.etl.grid import generate_grid
    return generate_grid(tmp_config)


@pytest.fixture
def features_df(tmp_config):
    """Ejecuta el pipeline ETL completo + build_features en tmp_path."""
    from src.etl.etl_transactions import run as run_tx
    from src.etl.etl_mobility import run as run_mob
    from src.etl.etl_satellite import run as run_sat
    from src.etl.etl_services import run as run_srv
    from src.etl.etl_socioeconomic import run as run_soc
    from src.features.build_features import build_features

    run_tx(tmp_config)
    run_mob(tmp_config)
    run_sat(tmp_config)
    run_srv(tmp_config)
    run_soc(tmp_config)
    return build_features(tmp_config)


# =============================================================================
# Tests: Rejilla Canónica
# =============================================================================

class TestCanonicalGrid:
    """Tests para la rejilla H3 canónica."""

    def test_grid_count(self, canonical_grid, base_config):
        """El número de celdas debe coincidir con la configuración."""
        expected = sum(c["num_cells"] for c in base_config["cities"].values())
        assert len(canonical_grid) == expected, (
            f"Esperaba {expected} celdas, obtuvo {len(canonical_grid)}"
        )

    def test_grid_columns(self, canonical_grid):
        """Debe contener cell_id, city, lat, lon."""
        for col in ["cell_id", "city", "lat", "lon"]:
            assert col in canonical_grid.columns, f"Falta columna {col}"

    def test_grid_no_duplicates(self, canonical_grid):
        """No debe haber cell_ids duplicados."""
        assert canonical_grid["cell_id"].is_unique, "Hay cell_ids duplicados"

    def test_grid_no_nulls(self, canonical_grid):
        """No debe haber valores nulos."""
        assert canonical_grid.isnull().sum().sum() == 0, "Hay valores nulos"

    def test_grid_cities(self, canonical_grid):
        """Debe incluir Quito y Guayaquil."""
        cities = set(canonical_grid["city"])
        assert "Quito" in cities and "Guayaquil" in cities

    def test_grid_deterministic(self, tmp_config):
        """La rejilla debe ser reproducible (misma semilla, misma rejilla)."""
        from src.etl.grid import generate_grid
        g1 = generate_grid(tmp_config)
        g2 = generate_grid(tmp_config)
        pd.testing.assert_frame_equal(g1, g2)


# =============================================================================
# Tests: Transacciones Inmobiliarias
# =============================================================================

class TestETLTransactions:
    """Tests para el módulo etl_transactions.py."""

    def test_transaction_count(self, etl_transactions):
        """Debe generar al menos una transacción."""
        assert len(etl_transactions) > 0, "Debe generar al menos una transacción"

    def test_transaction_columns(self, etl_transactions):
        """Debe contener todas las columnas esperadas."""
        expected_cols = {
            "transaction_id", "date", "lat", "lon", "cell_id",
            "property_type", "price_usd", "area_m2", "price_per_m2",
            "bedrooms", "bathrooms", "city",
        }
        actual_cols = set(etl_transactions.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_no_null_transactions(self, etl_transactions):
        """No debe haber valores nulos en columnas críticas."""
        critical_cols = ["transaction_id", "date", "lat", "lon", "cell_id",
                         "property_type", "price_usd", "area_m2", "city"]
        nulls = etl_transactions[critical_cols].isnull().sum()
        assert nulls.sum() == 0, f"Valores nulos encontrados:\n{nulls[nulls > 0]}"

    def test_valid_prices(self, etl_transactions):
        """Los precios deben ser positivos y realistas."""
        assert (etl_transactions["price_usd"] > 0).all(), "Precios no deben ser negativos"
        assert etl_transactions["price_usd"].max() < 2_000_000, "Precio máximo irreal"

    def test_price_per_m2_calculated(self, etl_transactions):
        """price_per_m2 debe ser consistente con price_usd / area_m2."""
        calculated = etl_transactions["price_usd"] / etl_transactions["area_m2"]
        np.testing.assert_allclose(
            etl_transactions["price_per_m2"].values,
            calculated.values,
            rtol=0.01,
        )

    def test_property_types_valid(self, etl_transactions):
        """Los tipos de propiedad deben ser válidos."""
        valid_types = {"apartment", "house", "lot"}
        actual_types = set(etl_transactions["property_type"].unique())
        assert actual_types.issubset(valid_types), f"Tipos inválidos: {actual_types}"

    def test_cities_valid(self, etl_transactions):
        """Las ciudades deben ser Quito o Guayaquil."""
        valid_cities = {"Quito", "Guayaquil"}
        actual_cities = set(etl_transactions["city"].unique())
        assert actual_cities.issubset(valid_cities), f"Ciudades inválidas: {actual_cities}"

    def test_coordinates_within_bbox(self, etl_transactions, base_config):
        """Las coordenadas deben estar dentro de los bounding boxes configurados."""
        for city_name, city_config in base_config["cities"].items():
            city_mask = etl_transactions["city"] == city_name
            if city_mask.sum() == 0:
                continue
            city_data = etl_transactions[city_mask]
            bbox = city_config["bbox"]
            assert city_data["lat"].between(bbox["min_lat"], bbox["max_lat"]).all()
            assert city_data["lon"].between(bbox["min_lon"], bbox["max_lon"]).all()

    def test_date_range(self, etl_transactions, base_config):
        """Las fechas deben estar dentro del rango configurado."""
        start = pd.to_datetime(base_config["transactions"]["start_date"])
        end = pd.to_datetime(base_config["transactions"]["end_date"])
        dates = pd.to_datetime(etl_transactions["date"])
        assert dates.min() >= start
        assert dates.max() <= end


# =============================================================================
# Tests: Movilidad
# =============================================================================

class TestETLMobility:
    """Tests para el módulo etl_mobility.py."""

    def test_mobility_count(self, etl_mobility):
        """Debe generar datos para múltiples celdas."""
        assert len(etl_mobility) > 0, "Debe generar datos de movilidad"

    def test_mobility_columns(self, etl_mobility):
        """Debe contener las columnas esperadas."""
        expected_cols = {
            "cell_id", "avg_travel_time_cbd_min", "transit_stops_count",
            "walkability_score", "connectivity_index",
        }
        actual_cols = set(etl_mobility.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_travel_time_positive(self, etl_mobility):
        """El tiempo de viaje debe ser positivo."""
        assert (etl_mobility["avg_travel_time_cbd_min"] > 0).all()

    def test_walkability_range(self, etl_mobility):
        """El índice de caminabilidad debe estar entre 0 y 100."""
        assert etl_mobility["walkability_score"].between(0, 100).all()

    def test_no_null_mobility(self, etl_mobility):
        """No debe haber valores nulos."""
        assert etl_mobility.isnull().sum().sum() == 0


# =============================================================================
# Tests: Satélite
# =============================================================================

class TestETLSatellite:
    """Tests para el módulo etl_satellite.py."""

    def test_satellite_count(self, etl_satellite):
        """Debe generar datos satelitales."""
        assert len(etl_satellite) > 0, "Debe generar datos satelitales"

    def test_satellite_columns(self, etl_satellite):
        """Debe contener las columnas esperadas."""
        expected_cols = {"cell_id", "ndvi_mean", "built_up_index", "green_space_ratio"}
        actual_cols = set(etl_satellite.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_ndvi_range(self, etl_satellite):
        """NDVI debe estar entre -1 y 1."""
        assert etl_satellite["ndvi_mean"].between(-1, 1).all()

    def test_green_space_range(self, etl_satellite):
        """El ratio de espacio verde debe estar entre 0 y 1."""
        assert etl_satellite["green_space_ratio"].between(0, 1).all()

    def test_no_null_satellite(self, etl_satellite):
        """No debe haber valores nulos."""
        assert etl_satellite.isnull().sum().sum() == 0


# =============================================================================
# Tests: Servicios
# =============================================================================

class TestETLServices:
    """Tests para el módulo etl_services.py."""

    def test_services_count(self, etl_services):
        """Debe generar datos de servicios."""
        assert len(etl_services) > 0, "Debe generar datos de servicios"

    def test_services_columns(self, etl_services):
        """Debe contener las columnas esperadas."""
        expected_cols = {"cell_id", "schools_count", "hospitals_count", "parks_count"}
        actual_cols = set(etl_services.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_counts_non_negative(self, etl_services):
        """Los conteos no deben ser negativos."""
        count_cols = ["schools_count", "hospitals_count", "parks_count",
                      "supermarkets_count", "banks_count", "restaurants_count"]
        for col in count_cols:
            if col in etl_services.columns:
                assert (etl_services[col] >= 0).all(), f"{col} no debe ser negativo"

    def test_no_null_services(self, etl_services):
        """No debe haber valores nulos."""
        assert etl_services.isnull().sum().sum() == 0


# =============================================================================
# Tests: Socioeconómico
# =============================================================================

class TestETLSocioeconomic:
    """Tests para el módulo etl_socioeconomic.py."""

    def test_socioeconomic_count(self, etl_socioeconomic):
        """Debe generar datos socioeconómicos."""
        assert len(etl_socioeconomic) > 0, "Debe generar datos socioeconómicos"

    def test_socioeconomic_columns(self, etl_socioeconomic):
        """Debe contener las columnas esperadas."""
        expected_cols = {"cell_id", "population_density", "median_income_usd",
                         "employment_rate"}
        actual_cols = set(etl_socioeconomic.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_income_positive(self, etl_socioeconomic):
        """El ingreso mediano debe ser positivo."""
        assert (etl_socioeconomic["median_income_usd"] > 0).all()

    def test_rates_range(self, etl_socioeconomic):
        """Las tasas deben estar en rangos plausibles."""
        # education_level = años de escolaridad (INEC: ~10-16 años)
        if "education_level" in etl_socioeconomic.columns:
            assert etl_socioeconomic["education_level"].between(0, 20).all(), \
                "education_level (años) fuera de rango"
        for col in ["employment_rate", "internet_penetration"]:
            if col in etl_socioeconomic.columns:
                assert etl_socioeconomic[col].between(0, 1).all(), f"{col} fuera de rango"

    def test_no_null_socioeconomic(self, etl_socioeconomic):
        """No debe haber valores nulos."""
        assert etl_socioeconomic.isnull().sum().sum() == 0


# =============================================================================
# Tests: Consistencia entre ETL
# =============================================================================

class TestCrossETLConsistency:
    """Todos los ETL deben usar la misma rejilla de celdas."""

    def test_same_cells_across_datasets(
        self, etl_transactions, etl_mobility, etl_satellite, etl_services, etl_socioeconomic
    ):
        """Las celdas deben coincidir entre los 5 datasets."""
        sets = [
            set(etl_transactions["cell_id"]),
            set(etl_mobility["cell_id"]),
            set(etl_satellite["cell_id"]),
            set(etl_services["cell_id"]),
            set(etl_socioeconomic["cell_id"]),
        ]
        common = set.intersection(*sets)
        assert len(common) == len(sets[0]), (
            f"Las rejillas no coinciden: {len(common)}/{len(sets[0])} celdas comunes"
        )


# =============================================================================
# Tests: Feature Engineering
# =============================================================================

class TestBuildFeatures:
    """Tests para el módulo build_features.py."""

    def test_features_not_empty(self, features_df):
        """La tabla de features no debe estar vacía."""
        assert len(features_df) > 0, "La tabla de features no debe estar vacía"

    def test_features_has_cell_id(self, features_df):
        """Debe tener columna cell_id."""
        assert "cell_id" in features_df.columns, "Debe tener columna cell_id"

    def test_features_has_latlon(self, features_df):
        """Debe tener lat/lon completos (sin nulos)."""
        assert "lat" in features_df.columns and "lon" in features_df.columns
        assert features_df["lat"].isnull().sum() == 0, "Hay celdas sin latitud"
        assert features_df["lon"].isnull().sum() == 0, "Hay celdas sin longitud"

    def test_features_has_target(self, features_df):
        """Debe tener columna objetivo anualizada."""
        assert "annualized_valuation" in features_df.columns, "Falta el target"

    def test_features_no_duplicate_cells(self, features_df):
        """No debe haber celdas duplicadas."""
        assert features_df["cell_id"].is_unique, "Hay celdas duplicadas"

    def test_features_no_inf(self, features_df):
        """No debe haber valores infinitos."""
        numeric = features_df.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.values).all(), "Hay valores infinitos"

    def test_features_row_count_matches_grid(self, features_df, base_config):
        """El número de filas debe coincidir con las celdas configuradas."""
        total_cells = sum(c["num_cells"] for c in base_config["cities"].values())
        assert len(features_df) == total_cells, (
            f"Esperaba {total_cells} filas, obtuvo {len(features_df)}"
        )
