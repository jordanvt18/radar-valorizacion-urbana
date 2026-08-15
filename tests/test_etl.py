"""
Tests para los módulos ETL del Radar de Valorización Urbana.

Cobertura:
- Generación de transacciones inmobiliarias
- Generación de datos de movilidad
- Generación de datos satelitales
- Generación de datos de servicios
- Generación de datos socioeconómicos
- Feature engineering
"""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def config():
    """Carga la configuración del proyecto desde config.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def etl_transactions(config):
    """Ejecuta el ETL de transacciones y retorna el DataFrame resultante."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.etl.etl_transactions import generate_transactions
    df = generate_transactions(config)
    return df


@pytest.fixture(scope="module")
def etl_mobility(config):
    """Ejecuta el ETL de movilidad y retorna el DataFrame resultante."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.etl.etl_mobility import generate_mobility
    df = generate_mobility(config)
    return df


@pytest.fixture(scope="module")
def etl_satellite(config):
    """Ejecuta el ETL satelital y retorna el DataFrame resultante."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.etl.etl_satellite import generate_satellite_features
    df = generate_satellite_features(config)
    return df


@pytest.fixture(scope="module")
def etl_services(config):
    """Ejecuta el ETL de servicios y retorna el DataFrame resultante."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.etl.etl_services import generate_services
    df = generate_services(config)
    return df


@pytest.fixture(scope="module")
def etl_socioeconomic(config):
    """Ejecuta el ETL socioeconómico y retorna el DataFrame resultante."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.etl.etl_socioeconomic import generate_socioeconomic
    df = generate_socioeconomic(config)
    return df


@pytest.fixture(scope="module")
def built_features(config, etl_transactions, etl_mobility, etl_satellite,
                    etl_services, etl_socioeconomic):
    """Ejecuta el feature engineering y retorna el DataFrame resultante."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.features.build_features import build_features
    df = build_features(
        transactions_df=etl_transactions,
        mobility_df=etl_mobility,
        satellite_df=etl_satellite,
        services_df=etl_services,
        socioeconomic_df=etl_socioeconomic,
        config=config,
    )
    return df


# =============================================================================
# Tests: Transacciones Inmobiliarias
# =============================================================================

class TestETLTransactions:
    """Tests para el módulo etl_transactions.py."""

    def test_transaction_count(self, etl_transactions, config):
        """El número de transacciones debe ser positivo y razonable."""
        assert len(etl_transactions) > 0, "Debe generar al menos una transacción"
        assert len(etl_transactions) == config["transactions"]["total_count"], (
            f"Esperaba {config['transactions']['total_count']} transacciones, "
            f"obtuvo {len(etl_transactions)}"
        )

    def test_transaction_columns(self, etl_transactions):
        """Debe contener todas las columnas esperadas."""
        expected_cols = {
            "transaction_id", "date", "lat", "lon", "h3_index",
            "property_type", "price_usd", "area_m2", "price_per_m2",
            "bedrooms", "bathrooms", "city",
        }
        actual_cols = set(etl_transactions.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_no_null_transactions(self, etl_transactions):
        """No debe haber valores nulos en columnas críticas."""
        critical_cols = ["transaction_id", "date", "lat", "lon", "h3_index",
                         "property_type", "price_usd", "area_m2", "city"]
        nulls = etl_transactions[critical_cols].isnull().sum()
        assert nulls.sum() == 0, f"Valores nulos encontrados:\n{nulls[nulls > 0]}"

    def test_valid_prices(self, etl_transactions, config):
        """Los precios deben estar en rangos realistas."""
        price_min = config["transactions"]["property_types"]["apartment"]["price_min"]
        price_max = config["transactions"]["property_types"]["house"]["price_max"]
        assert etl_transactions["price_usd"].min() >= 0, "Los precios no deben ser negativos"
        assert etl_transactions["price_usd"].min() >= price_min * 0.5, (
            f"Precio mínimo ({etl_transactions['price_usd'].min()}) demasiado bajo"
        )
        assert etl_transactions["price_usd"].max() <= price_max * 3, (
            f"Precio máximo ({etl_transactions['price_usd'].max()}) demasiado alto"
        )

    def test_price_per_m2_calculated(self, etl_transactions):
        """price_per_m2 debe ser consistente con price_usd / area_m2."""
        calculated = etl_transactions["price_usd"] / etl_transactions["area_m2"]
        np.testing.assert_allclose(
            etl_transactions["price_per_m2"].values,
            calculated.values,
            rtol=0.01,
            err_msg="price_per_m2 no coincide con price_usd / area_m2"
        )

    def test_property_types_valid(self, etl_transactions, config):
        """Los tipos de propiedad deben ser válidos."""
        valid_types = set(config["transactions"]["property_types"].keys())
        actual_types = set(etl_transactions["property_type"].unique())
        assert actual_types.issubset(valid_types), (
            f"Tipos de propiedad inválidos: {actual_types - valid_types}"
        )

    def test_cities_valid(self, etl_transactions, config):
        """Las ciudades deben ser Quito o Guayaquil."""
        valid_cities = {c for c in config["cities"].keys()}
        actual_cities = set(etl_transactions["city"].unique())
        assert actual_cities.issubset(valid_cities), (
            f"Ciudades inválidas: {actual_cities - valid_cities}"
        )

    def test_coordinates_within_bbox(self, etl_transactions, config):
        """Las coordenadas deben estar dentro de los bounding boxes configurados."""
        for city_name, city_config in config["cities"].items():
            city_mask = etl_transactions["city"] == city_name
            if city_mask.sum() == 0:
                continue
            city_data = etl_transactions[city_mask]
            bbox = city_config["bbox"]
            assert city_data["lat"].between(bbox["min_lat"], bbox["max_lat"]).all(), (
                f"Latitudes fuera de bbox para {city_name}"
            )
            assert city_data["lon"].between(bbox["min_lon"], bbox["max_lon"]).all(), (
                f"Longitudes fuera de bbox para {city_name}"
            )

    def test_h3_index_format(self, etl_transactions):
        """Los índices H3 deben ser strings no vacíos."""
        assert etl_transactions["h3_index"].dtype == object, "h3_index debe ser string"
        assert etl_transactions["h3_index"].str.len().min() > 0, "h3_index no debe estar vacío"

    def test_date_range(self, etl_transactions, config):
        """Las fechas deben estar dentro del rango configurado."""
        start = pd.to_datetime(config["transactions"]["start_date"])
        end = pd.to_datetime(config["transactions"]["end_date"])
        dates = pd.to_datetime(etl_transactions["date"])
        assert dates.min() >= start, f"Fecha mínima ({dates.min()}) anterior al inicio ({start})"
        assert dates.max() <= end, f"Fecha máxima ({dates.max()}) posterior al fin ({end})"


# =============================================================================
# Tests: Movilidad
# =============================================================================

class TestETLMobility:
    """Tests para el módulo etl_mobility.py."""

    def test_mobility_count(self, etl_mobility):
        """Debe generar datos para múltiples celdas H3."""
        assert len(etl_mobility) > 0, "Debe generar datos de movilidad"

    def test_mobility_columns(self, etl_mobility):
        """Debe contener las columnas esperadas."""
        expected_cols = {
            "h3_index", "avg_travel_time_cbd_min", "transit_stops_count",
            "road_density_km", "peak_hour_speed_kmh", "walkability_score",
        }
        actual_cols = set(etl_mobility.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_travel_time_positive(self, etl_mobility):
        """El tiempo de viaje debe ser positivo."""
        assert (etl_mobility["avg_travel_time_cbd_min"] > 0).all(), (
            "El tiempo de viaje debe ser positivo"
        )

    def test_walkability_range(self, etl_mobility):
        """El índice de caminabilidad debe estar entre 0 y 100."""
        assert etl_mobility["walkability_score"].between(0, 100).all(), (
            "walkability_score debe estar entre 0 y 100"
        )

    def test_transit_stops_non_negative(self, etl_mobility):
        """El número de paradas de transporte no debe ser negativo."""
        assert (etl_mobility["transit_stops_count"] >= 0).all(), (
            "transit_stops_count no debe ser negativo"
        )

    def test_no_null_mobility(self, etl_mobility):
        """No debe haber valores nulos."""
        nulls = etl_mobility.isnull().sum()
        assert nulls.sum() == 0, f"Valores nulos encontrados:\n{nulls[nulls > 0]}"


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
        expected_cols = {
            "h3_index", "ndvi_mean", "ndvi_std", "built_up_index",
            "green_space_ratio", "night_lights_intensity",
        }
        actual_cols = set(etl_satellite.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_ndvi_range(self, etl_satellite):
        """NDVI debe estar entre -1 y 1."""
        assert etl_satellite["ndvi_mean"].between(-1, 1).all(), (
            "ndvi_mean debe estar entre -1 y 1"
        )

    def test_green_space_ratio_range(self, etl_satellite):
        """El ratio de espacio verde debe estar entre 0 y 1."""
        assert etl_satellite["green_space_ratio"].between(0, 1).all(), (
            "green_space_ratio debe estar entre 0 y 1"
        )

    def test_built_up_index_range(self, etl_satellite):
        """El índice de área construida debe estar entre 0 y 1."""
        assert etl_satellite["built_up_index"].between(0, 1).all(), (
            "built_up_index debe estar entre 0 y 1"
        )

    def test_no_null_satellite(self, etl_satellite):
        """No debe haber valores nulos."""
        nulls = etl_satellite.isnull().sum()
        assert nulls.sum() == 0, f"Valores nulos encontrados:\n{nulls[nulls > 0]}"


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
        expected_cols = {
            "h3_index", "schools_count", "hospitals_count", "parks_count",
            "shopping_count", "bank_count", "restaurants_count",
            "nearest_school_km", "nearest_hospital_km", "nearest_park_km",
        }
        actual_cols = set(etl_services.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_counts_non_negative(self, etl_services):
        """Los conteos de servicios no deben ser negativos."""
        count_cols = ["schools_count", "hospitals_count", "parks_count",
                      "shopping_count", "bank_count", "restaurants_count"]
        for col in count_cols:
            assert (etl_services[col] >= 0).all(), f"{col} no debe ser negativo"

    def test_distances_non_negative(self, etl_services):
        """Las distancias no deben ser negativas."""
        dist_cols = ["nearest_school_km", "nearest_hospital_km", "nearest_park_km"]
        for col in dist_cols:
            assert (etl_services[col] >= 0).all(), f"{col} no debe ser negativo"

    def test_no_null_services(self, etl_services):
        """No debe haber valores nulos."""
        nulls = etl_services.isnull().sum()
        assert nulls.sum() == 0, f"Valores nulos encontrados:\n{nulls[nulls > 0]}"


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
        expected_cols = {
            "h3_index", "population_density", "median_income_usd",
            "education_index", "employment_rate", "internet_penetration",
            "crime_index",
        }
        actual_cols = set(etl_socioeconomic.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Columnas faltantes: {missing}"

    def test_income_positive(self, etl_socioeconomic):
        """El ingreso mediano debe ser positivo."""
        assert (etl_socioeconomic["median_income_usd"] > 0).all(), (
            "median_income_usd debe ser positivo"
        )

    def test_rate_columns_range(self, etl_socioeconomic):
        """Las tasas e índices deben estar en [0, 1] o [0, 100] según corresponda."""
        assert etl_socioeconomic["education_index"].between(0, 1).all(), (
            "education_index debe estar entre 0 y 1"
        )
        assert etl_socioeconomic["employment_rate"].between(0, 1).all(), (
            "employment_rate debe estar entre 0 y 1"
        )
        assert etl_socioeconomic["internet_penetration"].between(0, 1).all(), (
            "internet_penetration debe estar entre 0 y 1"
        )
        assert etl_socioeconomic["crime_index"].between(0, 100).all(), (
            "crime_index debe estar entre 0 y 100"
        )

    def test_no_null_socioeconomic(self, etl_socioeconomic):
        """No debe haber valores nulos."""
        nulls = etl_socioeconomic.isnull().sum()
        assert nulls.sum() == 0, f"Valores nulos encontrados:\n{nulls[nulls > 0]}"


# =============================================================================
# Tests: Feature Engineering
# =============================================================================

class TestBuildFeatures:
    """Tests para el módulo build_features.py."""

    def test_features_not_empty(self, built_features):
        """La tabla de features no debe estar vacía."""
        assert len(built_features) > 0, "La tabla de features no debe estar vacía"

    def test_features_has_h3_index(self, built_features):
        """Debe tener columna h3_index."""
        assert "h3_index" in built_features.columns, "Debe tener columna h3_index"

    def test_features_has_target(self, built_features):
        """Debe tener una columna objetivo (target)."""
        target_candidates = [c for c in built_features.columns
                             if "valor" in c.lower() or "target" in c.lower()
                             or "annual" in c.lower() or "appreciation" in c.lower()
                             or "growth" in c.lower()]
        assert len(target_candidates) > 0, (
            f"No se encontró columna objetivo. Columnas: {list(built_features.columns)}"
        )

    def test_features_no_duplicate_h3(self, built_features):
        """No debe haber celdas H3 duplicadas."""
        dupes = built_features["h3_index"].duplicated().sum()
        assert dupes == 0, f"Encontradas {dupes} celdas H3 duplicadas"

    def test_features_numeric(self, built_features):
        """La mayoría de columnas deben ser numéricas."""
        numeric_cols = built_features.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 5, (
            f"Esperaba al menos 5 columnas numéricas, encontró {len(numeric_cols)}"
        )

    def test_features_no_inf(self, built_features):
        """No debe haber valores infinitos en columnas numéricas."""
        numeric_cols = built_features.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(built_features[numeric_cols].values).sum()
        assert inf_count == 0, f"Encontrados {inf_count} valores infinitos"

    def test_features_row_count_matches_cells(self, built_features, config):
        """El número de filas debe ser consistente con el número de celdas configuradas."""
        total_cells = sum(c["num_cells"] for c in config["cities"].values())
        # Permitir cierta variación por celdas sin datos
        assert len(built_features) <= total_cells * 2, (
            f"Demasiadas filas: {len(built_features)} vs {total_cells} celdas esperadas"
        )
