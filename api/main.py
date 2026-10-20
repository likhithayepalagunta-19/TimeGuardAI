from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from timeguard.features import COLUMNS
from timeguard.runtime import ModelRuntime


class RULRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    engine_id: int = Field(default=1, ge=1)
    cycle: int = Field(default=30, ge=1, le=10000)
    observations: list[dict[str, float]] | None = Field(default=None, min_length=30, max_length=400)
    offsets: dict[str, float] = Field(default_factory=dict, max_length=21)
    packet_loss: bool = False
    horizon: float = Field(default=45, ge=1, le=125)
    explain: bool = True

    @model_validator(mode='after')
    def validate_window(self):
        if self.observations is not None and any(set(row) != set(COLUMNS) for row in self.observations):
            raise ValueError('Observation rows must have exactly the 26 C-MAPSS columns')
        if any(k not in COLUMNS[5:] or not -100 <= v <= 100 for k, v in self.offsets.items()):
            raise ValueError('Offsets require s1..s21 and a finite value within [-100,100]')
        return self


class DiagnosisRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    type: Literal['L', 'M', 'H'] = 'L'
    air_temperature: float = Field(default=300, ge=250, le=400)
    process_temperature: float = Field(default=310, ge=250, le=450)
    rotational_speed: float = Field(default=1500, ge=100, le=5000)
    torque: float = Field(default=40, ge=0, le=150)
    tool_wear: float = Field(default=100, ge=0, le=500)


class Interval(BaseModel):
    lower: float
    upper: float
    nominal_coverage: float
    benchmark_empirical_coverage: float
    clean_benchmark_request: bool


class AttributionEntry(BaseModel):
    sensor: str
    contribution: float


class Attribution(BaseModel):
    method: str
    units: str
    baseline_rul: float
    all_contributions_sum: float
    top: list[AttributionEntry]
    note: str


class RULResponse(BaseModel):
    engine_id: int
    cycle: int
    observed_cycle: int | None
    point_rul: float | None
    risk_aware_rul: float | None
    interval: Interval | None
    alert_state: str
    quality: str
    attribution: Attribution | None
    model_version: str
    provenance: Literal['live_model']
    note: str
    latency_ms: float


class DiagnosisResponse(BaseModel):
    failure_probability: float
    modes: dict[str, float]
    predicted_modes: list[str]
    threshold: float
    probabilities_calibrated: bool
    model_version: str
    provenance: Literal['live_model']
    note: str
    latency_ms: float


def create_app(root: Path = ROOT) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = None
        try:
            app.state.runtime = ModelRuntime(root)
        except Exception:
            logging.exception('Verified model startup failed')
        yield
        app.state.runtime = None

    app = FastAPI(
        title='TimeGuard AI — Uncertainty-Aware Predictive Maintenance',
        description=(
            'Uncertainty-Aware Predictive Maintenance & Diagnostics Workbench. '
            'Remaining useful life forecasts, conformal prediction intervals, local sensor '
            'attributions and multi-label manufacturing diagnostics. Verified model artifacts '
            'run locally in memory; maintenance decisions require human review.'
        ),
        version='2.1.1',
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=False,
                       allow_methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type'])

    def runtime(request):
        value = getattr(request.app.state, 'runtime', None)
        if value is None:
            raise HTTPException(503, 'Models unavailable: verify committed artifacts and runtime versions')
        return value

    @app.get('/health')
    def health(request: Request):
        value = getattr(request.app.state, 'runtime', None)
        if value is None:
            return JSONResponse({'ready': False, 'status': 'unavailable', 'loaded_artifacts': []}, status_code=503)
        return {'ready': True, 'status': 'ready', 'model_version': value.version,
                'loaded_artifacts': ['rul.joblib', 'modes.joblib', 'cmapss_replay.csv', 'TreeSHAP'],
                'verified_files': len(value.manifest['files']),
                'benchmark': {'rul_rmse_raw': value.summary['median']['rmse_raw'],
                              'nasa_score_raw': value.summary['median']['nasa_score_raw'],
                              'nominal_coverage': .9, 'empirical_coverage': value.summary['coverage_90_raw'],
                              'ai4i_binary': value.mode_summary['labels']['binary']}}

    @app.get('/api/engines')
    def engines(request: Request):
        return runtime(request).engine_list()

    @app.post('/api/predict/rul', response_model=RULResponse)
    def rul(body: RULRequest, request: Request):
        start = time.perf_counter()
        try:
            result = runtime(request).predict_rul(**body.model_dump())
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        result['latency_ms'] = round((time.perf_counter() - start) * 1000, 3)
        return result

    @app.post('/api/diagnose/ai4i', response_model=DiagnosisResponse)
    def diagnose(body: DiagnosisRequest, request: Request):
        start = time.perf_counter()
        result = runtime(request).diagnose(body.model_dump())
        result['latency_ms'] = round((time.perf_counter() - start) * 1000, 3)
        return result

    if (root / 'portal').is_dir():
        app.mount('/', StaticFiles(directory=root / 'portal', html=True), name='portal')
    return app


app = create_app()
