import asyncio
import importlib.util
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
path = Path(__file__).with_name("3.prometheus_exporter.py")
spec = importlib.util.spec_from_file_location("submission_prometheus_exporter", path)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_application_lifespan_starts():
    async def start():
        async with exporter.app.router.lifespan_context(exporter.app):
            pass

    asyncio.run(start())


def test_feature_contract_path():
    assert exporter.load_feature_names() == json.loads(
        (
            PACKAGE_ROOT
            / "Membangun_model"
            / "namadataset_preprocessing"
            / "feature_names.json"
        ).read_text(encoding="utf-8")
    )
