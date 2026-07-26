from pathlib import Path


def test_reference_shaped_monitoring_files_exist():
    root = Path(__file__).resolve().parent
    required = (
        "1.bukti_serving",
        "2.prometheus.yml",
        "3.prometheus_exporter.py",
        "4.bukti monitoring Prometheus",
        "5.bukti monitoring Grafana",
        "6.bukti alerting Grafana",
        "7.inference.py",
        "README.md",
        "docker-compose.yml",
        "grafana/dashboards/bank-marketing.json",
    )
    assert all((root / path).exists() for path in required)
