from pathlib import Path


def test_service_down_alert_uses_prometheus_scrape_health():
    rules = Path(__file__).with_name("alert_rules.yml").read_text(encoding="utf-8")
    assert 'expr: up{job="bank-marketing-exporter"} == 0' in rules
