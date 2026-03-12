from attendance_system.config import load_config


def test_load_default_config_file():
    cfg = load_config('config.yaml')
    assert cfg.detector.model_name
    assert cfg.embedder.model_name
    assert 0.0 <= cfg.matcher.cosine_threshold <= 1.0
