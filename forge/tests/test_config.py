"""Tests for ForgeConfig."""
import os
import pytest


class TestConfigDefaults:
    def test_defaults_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("FORGE_"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import ForgeConfig
        config = ForgeConfig.load()
        assert config.db_backend == "sqlite"
        assert config.workers == 50
        assert config.adapter == "auto"

    def test_default_batch_size(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("FORGE_"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import ForgeConfig
        config = ForgeConfig.load()
        assert config.batch_size == 5

    def test_default_ollama_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("FORGE_"):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import ForgeConfig
        config = ForgeConfig.load()
        assert config.ollama_model == "gemma4:26b"

    def test_to_db_config_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("FORGE_"):
                monkeypatch.delenv(k, raising=False)
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import ForgeConfig
        config = ForgeConfig.load()
        db_config = config.to_db_config()
        assert "db_path" in db_config

    def test_to_db_config_postgres(self):
        from forge.config import ForgeConfig
        config = ForgeConfig(
            db_backend="postgres",
            db_host="myhost",
            db_port=5432,
            db_user="forge",
            db_password="secret",
            db_name="forgedb",
        )
        db_config = config.to_db_config()
        assert "db_host" in db_config
        assert db_config["db_host"] == "myhost"

    def test_as_dict_contains_all_fields(self):
        from forge.config import ForgeConfig
        config = ForgeConfig()
        d = config.as_dict()
        assert "db_backend" in d
        assert "workers" in d
        assert "adapter" in d
        assert "anthropic_api_key" in d


class TestConfigEnvOverrides:
    def test_env_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("FORGE_WORKERS", "123")
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import ForgeConfig
        config = ForgeConfig.load()
        assert config.workers == 123

    def test_anthropic_key_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-123")
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import ForgeConfig
        config = ForgeConfig.load()
        assert config.anthropic_api_key == "sk-test-key-123"

    def test_cli_args_override_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("FORGE_WORKERS", "50")
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import ForgeConfig
        config = ForgeConfig.load(cli_args={"workers": "200"})
        assert config.workers == 200


class TestConfigTOML:
    def test_set_and_read_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        for k in list(os.environ):
            if k.startswith("FORGE_"):
                monkeypatch.delenv(k, raising=False)
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import cli_config_set, ForgeConfig
        cli_config_set("workers", "100")
        config = ForgeConfig.load()
        assert config.workers == 100

    def test_special_chars_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from forge.config import _safe_toml_value
        # Verify escaping works
        val = _safe_toml_value('P@ss"w0rd')
        # Inner quotes should be escaped
        assert val.startswith('"') and val.endswith('"')
        inner = val[1:-1]
        # All double quotes inside should be escaped
        assert '\\"' in inner or '"' not in inner


class TestConfigShow:
    def test_show_masks_secrets(self):
        from forge.config import ForgeConfig
        config = ForgeConfig(anthropic_api_key="sk-ant-api-12345678")
        output = config.show()
        # Should mask the API key
        assert "sk-a****" in output or "****" in output
        assert "sk-ant-api-12345678" not in output

    def test_show_returns_string(self):
        from forge.config import ForgeConfig
        config = ForgeConfig()
        output = config.show()
        assert isinstance(output, str)
        assert "FORGE Configuration" in output


class TestConfigSetAndSave:
    def test_set_validates_key(self):
        from forge.config import ForgeConfig
        config = ForgeConfig()
        with pytest.raises(ValueError, match="Unknown config key"):
            config.set("nonexistent_key", "value")

    def test_set_converts_int(self):
        from forge.config import ForgeConfig
        config = ForgeConfig()
        config.set("workers", "200")
        assert config.workers == 200

    def test_save_creates_toml_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        import importlib
        import forge.config
        importlib.reload(forge.config)
        from forge.config import ForgeConfig
        config = ForgeConfig(workers=99, adapter="claude")
        config.save()
        toml_path = tmp_path / ".forge" / "config.toml"
        assert toml_path.exists()
        content = toml_path.read_text()
        assert "99" in content


class TestMaskSecret:
    def test_mask_short_secret(self):
        from forge.config import _mask_secret
        assert _mask_secret("abc") == "****"

    def test_mask_long_secret(self):
        from forge.config import _mask_secret
        result = _mask_secret("sk-ant-12345678")
        assert result.startswith("sk-a")
        assert result.endswith("****")

    def test_mask_empty(self):
        from forge.config import _mask_secret
        assert _mask_secret("") == "(not set)"
