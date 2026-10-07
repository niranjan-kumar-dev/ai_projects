"""Environment-based database selection in config.Settings (no .env, no DB)."""

import pytest
from pydantic import ValidationError

from chat_with_website.config import Settings

LOCAL = "postgresql://dev:devpass@localhost:5432/dev_db"
PROD = "postgresql://prod:prodpass@db.example.com:5432/prod_db"


def make(**kw) -> Settings:
    # _env_file=None: ignore the real .env; pass values explicitly
    return Settings(_env_file=None, **kw)


def test_development_uses_local_url():
    s = make(app_env="development", local_database_url=LOCAL, production_database_url=PROD)
    assert s.database_url == LOCAL
    assert s.database_label() == "development → localhost:5432/dev_db"


def test_production_uses_production_url():
    s = make(app_env="production", local_database_url=LOCAL, production_database_url=PROD)
    assert s.database_url == PROD
    assert s.database_label() == "production → db.example.com:5432/prod_db"


def test_app_env_is_case_insensitive_and_defaults_to_development():
    assert make(app_env=" Production ", production_database_url=PROD).app_env == "production"
    assert make(local_database_url=LOCAL).app_env == "development"


def test_missing_local_url_in_development_is_a_clear_error():
    with pytest.raises(ValidationError, match="LOCAL_DATABASE_URL is not set"):
        make(app_env="development", production_database_url=PROD)


def test_missing_production_url_in_production_is_a_clear_error():
    with pytest.raises(ValidationError, match="PRODUCTION_DATABASE_URL is not set"):
        make(app_env="production", local_database_url=LOCAL)


def test_invalid_app_env_is_rejected():
    with pytest.raises(ValidationError, match="APP_ENV must be one of"):
        make(app_env="test123", local_database_url=LOCAL)


def test_explicit_database_url_overrides_both():
    override = "postgresql://x:y@override:5432/o"
    s = make(app_env="production", local_database_url=LOCAL, production_database_url=PROD, database_url=override)
    assert s.database_url == override


def test_blank_database_url_counts_as_unset():
    s = make(app_env="development", local_database_url=LOCAL, database_url="")
    assert s.database_url == LOCAL


def test_logs_never_contain_the_password():
    s = make(app_env="production", production_database_url=PROD)
    assert "prodpass" not in s.safe_database_url()
    assert "prodpass" not in s.database_label()
    assert "prod" in s.safe_database_url()  # user is fine to show
