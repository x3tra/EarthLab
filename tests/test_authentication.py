import pytest
from pydantic import SecretStr

from nasa_agri_intelligence.collection.authentication import EarthdataAuthenticator
from nasa_agri_intelligence.config import EarthdataSettings


def test_existing_appeears_token_is_used_as_bearer_auth() -> None:
    settings = EarthdataSettings(
        _env_file=None,
        nasa_app_ears_token=SecretStr("test-token"),
    )

    authenticator = EarthdataAuthenticator(settings)

    assert authenticator.session.headers["Authorization"] == "Bearer test-token"


def test_authentication_requires_token_or_earthdata_credentials() -> None:
    settings = EarthdataSettings(_env_file=None)

    with pytest.raises(ValueError, match="NASA_APP_EARS_TOKEN"):
        EarthdataAuthenticator(settings)


def test_empty_token_is_treated_as_missing() -> None:
    settings = EarthdataSettings(
        _env_file=None,
        nasa_app_ears_token=SecretStr(""),
    )

    with pytest.raises(ValueError, match="NASA_APP_EARS_TOKEN"):
        EarthdataAuthenticator(settings)
