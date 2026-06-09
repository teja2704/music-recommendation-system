import os

from project_config import PROJECT_ROOT


def get_required_environment_variable(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Configure it in your shell or deployment platform; "
            f"see {PROJECT_ROOT / '.env.example'} for the required variable names."
        )
    return value


def get_spotify_client_credentials():
    return (
        get_required_environment_variable("SPOTIFY_CLIENT_ID"),
        get_required_environment_variable("SPOTIFY_CLIENT_SECRET"),
    )


def get_spotify_redirect_uri():
    return os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
