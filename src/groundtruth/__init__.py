"""Groundtruth - an evaluation and regression-testing framework for RAG systems."""

__version__ = "0.1.0"


def _use_system_certificates() -> None:
    """Make Python's TLS trust the OS certificate store.

    On a corporate network that performs TLS inspection, the proxy's root CA lives
    in the Windows certificate store but not in certifi's bundle, so model downloads
    from huggingface.co fail with CERTIFICATE_VERIFY_FAILED even though curl and the
    browser work fine. `truststore` bridges the two.

    Best-effort: on a machine without TLS interception this changes nothing, and if
    truststore is unavailable we fall through to certifi rather than failing import.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


def _load_dotenv() -> None:
    """Load a local `.env` (see `.env.example`) into the environment, if present.

    This is how `GROQ_API_KEY` and the optional `GROUNDTRUTH_GENERATION_PROVIDER`
    switch (see config.py) reach the process without the user having to `set`/
    `export` them by hand every session. Best-effort: a missing `.env`, or a missing
    `python-dotenv`, changes nothing rather than failing import.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


_use_system_certificates()
_load_dotenv()
