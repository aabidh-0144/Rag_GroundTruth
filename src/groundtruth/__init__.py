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


_use_system_certificates()
