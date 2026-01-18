"""
Runtime utilities for packaged apps.

Handles differences between development and packaged (PyInstaller) environments.
"""
import os
import sys
import warnings
from pathlib import Path

# Try to import certifi for SSL certificate handling
try:
    import certifi
    _certifi_available = True
except ImportError:
    _certifi_available = False


def is_packaged() -> bool:
    """
    Check if running as a packaged app (PyInstaller).

    Returns:
        True if running from PyInstaller bundle, False in development
    """
    return getattr(sys, 'frozen', False)


def get_resource_path(relative_path: str) -> Path:
    """
    Get the path to a resource file, handling both dev and packaged environments.

    In development: Returns path relative to project root
    In packaged: Returns path inside sys._MEIPASS (PyInstaller temp dir)

    Args:
        relative_path: Relative path from resource root

    Returns:
        Absolute Path to the resource

    Example:
        get_resource_path("certifi/cacert.pem") -> Path to certificate bundle
    """
    if is_packaged():
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(getattr(sys, '_MEIPASS', '.'))
    else:
        # Development environment
        # backend/runtime_utils.py -> project_root
        base_path = Path(__file__).resolve().parents[2]

    return base_path / relative_path


def ensure_certificates() -> None:
    """
    Ensure SSL certificates are available for HTTPS requests.

    This sets environment variables that the `requests` library uses
    to locate certificate bundles.

    In packaged apps, we check for bundled certificates first.
    Falls back to system certifi installation.

    Raises:
        RuntimeError: If certificates cannot be located
    """
    if not _certifi_available:
        warnings.warn(
            "certifi module not available. SSL certificate verification may fail.",
            RuntimeWarning
        )
        return

    # Get default certifi bundle location
    default_cert_path = certifi.where()

    if is_packaged():
        # Check if certifi bundle exists in packaged environment
        # PyInstaller should bundle it with --add-data
        bundled_cert = get_resource_path('certifi') / 'cacert.pem'

        if bundled_cert.exists():
            # Use bundled certificates
            os.environ['REQUESTS_CA_BUNDLE'] = str(bundled_cert)
            os.environ['SSL_CERT_FILE'] = str(bundled_cert)
            return

        # Also check for certifi bundle in _MEIPASS/certifi/
        meipass_cert = Path(getattr(sys, '_MEIPASS', '.')) / 'certifi' / 'cacert.pem'
        if meipass_cert.exists():
            os.environ['REQUESTS_CA_BUNDLE'] = str(meipass_cert)
            os.environ['SSL_CERT_FILE'] = str(meipass_cert)
            return

    # Fallback to system certifi
    cert_path = Path(default_cert_path)
    if cert_path.exists():
        os.environ['REQUESTS_CA_BUNDLE'] = default_cert_path
        os.environ['SSL_CERT_FILE'] = default_cert_path
    else:
        warnings.warn(
            f"SSL certificate bundle not found at expected location: {default_cert_path}\n"
            f"HTTPS requests may fail. Consider installing certifi.",
            RuntimeWarning
        )


def get_bundle_info() -> dict:
    """
    Get information about the runtime environment.

    Returns:
        Dict with keys:
        - packaged: bool - True if running from PyInstaller
        - meipass: str | None - PyInstaller temp dir (only if packaged)
        - executable: str - Path to Python executable
        - certifi_bundle: str | None - Path to SSL certificate bundle
        - certifi_exists: bool - Whether certifi bundle exists
    """
    info = {
        'packaged': is_packaged(),
        'executable': sys.executable,
    }

    if is_packaged():
        info['meipass'] = getattr(sys, '_MEIPASS', None)

    if _certifi_available:
        info['certifi_bundle'] = certifi.where()
        info['certifi_exists'] = Path(certifi.where()).exists()
    else:
        info['certifi_bundle'] = None
        info['certifi_exists'] = False

    return info


def log_environment_info(logger=None) -> None:
    """
    Log detailed environment information for diagnostics.

    Args:
        logger: Optional logger instance. If None, uses print.
    """
    import platform

    info = get_bundle_info()

    lines = [
        "=" * 60,
        "INSIGHT RUNTIME ENVIRONMENT INFO",
        "=" * 60,
        f"Platform: {platform.platform()}",
        f"Python version: {sys.version}",
        f"Python executable: {sys.executable}",
        f"Packaged (PyInstaller): {info['packaged']}",
    ]

    if info['packaged']:
        lines.extend([
            f"_MEIPASS: {info.get('meipass', 'N/A')}",
            f"frozen: {getattr(sys, 'frozen', False)}",
        ])

    if info['certifi_bundle']:
        lines.extend([
            f"certifi bundle: {info['certifi_bundle']}",
            f"certifi exists: {info['certifi_exists']}",
        ])
    else:
        lines.append("certifi: NOT AVAILABLE")

    lines.extend([
        f"Working directory: {os.getcwd()}",
        f"INSIGHT_WORKSPACE_DIR: {os.getenv('INSIGHT_WORKSPACE_DIR', 'NOT SET')}",
        "=" * 60,
    ])

    output = "\n".join(lines)

    if logger:
        logger.info(output)
    else:
        print(output, file=sys.stderr)


def verify_ssl_connection(test_url: str = "https://huggingface.co") -> dict:
    """
    Verify that SSL connections work.

    Args:
        test_url: URL to test against

    Returns:
        Dict with keys:
        - success: bool - True if connection succeeded
        - error: str | None - Error message if failed
        - cert_used: str | None - Path to certificate bundle used
    """
    result = {
        'success': False,
        'error': None,
        'cert_used': os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE'),
    }

    try:
        import urllib.request
        import ssl

        # Create SSL context
        context = ssl.create_default_context()

        # Try to open the URL
        with urllib.request.urlopen(test_url, context=context, timeout=10) as response:
            if response.status == 200:
                result['success'] = True
    except Exception as e:
        result['error'] = str(e)

    return result


# Auto-initialize certificates on import
# This should be called as early as possible in startup
if _certifi_available:
    try:
        ensure_certificates()
    except Exception as e:
        warnings.warn(f"Failed to initialize certificates: {e}", RuntimeWarning)
else:
    warnings.warn("certifi module not available. SSL certificate verification may fail.", RuntimeWarning)
