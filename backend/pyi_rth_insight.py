"""
PyInstaller runtime hook for Insight Engine.

This hook is executed before the main script and sets up SSL certificate
paths for HTTPS requests (e.g., to HuggingFace for model downloads).

The hook checks for bundled certifi certificates and sets environment
variables that the requests library uses.

Place this file in the same directory as your .spec file.
PyInstaller will automatically include it as a runtime hook.
"""
import os
import sys
from pathlib import Path


def setup_certificates():
    """
    Setup SSL certificates for HTTPS requests.

    1. Check if running in PyInstaller bundle (sys._MEIPASS exists)
    2. Look for bundled certifi certificate in _MEIPASS/certifi/cacert.pem
    3. Set REQUESTS_CA_BUNDLE and SSL_CERT_FILE environment variables
    """
    meipass = getattr(sys, '_MEIPASS', None)

    if meipass:
        # Running from PyInstaller bundle
        cert_bundle = Path(meipass) / 'certifi' / 'cacert.pem'

        if cert_bundle.exists():
            # Use bundled certificates
            os.environ['REQUESTS_CA_BUNDLE'] = str(cert_bundle)
            os.environ['SSL_CERT_FILE'] = str(cert_bundle)
            print(f"[SSL] Using bundled certificates: {cert_bundle}", file=sys.stderr)
        else:
            # Fallback: try to find certifi in the bundle
            certifi_path = Path(meipass) / 'certifi'
            if certifi_path.exists():
                # Look for any .pem file
                for pem_file in certifi_path.rglob('*.pem'):
                    os.environ['REQUESTS_CA_BUNDLE'] = str(pem_file)
                    os.environ['SSL_CERT_FILE'] = str(pem_file)
                    print(f"[SSL] Using bundled certificates: {pem_file}", file=sys.stderr)
                    break

            # If still not found, try importing certifi
            try:
                import certifi
                bundle = certifi.where()
                if Path(bundle).exists():
                    os.environ['REQUESTS_CA_BUNDLE'] = bundle
                    os.environ['SSL_CERT_FILE'] = bundle
                    print(f"[SSL] Using certifi bundle: {bundle}", file=sys.stderr)
            except ImportError:
                print("[SSL] WARNING: No SSL certificates found. HTTPS requests may fail.", file=sys.stderr)


def setup_environment():
    """
    Setup other environment variables needed for packaged app.
    """
    # Set library path for macOS/Linux if needed
    if sys.platform.startswith('darwin'):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            # Add _MEIPASS to DYLD_LIBRARY_PATH for dynamic libraries
            dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
            if dyld_path:
                os.environ['DYLD_LIBRARY_PATH'] = f"{meipath}:{dyld_path}"
            else:
                os.environ['DYLD_LIBRARY_PATH'] = meipass


# Run setup on import
if __name__ == 'pyi_rth_insight':
    setup_certificates()
    setup_environment()
else:
    # Also run when imported directly
    setup_certificates()
    setup_environment()
