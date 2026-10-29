"""Muse AI automation settings - store cookies persistently for automatic operation."""
import json
import os
import urllib.request
import urllib.error

from app.db import SessionLocal
from app.models import AppSetting

MUSE_COOKIES_KEY = "muse_cookies"
MUSE_COOKIE_NAMES = ("hatch_sess", "hatch_gw", "hatch_vml", "hatch_native_auth_device")


def get_muse_cookies() -> dict[str, str]:
    """Get stored Muse cookies from AppSetting."""
    with SessionLocal() as db:
        row = db.get(AppSetting, MUSE_COOKIES_KEY)
        if not row or not row.value:
            return {}
        try:
            return json.loads(row.value)
        except Exception:
            return {}


def set_muse_cookies(cookies: dict[str, str], admin_id: int | None = None) -> None:
    """Store Muse cookies in AppSetting."""
    with SessionLocal() as db:
        row = db.get(AppSetting, MUSE_COOKIES_KEY)
        value = json.dumps(cookies, ensure_ascii=False)
        if row:
            row.value = value
        else:
            row = AppSetting(key=MUSE_COOKIES_KEY, value=value)
            db.add(row)
        db.commit()


def clear_muse_cookies() -> None:
    """Remove stored Muse cookies."""
    with SessionLocal() as db:
        row = db.get(AppSetting, MUSE_COOKIES_KEY)
        if row:
            db.delete(row)
            db.commit()


def muse_cookies_configured() -> bool:
    """Check if all required Muse cookies are stored."""
    cookies = get_muse_cookies()
    return all(cookies.get(k) for k in MUSE_COOKIE_NAMES)


def _get_muse2api_base() -> str:
    return f"http://127.0.0.1:{os.environ.get('MUSE2API_PORT', '18610')}"


def auto_import_cookies_to_muse2api() -> bool:
    """Tự động gửi cookies từ DB vào muse2api gateway.

    Trả về True nếu thành công. Nếu muse2api không chạy, trả False.
    """
    cookies = get_muse_cookies()
    if not cookies:
        return False

    admin_key = None
    state_dir = os.path.expanduser("~/.musekit")
    admin_key_path = os.path.join(state_dir, "admin_key")
    if os.path.exists(admin_key_path):
        try:
            admin_key = open(admin_key_path).read().strip()
        except Exception:
            pass

    if not admin_key:
        return False

    base = _get_muse2api_base()
    payload = json.dumps({"label": "muse-main", "cookies": cookies}).encode()

    try:
        req = urllib.request.Request(
            f"{base}/admin/accounts",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {admin_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 201)
    except Exception:
        return False