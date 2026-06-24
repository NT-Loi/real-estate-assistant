import os
from pathlib import Path

# --- Directory Paths ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("BDS_DATA_DIR", ROOT_DIR / "data"))
LAW_DIR = DATA_DIR / "laws"


def load_dotenv(path: Path = ROOT_DIR / ".env") -> None:
    """Load simple KEY=VALUE pairs from .env without overriding exported env vars."""
    if not path.exists():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


load_dotenv()

# --- batdongsan.com.vn Configurations ---
BASE_URL = "https://batdongsan.com.vn"
LISTING_URLS = {
    "ban": f"{BASE_URL}/nha-dat-ban",
    "cho-thue": f"{BASE_URL}/nha-dat-cho-thue",
}
PROJECT_URL = f"{BASE_URL}/du-an-bat-dong-san"
NEWS_URL = f"{BASE_URL}/tin-tuc"
WIKI_URL = f"{BASE_URL}/wiki"

WIKI_CATEGORIES = {
    "mua-bds": "Mua BĐS",
    "ban-bds": "Bán BĐS",
    "thue-bds": "Thuê BĐS",
    "tai-chinh": "Tài chính BĐS",
    "quy-hoach-phap-ly": "Quy hoạch - Pháp lý",
    "noi-ngoai-that": "Nội - Ngoại thất",
    "phong-tuc": "Phong tục",
}

# --- API Keys and External Credentials ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# We can store cookies string directly in an env var or a file
TIKTOK_COOKIES_FILE = ROOT_DIR / "cookies.txt"

# --- Crawling and Request Settings ---
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2.0"))
STEALTH_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    window.chrome = { runtime: {} };
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    );
"""

CONTEXT_OPTS = {
    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "viewport": {"width": 1440, "height": 900},
    "locale": "vi-VN",
}
