import re
import trafilatura


def download_and_extract(url):
    """
    تحميل الصفحة واستخراج النص النظيف منها
    """

    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return ""

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )

        return clean_text(text)

    except Exception:
        return ""


def clean_text(text):

    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)

    return text.strip()
