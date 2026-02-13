# preprocessing/clean_text.py
import re
import html
import emoji

URL = r"(https?://\S+|www\.\S+)"
USER = r"@\w+"
HASHTAG = r"#\w+"
MULTISPACE = r"\s+"

def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = s.replace("\u200d", "")  # zero-width joiners
    s = re.sub(URL, " ", s)
    s = re.sub(USER, " ", s)
    s = re.sub(HASHTAG, " ", s)
    s = emoji.replace_emoji(s, replace=" ")
    s = re.sub(r"[^A-Za-z0-9'!?., ]+", " ", s)  # english only pass
    s = re.sub(MULTISPACE, " ", s)
    return s.strip().lower()
