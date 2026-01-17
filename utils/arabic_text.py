import re

# إزالة التشكيل
TASHKEEL_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

def remove_tashkeel(text: str) -> str:
    return re.sub(TASHKEEL_PATTERN, "", text)


def remove_tatweel(text: str) -> str:
    return text.replace("ـ", "")


def remove_numbers_and_symbols(text: str) -> str:
    return re.sub(r"[^\u0600-\u06FF\s]", " ", text)


def remove_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_arabic_text(text: str) -> str:
    text = remove_tashkeel(text)
    text = remove_tatweel(text)
    text = remove_numbers_and_symbols(text)
    text = remove_extra_spaces(text)
    return text
AR_STOPWORDS = {
    "في","على","من","الى","إلى","عن","مع","هذا","هذه","هذي","ذلك","تلك",
    "انا","أانا","انت","أنت","انتي","أنتي","هو","هي","هم","هن","نحن",
    "و","او","أو","ثم","لكن","بل","كما","قد","لم","لن","لا","ما",
    "كان","كانت","يكون","تكون","بس","مرة","جدا","مره"
}

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    kept = [t for t in tokens if t not in AR_STOPWORDS]
    return " ".join(kept)
def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text
