import datetime


def now_datetime_text(format: str = "%Y%m%d-%H%M%S") -> str:
    now = datetime.datetime.now()
    text = now.strftime(format)
    return text
