from datetime import datetime


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_code(code: str):

    code = str(code).strip()

    if "." in code:
        code = code.split(".")[0]

    return code.zfill(6)


def get_market(code):

    if code.startswith("6"):
        return 1

    return 0