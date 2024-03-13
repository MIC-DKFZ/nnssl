import os
from valohai.config import is_running_in_valohai


def get_valohai_api_token():
    if is_running_in_valohai():
        return os.environ["VALOHAI_API_TOKEN"]
    else:
        import secret_token
        return secret_token.valohai_api_token
