from nnssl.apitoken import get_valohai_api_token
import requests


def get_valohai_projects() -> dict:
    headers = get_auth_header()
    # ------------------------------- Get Projects ------------------------------- #
    resp = requests.get("https://app.valohai.com/api/v0/projects/", headers=headers)
    projects = {}
    for r in resp.json()["results"]:
        projects[r["name"]] = r
    return projects


def get_valohai_stores() -> dict:
    # ------------------------------- Get Stores ------------------------------- #
    store_ids = requests.get(url="https://app.valohai.com/api/v0/stores/", headers=get_auth_header())
    stores = {}
    for s in store_ids.json()["results"]:
        stores[s["name"]] = s
    return stores


def get_andrei_adopt_output():
    """Gets the outputs of the execution with the ID 018e1622-540c-4ed6-ed44-72c3194758a7."""
    return get_execution_output("018e1622-540c-4ed6-ed44-72c3194758a7")


def get_auth_header() -> dict[str, str]:
    return {"Authorization": "Token %s" % get_valohai_api_token()}


def get_execution_output(execution_id):
    url = f"https://app.valohai.com/api/v0/executions/{execution_id}/outputs/"
    header = get_auth_header()
    response = requests.get(url=url, headers=header)
    response_json = response.json()
    return response_json


def get_andrei_adopt_output():
    """Gets the outputs of the execution with the ID 018e1622-540c-4ed6-ed44-72c3194758a7."""
    return get_execution_output("018e1622-540c-4ed6-ed44-72c3194758a7")


def check_for_datum_uuid(query: dict) -> str | None:
    """Takes some infos about a file and checks if it's already in the Valohai dataset."""
    header = {
        "Authorization": "Token %s" % get_valohai_api_token(),
    }
    apendix = ""
    for cnt, (k, v) in enumerate(query.items()):
        if cnt == 0:
            apendix = f"?{k}={v}"
        else:
            apendix += f"&{k}={v}"
    query_url = f"https://app.valohai.com/api/v0/data/{apendix}"
    # query_url = f"https://app.valohai.com/api/v0/data/?name={name}&store={store}&project={project_id}"

    response = requests.get(url=query_url, headers=header)
    response_json = response.json()
    if response_json["count"] == 0:
        return None
    else:
        return response_json["results"][0]["id"]


def get_dataset_uid_by_name(name: str) -> str | None:
    """Gets a dataset_uid by name. Returns None if it doesn't exist."""
    headers = get_auth_header()
    existing_datasets = requests.get("https://app.valohai.com/api/v0/datasets/", headers=headers).json()

    for ds in existing_datasets["results"]:
        if ds["name"] == name:
            return ds["id"]
    return


def maybe_create_new_valohai_dataset(dataset_name: str, owner: int = 0) -> str:
    """
    Checks is the dataset_name is already in valohai.
    If not creates it and returns the `datum_id` of it.
    """

    # Check users to make sure we set it to the right one.
    # headers = get_auth_header_token()
    # usrs = requests.get("https://app.valohai.com/api/v0/users/", headers=headers).json()
    # orgs = requests.get("https://app.valohai.com/api/v0/organizations/", headers=headers).json()
    # Manually check which one Floy is.
    ds_uid = get_dataset_uid_by_name(dataset_name)
    if ds_uid is not None:
        return ds_uid
    # --------------------- If not exists we create a new one -------------------- #
    post_url = f"https://app.valohai.com/api/v0/datasets/"
    post_request_body = {
        "name": dataset_name,
        "owner": owner,
    }
    response = requests.post(post_url, post_request_body, headers=get_auth_header()).json()
    ds_uid = response["id"]
    return ds_uid


def maybe_create_new_dataset_version(
    dataset_name: str, version: str, files: list[dict[str, str]], owner: int
) -> dict:
    """
    Creates a new version of a dataset.

    :arg dataset_name: The name of the dataset.
    :arg version: The version of the dataset.
    :arg files: `datum_id` of the files to be added to the dataset.
    """
    ds_id = maybe_create_new_valohai_dataset(dataset_name, owner)

    post_url = f"https://app.valohai.com/api/v0/dataset-versions/"
    post_request_body = {
        "name": version,
        "dataset": ds_id,
        "files": files,
    }
    response = requests.post(post_url, json=post_request_body, headers=get_auth_header())
    return response


def get_all_prev_dataset_version(dataset_version_id: str) -> set[str]:
    url = f"https://app.valohai.com/api/v0/dataset-versions/{dataset_version_id}/"
    response = requests.get(url, headers=get_auth_header())
    if response.status_code != 200:
        raise Exception(f"Error getting dataset versions. Status code: {response.status_code}")
    dataset_content = response.json()
    if dataset_content["previous_version"] is None:
        return set(dataset_version_id)
    else:
        all_prev_ids = get_all_prev_dataset_version(dataset_content["previous_version"])
        all_prev_ids.add(dataset_version_id)
        return all_prev_ids


def get_dataset_versions(dataset_id: str) -> dict:
    url = f"https://app.valohai.com/api/v0/datasets/{dataset_id}#versions"
    response = requests.get(url, headers=get_auth_header())
    if response.status_code != 200:
        raise Exception(f"Error getting dataset versions. Status code: {response.status_code}")
    dataset_content = response.json()
    latest_version = dataset_content["latest_version"]
    all_versions = get_all_prev_dataset_version(latest_version["id"])
    return dataset_content["results"]


def get_name_from_datum_uid(datum_uids: list[str]) -> list[str]:
    all_infos = []
    for datum_uid in datum_uids:
        url = f"https://app.valohai.com/api/v0/data/{datum_uid}"
        response = requests.get(url, headers=get_auth_header())
        all_infos.append(response.json()["name"])
    return all_infos


def get_datum_uids_in_dataset_content(dataset_version_id: str) -> list[str]:
    url = f"https://app.valohai.com/api/v0/dataset-versions/{dataset_version_id}"

    # Make the API request to get the dataset version's files
    response = requests.get(url, headers=get_auth_header())

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        dataset_files = response.json()["files"]
        dataset_uids = [df["datum"] for df in dataset_files]
        return dataset_uids
    raise Exception(f"Error getting dataset content. Status code: {response.status_code}")


def convert_andrei_adtop_to_lookup(andrei_adopt_output: list[dict]) -> dict[str, dict]:
    """Converts the output of get_andrei_adopt_output to a lookup table."""
    lookup = {}
    for pat in andrei_adopt_output:
        lookup_key = pat["name"].split("/")[-1].split(".")[0]
        lookup[lookup_key] = pat
    return lookup
