from pathlib import Path
import sys

from mic_rocket.wp2.database.data_store import DataStorageInterface, Query, Conditions
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *
from loguru import logger

from nnssl.data.raw_dataset import Dataset, Subject, Session, Image, Collection, AssociatedMasks

# Possible Query options: where, limit
# Possible Conditions options: equal, range, above, below, not_in, in_
# Possible tags to query can be found in this document (You can query based on Image and Dataset tags):
# https://docs.google.com/document/d/1TEG76ZM-YhNwThcKNOoD3BR99jnybFOY6oIP-ju5bns/edit?usp=sharing

# FYIs:
# - Getting all images (all their metadata) can take some time if there is simultaneously a dataset ingestion happening.
# - Limiting the returned tags of the results improves the querying speed as less data needs to be send over the network.

REQUIRED_KEYS = [
    "a_direction",
    "a_origin",
    "a_spacing",
    "a_shape",
    "data_storage_filepath",
    "dataset_id",
    "id",
    "patient_id",
    "r_body_part_examined",
    "r_data_type",
    "r_gender",
    "r_time_point",
    "d_anonymization_mask_path",
    "d_foreground_mask_path",
    "d_has_patient_annotation",
    "r_manufacturer",
    "r_manufacturer_model_name",
    "r_modality",
    "r_nationality",
    "state",
]
DROP_KEYS = [
    "c_body_part_examined",
    "c_data_type",
    "c_gender",
    "c_manufacturer",
    "c_manufacturer_model_name",
    "c_modality",
    "c_nationality",
    "c_patient_name",
    "c_time_point",
    "q_is_anonymized",
    "q_is_corrupted",
    "q_noise_level",
    "q_reversed_z_orientation",
    "q_spatial_dimensionality",
    "q_unified_sequence_name",
    "q_valid_z_spacing",
    "r_acquisition_institution",
    "raw_filepaths",
    "raw_metadata",
    "r_patient_name",
]


def get_rocket_token() -> str:
    rocket_token = os.environ["rocket_token"]
    return rocket_token


def query_rocket_database() -> Collection:
    # Init the interface
    interface = DataStorageInterface(
        base_url="http://e230-pc40.dkfz-heidelberg.de",
        auth_token=get_rocket_token(),
    )  # Fabian Workstation: http://e230-pc40.dkfz-heidelberg.de  Central Database: http://10.128.131.202

    # Example Get all images
    # Construct the query
    print("Get all images:")
    q = Query()  # .limit(*REQUIRED_KEYS)

    # Get the paginated query result
    # paginated_result is an iterator that will query always 100 items at once from the database for performance reasons
    paginated_result = interface.query(q)

    # Get all items from the query
    i = 0
    results = []
    for res in tqdm(paginated_result):
        results.append(res)
        # if i > 20:
        #     break
        # i += 1

    # result = [item for item in tqdm(paginated_result, desc="Querying")]

    print("Number of queried images: ", len(results))
    print("-----")
    return results


# Examples:
def karols_old_queries():
    print("Get all CT images:")
    q = Query().where(Conditions.equal("r_modality", "CT"))
    paginated_result = interface.query(q)
    result = [item for item in tqdm(paginated_result, desc="Querying")]
    print("Number of queried images: ", len(result))
    print("-----")

    print("Get all MR images:")
    q = Query().where(Conditions.equal("r_modality", "MR"))
    paginated_result = interface.query(q)
    result = [item for item in tqdm(paginated_result, desc="Querying")]
    print("Number of queried images: ", len(result))
    print("-----")

    print("Get all non CT and MR images:")
    q = Query().where(Conditions.not_in("r_modality", ["CT", "MR"]))
    paginated_result = interface.query(q)
    result = [item for item in tqdm(paginated_result, desc="Querying")]
    print("Number of queried images: ", len(result))
    print("-----")

    print("Get all images from the manufacturer 'Varian Imaging Laboratories, Switzerland':")
    q = Query().where(Conditions.equal("r_manufacturer", "Varian Imaging Laboratories, Switzerland"))
    paginated_result = interface.query(q)
    result = [item for item in tqdm(paginated_result, desc="Querying")]
    print("Number of queried images: ", len(result))
    print("-----")

    print("Get all CT images with a shape larger than (50, 256, 256):")
    q = (
        Query()
        .where(Conditions.equal("r_modality", "CT"))
        .where(Conditions.above("a_shape", 49, index=0))
        .where(Conditions.above("a_shape", 256, index=1))
        .where(Conditions.above("a_shape", 256, index=2))
    )
    paginated_result = interface.query(q)
    result = [item for item in tqdm(paginated_result)]
    print("Number of queried images: ", len(result))
    print("-----")

    print("Get all CT images, but limit the returned tags per image to only include the 'a_spacing' tag:")
    q = Query().where(Conditions.equal("r_modality", "CT")).limit("a_spacing")
    paginated_result = interface.query(q)
    result = [item for item in tqdm(paginated_result, desc="Querying")]
    if len(result) > 0:
        print("First query result: ", result[0])
    else:
        print("No results for this query.")
    print("-----")

    print("Get all CT images, but limit the returned tags per image to only include the 'data_storage_filepath' tag:")
    q = Query().where(Conditions.equal("r_modality", "CT")).limit("data_storage_filepath")
    paginated_result = interface.query(q)
    result = [item for item in tqdm(paginated_result, desc="Querying")]
    if len(result) > 0:
        print("First query result: ", result[0])
    else:
        print("No results for this query.")
    print("-----")


def get_rocketv0_query() -> tuple[str, Query]:
    """Basic query to get all images."""
    q = Query()
    return "rocket_v0", q


def get_rocketv1_query() -> tuple[str, Query]:
    """Query that removes images with shape < 10 and spacing > 8."""
    MIN_SHAPE = 10  # A lot have 16 in z, so we filter below.
    MAX_SPACING = 8  # Often Z around 5/6, so we keep that in.
    q: Query
    q = (
        Query()
        .where(Conditions.above("a_shape", MIN_SHAPE, index=0))
        .where(Conditions.above("a_shape", MIN_SHAPE, index=1))
        .where(Conditions.above("a_shape", MIN_SHAPE, index=2))
        .where(Conditions.below("a_spacing", MAX_SPACING, index=0))
        .where(Conditions.below("a_spacing", MAX_SPACING, index=1))
        .where(Conditions.below("a_spacing", MAX_SPACING, index=2))
        # .where(Conditions.in_("r_modality", ["CT", "MR", "PET"]))
    )

    return "rocket_v1", q


def save_images_to_collection(all_images: list[dict], filename: str):
    collection = Collection(collection_index=800, collection_name="Rocket_v1")
    # Add images to dataset

    modalities_to_exclude = ["SEG", "RTSTRUCT", "RTDOSE", "RTPLAN", "SM", "OCT", "DM", "DF"]

    for image in tqdm(all_images):
        dataset_id = image["dataset_id"]
        subject_id = image["patient_id"]
        session_id = image["r_time_point"]
        image_id = image["id"]
        image_info = {
            "direction": image["a_direction"],
            "origin": image["a_origin"],
            "spacing": image["a_spacing"],
            "shape": image["a_shape"],
            "r_body_part_examined": image["r_body_part_examined"],
            "r_data_type": image["r_data_type"],
        }
        subject_info = {
            "gender": image["r_gender"],
        }
        if image["r_modality"] in modalities_to_exclude:
            continue

        dataset: Dataset
        session: Session
        image: Image

        if dataset_id not in collection.datasets.keys():
            dataset = Dataset(
                dataset_index=dataset_id,
            )
            collection.datasets[dataset_id] = dataset
        else:
            dataset = collection.datasets[image["dataset_id"]]

        if subject_id not in dataset.subjects.keys():
            subject = Subject(subject_id=subject_id, subject_info=subject_info)
            subject.subject_id = subject_id
            dataset.subjects[subject.subject_id] = subject
        else:
            subject = dataset.subjects[subject_id]

        if session_id not in subject.sessions.keys():
            if session_id is None:
                session_id = f"unknown_session__{len(subject.sessions)}"
            session = Session(session_id=session_id)
            subject.sessions[session.session_id] = session
        else:
            session = subject.sessions[session_id]

        # --------------------------- Always add the image --------------------------- #
        image_path = "$RocketDBRoot/" + image["data_storage_filepath"]
        anon_mask_path = (
            "$RocketDBRoot/" + image["d_anonymization_mask_path"] if image["d_anonymization_mask_path"] else None
        )
        anatomy_mask_path = (
            "$RocketDBRoot/" + image["d_foreground_mask_path"] if image["d_foreground_mask_path"] else None
        )
        image = Image(
            name=image_id,
            image_path=image_path,
            modality=image["r_modality"],
            image_info=image_info,
            associated_masks=AssociatedMasks(
                anonymization_mask=anon_mask_path,
                anatomy_mask=anatomy_mask_path,
            ),
        )
        session.images.append(image)
    rocket_dict = collection.to_dict()
    save_json(rocket_dict, f"{filename}.json")
    return collection


def query_modalities_in_db() -> set[str]:
    interface = DataStorageInterface(
        base_url="http://e230-pc40.dkfz-heidelberg.de",
        auth_token=get_rocket_token(),
    )  # Fabian Workstation: http://e230-pc40.dkfz-heidelberg.de  Central Database: http://
    q = Query().limit("r_modality")
    paginated_result = interface.query(q)
    result = [item for item in tqdm(paginated_result, desc="Querying")]
    modalities = set(result)
    return modalities


def query_rocket_db() -> Collection:
    interface = DataStorageInterface(
        base_url="http://e230-pc40.dkfz-heidelberg.de",
        auth_token=get_rocket_token(),
    )  # Fabian Workstation: http://e230-pc40.dkfz-heidelberg.de  Central Database: http://10.128.131.202

    filename, q = get_rocketv1_query()  # Apply filters of interest.
    q.limit(*REQUIRED_KEYS)

    if not os.path.exists("query_" + filename + ".json"):
        logger.info(f"Querying RocketDB for {filename}")
        paginated_result = interface.query(q)
        result = [item for item in tqdm(paginated_result, desc="Querying")]
        save_json(result, "query_" + filename + ".json")

    all_images = load_json("query_" + filename + ".json")
    collection = save_images_to_collection(all_images, filename)
    return collection


if __name__ == "__main__":
    # modalities = query_modalities_in_db()
    query_rocket_db()
