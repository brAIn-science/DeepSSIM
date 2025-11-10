import os

# This function extracts the UID and name of an image from its file path.
# The UID corresponds to the parent directory, and the name is the filename without the ".png" extension.
# The extracted values are concatenated using "___" as a separator.
# Author: Lemuel Puglisi

def extract_image_identifier(path: str) -> str:
    uid = path.split('/')[-2]
    name = path.split('/')[-1].replace('.png', '')
    return uid + '___' + name

# This function reconstructs the file path of an image from its identifier.
# The identifier follows the format "UID___FILENAME".
# The function returns the full path by joining the base directory, UID, and filename.
# Author: Antonio Scardace

def get_image_path(identifier: str, base_path: str) -> str:
    uid, name = identifier.split('___')
    return os.path.join(base_path, uid, name + '.png')