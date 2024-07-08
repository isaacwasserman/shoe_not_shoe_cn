import hashlib
from safetensors.torch import save_file
from safetensors.torch import load_file
import torch
import os

HASH_START = 0x100000
HASH_LENGTH = 0x10000
SAFETENSORS_STR = "safetensors"
CKPT_STR = "ckpt"

PYTORCH_FILE_EXTENSIONS = {f".{CKPT_STR}"}
SAFETENSORS_FILE_EXTENSIONS = {f".{SAFETENSORS_STR}"}

FILE_RAD_KEY = '-file_radio-'
DIRECTORY_RAD_KEY = '-directory_radio-'
TYPE_SELECTOR_INP_RAD_KEY = '-type_selector_input_radio-'
FILE_LBL = 'File'
DIRECTORY_LBL = 'Directory'

CONVERT_FILE_BTN_KEY = '-convert_file_button-'
CONVERT_DIRECTORY_BTN_KEY = '-convert_directory_button-'
ADD_SUFFIX_CHKBOX_KEY = '-add_suffix_chkbox-'
CONVERT_FILE_BTN_LBL = 'CONVERT FILE'
CONVERT_DIRECTORY_BTN_LBL = 'CONVERT DIRECTORY'
ADD_SUFFIX_CHKBOX_LBL = 'Add Suffix'
FONT = 'Arial 12'
CONSOLE_FONT = 'Arial 11'

FORMAT_SELECTOR_COMBO_KEY = '-format_selector_combo-'
CONSOLE_ML_KEY = '-console_ml-'
PBAR_KEY = 'progress_bar'

CONVERTING_TXT = 'Converting...'

def get_file_hash(filename):
    with open(filename, "rb") as file:
        m = hashlib.sha256()
        file.seek(HASH_START)
        m.update(file.read(HASH_LENGTH))
        return m.hexdigest()[0:8]

def save_checkpoint(weights, filename):
    with open(filename, "wb") as f:
        torch.save(weights, f)

def convert_to_ckpt(filename, suffix=False):
    model_hash = get_file_hash(filename)
    device = "cpu"
    weights = load_file(filename, device=device)

    try:
        weights = load_file(filename, device=device)
    except Exception as e:
        print(f'Error: {e}')

    try:
        print(f'{CONVERTING_TXT} {filename} [{model_hash}] to {CKPT_STR}.')
        weights = load_file(filename, device=device)
        weights["state_dict"] = weights
        checkpoint_filename = f"{os.path.splitext(filename)[0]}-cnvrtd.{CKPT_STR}" if suffix else f"{os.path.splitext(filename)[0]}.{CKPT_STR}"
        save_checkpoint(weights, checkpoint_filename)
        print(f'Saving {checkpoint_filename} [{get_file_hash(checkpoint_filename)}].')
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            print(" File Not Found")
        else:
            print(f'Error: {e}')