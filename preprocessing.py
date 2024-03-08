from PIL import Image
import glob
import os
from pathlib import Path
import shutil

def preprocess():
    groupA = 'smoking'
    groupB = 'not_smoking'
    shrink(input_filepath = './data/', directory = groupA, output_filepath= './data/'+ groupA +'_processed/')
    shrink(input_filepath = './data/', directory = groupB, output_filepath= './data/'+ groupB +'_processed/')

    #testA, testB, trainA, and trainB
    split_data('./data/smoking_processed/', 'A', split=.1)
    split_data('./data/not_smoking_processed/', 'B', split=.1)

def split_data(input_filepath = './data/', group = 'A', split=.1):
    """Splits data into test and train sets"""
    dataset = glob.glob(input_filepath + '*')
    split_index = max(split, 1-split)
    training = dataset[(int)(len(dataset)*split_index):]
    testing = dataset[:(int)(len(dataset)*split_index)]

    output_filepath = f'./data/train{group}/'
    Path(output_filepath).mkdir(parents=True, exist_ok=True)
    for filepath in training:
        img = Image.open(filepath)
        filename = os.path.basename(filepath)
        img.thumbnail((128,128), Image.LANCZOS)
        if img.mode in ["RGBA", "P"]:
            img = img.convert("RGB")
        img.save(os.path.join(output_filepath, filename))

    output_filepath = f'./data/test{group}/'
    Path(output_filepath).mkdir(parents=True, exist_ok=True)
    for filepath in testing:
        img = Image.open(filepath)
        filename = os.path.basename(filepath)
        img.thumbnail((128,128), Image.LANCZOS)
        if img.mode in ["RGBA", "P"]:
            img = img.convert("RGB")
        img.save(os.path.join(output_filepath, filename))

    return training, testing


def filter(files):
    """Filters out undesirable filetypes"""
    filtered_files = []
    for i in range(len(files)):
        file = files[i]
        filename, file_extension = os.path.splitext(file)
        if file_extension in ['.png', '.jpg', '.jpeg', '.JPEG', '.JPG']:
            filtered_files.append(file)
        else:
            print(f'The following filetype is not currently accepted: {file}')
    return filtered_files
    
def shrink(input_filepath = './data/', directory = 'smoking', output_filepath = './smoking_processed/'):
    """downsamples to size 128x128"""
    files = glob.glob(input_filepath + directory + '/*')
    files = filter(files)
    Path(output_filepath).mkdir(parents=True, exist_ok=True)

    for filepath in files:
        img = Image.open(filepath)
        filename = os.path.basename(filepath)
        img.thumbnail((128,128), Image.LANCZOS)
        if img.mode in ["RGBA", "P"]:
            img = img.convert("RGB")
        img.save(os.path.join(output_filepath, filename))

def clear_directory(directory):
    """Deletes all files in a directory. Useful during debugging."""
    if os.path.exists(directory):
        shutil.rmtree(directory)

def clear_directories(directories = ['./data/smoking_processed/', './data/not_smoking_processed/']):
    """Deletes all files in a list of directories. Useful during debugging."""
    for directory in directories:
        clear_directory(directory)