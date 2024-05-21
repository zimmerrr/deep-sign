from datasets import Dataset
import os 
import numpy as np
import glob
import tqdm

actions = np.array([
    'hello', 'thanks', 'iloveyou', 'idle',
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y',
    'Z',
    ])

DATA_PATH = os.path.join('../Data') 
NUM_EXAMPLES = 120
NUM_SEQUENCE = 30


def gen():
    for gesture_path in tqdm.tqdm(glob.glob('../Data/**')[:1], '[GENERATING DICTIONARY]'):
        if not os.path.isdir(gesture_path):
            continue 
        gesture_label = os.path.basename(os.path.normpath(gesture_path))
        
        samples = {}

        for sample_path in tqdm.tqdm(glob.glob(os.path.join(gesture_path, '**/*.npy')), '[LOADING SAMPLES FOR GESTURE: ' + gesture_label + ']'):
            sample_label = os.path.dirname(os.path.normpath(sample_path))
            sample_label = os.path.basename(sample_label)
            
            if not sample_label in samples:
                samples[sample_label] = []
            samples[sample_label].append(np.load(sample_path))

        for key in samples:
            yield {"label": gesture_label, "sequence": samples[key]}


if __name__ == '__main__':
    ds = Dataset.from_generator(gen)
    ds.save_to_disk("../cache")
