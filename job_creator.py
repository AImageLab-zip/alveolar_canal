import os
import pathlib
import utils
from os import path
import pandas as pd
import random
import string
from functools import reduce
from operator import getitem
import yaml
import re
from datetime import datetime

multi_labels = {
    'UNLABELED': 3,
    'BACKGROUND': 2,
    'INSIDE': 1,
    'CONTOUR': 0
}
binary_labels = {
    'BACKGROUND': 0,
    'INSIDE': 1
}

EXCELL_PATH = '/homes/mcipriano/experiments.xlsx'
SBATCH_OUTDIR = '/homes/mcipriano/sbatch_scripts/generated_sbatch.sh'
SRUN_COMMAND = '$python -u /homes/mcipriano/projects/alveolar_canal_3Dtraining/main.py --base_config '

EXCLUDED_COLUMNS = ['todo', 'num_labels', 'best_score', 'title', 'date', 'note', 'test_patients_id']  # those excell values dont go to the yaml


def set_nested_item(dataDict, mapList, val):
    """Set item in nested dictionary"""
    reduce(getitem, mapList[:-1], dataDict)[mapList[-1]] = val
    return dataDict


if __name__ == '__main__':

    df = pd.read_excel(EXCELL_PATH)
    config = utils.load_config_yaml(path.join('configs', 'base', 'base_config.yaml'))  # load base config
    titles = []
    yaml_dirs = []
    for i, r in df[df.todo.eq(True)].iterrows():
        # creating an unique title and saving it for the sbatch file
        title = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(9))
        config['title'] = title
        titles.append(title)

        # choosing label-set if specified
        if 'num_labels' in r.keys():
            config['data-loader']['labels'] = binary_labels if r['num_labels'] == 2 else multi_labels

        # saving the cvs keys-values to the yaml dictionary
        for k in r[r.notna()].drop(EXCLUDED_COLUMNS, errors='ignore').keys():
            # return list of values if this value is a comma separeted list in yaml
            val = r[k]
            if type(val) == str and len(val.split(',')) > 1:
                val = [int(v.strip()) if v.strip().isnumeric() else v.strip() for v in val.split(',')]
            set_nested_item(config, k.split('.'), val)

        df.loc[i, 'todo'] = False
        df.loc[i, 'title'] = title
        current_datetime = datetime.now()
        df.loc[i, 'date'] = current_datetime.strftime('%x %X')
        # folders for this experiment
        project_dir = path.join('/nas/softechict-nas-2/mcipriano/results/maxillo/3D/', title)
        pathlib.Path(os.path.join(project_dir)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(project_dir, 'logs')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(project_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(project_dir, 'files')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(project_dir, 'numpy')).mkdir(parents=True, exist_ok=True)
        # saving the yaml dict as a file for this experiment
        yaml_dir = os.path.join(project_dir, 'logs', 'config.yaml')
        with open(yaml_dir, 'w') as file:
            yaml.dump(config, file)
        yaml_dirs.append(yaml_dir)

    df.to_excel(EXCELL_PATH, index=False)
    file = open(os.path.join(SBATCH_OUTDIR), 'r')
    sbatch = file.read()
    file.close()

    pos = sbatch.rfind('-eq') + 5
    start_id = int(re.findall(r'\d+', sbatch[pos:pos + 4])[0])
    next_id = start_id
    for yaml_dir in yaml_dirs:
        next_id = next_id + 1
        sbatch += '\nif [ "$SLURM_ARRAY_TASK_ID" -eq "{}" ]; then\n\t {} {}\nfi\n\n'.format(next_id, SRUN_COMMAND, yaml_dir)

    for line in sbatch.splitlines():
        if line.startswith('#SBATCH --array='):
            sbatch = sbatch.replace(line, '#SBATCH --array={}-{}'.format(start_id + 1, next_id))
            break

    file = open(os.path.join(SBATCH_OUTDIR), 'w')
    file.write(sbatch)
    file.close()
    print("local folders and config files created.")