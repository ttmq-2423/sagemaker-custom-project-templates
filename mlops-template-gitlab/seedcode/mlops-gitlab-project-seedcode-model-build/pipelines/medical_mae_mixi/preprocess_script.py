import subprocess
import tarfile
import os

os.chdir('/opt/ml/processing/input')
subprocess.run("pip install -r requirements_processing.txt", shell=True, check=True)


subprocess.run([
    "python", "processing.py",
    "--input_size", "224",
    "--random_resize_range", "0.5", "1.0",
    "--datasets_names", "chexpert"
], check=True)

input_file = 'dataset_train.pkl'
output_file = '/opt/ml/processing/output/data_train.tar.gz'

with tarfile.open(output_file, 'w:gz') as tar:
    tar.add(input_file)

print("Processing completed and output uploaded to S3.")
