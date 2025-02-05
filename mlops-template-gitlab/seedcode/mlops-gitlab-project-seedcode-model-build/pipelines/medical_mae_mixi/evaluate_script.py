import json
import os
import subprocess
import shutil
import tarfile

source_dir = '/opt/ml/input/data/code'  
destination_dir = '/opt/ml/code'  

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        source_file = os.path.join(root, file)
        
        relative_path = os.path.relpath(root, source_dir)  
        destination_folder = os.path.join(destination_dir, relative_path) 

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        destination_file = os.path.join(destination_folder, file)

        shutil.copy(source_file, destination_file)

        print(f'Successfully copied: {source_file} -> {destination_file}')

model_output_dir = '/opt/ml/model' 

result = subprocess.run([
    "python", "evaluate.py",
    "--batch_size", "8",
    "--finetune", "Pretrain_densenet121.pth",
    "--model", "densenet121",
    "--data_path", "/opt/ml/input/data/training/",
    "--num_workers", "11",
    "--test_list", "/opt/ml/input/data/training/test1.csv",
    "--nb_classes", "5",
    "--eval_interval", "10",
    "--dataset", "chexpert",
    "--aa", "rand-m6-mstd0.5-inc1",
    "--device", "cpu"
], capture_output=True, text=True, check=True)

with open(os.path.join(model_output_dir, "result.txt"), "w") as f:
   f.write(result.stdout)
   f.write(result.stderr)
print(result.stdout)
print(result.stderr)

auc_avg = 0 
for line in result.stdout.splitlines():
  if "AUC avg: " in line:
    auc_avg = float(line.split("AUC avg: ")[1].split()[0])

print(f"Extracted auc value: {auc_avg}")

report_dict = {
    "regression_metrics": {
        "auc": {
            "value": auc_avg,
             }
    }
}

with open(os.path.join(model_output_dir, "evaluation.json"), "w") as f:
    json.dump(report_dict, f)

print(f"Metrics report created successfully to {model_output_dir}")