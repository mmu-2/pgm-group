import base64
import os
from datetime import datetime
from time import gmtime, strftime

import numpy as np
import pytz
from PIL import Image
from yattag import Doc

from scoring.clip_embedder import (acquisition_model, get_image_embeddings,
                                   get_text_embeddings, max_few_shot_score,
                                   prompt_proximity_score)
from scoring.cv_score import convexity_score
import io
import pandas as pd
from cas import run_train
import itertools     

class CASOpt():
    def __init__(self):
        self.dataroot = "./cas/smoking/"
        self.dataroot2 = "./cas/smoking/"
        self.epochs = 30
        self.batch_size = 32
        self.output_dir = "./cas/smoking/"
        self.learning_rate = 0.0001
        self.momentum = 0.9

device = "cuda"

EST = pytz.timezone('EST') 
datetime_utc = datetime.now(EST) 
now = datetime_utc.strftime('%Y-%m-%d_%H:%M:%S')
print(now)

walking_path = "./pytorch-CycleGAN-and-pix2pix/test_results/"
few_shot_path = "pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/pop_processed"
report_path = f"./results/reports_{now}"
os.makedirs(report_path, exist_ok=True)

# few shot images
samples = [0, 10, 20, 30, 40]
ground_truth_images = [os.path.join(few_shot_path, f"{p}.png") for p in samples]

# prompt proximity score
prompt = "a picture of a popsicle stick"

def burst_score(list_of_images):
    image_embeddings = get_image_embeddings(list_of_images, processor, model, device)
    few_shot_embeddings = get_image_embeddings(ground_truth_images, processor, model, device)
    text_prompt = get_text_embeddings([prompt], processor, model, device)
    pps = prompt_proximity_score(image_embeddings, text_prompt)
    mfss = max_few_shot_score(image_embeddings, few_shot_embeddings)
    cvs, imgs_out, imgs_mask = convexity_score(list_of_images, modifiers = {})

    scores = {
        "prompt proximity score": pps,
        "max few-shot score": mfss,
        "convexity score": cvs
    }
    assets = {
        "annotated images": imgs_out,
        "masks": imgs_mask
    }
    return scores, assets

def save_report(report, output_path):
    final_report = f"""
    <html>
    <head><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css" integrity="sha384-xOolHFLEh07PJGoPkLv1IbcEPTNtaed2xpHsD9ESMhqIYd0nLMwNLD69Npy4HI+N" crossorigin="anonymous"></head>
    <body>
    {report}
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-Fy6S3B9q64WdZWQUiU+q4/2Lc9npb8tCaSX9FK7E8HnRr0Jz8D6OP9dO5Vg3Q9ct" crossorigin="anonymous"></script>
    </body>
    </html>
    """
    with open(output_path, "w") as f:
        f.write(final_report)
    
def arr_to_b64(arr):
    im = Image.fromarray(arr.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    return f"data:image/png;base64, {base64.b64encode(rawBytes.read()).decode('ascii')}"

def generate_report(scores, assets, imgs_paths, title):
    doc, tag, text = Doc().tagtext()
    samples_per_row = 3
    rr, cc = (len(imgs_paths) + samples_per_row - 1) // samples_per_row, samples_per_row
    with tag("div", klass="border border-2 border-primary container-fluid"):
        with tag("h3"):
            text(title)
        with tag("ul"):
            with tag("li"):
                text(f"Mean Prompt Proximity Score (pps): {np.mean(scores['prompt proximity score']):0.4f}" )
            with tag("li"):
                text(f"Mean Max Few-Shot Score (pps): {np.mean(scores['max few-shot score']):0.4f}" )
            with tag("li"):
                text(f"Mean Convexity Score (cs): {np.mean(scores['convexity score']):0.4f}" )
        for r in range(rr):
            with tag("div", klass="row justify-content-center"):
                for c in range(cc):
                    i = r * samples_per_row + c
                    if i >= len(imgs_paths):
                        break
                    with tag("div", klass="col-4"):
                        with tag("div", klass="row"):
                            text(os.path.basename(imgs_paths[i]))
                            doc.stag('br')
                            text(f"pps: {scores['prompt proximity score'][i]:0.4f}")
                            text(f", mfss: {scores['max few-shot score'][i]:0.4f}")
                            text(f", cs: {scores['convexity score'][i]:0.4f}")
                        with tag("div", klass="row"):
                            with tag("div", klass="col-6 border border-secondary"):
                                doc.stag('img', src=arr_to_b64(assets["annotated images"][i]), klass="img-fluid", width="200px")
                            with tag("div", klass="col-6 border border-secondary"):
                                doc.stag('img', src=arr_to_b64(assets["masks"][i]*255), klass="img-fluid", width="200px")
    return doc.getvalue()

overall_scores = []
scores_dict = dict()

fake_val_images = f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/val_results/diff_aug'
real_val_images = (f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/valA', f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/valB')
real_test_images = (f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/testA', f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/testB')

cas_args = CASOpt()

option_names = ['brightness', 'saturation', 'contrast', 'translation', 'cutout']

path_names = []  
for r in range(len(option_names) + 1):
    for subset in itertools.combinations(option_names, r):
        options_str = '_'.join(option.lstrip('-') for option in subset)
        path_names.append(f'{options_str}')
path_names += ["diff_aug", "specnorm", "diff_specnorm", "raw"]

path_names.remove('')

cas_dict = dict()
for path_name in path_names:
    print(f"Calculating CAS for {path_name}")
    fake_val_images = f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/val_results/{path_name}'
    real_val_images = (f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/valA', f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/valB')
    real_test_images = (f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/testA', f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/datasets/smoke-pop/testB')
    
    fake_score, real_score = run_train(cas_args, fake_val_images, real_val_images, real_test_images)
    
    fake_val_images = f'/data/taeyoun_kim/reward_gan/pgm-group/pytorch-CycleGAN-and-pix2pix/val_results/raw'
    fake_score_raw, real_score_raw = run_train(cas_args, fake_val_images, real_val_images, real_test_images)
    cas_dict[path_name] = (real_score - fake_score, real_score_raw - fake_score_raw)

    
processor, model = acquisition_model(device)
pps_threshold = 0.31
mfs_threshold = 0.924
for root, dirs, files in os.walk(walking_path):
    if not any([f.endswith("fake_B.png") for f in files]):
        continue
    identifier = root.split("/")[-1]
    print(f"Processing: {identifier}")
    imgs_list = [os.path.join(root, file) for file in files if file.endswith("fake_B.png")]

    scores, assets = burst_score(imgs_list)
    prompt_scores = [round(score, 3) for score in scores["prompt proximity score"]]
    prompt_scores = sorted(prompt_scores, reverse=True)
    pps_count = len([score for score in prompt_scores if score > pps_threshold])
    
    prompt_scores = [round(score, 3) for score in scores["max few-shot score"]]
    prompt_scores = sorted(prompt_scores, reverse=True)
    mfs_count = len([score for score in prompt_scores if score > mfs_threshold])

    scores_dict[identifier] = scores
    overall_scores.append([
        identifier, 
        round(np.mean(scores["prompt proximity score"]), 4),
        round(np.mean(scores["max few-shot score"]), 3),
        round(np.mean(scores["convexity score"]), 2),
        pps_count,
        mfs_count
    ])
    report = generate_report(scores, assets, imgs_list, root)
  
    save_report(report, os.path.join(report_path, identifier + ".html"))

modified_overall_scores = []
for overall_score in overall_scores:
    identifier = overall_score[0]
    modified_overall_scores.append([
        overall_score[0], 
        overall_score[1],
        overall_score[2],
        overall_score[4],
        overall_score[5],
        overall_score[3],
        round(cas_dict[identifier][0], 2),
        round(cas_dict[identifier][1], 2)
    ])

df = pd.DataFrame(modified_overall_scores, columns=["report", "Prompt Proximity Score", "Max Few-shot Score", "PPS Count", "MFS Count", "Convexity Score", "CAS", "CAS Base"])

df.to_csv(os.path.join(report_path, "reports.csv"))

if "report_path" not in globals():
    report_path = "./results/reports_2024-04-21_00:52:36"
pps = "Prompt Proximity Score"
mfss = "Max Few-shot Score"
cs = "Convexity Score"

df = pd.read_csv(os.path.join(report_path, "reports.csv"), index_col=0)
# df = df.sort_values(pps, ascending=False).head(10)
df = df.sort_values(pps, ascending=False)
df.reset_index(inplace=True, drop=True)
print("latex:\n", df.to_latex(index=False))
latex_output = df.to_latex(index=False)
with open('latex_table.txt', 'w') as f:
    f.write("latex:\n")
    f.write(latex_output)