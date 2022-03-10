import os
import datetime
import time
import csv
import torch.optim as optim

import style
import vgg19



# Values to edit for testing
CONTENT_WEIGHT = 400.0
STYLE_WEIGHT = 1700.0
ITERATIONS = 250
# alternative to iterations
# STOP_AT_LOSS = 2_900_000


# Describe main points about current test
SETUP_DESCRIPTION = f"""

    Model:  VGG19 - avg pooling (Pre-trained on Imagenet dataset),
    Optimizer: LBFGS,
    Iterations: {ITERATIONS},
    Style Weight: {STYLE_WEIGHT},
    Content Weight: {CONTENT_WEIGHT}

    Notes: testing new set of styles (van gogh)
    

"""

# Init stylizer
stylizer = style.Stylizer(style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT)


# Dirs
CONTENT_DIR = os.getcwd() + '\\images'
STYLE_DIR = os.getcwd() + '\\styles'
RESULT_DIR = os.getcwd() + '\\results'


# Testing script start

# Load images
results_dir_name = RESULT_DIR + "\\" + \
    str(datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))

content_images = [CONTENT_DIR + "\\" + f for f in os.listdir(CONTENT_DIR)]
style_images = [STYLE_DIR + "\\" + f for f in os.listdir(STYLE_DIR)]

# Get all pairs of content, style
pairs = [(c, s) for c in content_images for s in style_images]

statistics = []

# Make new result dir
os.makedirs(results_dir_name)
print(f"Made results directory \"{results_dir_name}\"")

# Write and save description text
desc_path = results_dir_name + "\\description.txt"

desc_file = open(desc_path, 'w')
desc_file.write(SETUP_DESCRIPTION)
desc_file.close()

print(f'Saved description in \"{desc_path}\"')

# Apply stylizer and record stats
count = 1
for pair in pairs:
    content = pair[0].split("\\")[-1].replace(".jpg", "")
    style = pair[1].split("\\")[-1].replace(".jpg", "")
    out_path = results_dir_name + "\\" + f'{content}_as_{style}.jpg'

    # Normal iteration run 
    time_taken = stylizer.run_standard(pair[0], pair[1], out_path, ITERATIONS)

    # Stop at loss run
    # time_taken = stylizer.run_to_loss(pair[0], pair[1], out_path, STOP_AT_LOSS)

    item_stats = {
        "time taken": time_taken,
        "content image": content,
        "style image": style
    }

    statistics.append(item_stats)

    print(f' ===== {count}/{len(pairs)}  ===== ')
    count += 1

# Write and save statistics
keys = statistics[0].keys()
stats_path = results_dir_name + '\\statistics.csv'

with open(stats_path, 'w') as out:
    writer = csv.DictWriter(out, keys)
    writer.writeheader()
    writer.writerows(statistics)

print(f'Saved statistics in \"{stats_path}\"')


print("Testing done")
