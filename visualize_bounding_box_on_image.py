'''
-*- coding: utf-8 -*-
Copyright (C) 2019/5/5 
Author: Xin Qian

This script visualizes the answer bounding box on DVQA dataset.

    Usage: pythonw visualize_bounding_box_on_image.py

Note: Example question answer pair data is in this below format.

bbox_answer: If the answer is a text in the bar_chart, bounding box in form of [x,y,w,h], else []

        [
            {
                "question": "Which group has the smallest summed value?",
                "question_id": 299202,
                "template_id": "reasoning",
                "answer": "pair",
                "image": "bar_val_easy_00000992.png",
                "answer_bbox": [
                    142.43141274802392,
                    355.969399435541,
                    59.15603800820509,
                    59.15603800820507
                ]
            },
        ]


Note: To get a proof-of-concept example (who has bounding box for its answer), use this bash command:

grep -o -E '.{0,40}bar_val_easy_00000992.png.{0,40}' data/qa/val_easy_qa.json

NOTE: to pretty print a JSON file: use python -m json.tool [filename] and then grep line by line

    python -m json.tool data/qa/val_easy_qa.json > data/qa/val_easy_qa_pretty.json

    grep -A 10 -B 10 "bar_val_easy_00000992.png" data/qa/val_easy_qa_pretty.json


'''

image_name = "bar_val_easy_00000992.png"
image_dir = "data/images/unsplitted_images"  # this is a local directory path
answer_bbox = [
    142.43141274802392,
    355.969399435541,
    59.15603800820509,
    59.15603800820507
]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

im = np.array(Image.open(os.path.join(image_dir, image_name)), dtype=np.uint8)
# plt.imshow(im)
# plt.show()

#
# # Create figure and axes
fig, ax = plt.subplots(1)
#
# # Display the image
ax.imshow(im)
#
# # Create a Rectangle patch
rect = patches.Rectangle((answer_bbox[0], answer_bbox[1]), answer_bbox[2], answer_bbox[3], linewidth=1, edgecolor='r', facecolor='none')
#
# # Add the patch to the Axes
ax.add_patch(rect)
#
plt.show()
