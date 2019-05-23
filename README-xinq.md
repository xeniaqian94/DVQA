#### To train

1. Download and extract DVQA dataset from https://github.com/kushalkafle/DVQA_dataset
    Such that data/ directory contains:
        1) images/ (who has unsplitted_images/) and 
        2) qa/ (train_qa.json, etc.)
        
 Each image has dimension of 448 × 448. Seems to have 4 channels (RGBA) - to be verified. 
     
 Each json has below format 
     
      [
          {
            "question": "How many algorithms have accuracy lower than 9 in at least one dataset?",
            "question_id": 3101,
            "template_id": "reasoning",
            "answer": "two",
            "image": "bar_val_hard_00000001.png",
            "answer_bbox": []
          },
          {
            "question": "Which algorithm has highest accuracy for any dataset?",
            "question_id": 3102,
            "template_id": "reasoning",
            "answer": "brave",  // could be  "answer": "yes"
            "image": "bar_val_hard_00000001.png",
            "answer_bbox": [
              302.5178277191559,
              394.48,
              82.5,
              40
            ]
          }
        ]
2. Preprocessing question data
```
    python preprocess.py [DVQA directory] 
    e.g. python preprocess.py data/

```
3. Run train.py
```
    python train.py [DVQA directory]
    e.g. python preprocess.py data/
```



#### TODO list

0. Unzip data
1. Implement the preprocess.py, reads all dictionary and build vocabulary / classification classes
2. Implement the YES baseline, numbers should perfectly match with Table 3?
3. 



