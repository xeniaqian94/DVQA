import os
import sys
import json
import pickle
import ast
import nltk
import tqdm
import h5py
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
from collections import defaultdict

'''
Usage: python preprocess.py data/ True/False

'''


def process_question(root, split, word_dic=None, answer_dic=None, question_subdir='qa', qs_level="",
                     image_dir="data/images/unsplitted_images", save_image_to_HDF5=True, with_bounding_box=False):
    # question_subdir (for CLEVR) = "questions",
    # qs_level can be either "_easy" or "_hard"
    print("save_image_to_HDF5", save_image_to_HDF5)
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(
            os.path.join(root, question_subdir, split + qs_level + "_qa.json")) as f:  # f'CLEVR_{split}_questions.json'
        data = json.load(f)

    result = []
    word_index = 1  # my guess, 0 reserved for OOV?
    answer_index = 0
    miss_image_count = 0

    if os.path.exists(os.path.join(os.path.split(image_dir)[0], split + '_IMAGES_.hdf5')):
        os.remove(os.path.join(os.path.split(image_dir)[0], split + '_IMAGES_.hdf5'))

    image_filename_2_hdf5_idx_dict = defaultdict(lambda: len(image_filename_2_hdf5_idx_dict))

    for i, question in enumerate(tqdm.tqdm(data)):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        ## TODO: for captioning, maybe not for classification
        # word_map['<unk>'] = len(word_map) + 1
        # word_map['<start>'] = len(word_map) + 1
        # word_map['<end>'] = len(word_map) + 1
        # word_map['<pad>'] = 0

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']
        # bbox = question['answer_bbox']

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        if not os.path.exists(os.path.join(image_dir, question['image'])):
            miss_image_count += 1
            continue

        already_saved_image_in_HDF5 = (True if question['image'] in image_filename_2_hdf5_idx_dict else False)

        hdf5_idx_for_this_image = image_filename_2_hdf5_idx_dict[question['image']]
        result.append(
            (
                hdf5_idx_for_this_image,
                question['image'],
                question_token,
                answer,
                question['template_id'],  # question/answer class
            )
        )

        ## TODO: add the answer bounding box
        # "answer_bbox": []

        if already_saved_image_in_HDF5:
            continue

    print("\nReading %s images and qa pairs, storing to file...\n" % split, save_image_to_HDF5)
    if save_image_to_HDF5:
        with h5py.File(os.path.join(os.path.split(image_dir)[0], split + '_IMAGES_.hdf5'),
                       'a') as h:  # save to HDF5 file for faster data IO

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(image_filename_2_hdf5_idx_dict), 3, 224, 224),
                                      dtype='uint8')  ## NOTE: question: shall we resize??

            for i, image_path in enumerate(tqdm.tqdm(image_filename_2_hdf5_idx_dict.keys())):
                # img = np.asarray(Image.open(os.path.join(image_dir, question['image'])).convert('RGB'))
                img = imread(os.path.join(image_dir, image_path), mode="RGB")  # (448,448,3)
                img = imresize(img, (224, 224))  # (224,224,3)

                img = img.transpose(2, 0, 1)  # TODO: change back switch dimension
                assert img.shape == (3, 224, 224)  # TODO: change magic number

                # assert img.shape == (224,224,3)

                assert np.max(img) <= 255

                hdf5_idx_for_this_image = image_filename_2_hdf5_idx_dict[image_path]
                try:
                    images[hdf5_idx_for_this_image] = img
                except:
                    print("img error save hdf5", question['image'])
                h.flush()

    print("Length of questions/results", len(result))
    print("Miss_image_count", miss_image_count)

    with open(f'data/{split}' + str("_w_bbx" if with_bounding_box else "") + '.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


# resize = transforms.Resize([128, 128]) # No need, images are 448*448


# def process_image(path, output_dir):
#     images = os.listdir(path)
#
#     if not os.path.isdir(output_dir):
#         os.mkdir(output_dir)
#
#     for imgfile in tqdm.tqdm(images):
#         img = Image.open(os.path.join(path, imgfile)).convert('RGB')
#         img = resize(img)
#         img.save(os.path.join(output_dir, imgfile))


if __name__ == '__main__':
    root = sys.argv[1]
    save_image_to_HDF5 = bool(ast.literal_eval(sys.argv[2]))
    print("save_image_to_HDF5", save_image_to_HDF5)
    word_dic, answer_dic = process_question(root, 'train',
                                            save_image_to_HDF5=save_image_to_HDF5)  # This step takes ~3 mins

    with open('data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

    process_question(root, 'val_easy', word_dic, answer_dic,
                     save_image_to_HDF5=save_image_to_HDF5)  # CLEVR has validation set; DVQA only has test set
    process_question(root, 'val_hard', word_dic, answer_dic,
                     save_image_to_HDF5=save_image_to_HDF5)  # CLEVR has validation set; DVQA only has test set

    print("[SANITY CHECK] unique answers in train set == 1076? (Section 4.4)",
          len(
              answer_dic))  # is indeed the same lol . If 1576, it's fine. Because the 500 more are from val_hard, not saved in answer_dic

    # process_image(
    #     os.path.join(sys.argv[1], 'images/unsplitted_images',split="train"),
    #     os.path.join(sys.argv[1], 'images/train_preprocessed'),
    # )
    # process_image(
    #     os.path.join(sys.argv[1], 'images/unsplitted_images',split="val_hard"),
    #     os.path.join(sys.argv[1], 'images/val_hard_preprocessed'),
    # )
    # process_image(
    #     os.path.join(sys.argv[1], 'images/unsplitted_images',split="val_easy"),
    #     os.path.join(sys.argv[1], 'images/val_easy_preprocessed'),
    # )
