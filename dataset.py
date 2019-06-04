from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from functools import lru_cache
import sys
import torchvision
import torch.nn as nn
from tqdm import tqdm
from unet import UNetWithResnet50Encoder
from PIL import Image, ImageDraw
import random
import copy
from os import listdir
from os.path import isfile, join

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def detokenize(self, words):
        sentence = []
        for w in words:
            sentence.append(self.idx2word[w])
        return sentence

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(question, answer):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'question'    : question['question'],
        'answer'      : answer}
    return entry

def resize(dataroot, base_folder,coco_folder_resize, size):
    print(coco_folder_resize + " does not exists, resizing original")
    coco_folder_resize = os.path.join(dataroot, coco_folder_resize)

    os.mkdir(coco_folder_resize)
    coco_folder = os.path.join(dataroot, base_folder)

    images = [f for f in listdir(coco_folder) if isfile(join(coco_folder,f))]

    resize = size
    crop = size

    _transforms = []
    if resize is not None:
        _transforms.append(transforms.Resize(resize))
    if crop is not None:
        _transforms.append(transforms.CenterCrop(crop))
    transform = transforms.Compose(_transforms)

    for i, o in enumerate(images):
        with open(os.path.join(coco_folder, o), 'rb') as f:
            img = Image.open(f).convert('RGB')
            x = transform(img)
            x.save(os.path.join(coco_folder_resize, o), "JPEG", quality=100)
        if i % 10000 == 0:
            print(i)



def _load_dataset(dataroot, name):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if name == "test2015":
        question_path = os.path.join(
            dataroot, 'OpenEnded_mscoco_test2015_questions.json'
        )
    elif name == "testgqa":
        question_path = os.path.join(
            dataroot, 'OpenEnded_mscoco_testgqa_questions.json'
        )
    else:
        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' %
                      (name + '2014')
        )

    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    if 'test'!=name[:4]: # train, val

        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        i = 0
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            # if question['question_id'] == 100500000:
            #     print(i)
            #     sys.exit()

            entries.append(_create_entry(question, answer))
            i+=1
    else:
        entries = []
        for question in questions:
            entries.append(_create_entry(question, None))
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', size=64, npy=False, layer=4, finetune=False):

        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.entries = _load_dataset(dataroot, name)
        self.size = size
        self.npy = npy
        self.layer = layer
        self.finetune = finetune
        self.dataroot = dataroot

        resize = size
        crop = size
        _transforms = []

        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        features_filename = os.path.join(dataroot, "%s_unet_%s.pkl" % (name, str(size)))
        print("Loading %s" % features_filename)
        if not os.path.exists(features_filename):
            self.compute_features(self.entries, dataroot, name, layer, size, features_filename)
        self.features = pickle.load(open(features_filename, 'rb'))

        random_key = list(self.features.keys())[0]
        self.v_dim = self.features[random_key][0]["layer_%s" % str(self.layer)].shape[0]

        # self.b_dim = self.features[random_key][0]["layer_4"].shape[0]*2

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):

        # if not self.use_resnet:
        #     self.features = torch.from_numpy(self.features)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']

            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None


    def _read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img_black = copy.copy(img)

            q = random.randint(0, 75)  # change if you want
            p = random.randint(0, 75)  # change if you want
            r = random.randint(0, 75)  # change if you want
            s = random.randint(0, 75)  # change if you want
            draw = ImageDraw.Draw(img_black)
            draw.rectangle(((q, p), (r, s)), fill="black")
            # print(type(draw))
            # img_black.save("1", "JPEG")
            # img.save("2", "JPEG")            #
            # sys.exit()
            return self.transform(img_black), self.transform(img)


    def compute_features(self, entries, dataroot, name, layer, size, features_filename):
        model = UNetWithResnet50Encoder(finetune=False).cuda()

        base_folder = name + '2014' if 'test' != name[:4] else name
        coco_folder_resize = '%s_%s' % (str(base_folder), str(size))

        print("Reading in folder %s/%s" % (dataroot,coco_folder_resize))
        if not os.path.exists(os.path.join(dataroot, coco_folder_resize)):
            resize(dataroot, base_folder,coco_folder_resize, size)

        print(features_filename, "not found, extracting...")
        features = {}
        for i in tqdm(range(len(entries))):
            entry = entries[i]
            img_id = entry["image_id"]
            if img_id in features:
                continue
            filename = os.path.join(dataroot, coco_folder_resize, 'COCO_%s_%s.jpg' % (str(base_folder),str(img_id).zfill(12)))
            assert os.path.exists(filename), filename + " does not exists"
            image_box, image_orig = self._read_image(filename)
            image_orig_np16 = image_orig.numpy().astype(np.float16)
            image_box_np16  = image_box.numpy().astype(np.float16)
            inp = torch.unsqueeze(image_box, 0).cuda() # 1,dim,x,x
            _, pre_pools = model.encode(inp)
            for r in pre_pools.keys():
                pre_pools[r] = pre_pools[r].squeeze(0).cpu().numpy().astype(np.float16)
            features[img_id] = [pre_pools, image_orig_np16, image_box_np16]
        print("Dumping in filename %s with size %s" % (features_filename, str(len(features))))
        pickle.dump(features, open(features_filename,'wb+'))
        del model

    def __getitem__(self, index):
        entry = self.entries[index]

        img_id = entry["image_id"]
        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        question_raw = entry['question']

        target = torch.tensor(0)
        if None!=answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)

        pre_pools, image, image_box = self.features[img_id]
        for r in pre_pools.keys():
            pre_pools[r] = pre_pools[r].astype(np.float32)
        image = image.astype(np.float32)




        return [question_id, pre_pools, image, question], [image, target]
        # return [question_id, pre_pools, image, question], [image_orig,target]

    def __len__(self):
        return len(self.entries)
