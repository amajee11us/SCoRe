import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 10],
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = '/archive/datasets/coco/cocosplit_fsdet/datasplit/trainvalno5k.json'
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
        if a['iscrowd'] == 1 or (a['category_id'] not in list(ID2CLASS.keys())):
            # print("Skipping image_id {}, with cat_id {} and crowd {}".format(a['id'], a['category_id'], a['iscrowd']))
            continue
        anno[a['category_id']].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in ID2CLASS.keys():
            img_ids = {}
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []
            for shots in [10, 30]:
                while True:
                    print("Class {}, number of instances {}, shots {}".format(c, len(list(img_ids.keys())), shots))
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                new_data = {
                    'info': data['info'],
                    'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(data_path, ID2CLASS[c], shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    save_dir = os.path.join('datasets', 'cocosplit', 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    ID2CLASS = {
        1: 'truck',
        2: 'traffic light',
        3: 'fire hydrant',
        4: 'stop sign',
        5: 'parking meter',
        6: 'bench',
        7: 'elephant',
        8: 'bear',
        9: 'zebra',
        10: 'giraffe',
        11: 'backpack',
        12: 'umbrella',
        13: 'handbag',
        14: 'tie',
        15: 'suitcase',
        16: 'frisbee',
        17: 'skis',
        18: 'snowboard',
        19: 'sports ball',
        20: 'kite',
        21: 'baseball bat',
        22: 'baseball glove',
        23: 'skateboard',
        24: 'surfboard', 
        25: 'tennis racket',
        26: 'wine glass',
        27: 'cup',
        28: 'fork',
        29: 'knife',
        30: 'spoon',
        31: 'bowl',
        32: 'banana',
        33: 'apple',
        34: 'sandwich',
        35: 'orange',
        36: 'broccoli',
        37: 'carrot',
        38: 'hot dog', 
        39: 'pizza',
        40: 'donut',
        41: 'cake',
        42: 'bed',
        43: 'toilet',
        44: 'laptop',
        45: 'mouse',
        46: 'remote',
        47: 'keyboard',
        48: 'cell phone',
        49: 'microwave',
        50: 'oven',
        51: 'toaster',
        52: 'sink',
        53: 'refrigerator', 
        54: 'book',
        55: 'clock',
        56: 'vase',
        57: 'scissors',
        58: 'teddy bear',
        59: 'hair drier',
        60: 'toothbrush',
        61: 'person',
        62: 'bicycle',
        63: 'car',
        64: 'motorcycle',
        65: 'airplane',
        66: 'bus',
        67: 'train',
        68: 'boat', 
        69: 'bird',
        70: 'cat',
        71: 'dog',
        72: 'horse',
        73: 'sheep',
        74: 'cow',
        75: 'bottle',
        76: 'chair',
        77: 'couch',
        78: 'potted plant',
        79: 'dining table',
        80: 'tv',
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
