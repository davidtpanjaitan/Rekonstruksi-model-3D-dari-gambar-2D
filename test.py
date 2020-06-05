import os, sys
import glob
from PIL import Image
import tqdm
import argparse
import pickle

from segmentation.use_tiramisu import TiramisuEvaluator
from segmentation.separate_objects import separate_by_contour
from reconstruction.use_pifu import PIFuEvaluator, opt

def parse_args(args):
    parser = argparse.ArgumentParser(description='Some arguments to configure the networks the app uses')
    parser.add_argument('--seg_net',
                        help='Path to the tiramisu reconstruction network weights',
                        default='segmentation/models/tiramisu_multicoco_minloss52.h5')
    parser.add_argument('--disable_multi_object',
                        help='with this flag, each image will only produce one reconstructed object',
                        action='store_true')
    parser.add_argument('--disable_pifu',
                        help='Do not process reconstruction in this run',
                        action='store_true')
    parser.add_argument('--disable_tiramisu',
                        help='Do not process segmentation in this run',
                        action='store_true')

    return parser.parse_known_args(args)[0]

args = parse_args(sys.argv[1:])
print('args parsed!')

test_images = glob.glob(os.path.join('test_images', '*'))
test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
test_masks = [f[:-4]+'_mask.png' for f in test_images]

if not args.disable_tiramisu:
    seg = TiramisuEvaluator(net_path=args.seg_net)
    for i, mask in enumerate(test_masks):
        if not os.path.exists(mask):
            print('creating mask for {}'.format(test_images[i]))
            img = Image.open(test_images[i]).convert('RGB')
            new_mask = seg.segment(img, img.width, img.height)
            new_mask.save(mask)
    del seg

if not args.disable_multi_object:
    test_images, test_masks = separate_by_contour(test_images, test_masks)
    if args.disable_pifu:
        with open('imagelist.pkl', 'wb') as of:
            pickle.dump((test_images, test_masks), of)

if not args.disable_pifu:
    if args.disable_multi_object and os.path.exists('imagelist.pkl'):
        with open('imagelist.pkl', 'rb') as inf:
            test_images, test_masks = pickle.load(inf)
        os.remove('imagelist.pkl')
    rec = PIFuEvaluator(opt)
    for img, mask in tqdm.tqdm(zip(test_images, test_masks)):
        try:
            print(img, mask)
            data = rec.load_image(img, mask)
            rec.eval(data, True)
        except Exception as e:
            print("error:", e.args)