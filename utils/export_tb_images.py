from collections import defaultdict, namedtuple
from typing import List

import os

import tensorflow as tf
import imageio
from tensorboard.compat.proto import event_pb2

TensorBoardImage = namedtuple("TensorBoardImage", ["topic", "image", "cnt"])

def extract_images_from_event(event_filename: str, image_tags: List[str]):
    topic_counter = defaultdict(lambda: 0)

    serialized_examples = tf.data.TFRecordDataset(event_filename)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for v in event.summary.value:
            if v.tag in image_tags:

                if v.HasField('tensor'):  # event for images using tensor field
                    s = v.tensor.string_val[2]  # first elements are W and H

                    tf_img = tf.image.decode_image(s)  # [H, W, C]
                    np_img = tf_img.numpy()

                    topic_counter[v.tag] += 1

                    cnt = topic_counter[v.tag]
                    tbi = TensorBoardImage(topic=v.tag, image=np_img, cnt=cnt)

                    yield tbi

                if v.HasField('image'):  # event for images using tensor field
                    s = v.image.encoded_image_string

                    tf_img = tf.image.decode_image(s)  # [H, W, C]
                    np_img = tf_img.numpy()

                    topic_counter[v.tag] += 1

                    cnt = topic_counter[v.tag]
                    tbi = TensorBoardImage(topic=v.tag, image=np_img, cnt=cnt)

                    yield tbi


if __name__ == '__main__':
    out_root = '/mnt/raid/dkomorowicz/plots'

    tags = ['val/GT_pred_depth', 'val/GT_pred_static', 'val/path']

    in_path = '/mnt/raid/dkomorowicz/david_save_nerfw/logs'
    for exp_name in os.listdir(in_path):
        for version in os.listdir(os.path.join(in_path, exp_name)):
            dirname = os.path.join(in_path, exp_name, version, 'tf')
            filename = os.listdir(dirname)

            for tag, img, num in extract_images_from_event(os.path.join(dirname, filename[0]), tags):
                out_path = os.path.join(out_root, exp_name, version, tag.split('/')[-1])
                os.makedirs(out_path, exist_ok=True)
                fname = f"image_{num:05d}.png"
                imageio.imwrite(os.path.join(out_path, fname), img)
