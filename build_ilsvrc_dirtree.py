import os
from os import path
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "rel_image_list",
        help="Text file listing images in relative path and assigned label")
    parser.add_argument("image_dir", help="Where the images are stored")
    parser.add_argument(
        "folder_to_class_list", help="text file mapping folders to classes")
    parser.add_argument("outdir", help="Where to create the folder with links")
    return parser.parse_args()


def build_symlink_dir(outdir, inputdir, image_list, label2folder):
    for folder in label2folder.values():
        outpath = path.join(outdir, folder)
        if not path.exists(outpath):
            os.makedirs(outpath)
#    import ipdb; ipdb.set_trace()
    for line in image_list:
        if line.isspace(): continue  # to avoid empty lines
        (rel_path, label) = line.split()
        folder = label2folder[label.strip()]
        source_path = path.join(inputdir, rel_path.strip())
        dest_path = path.join(outdir, folder, rel_path.strip())
        os.symlink(source_path, dest_path)


if __name__ == '__main__':
    args = get_args()
    with open(args.folder_to_class_list, 'rt') as tf:
        lines = tf.readlines()
        label2folder = {}
        for line in lines:
            (folder, label) = line.split()
            label2folder[label.strip()] = folder.strip()
    with open(args.rel_image_list, 'rt') as tf:
        build_symlink_dir(args.outdir, args.image_dir, tf.readlines(), label2folder)
