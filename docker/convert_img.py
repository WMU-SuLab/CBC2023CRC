import os

import click
from skimage import io


def convert_img(file_path, output_dir_path, new_ext):
    print(file_path)
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f'{name}.{new_ext}'
    new_file_path = os.path.join(output_dir_path, new_file_name)
    img = io.imread(file_path)
    io.imsave(new_file_path, img)


@click.command()
@click.option('-i', '--input_file_path', type=str, default='')
@click.option('-d', '--input_dir_path', type=str, default='')
@click.option('-o', '--output_dir_path', type=str, default='./data/out/images')
@click.option('-e', '--new_ext', type=str, default='png')
def main(input_file_path: str, input_dir_path: str, output_dir_path: str, new_ext: str):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)
    if input_file_path and os.path.exists(input_file_path) and os.path.isfile(input_file_path):
        with open(input_file_path, 'r') as f:
            file_paths = f.read().strip().split('\n')
        for file_path in file_paths:
            convert_img(file_path, output_dir_path, new_ext)
    if input_dir_path and os.path.exists(input_dir_path):
        for file_name in os.listdir(input_dir_path):
            file_path = os.path.join(input_dir_path, file_name)
            convert_img(file_path, output_dir_path, new_ext)


if __name__ == '__main__':
    main()
