from collections import defaultdict
import csv
import os
import shutil


def categorise_image(train_data_dir, dst_dir, train_csv_file):
    csv.register_dialect('train_dialect', delimiter =',',skipinitialspace=True)

    label_images = defaultdict(list)

    with open(train_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, dialect='train_dialect')
        next(csv_reader, None)#Skip the header
        for image, label in csv_reader:
            label_images[label.strip()].append(image.strip())

        temp_folder = dst_dir
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        for label, images in label_images.iteritems():
            label_folder = os.path.join(temp_folder,label)
            print('label folder:',label_folder)
            if not os.path.exists(label_folder):
                os.mkdir(label_folder)
                for image in images:
                    src = os.path.join(train_data_dir,image)
                    dst = os.path.join(label_folder, image)
                    print('src:',src,'dst:',dst, 'lf:', label_folder)
                    shutil.copy(src, dst)

        print(label_images.count())

categorise_image('train/', 'label_images', 'train.csv')
