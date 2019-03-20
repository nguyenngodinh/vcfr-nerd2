from collections import defaultdict
import csv
import os
import shutil
import xlsxwriter

def label_images(csv_file_dir):
    csv.register_dialect('train_dialect', delimiter = ',', skipinitialspace=True)
    label_i = defaultdict(list)

    with open(csv_file_dir) as csv_file:
        csv_reader = csv.reader(csv_file, dialect='train_dialect')
        #skip the heaser
        next(csv_reader, None)
        count = 0
        for image, label in csv_reader:
            label_i[label.strip()].append(image.strip())
            count += 1

        print('Read:', count, ' data')

    return label_i


def categorise_image(train_data_dir, dst_dir, train_csv_file):
    labels = label_images(train_csv_file)

    temp_folder = dst_dir
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    for label, images in labels.iteritems():
        label_folder = os.path.join(temp_folder,label)
        print('label folder:',label_folder)
        if not os.path.exists(label_folder):
            os.mkdir(label_folder)
            for image in images:
                src = os.path.join(train_data_dir,image)
                dst = os.path.join(label_folder, image)
                print('src:',src,'dst:',dst, 'lf:', label_folder)
                shutil.copy(src, dst)

def write_xls(label_imgs, outfile):
    
    wb = xlsxwriter.Workbook(outfile)
    ws = wb.add_worksheet()
    
    counter=0
#    for label, imgs in label_imgs.iteritems():
    for label in sorted(label_imgs.iterkeys()):
        imgs = label_imgs[label]
        col_l = 0
        row_l = counter
        ws.set_column(col_l, 10)

        ws.write(row_l, col_l, label)
        img_col = 2
        for img in imgs:
            col_i = img_col
            row_i = counter
            img_col += 1
            ws.set_column(col_i, 30)
            ws.write(row_i, col_i, img)
        ws.write(counter, 1, img_col-2)
        counter += 1
    wb.close()

def count(list_):
    count =0
    for l in list_:
        count +=1
    return count

#labels = label_images('train.csv')
#write_xls(labels, 'labels.xlsx')

#categorise_image('train/', 'label_images', 'train.csv')
