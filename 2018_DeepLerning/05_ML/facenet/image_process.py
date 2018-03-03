import os
import shutil
base_dir = os.getcwd() + '/CAS-PEAL-R1/'
out_dir = os.getcwd() + '/lfw/'
num = 0

for cas_dir in os.listdir(base_dir):
    if cas_dir == 'FRONTAL' or cas_dir == 'pose':
        process_dir = base_dir + cas_dir
        for root, dirs, imgs in os.walk(process_dir):
            for i in range(len(imgs)):
                if(imgs[i].endswith('tif')):
                    img_no = imgs[i].split('_')[1]
                    img_path = root + '/' + imgs[i]
                    out_lfw = out_dir + img_no 
                    if os.path.exists(out_lfw) == False:
                        os.mkdir(out_lfw)
                    shutil.copy(img_path,out_lfw)
                    num = num + 1
                    if num % 500 == 0 and num >= 500:
                        print('already processed ' + str(num) + ' numbers images')