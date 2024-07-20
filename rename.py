import os
import shutil
from tqdm import tqdm
root = 'datasets/expend/expend_lines_15000/lines'
save_root = 'datasets/expend/expend_lines_50000/lines'
names = os.listdir(root)
for i, name in tqdm(enumerate(names)):
    newname = str(i)
    sub_root = os.path.join(save_root,newname)
    if not os.path.exists(sub_root):
        os.makedirs(sub_root)
    imgs = os.listdir(os.path.join(root,name))
    for img in imgs:
        img_path = os.path.join(root,name,img)
        shutil.copy(img_path,os.path.join(sub_root,img))
    
