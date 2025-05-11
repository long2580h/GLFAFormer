import argparse
import os
import torch
from thop import profile
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
from models import get_model
from PIL import Image 
import pickle
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import time
import shutil
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
from torch import nn
SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}



def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 模型输出为二分类概率（sigmoid处理后的值）
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # 转换为numpy数组
    all_probs = np.array(all_probs).squeeze()
    all_labels = np.array(all_labels).squeeze()

    # 计算阈值化后的预测类别（默认阈值为0.5）
    preds = (all_probs >= 0.5).astype(int)

    # 核心指标计算
    roc_auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, preds)

    # ROC曲线绘制
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    return {
        'ROC-AUC': roc_auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Confusion Matrix': cm
    }




def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)



def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, find_thres=False):

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Length of dataset: %d" %(len(loader)))
        for img, label in loader:
            in_tens = img.cuda()

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())


    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # ================== save this if you want to plot the curves =========== # 

    # torch.save( torch.tensor(y_true),  'baseline_predication_for_pr_roc_curve_true.pth' )
    # torch.save(torch.tensor(y_pred), 'pr_roc_curve_pred_progan_mine.pth')
    # torch.save(torch.stack([torch.tensor(y_true),torch.tensor(y_pred)]), 'pr_roc_curv_biggan_mine.pth')
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    # print(y_pred[y_true == 0] > best_thres)
    # print(y_true[y_true == 0])
    # assert 0
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

    
    



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 




def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        data_mode, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None):

        # assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])


    def read_path(self, real_path, fake_path, data_mode, max_sample):

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)


        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        # assert len(real_list) == len(fake_list)

        return real_list, fake_list



    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)


        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label


def count_parameters(model: nn.Module) -> int:
    """统计PyTorch模型的参数量"""
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")


    opt = parser.parse_args()

    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)
    model = get_model(opt.arch)
    # model = tf.keras.models.load_model(args.MODEL)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    if 'optimizer' in state_dict.keys():
        # optimizer = state_dict.get('optimizer')
        # state = optimizer.get('state')
        state_dict1 = {'weight' : state_dict.get('model').get('weight'),'bias':state_dict.get('model').get('bias')}
        model.fc.load_state_dict(state_dict1)
    else:
        model.fc.load_state_dict(state_dict)
    print ("Model loaded..")
    model.eval()
    model.cuda()
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")
    inputs = torch.randn(96, 3, 224, 224)
    flops, params = profile(model, inputs=(inputs,))
    print(f"FLOPs: {flops / 1e9:.1f}G, Params: {params / 1e6:.1f}M")
    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [ dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode, key=None) ]



    for dataset_path in (dataset_paths):
        set_seed()

        dataset = RealFakeDataset(  dataset_path['real_path'], 
                                    dataset_path['fake_path'], 
                                    dataset_path['data_mode'], 
                                    opt.max_sample, 
                                    opt.arch,
                                    jpeg_quality=opt.jpeg_quality, 
                                    gaussian_sigma=opt.gaussian_sigma,
                                    )

        start_time = time.time()
        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")
        # test_metrics = evaluate_model(model, loader, 'cuda')
        # print(f"Final Test Metrics:")
        # print(f"- ROC-AUC: {test_metrics['ROC-AUC']:.4f}")
        # print(f"- Precision: {test_metrics['Precision']:.4f}")
        # print(f"- Recall: {test_metrics['Recall']:.4f}")
        # print(f"- F1: {test_metrics['F1']:.4f}")
        # print("Confusion Matrix:\n", test_metrics['Confusion Matrix'])
        # with open( os.path.join(opt.result_folder,'ap.txt'), 'a') as f:
        #     f.write(dataset_path['key']+': ' + str(round(ap*100, 2))+'\n' )
        #
        # with open( os.path.join(opt.result_folder,'acc0.txt'), 'a') as f:
        #     f.write(dataset_path['key']+': ' + str(round(r_acc0*100, 2))+'  '+str(round(f_acc0*100, 2))+'  '+str(round(acc0*100, 2))+'\n' )

