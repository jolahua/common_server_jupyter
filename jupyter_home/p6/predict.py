#!/usr/bin/python
# -*- coding:utf-8 -*-


import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Image read
    image = Image.open(image)
    
    # use pytorch less efficiency
#     trans = transforms.Compose([ transforms.Resize(256),
#                                  transforms.CenterCrop(224),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])])

#     image_tensor = trans(image)
    
    # use numpy 1.5 times efficiency
    image = image.resize((256, 256))
    image.thumbnail((224, 224))
    
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None, show=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.set_title(title)
    ax.imshow(image)
    
    return ax


def test():

    test_dir = 'flowers/test'
    image_dir = test_dir + '/' + random.choice(os.listdir(test_dir))
    image_path = image_dir + '/' + random.choice(os.listdir(image_dir))
    print(image_path)
    imshow(process_image(image_path), title='Flowers')

    
def predict(image_path, model, topk=5, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    """
    torch.topk:
    第一个参数需要的是tensor对象  第二个就是需要的前几个值
    """
    if torch.cuda.is_available() and device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'
        
    image_pil = process_image(image_path)  # PIL格式
    
    image_tensor = torch.tensor(image_pil)

    # 第一种方法， 直接一层转换
#     image_tensor = image_tensor.type(torch.cuda.FloatTensor).resize(1, 3, 224, 224).to(device)
    
    # 第二种方法，先增加维度在转float
    image_tensor = image_tensor.unsqueeze_(0).float().to(device)

#     model = model.float()
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model.forward(image_tensor)
        
    output = torch.exp(output)
    probs, classes = torch.topk(output, topk, sorted=True)
    
    return probs.tolist()[0], classes.tolist()[0]
    
    
def show_type_rank(probs, classes, image_path, cat_to_name):
    model.class_to_idx = cat_to_name   # 这个在load_state的时候就应该加载了，  这里作为测试用  需要重新训练模型并保存
    class_to_idx = {i:j for j, i in train_datasets.class_to_idx.items()}  # 因为使用训练集训练 所以先对应到训练集  在对应到json
#     classes = [model.class_to_idx[str(classs)] for classs in classes]  #  没有使用训练集先对应
    classes = [model.class_to_idx[str(class_to_idx[classs])] for classs in classes]
    
    probs.reverse()
    classes.reverse()
    
    plt.figure(figsize = [10, 5]) 
    ax1 = plt.subplot(1, 2, 1)
    ax1 = imshow(process_image(image_path), ax=ax1, title=classes[-1])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = plt.subplot(1, 2, 2)
    y_len = np.arange(len(classes))
    ax2.barh(y_len, probs, height=0.7, align='center', tick_label=probs)
    ax2.set_yticks(y_len)
    ax2.set_yticklabels(classes)
    ax2.set_title('F_classif scores of the features.')
    
    plt.show()
    return classes 


def main(model, criterion, optimizer, input_image, top_k, cat_to_name, device):
    probs, classes = predict(image_path, model, device)
    print(probs)
    print(classes)
    # [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
    # ['70', '3', '45', '62', '55']

    show_type_rank(probs, classes, image_path)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Predict Image')
    parser.add_argument('input', type=str, help='the image you want to predict')
    parser.add_argument('checkpoint', type=str, help='your model path', default='save_directory/checkpoint.pth')
    parser.add_argument('--top_k', type=int, help='the numbers of flowers you want to show', default=5')
    parser.add_argument('--category_names', type=str, help='your class_to_idx path', default='cat_to_name.json')
    parser.add_argument('--gpu', help='weather use gpu', default='cuda', nargs='?')

    args = parser.parse_args()
    args.gpu = 'cuda' if args.gpu is None else ''
    
    cat_to_name = get_class_to_ids(args.category_names)
    # trainloaders, validloaders, testloaders, model_type, epochs, learning_rate, hidden_units, device
    net = Net()
    model, criterion, optimizer = net.load_model(args.checkpoint)
    print('Predict Start ...')
    time1 = time.time()                    
    main(model, criterion, optimizer, args.input, args.top_k, args.category_name, args.gpu)
    print('Predicting Finishing ...   Timing: %s seconds' % (str(int(time.timne() - time1))))                    
