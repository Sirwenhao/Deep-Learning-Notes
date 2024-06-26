import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import create_regnet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # load image
    
    img_path = r''
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.show(img)
    
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    
    json_path = r''
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    
    with open(json_path, 'r') as f:
        class_indict = json.load(f)
        
    # create model
    
    model = create_regnet(model_name="RegNetY_400MF", num_classes=5).to(device)
    model_weight_path = r''
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        
    print_res = "class: {}  probL {:.5}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    
    for i in range(len(predict)):
        print("class: {:10} prob: {:.5}".format(class_indict[str(i)], predict[predict_cla].numpy()))
        plt.show()
        
if __name__ == "__main__":
    main()
        
        
    
    

