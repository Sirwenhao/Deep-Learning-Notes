import cv2
import numpy as np

class ActivationAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.append(
                    target_layer.register_forward_hook(
                        self.save_activastion))
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backwarrd_hook(
                        self.save_gradients))
    
    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
        
    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients += [grad.cpu().detach()]
        
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)
        
    def release(self):
        for handle in self.handles:
            handle.remove()
            
class GradCAM:
    def __init__(self, model, target_layers, reshape_transform=None, use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationAndGradients(
            self.model, target_layers, reshape_transform)
        
    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdim=True)
        
    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss
        
    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        return cam
        
    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height
        
    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in slef.activations_and_grads_gradients]
        target_size = self.get_target_width_height(input_tensor)
        
        cam_per_target_layer = []
        
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

        
    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maxmium(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)
    
    
    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)
        return result
    
    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()
            
        # 正向传播得到网络输出logits
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)
            
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f'category id:{target_category}')
        else:
            assert (len(target_category) == input_tensor.size(0))
            
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retrain_graph=True)
        
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)
        
    def __del__(self):
        self.activations_and_grads.release()
        
    def __enter__():
        return self
        
    def __exit__(self):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f'An exception occurred in CAM with block: {exc_type}. Message: {exc_value}')
        return True
        
    def show_cam_on_image(img, mask, use_rgb, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        
        if np.max(img) > 1:
            raise Exception("The input image should np.float32 in the range [0, 1]")
            
        cam = heatmap + img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
        
    def center_crop_img(img, size):
        h, w, c = img.shape
        
        if w == h == size:
            return img
            
        if  w < h:
            ratio = size / w
            new_w = size
            new_h = int(h * ratio)
        else:
            ratio = size / h
            new_h = size
            new_w = int(w * ratio)
            
        img = cv2.resize(img, dsize=(new_w, new_h))
        
        if new_w == size:
            h = (new_h - size) // 2
            img = img[h: h+size]
        else:
            w = (new - size) // 2
            img = img[:, w:w+size]
        return img
        
        
            
              
