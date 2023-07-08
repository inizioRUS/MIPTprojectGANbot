import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import copy


class MLPart:
    def __init__(self):
        self.device = "cpu"
        self.imsize = 256 if torch.cuda.is_available() else 256
        self.loader = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),  # уменьшаем картинку
            transforms.ToTensor()])  # делаем из картинки тензор
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    def image_loader(self, image_byte):
        image = Image.open(io.BytesIO(image_byte))
        # создаем фейковый дименшн для батча
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def load_image(self, first, second):
        return self.image_loader(first), self.image_loader(second)

    class ContentLoss(nn.Module):
        def __init__(self, target):
            super().__init__()

            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img):
        content_layers = self.content_layers_default
        style_layers = self.style_layers_default
        cnn = copy.deepcopy(cnn)

        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)

                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):

        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self, normalization_mean, normalization_std,
                           content_img, style_img, num_steps=320,
                           style_weight=300000, content_weight=1):
        """ Поехали! """
        input_img = content_img.clone()

        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(self.cnn, normalization_mean,
                                                                              normalization_std,
                                                                              style_img, content_img)
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # корректируем значения, чтобы они лежали в пределах `[0..1]`
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)
        input_img.data.clamp_(0, 1)
        unloader = transforms.ToPILImage()
        image = input_img.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def gram_matrix(self, input):
        a, b, c, d = input.size()

        features = input.view(a * b, c * d)

        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()

        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
