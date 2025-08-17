import argparse
from torchvision import transforms
import argparse
import torch
import torchvision
from config import APPLICATION_NAME, APPLICATION_DESCRIPTION, DEVICE
from restyler.model import VGGNet
from restyler.utils import load_image


def get_arguments():
    parser = argparse.ArgumentParser(prog=APPLICATION_NAME,
                                     description=APPLICATION_DESCRIPTION)
    
    parser.add_argument('--content', type=str, default='data/content.jpg')
    parser.add_argument('--style', type=str, default='data/style.jpg')
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--total_step', type=int, default=2000)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--style_weight', type=float, default=300)
    parser.add_argument('--lr', type=float, default=0.003)

    return parser.parse_args()

def main():
    config = get_arguments()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                std=(0.229, 0.224, 0.225))])

    content = load_image(config.content, transform, max_size=config.max_size)
    style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])

    target = content.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([target], lr=config.lr, betas=[0.5, 0.999])
    vgg = VGGNet().to(DEVICE).eval()

    for step in range(config.total_step):
        
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            content_loss += torch.mean((f1 - f2)**2)

            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            style_loss += torch.mean((f1 - f3)**2) / (c * h * w) 
        
        loss = content_loss + config.style_weight * style_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % config.log_step == 0:
            print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}' 
                    .format(step+1, config.total_step, content_loss.item(), style_loss.item()))

        if (step+1) % config.sample_step == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, 'output-{}.png'.format(step+1))
    
if __name__ == '__main__':
    main()