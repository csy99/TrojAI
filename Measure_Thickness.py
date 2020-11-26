from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import warnings

from scipy.sparse.linalg import svds

from sklearn.metrics import f1_score
import torch.nn.functional as F


warnings.filterwarnings("ignore")



class Attacks(object):
    """
    An abstract class representing attacks.
    Arguments:
        name (string): name of the attack.
        model (nn.Module): a model to attack.
    .. note:: device("cpu" or "cuda") will be automatically determined by a given model.
    """
    def __init__(self, name, model):
        self.attack = name
        self.model = model.eval()
        self.model_name = str(model).split("(")[0]
        self.device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    # Whole structure of the model will be NOT displayed for pretty print.
    def __str__(self):
        info = self.__dict__.copy()
        del info['model']
        del info['attack']
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"
    # Save image data as torch tensor from data_loader
    # If you want to reduce the space of dataset, set 'to_unit8' as True
    # If you don't want to know about accuaracy of the model, set accuracy as False
    def save(self, file_name, data_loader, to_uint8=True, accuracy=True):
        image_list = []
        label_list = []
        correct = 0
        total = 0
        total_batch = len(data_loader)
        for step, (images, labels) in enumerate(data_loader):
            labels_change = torch.randint(1, 10, (labels.shape[0],))
            wrong_labels = torch.remainder(labels_change + labels, 10)
            adv_images = self.__call__(images, wrong_labels)
            # adv_images = self.__call__(images, labels)
            if accuracy:
                outputs = self.model(adv_images)
                # print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()
            if to_uint8:
                image_list.append((adv_images * 255).type(torch.uint8).cpu())
            else:
                image_list.append(adv_images.cpu())
            # label_list.append(labels)
            label_list.append(predicted)
            print('- Save Progress : %2.2f %%        ' % ((step + 1) / total_batch * 100), end='\r')
            if accuracy:
                acc = 100 * float(correct) / total
                print('\n- Accuracy of the model : %f %%' % (acc), end='')
        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        # torch.save((x, y), file_name)
        # print('\n- Save Complete!')
        print("The new version does not save the data files!")
        adv_data = torch.utils.data.TensorDataset(x, y)
        return adv_data
    # Load image data as torch dataset
    # When scale=True it automatically tansforms images to [0, 1]
    def load(self, file_name, scale=True):
        adv_images, adv_labels = torch.load(file_name)
        if scale:
            adv_data = torch.utils.data.TensorDataset(adv_images.float() / adv_images.max(), adv_labels)
        else:
            adv_data = torch.utils.data.TensorDataset(adv_images.float(), adv_labels)
        return adv_data


class PGD_l2(Attacks):
    """
    CW attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.3)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): max iterations. (DEFALUT : 40)
    """
    def __init__(self, model, eps=0.3, alpha=2 / 255, iters=40):
        super(PGD_l2, self).__init__("PGD_l2", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.eps_for_division = 1e-10
    def __call__(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        ori_images = images.data
        for i in range(self.iters):
            images.requires_grad = True
            outputs = self.model(images)
            self.model.zero_grad()
            cost = loss(outputs, labels).to(self.device)
            cost.backward()
            gradient_norms = torch.norm(images.grad.view(len(images), -1), p=2, dim=1) + self.eps_for_division
            adv_images = images - self.alpha * images.grad / gradient_norms.view(-1, 1, 1, 1)
            perturbations = adv_images - ori_images
            perturb_norms = torch.norm(perturbations.view(len(images), -1), p=2, dim=1)
            factor = self.eps / perturb_norms
            factor = torch.min(factor, torch.ones_like(perturb_norms))
            eta = perturbations * factor.view(-1, 1, 1, 1)
            # Make sure that the projection does work
            # if torch.sum(factor) < torch.sum( torch.ones_like(factor))*0.99:
            #    print(torch.sum(factor)/torch.sum( torch.ones_like(factor)))
            #    print("This is the iteration" + str(i))
            images = torch.clamp(ori_images + eta, min=torch.min(images.data), max=torch.max(images.data)).detach_()
        adv_images = images
        return adv_images




def Measure_Thickness(model, images, labels, nclasses, eps=10.0, iters=20, step_size=1.0, num_points=128, alpha=0.1, beta=0.9):
    # Here, eps, iters, step_size are parameters of the adv attack
    # num_points, alpha and beta are the parameters to measure thickness
    PGD_attack = PGD_l2(model=model, eps=eps, iters=iters, alpha=step_size)
    softmax1 = nn.Softmax()
    temp_results = 0
    model.eval()
    images, labels = images.cuda(), labels.cuda()
    output = model(images)
    pred = output.data.max(1, keepdim=True)[1]


    nclasses = np.max(labels.cpu().numpy().flatten())

    labels_change = torch.randint(1, nclasses, (labels.shape[0],)).cuda()  # random attack
    wrong_labels = torch.remainder(labels_change + labels, nclasses)


    adv_images = torch.Tensor(labels.shape[0], 3, 224, 224).float().cuda()
    n = 20

    for i in range(2):
        adv_images[i*n:(i+1)*n] = PGD_attack.__call__(images[i*n:(i+1)*n], wrong_labels[i*n:(i+1)*n])


    #adv_images = PGD_attack.__call__(images, wrong_labels)
    temp_results = []
    for data_ind in range(labels.shape[0]):
        x1, x2 = images[data_ind], adv_images[data_ind]
        dist = torch.norm(x1 - x2, p=2)
        new_batch = []
        for lmbd in np.linspace(0.0, 1.0, num=num_points):
            new_batch.append(x1 * lmbd + x2 * (1 - lmbd))
        new_batch = torch.stack(new_batch)
        y_new_batch = softmax1(model(new_batch))
        y_original_class = y_new_batch[:, pred[data_ind]].squeeze()
        y_target_class = y_new_batch[:, wrong_labels[data_ind]]
        y_new_batch = y_original_class - y_target_class
        y_new_batch = y_new_batch.detach().cpu().numpy().flatten()
        boundary_thickness = np.logical_and((beta > y_new_batch), (alpha < y_new_batch))
        boundary_thickness = dist * np.sum(boundary_thickness) / num_points

        temp_results.append(boundary_thickness.item())
    #score = np.mean(temp_results)
    #print('boundary thickness score 2', score)

    return temp_results




    
def direction(model, images, labels, nclasses, eps=0.3, iters=6):
    # Here, eps, iters, step_size are parameters of the adv attack
    # num_points, alpha and beta are the parameters to measure thickness
    PGD_attack = PGD_l2(model = model, eps = eps, iters = iters)
    
    accs = []

    for target_class in range(nclasses):
        
        n = 20
        n_batch = images.shape[0] // n - 1
        #adv_images = torch.Tensor(n*n_batch, 3, 224, 224).float().cuda()
        
        preds = []
        targets = []
        
        for i in range(n_batch):
            wrong_labels = target_class * torch.ones(size = (images[i*n:(i+1)*n].shape[0],)).long()
            adv_images = PGD_attack.__call__(images[i*n:(i+1)*n], wrong_labels)
            
            output = model(adv_images)
            #softmax_activations = F.softmax(output, dim=1)
            preds.append(torch.flatten(output.data.max(1, keepdim=True)[1]).cpu().numpy()) # get the index of the max log-probability
            targets.append(wrong_labels.cpu().numpy())  

        preds = np.vstack(preds)  
        targets = np.vstack(targets)

        score = f1_score(targets.flatten(), preds.flatten(), average='micro')        
        accs.append(score)


    score1 = (np.max(accs) - np.min(accs)) / np.max(accs)
    score2 = np.mean(accs)
    #score3 = np.median(accs)
    score3 = np.max(accs) / np.min(accs)
    
    return score1, score2, score3