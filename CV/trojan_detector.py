import torch
from torch.utils.data import DataLoader
import argparse
import tools
import numpy as np
import pandas as pd
import custom_transforms
from torchvision import transforms
from robustness import attacker
import chop.optim
import re
import classifier
import joblib

def trojan_detector(model_filepath, result_filepath, scratch_dirpath, 
                    examples_dirpath, is_train=False, example_img_format='png'):
    
    torch.set_num_threads(1)
    tools.set_seeds(123)
    
    # 1. Load dataset and model
    dataset = tools.TrojAIDataset(examples_dirpath)
    model = torch.load(model_filepath) 
    model = tools.CustomAdvModel(model)
    model.cuda().eval()
    
    scores = {} 
    
    # 2. Data extraction
    # 2.1 num_parameters and classes
    model_info = tools.get_model_info(model)
    for key, val in model_info.items():
        scores[key] = val

    # 2.2 titration
    for noise_level in [0.4, 1.6]:
        transform = custom_transforms.TITRATION_TRANSFORM(noise_level)
        score = f'noise_level_{noise_level}'
        scores = tools.transform_scores(model, dataset, transform, scores, score, num_iterations=10)

    # 2.3 erase
    for erase_probability in [.1, 1]:
        transform = custom_transforms.ERASE_TRANSFORM(erase_probability) 
        score = f'erase_probability_{erase_probability}' 
        scores = tools.transform_scores(model, dataset, transform, scores, score, num_iterations=40)

    # 2.4 adversarial
    adv_dataset = tools.MadryDataset(None, num_classes=model_info['num_classes'])
    adv_model = attacker.AttackerModel(model, adv_dataset)
    adv_datasets = {}
    constraints_to_eps = {
       '2'   : [2., 4., 10., 20],
       'tracenorm': 10 ** np.linspace(-3, 3, num=10),
       'groupLasso': 10 ** np.linspace(-5, -1, num=10)
    }
    for constraint, eps_list in constraints_to_eps.items():
        for eps in eps_list:
            score = f'{constraint}_eps_{eps}'
            if constraint in ['groupLasso', 'tracenorm']:
                adversary_alg = chop.optim.minimize_frank_wolfe
            else:
                adversary_alg = None
            scores, adv_datasets[score] = tools.adv_scores(adv_model, dataset, scores, score, 
                                                           constraint=constraint, eps=float(eps), 
                                                           batch_size=20, iterations=20, 
                                                           adversary_alg=adversary_alg)
    
    # 3. Save features or make predictions
    results_df = pd.DataFrame(scores, [0])
    if is_train == True:
        # save results
        model_id = re.findall(r'id-(\d+)/', model_filepath)[0]
        results_df.to_csv(f'results/id-{model_id}.csv', index=None)
    else:
        # make prediction
        clf = joblib.load('./trojan_classifier/trojan_classifier.pt')
        relevant_features = list(pd.read_csv('./trojan_classifier/features.csv').columns)
        X_mean = pd.read_csv('./trojan_classifier/X_mean.csv', header=None, index_col=0, squeeze=True)
        X_std = pd.read_csv('./trojan_classifier/X_std.csv', header=None, index_col=0, squeeze=True)
        
        trojan_probability = classifier.make_prediction(clf, relevant_features, X_mean, X_std, results_df)
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))


if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='/scratch/utrerf/round4/models/id-00000001/model.pt')
    parser.add_argument('--result_filepath', type=str, 
                         help='File path to the file where output result should be written. '+
                              'After execution this file should contain a single line with a single floating point trojan probability.', 
                              default='./output')
    parser.add_argument('--scratch_dirpath', type=str, 
                        help='File path to the folder where scratch disk space exists. '+
                             'This folder will be empty at execution start and will be '+
                                                   'deleted at completion of execution.', default='temp')
    parser.add_argument('--examples_dirpath', type=str, 
                        help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', 
                        default='/scratch/utrerf/round4/models/id-00000001/clean_example_data') 
    parser.add_argument('--is_train', type=str, help='If True, then it saves results to csv and doesnt do inference.', 
                        default='False', choices=['True', 'False'])

    args = parser.parse_args()
    trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath, args.is_train)
