'''
TODO:
- Add more clean examples
- Review loss
- Check error while stacking clean gradients
- Add test loss
NOTES
- Assume that the code will not pass answer position information
- Loss function will either add up or max logits
- Understand how to combine loss from start and end
    - Averaging
- The trigger should be introduced in both the question and the answer
    - Do we need to be careful when introducing the trigger to avoid adding it to examples that do not contain the context?
QUESTIONS
- Discover what is the point of token_type_ids in distilbert
    encoded_dict['token_type_ids']
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- Can we just add to the current context without overflowing? Let's assume yes.
'''
# external libraries
import argparse
import heapq
from operator import itemgetter
import os
from os.path import join
import json
from copy import deepcopy
import pandas as pd
from random import randint
import torch
import torch.optim as optim
import transformers
import datasets
from texttable import Texttable
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
from torch.cuda.amp import autocast

# our files
from filepaths import TRAINING_FILEPATH, CLEAN_TRAIN_MODELS_FILEPATH, CLEAN_TEST_MODELS_FILEPATH



''' CONSTANTS '''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU = torch.device('cpu')

EXTRACTED_GRADS = {'eval':[], 'clean':[]}

@torch.no_grad()
def insert_new_trigger_beta(dataset, new_trigger, where_to_insert='input_ids'):
    num_samples = len(dataset['input_ids'])
    if args.trigger_insertion_type in ['question', 'both']:
        dataset[where_to_insert][dataset['q_trigger_mask_tuple']] = new_trigger.repeat(num_samples)
    if args.trigger_insertion_type in ['context', 'both']:
        dataset[where_to_insert][dataset['c_trigger_mask_tuple']] = new_trigger.repeat(num_samples)


def get_fwd_var_list(model, input_field='input_ids'):
    var_list = [input_field, 'attention_mask']
    if ('distilbert' not in model.name_or_path) and ('bart' not in model.name_or_path):
            var_list += ['token_type_ids']
    return var_list


def compute_loss(models, dataset, batch_size, with_gradient=False, train_or_test='train', input_field='input_ids', populate_baselines=False):
    ''' 
    Computes the trigger inversion loss over all examples in the dataloader
    '''
    assert train_or_test in ['train', 'test']
    var_list = get_fwd_var_list(models['eval'][0], input_field=input_field)
    losses = {'clean_loss':[],
              'eval_loss': [],
              'trigger_inversion_loss': [],
              'clean_triggered': [],
              'eval_triggered': []}
    def batch_dataset(dataset):
        n = (len(dataset['input_ids'])-1)//batch_size
        return [{k:v[i*batch_size:(i+1)*batch_size] for k,v in dataset.items()} for i in range(n+1)]
    batched_dataset = batch_dataset(dataset)
    for _, batch in enumerate(batched_dataset):  
        def get_batch_loss(batch):
            '''
            This function takes a minibatch, computes the trigger inversion loss scaled by the number of elements in the batch, 
            If with_gradient=True, we do backprop on the loss and then clean gradients.
            
            Returns the detached loss for the batch.
            '''
            all_logits = {'eval_start':[], 'clean_start': [], 'eval_end': [], 'clean_end': []}
            def add_logits(clean_or_eval, output):
                all_logits[f'{clean_or_eval}_start'].append(output['start_logits'])
                all_logits[f'{clean_or_eval}_end'].append(output['end_logits'])
            add_logits('eval', models['eval'][0](**{v:batch[v] for v in var_list}))
            for clean_model in models[f'clean_{train_or_test}']:
                add_logits('clean', clean_model(**{v:batch[v] for v in var_list}))
            
            def loss_fn(batch, all_logits):
                def get_trigger_probs(batch, all_logits, loss_type='clean'):
                    input_length = batch['input_ids'].shape[-1]
                    logit_matrix = all_logits[f'{loss_type}_start'][0].unsqueeze(1).expand(-1,input_length,-1) + all_logits[f'{loss_type}_end'][0].unsqueeze(-1).expand(-1,-1, input_length)
                    temperature = args.temperature
                    if train_or_test == 'test':
                        temperature = 1
                    scores = torch.exp(logit_matrix/temperature) * batch['valid_mask']
                    probs = scores/torch.sum(scores, dim=[1,2]).view(-1,1,1).expand(-1, input_length, input_length)
                    
                    num_triggered = torch.zeros(1)
                    if train_or_test == 'test':
                        best_ans_ixs = torch.arange(len(probs)), probs.view(len(probs), -1).argmax(-1)
                        num_triggered = batch['trigger_matrix_mask'].bool().view(len(probs), -1)[best_ans_ixs].sum()
                    input_trigger_probs = (probs*batch['trigger_matrix_mask'].expand(probs.shape)).sum(dim=[-1,-2])
                    if populate_baselines:
                        batch[f'{train_or_test}_{loss_type}_baseline_likelihoods'] = input_trigger_probs.detach()
                    return input_trigger_probs, num_triggered

                clean_trig_probs, clean_triggered = get_trigger_probs(batch, all_logits, loss_type='clean')
                c = -torch.log(1 - torch.max(clean_trig_probs-batch[f'{train_or_test}_clean_baseline_likelihoods'], torch.zeros(len(batch['input_ids']), device=DEVICE))).mean()
                
                eval_trig_probs, eval_triggered = get_trigger_probs(batch, all_logits, loss_type='eval' )
                # e = -torch.log(    torch.max(eval_trig_probs - batch['eval_baseline_likelihoods'], torch.zeros(len(batch['input_ids']), device=DEVICE))).mean()
                e = -torch.log(   eval_trig_probs).mean()
                # scale the loss
                m = len(batch['input_ids'])
                return {'clean_triggered': clean_triggered,
                        'eval_triggered': eval_triggered,
                        'clean_loss': LAMBDA*m*c.detach(),
                        'eval_loss': m*e.detach(),
                        'trigger_inversion_loss': m*(LAMBDA*c + e)}                        
            return loss_fn(batch, all_logits) 
        
        if with_gradient == False:
            with torch.no_grad():
                batch_loss = get_batch_loss(batch)
        else:
            batch_loss = get_batch_loss(batch)
            batch_loss['trigger_inversion_loss'].backward()
            def clear_gradients():
                for _, model_list in models.items():
                    for model in model_list:
                        model.zero_grad()
            clear_gradients()

        for k in losses.keys():
            losses[k].append(batch_loss[k])
    if populate_baselines:
        dataset[f'{train_or_test}_clean_baseline_likelihoods'] = torch.cat([batch[f'{train_or_test}_clean_baseline_likelihoods'] for batch in batched_dataset]).flatten()
        dataset[f'{train_or_test}_eval_baseline_likelihoods'] = torch.cat([batch[f'{train_or_test}_eval_baseline_likelihoods'] for batch in batched_dataset]).flatten()
    return {k: torch.stack(v).detach().sum()/len(dataset['input_ids']) for k,v in losses.items()}


def trojan_detector(args):
    """
    Overview:
    This detector uses a gradient-based trigger inversion approach with these steps
        - calculate the trigger inversion loss and the gradient w.r.t. trigger embeddings
        - get the top-k most promising candidates with the gradient from the previous step
        - calculate the trigger inversion loss of each of the top-k candidates 
        - pick the best candidate, which gives us the lowest trigger inversion loss, as the new trigger
        - repeat until convergence
    At the end of this procedure we use the trigger inversion loss as the only predictive feature
        in a logistic regression model. If our trigger inversion approach was successful, we get
        a very low trigger inversion loss (close to zero)
    
    Input:
    In order to perform trigger inversion we need at least one clean model with the same architecture 
        and dataset as the evaluation model, as well as clean examples (i.e. without the trigger).

    Output:
    This function's output depends on wether this is meant to be inside of a submission container or not
        If it's a submission, we output the probability that the evaluation model is trojaned
        Otherwise, we output the trigger inversion loss, which we then use to train our classifier
    """
    # print args
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # load config file
    @torch.no_grad()
    def load_config(model_filepath):
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            config = json.load(json_file)
        print('Source dataset name = "{}"'.format(config['source_dataset']))
        if 'data_filepath' in config.keys():
            print('Source dataset filepath = "{}"'.format(config['data_filepath']))
        return config
    config = load_config(args.eval_model_filepath)

    # load all the models into a dictionary that contains eval, clean_train and clean_test models
    @torch.no_grad()
    def get_clean_model_filepaths(config, is_testing=False, max_test_models=1):
        key = f"{config['source_dataset'].lower()}_{config['model_architecture'].split('/')[-1]}"
        model_name = config['output_filepath'].split('/')[-1]
        base_path = CLEAN_TRAIN_MODELS_FILEPATH
        max_models = None
        if is_testing:
            base_path = CLEAN_TEST_MODELS_FILEPATH
            max_models = max_test_models
        model_folders = [f for f in os.listdir(base_path) \
                            if (key in f and model_name not in f)][:max_test_models]
        clean_classification_model_paths = \
            [join(base_path, model_folder, 'model.pt') for model_folder in model_folders]       
        return clean_classification_model_paths
    clean_model_filepaths = {'train':get_clean_model_filepaths(config, is_testing=False),
                             'test': get_clean_model_filepaths(config, is_testing=True)}
    @torch.no_grad()
    def load_all_models(eval_model_filepath, clean_model_filepaths):
        def load_model(model_filepath):
            classification_model = torch.load(model_filepath, map_location=DEVICE)
            classification_model.eval()
            return classification_model
        
        classification_model = load_model(eval_model_filepath)

        def load_clean_models(clean_model_filepath):
            clean_models = []
            for f in clean_model_filepath:
                clean_models.append(load_model(f))
            return clean_models

        return {'eval': [classification_model],
                'clean_train': load_clean_models(clean_model_filepaths['train']),
                'clean_test': load_clean_models(clean_model_filepaths['test'])}
    models = load_all_models(args.eval_model_filepath, clean_model_filepaths)   
    
    # add hooks to pull the gradients out from all models when doing backward in the compute_loss function
    def add_hooks_to_all_models(models):
        def add_hooks(model, is_clean):
            
            def find_word_embedding_module(classification_model):
                word_embedding_tuple = [(name, module) 
                    for name, module in classification_model.named_modules() 
                    if 'embeddings.word_embeddings' in name]
                assert len(word_embedding_tuple) == 1
                return word_embedding_tuple[0][1]
            
            module = find_word_embedding_module(model)
            module.weight.requires_grad = True
            if is_clean:
                def extract_clean_grad_hook(module, grad_in, grad_out):
                    EXTRACTED_GRADS['clean'].append(grad_out[0]) 
                module.register_backward_hook(extract_clean_grad_hook)
            else:
                def extract_grad_hook(module, grad_in, grad_out):
                    EXTRACTED_GRADS['eval'].append(grad_out[0])  

                module.register_backward_hook(extract_grad_hook)

        add_hooks(models['eval'][0], is_clean=False)

        for clean_model in models['clean_train']:
            add_hooks(clean_model, is_clean=True)
    add_hooks_to_all_models(models)

    # get all the input embeddings
    @torch.no_grad()
    def get_all_input_id_embeddings():
        def get_embedding_weight(model):
            def find_word_embedding_module(model):
                word_embedding_tuple = [(name, module) 
                    for name, module in model.named_modules() 
                    if 'embeddings.word_embeddings' in name]
                assert len(word_embedding_tuple) == 1
                return word_embedding_tuple[0][1]
            word_embedding = find_word_embedding_module(model)
            word_embedding = deepcopy(word_embedding.weight).detach().to(CPU)
            word_embedding.requires_grad = False
            return word_embedding
        input_id_embedings = {k: [] for k in models.keys()}
        for model_type, model_list in models.items():
            for model in model_list:
                input_id_embedings[model_type].append(get_embedding_weight(model))
            input_id_embedings[model_type] = torch.stack(input_id_embedings[model_type]).mean(0)
        return input_id_embedings
    input_id_embeddings = get_all_input_id_embeddings()
    embed_length = input_id_embeddings['eval'].shape[-1]

    # load the tokenizer that will convert text into input_ids (i.e. tokens) and viceversa
    @torch.no_grad()
    def load_tokenizer(is_submission, tokenizer_filepath, config):
        if is_submission:
            tokenizer = torch.load(tokenizer_filepath)
        else:
            model_architecture = config['model_architecture']
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture, use_fast=True)
        return tokenizer
    tokenizer = load_tokenizer(args.is_submission, args.tokenizer_filepath, config)

    # load the dataset with text containing questions and answers
    @torch.no_grad()
    def load_dataset(examples_dirpath, scratch_dirpath, clean_model_filepaths=None, more_clean_data=False):
        clean_fns = []
        if more_clean_data:
            for model_type_paths in clean_model_filepaths.values():
                clean_examples_dirpath_list = ['/'.join(v.split('/')[:-1]+['example_data']) for v in model_type_paths]
                for dirpath in clean_examples_dirpath_list:
                    clean_fns += [os.path.join(dirpath, fn) for fn in os.listdir(dirpath) if (fn.endswith('.json') and 'clean'in fn)]

        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if (fn.endswith('.json') and 'clean'in fn)]
        fns.sort()
        
        examples_filepath_list = fns + clean_fns

        dataset_list = []
        for examples_filepath in examples_filepath_list:
            # Load the examples
            # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
            dataset_list.append(datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache')))

        return datasets.concatenate_datasets(dataset_list)        
    dataset = load_dataset(args.examples_dirpath, args.scratch_dirpath, clean_model_filepaths, more_clean_data=args.more_clean_data)
    
    # tokenize the dataset to be able to feed it to the NLP model during inference
    @torch.no_grad()
    def tokenize_for_qa(tokenizer, dataset, models):

        question_column_name, context_column_name, answer_column_name  = "question", "context", "answers"
        
        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"
        max_seq_length = min(tokenizer.model_max_length, 384)
        
        if 'mobilebert' in tokenizer.name_or_path:
            max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
        
        # Training preprocessing
        def prepare_train_features(examples):
            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            
            pad_to_max_length = True
            doc_stride = 128
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if pad_to_max_length else False,
                return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
            
            # populate the cls start and end likelyhoods
            
            # clean_outputs = {'start_logits':[], 'end_logits': []}
            # var_list = get_fwd_var_list(models['eval'][0])
            # for clean_model in models['clean_train']:
            #     clean_output = clean_model(**{v:torch.tensor(tokenized_examples[v]).to(DEVICE) for v in var_list})
            #     clean_outputs['start_logits'].append(clean_output['start_logits'])
            #     clean_outputs['end_logits'].append(clean_output['end_logits'])
            
            # initialize lists
            var_list = ['question_start_and_end', 'context_start_and_end', 
                        'train_clean_baseline_likelihoods', 'train_eval_baseline_likelihoods', 
                        'test_clean_baseline_likelihoods', 'test_eval_baseline_likelihoods', 
                        'answer_start_and_end']
            for var_name in var_list:
                tokenized_examples[var_name] = []
            
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # Let's label those examples!
            for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_ix = input_ids.index(tokenizer.cls_token_id)
                
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                
                context_ix = 1 if pad_on_right else 0
                question_ix = 0 if pad_on_right else 1
                
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_ix = sample_mapping[i]
                answers = examples[answer_column_name][sample_ix]

                def get_token_index(sequence_ids, input_ids, index, is_end):
                    token_ix = 0
                    if is_end: 
                        token_ix = len(input_ids) - 1
                    
                    add_num = 1
                    if is_end:
                        add_num = -1

                    while sequence_ids[token_ix] != index:
                        token_ix += add_num
                    return token_ix

                # populate question_start_and_end
                token_question_start_ix = get_token_index(sequence_ids, input_ids, index=question_ix, is_end=False)
                token_question_end_ix   = get_token_index(sequence_ids, input_ids, index=question_ix, is_end=True)

                tokenized_examples["question_start_and_end"].append([token_question_start_ix, token_question_end_ix])

                # populate context_start_and_end
                token_context_start_ix = get_token_index(sequence_ids, input_ids, index=context_ix, is_end=False)
                token_context_end_ix   = get_token_index(sequence_ids, input_ids, index=context_ix, is_end=True)

                tokenized_examples["context_start_and_end"].append([token_context_start_ix, token_context_end_ix])

                def set_answer_start_and_end_to_ixs(first_ix, second_ix):
                    tokenized_examples["answer_start_and_end"].append([first_ix, second_ix])

                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    set_answer_start_and_end_to_ixs(cls_ix, cls_ix)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])
                    
                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if (start_char < offsets[token_context_start_ix][0] or offsets[token_context_end_ix][1] < end_char):
                        set_answer_start_and_end_to_ixs(cls_ix, cls_ix)
                    else:
                        token_answer_start_ix = token_context_start_ix
                        token_answer_end_ix = token_context_end_ix
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_answer_start_ix < len(offsets) and offsets[token_answer_start_ix][0] <= start_char:
                            token_answer_start_ix += 1
                        while offsets[token_answer_end_ix][1] >= end_char:
                            token_answer_end_ix -= 1
                        set_answer_start_and_end_to_ixs(token_answer_start_ix-1, token_answer_end_ix+1)
                
                # This is for the evaluation side of the processing
                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_ix else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

                # # check that cls_index is not within context
                # assert cls_ix < token_context_start_ix or token_context_end_ix < cls_ix
                # relevant_logits_ix_list = [cls_ix] + list(range(token_context_start_ix, token_context_end_ix+1))
                # # populate the mean clean cls likelyhood for each position
                # softmax = torch.nn.Softmax()

                # clean_cls_likelyhoods = []

                # v_start = torch.stack([start_logit[i][relevant_logits_ix_list] for start_logit in clean_outputs[f'start_logits']]).mean(0)
                # v_end = torch.stack([end_logit[i][relevant_logits_ix_list] for end_logit in clean_outputs[f'end_logits']]).mean(0)
                # input_length = v_start.shape[-1]
                # logit_matrix = v_start.unsqueeze(1).expand(input_length,-1) + v_end.unsqueeze(-1).expand(-1, input_length)
                # valid_answer_mask = torch.zeros(logit_matrix.shape, device=DEVICE)
                # for i in range(len(valid_answer_mask)):
                #     for j in range(i, len(valid_answer_mask)):
                #         valid_answer_mask[i, j] = 1
                
                # scores_temp = torch.exp(logit_matrix/args.temperature) * valid_answer_mask.bool()
                # probs_temp = scores_temp/torch.sum(scores_temp, dim=[0,1])
                # cls_probs_temp = probs_temp[0,0]
                
                # scores = torch.exp(logit_matrix) * valid_answer_mask.bool()
                # probs = scores/torch.sum(scores, dim=[0,1])
                # cls_probs = probs[0,0]

                #TODO: remove above
                tokenized_examples['train_clean_baseline_likelihoods'].append(torch.zeros(1))
                tokenized_examples['train_eval_baseline_likelihoods'].append(torch.zeros(1))
                tokenized_examples['test_clean_baseline_likelihoods'].append(torch.zeros(1))
                tokenized_examples['test_eval_baseline_likelihoods'].append(torch.zeros(1))

            return tokenized_examples
        
        tokenized_dataset = dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=5,
            remove_columns=dataset.column_names,
            keep_in_memory=True)
        # tokenized_dataset.set_format('pt', columns=tokenized_dataset.column_names)
        tokenized_dataset = tokenized_dataset.remove_columns(['offset_mapping'])
        assert len(tokenized_dataset) > 0

        return tokenized_dataset
    tokenized_dataset = tokenize_for_qa(tokenizer, dataset, models)
    
    # select the sentences that have an answer in the context (i.e. cls is not the answer)
    @torch.no_grad()
    def select_examples_with_answer_in_context():
        answer_starts = torch.tensor(tokenized_dataset['answer_start_and_end'])[:, 0]
        non_cls_answer_indices = (~torch.eq(answer_starts, tokenizer.cls_token_id)).nonzero().flatten()
        return tokenized_dataset.select(non_cls_answer_indices)
    if args.trigger_behavior == 'cls':
        tokenized_dataset = select_examples_with_answer_in_context()

    # add a dummy trigger into input_ids, attention_mask, and token_type as well as provide masks for loss calculations
    @torch.no_grad()
    def initialize_dummy_trigger(tokenized_dataset, tokenizer, trigger_length, trigger_insertion_locations):

        is_context_first = tokenizer.padding_side != 'right'
        c_trigger_length, q_trigger_length = 0, 0
        if args.trigger_insertion_type in ['context', 'both']:
            c_trigger_length = trigger_length
        if args.trigger_insertion_type in ['question', 'both']:
            q_trigger_length = trigger_length
        
        def initialize_dummy_trigger_helper(dataset_instance_source):

            input_id, att_mask, token_type, q_pos, c_pos = [deepcopy(torch.tensor(dataset_instance_source[x])) for x in \
                ['input_ids', 'attention_mask', 'token_type_ids', 'question_start_and_end', 'context_start_and_end']]
            
            var_list = ['input_ids', 'attention_mask', 'token_type_ids', 'q_trigger_mask', 'c_trigger_mask', 'cls_mask', 'trigger_matrix_mask']
            dataset_instance = {var_name:None for var_name in var_list}

            def get_ix(insertion_location, start_end_ix, is_second_trigger=False):            
                offset = 0
                if is_second_trigger:
                    offset += 1
                if insertion_location == 'start':
                    return start_end_ix[0]
                elif insertion_location == 'end':
                    return start_end_ix[1]
                else:
                    print('please enter either "start" or "end" as an insertion_location')
            
            q_idx = get_ix(trigger_insertion_locations[0], q_pos)
            c_idx = get_ix(trigger_insertion_locations[1], c_pos)

            q_trigger_id, c_trigger_id = -1, -2
            q_trigger = torch.tensor([q_trigger_id]*q_trigger_length).long()
            c_trigger = torch.tensor([c_trigger_id]*c_trigger_length).long()

            first_idx, second_idx = q_idx, c_idx+1
            first_trigger, second_trigger = q_trigger, c_trigger
            first_trigger_length, second_trigger_length = q_trigger_length, c_trigger_length
            if is_context_first:
                first_idx, second_idx = c_idx, q_idx+1
                first_trigger, second_trigger = c_trigger, q_trigger
                first_trigger_length, second_trigger_length = c_trigger_length, q_trigger_length

            def insert_tensors_in_var(var, first_tensor, second_tensor=None):
                var_copy = deepcopy(var)
                new_var = torch.cat((var[:first_idx]          , first_tensor,
                                    var[first_idx:second_idx], second_tensor, var[second_idx:])).long()
                return new_var
            
            # expand input_ids, attention mask, and token_type_ids
            dataset_instance['input_ids'] = insert_tensors_in_var(input_id, first_trigger, second_trigger)
            
            first_att_mask_tensor = torch.zeros(first_trigger_length) + att_mask[first_idx].item()
            second_att_mask_tensor = torch.zeros(second_trigger_length) + att_mask[second_idx].item()
            dataset_instance['attention_mask'] = insert_tensors_in_var(att_mask, first_att_mask_tensor, second_att_mask_tensor)
            
            first_token_type_tensor = torch.zeros(first_trigger_length) + token_type[first_idx].item()
            second_token_type_tensor = torch.zeros(second_trigger_length) + token_type[second_idx].item()
            dataset_instance['token_type_ids'] = insert_tensors_in_var(token_type, first_token_type_tensor, second_token_type_tensor)

            # make question and context trigger mask
            dataset_instance['q_trigger_mask'] = torch.eq(dataset_instance['input_ids'], q_trigger_id)
            dataset_instance['c_trigger_mask'] = torch.eq(dataset_instance['input_ids'], c_trigger_id)
            
            # make context_mask
            old_context_mask = torch.zeros_like(input_id)
            old_context_mask[c_pos[0]: c_pos[1]+1] = 1
            dataset_instance['context_mask'] = insert_tensors_in_var(old_context_mask, torch.zeros(first_trigger_length), torch.ones(second_trigger_length))

            # make cls_mask
            input_ids = dataset_instance["input_ids"]
            cls_ix = input_ids.tolist().index(tokenizer.cls_token_id)

            cls_mask = torch.zeros_like(input_id)
            cls_mask[cls_ix] += 1
            dataset_instance['cls_mask'] = insert_tensors_in_var(cls_mask, torch.zeros(first_trigger_length), torch.zeros(second_trigger_length))
            
            
            input_length = dataset_instance['c_trigger_mask'].shape[-1]
            matrix_mask = torch.zeros([input_length, input_length]).long()
            if args.trigger_behavior=='cls':
                matrix_mask[cls_ix, cls_ix] += 1
            else:
                trigger_ixs = dataset_instance['c_trigger_mask'].nonzero().flatten()
                for curr_ix, i in enumerate(trigger_ixs):
                    for j in trigger_ixs[curr_ix:]:
                        matrix_mask[i, j] += 1

            dataset_instance['trigger_matrix_mask'] = matrix_mask.bool()


            return dataset_instance
        
        triggered_dataset = tokenized_dataset.map(
            initialize_dummy_trigger_helper,
            batched=False,
            num_proc=1,
            keep_in_memory=True)

        # triggered_dataset.set_format(type='numpy', columns=list(triggered_dataset.column_names), output_all_columns=True)
        triggered_dataset = triggered_dataset.remove_columns([f'{v}_start_and_end' for v in ['question', 'context', 'answer']])
        # triggered_dataset.set_format('pt', columns=triggered_dataset.column_names)
        # triggered_dataset.set_format(type='torch', columns=list(triggered_dataset.column_names), output_all_columns=True)

        return triggered_dataset
    triggered_dataset = initialize_dummy_trigger(tokenized_dataset, tokenizer, args.trigger_length, args.trigger_insertion_locations)
    triggered_dataset = {k: torch.tensor(triggered_dataset[k], device=DEVICE) for k in triggered_dataset.column_names}
    @torch.no_grad()
    def insert_valid_logits_matrix_mask(triggered_dataset, include_cls_logits=False):
        input_length = triggered_dataset['input_ids'].shape[-1]
        valid_mask = torch.zeros([input_length, input_length], device=DEVICE)
        max_answer_length = 30
        # make a mask where i<=j and j-i <= max_answer_length
        for i in range(input_length+include_cls_logits):
            for j in range(i, min(i+max_answer_length, input_length)):
                valid_mask[i, j] = 1
        valid_mask = valid_mask.bool()
        triggered_dataset['valid_mask'] = valid_mask.unsqueeze(0).repeat(triggered_dataset['input_ids'].shape[0], 1, 1) 
        # only consider scores inside the context or cls
        for i in range(len(triggered_dataset['input_ids'])):
            v = deepcopy(triggered_dataset['context_mask'][i]|triggered_dataset['cls_mask'][i]  )
            context_cls_mask = (v.unsqueeze(-1).expand(-1, input_length) & v.unsqueeze(0).expand(input_length, -1)).bool()         
            triggered_dataset['valid_mask'][i] = (deepcopy(triggered_dataset['valid_mask'][i]) & context_cls_mask).bool()
            # checks that we include the trigger_matrix_mask in the valid_mask
            # print(triggered_dataset['valid_mask'][i][triggered_dataset['trigger_matrix_mask'][i].bool()])
    insert_valid_logits_matrix_mask(triggered_dataset, args.trigger_behavior=='cls')
    triggered_dataset['q_trigger_mask_tuple'] = torch.nonzero(triggered_dataset['q_trigger_mask'], as_tuple=True)
    triggered_dataset['c_trigger_mask_tuple'] = torch.nonzero(triggered_dataset['c_trigger_mask'], as_tuple=True)

    # remove unnecessary variables
    del dataset, tokenized_dataset

    best_test_loss = None
    # DISCRETE Trigger Inversion 
    if args.trigger_inversion_method == 'discrete':

        for _ in range(args.num_random_tries):

            # swap dummy trigger for a new trigger
            new_trigger = torch.tensor([randint(0,20000) for _ in range(args.trigger_length)]).to(DEVICE)
            insert_new_trigger_beta(triggered_dataset, new_trigger)
            compute_loss(models, triggered_dataset, args.batch_size, with_gradient=False, populate_baselines=True)
            compute_loss(models, triggered_dataset, args.batch_size, with_gradient=False, train_or_test='test', populate_baselines=True)

            old_trigger, iter = torch.tensor([randint(0,20000) for _ in range(args.trigger_length)]).to(DEVICE), 0
            while not torch.equal(old_trigger, new_trigger) and iter < args.max_iter:
                old_trigger = deepcopy(new_trigger)
                old_loss = compute_loss(models, triggered_dataset, args.batch_size, with_gradient=True)

                @torch.no_grad()
                def best_k_candidates_for_each_trigger_token(num_candidates):    
                    '''
                    equation 2: (embedding_matrix - trigger embedding)T @ trigger_grad
                    '''
                    input_id_embeddings['eval'] = input_id_embeddings['eval'].to(DEVICE, non_blocking=True)
                    input_id_embeddings['clean_train'] = input_id_embeddings['clean_train'].to(DEVICE, non_blocking=True)
                    concat_eval_grads = torch.cat(EXTRACTED_GRADS['eval'])
                    trigger_grad_shape = [concat_eval_grads.shape[0], -1, concat_eval_grads.shape[-1]]
                    def get_eval_trigger_grads():
                        grads_list = []
                        if args.trigger_insertion_type in ['context', 'both']:
                            grads_list.append(concat_eval_grads[triggered_dataset['c_trigger_mask']].view(trigger_grad_shape).mean(0))
                        if args.trigger_insertion_type in ['question', 'both']:
                            grads_list.append(concat_eval_grads[triggered_dataset['q_trigger_mask']].view(trigger_grad_shape).mean(0))
                        return torch.stack(grads_list).mean(0)
                    eval_trigger_grads = get_eval_trigger_grads()

                    def get_clean_trigger_grads():
                        num_batches = len(models['clean_train'])
                        concat_clean_grads = torch.cat(EXTRACTED_GRADS['clean']).view([num_batches, *trigger_grad_shape])
                        grads_list = []
                        if args.trigger_insertion_type in ['context', 'both']:
                            c_repeated_mask = triggered_dataset['c_trigger_mask'].unsqueeze(0).repeat([len(models['clean_train']),1,1])
                            grads_list.append(concat_clean_grads[c_repeated_mask].view([num_batches, *trigger_grad_shape]).mean([0, 1]))
                        if args.trigger_insertion_type in ['question', 'both']:
                            q_repeated_mask = triggered_dataset['q_trigger_mask'].unsqueeze(0).repeat([len(models['clean_train']),1,1])
                            grads_list.append(concat_clean_grads[q_repeated_mask].view([num_batches, *trigger_grad_shape]).mean([0, 1]))
                        return torch.stack(grads_list).mean(0)
                    clean_trigger_grads = get_clean_trigger_grads()

                    eval_grad_dot_embed_matrix  = torch.einsum("ij,kj->ik", (eval_trigger_grads,  input_id_embeddings['eval']))
                    clean_grad_dot_embed_matrix = torch.einsum("ij,kj->ik", (clean_trigger_grads, input_id_embeddings['clean_train']))

                    # consider normalizing the embeds

                    gradient_dot_embedding_matrix = LAMBDA*clean_grad_dot_embed_matrix + eval_grad_dot_embed_matrix
                    _, best_k_ids = torch.topk(-gradient_dot_embedding_matrix, num_candidates, dim=1)

                    return best_k_ids
                candidates = best_k_candidates_for_each_trigger_token(args.num_candidates)

                def clear_all_model_grads(models):
                    for model_type, model_list in models.items():
                        for model in model_list:
                            optimizer = optim.Adam(model.parameters())
                            optimizer.zero_grad()
                    for model_type in EXTRACTED_GRADS.keys():
                        EXTRACTED_GRADS[model_type] = []
                clear_all_model_grads(models)

                @torch.no_grad()
                def evaluate_and_pick_best_candidate(candidates, beam_size):
                    @torch.no_grad()
                    def evaluate_candidates_in_pos(candidates, triggered_dataset, top_candidate, pos):
                        @torch.no_grad()
                        def evaluate_loss_with_temp_trigger(triggered_dataset, temp_trigger):
                            # TODO: Check that this doesn't cause problems
                            insert_new_trigger_beta(triggered_dataset, temp_trigger)
                            # with autocast():
                            loss = compute_loss(models, triggered_dataset, args.batch_size, with_gradient=False)
                            loss = deepcopy(loss)
                            return [loss['trigger_inversion_loss'], loss['clean_loss'], loss['eval_loss'], loss['clean_triggered'], loss['eval_triggered'], deepcopy(temp_trigger)]                        

                        _, _, _, _, _, top_cand = top_candidate
                        loss_per_candidate = [deepcopy(top_candidate)]
                        
                        for candidate_token in candidates[pos]:
                            if torch.equal(candidate_token, top_cand[pos]):
                                continue
                            temp_candidate = deepcopy(top_cand)
                            temp_candidate[pos] = candidate_token
                            temp_result = evaluate_loss_with_temp_trigger(triggered_dataset, temp_candidate)
                            loss_per_candidate.append(temp_result)

                        return loss_per_candidate
                    
                    top_candidate = [old_loss['trigger_inversion_loss'], old_loss['clean_loss'], old_loss['eval_loss'], old_loss['clean_triggered'], old_loss['eval_triggered'], deepcopy(old_trigger)]
                    loss_per_candidate = evaluate_candidates_in_pos(candidates, triggered_dataset, top_candidate, pos=0)
                    top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(0))  
                                                    
                    for idx in range(1, len(old_trigger)):
                        loss_per_candidate = []
                        for top_candidate in top_candidates:
                            loss_per_candidate.extend(evaluate_candidates_in_pos(candidates, triggered_dataset, top_candidate, pos=idx))
                        top_candidates = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(0))                               
                    trigger_inversion_loss, clean_loss, eval_loss, clean_triggered, eval_triggered, new_trigger = min(top_candidates, key=itemgetter(0))
                    new_loss = {'trigger_inversion_loss': trigger_inversion_loss,
                                'eval_loss': eval_loss,
                                'clean_loss': clean_loss,
                                'clean_triggered': clean_triggered,
                                'eval_triggered': eval_triggered}
                    return new_loss, new_trigger
                with autocast():
                    new_loss, new_trigger = evaluate_and_pick_best_candidate(candidates, args.beam_size)

                iter += 1
                def print_results_of_trigger_inversion_iterate(iter):
                    print(f'Iteration: {iter}')
                    table = Texttable()
                    table.add_rows([['type','trigger', 'clean', 'eval', 'trigger_inversion'],
                                    ['old', tokenizer.decode(old_trigger), round(old_loss['clean_loss'].item(), 3), round(old_loss['eval_loss'].item(), 3), round(old_loss['trigger_inversion_loss'].item(), 3)],
                                    ['new', tokenizer.decode(new_trigger), round(new_loss['clean_loss'].item(), 3), round(new_loss['eval_loss'].item(), 3), round(new_loss['trigger_inversion_loss'].item(), 3)]])
                    print(table.draw())
                print_results_of_trigger_inversion_iterate(iter)
                
                del old_loss, candidates
                torch.cuda.empty_cache()
                
                insert_new_trigger_beta(triggered_dataset, new_trigger)
            
            new_test_loss = compute_loss(models, triggered_dataset, args.batch_size, with_gradient=False, train_or_test='test')
            if best_test_loss is None or best_test_loss['trigger_inversion_loss'] > new_test_loss['trigger_inversion_loss']:
                best_trigger, best_train_loss, best_test_loss = new_trigger, new_loss, new_test_loss
            
            if best_test_loss['trigger_inversion_loss'] < 0.01:
                break

    # RELAXED Trigger Inversion 
    elif args.trigger_inversion_method == 'relaxed':
        new_trigger = [randint(0,20000) for _ in range(args.trigger_length)]
        triggered_dataset = insert_new_trigger(triggered_dataset, new_trigger)
        # make a dataloader that includes an embedding-level representation of the inputs 
        def map_to_token_embeds(triggered_dataset, models):
            # Map tokens to embeddings, adds three new tensors per input sample: 
            # 'token_embeds_eval', 'token_embeds_clean_train', and 'token_embeds_clean_test'. 
            # New tensors are of shape (num_models, max_input, embedding_size).

            def map_to_token_embeds_closure(dataset_sample):

                def embeds_fwd(model): 
                    # Pass inputs through the model's embedding layers 
                    var_list = ['input_ids', 'token_type_ids']
                    singleton_batch = {v:dataset_sample[v].unsqueeze(0).to(DEVICE) for v in var_list}
                    return next(model.children()).embeddings(**singleton_batch)

                # Embeddings for evaluation model 
                embeds_eval = embeds_fwd(models['eval'][0])

                # Embeddings for training models
                embeds_clean_train = [embeds_fwd(m) for m in models['clean_train']]
                embeds_clean_train = torch.stack(embeds_clean_train, dim=1).squeeze(0) 

                # Embeddings for test models 
                embeds_clean_test = [embeds_fwd(m) for m in models['clean_test']]
                embeds_clean_test = torch.stack(embeds_clean_test, dim=1).squeeze(0) 

                dataset_sample['token_embeds_eval'] = embeds_eval
                dataset_sample['token_embeds_clean_train'] = embeds_clean_train
                dataset_sample['token_embeds_clean_test'] = embeds_clean_test

                return dataset_sample 

            # BUG | Currently `map` is converting new tensors to lists of lists of tensors 
            token_embeds_dataset = triggered_dataset.map(
                map_to_token_embeds_closure,
                batched=False,
                num_proc=1,
                keep_in_memory=True)

            return token_embeds_dataset
        embeds_dataset = map_to_token_embeds(triggered_dataset, models)
        embeds_dataset = {k: v.to(DEVICE) for k,v in embeds_dataset.column_names}
        # embeds_dataloader = torch.utils.data.DataLoader(embeds_dataset, batch_size=args.batch_size, shuffle=False)

        def condensed_input_embeds(input_id_embeddings):
            return torch.stack(input_id_embeddings['eval'], input_id_embeddings['clean_train']).mean()
        condensed_input_embeds = condensed_input_embeds(input_id_embeddings)
        
        import chop
        constraint = chop.constraints.Polytope(vertices=condensed_input_embeds)

        rand_adv_token_id = torch.randint(0, len(condensed_input_embeds))

        # minimize_pairwise_frank_wolfe
        #     give the index of the starting token
        #     returns the soluton as an object
        #         x is the embedding
        #         active_set is the lambda dictionary = {token_index : weight}

        # TODO: define loss using chop.utils.closure, with only 1 argument: the embedding we're optimizing over
        @chop.utils.closure
        def relaxed_loss_fn(adv_embedding):
            # adv_embedding should be (768 vector)        
            # TODO(trusty_patches): insert adv_embedding in the input as embeds within embeds_dataset
            # Assume we only have a single-token trigger
            # TODO(trusty_patches): 
            #   compute_loss(models, embeds_dataset, args.batch_size, with_gradient=True, train_or_test='train', input_field='inputs_embeds')
            # TODO: Might need/want to remove hooks from the model? Maybe consider branching if errors with CUDA?
            return NotImplementedError

        result = chop.optim.minimize_pairwise_frank_wolfe(
                            relaxed_loss_fn,
                            rand_adv_token_id,
                            constraint,
                            step='backtracking',
                            lipschitz=None,
                            max_iter=200,
                            tol=1e-6,
                            callback=None
                            )



    
    def save_results():
        def check_if_folder_exists(folder):
            if not os.path.isdir(folder):
                os.mkdir(folder)
        parent_folder = 'results'
        check_if_folder_exists(parent_folder)
        folder = f'results/'+\
                 f'lambda_{args.lmbda}'+\
                 f'_method_{args.trigger_inversion_method}'+\
                 f'_candidates_{args.num_candidates}'+\
                 f'_trigger_len_{args.trigger_length}'+\
                 f'_rand_starts_{args.num_random_tries}'+\
                 f'_bs_{args.batch_size}'+\
                 f'_locs_{args.q_trigger_insertion_location}_{args.c_trigger_insertion_location}'+\
                 f'_beam_{args.beam_size}'+\
                 f'_more_data_{args.more_clean_data}'+\
                 f'_insertion_{args.trigger_insertion_type}'+\
                 f'_behavior_{args.trigger_behavior}'+\
                 f'_temperature_{args.temperature}'+\
                 f'_max_iter_{args.max_iter}'
        check_if_folder_exists(folder)
        df = pd.DataFrame(columns=['trigger_token_ids', 
                                   'decoded_trigger', 
                                   'clean_triggered',
                                   'eval_triggered',
                                   'train_trigger_inversion_loss', 
                                   'test_eval_loss',
                                   'test_clean_loss',
                                   'test_trigger_inversion_loss'])
        df.loc[0] = [new_trigger, 
                     tokenizer.decode(best_trigger), 
                     round(best_train_loss['trigger_inversion_loss'].item(), 5), 
                     round(best_test_loss['clean_triggered'].item(), 5),
                     round(best_test_loss['eval_triggered'].item(), 5),
                     round(best_test_loss['eval_loss'].item(), 5),
                     round(best_test_loss['clean_loss'].item(), 5),
                     round(best_test_loss['trigger_inversion_loss'].item(), 5)]
        df.to_csv(os.path.join(folder, f'{args.model_num}.csv'))
    save_results()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Trojan Detector for Question & Answering Tasks.')

    # remember to switch this to 1 when making a container
    parser.add_argument('--is_submission', dest='is_submission', action='store_true', help='Flag to determine if this is a submission to the NIST server',  )
    parser.add_argument('--model_num',     default=45,               type=int, help="model number - only used if it's not a submission")                    
    parser.add_argument('--batch_size',    default=10,                 type=int, help='What batch size', )
    parser.add_argument('--more_clean_data', dest='more_clean_data', action='store_true', help='Flag to determine if we want to grab clean examples from the clean models',  )

    # trigger inversion variables
    parser.add_argument('--trigger_behavior',             default='cls',          type=str,   help='Where does the trigger point to?', choices=['self', 'cls'])
    parser.add_argument('--trigger_insertion_type',       default='both',         type=str,   help='Where is the trigger inserted', choices=['context', 'question', 'both'])
    parser.add_argument('--num_random_tries',             default=2,              type=int,   help='How many random starts do we try')
    parser.add_argument('--trigger_length',               default=20,             type=int,   help='How long do we want the trigger to be')
    parser.add_argument('--trigger_inversion_method',     default='discrete',     type=str,   help='Which trigger inversion method do we use', choices=['discrete', 'relaxed'])
    parser.add_argument('--lmbda',                        default=1.,           type=float, help='Weight on the clean loss')
    parser.add_argument('--temperature',                  default=8,             type=float, help='Temperature parameter to divide logits by')
    parser.add_argument('--q_trigger_insertion_location', default='start',        type=str,   help='Where in the question do we want to insert the trigger', choices=['start', 'end'])
    parser.add_argument('--c_trigger_insertion_location', default='start',        type=str,   help='Where in the context do we want to insert the trigger', choices=['start', 'end'])
    
    # discrete trigger inversion specific variables
    parser.add_argument('--max_iter',       default=100,type=int,   help='Max num of iterations')
    parser.add_argument('--num_candidates', default=1,  type=int, help='How many candidates do we want to evaluate in each position during the discrete trigger inversion')
    parser.add_argument('--beam_size',      default=1,   type=int, help='How big do we want the beam size during the discrete trigger inversion')
    
    # continuous trigger inversion specific variables
    parser.add_argument('--beta',           default=.01,      type=float, help='Weight on the sparsity loss')
    parser.add_argument('--rtol',           default=1e-3,     type=float, help="How different are the w's before we stop")

    # do not change these ones
    parser.add_argument('--model_filepath',     type=str, help='Filepath to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='Filepath to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
    parser.add_argument('--result_filepath',    type=str, help='Filepath to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath',    type=str, help='Filepath to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath',   type=str, help='Filepath to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')
    
    parser.set_defaults(is_submission=False, more_clean_data=True, invert_trigger_pointing_to_cls=False, exclude_cls_examples=False, invert_question_and_context_trigger=False)
    args = parser.parse_args()
    
    def modify_args_for_training(args):
        if args.is_submission == 0:
            metadata = pd.read_csv(join(TRAINING_FILEPATH, 'METADATA.csv'))

            id_str = str(100000000 + args.model_num)[1:]
            model_id = 'id-'+id_str

            args.model_filepath = join(TRAINING_FILEPATH, 'models', model_id, 'model.pt')
            args.examples_dirpath = join(TRAINING_FILEPATH, 'models', model_id, 'example_data')
            return args
        else:
            return args
    args = modify_args_for_training(args)

        # modify args and export LAMBDA
    args.eval_model_filepath = args.model_filepath
    args.trigger_insertion_locations = [args.q_trigger_insertion_location, args.c_trigger_insertion_location]
    global LAMBDA
    LAMBDA = args.lmbda

    trojan_detector(args)