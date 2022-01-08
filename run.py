#import functionality

from comet_ml import Experiment

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os
import random 
import pickle

import models


experiment = Experiment(project_name="project_name")
hyper_params = {
    # data format:
    # <data_dir>/<data>/<data>.train.c2c
    # <data_dir>/<data>/<data>.test.c2c
    # <data_dir>/<data>/<data>.val.c2c
    # <data_dir>/<data>/<data>.dict.c2c
    "dataset_name": 'mixed_slt_opt_multiclass',
    "data_dir": 'data_preprocessed_c2c',
    # load: load saved checkpoint/model? (True to load from save: <save_dir>/<data>-model.pt)
    "load": False,
    # save_dir: model location (<save_dir>/<data>-model.pt)
    "save_dir": 'checkpoints',
    # log_dir: log location (<log_dir>/<data>-log.txt)
    "log_dir": 'logs',
    # embedding dim: size of fully connected layer (vector compression)
    "embedding_dim": 128,
    # lstm dim: hidden dim of lstm
    "lstm_dim": 16,
    # dropout: dropout rate for prevention of coadaption of neurons
    "dropout": 0.25,
    # batch_size, chunks: training/evaluation uses batch_size*chunks examples at a time
    "batch_size": 128,
    "chunks": 10,
    # max_length: max number of paths in examples
    "max_length": 200,
    # max_path_length: max path segments in paths
    "max_path_length": 10,
    # n_epochs: number of epochs to train
    "n_epochs": 15,
}
experiment.log_parameters(hyper_params)


# setup parameters

SEED = 1234
DATA_DIR = hyper_params["data_dir"]
DATASET = hyper_params["dataset_name"]
EMBEDDING_DIM = hyper_params["embedding_dim"]
LSTM_DIM = hyper_params["lstm_dim"]
DROPOUT = hyper_params["dropout"]
BATCH_SIZE = hyper_params["batch_size"]
CHUNKS = hyper_params["chunks"]
MAX_LENGTH = hyper_params["max_length"]
MAX_PATH_LENGTH = hyper_params["max_path_length"]
LOG_EVERY = 100 #print log of results after every LOG_EVERY batches
N_EPOCHS = hyper_params["n_epochs"]
LOG_DIR = hyper_params["log_dir"]
SAVE_DIR = hyper_params["save_dir"]
LOG_PATH = os.path.join(LOG_DIR, f'{DATASET}-log.txt')
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f'{DATASET}-model.pt')
LOAD = hyper_params["load"]

# set random seeds for reproducability

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#load counts of each token in dataset

with open(f'{DATA_DIR}/{DATASET}/{DATASET}.dict.c2c', 'rb') as file:
    word2count = pickle.load(file)
    node2count = pickle.load(file)
    target2count = pickle.load(file)
    n_training_examples = pickle.load(file)

# create vocabularies, initialized with unk and pad tokens

word2idx = {'<unk>': 0, '<pad>': 1}
node2idx = {'<unk>': 0, '<pad>': 1 }
target2idx = {'<unk>': 0, '<pad>': 1}

idx2word = {}
idx2node = {}
idx2target = {}


for w in word2count.keys():
    word2idx[w] = len(word2idx)
    
for k, v in word2idx.items():
    idx2word[v] = k
    
for p in node2count.keys():
    node2idx[p] = len(node2idx)
    
for k, v in node2idx.items():
    idx2node[v] = k
    
for t in target2count.keys():
    target2idx[t] = len(target2idx)
    
for k, v in target2idx.items():
    idx2target[v] = k

model = models.Code2Class(len(word2idx), len(node2idx), EMBEDDING_DIM, len(target2idx), DROPOUT, MAX_PATH_LENGTH, LSTM_DIM, BATCH_SIZE, MAX_LENGTH)

if LOAD:
    print(f'Loading model from {MODEL_SAVE_PATH}')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

optimizer = optim.Adam(model.parameters())

criterion = nn.MultiLabelSoftMarginLoss() #nn.CrossEntropyLoss()

device = torch.device('cuda')

model = model.to(device)
criterion = criterion.to(device)

def calculate_f1_at_k(fx, y, k):
    """
    Calculate precision, recall and F1 score
    - Takes top-k predictions
    - Converts to strings
    - Calculates TP, FP and FN
    - Calculates precision, recall and F1 score

    fx = [batch size, output dim]
     y = [batch size]
    """
    # take top k predictions
    pred_idxs = fx.topk(k)[1]
    # convert to strings
    pred_names = [[idx2target[i.item()] for i in preds] for preds in pred_idxs]
    # take true values
    original_names = []
    for i in range(len(y)):
        original_names.append([])
    for i, ex in enumerate(y.tolist()):
        for j, v in enumerate(ex):
            if v == 1:
                original_names[i].append(idx2target[j])
    # calculate TP, FP and FN
    true_positive, false_positive, false_negative = 0, 0, 0
    for p, o in zip(pred_names, original_names):
        predicted_subtokens = p
        original_subtokens = o
        for subtok in predicted_subtokens:
            if subtok in original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in original_subtokens:
            if not subtok in predicted_subtokens:
                false_negative += 1
    try:
        # calculates precision, recall and F1 score
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1

def parse_line(line):
    """
    Takes a string 'x y1,p1,z1 y2,p2,z2 ... yn,pn,zn and splits into name (x) and tree [[y1,p1,z1], ...]
    """
    name, *tree = line.split(' ')
    tree = [t.split(',') for t in tree if t != '' and t != '\n']
    return name, tree

def file_iterator(file_path):
    """
    Takes a file path and creates and iterator
    For each line in the file, parse into a name and tree
    Pad tree to maximum length
    Yields example:
    - example_name = 'target'
    - example_body = [['left_node','path','right_node'], ...]
    """
    with open(file_path, 'r') as f:

        for line in f:

            #each line is an example

            #each example is made of the function name and then a sequence of triplets
            #the triplets are (node, path, node)

            example_name, example_body = parse_line(line)

            #max length set while preprocessing, make sure none longer

            example_length = len(example_body)

            if example_length > MAX_LENGTH:
                random.shuffle(example_body)
                example_body = example_body[:200]
                example_length = len(example_body)
            assert example_length <= MAX_LENGTH
            
            #need to pad all to maximum length
            for i in range (MAX_LENGTH - example_length):
                example_body += [['<pad>', '<pad>', '<pad>']]
            
            assert len(example_body) == MAX_LENGTH

            yield example_name, example_body, example_length

def numericalize(examples, n):
    """
    Examples are a list of list of lists, i.e. examples[0] = [['left_node','path','right_node'], ...]
    n is how many batches we are getting our of `examples`
    
    Get a batch of raw (still strings) examples
    Create tensors to store them all
    Numericalize each raw example within the batch and convert whole batch tensor
    Yield tensor batch
    """

    assert n*BATCH_SIZE <= len(examples)

    for i in range(n):

        #get the raw data
                    
        raw_batch_name, raw_batch_body, batch_lengths = zip(*examples[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
        
        #create a tensor to store the batch
        
        tensor_n = torch.zeros((BATCH_SIZE, len(target2idx))).long() #name
        tensor_l = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #left node
        #tensor_p = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #path
        tensor_p = torch.zeros((BATCH_SIZE, MAX_LENGTH, MAX_PATH_LENGTH)).long() #path
        tensor_r = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #right node
        mask = torch.ones((BATCH_SIZE, MAX_LENGTH)).float() #mask
        
        #for each example in our raw data
        
        for j, (names, body, length) in enumerate(zip(raw_batch_name, raw_batch_body, batch_lengths)):

            # pad paths to MAX PATH LENGTH
            for path in body:
                path[1] = path[1].split('|')
                for k in range(MAX_PATH_LENGTH-len(path[1])):
                    path[1] += ['<pad>']

            names = names.split('|')

            #convert to idxs using vocab
            #use <unk> tokens if item doesn't exist inside vocab
            temp_names = [target2idx.get(name, target2idx['<unk>']) for name in names]
            temp_n = [0]*len(target2idx)
            for idx in temp_names:
                temp_n[idx] = 1
            temp_l, temp_p, temp_r = zip(*[(word2idx.get(l, word2idx['<unk>']), [node2idx.get(p, node2idx['<unk>']) for p in path], word2idx.get(r, word2idx['<unk>'])) for l, path, r in body])
            # cut off paths at MAX PATH LENGTH
            temp_p = tuple(path[:MAX_PATH_LENGTH] for path in temp_p)
            
            #store idxs inside tensors
            tensor_n[j,:] = torch.LongTensor(temp_n)
            tensor_l[j,:] = torch.LongTensor(temp_l)
            l = 0
            for path in temp_p:
                tensor_p[j,l,:] = torch.LongTensor(path)
                l+=1
            #tensor_p[j,:] = torch.LongTensor(temp_p)
            tensor_r[j,:] = torch.LongTensor(temp_r)   
            
            #create masks
            mask[j, length:] = 0

        yield tensor_n, tensor_l, tensor_p, tensor_r, mask

def get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, k, optimizer=None):
    """
    Takes inputs, calculates loss and other metrics, then calculates gradients and updates parameters

    if optimizer is None, then we are doing evaluation so no gradients are calculated and no parameters are updated
    """

    if optimizer is not None:
        optimizer.zero_grad()

    fx = model(tensor_l, tensor_p, tensor_r)

    loss = criterion(fx, tensor_n)

    precision, recall, f1 = calculate_f1_at_k(fx, tensor_n, k)
    
    if optimizer is not None:
        loss.backward()
        optimizer.step()    

    return loss.item(), precision, recall, f1

def train(model, file_path, optimizer, criterion):
    """
    Training loop for the model
    Dataset is too large to fit in memory, so we stream it
    Get BATCH_SIZE * CHUNKS examples at a time (default = 1024 * 10 = 10,240)
    Shuffle the BATCH_SIZE * CHUNKS examples
    Convert raw string examples into numericalized tensors
    Get metrics and update model parameters

    Once we near end of file, may have less than BATCH_SIZE * CHUNKS examples left, but still want to use
    So we calculate number of remaining whole batches (len(examples)//BATCH_SIZE) then do that many updates
    """
    n_batches = 0

    epoch_loss = 0
    epoch_r = 0
    epoch_p = 0
    epoch_f1 = 0

    model.train()

    examples = []

    for example_name, example_body, example_length in file_iterator(file_path):

        examples.append((example_name, example_body, example_length))

        if len(examples) >= (BATCH_SIZE * CHUNKS):

            random.shuffle(examples)

            for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, CHUNKS):

                #place on gpu

                tensor_n = tensor_n.to(device)
                tensor_l = tensor_l.to(device)
                tensor_p = tensor_p.to(device)
                tensor_r = tensor_r.to(device)

                #put into model
                loss, p, r, f1 = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, 1, optimizer)

                epoch_loss += loss
                epoch_p += p
                epoch_r += r
                epoch_f1 += f1

                n_batches += 1

                if n_batches % LOG_EVERY == 0:

                    loss = epoch_loss / n_batches
                    precision = epoch_p / n_batches
                    recall = epoch_r / n_batches
                    f1 = epoch_f1 / n_batches

                    log = f'\t| Batches: {n_batches} | Completion: {((n_batches*BATCH_SIZE)/n_training_examples)*100:03.3f}% |\n'
                    log += f'\t| Loss: {loss:02.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
                    with open(LOG_PATH, 'a+') as f:
                        f.write(log+'\n')
                    print(log)

            examples = []

        else:
            pass

    #outside of `file_iterator`, but will probably still have some examples left over
    random.shuffle(examples)

    #get amount of batches we have left
    n = len(examples)//BATCH_SIZE

    #train with remaining batches
    for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, n):

        #place on gpu

        tensor_n = tensor_n.to(device)
        tensor_l = tensor_l.to(device)
        tensor_p = tensor_p.to(device)
        tensor_r = tensor_r.to(device)

        #put into model

        loss, p, r, f1 = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, 1, optimizer)

        epoch_loss += loss
        epoch_p += p
        epoch_r += r
        epoch_f1 += f1

        n_batches += 1

    return epoch_loss / n_batches, epoch_p / n_batches, epoch_r / n_batches, epoch_f1 / n_batches

def evaluate(model, file_path, criterion, k):
    """
    Similar to training loop, but we do not pass optimizer to get_metrics
    Also wrap get_metrics in `torch.no_grad` to avoid calculating gradients
    """

    n_batches = 0
    
    epoch_loss = 0
    epoch_r = 0
    epoch_p = 0
    epoch_f1 = 0
    
    model.eval()
    
    examples = []
    
    for example_name, example_body, example_length in file_iterator(file_path):

        examples.append((example_name, example_body, example_length))

        if len(examples) >= (BATCH_SIZE * CHUNKS):

            random.shuffle(examples)

            for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, CHUNKS):

                #place on gpu

                tensor_n = tensor_n.to(device)
                tensor_l = tensor_l.to(device)
                tensor_p = tensor_p.to(device)
                tensor_r = tensor_r.to(device)

                #put into model
                with torch.no_grad():
                    loss, p, r, f1 = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, k)

                epoch_loss += loss
                epoch_p += p
                epoch_r += r
                epoch_f1 += f1
                
                n_batches += 1
                                    
                if n_batches % LOG_EVERY == 0:
            
                    loss = epoch_loss / n_batches
                    precision = epoch_p / n_batches
                    recall = epoch_r / n_batches
                    f1 = epoch_f1 / n_batches

                    log = f'\t| Batches: {n_batches} |\n'
                    log += f'\t| Loss: {loss:02.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
                    with open(LOG_PATH, 'a+') as f:
                        f.write(log+'\n')
                    print(log)

            examples = []
                            
        else:
            pass
      
    #outside of for line in f, but will still have some examples left over

    random.shuffle(examples)

    n = len(examples)//BATCH_SIZE

    for tensor_n, tensor_l, tensor_p, tensor_r, mask in numericalize(examples, n):
            
        #place on gpu

        tensor_n = tensor_n.to(device)
        tensor_l = tensor_l.to(device)
        tensor_p = tensor_p.to(device)
        tensor_r = tensor_r.to(device)
            
        #put into model
        with torch.no_grad():
            loss, p, r, f1 = get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion, k)

        epoch_loss += loss
        epoch_p += p
        epoch_r += r
        epoch_f1 += f1
        
        n_batches += 1

    return epoch_loss / n_batches, epoch_p / n_batches, epoch_r / n_batches, epoch_f1 / n_batches

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

if not os.path.isdir(f'{LOG_DIR}'):
    os.makedirs(f'{LOG_DIR}')

if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)


with experiment.train():
    for epoch in range(N_EPOCHS):

        log = f'Epoch: {epoch+1:02} - Training'
        with open(LOG_PATH, 'a+') as f:

            f.write(log+'\n')
        print(log)

        train_loss, train_p, train_r, train_f1 = train(model, f'{DATA_DIR}/{DATASET}/{DATASET}.train.c2c', optimizer, criterion)
        experiment.log_metric("loss", train_loss, step=epoch)
        experiment.log_metric("precision", train_p, step=epoch)
        experiment.log_metric("recall", train_r, step=epoch)
        experiment.log_metric("f1", train_f1, step=epoch)

        log = f'Epoch: {epoch+1:02} - Validation'
        with open(LOG_PATH, 'a+') as f:
            f.write(log+'\n')
        print(log)

        valid_loss, valid_p_1, valid_r_1, valid_f1_1 = evaluate(model, f'{DATA_DIR}/{DATASET}/{DATASET}.val.c2c', criterion, 1)
        experiment.log_metric("validation_loss", valid_loss, step=epoch)
        experiment.log_metric("validation_p_at1", valid_p_1, step=epoch)
        experiment.log_metric("validation_r_at1", valid_r_1, step=epoch)
        experiment.log_metric("validation_f1_at1", valid_f1_1, step=epoch)

        valid_loss, valid_p_3, valid_r_3, valid_f1_3 = evaluate(model, f'{DATA_DIR}/{DATASET}/{DATASET}.val.c2c', criterion, 3)
        experiment.log_metric("validation_loss", valid_loss, step=epoch)
        experiment.log_metric("validation_p_at3", valid_p_3, step=epoch)
        experiment.log_metric("validation_r_at3", valid_r_3, step=epoch)
        experiment.log_metric("validation_f1_at3", valid_f1_3, step=epoch)

        valid_loss, valid_p_5, valid_r_5, valid_f1_5 = evaluate(model, f'{DATA_DIR}/{DATASET}/{DATASET}.val.c2c', criterion, 5)
        experiment.log_metric("validation_loss", valid_loss, step=epoch)
        experiment.log_metric("validation_p_at5", valid_p_5, step=epoch)
        experiment.log_metric("validation_r_at5", valid_r_5, step=epoch)
        experiment.log_metric("validation_f1_at5", valid_f1_5, step=epoch)


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        log = f'| Epoch: {epoch+1:02} |\n'
        log += f'| Train Loss: {train_loss:.3f} | Train Precision: {train_p:.3f} | Train Recall: {train_r:.3f} | Train F1: {train_f1:.3f} |\n'
        log += f'| Val. Loss: {valid_loss:.3f} | Val. Precision: {valid_p_1:.3f} | Val. Recall: {valid_r_1:.3f} | Val. F1: {valid_f1_1:.3f} |'
        with open(LOG_PATH, 'a+') as f:
            f.write(log+'\n')
        print(log)


with experiment.test():
    log = 'Testing'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    test_loss, test_p_1, test_r_1, test_f1_1 = evaluate(model, f'{DATA_DIR}/{DATASET}/{DATASET}.test.c2c', criterion, 1)
    experiment.log_metric("loss", test_loss, step=N_EPOCHS)
    experiment.log_metric("precision_at1", test_p_1, step=N_EPOCHS)
    experiment.log_metric("recall_at1", test_r_1, step=N_EPOCHS)
    experiment.log_metric("f1_at1", test_f1_1, step=N_EPOCHS)

    test_loss, test_p_3, test_r_3, test_f1_3 = evaluate(model, f'{DATA_DIR}/{DATASET}/{DATASET}.test.c2c', criterion, 3)
    experiment.log_metric("loss", test_loss, step=N_EPOCHS)
    experiment.log_metric("precision_at3", test_p_3, step=N_EPOCHS)
    experiment.log_metric("recall_at3", test_r_3, step=N_EPOCHS)
    experiment.log_metric("f1_at3", test_f1_3, step=N_EPOCHS)

    test_loss, test_p_5, test_r_5, test_f1_5 = evaluate(model, f'{DATA_DIR}/{DATASET}/{DATASET}.test.c2c', criterion, 5)
    experiment.log_metric("loss", test_loss, step=N_EPOCHS)
    experiment.log_metric("precision_at5", test_p_5, step=N_EPOCHS)
    experiment.log_metric("recall_at5", test_r_5, step=N_EPOCHS)
    experiment.log_metric("f1_at5", test_f1_5, step=N_EPOCHS)

    log = f'| Test Loss: {test_loss:.3f} | Test Precision: {test_p_1:.3f} | Test Recall: {test_r_1:.3f} | Test F1: {test_f1_1:.3f} |'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

