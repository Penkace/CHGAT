import dill
import json
import math
import os
import torch


from chgatmodel import classifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report, auc, roc_auc_score, roc_curve
from torch.optim import AdamW
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Dataset
from tqdm import tqdm
from transformers import BertTokenizer

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
torch.distributed.init_process_group(backend="nccl")
# local_rank = int(os.environ['LOCAL_RANK'])
if num_gpus > 1:
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    
else:
    local_rank = 0
DEVICE = torch.device("cuda", local_rank)



TASK = 'ng'
class myDataset(Dataset):
    def __init__(self, examples, fields, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            examples: List of Examples.
            fields (List(tuple(str, Field))): The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        if num_gpus != 1:
            per_worker = int(math.ceil(len(examples) / float(num_gpus)))
            worker_id = local_rank
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(examples))
            examples = examples[iter_start:iter_end]
        self.examples = examples
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]
    # def __iter__(self):
    #     # worker_info = torch.utils.data.get_worker_info()
    #     # print(worker_info.id)
    #     if num_gpus == 1:
    #         for x in self.examples:
    #             yield x
    #     else:
    #         per_worker = int(math.ceil(len(self.examples) / float(num_gpus)))
    #         worker_id = local_rank
    #         iter_start = worker_id * per_worker
    #         iter_end = min(iter_start + per_worker, len(self.examples))
    #         self.examples = self.examples[iter_start:iter_end]
    #         for x in self.examples[iter_start:iter_end]:
    #             yield x
    #     for x in self.examples:
    #         yield x
# if worker_info is None:  # single-process data loading, return the full iterator
#     iter_start = self.start
#     iter_end = self.end
# else:  # in a worker process
#     # split workload
#     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#     worker_id = worker_info.id
#     iter_start = self.start + worker_id * per_worker
#     iter_end = min(iter_start + per_worker, self.end)
#     return iter(range(iter_start, iter_end))
        

class pyTokenizer(object):
    def __init__(self, config):
        super(pyTokenizer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(config['chinese_bert_path'], do_lower_case=True)
        self.pad_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.name_max_length = config['name_max_length']
        self.pronunciation_max_lenth = config['pronunciation_max_lenth']
    
    def encode(self, text):
        text1, text2 = text.split('|')
        # text = self.tokenizer.encode(text1,add_special_tokens=False)
        text1 = self.tokenizer.encode(text1, max_length=self.name_max_length, pad_to_max_length=True, truncation=True)
        text2 = self.tokenizer.encode(text2, max_length=self.pronunciation_max_lenth, pad_to_max_length=True, truncation=True)
        # print(text1)
        return text1+text2

# def load_data(config):
#     train = None
#     valid = None
#     test = None
#     if not config['tokenized']:
#         tokenizer = pyTokenizer(config)
#         label_field = Field(sequential=False, use_vocab=False, batch_first=True)
#         text_field = Field(batch_first=True, use_vocab=False, tokenize=tokenizer.encode)
#         fields = [('label', label_field), ('name', text_field)]

#         train, valid, test = TabularDataset.splits(path=config['data_dir']+"{}_pinyin/".format(TASK), \
#             train='train.tsv', validation='dev.tsv', test ='test.tsv', format='TSV', fields=fields, \
#             skip_header=True)
#     else:
#         examples = torch.load(config['data_dir']+"{}_pinyin/train_examples.pkl".format(TASK), pickle_module=dill)
#         fields = torch.load(config['data_dir']+"{}_pinyin/train_fields.pkl".format(TASK), pickle_module=dill)
#         train = Dataset(examples, fields=fields)
#         examples = torch.load(config['data_dir']+"{}_pinyin/dev_examples.pkl".format(TASK), pickle_module=dill)
#         fields = torch.load(config['data_dir']+"{}_pinyin/dev_fields.pkl".format(TASK), pickle_module=dill)
#         valid =Dataset(examples, fields=fields)
#         examples = torch.load(config['data_dir']+"{}_pinyin/test_examples.pkl".format(TASK), pickle_module=dill)
#         fields = torch.load(config['data_dir']+"{}_pinyin/test_fields.pkl".format(TASK), pickle_module=dill)
#         test = Dataset(examples, fields=fields)
        
#     train_iter, valid_iter, test_iter = BucketIterator.splits((train, valid, test), \
#         batch_sizes=(config['train_batch_size'], config['dev_batch_size'], config['test_batch_size']), \
#         device=config['device'], sort=False)
    
#     return train_iter, valid_iter, test_iter

def load_data(config):
    valid = None
    test = None
    if not config['tokenized']:
        tokenizer = pyTokenizer(config)
        label_field = Field(sequential=False, use_vocab=False, batch_first=True)
        text_field = Field(batch_first=True, use_vocab=False, tokenize=tokenizer.encode)
        fields = [('label', label_field), ('name', text_field)]

        # valid, test = TabularDataset.splits(path=config['data_dir']+"{}_pinyin/".format(TASK), \
        #     validation='dev.tsv', test ='test.tsv', format='TSV', fields=fields, \
        #     skip_header=True)
        valid, test = TabularDataset.splits(path=config['data_dir']+"{}_pinyin/".format(TASK), \
            validation='train_0.tsv', test ='train_0.tsv', format='TSV', fields=fields, \
            skip_header=True)
    else:
        examples = torch.load(config['data_dir']+"{}_pinyin/dev_examples.pkl".format(TASK), pickle_module=dill)
        fields = torch.load(config['data_dir']+"{}_pinyin/dev_fields.pkl".format(TASK), pickle_module=dill)
        valid = Dataset(examples, fields=fields)
        examples = torch.load(config['data_dir']+"{}_pinyin/test_examples.pkl".format(TASK), pickle_module=dill)
        fields = torch.load(config['data_dir']+"{}_pinyin/test_fields.pkl".format(TASK), pickle_module=dill)
        test = Dataset(examples, fields=fields)
        
    valid_iter, test_iter = BucketIterator.splits((valid, test), \
        batch_sizes=(config['dev_batch_size'], config['test_batch_size']), \
        device=config['device'], sort=False)
    # valid_iter = DataLoader(valid, batch_size=config['dev_batch_size'])
    # test_iter = DataLoader(test, batch_size=config['test_batch_size'])
    return valid_iter, test_iter

def load_model(config, n_gpu = 0):
    model = classifier(config, num_labels = 2)
    model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    # if config['device'] == 'cuda':
    #     n_gpu = torch.cuda.device_count()
    #     if n_gpu > 1:
    #         model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.08},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=config['learning_rate'])

    return model, optimizer, n_gpu

def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader, config, n_gpu, label_list):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_acc = 0
    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 10000000000000

    test_best_acc = 0
    test_best_precision = 0
    test_best_recall = 0
    test_best_f1 = 0
    test_best_loss = 1000000000000000

    # model.train()
    for idx in range(int(config['num_train_epochs'])):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 10)
        print("EPOCH: ", str(idx))
        for train_index in range(config['num_train_file']):
            train_dataset = None
            # print(num_gpus,local_rank)
            if not config['tokenized']:
                tokenizer = pyTokenizer(config)
                label_field = Field(sequential=False, use_vocab=False, batch_first=True)
                text_field = Field(batch_first=True, use_vocab=False, tokenize=tokenizer.encode)
                fields = [('label', label_field), ('name', text_field)]
                train = TabularDataset(path=config['data_dir']+"{}_pinyin/train_{}.tsv".format(TASK, train_index), format='TSV', fields=fields, skip_header=True)
                train_dataset = myDataset(train.examples, fields=fields)
            else:
                examples = torch.load(config['data_dir']+"{}_pinyin/train_{}_examples.pkl".format(TASK, train_index), pickle_module=dill)
                fields = torch.load(config['data_dir']+"{}_pinyin/train_{}_fields.pkl".format(TASK, train_index), pickle_module=dill)
                # print('here')
                train_dataset = myDataset(examples, fields=fields)
            train_dataloader = BucketIterator(train_dataset, batch_size=config['train_batch_size'], device=config['device'], sort=False)
            # train_dataloader = DataLoader(train_dataset, batch_size=config['train_batch_size']) #,sampler=DistributedSampler(train_dataset)
            for step, batch in tqdm(enumerate(train_dataloader)):
                model.train()
                (label_ids, input_ids),_ = batch
                if config['tokenized']:
                    input_ids = input_ids[0]
                if torch.cuda.is_available():
                    label_ids = label_ids.cuda()
                    input_ids = input_ids.cuda()
                loss, _= model(input_ids, label_ids)
    #             print(type(hgat_loss))
                # if n_gpu > 1:
                #     loss = loss.mean()

                # sum_loss = loss

                loss.backward()
                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if ((nb_tr_steps+1) % config['checkpoint'] == 0) and (local_rank == 0):
                    print("-*-" * 15)
                    print("current training loss is : ")
                    print("classification loss, hgat loss")
                    print(loss.item())
                    tmp_dev_loss, tmp_dev_acc, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config['tokenized'])
                    print("......" * 10)
                    print("DEV: loss, acc, f1")
                    print(tmp_dev_loss, tmp_dev_acc, tmp_dev_f1)

                    if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
                        dev_best_acc = tmp_dev_acc
                        dev_best_loss = tmp_dev_loss
                        dev_best_f1 = tmp_dev_f1

                        tmp_test_loss, tmp_test_acc, tmp_test_f1 = eval_checkpoint(model, test_dataloader, config['tokenized'])
                        print("......" * 10)
                        print("TEST: loss, acc, f1")
                        print(tmp_test_loss, tmp_test_acc, tmp_test_f1)

                        test_best_acc = tmp_test_acc
                        test_best_loss = tmp_test_loss
                        test_best_f1 = tmp_test_f1

                        # export model
                        if config['export_model']:
                            model_to_save = model.module if hasattr(model, "module") else model
                            output_model_file = os.path.join(config['output_dir'], "{}/pytorch_model{}.bin".format(TASK,nb_tr_steps))
                            torch.save(model_to_save.state_dict(), output_model_file)

                    print("-*-" * 15)

    # export a trained mdoel
    
    if (local_rank == 0):
        print("-*-" * 15)
        print("current training loss is : ")
        print("classification loss, hgat loss")
        print(loss.item())
        tmp_dev_loss, tmp_dev_acc, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config['tokenized'])
        print("......" * 10)
        print("DEV: loss, acc, f1")
        print(tmp_dev_loss, tmp_dev_acc, tmp_dev_f1)

        if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
            dev_best_acc = tmp_dev_acc
            dev_best_loss = tmp_dev_loss
            dev_best_f1 = tmp_dev_f1

            tmp_test_loss, tmp_test_acc, tmp_test_f1 = eval_checkpoint(model, test_dataloader, config['tokenized'])
            print("......" * 10)
            print("TEST: loss, acc, f1")
            print(tmp_test_loss, tmp_test_acc, tmp_test_f1)

            test_best_acc = tmp_test_acc
            test_best_loss = tmp_test_loss
            test_best_f1 = tmp_test_f1
        if config['export_model']:
            model_to_save = model
            output_model_file = os.path.join(config['output_dir'], "{}/pytorch_model.bin".format(TASK))
            torch.save(model_to_save.state_dict(), output_model_file)
        print("=&=" * 15)
        print("DEV: current best precision, recall, f1, acc, loss ")
        print(dev_best_precision, dev_best_recall, dev_best_f1, dev_best_acc, dev_best_loss)
        print("TEST: current best precision, recall, f1, acc, loss ")
        print(test_best_precision, test_best_recall, test_best_f1, test_best_acc, test_best_loss)
        print("=&=" * 15)


def eval_checkpoint(model_object, eval_dataloader, tokenized):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    eval_loss = 0
    eval_accuracy = []
    eval_f1 = []
    logits_all = []
    labels_all = []
    eval_steps = 0
    for batch in tqdm(eval_dataloader):
        (label_ids, input_ids),_ = batch
        if tokenized:
            input_ids = input_ids[0]
        if torch.cuda.is_available():
            label_ids = label_ids.cuda()
            input_ids = input_ids.cuda()
        with torch.no_grad():
            tmp_eval_loss, logits = model_object(input_ids, label_ids)

        
        label_ids = label_ids.to("cpu").numpy()
        
        logits = logits.detach().cpu().numpy()

        eval_loss += tmp_eval_loss.mean().item()

        logits_all.extend(logits.tolist())
        labels_all.extend(label_ids.tolist())
        eval_steps += 1

    average_loss = round(eval_loss / eval_steps, 4)
    eval_accuracy = float(accuracy_score(y_true=labels_all, y_pred=logits_all))
    matric = precision_recall_fscore_support(labels_all, logits_all, average = 'binary')
    eval_f1 = matric[2]
    fpr, tpr, threshold = roc_curve(labels_all, logits_all, pos_label=2)
    print(auc(fpr, tpr))
    print('AUROC')
    print(roc_auc_score(labels_all, logits_all))
    print('Classification Report:')
    print(classification_report(labels_all, logits_all, labels=[0,1], digits=4))

    return average_loss, eval_accuracy, eval_f1  # eval_precision, eval_recall, eval_f1

def main():
    config = json.load(open('./config/CHGAT/{}.json'.format(TASK)))
    # train_loader, dev_loader, test_loader = load_data(config)
    dev_loader, test_loader = load_data(config)
    train_loader = None
    model, optimizer, n_gpu = load_model(config, num_gpus)
    # print(n_gpu)
    train(model, optimizer, train_loader, dev_loader, test_loader, config, n_gpu, [0, 1])


if __name__ == "__main__":
    main()