# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
from transformers import LxmertForQuestionAnswering, LxmertConfig, LxmertTokenizer


DataTuple = collections.namedtuple("DataTuple", "dataset loader evaluator")


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        # load the model
        self.tokenizer = LxmertTokenizer.from_pretrained("/ssd-playpen/avmendoz/local_lxmert-base-uncased")
        self.model = LxmertForQuestionAnswering.from_pretrained("/ssd-playpen/avmendoz/local_lxmert-base-uncased")
        # load the data
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        self.model.resize_num_qa_labels(self.train_tuple.dataset.num_answers)
        if args.valid != "":
            valid_bsize = 32 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize, shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        # Load pre-trained weights
        # GPU options
        self.model = self.model.cuda()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss()
        if "bert" in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam

            self.optim = BertAdam(
                list(self.model.parameters()), lr=args.lr, warmup=0.1, t_total=t_total
            )
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (
            (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        )

        best_valid = 0.0
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(
                enumerate(loader)
            ):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                inputs = self.tokenizer(
                    sent,
                    padding="max_length",
                    max_length=20,
                    truncation=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    add_special_tokens=True
                )
                token_type_ids=torch.tensor(inputs.token_type_ids).cuda()
                input_ids = torch.tensor(inputs.input_ids).cuda()
                attention_mask = torch.tensor(inputs.attention_mask).cuda()
                max_value, target = target.max(1)
                output = self.model(input_ids=input_ids, visual_feats=feats, visual_pos=boxes, labels=target, token_type_ids=token_type_ids, return_dict=True, output_attentions=False)
                loss = output.loss
                logit = output.question_answering_score


                #assert logit.dim() == target.dim() == 2
                #if args.mce_loss:
                #    max_value, target = target.max(1)
                #    loss = self.mce_loss(logit, target) * logit.size(1)
                #else:
                #    loss = self.bce_loss(logit, target)
                #    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (
                epoch,
                evaluator.evaluate(quesid2ans) * 100.0,
            )

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (
                    epoch,
                    valid_score * 100.0,
                ) + "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.0)

            print(log_str, end="")

            with open(self.output + "/log_test.log", "a") as f:
                f.write(log_str)
                f.flush()

        self.save("WOOT")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, target = datum_tuple[:5]  # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                inputs = self.tokenizer(sent, padding="max_length", max_length=20, truncation=True)
                token_type_ids=torch.tensor(inputs.token_type_ids).cuda()
                input_ids = torch.tensor(inputs.input_ids).cuda()
                #max_value, target = target.max(1)
                output = self.model(input_ids=input_ids, visual_feats=feats, visual_pos=boxes, token_type_ids=token_type_ids, return_dict=True)
                loss = output.loss
                logit = output.question_answering_score
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if ".module" in key:
                state_dict[key.replace(".module", "")] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    #conf = LxmertConfig(num_qa_labels=1536)
    #model = LxmertForQuestionAnswering(conf)
    #model = model.from_pretrained("unc-nlp/lxmert-base-uncased", cache_dir="ssd-playpen/avmendoz")
    #gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased", cache_dir="ssd-playpen/avmendoz")
    gqa = GQA()
    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False  # Always loading all data in test
        if "submit" in args.test:
            raise Exception("not testing")
            gqa.predict(
                get_tuple(
                    args.test, bs=args.batch_size, shuffle=False, drop_last=False
                ),
                dump=os.path.join(args.output, "submit_predict.json"),
            )
        else:
            result = gqa.evaluate(
                get_tuple(
                    "valid", bs=args.batch_size, shuffle=False, drop_last=False
                ),
                dump=os.path.join(args.output, "testdev_predict_aug.json"),
            )
            print(result)
    else:
        # raise Exception("model is not testing")
        print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print("Splits in Train data:", gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print("whao")
            print("Splits in Valid data:", gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)
