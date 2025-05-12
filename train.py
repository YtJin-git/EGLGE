#
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
import pprint

import numpy as np
import torch
import torch.nn as nn
import tqdm
import yaml
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from models.model_factory import get_model
from parameters import parser
import torch.optim as optim

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *
from models.gcn import spm_to_tensor
from models.EGLGE import EstimateAdj


torch.multiprocessing.set_sharing_strategy('file_system')


def train_model(model, config, train_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    model.train()
    best_val_metric = 0
    best_test_metric = 0
    best_val_loss = 1e5
    best_test_loss = 1e5
    best_epoch = 0
    final_model_state = None

    val_results = []
    test_results = []

    scheduler = get_scheduler(model.optimizer, config, len(train_dataloader))
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses, epoch_mask_loss = [], []
        step = 0
        for bid, batch in enumerate(train_dataloader):

            # Optimize M
            if step % config.train_mask_step == 0:
                model.train_or_not(False)
                model.eval()
                model.estimator.train()
                model.optimizer_adj.zero_grad()
                predict = model(batch, train_dataset.train_pairs)
                loss_mask = model.loss_calu(predict, batch)
                loss_mask.backward()
                model.optimizer_adj.step()
                data = proximal_op(spm_to_tensor(model.adj_matrix).to_dense().cuda(), model.estimator.estimated_adj,
                                   config.beta)
                model.estimator.estimated_adj.data.copy_(data)

            # Optimize W
            model.train_or_not(True)
            model.train()
            model.estimator.eval()
            model.optimizer.zero_grad()
            predict = model(batch, train_dataset.train_pairs)
            loss = model.loss_calu(predict, batch)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                model.optimizer.step()
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss.item())
            epoch_mask_loss.append(loss_mask.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:]), "mask loss": np.mean(epoch_mask_loss[-50:])})
            progress_bar.update()
            step = step + 1

        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} "
                           f"train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        test_flag = True
        if test_flag:
            print("--- Evaluating test dataset on Closed World ---")
            test_result = evaluate(model, test_dataset, config)
            test_results.append(test_result)

            if config.val_metric == 'best_loss' and test_result['loss'] < best_val_loss:
                best_val_loss = test_result['loss']
                best_epoch = i
                torch.save(model.state_dict(), os.path.join(
                    config.save_path, "test_best.pt"))
            if config.val_metric != 'best_loss' and test_result[config.val_metric] > best_val_metric:
                best_val_metric = test_result[config.val_metric]
                best_epoch = i
                torch.save(model.state_dict(), os.path.join(
                    config.save_path, "test_best.pt"))


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    test_saved_results['loss'] = loss_avg
    return test_saved_results





if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    print(config)
    # set the seed value
    set_seed(config.seed)

    dataset_path = config.dataset_path
    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural',
                                       same_prim_sample=config.same_prim_sample,
                                       open_world=config.open_world)

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural',
                                      open_world=config.open_world)

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)
    model = get_model(config, attributes=attributes, classes=classes, offset=offset, dset=train_dataset).cuda()
    model = model.cuda()
    os.makedirs(config.save_path, exist_ok=True)
    train_model(model, config, train_dataset, test_dataset)
    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")
