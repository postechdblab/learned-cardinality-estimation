"""Model training."""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import naru.common as common
from naru.common import *
#import naru.common.config as config
import naru.estimators as estimators_lib
import naru.datasets as datasets
import naru.made as made
import naru.transformer as transformer

def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    print(s)
    return ret


def RunEpoch(split,
             model,
             opt,
             train_data,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, xb in enumerate(loader):
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if config['constant-lr']:
                    lr = config['constant-lr']
                elif config['warmups']:
                    t = config['warmups']
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                        .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))

                    loss = (-logps).mean()

        losses.append(loss.item())

        if step % log_every == 0:
            if split == 'train':
                print(
                    '{} Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(model.model_id, epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, lr))
            else:
                print('{} Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(model.model_id, epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print('%d %s epoch average loss: %f' % (model.model_id ,split, np.mean(losses)))
    if return_losses:
        return losses
    return np.mean(losses)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('{} Number of model parameters: {} (~= {:.1f}MB)'.format(model.model_id, num_params, mb))
    print(model)
    return mb


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def MakeMade(model_id, scale, cols_to_train, seed, fixed_ordering=None):
    if config['inv-order']:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        model_id=model_id,
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        config['layers'] if config['layers'] > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=config['input-encoding'],
        output_encoding=config['output-encoding'],
        embed_size=32,
        seed=seed,
        do_direct_io_connections=config['direct-io'],
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=config['residual'],
        fixed_ordering=fixed_ordering,
        column_masking=config['column-masking'],
    ).to(DEVICE)
    
    print(f'{model_id} Free GPU memory after MADE')
    get_gpu_memory()


    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=config['blocks'],
        d_model=config['dmodel'],
        d_ff=config['dff'],
        num_heads=config['heads'],
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=config['transformer-act'],
        fixed_ordering=fixed_ordering,
        column_masking=config['column-masking'],
        seed=seed,
    ).to(DEVICE)


def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


def TrainNARU(model_id, table_path, model_path, seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

    print(f'{model_id} Free GPU memory before TrainTask {model_id}')
    get_gpu_memory()

    #assert config['dataset'] in ['dmv-tiny', 'dmv', 'custom']
    assert config['dataset'] in ['custom']
    if config['dataset'] == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif config['dataset'] == 'dmv':
        table = datasets.LoadDmv()
    elif config['dataset'] == 'custom':
        table = datasets.LoadCustom(table_path)
    
    print('LoadCustom done')

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns
                                           ]).size(), [2])[0]
    fixed_ordering = None

    print(model_id, table)

    table_train = table

    if config['heads'] > 0:
        model = MakeTransformer(cols_to_train=table.columns,
                                fixed_ordering=fixed_ordering,
                                seed=seed)
    else:
        if config['dataset'] in ['dmv-tiny', 'dmv', 'custom']:
            model = MakeMade(
                model_id=model_id,
                scale=config['fc-hiddens'],
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
        else:
            assert False, config['dataset']

    mb = ReportModel(model)

    if not isinstance(model, transformer.Transformer):
        print('Applying InitWeight()')
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), 2e-4)

    bs = config['bs']
    log_every = 200

    train_data = common.TableDataset(table_train)

    train_losses = []
    train_start = time.time()
    for epoch in range(config['epochs']):

        mean_epoch_train_loss = RunEpoch('train',
                                         model,
                                         opt,
                                         train_data=train_data,
                                         val_data=train_data,
                                         batch_size=bs,
                                         epoch_num=epoch,
                                         log_every=log_every,
                                         table_bits=table_bits)

        if epoch % 1 == 0:
            print('{} epoch {} train loss {:.4f} nats / {:.4f} bits'.format(
                model_id,
                epoch, mean_epoch_train_loss,
                mean_epoch_train_loss / np.log(2)))
            since_start = time.time() - train_start
            print('{} time since start: {:.1f} secs'.format(model_id, since_start))

        train_losses.append(mean_epoch_train_loss)

    since_start = time.time() - train_start
    print('{} time since start: {:.1f} secs'.format(model_id, since_start))

    print(f'Training {model_id} done; evaluating likelihood on full data:')
    all_losses = RunEpoch('test',
                          model,
                          train_data=train_data,
                          val_data=train_data,
                          opt=None,
                          batch_size=1024,
                          log_every=500,
                          table_bits=table_bits,
                          return_losses=True)
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    #assert not os.path.isfile(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'{model_id} saved to {model_path}')

    print(f'{model_id} Free GPU memory after TrainTask {model_id}')
    get_gpu_memory()

    model.eval()

    est = estimators_lib.ProgressiveSampling(model.model_id, model, 
            table,
            config['psample'],
            device=DEVICE,
            shortcircuit=config['column-masking'])

    print(f'{model_id} Free GPU memory after ProgressiveSampling {model_id}')
    get_gpu_memory()

    return est


def TrainBN(model_id, table_path, model_path, seed=0):
    torch.manual_seed(0)
    np.random.seed(0)
    
    print('LoadCustom ...')

    #assert config['dataset'] in ['dmv-tiny', 'dmv', 'custom']
    assert config['dataset'] in ['custom']
    if config['dataset'] == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif config['dataset'] == 'dmv':
        table = datasets.LoadDmv()
    elif config['dataset'] == 'custom':
        table = datasets.LoadCustom(table_path)

    print('LoadCustom done')

    train_data = common.TableDataset(table) 
    
    print('TableDataset done')

    estimator = estimators_lib.BayesianNetwork(train_data,
                                               rows_to_use=10000, #config['bn-samples'],
                                               algorithm='chow-liu',
                                               topological_sampling_order=True,
                                               root=config['bn-root'],
                                               max_parents=2,
                                               use_pgm=True, #XXX following BayesCard
                                               discretize=None, #100 XXX None causes seg fault!! 
                                               discretize_method='equal_freq')
    #estimator.name = str(estimator)

    return estimator
