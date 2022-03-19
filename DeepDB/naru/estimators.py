"""A suite of cardinality estimators.

In practicular, inference algorithms for autoregressive density estimators can
be found in 'ProgressiveSampling'.
"""
import bisect
import collections
import json
import operator
import time
import sys
import faulthandler

import numpy as np
import pandas as pd
import torch

import naru.common as common
from naru.common import *
import naru.made as made
import naru.transformer as transformer

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))


def QueryToPredicate(columns, operators, vals, wrap_as_string_cols=None):
    """Converts from (c,o,v) to sql string (for Postgres)."""
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]
    v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if wrap_as_string_cols is not None:
        for i in range(len(columns)):
            if columns[i].name in wrap_as_string_cols:
                v_s[i] = "'" + str(v_s[i]) + "'"

    preds = [
        c.pg_name + ' ' + o + ' ' + v
        for c, o, v in zip(columns, operators, v_s)
    ]
    s = ' and '.join(preds)
    return ' where ' + s


def FillInUnqueriedColumns(table, columns, operators, vals):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


class ProgressiveSampling(CardEst):
    """Progressive sampling."""

    def __init__(
            self,
            model_id,
            model,
            table,
            r,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False # Skip sampling on wildcards?
    ):
        super(ProgressiveSampling, self).__init__()
        torch.set_grad_enabled(False)
        self.model_id = model_id
        self.model = model
        self.table = table
        self.shortcircuit = shortcircuit
        self.scope = list(range(len(table.columns)))

        if r <= 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(1, self.model.nin, device=device))

        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        if 'MADE' in str(model):
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        print(f'{model_id} Setting masked_weight in MADE, do not retrain!')
        for p in model.parameters():
            p.detach_()
            p.requires_grad = False
        self.init_logits.detach_()

        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros)

            # For transformer, need to flatten [num cols, d_model].
            self.inp = self.inp.view(self.num_samples, -1)

    def MoveToCPU(self):
        #print(f'{self.model_id} Free GPU memory before MoveToCPU')
        get_gpu_memory()

        if DEVICE == 'cuda':
            self.init_logits = self.init_logits.to('cpu')
            self.kZeros = self.kZeros.to('cpu')
            self.model = self.model.to('cpu')

        #print(f'{self.model_id} Free GPU memory after MoveToCPU')
        get_gpu_memory()

    def MoveToGPU(self):
        #print(f'{self.model_id} Free GPU memory before MoveToGPU')
        get_gpu_memory()

        if DEVICE == 'cuda':
            self.init_logits = self.init_logits.to(DEVICE)
            self.kZeros = self.kZeros.to(DEVICE)
            self.model = self.model.to(DEVICE)

        #print(f'{self.model_id} Free GPU memory after MoveToGPU')
        get_gpu_memory()


    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        return 'psample_{}'.format(n)

    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  X_feature_scope=None, 
                  X_inverted_features=None,
                  inp=None):
        ncols = len(columns)
        logits = self.init_logits
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []

        #print(f'model.nin: {self.model.nin}')
        #print(f'logits: {logits.shape} {logits}')
        assert self.shortcircuit
        

       
        if X_feature_scope is not None:
            X_multi = torch.ones([num_samples,1], device=self.device)
            #print(f'X_multi shape {X_multi.shape}')

        # Use the query to filter each column's domain.
        valid_i_list = [None] * ncols  # None means all valid.
        for i in range(ncols):
            natural_idx = ordering[i]

            # Column i.
            if vals is None:
                if operators[natural_idx] is None:
                    continue
                #print(f'vals is None, operators {operators[natural_idx]}')
                valid_i = operators[natural_idx]
            else:
                op = operators[natural_idx]
                if op is not None:
                    # There exists a filter.
                    valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                      vals[natural_idx]).astype(np.float32,
                                                                copy=False)
                    
                else:
                    continue

            # This line triggers a host -> gpu copy, showing up as a
            # hotspot in cprofile.
            valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)

        # Fill in wildcards, if enabled.
        if self.shortcircuit:
            for i in range(ncols):
                natural_idx = i if ordering is None else ordering[i]
                if operators[natural_idx] is None and natural_idx != ncols - 1:
                    if natural_idx == 0:
                        self.model.EncodeInput(
                            None,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]])
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                                 1]
                        r = self.model.input_bins_encoded_cumsum[natural_idx]
                        self.model.EncodeInput(None,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r])

        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]

            X_feature_i: bool = X_feature_scope is not None and natural_idx in X_feature_scope

            # If wildcard enabled, 'logits' wasn't assigned last iter.
            if not self.shortcircuit or operators[natural_idx] is not None:
                #logits_i = self.model.logits_for_col(natural_idx, logits, True)
                #print(f'logits_{i}: {logits_i.shape} {logits_i}')
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1)

                valid_i = valid_i_list[i]
                if valid_i is not None:
                    probs_i *= valid_i
                probs_i_summed = probs_i.sum(1)

                masked_probs.append(probs_i_summed)

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
            elif X_feature_i: 
                #logits_i = self.model.logits_for_col(natural_idx, logits, True)
                #print(f'logits_{i}: {logits_i.shape} {logits_i}')
                
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1)
                
                # no need!
                #valid_i = valid_i_list[i] 

            if i < ncols - 1:
                # Num samples to draw for column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples if num_samples else int(
                        self.r * self.dom_sizes[natural_idx])
                #print(f'num_{i}: {num_i}')

                if self.shortcircuit and operators[natural_idx] is None:
                    data_to_encode = None
                    if X_feature_i:
                        #print(f'probs_{i}: {probs_i.shape}, {probs_i}')
                        samples_i = torch.multinomial(
                            probs_i, num_samples=num_i,
                            replacement=True)  # [bs, num_i]
                else:
                    #print(f'probs_{i}: {probs_i.shape}, {probs_i}')
                    samples_i = torch.multinomial(
                        probs_i, num_samples=num_i,
                        replacement=True)  # [bs, num_i]
                    data_to_encode = samples_i.view(-1, 1)

                if X_feature_i: 
                    feature_idx = X_feature_scope.index(natural_idx)
                    #print(f'samples_{i}: {samples_i.shape}')
                    #print(f'all_distinct_values: {columns[natural_idx].all_distinct_values.shape}')
                    indices = samples_i.squeeze()
                    values  = torch.from_numpy(columns[natural_idx].all_distinct_values).to(self.device)
                    #print(f'{self.model_id} Free GPU memory after values')
                    #get_gpu_memory()


                    #print(f'indices: {indices.shape}')
                    #print(f'values: {values.shape}')
                    X_new = torch.index_select(values, 0, indices)
                    #print(f'X_new: {X_new.shape}')
                    if X_inverted_features[feature_idx]:
                        X_new.pow_(-1)
                    #print(f'X_new: {X_new}')
                    X_multi *= X_new.view(-1, 1)
                    if i == 0:
                        assert samples_i.shape[1] == num_samples
                    else:
                        assert samples_i.shape[0] == num_samples
                #print(f'inp: {inp.shape} {inp}')

                # Encode input: i.e., put sampled vars into input buffer.
                if data_to_encode is not None:  # Wildcards are encoded already.
                    if not isinstance(self.model, transformer.Transformer):
                        if natural_idx == 0:
                            self.model.EncodeInput(
                                data_to_encode,
                                natural_col=0,
                                out=inp[:, :self.model.
                                        input_bins_encoded_cumsum[0]])
                        else:
                            l = self.model.input_bins_encoded_cumsum[natural_idx
                                                                     - 1]
                            r = self.model.input_bins_encoded_cumsum[
                                natural_idx]
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                    else:
                        assert False
                        # Transformer.  Need special treatment due to
                        # right-shift.
                        l = (natural_idx + 1) * self.model.d_model
                        r = l + self.model.d_model
                        if i == 0:
                            # Let's also add E_pos=0 to SOS (if enabled).
                            # This is a no-op if disabled pos embs.
                            self.model.EncodeInput(
                                data_to_encode,  # Will ignore.
                                natural_col=-1,  # Signals SOS.
                                out=inp[:, :self.model.d_model])

                        if transformer.MASK_SCHEME == 1:
                            # Should encode natural_col \in [0, ncols).
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                        elif natural_idx < self.model.nin - 1:
                            # If scheme is 0, should not encode the last
                            # variable.
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                #else:
                #    print(f'wildcard!')

                # Actual forward pass.
                next_natural_idx = i + 1 if ordering is None else ordering[i + 1]
                if self.shortcircuit and operators[next_natural_idx] is None:
                    if X_feature_scope is None or next_natural_idx not in X_feature_scope:
                        # If next variable in line is wildcard, then don't do
                        # this forward pass.  Var 'logits' won't be accessed.
                        continue

                if hasattr(self.model, 'do_forward'):
                    # With a specific ordering.
                    assert False
                    logits = self.model.do_forward(inp, ordering)
                else:
                    if self.traced_fwd is not None:
                        assert False
                        logits = self.traced_fwd(inp)
                    else:
                        logits = self.model.forward_with_encoded_input(inp)
                #print(f'logits_{i+1} pre: {logits}')

        # Doing this convoluted scheme because m_p[0] is a scalar, and
        # we want the corret shape to broadcast.
        #print(f'masked_probs: {masked_probs}')
        if len(masked_probs) == 0:
            return 1
        elif len(masked_probs) == 1:
            p = masked_probs[0]
        else:
            p = masked_probs[1]
            for ls in masked_probs[2:]:
                #print(f'masked_probs shape {ls.shape}')
                p *= ls
            p *= masked_probs[0]

        #normalize = p.mean().item()
        #print(f'X_multi shape: {X_multi.shape}')
        #print(f'X_multi: {X_multi}')
        
        if len(p) == 1:
            return p.item()*X_multi.mean().item()
        p *= X_multi.view(1,-1).squeeze()

        return p.mean().item() #/ normalize

    def Query(self, columns, operators, vals=None, X_feature_scope=None, X_inverted_features=None, num_samples=None):
        assert num_samples is not None
        # XXX move from init to this
        with torch.no_grad():
            self.num_samples = num_samples
            self.kZeros = torch.zeros(num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros)

            # For transformer, need to flatten [num cols, d_model].
            self.inp = self.inp.view(num_samples, -1)

        # Massages queries into natural order.
        if vals is not None:
            columns, operators, vals = FillInUnqueriedColumns(
                    self.table, columns, operators, vals)

        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)
        #print(f'orderings: {orderings}')

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * len(columns)
        for natural_idx in range(len(columns)):
            inv_ordering[ordering[natural_idx]] = natural_idx

        with torch.no_grad():
            inp_buf = self.inp.zero_()
            # Fast (?) path.
            if num_orderings == 1:
                ordering = orderings[0]
                self.OnStart()
                p = self._sample_n(
                    self.num_samples,
                    ordering if isinstance(
                        self.model, transformer.Transformer) else inv_ordering,
                    columns,
                    operators,
                    vals,
                    inp=inp_buf,
                    X_feature_scope=X_feature_scope,
                    X_inverted_features=X_inverted_features)
                self.OnEnd()
                #return np.ceil(p * self.cardinality).astype(dtype=np.int32,
                #                                            copy=False)
                return p

            # Num orderings > 1.
            assert False


class BayesianNetwork(CardEst):
    """Progressive sampling with a pomegranate bayes net."""

    def build_discrete_mapping(self, table, discretize, discretize_method):
        assert discretize_method in ["equal_size",
                                     "equal_freq"], discretize_method
        self.max_val = [0] * len(table[0]) 
        self.mean_val = [[]] * len(table[0])
        self.assignments = [None] * len(table[0])
        #if not discretize:
        #    return {}
        table = table.copy()
        #mapping = {}
        for col_id in range(len(table[0])):
            col = table[:, col_id]
            if max(col) > discretize:
                if discretize_method == "equal_size":
                    denom = (max(col) + 1) / discretize
                    #fn = lambda v: np.floor(v / denom)
                elif discretize_method == "equal_freq":
                    self.assignments[col_id] = {} 
                    per_bin = len(col) // discretize
                    counts = collections.defaultdict(int)
                    for x in col:
                        counts[int(x)] += 1
                    i = 0 # max bin
                    exp = 0
                    bin_size = 0
                    for k, count in sorted(counts.items()):
                        if bin_size > 0 and bin_size + count >= per_bin:
                            self.mean_val[col_id].append(exp / count if exp > 0 else 0)
                            exp = 0
                            bin_size = 0
                            i += 1
                        self.assignments[col_id][k] = i
                        self.max_val[col_id] = i
                        exp += count * k
                        bin_size += count
                    self.assignments[col_id] = np.array(
                        [self.assignments[col_id][i] for i in range(int(max(col) + 1))])
            else:
                self.max_val[col_id] = max(col)

    def apply_discrete_mapping(self, table):
        if self.discretize is not None:
            table = table.copy()
            for col_id in range(len(table[0])):
                if self.assignments[col_id] is not None:
                    #fn = discrete_mapping[col_id]
                    fn = lambda v: self.assignments[col_id][v] 
                    table[:, col_id] = fn(table[:, col_id].astype(np.int32))
        return table

    def apply_discrete_mapping_to_value(self, value, col_id):
        if self.assignments[col_id] is None:
            return value
        return self.assignments[col_id][value.astype(np.int32)]

    def __init__(self,
                 dataset,
                 rows_to_use,
                 algorithm="greedy",
                 max_parents=-1,
                 topological_sampling_order=True,
                 use_pgm=True,
                 discretize=None,
                 discretize_method="equal_size",
                 root=None):
        CardEst.__init__(self)

        from pomegranate import BayesianNetwork
        self.discretize = discretize
        self.discretize_method = discretize_method
        self.dataset = dataset
        self.original_table = self.dataset.tuples.numpy()
        self.algorithm = algorithm
        self.topological_sampling_order = topological_sampling_order
        self.rows_to_use = rows_to_use 
        if discretize is not None:
            self.build_discrete_mapping(self.original_table, discretize, discretize_method)
        self.discrete_table = self.apply_discrete_mapping(self.original_table)
        print(f'table {self.discrete_table.shape} {self.discrete_table} algorithm {self.algorithm} max_parents {max_parents} root {root}')
        print('calling BayesianNetwork.from_samples...', end='')
        sys.stdout.flush()
        #XXX seg fault!!!
        faulthandler.enable()
        self.scope = list(range(len(dataset.table.columns)))
        #XXX state_names should be strings in order to easily run inference.query(...)
        self.node_names = [str(i) for i in self.scope]
        t = time.time()
        if len(self.discrete_table) <= self.rows_to_use:
            self.model = BayesianNetwork.from_samples(X=self.discrete_table,
                                                      algorithm=self.algorithm,
                                                      state_names=self.node_names,
                                                      max_parents=max_parents,
                                                      n_jobs=8,
                                                      root=root)
        else:
            idx = np.random.randint(len(self.discrete_table), size=self.rows_to_use)
            self.model = BayesianNetwork.from_samples(X=self.discrete_table[idx,:],
                                                      algorithm=self.algorithm,
                                                      state_names=self.node_names,
                                                      max_parents=max_parents,
                                                      n_jobs=8,
                                                      root=root)
        
        print('done, took', time.time() - t, 'secs.')
        sys.stdout.flush()

        self.use_pgm = use_pgm

        if topological_sampling_order:
            self.sampling_order = []
            while len(self.sampling_order) < len(self.model.structure):
                for i, deps in enumerate(self.model.structure):
                    if i in self.sampling_order:
                        continue  # already ordered
                    if all(d in self.sampling_order for d in deps):
                        self.sampling_order.append(i)
                print("Building sampling order", self.sampling_order)
        else:
            self.sampling_order = list(range(len(self.model.structure)))
        print("Using sampling order", self.sampling_order)
        sys.stdout.flush()

        if use_pgm:
            from pgmpy.models import BayesianModel
            data = pd.DataFrame(self.discrete_table.astype(np.int64), columns=self.node_names)
            spec = []
            orphans = []
            for i, parents in enumerate(self.model.structure):
                assert type(i) == int
                for p in parents:
                    assert type(p) == int
                    spec.append((self.node_names[p], self.node_names[i]))
                    #spec.append((p, i))
                if not parents:
                    orphans.append(self.node_names[i])
                    #orphans.append(i)
            print("Model spec", spec)
            model = BayesianModel(spec)
            for o in orphans:
                model.add_node(o)
            print('calling pgm.BayesianModel.fit...', end='')
            t = time.time()
            model.fit(data)
            print('done, took', time.time() - t, 'secs.')
            sys.stdout.flush()
            self.model = model

    def __str__(self):
        return "bn-{}-{}-{}-{}-bytes-{}-{}-{}".format(
            self.algorithm,
            self.rows_to_use,
            "topo" if self.topological_sampling_order else "nat",
            self.size,
            self.json_size,
            self.discretize,
            self.discretize_method if self.discretize else "na",
            "pgmpy" if self.use_pgm else "pomegranate")


    def Query(self, columns, operators, vals=None, X_feature_scope=None, X_inverted_features=None, num_samples=None):
        if vals is not None:
            if len(columns) != len(self.dataset.table.columns):
                columns, operators, vals = FillInUnqueriedColumns(
                        self.dataset.table, columns, operators, vals)

        self.OnStart()
        ncols = len(columns)
        nrows = self.discrete_table.shape[0]
        assert ncols == self.discrete_table.shape[1], (
            ncols, self.discrete_table.shape)

        def draw_conditional_pgm(evidence, col_id):
            """PGM version of draw_conditional()"""

            if operators[col_id] is None:
                op = None
                val = None
            else:
                if vals is not None:
                    op = OPS[operators[col_id]]
                    val = self.apply_discrete_mapping_to_value(
                        self.dataset.table.val_to_bin_funcs[col_id](vals[col_id]),
                        col_id)
                    if self.discretize:
                        # avoid some bad cases
                        if val == 0 and operators[col_id] == "<":
                            val += 1
                        elif val == self.max_val[col_id] and operators[
                                col_id] == ">":
                            val -= 1
                else:
                    #XXX convert to bin values
                    # here, only consider valid ones
                    if self.discretize:
                        operators[col_id] = [self.apply_discrete_mapping_to_value(
                            self.dataset.table.val_to_bin_funcs[col_id](v),
                            col_id)
                            for v in operators[col_id]]
                        op = operators[col_id] = set(operators[col_id])
                    else:
                        op = operators[col_id]

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                #print(f'distribution: {len(distribution)}, {distribution}')
                #print(f'all_distinct_values: {len(self.dataset.table.columns[col_id].all_distinct_values)}')
                #print(f'max_val: {self.max_val[col_id]}') 
                assert len(distribution) == len(self.dataset.table.columns[col_id].all_distinct_values) \
                        or len(distribution) == self.max_val[col_id] + 1

                for k, v in enumerate(distribution):

                    if vals is None:
                        if k in op: 
                            p += v
                    else:
                        if op(k, val):
                            p += v
                return p

            from pgmpy.inference import VariableElimination
            model_inference = VariableElimination(self.model)
            xi_distribution = []
            for row in evidence:
                e = {}
                for i, v in enumerate(row):
                    if v is not None:
                        e[str(i)] = v
                #print(f'order = {self.sampling_order}, col_id = {col_id}, evidence = {e}')
                if col_id > 0:
                    result = model_inference.query(variables=[str(col_id)], evidence=e, show_progress=False) #XXX error
                else:
                    result = model_inference.query(variables=[str(col_id)], show_progress=False)
                #xi_distribution.append(result[col_id].values)
                xi_distribution.append(result.values)

            xi_marginal = [prob_match(d) for d in xi_distribution]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in enumerate(d):
                    if not op:
                        keys.append(k)
                        prob.append(p)
                    else:
                        if vals is None:
                            if k in operators[col_id]:
                                keys.append(k)
                                prob.append(p)
                        elif op(k, val):
                            keys.append(k)
                            prob.append(p)

                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        def draw_conditional(evidence, col_id):
            """Draws a new value x_i for the column, and returns P(x_i|prev).
            Arguments:
                evidence: shape [BATCH, ncols] with None for unknown cols
                col_id: index of the current column, i
            Returns:
                xi_marginal: P(x_i|x0...x_{i-1}), computed by marginalizing
                    across the range constraint
                match_rows: the subset of rows from filtered_rows that also
                    satisfy the predicate at column i.
            """

            if operators[col_id] is None:
                op = None
                val = None
            else:
                if vals is not None:
                    op = OPS[operators[col_id]]
                    val = self.apply_discrete_mapping_to_value(
                        self.dataset.table.val_to_bin_funcs[col_id](vals[col_id]),
                        col_id)
                    if self.discretize:
                        # avoid some bad cases
                        if val == 0 and operators[col_id] == "<":
                            val += 1
                        elif val == self.max_val[col_id] and operators[
                                col_id] == ">":
                            val -= 1
                else:
                    if self.discretize:
                        operators[col_id] = [self.apply_discrete_mapping_to_value(
                            self.dataset.table.val_to_bin_funcs[col_id](v),
                            col_id)
                            for v in operators[col_id]]
                        operators[col_id] = set(operators[col_id])


            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in distribution.items():
                    if vals is None:
                        if val in operators[col_id]:
                            p += v
                    else:
                        if op(k, val):
                            p += v
                return p

            xi_distribution = self.model.predict_proba(evidence,
                                                       max_iterations=1,
                                                       n_jobs=-1)
            xi_marginal = [
                prob_match(d[col_id].parameters[0]) for d in xi_distribution
            ]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in d[col_id].parameters[0].items():
                    if not op:
                        keys.append(k)
                        prob.append(p)
                    else:
                        if vals is None:
                            if k in bin_valid_val:
                                keys.append(k)
                                prob.append(p)
                        elif op(k, val):
                            keys.append(k)
                            prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        p_estimates = [1. for _ in range(num_samples)]
        evidence = [[None] * ncols for _ in range(num_samples)]
        for col_id in self.sampling_order:
            if self.use_pgm:
                xi_marginal, xi_samples = draw_conditional_pgm(evidence, col_id)
            else:
                xi_marginal, xi_samples = draw_conditional(evidence, col_id)
            
            for ev_list, xi in zip(evidence, xi_samples):
                ev_list[col_id] = xi
            if X_feature_scope is not None and col_id in X_feature_scope:
                feature_idx = X_feature_scope.index(col_id)
                for i in range(num_samples):
                    if self.discretize:
                        X = self.mean_val[col_id][xi_samples[i]]
                        p_estimates[i] *= xi_marginal[i] * (1. / X if X_inverted_features[feature_idx] else X)
                    else:
                        p_estimates[i] *= xi_marginal[i] * (1. / xi_samples[i] if X_inverted_features[feature_idx] else xi_samples[i])
            else:
                for i in range(num_samples):
                    p_estimates[i] *= xi_marginal[i]

        self.OnEnd()
        #return int(np.mean(p_estimates) * nrows)
        return np.mean(p_estimates)
