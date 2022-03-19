import logging
from time import perf_counter

import numpy as np
from spn.algorithms.Inference import likelihood
from spn.structure.Base import Product

from rspn.code_generation.convert_conditions import convert_range
from rspn.structure.base import Sum
from rspn.algorithms.ranges import NominalRange, NumericRange

logger = logging.getLogger(__name__)


def expectation_spn(spn, feature_scope, inverted_features, ranges, node_expectation=None, node_likelihoods=None,
                use_generated_code=False, spn_id=None, meta_types=None, gen_code_stats=None):
    """Compute the Expectation:
        E[1_{conditions} * X_feature_scope]
        First factor is one if condition is fulfilled. For the second factor the variables in feature scope are
        multiplied. If inverted_features[i] is True, variable is taken to denominator.
        The conditional expectation would be E[1_{conditions} * X_feature_scope]/P(conditions)
    """

    # evidence_scope = set([i for i, r in enumerate(ranges) if not np.isnan(r)])
    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    if len(relevant_scope) == 0:
        return np.ones((ranges.shape[0], 1))

    if ranges.shape[0] == 1:

        applicable = True
        if use_generated_code:
            boolean_relevant_scope = [i in relevant_scope for i in range(len(meta_types))]
            boolean_feature_scope = [i in feature_scope for i in range(len(meta_types))]
            applicable, parameters = convert_range(boolean_relevant_scope, boolean_feature_scope, meta_types, ranges[0],
                                                   inverted_features)

        # generated C++ code
        if use_generated_code and applicable:
            time_start = perf_counter()
            import optimized_inference

            spn_func = getattr(optimized_inference, f'spn{spn_id}')
            result = np.array([[spn_func(*parameters)]])

            time_end = perf_counter()

            if gen_code_stats is not None:
                gen_code_stats.calls += 1
                gen_code_stats.total_time += (time_end - time_start)

            # logger.debug(f"\t\tGenerated Code Latency: {(time_end - time_start) * 1000:.3f}ms")
            return result

        # lightweight non-batch version
        else:
            depth = 0 
            est = expectation_recursive(spn, feature_scope, inverted_features, relevant_scope, evidence,
                                        node_expectation, node_likelihoods,depth)
            return np.array([[est]])
    # full batch version
    assert False
    return expectation_recursive_batch(spn, feature_scope, inverted_features, relevant_scope, evidence,
                                       node_expectation, node_likelihoods)


def expectation_nc(nc, feature_scope, inverted_features, ranges, node_expectation=None, node_likelihoods=None,
                use_generated_code=False, spn_id=None, meta_types=None, gen_code_stats=None):
    """Compute the Expectation:
        E[1_{conditions} * X_feature_scope]
        First factor is one if condition is fulfilled. For the second factor the variables in feature scope are
        multiplied. If inverted_features[i] is True, variable is taken to denominator.
        The conditional expectation would be E[1_{conditions} * X_feature_scope]/P(conditions)
    """

    assert len(nc.scope) == len(nc.table.columns) == len(ranges[0])

    # evidence_scope = set([i for i, r in enumerate(ranges) if not np.isnan(r)])
    logger.debug(f"NC Expectation with feature_scope {feature_scope}, inverted_features {inverted_features}, ranges {ranges}, spn_id {spn_id}")
    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    if len(relevant_scope) == 0:
        return np.ones((ranges.shape[0], 1))

    # convert ranges to cols, ops, vals
    cols = nc.table.columns
    if ranges.shape[0] == 1:
        assert len(cols) == len(ranges[0])
        valids = [None] * len(cols)
        for i in range(len(cols)):
            if ranges[0][i] is not None:
                rang = ranges[0][i]
                r = rang.get_ranges()
                dvs = cols[i].all_distinct_values
                if isinstance(rang, NominalRange):
                    #logger.debug(f'isin {r}')
                    valids[i] = np.isin(dvs, r).astype(np.float32, copy=False) 
                else:
                    valid = np.zeros_like(dvs, np.bool)
                    for k, s in enumerate(r):
                        left  = s[0]
                        right = s[1]
                        lower_idx = np.searchsorted(dvs, left, side='left')
                        higher_idx = np.searchsorted(dvs, right, side='right')
                        for j in np.arange(lower_idx, higher_idx):
                            #TODO change iteration to masking
                            if dvs[j] == rang.null_value:
                                continue
                            if dvs[j] == left  and not rang.inclusive_intervals[k][0]:
                                continue
                            if dvs[j] == right and not rang.inclusive_intervals[k][1]:
                                continue
                            valid[j] = True
                    #logger.debug(f'valid: {valids[i]}')
                    valids[i] = valid.astype(np.float32, copy=False)
                if not valids[i].any():
                    logger.debug(f'INVALID RANGE DETECTED')
                    return np.array([[0.]])


        # call NARU's Query
        nc.MoveToGPU()
        # valids as operators, vals = None
        est = nc.Query(cols, valids, X_feature_scope=feature_scope, X_inverted_features=inverted_features, num_samples=2048) 
        nc.MoveToCPU()

        return np.array([[est]])

        # scan and get expectation
        ex = 0
        for _, row in nc.table.data.iterrows():
            valid = True
            for i in range(len(nc.table.columns)):
                rang = ranges[0][i]
                if rang is not None:
                    # if this value is null
                    if row[i] == rang.null_value:
                        valid = False
                        break
                    if not rang.in_range(row[i]):
                        valid = False
                        break
            if valid:
                prod = 1
                for i,f in enumerate(feature_scope):
                    prod *= pow(row[f], -1) if inverted_features[i] else row[f]
                ex += prod
        ex = ex / nc.table.cardinality

        logger.debug(f'NC Expectation scan result is {ex}')
        return np.array([[ex]])

    assert False

def expectation_bn(bn, feature_scope, inverted_features, ranges, node_expectation=None, node_likelihoods=None,
                use_generated_code=False, spn_id=None, meta_types=None, gen_code_stats=None):
    """Compute the Expectation:
        E[1_{conditions} * X_feature_scope]
        First factor is one if condition is fulfilled. For the second factor the variables in feature scope are
        multiplied. If inverted_features[i] is True, variable is taken to denominator.
        The conditional expectation would be E[1_{conditions} * X_feature_scope]/P(conditions)
    """

    # evidence_scope = set([i for i, r in enumerate(ranges) if not np.isnan(r)])
    logger.debug(f"BN Expectation with feature_scope {feature_scope}, inverted_features {inverted_features}, ranges {ranges}, node_expectation {node_expectation}, node_likelihoods {node_likelihoods}, use_generated_code {use_generated_code}, spn_id {spn_id}, meta_types {meta_types}, gen_code_stats {gen_code_stats}")
    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    if len(relevant_scope) == 0:
        return np.ones((ranges.shape[0], 1))

    # convert ranges to cols, ops, vals
    cols = bn.dataset.table.columns
    if ranges.shape[0] == 1:
        logger.debug(f'bn scope {bn.scope}, cols {len(cols)}, ranges {len(ranges[0])}')
        assert len(cols) == len(ranges[0])
        valids = [None] * len(cols)
        for i in range(len(cols)):
            if ranges[0][i] is not None:
                rang = ranges[0][i]
                r = rang.get_ranges()
                dvs = cols[i].all_distinct_values
                if isinstance(rang, NominalRange):
                    logger.debug(f'isin {r}')
                    valids[i] = set(r)
                    logger.debug(f'valid: {valids[i]}')
                else:
                    valid = set() 
                    for k, s in enumerate(r):
                        left  = s[0]
                        right = s[1]
                        lower_idx = np.searchsorted(dvs, left, side='left')
                        higher_idx = np.searchsorted(dvs, right, side='right')
                        for j in np.arange(lower_idx, higher_idx):
                            if dvs[j] == rang.null_value:
                                continue
                            if dvs[j] == left  and not rang.inclusive_intervals[k][0]:
                                continue
                            if dvs[j] == right and not rang.inclusive_intervals[k][1]:
                                continue
                            valid.add(dvs[j])
                    valids[i] = valid
                    logger.debug(f'valid: {valids[i]}')
                if len(valids[i]) == 0:
                    logger.debug(f'INVALID RANGE DETECTED')


        # call NARU's Query
        # valids as operators, vals = None
        est = bn.Query(cols, valids, X_feature_scope=feature_scope, X_inverted_features=inverted_features, num_samples=2048) 
        logger.debug(f'BN Expectation result is {est}')

        return np.array([[est]])

    assert False


def expectation_recursive_batch(node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                                node_likelihoods):
    if isinstance(node, Product):

        llchildren = np.concatenate(
            [expectation_recursive_batch(child, feature_scope, inverted_features, relevant_scope, evidence,
                                         node_expectation, node_likelihoods)
             for child in node.children if
             len(relevant_scope.intersection(child.scope)) > 0], axis=1)
        return np.nanprod(llchildren, axis=1).reshape(-1, 1)

    elif isinstance(node, Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.full((evidence.shape[0], 1), np.nan)

        llchildren = np.concatenate(
            [expectation_recursive_batch(child, feature_scope, inverted_features, relevant_scope, evidence,
                                         node_expectation, node_likelihoods)
             for child in node.children], axis=1)

        relevant_children_idx = np.where(np.isnan(llchildren[0]) == False)[0]
        if len(relevant_children_idx) == 0:
            return np.array([np.nan])

        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        b = np.array(node.weights)[relevant_children_idx] / weights_normalizer

        return np.dot(llchildren[:, relevant_children_idx], b).reshape(-1, 1)

    else:
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((evidence.shape[0], 1))

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                exps[:] = node_expectation[t_node](node, evidence, inverted=inverted)
                return exps
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return likelihood(node, evidence, node_likelihood=node_likelihoods)


def nanproduct(product, factor):
    if np.isnan(product):
        if not np.isnan(factor):
            return factor
        else:
            return np.nan
    else:
        if np.isnan(factor):
            return product
        else:
            return product * factor


def expectation_recursive(node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                          node_likelihoods,depth):
    depth += 1
    if isinstance(node, Product):

        product = np.nan
        for child in node.children:
            if len(relevant_scope.intersection(child.scope)) > 0:
                factor = expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                               node_expectation, node_likelihoods,depth)
                product = nanproduct(product, factor)
        return product

    elif isinstance(node, Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.nan

        llchildren = [expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods,depth)
                      for child in node.children]

        relevant_children_idx = np.where(np.isnan(llchildren) == False)[0]

        if len(relevant_children_idx) == 0:
            return np.nan

        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        weighted_sum = sum(node.weights[j] * llchildren[j] for j in relevant_children_idx)
        return weighted_sum / weights_normalizer

    else:
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]
                return node_expectation[t_node](node, evidence, inverted=inverted).item()
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return node_likelihoods[type(node)](node, evidence).item()
