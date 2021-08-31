# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .offline_algorithm import OfflineAlgorithm
from .stochastic_virtual import StochasticVirtual
from .stochastic_optimistic import StochasticOptimistic
from .stochastic_modified_virtual import StochasticModifiedVirtual
from .stochastic_virtual_ref import StochasticVirtualRef
from .stochastic_single_ref import StochasticSingleRef
from .base import Algorithm, RandomAlgorithm, AlgorithmType

from dataclasses import dataclass, field
from typing import Iterable, List, Union, Optional
import tqdm


@dataclass
class OnlineParams:
    online_type: List[AlgorithmType] = field(
        default_factory=lambda: [AlgorithmType.STOCHASTIC_VIRTUAL]
    )
    N: int = 5  # Here for backward compatibility
    K: int = 1
    reference_rank: Optional[int] = None  # This in only used by Single-Ref
    threshold: Optional[int] = None  # This will be reset in create_online_algorithm
    exhaust: bool = False  # Exhaust K


def create_algorithm(
    online_type: Union[AlgorithmType, List[AlgorithmType]],
    params: OnlineParams = OnlineParams(),
    N: Optional[int] = None,
):
    if isinstance(online_type, AlgorithmType):
        online_type = (online_type,)

    # Here for backward compatibility
    if N is None:
        print(
            "Warning: OnlineParams.N will be removed in future version. You need to directly pass N to the function instead."
        )
        N = params.N

    list_algorithms = []
    for alg_type in online_type:
        if alg_type == AlgorithmType.STOCHASTIC_VIRTUAL:
            algorithm = StochasticVirtual(N, params.K, params.threshold, params.exhaust)
        elif alg_type == AlgorithmType.STOCHASTIC_OPTIMISTIC:
            algorithm = StochasticOptimistic(
                N, params.K, params.threshold, params.exhaust
            )
        elif alg_type == AlgorithmType.STOCHASTIC_MODIFIED_VIRTUAL:
            algorithm = StochasticModifiedVirtual(
                N, params.K, params.threshold, params.exhaust
            )
        elif alg_type == AlgorithmType.STOCHASTIC_VIRTUAL_REF:
            algorithm = StochasticVirtualRef(
                params.N,
                params.K,
                params.reference_rank,
                params.threshold,
                params.exhaust,
            )
        elif alg_type == AlgorithmType.STOCHASTIC_SINGLE_REF:
            algorithm = StochasticSingleRef(
                N, params.K, params.reference_rank, params.threshold, params.exhaust
            )
        elif alg_type == AlgorithmType.OFFLINE:
            algorithm = OfflineAlgorithm(params.K)
        elif alg_type == AlgorithmType.RANDOM:
            algorithm = RandomAlgorithm(N, params.K)
        else:
            raise ValueError(f"Unknown online algo type: '{alg_type}'.")

        list_algorithms.append(algorithm)

    return list_algorithms


def create_online_algorithm(
    params: OnlineParams = OnlineParams(), N: Optional[int] = None
) -> (Algorithm, Algorithm):
    return create_algorithm(
        [AlgorithmType.OFFLINE] + list(params.online_type), params, N
    )


def compute_indices(
    data_stream: Iterable,
    algorithm_list: Union[Algorithm, List[Algorithm]],
    pbar_flag=False,
) -> Union[Iterable, List[Iterable]]:
    if isinstance(algorithm_list, Algorithm):
        algorithm_list = (algorithm_list,)

    for algorithm in algorithm_list:
        algorithm.reset()

    if pbar_flag:
        pbar = tqdm.tqdm(total=len(data_stream))

    index = 0
    for data in data_stream:
        if not isinstance(data, Iterable):
            data = [data]
        for value in data:
            value = float(value)
            for algorithm in algorithm_list:
                algorithm.action(value, index)
            index += 1

        if pbar_flag:
            pbar.update()

    if pbar_flag:
        pbar.close()

    indices_list = dict(
        (algorithm.name, algorithm.get_indices()) for algorithm in algorithm_list
    )
    return indices_list


def compute_competitive_ratio(
    online_indices: Iterable, offline_indices: Iterable, knapsack=False
) -> int:
    if knapsack:
        online_value = compute_knapsack_online_value(online_indices)
        offline_value = compute_knapsack_online_value(offline_indices)
        comp_ratio = online_value / offline_value
    else:
        online_indices = set([x[1] for x in online_indices])
        offline_indices = set([x[1] for x in offline_indices])
        comp_ratio = len(list(online_indices & offline_indices))
    return comp_ratio


def compute_knapsack_online_value(online_indices: Iterable) -> float:
    if len(online_indices) > 0:
        online_value = sum([x[0] for x in online_indices])
    else:
        online_value = 0.0
    return online_value
