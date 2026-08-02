import torch
import torch.nn as nn
from tqdm.auto import trange
import os
import logging
import math
import time
from .config import *
from any_precision.analyzer.analyzer import ModelAnalyzer
from typing import List, Tuple, Sequence, Dict
from itertools import chain
from tqdm import tqdm
try:
    from transformers import Gemma3Config
except ImportError:
    class Gemma3Config:
        pass


@torch.no_grad()
def get_inps(
    analyzer: ModelAnalyzer,
    data: Sequence,
    model_seqlen: int,
    devices: List[torch.device],
    offload_activations: bool,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    logging.info("Catching layer inputs from data")
    layers = analyzer.get_layers()
    model = analyzer.model
    device = devices[0] if not offload_activations else torch.device("cpu")

    if isinstance(data, torch.Tensor) and data.shape[0] == 1:  # given a single long tensor, split it into sequences
        assert data.ndim == 2, "data must be either a single tensor with a long sequence or a list of pre-cut sequences"
        num_sequences, num_tokens_dropped = data.numel() // model_seqlen, data.numel() % model_seqlen
        data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
        print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")
        del num_sequences, num_tokens_dropped

    assert all(sequence.shape[1] == model_seqlen for sequence in data)
    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    if isinstance(model.config, Gemma3Config):
        hidden_size = model.config.text_config.hidden_size
    else:
        assert hasattr(model.config, "hidden_size"), f"Model config has no hidden_size: {model.config}"
        hidden_size = model.config.hidden_size

    inps = [
        torch.zeros(
            (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, hidden_size),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
        )
        for i in range(len(devices))
    ]
    if isinstance(model.config, Gemma3Config):
        forward_arg_names = ["attention_mask", "position_ids", "position_embeddings_global", "position_embeddings_local", "cache_position"]
    else:
        forward_arg_names = ["attention_mask", "position_ids", "position_embeddings"]

    cache = {"i": 0}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in tqdm(data, desc="Catching layer inputs from data"):
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers

    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    torch.cuda.empty_cache()
    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps), "internal error: found empty rows in inps"
    return inps, forward_args


@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, is_after_quant: bool, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    :param layer: transformer layer with one or more linear layer to be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :note: outs_tensor must contain previous activations with which to compute MSE loss
    :param compute_mse: if True, return a list of sample-wise mse losses; if False, return an empty sequence
    :param is_after_quant: if True, calculate outputs after quantization; if False, calculate outputs before quantization
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a list of mean squared errors for each sequence
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    description = "Calculating outputs for next layer"
    for j in trange(len(inps_tensor), desc=description, leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses


@torch.no_grad()
def update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    is_after_quant: bool,
    **forward_args,
) -> Sequence[float]:
    """Parallel version of update_outs_and_compute_losses; works on lists of input/output tensors"""
    layer.to(devices[0])
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [update_outs for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], inps[i], outs[i], compute_mse, is_after_quant))
        processed_args = {}
        for k, v in forward_args.items():
            if isinstance(v, torch.Tensor):
                processed_args[k] = v.to(devices[i], non_blocking=True)
            elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                processed_args[k] = tuple(x.to(devices[i], non_blocking=True) for x in v)
            else:
                processed_args[k] = v
        kwargs_by_device.append(processed_args)
    out_losses_by_device: Sequence[Sequence[float]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))


import os
import logging
import time
import torch
import torch.nn as nn
from typing import Sequence, Dict, Any, Optional, Tuple, List
from tqdm import trange


##############################################################################
# 1) SaliencyEngine that stores entire (N, seq_len, G) for that sub-layer
##############################################################################

class SaliencyEngine(nn.Module):
    """
    Holds a (N, seq_len, G) saliency buffer for one sub-layer's entire dataset,
    accumulates X^T X in shape (D, D, G).
    """
    def __init__(
        self,
        in_features: int,
        saliency: torch.Tensor,   # shape (N, seq_len, G)
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.index = 0  # track how many samples we've consumed

        # in_features = layer.weight.shape[1], assuming it's nn.Linear
        num_groups = saliency.shape[-1]

        # store saliency in a buffer
        self.register_buffer("saliencies", saliency.to(dtype))
        self.nsamples = saliency.shape[0]

        # Hessian buffer (D, D, G)
        self.register_buffer(
            "XTX",
            torch.zeros(in_features, in_features, num_groups, dtype=dtype, device=self.device),
        )

    @torch.no_grad()
    def add_batch(self, X: torch.Tensor):
        """
        X: shape [batch_size, seq_len, in_features].

        We'll slice self.saliencies[index : index + batch_size],
        do the einsum => accumulate into self.XTX,
        then index += batch_size.
        """

        bsz = X.shape[0]
        # slice out shape => (bsz, seq_len, G)
        sal_batch = self.saliencies[self.index : self.index + bsz].to(self.device)
        self.index += bsz

        # Flatten
        if X.dim() == 3:
            X = X.reshape(-1, X.shape[-1])  # => [bsz*seq_len, D]
            sal_batch = sal_batch.reshape(-1, sal_batch.shape[-1])  # => [bsz*seq_len, G]

        X = X.to(self.XTX.dtype)
        S = sal_batch.to(self.XTX.dtype)

        sal_weighted_X = torch.einsum("nj,ng->njg", X, S)
        block = torch.einsum("ni,njg->ijg", X, sal_weighted_X)
        if torch.isnan(block).any():
            raise ValueError(f"batch {self.index} XTX is nan")
        else:
            self.XTX.add_(block)
        if torch.isnan(self.XTX).any():
            raise ValueError(f"batch {self.index} XTX is nan")

    def __repr__(self):
        return f"SaliencyEngine(XTX.shape={tuple(self.XTX.shape)}, index/nsamples={self.index}/{self.nsamples})"


class _LayerWrapperThatAccumulatesSaliency(nn.Module):
    """
    Intercepts the sub-layer's forward. On each call:
      1) reads the current batch size from 'input'
      2) calls engine.add_batch(input)
      3) calls real sub-layer forward
    """
    def __init__(self, real_layer: nn.Module, engine: SaliencyEngine):
        super().__init__()
        self.wrapped_layer = real_layer
        self.engine = engine

    def forward(self, input, *args, **kwargs):
        self.engine.add_batch(input)
        return self.wrapped_layer(input, *args, **kwargs)

def print_gpu_usage(message: str):
    import torch
    allocated_memory = torch.cuda.memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"{message}: {allocated_memory / 1024 ** 3:.1f} GB / {total_memory / 1024 ** 3:.1f} GB")


def print_cpu_memory_usage(message: str):
    import psutil
    memory_info = psutil.virtual_memory()
    used_memory = memory_info.used
    total_memory = memory_info.total
    print(f"{message}: {used_memory / 1024 ** 3:.1f} GB / {total_memory / 1024 ** 3:.1f} GB")

##############################################################################
# 2) Single-Device version: init_saliency_engines_single_wrapper
##############################################################################

def init_saliency_engines_single_wrapper(
    layer: nn.Module,
    sublayer_names: List[str],
    inp: torch.Tensor,
    layer_saliencies: Dict[str, torch.Tensor],  # { module_name -> (N, seq_len, G) }
    **forward_args,
) -> Dict[str, SaliencyEngine]:
    """
    Single-device version:
      - For each sublayer_name in sublayer_names, create SaliencyEngine with layer_saliencies[name].
      - Wrap the sub-layer with _LayerWrapperThatAccumulatesSaliency(engine).
      - Forward pass over 'inp' => fill 'out' => accumulate X^T X in each engine's .XTX.
      - Unwrap.
      - Return { sublayer_name -> SaliencyEngine }.
    """

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    layer = layer.to(device)

    # 1) find the actual sub-layers
    found_sublayers = _find_sublayers(layer)
    sublayers = {nm: found_sublayers[nm] for nm in sublayer_names if nm in found_sublayers}

    # 2) Build an engine for each sub-layer
    engines = {}
    for nm, submodule in sublayers.items():
        if nm not in layer_saliencies:
            raise ValueError(f"No saliency found for sublayer '{nm}' in layer_saliencies keys = {list(layer_saliencies.keys())}")
        engine = SaliencyEngine(submodule.weight.shape[1], layer_saliencies[nm], dtype=torch.float32, device=device)
        engines[nm] = engine

    # 3) Wrap sub-layers
    _wrap_sublayers(layer, engines)

    processed_args = {}
    for k, v in forward_args.items():
        if isinstance(v, torch.Tensor):
            processed_args[k] = v.to(device, non_blocking=True)
        elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
            processed_args[k] = tuple(x.to(device, non_blocking=True) for x in v)
        else:
            processed_args[k] = v
    forward_args = processed_args

    with torch.no_grad():
        # 4) Forward pass over 'inp'
        for i in trange(len(inp), desc="capturing saliency weighted XTX", leave=False):
            local_inp = inp[i].to(device).unsqueeze(0)   # => [1, seq_len, D]
            out_batch = layer(local_inp, **forward_args)[0]

    # 5) Unwrap
    _unwrap_sublayers(layer)

    return engines


def init_saliency_engines_parallel_wrapper(
    devices: Sequence[torch.device],
    layer: nn.Module,
    sublayer_names: List[str],
    inps: Sequence[torch.Tensor],
    layer_saliencies_by_device: Sequence[Dict[str, torch.Tensor]],
    **forward_args,
) -> Dict[str, SaliencyEngine]:
    """
    Parallel version. Each device i calls init_saliency_engines_single_wrapper(...) on a replica,
    using sublayer_names + inps[i] + outs[i] + layer_saliencies_by_device[i].
    Then we combine the partial XTX in the main engine (device[0]).
    """
    from torch.nn.parallel import replicate, parallel_apply

    layer.to(devices[0])
    # replicate the layer
    layer_replicas = replicate(layer, devices=devices, detach=True)
    layer_replicas[0] = layer

    funcs = [init_saliency_engines_single_wrapper for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []

    for i, dev in enumerate(devices):
        # We'll pass (layer_replicas[i], sublayer_names, inps[i], layer_saliencies_by_device[i])
        inputs_by_device.append((layer_replicas[i], sublayer_names, inps[i], layer_saliencies_by_device[i]))
        # forward_args -> dev
        dev_kwargs = {}
        for k,v in forward_args.items():
            if isinstance(v, torch.Tensor):
                dev_kwargs[k] = v.to(dev, non_blocking=True)
            elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                dev_kwargs[k] = tuple(x.to(dev, non_blocking=True) for x in v)
            else:
                dev_kwargs[k] = v
        kwargs_by_device.append(dev_kwargs)

    partial_results: List[Dict[str, SaliencyEngine]] = parallel_apply(
        funcs, inputs_by_device, kwargs_by_device, devices=devices
    )
    # partial_results[i] is { sublayer_name -> SaliencyEngine } from device i

    # Merge them on device[0]
    main_engines = partial_results[0]
    for nm, main_engine in main_engines.items():
        total_nsamples = main_engine.nsamples
        for i in range(1, len(devices)):
            eng_i = partial_results[i][nm]
            main_engine.XTX.add_(eng_i.XTX.to(main_engine.device))
            total_nsamples += eng_i.nsamples
        main_engine.nsamples = total_nsamples

    return main_engines

##############################################################################
# 4) Updated "accumulate_saliency_weighted_hessians" using saliency_path
##############################################################################

def accumulate_saliency_weighted_hessians(
    analyzer,
    data: List[torch.Tensor],
    saliency_path: str,             # Path containing l{L}.pt => { sublayer_name -> saliency Tensor }
    output_folder: str,
    num_groups: int
) -> bool:
    """
    1) get_inps(...) => inps, forward_args
    2) Prepare outs
    3) For each layer L in [0..num_layers-1]:
       - update_outs_parallel(...) so 'outs' = layer L's outputs
       - load the saliencies for layer L => a dict { sublayer_name -> (N, seq_len, G) }
         or for multi-GPU, a list of per-device dicts
       - call init_saliency_engines_parallel_wrapper(...) or single_wrapper(...) to accumulate
         the Hessians in .XTX
       - save the final Hessians to output_folder/l{L}.pt
       - swap inps, outs
    
    Returns True if the hessians are already cached, False otherwise.
    """

    if output_folder and os.path.exists(output_folder):
        if all(os.path.exists(os.path.join(output_folder, f"l{i}.pt")) 
                for i in range(len(analyzer.get_layers()))):
            logging.info(f"Cached hessians found in {output_folder}")
            return True

    # 0) Setup devices
    if torch.cuda.is_available():
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]

    model_seqlen = data[0].shape[-1]
    if data[0].dim() == 1:
        data = [d.unsqueeze(0) for d in data]

    os.makedirs(output_folder, exist_ok=True)

    inps, forward_args = get_inps(
        analyzer=analyzer,
        data=data,
        model_seqlen=model_seqlen,
        devices=devices,
        offload_activations=True,
    )

    outs = [torch.zeros_like(inp_tensor) for inp_tensor in inps]

    # 2) Layers
    layers = analyzer.get_layers()
    num_layers = len(layers)

    processed_layers = []
    for l in range(num_layers):
        if os.path.exists(os.path.join(output_folder, f"l{l}.pt")):
            processed_layers.append(l)
        
    logging.info(f"Processed layers: {processed_layers}")

    module_names = analyzer.module_names  # e.g. sub-layer names
    from .utils import get_progress_bar     # adapt if needed
    pb = get_progress_bar(num_layers, "Accumulating saliency Hessians blockwise")

    # 3) Blockwise
    for l in range(num_layers):

        layer = layers[l]
        layer.to(torch.device("cpu"))

        if l in processed_layers:
            logging.info(f"Skipping layer {l} because it has already been processed")
            torch.cuda.empty_cache()

            # # (A) update_outs_parallel => outs
            update_outs_parallel(
                devices=devices,
                layer=layer,
                inps=inps,
                outs=outs,
                compute_mse=False,
                is_after_quant=False,
                **forward_args
            )

            layer.to(torch.device("cpu"))

            # (E) Swap inps, outs
            inps, outs = outs, inps

            torch.cuda.empty_cache()
            layers[l] = None

            pb.update(1)
            continue


        # (B) Load saliencies for layer l
        #     We expect e.g. saliency_path/l{l}.pt => either:
        #       { sublayer_name -> (N, seq_len, G) } in single-GPU
        #     or a list of length len(devices), each is { sublayer_name -> partial saliency } for multi-GPU.
        file_path = os.path.join(saliency_path, f"l{l}.pt")
        # This might be either a dict or a list-of-dicts
        loaded = torch.load(file_path)

        orig_num_groups = list(loaded.values())[0].shape[-1]
        assert orig_num_groups % num_groups == 0, f"orig_num_groups {orig_num_groups} must be divisible by num_groups {num_groups}"
        group_subchannels = orig_num_groups // num_groups
        loaded = {k: v.view(v.shape[0], v.shape[1], num_groups, group_subchannels).mean(dim=-1) for k, v in loaded.items()}

        nsamples = list(loaded.values())[0].shape[0]
        nsamples_per_device = (nsamples - 1) // len(devices) + 1
        assert nsamples_per_device == inps[0].shape[0], f"nsamples_per_device {nsamples_per_device} must match inps[0].shape[0] {inps[0].shape[0]}"

        splitted_loaded = []
        for i in range(len(devices)):
            start = i * nsamples_per_device
            end = min(start + nsamples_per_device, nsamples)
            splitted_loaded.append({k: v[start:end] for k, v in loaded.items()})

        assert list(splitted_loaded[0].values())[0].shape[0] == nsamples_per_device

        # (C) Accumulate Hessians
        if len(devices) == 1:
            # single device
            # loaded must be { sublayer_name -> (N, seq_len, G) }
            saliency_handlers = init_saliency_engines_single_wrapper(
                layer,
                module_names,    # or the sub-layers you actually want
                inps[0],
                splitted_loaded[0],         # => layer_saliencies
                **forward_args
            )
        else:
            # multi-gpu => loaded presumably is a list of length len(devices),
            # each element is { sublayer_name -> partial saliency } for that device
            if not isinstance(splitted_loaded, list) or len(splitted_loaded) != len(devices):
                raise ValueError(f"Expected saliencies for layer {l} to be a list of length {len(devices)}")
            # loaded[i]: { sublayer_name -> (partialN, seq_len, G) }
            saliency_handlers = init_saliency_engines_parallel_wrapper(
                devices,
                layer,
                module_names,
                inps,
                splitted_loaded,  # layer_saliencies_by_device
                **forward_args
            )

        # (D) Save final Hessians => output_folder/l{l}.pt
        result_dict = {}
        for nm, engine in saliency_handlers.items():
            result_dict[nm] = engine.XTX.detach().cpu().float()

        out_file = os.path.join(output_folder, f"l{l}.pt")
        torch.save(result_dict, out_file)
        logging.info(f"[Layer {l}] Saved saliency-weighted Hessians to {out_file}")

        del result_dict


        import gc;gc.collect()
        

        del saliency_handlers
        torch.cuda.empty_cache()

        # # (A) update_outs_parallel => outs
        update_outs_parallel(
            devices=devices,
            layer=layer,
            inps=inps,
            outs=outs,
            compute_mse=False,
            is_after_quant=False,
            **forward_args
        )

        layer.to(torch.device("cpu"))

        # (E) Swap inps, outs
        inps, outs = outs, inps

        torch.cuda.empty_cache()

        layers[l] = None

        pb.update(1)

    pb.close()
    logging.info("Done accumulating saliency-weighted Hessians for all layers.")
    return False


##############################################################################
# 5) Minimal Helper Stubs
##############################################################################

def _find_sublayers(layer: nn.Module) -> Dict[str, nn.Module]:
    """
    Recursively gather { full_name: submodule } for all sub-layers (e.g. nn.Linear).
    Adapt as needed.
    """
    result = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear) and name:
            result[name] = module
    return result

def _wrap_sublayers(layer: nn.Module, engines: Dict[str, SaliencyEngine]):

    for name, submodule in layer.named_modules():
        if name in engines.keys():
            wrapper = _LayerWrapperThatAccumulatesSaliency(submodule, engines[name])
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = getattr(layer, parent_name)
            setattr(parent_module, child_name, wrapper)

def _unwrap_sublayers(layer: nn.Module):

    for name, submodule in layer.named_modules():
        if isinstance(submodule, _LayerWrapperThatAccumulatesSaliency):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = getattr(layer, parent_name)
            setattr(parent_module, child_name, submodule.wrapped_layer)



def _prepare_hnll_tokens_and_labels(tokens: torch.Tensor, pad_token_id: Optional[int] = None):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    labels = tokens.clone()
    attention_mask = None
    if pad_token_id is not None:
        pad_mask = labels.eq(pad_token_id)
        if pad_mask.any():
            labels = labels.masked_fill(pad_mask, -100)
            attention_mask = (~pad_mask).long()

    valid_tokens = int(labels[..., 1:].ne(-100).sum().item())
    return tokens, labels, attention_mask, valid_tokens

def _rademacher_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor).bernoulli_(0.5).mul_(2.0).sub_(1.0)


def reduce_channels_by_group_mean(diag_sample: torch.Tensor, num_groups: int) -> torch.Tensor:
    if diag_sample.shape[-1] % num_groups != 0:
        raise ValueError(
            f"output channels {diag_sample.shape[-1]} must be divisible by num_groups {num_groups}"
        )
    channels_per_group = diag_sample.shape[-1] // num_groups
    return diag_sample.reshape(*diag_sample.shape[:-1], num_groups, channels_per_group).mean(dim=-1)



def _resolve_hnll_curvature_dtype(dtype: str, device: torch.device, current_dtype: torch.dtype) -> torch.dtype:
    if dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return current_dtype
    if dtype == "current":
        return current_dtype
    if dtype == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            logging.warning("Requested HNLL BF16 curvature pass, but CUDA BF16 is not supported; using current dtype.")
            return current_dtype
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported HNLL curvature dtype {dtype!r}; expected auto/current/bf16/fp16/fp32")

def _cuda_sync_if_profiled(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _sdpa_backend_context(backend: str):
    from contextlib import nullcontext
    if backend == "auto" or not torch.cuda.is_available():
        return nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend_map = {
            "math": [SDPBackend.MATH],
            "flash": [SDPBackend.FLASH_ATTENTION],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "cudnn": [SDPBackend.CUDNN_ATTENTION],
        }
        return sdpa_kernel(backend_map[backend])
    except Exception:
        pass

    if backend == "math":
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
        except Exception:
            return nullcontext()

    return nullcontext()


def _sdpa_math_context():
    return _sdpa_backend_context("math")


def _validate_hessian_builder_inputs(x: torch.Tensor, stats: torch.Tensor):
    if x.ndim != 2:
        raise ValueError(f"x must have shape [tokens, input_dim], got {tuple(x.shape)}")
    if stats.ndim != 2:
        raise ValueError(f"stats must have shape [tokens, num_groups], got {tuple(stats.shape)}")
    if x.shape[0] != stats.shape[0]:
        raise ValueError(f"x tokens {x.shape[0]} must match stats tokens {stats.shape[0]}")
    if not torch.isfinite(x).all():
        raise ValueError("x contains non-finite values")
    if not torch.isfinite(stats).all():
        raise ValueError("stats contains non-finite values")
    if (stats < 0).any():
        raise ValueError("negative stats passed to Hessian builder; positive projection must happen upstream")


def build_group_hessians_legacy(x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    _validate_hessian_builder_inputs(x, stats)
    x = x.float()
    stats = stats.float()
    hessians = torch.empty(
        stats.shape[1], x.shape[1], x.shape[1],
        device=x.device, dtype=torch.float32,
    )
    for group_idx in range(stats.shape[1]):
        sqrt_s = torch.sqrt(stats[:, group_idx]).unsqueeze(-1)
        x_weighted = x * sqrt_s
        hessians[group_idx].copy_(x_weighted.T @ x_weighted)
    return hessians


def build_group_hessians_batched(x: torch.Tensor, stats: torch.Tensor, group_chunk_size: int) -> torch.Tensor:
    _validate_hessian_builder_inputs(x, stats)
    if group_chunk_size <= 0:
        raise ValueError(f"group_chunk_size must be positive, got {group_chunk_size}")

    x = x.float()
    stats = stats.float()
    hessians = torch.empty(
        stats.shape[1], x.shape[1], x.shape[1],
        device=x.device, dtype=torch.float32,
    )
    for start in range(0, stats.shape[1], group_chunk_size):
        end = min(start + group_chunk_size, stats.shape[1])
        sqrt_s = torch.sqrt(stats[:, start:end]).transpose(0, 1)
        x_weighted = sqrt_s.unsqueeze(-1) * x.unsqueeze(0)
        h_chunk = torch.bmm(x_weighted.transpose(1, 2), x_weighted)
        hessians[start:end].copy_(h_chunk)
        del sqrt_s, x_weighted, h_chunk
    return hessians


def build_shared_x_group_hessians(x: torch.Tensor, module_group_stats, group_chunk_size: int):
    if x.ndim != 2:
        raise ValueError(f"x must have shape [tokens, input_dim], got {tuple(x.shape)}")
    module_names = list(module_group_stats.keys())
    group_counts = []
    stats_list = []
    for module_name in module_names:
        stats = module_group_stats[module_name]
        _validate_hessian_builder_inputs(x, stats)
        group_counts.append(stats.shape[1])
        stats_list.append(stats.float())

    all_stats = torch.cat(stats_list, dim=1)
    all_hessians = build_group_hessians_batched(x.float(), all_stats, group_chunk_size)
    split_hessians = torch.split(all_hessians, group_counts, dim=0)
    return {module_name: hessian for module_name, hessian in zip(module_names, split_hessians)}


def build_group_hessians_from_curvature(x: torch.Tensor, s_group: torch.Tensor) -> torch.Tensor:
    return build_group_hessians_batched(x, s_group, group_chunk_size=s_group.shape[1])


def _shared_x_semantic_id(layer_idx: int, module_name: str):
    if module_name.endswith("self_attn.q_proj") or module_name.endswith("self_attn.k_proj") or module_name.endswith("self_attn.v_proj"):
        return (layer_idx, "attention_input")
    if module_name.endswith("mlp.gate_proj") or module_name.endswith("mlp.up_proj"):
        return (layer_idx, "mlp_input")
    return (layer_idx, "module_input", module_name)


def _build_hnll_sample_hessians(
    module_keys,
    xs,
    group_accums,
    num_probes: int,
    builder: str,
    group_chunk_size: int,
    validate_shared_x: bool,
):
    if builder not in ("legacy", "batched", "batched_shared_x"):
        raise ValueError(f"Unsupported nll_hessian_builder={builder!r}")
    if builder in ("batched", "batched_shared_x") and group_chunk_size <= 0:
        raise ValueError(f"nll_hessian_group_chunk_size must be positive for {builder}, got {group_chunk_size}")

    x_by_key = {}
    stats_by_key = {}
    for key, x in zip(module_keys, xs):
        raw_s = group_accums[key].div(float(num_probes))
        s_pos = raw_s.clamp_min(0.0)
        x_by_key[key] = x.detach().reshape(-1, x.shape[-1]).float()
        stats_by_key[key] = s_pos

    if builder == "legacy":
        return {key: build_group_hessians_from_curvature(x_by_key[key], stats_by_key[key]).detach().cpu() for key in module_keys}

    if builder == "batched":
        return {
            key: build_group_hessians_batched(x_by_key[key], stats_by_key[key], group_chunk_size).detach().cpu()
            for key in module_keys
        }

    grouped_keys = {}
    for key in module_keys:
        layer_idx, module_name = key
        shared_id = _shared_x_semantic_id(layer_idx, module_name)
        grouped_keys.setdefault(shared_id, []).append(key)

    hessians = {}
    for shared_id, keys in grouped_keys.items():
        base_x = x_by_key[keys[0]]
        can_fuse = len(keys) > 1
        if validate_shared_x and can_fuse:
            for key in keys[1:]:
                if base_x.shape != x_by_key[key].shape or not torch.equal(base_x, x_by_key[key]):
                    logging.warning(f"HNLL shared-X validation failed for {shared_id}; falling back to per-module batched builder")
                    can_fuse = False
                    break

        if not can_fuse:
            for key in keys:
                hessians[key] = build_group_hessians_batched(x_by_key[key], stats_by_key[key], group_chunk_size).detach().cpu()
            continue

        module_group_stats = {key: stats_by_key[key] for key in keys}
        for key, hessian in build_shared_x_group_hessians(base_x, module_group_stats, group_chunk_size).items():
            hessians[key] = hessian.detach().cpu()

    return hessians

def accumulate_nll_ggn_hessians(
    analyzer,
    data: List[torch.Tensor],
    output_folder: str,
    num_groups: int,
    num_probes: int = 1,
    random_state: Optional[int] = None,
    layer_chunk_size: int = 0,
    profile: bool = False,
    curvature_dtype: str = "auto",
    hessian_builder: str = "legacy",
    hessian_group_chunk_size: int = 4,
    validate_shared_x: bool = False,
) -> bool:
    """Accumulate PSD NLL-GGN Hessians using first-order logits-covariance probes.

    The estimator uses u = one_hot(y~p) - p at valid next-token logits, so
    E[u u^T] = Diag(p) - p p^T. Backpropagating logits·u to each target
    linear output Z gives J_Z^T u; squaring estimates diag(J_Z^T C J_Z).
    """
    if num_probes < 1:
        raise ValueError(f"num_probes must be >= 1, got {num_probes}")
    if layer_chunk_size < 0:
        raise ValueError(f"layer_chunk_size must be >= 0, got {layer_chunk_size}")

    layers = analyzer.get_layers()
    num_layers = len(layers)
    if output_folder and os.path.exists(output_folder):
        if all(os.path.exists(os.path.join(output_folder, f"l{i}.pt")) for i in range(num_layers)):
            logging.info(f"Cached NLL-GGN hessians found in {output_folder}")
            return True

    if random_state is not None:
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_folder, exist_ok=True)
    model = analyzer.model
    model.eval()
    old_param = next(model.parameters())
    old_dtype = old_param.dtype
    ggn_dtype = _resolve_hnll_curvature_dtype(curvature_dtype, device, old_dtype)
    logging.info(f"NLL-GGN curvature dtype: {ggn_dtype} (requested={curvature_dtype}, original={old_dtype})")
    model.to(device=device, dtype=ggn_dtype)

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False

    if data and data[0].dim() == 1:
        data = [tokens.unsqueeze(0) for tokens in data]

    processed_layers = {
        layer_idx for layer_idx in range(num_layers)
        if os.path.exists(os.path.join(output_folder, f"l{layer_idx}.pt"))
    }
    effective_layer_chunk_size = num_layers if layer_chunk_size == 0 else layer_chunk_size
    logging.info(f"Processed NLL-GGN layers: {sorted(processed_layers)}")
    logging.info(f"NLL-GGN layer chunk size: {layer_chunk_size} (effective={effective_layer_chunk_size})")
    logging.info(f"NLL-GGN probes: {num_probes}")
    logging.info(f"NLL-GGN profiling: {profile}")
    logging.info(f"NLL-GGN Hessian builder: {hessian_builder}")
    logging.info(f"NLL-GGN Hessian group chunk size: {hessian_group_chunk_size}")
    logging.info(f"NLL-GGN validate shared-X: {validate_shared_x}")

    from .utils import get_progress_bar
    pb = get_progress_bar(num_layers, "Accumulating NLL-GGN Hessians")

    try:
        for batch_start in range(0, num_layers, effective_layer_chunk_size):
            batch_layer_indices = [
                idx for idx in range(batch_start, min(batch_start + effective_layer_chunk_size, num_layers))
                if idx not in processed_layers
            ]
            if not batch_layer_indices:
                for skipped_idx in range(batch_start, min(batch_start + effective_layer_chunk_size, num_layers)):
                    if skipped_idx in processed_layers:
                        pb.update(1)
                continue

            batch_modules = {layer_idx: analyzer.get_modules(layers[layer_idx]) for layer_idx in batch_layer_indices}
            results = {
                layer_idx: {
                    name: torch.zeros(num_groups, module.weight.shape[1], module.weight.shape[1], dtype=torch.float32)
                    for name, module in modules.items()
                }
                for layer_idx, modules in batch_modules.items()
            }
            valid_tokens_total = 0
            hooks = []
            captures = {}

            logging.info(
                f"[NLL-GGN] estimating layers {batch_layer_indices} with {num_probes} logits-covariance probe(s)"
            )

            def make_hook(layer_idx, module_name):
                key = (layer_idx, module_name)
                def hook(_module, inp, out):
                    if not torch.is_tensor(out):
                        raise TypeError(f"NLL-GGN expects tensor output for layer {layer_idx} {module_name}, got {type(out)}")
                    captures[key] = (inp[0], out)
                    return None
                return hook

            for layer_idx, modules in batch_modules.items():
                for module_name, module in modules.items():
                    hooks.append(module.register_forward_hook(make_hook(layer_idx, module_name)))

            try:
                for sample_idx, tokens in enumerate(tqdm(data, desc=f"NLL-GGN layers {batch_layer_indices[0]}-{batch_layer_indices[-1]}", leave=False)):
                    captures.clear()
                    tokens = tokens.to(device)
                    pad_token_id = getattr(analyzer.tokenizer, "pad_token_id", None)
                    tokens, labels, attention_mask, valid_tokens = _prepare_hnll_tokens_and_labels(tokens, pad_token_id)
                    if valid_tokens == 0:
                        logging.warning("Skipping NLL-GGN calibration batch with zero valid next-token labels")
                        continue
                    valid_tokens_total += valid_tokens

                    forward_kwargs = {"input_ids": tokens, "labels": labels}
                    if attention_mask is not None:
                        forward_kwargs["attention_mask"] = attention_mask.to(device)

                    if profile:
                        sample_start = time.perf_counter()
                        _cuda_sync_if_profiled(device, profile)
                        phase_start = time.perf_counter()

                    outputs = model(**forward_kwargs)
                    if not hasattr(outputs, "logits"):
                        raise RuntimeError("NLL-GGN requires model outputs.logits")
                    logits = outputs.logits

                    if profile:
                        _cuda_sync_if_profiled(device, profile)
                        forward_time = time.perf_counter() - phase_start
                        phase_start = time.perf_counter()

                    expected_keys = [
                        (layer_idx, module_name)
                        for layer_idx, modules in batch_modules.items()
                        for module_name in modules.keys()
                    ]
                    missing_keys = [key for key in expected_keys if key not in captures]
                    if missing_keys:
                        raise RuntimeError(f"NLL-GGN did not capture target module outputs: {missing_keys}")

                    module_keys = expected_keys
                    xs = [captures[key][0].detach() for key in module_keys]
                    zs = [captures[key][1] for key in module_keys]
                    group_accums = {
                        key: torch.zeros(
                            z.reshape(-1, z.shape[-1]).shape[0],
                            num_groups,
                            device=z.device,
                            dtype=torch.float32,
                        )
                        for key, z in zip(module_keys, zs)
                    }

                    logits_shift = logits[:, :-1, :]
                    valid_mask = labels[:, 1:].ne(-100)
                    if logits_shift.shape[:2] != valid_mask.shape:
                        raise RuntimeError(
                            f"NLL-GGN logits/label shift mismatch: logits={tuple(logits_shift.shape)}, mask={tuple(valid_mask.shape)}"
                        )
                    flat_probs = torch.softmax(logits_shift.float(), dim=-1).detach().reshape(-1, logits_shift.shape[-1])
                    valid_mask_f = valid_mask.float()

                    for probe_idx in range(num_probes):
                        sampled = torch.multinomial(flat_probs, num_samples=1).reshape(logits_shift.shape[:2])
                        selected = logits_shift.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).float()
                        expected_logit = (logits_shift.float() * flat_probs.reshape_as(logits_shift.float())).sum(dim=-1)
                        probe_scalar = ((selected - expected_logit) * valid_mask_f).sum()
                        grads = torch.autograd.grad(
                            probe_scalar,
                            zs,
                            retain_graph=probe_idx < num_probes - 1,
                            create_graph=False,
                            allow_unused=False,
                        )
                        for key, grad in zip(module_keys, grads):
                            diag_sample = grad.float().square().reshape(-1, grad.shape[-1])
                            if not torch.isfinite(diag_sample).all():
                                raise ValueError(
                                    f"NLL-GGN produced non-finite curvature sample at layers "
                                    f"{batch_layer_indices[0]}-{batch_layer_indices[-1]}, module {key}, probe={probe_idx}."
                                )
                            group_accums[key].add_(reduce_channels_by_group_mean(diag_sample, num_groups))

                    del flat_probs, logits_shift, valid_mask, valid_mask_f, grads, sampled, selected, expected_logit, probe_scalar
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    sample_hessians = _build_hnll_sample_hessians(
                        module_keys,
                        xs,
                        group_accums,
                        num_probes,
                        hessian_builder,
                        hessian_group_chunk_size,
                        validate_shared_x,
                    )
                    for key, hessian in sample_hessians.items():
                        layer_idx, module_name = key
                        if not torch.isfinite(hessian).all():
                            raise ValueError(f"NLL-GGN H contains non-finite values at layer {layer_idx}, module {module_name}")
                        results[layer_idx][module_name].add_(hessian.detach().cpu())

                    if profile:
                        _cuda_sync_if_profiled(device, profile)
                        total_time = time.perf_counter() - sample_start
                        ggn_time = time.perf_counter() - phase_start
                        logging.info(
                            f"[NLL-GGN][profile] layers={batch_layer_indices[0]}-{batch_layer_indices[-1]} "
                            f"sample={sample_idx + 1}/{len(data)} valid_tokens={valid_tokens} "
                            f"forward={forward_time:.3f}s ggn_backward_build={ggn_time:.3f}s total={total_time:.3f}s"
                        )

                    del outputs, logits, group_accums, sample_hessians
            finally:
                for hook in hooks:
                    hook.remove()

            for layer_idx in batch_layer_indices:
                out_file = os.path.join(output_folder, f"l{layer_idx}.pt")
                for module_name, hessian in results[layer_idx].items():
                    _log_hnll_hessian_stats(layer_idx, module_name, hessian, num_probes, valid_tokens_total)
                torch.save(results[layer_idx], out_file)
                logging.info(f"[Layer {layer_idx}] Saved NLL-GGN Hessians to {out_file}")
                pb.update(1)
            del results
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        pb.close()
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if "old_dtype" in locals() and "ggn_dtype" in locals() and ggn_dtype != old_dtype:
            model.to(dtype=old_dtype)

    logging.info("Done accumulating NLL-GGN Hessians for all layers.")
    return False

def _log_hnll_hessian_stats(layer_idx: int, module_name: str, hessian: torch.Tensor, num_probes: int, valid_tokens: int):
    h = hessian.detach().float()
    diag = torch.diagonal(h, dim1=-2, dim2=-1)
    logging.info(
        f"[HNLL][Layer {layer_idx}][{module_name}] valid_tokens={valid_tokens} probes={num_probes} "
        f"H trace={diag.sum(dim=-1).mean().item():.6e} "
        f"fro={torch.linalg.matrix_norm(h).mean().item():.6e} "
        f"diag_min={diag.min().item():.6e} diag_max={diag.max().item():.6e}"
    )


def accumulate_nll_hvp_hessians(
    analyzer,
    data: List[torch.Tensor],
    output_folder: str,
    num_groups: int,
    num_probes: int = 1,
    random_state: Optional[int] = None,
    layer_chunk_size: int = 0,
    profile: bool = False,
    curvature_dtype: str = "auto",
    hessian_builder: str = "legacy",
    hessian_group_chunk_size: int = 4,
    validate_shared_x: bool = False,
    sdpa_backend: str = "math",
    hvp_engine: str = "autograd",
    fd_epsilon: float = 1e-3,
    fd_batched_signs: bool = False,
) -> bool:
    """Accumulate Group-Trace Positive NLL Hessians via autograd HVP.

    This writes the same per-layer cache format consumed by layerwise_quantize.seed:
    output_folder/l{layer}.pt contains {module_name: H} with H shaped [G, D, D].
    `layer_chunk_size=0` uses all target layers in one global HVP chunk. Positive values set a memory-fallback chunk size and amortize the
    full-model forward over one or more target layers by using one joint Hutchinson
    HVP per probe for the selected layers.
    """
    if num_probes < 1:
        raise ValueError(f"num_probes must be >= 1, got {num_probes}")
    if layer_chunk_size < 0:
        raise ValueError(f"layer_chunk_size must be >= 0, got {layer_chunk_size}")
    if hvp_engine not in ("autograd", "finite_diff"):
        raise ValueError(f"Unsupported nll_hvp_engine={hvp_engine!r}")
    if fd_epsilon <= 0:
        raise ValueError(f"fd_epsilon must be positive, got {fd_epsilon}")

    layers = analyzer.get_layers()
    num_layers = len(layers)
    if output_folder and os.path.exists(output_folder):
        if all(os.path.exists(os.path.join(output_folder, f"l{i}.pt")) for i in range(num_layers)):
            logging.info(f"Cached HNLL hessians found in {output_folder}")
            return True

    if random_state is not None:
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logging.warning("HNLL HVP Hessian path uses a single device to avoid cross-device double-backward graphs.")

    os.makedirs(output_folder, exist_ok=True)
    model = analyzer.model
    model.eval()
    old_param = next(model.parameters())
    old_dtype = old_param.dtype
    hnll_dtype = _resolve_hnll_curvature_dtype(curvature_dtype, device, old_dtype)
    logging.info(f"HNLL curvature dtype: {hnll_dtype} (requested={curvature_dtype}, original={old_dtype})")
    model.to(device=device, dtype=hnll_dtype)

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False

    if data and data[0].dim() == 1:
        data = [tokens.unsqueeze(0) for tokens in data]

    processed_layers = {
        layer_idx for layer_idx in range(num_layers)
        if os.path.exists(os.path.join(output_folder, f"l{layer_idx}.pt"))
    }
    logging.info(f"Processed HNLL layers: {sorted(processed_layers)}")
    effective_layer_chunk_size = num_layers if layer_chunk_size == 0 else layer_chunk_size
    logging.info(f"HNLL layer chunk size: {layer_chunk_size} (effective={effective_layer_chunk_size})")
    logging.info(f"HNLL profiling: {profile}")
    logging.info(f"HNLL Hessian builder: {hessian_builder}")
    logging.info(f"HNLL Hessian group chunk size: {hessian_group_chunk_size}")
    logging.info(f"HNLL validate shared-X: {validate_shared_x}")
    logging.info(f"HNLL SDPA backend: {sdpa_backend}")
    logging.info(f"HNLL HVP engine: {hvp_engine}")
    logging.info(f"HNLL finite-difference epsilon: {fd_epsilon}")
    logging.info(f"HNLL finite-difference batched +/- signs: {fd_batched_signs}")

    from .utils import get_progress_bar
    pb = get_progress_bar(num_layers, "Accumulating HNLL HVP Hessians")

    try:
        with _sdpa_backend_context(sdpa_backend):
            for batch_start in range(0, num_layers, effective_layer_chunk_size):
                batch_layer_indices = [
                    idx for idx in range(batch_start, min(batch_start + effective_layer_chunk_size, num_layers))
                    if idx not in processed_layers
                ]
                if not batch_layer_indices:
                    for skipped_idx in range(batch_start, min(batch_start + effective_layer_chunk_size, num_layers)):
                        logging.info(f"Skipping HNLL layer {skipped_idx} because it has already been processed")
                        pb.update(1)
                    continue

                batch_modules = {layer_idx: analyzer.get_modules(layers[layer_idx]) for layer_idx in batch_layer_indices}
                results = {
                    layer_idx: {
                        name: torch.zeros(num_groups, module.weight.shape[1], module.weight.shape[1], dtype=torch.float32)
                        for name, module in modules.items()
                    }
                    for layer_idx, modules in batch_modules.items()
                }
                valid_tokens_total = 0
                hooks = []
                captures = {}
                perturbations = None
                perturbed_captures = {}

                logging.info(
                    f"[HNLL] estimating layers {batch_layer_indices} with global multi-layer HVP "
                    f"and {num_probes} probe(s)"
                )

                def make_hook(layer_idx, module_name):
                    key = (layer_idx, module_name)
                    def hook(_module, inp, out):
                        if not torch.is_tensor(out):
                            raise TypeError(f"HNLL HVP expects tensor output for layer {layer_idx} {module_name}, got {type(out)}")
                        captures[key] = (inp[0], out)
                        if perturbations is not None and key in perturbations:
                            perturbed_out = out + perturbations[key]
                            perturbed_captures[key] = perturbed_out
                            return perturbed_out
                        return None
                    return hook

                for layer_idx, modules in batch_modules.items():
                    for module_name, module in modules.items():
                        hooks.append(module.register_forward_hook(make_hook(layer_idx, module_name)))

                try:
                    for sample_idx, tokens in enumerate(tqdm(data, desc=f"HNLL HVP layers {batch_layer_indices[0]}-{batch_layer_indices[-1]}", leave=False)):
                        captures.clear()
                        tokens = tokens.to(device)
                        pad_token_id = getattr(analyzer.tokenizer, "pad_token_id", None)
                        tokens, labels, attention_mask, valid_tokens = _prepare_hnll_tokens_and_labels(tokens, pad_token_id)
                        if valid_tokens == 0:
                            logging.warning("Skipping HNLL calibration batch with zero valid next-token labels")
                            continue
                        valid_tokens_total += valid_tokens

                        forward_kwargs = {"input_ids": tokens, "labels": labels}
                        if attention_mask is not None:
                            forward_kwargs["attention_mask"] = attention_mask.to(device)
                        if profile:
                            sample_start = time.perf_counter()
                            _cuda_sync_if_profiled(device, profile)
                            phase_start = time.perf_counter()
                        outputs = model(**forward_kwargs)
                        loss_nll = outputs.loss * valid_tokens
                        if profile:
                            _cuda_sync_if_profiled(device, profile)
                            forward_time = time.perf_counter() - phase_start
                            phase_start = time.perf_counter()

                        expected_keys = [
                            (layer_idx, module_name)
                            for layer_idx, modules in batch_modules.items()
                            for module_name in modules.keys()
                        ]
                        missing_keys = [key for key in expected_keys if key not in captures]
                        if missing_keys:
                            raise RuntimeError(f"HNLL HVP did not capture target module outputs: {missing_keys}")

                        module_keys = expected_keys
                        xs = [captures[key][0].detach() for key in module_keys]
                        zs = [captures[key][1] for key in module_keys]
                        hvp_scale = float(2 ** math.ceil(math.log2(max(len(zs), 1))))
                        group_accums = {
                            key: torch.zeros(
                                z.reshape(-1, z.shape[-1]).shape[0],
                                num_groups,
                                device=z.device,
                                dtype=torch.float32,
                            )
                            for key, z in zip(module_keys, zs)
                        }

                        if hvp_engine == "autograd":
                            grad_zs = torch.autograd.grad(
                                loss_nll,
                                zs,
                                create_graph=True,
                                retain_graph=True,
                                allow_unused=False,
                            )
                            if profile:
                                _cuda_sync_if_profiled(device, profile)
                                first_backward_time = time.perf_counter() - phase_start
                                phase_start = time.perf_counter()

                            for probe_idx in range(num_probes):
                                probes = [_rademacher_like(z) for z in zs]
                                hvp_seed = torch.stack([
                                    (grad_z.float() * probe.float()).sum()
                                    for grad_z, probe in zip(grad_zs, probes)
                                ]).sum() / hvp_scale
                                hvps = torch.autograd.grad(
                                    hvp_seed,
                                    zs,
                                    retain_graph=probe_idx < num_probes - 1,
                                    create_graph=False,
                                    allow_unused=False,
                                )
                                for key, probe, hv in zip(module_keys, probes, hvps):
                                    diag_sample = (probe.float() * hv.float() * hvp_scale).reshape(-1, hv.shape[-1])
                                    if not torch.isfinite(diag_sample).all():
                                        raise ValueError(
                                            f"HNLL HVP produced non-finite curvature sample before Hessian build "
                                            f"at layers {batch_layer_indices[0]}-{batch_layer_indices[-1]}, module {key}, "
                                            f"probe={probe_idx}, hvp_scale={hvp_scale}. "
                                            f"Try --nll_hvp_layer_chunk_size with a smaller contiguous chunk."
                                        )
                                    group_accums[key].add_(reduce_channels_by_group_mean(diag_sample, num_groups))
                        else:
                            if profile:
                                first_backward_time = 0.0

                            def _raise_missing_perturbations():
                                missing_perturbed = [key for key in module_keys if key not in perturbed_captures]
                                if missing_perturbed:
                                    raise RuntimeError(f"HNLL finite-difference did not perturb target module outputs: {missing_perturbed}")

                            def finite_diff_grads(probe_dict, sign):
                                nonlocal perturbations, perturbed_captures
                                captures.clear()
                                perturbed_captures = {}
                                perturbations = {
                                    key: (sign * fd_epsilon * probe).to(device)
                                    for key, probe in probe_dict.items()
                                }
                                try:
                                    fd_outputs = model(**forward_kwargs)
                                    fd_loss = fd_outputs.loss * valid_tokens
                                    _raise_missing_perturbations()
                                    fd_zs = [perturbed_captures[key] for key in module_keys]
                                    grads = torch.autograd.grad(
                                        fd_loss,
                                        fd_zs,
                                        create_graph=False,
                                        retain_graph=False,
                                        allow_unused=False,
                                    )
                                finally:
                                    perturbations = None
                                del fd_outputs, fd_loss
                                return grads

                            def finite_diff_grads_batched_signs(probe_dict):
                                nonlocal perturbations, perturbed_captures
                                captures.clear()
                                perturbed_captures = {}
                                perturbations = {
                                    key: torch.cat((fd_epsilon * probe, -fd_epsilon * probe), dim=0).to(device)
                                    for key, probe in probe_dict.items()
                                }
                                batched_forward_kwargs = {
                                    "input_ids": torch.cat((tokens, tokens), dim=0),
                                    "labels": torch.cat((labels, labels), dim=0),
                                }
                                if attention_mask is not None:
                                    mask = forward_kwargs.get("attention_mask")
                                    batched_forward_kwargs["attention_mask"] = torch.cat((mask, mask), dim=0)
                                try:
                                    fd_outputs = model(**batched_forward_kwargs)
                                    fd_loss = fd_outputs.loss * (2 * valid_tokens)
                                    _raise_missing_perturbations()
                                    fd_zs = [perturbed_captures[key] for key in module_keys]
                                    grads = torch.autograd.grad(
                                        fd_loss,
                                        fd_zs,
                                        create_graph=False,
                                        retain_graph=False,
                                        allow_unused=False,
                                    )
                                finally:
                                    perturbations = None
                                del fd_outputs, fd_loss, batched_forward_kwargs
                                grads_plus = []
                                grads_minus = []
                                for grad in grads:
                                    grad_plus, grad_minus = grad.chunk(2, dim=0)
                                    grads_plus.append(grad_plus)
                                    grads_minus.append(grad_minus)
                                return grads_plus, grads_minus

                            zs = [z.detach() for z in zs]
                            captures.clear()
                            del outputs, loss_nll
                            fd_batched_signs_active = fd_batched_signs
                            for probe_idx in range(num_probes):
                                probes = [_rademacher_like(z) for z in zs]
                                probe_dict = {key: probe for key, probe in zip(module_keys, probes)}
                                if fd_batched_signs_active:
                                    try:
                                        grads_plus, grads_minus = finite_diff_grads_batched_signs(probe_dict)
                                    except RuntimeError as exc:
                                        if device.type == "cuda" and "out of memory" in str(exc).lower():
                                            logging.warning(
                                                "HNLL finite-difference batched +/- signs hit CUDA OOM; "
                                                "falling back to sequential +/- signs for this run."
                                            )
                                            torch.cuda.empty_cache()
                                            fd_batched_signs_active = False
                                            grads_plus = finite_diff_grads(probe_dict, 1.0)
                                            grads_minus = finite_diff_grads(probe_dict, -1.0)
                                        else:
                                            raise
                                else:
                                    grads_plus = finite_diff_grads(probe_dict, 1.0)
                                    grads_minus = finite_diff_grads(probe_dict, -1.0)
                                for key, probe, grad_plus, grad_minus in zip(module_keys, probes, grads_plus, grads_minus):
                                    hv = (grad_plus.float() - grad_minus.float()) / (2.0 * fd_epsilon)
                                    diag_sample = (probe.float() * hv).reshape(-1, hv.shape[-1])
                                    if not torch.isfinite(diag_sample).all():
                                        raise ValueError(
                                            f"HNLL finite-difference HVP produced non-finite curvature sample "
                                            f"at layers {batch_layer_indices[0]}-{batch_layer_indices[-1]}, module {key}, "
                                            f"probe={probe_idx}, epsilon={fd_epsilon}."
                                        )
                                    group_accums[key].add_(reduce_channels_by_group_mean(diag_sample, num_groups))
                        if profile:
                            _cuda_sync_if_profiled(device, profile)
                            hvp_time = time.perf_counter() - phase_start
                            phase_start = time.perf_counter()

                        if hvp_engine == "autograd":
                            del grad_zs, probes, hvp_seed, hvps
                        else:
                            del probes, probe_dict, grads_plus, grads_minus
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        sample_hessians = _build_hnll_sample_hessians(
                            module_keys,
                            xs,
                            group_accums,
                            num_probes,
                            hessian_builder,
                            hessian_group_chunk_size,
                            validate_shared_x,
                        )
                        for key, hessian in sample_hessians.items():
                            layer_idx, module_name = key
                            if not torch.isfinite(hessian).all():
                                raise ValueError(f"HNLL H contains non-finite values at layer {layer_idx}, module {module_name}")
                            results[layer_idx][module_name].add_(hessian.detach().cpu())

                        if profile:
                            _cuda_sync_if_profiled(device, profile)
                            build_h_time = time.perf_counter() - phase_start
                            total_time = time.perf_counter() - sample_start
                            allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
                            reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3) if device.type == "cuda" else 0.0
                            logging.info(
                                f"[HNLL][profile] layers={batch_layer_indices[0]}-{batch_layer_indices[-1]} "
                                f"sample={sample_idx + 1}/{len(data)} valid_tokens={valid_tokens} hvp_scale={hvp_scale:.0f} "
                                f"forward={forward_time:.3f}s first_backward={first_backward_time:.3f}s "
                                f"global_hvp={hvp_time:.3f}s build_h={build_h_time:.3f}s total={total_time:.3f}s "
                                f"cuda_alloc={allocated_gb:.2f}GiB cuda_reserved={reserved_gb:.2f}GiB"
                            )

                        for _name in ("outputs", "loss_nll", "grad_zs", "group_accums", "probes", "hvps", "hvp_seed", "sample_hessians", "grads_plus", "grads_minus"):
                            if _name in locals():
                                del locals()[_name]
                finally:
                    for hook in hooks:
                        hook.remove()

                for layer_idx in batch_layer_indices:
                    out_file = os.path.join(output_folder, f"l{layer_idx}.pt")
                    for module_name, hessian in results[layer_idx].items():
                        _log_hnll_hessian_stats(layer_idx, module_name, hessian, num_probes, valid_tokens_total)
                    torch.save(results[layer_idx], out_file)
                    logging.info(f"[Layer {layer_idx}] Saved HNLL HVP Hessians to {out_file}")
                    pb.update(1)
                del results
                torch.cuda.empty_cache()
    finally:
        pb.close()
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if "old_dtype" in locals() and "hnll_dtype" in locals() and hnll_dtype != old_dtype:
            model.to(dtype=old_dtype)

    logging.info("Done accumulating HNLL HVP Hessians for all layers.")
    return False

















