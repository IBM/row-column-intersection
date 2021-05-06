import logging
import torch
import os
import socket
from torch_util.hypers_base import HypersBase

logger = logging.getLogger(__name__)


DEBUG_MODE = False


def initialize():
    """
    initializes torch distributed
    :return: local_rank, global_rank, world_size
    """
    if "RANK" not in os.environ:
        local_rank = -1
        global_rank = 0
        world_size = 1
    else:
        if torch.cuda.device_count() == 0:
            err = f'No CUDA on {socket.gethostname()}'
            logger.error(err)
            raise ValueError(err)
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        env_master_addr = os.environ['MASTER_ADDR']
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        if os.environ['MASTER_ADDR'].startswith('file://'):
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method=os.environ['MASTER_ADDR'],
                                                 world_size=int(os.environ['WORLD_SIZE']),
                                                 rank=global_rank)
            logger.info("init-method file: {}".format(env_master_addr))
        else:
            torch.distributed.init_process_group(backend='nccl')
            logger.info("init-method master_addr: {} master_port: {}".format(
                env_master_addr, os.environ['MASTER_PORT']))

        logger.info(f"world_rank {global_rank} cuda_is_available {torch.cuda.is_available()} "
                    f"cuda_device_cnt {torch.cuda.device_count()} on {socket.gethostname()}")
        local_rank = int(global_rank % torch.cuda.device_count())
    return local_rank, global_rank, world_size


def to_tensor(hypers: HypersBase, tensor):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device=hypers.device)
    else:
        tensor = tensor.to(hypers.device).detach()
    return tensor


def all_gather(tensor, *, check_id=None):
    """
    all gather the tensor with dimensions [d0 x d1 x...], returning a tensor with dimensions [d0*world_size x d1 x...]
    :param tensor: this process's tensor
    :param check_id: identifier for this call to all_gather (to check that there is no cross talk)
    :return:
    """
    normal_gathered = _all_gather(tensor, debug=False, check_id=check_id)
    # in debug mode we validate there is no cross talk
    debug = DEBUG_MODE and tensor.nelement() > 1
    if debug:
        debug_gathered = _all_gather(tensor, debug=True, check_id=check_id)
        assert normal_gathered.shape == debug_gathered.shape
        assert (normal_gathered - debug_gathered).view(-1).norm() / normal_gathered.nelement() < 0.0001
    return normal_gathered


def _all_gather(tensor, *, debug, check_id=None):
    """
    all gather the tensor with dimensions [d0 x d1 x...], returning a tensor with dimensions [d0*world_size x d1 x...]
    :param tensor: this process's tensor
    :param check_id: identifier for this call to all_gather (to check that there is no cross talk)
    :return:
    """
    # in debug mode we validate there is no cross talk
    add_dim = False
    tensor = tensor.detach()
    if debug:
        add_dim = len(tensor.shape) == 1 or tensor[0].nelement() < 2
        if add_dim:
            tensor = tensor.unsqueeze(0)
        check_id = torch.tensor(check_id, dtype=tensor.dtype).item()
        rank = torch.distributed.get_rank()
        canary = torch.zeros(tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
        # fill canary
        canary.view(-1)[0] = rank
        canary.view(-1)[1] = check_id
        tensor = torch.cat([tensor, canary.unsqueeze(0)])
    gather_list = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gather_list, tensor)
    if debug:
        canaries = [tensor[-1:].squeeze(0) for tensor in gather_list]
        # check canaries
        for r in range(len(canaries)):
            assert canaries[r].view(-1)[0] == r
            assert canaries[r].view(-1)[1] == check_id
        gather_list = [tensor[:-1].unsqueeze(0) if add_dim else tensor[:-1] for tensor in gather_list]
    return torch.cat(gather_list, 0).detach()


def reduce(hypers: HypersBase, tensor, *, op=torch.distributed.ReduceOp.SUM, check_id=None):
    """
    all reduce the tensor, modifying the tensor
    :param tensor: the tensor that will be all-reduced
    :param op: operation to reduce (example: torch.distributed.ReduceOp.SUM)
    :param check_id: identifier for this call to all_reduce (to check that there is no cross talk)
    :return:
    """
    tensor = to_tensor(hypers, tensor)
    if hypers.world_size == 1:
        return tensor
    torch.distributed.all_reduce(tensor, op)
    return tensor

