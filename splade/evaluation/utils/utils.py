import os
import torch
from omegaconf import DictConfig, OmegaConf


def parse(d, name):
    return {k.replace(name + "_", ""): v for k, v in d.items() if name in k}


def rename_keys(d, prefix):
    return {prefix + "_" + k: v for k, v in d.items()}


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def restore_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    # strict = False => it means that we just load the parameters of layers which are present in both and
    # ignores the rest
    if len(missing_keys) > 0:
        print("~~ [WARNING] MISSING KEYS WHILE RESTORING THE MODEL ~~")
        print(missing_keys)
    if len(unexpected_keys) > 0:
        print("~~ [WARNING] UNEXPECTED KEYS WHILE RESTORING THE MODEL ~~")
        print(unexpected_keys)
    print("restoring model:", model.__class__.__name__)


def generate_bow(input_ids, output_dim, device, values=None):
    """from a batch of input ids, generates batch of bow rep
    """
    bs = input_ids.shape[0]
    bow = torch.zeros(bs, output_dim).to(device)
    if values is None:
        bow[torch.arange(bs).unsqueeze(-1), input_ids] = 1
    else:
        bow[torch.arange(bs).unsqueeze(-1), input_ids] = values
    return bow


def normalize(tensor, eps=1e-9):
    """normalize input tensor on last dimension
    """
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)


def get_initialize_config(exp_dict: DictConfig, train=False):
    # delay import to reduce dependencies
    from ..utils.hydra import hydra_chdir
    hydra_chdir(exp_dict)
    exp_dict["init_dict"]["fp16"] = exp_dict["config"].get("fp16", False)
    config = exp_dict["config"]
    init_dict = exp_dict["init_dict"]
    if train:
        os.makedirs(exp_dict.config.checkpoint_dir, exist_ok=True)
        OmegaConf.save(config=exp_dict, f=os.path.join(exp_dict.config.checkpoint_dir, "config.yaml"))
        model_training_config = None
    else:
        if config.pretrained_no_yamlconfig:
            model_training_config = config
        else:
            model_training_config = OmegaConf.load(os.path.join(config["checkpoint_dir"], "config.yaml"))["config"]

        #if HF: need to update config (except for adapters...).
        #if not "adapter_name" in config and "hf_training" in config:
        if  "hf_training" in config and config["hf_training"]:
            init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
            init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None
                   
    return exp_dict, config, init_dict, model_training_config


class L0:
    """non-differentiable
    """
    def __call__(self, batch_rep):
        return torch.count_nonzero(batch_rep, dim=-1).float().mean()
    
def clean_bow(bow,pad_id=None, cls_id=None, sep_id=None, mask_id=None):
    """clean a bag of words representation
    """
    if pad_id:
        bow[:, pad_id] = 0  # otherwise the pad tok is in bow
    if cls_id:
        bow[:, cls_id] = 0  # otherwise the pad tok is in bow
    if sep_id:
        bow[:, sep_id] = 0  # otherwise the pad tok is in bow
    if mask_id:
        bow[:, mask_id] = 0  # otherwise the pad tok is in bow

    return bow

def pruning(output, k,dim):
    topk, indices = torch.topk(output, int(k)) # last dim
    prune_docs = torch.zeros(output.size()).to(output.device)
    output = prune_docs.scatter(dim, indices, topk)
    return output