import torch
import yaml
import os

from pytorch_mil.util import yaml_util
from .models import InstanceSpaceNN, EmbeddingSpaceNN, AttentionNN, MultiHeadAttentionNN, ClusterGNN

MODEL_NAMES = ["EmbeddingSpaceNN", "InstanceSpaceNN", "AttentionNN", "MultiHeadAttentionNN", "ClusterGNN"]


def get_model_clz_from_name(model_name):
    if model_name == 'InstanceSpaceNN':
        return InstanceSpaceNN
    if model_name == 'EmbeddingSpaceNN':
        return EmbeddingSpaceNN
    if model_name == 'AttentionNN':
        return AttentionNN
    if model_name == 'MultiHeadAttentionNN':
        return MultiHeadAttentionNN
    if model_name == 'ClusterGNN':
        return ClusterGNN
    raise ValueError("No model class found for model name {:s}".format(model_name))


def create_model(device, model_clz, dataset_name, model_yobj_override=None):
    if dataset_name == "sival":
        yaml_path = "config/sival_models.yaml"
    else:
        raise NotImplementedError('Something')
    model_yobj, complete_yobj = parse_model_yobj(model_clz, yaml_path, model_yobj_override)
    n_classes = complete_yobj.n_classes
    n_expec_dims = complete_yobj.n_expec_dims
    return model_clz.from_yaml_obj(device, n_classes, n_expec_dims, model_yobj), model_yobj


def load_model(device, model_name, dataset_name, model_yobj_override=None, repeat=None):
    model_clz = get_model_clz_from_name(model_name)
    model, _ = create_model(device, model_clz, dataset_name, model_yobj_override=model_yobj_override)
    model_path, _, _ = get_model_save_path(dataset_name, model_name, repeat=repeat)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def save_model(model, model_name, dataset_name, repeat=None):
    model_path, model_dir, _ = get_model_save_path(dataset_name, model_name, repeat=repeat)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_path)


def get_model_save_path(dataset_name, model_name, repeat=None):
    base_dir = 'models/{:s}'.format(dataset_name)
    if repeat:
        model_dir = '{:s}/{:s}{:s}'.format(base_dir, dataset_name, model_name)
        model_filename = '{:s}{:s}_{:d}.pkl'.format(dataset_name, model_name, repeat)
    else:
        model_dir = base_dir
        model_filename = '{:s}{:s}.pkl'.format(dataset_name, model_name, repeat)
    return model_dir + "/" + model_filename, model_dir, model_filename


def parse_model_yobj(model_clz, yaml_path, model_yobj_override):
    # Load default model params
    with open(yaml_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)
        complete_yobj = yaml_util.create_yaml_obj(yaml_dict)
        model_yobj = getattr(complete_yobj, model_clz.__name__)
    # Override default model params if override is given
    if model_yobj_override is not None:
        yaml_util.override_yaml_obj(model_yobj, model_yobj_override)
    return model_yobj, complete_yobj
