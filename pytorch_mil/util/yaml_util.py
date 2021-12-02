from types import SimpleNamespace
from operator import attrgetter


def create_yaml_obj(yaml_dict):

    def create_obj_from_dict(d):
        obj = SimpleNamespace()
        for key, value in d.items():
            if type(value) == dict:
                value_obj = create_obj_from_dict(value)
            else:
                value_obj = value
            setattr(obj, key, value_obj)
        return obj

    yaml_obj = create_obj_from_dict(yaml_dict)

    def propagate_values(obj, nested_obj):
        for key, value in nested_obj.__dict__.items():
            if type(value) == SimpleNamespace:
                propagate_values(obj, value)
            else:
                if type(value) == str and value.startswith("$"):
                    look_up_key = value[2:-1]
                    look_up_value = attrgetter(look_up_key)(obj)
                    setattr(nested_obj, key, look_up_value)

    propagate_values(yaml_obj, yaml_obj)
    return yaml_obj


def override_yaml_obj(current, override):

    def _override(c, o):
        for key, value in o.__dict__.items():
            if not hasattr(c, key):
                raise ValueError('Override key missing in yaml obj: {:s}'.format(key))
            if type(value) == SimpleNamespace:
                _override(getattr(c, key), value)
            else:
                setattr(c, key, value)

    _override(current, override)
