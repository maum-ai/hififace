import importlib

def instantiate_from_config(config, params = None):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if params is None:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    else:
        return get_obj_from_str(config["target"])(**params, **config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)