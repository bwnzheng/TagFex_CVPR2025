
def method_dispatch(method_name, *args, **kwargs):
    if method_name == 'finetune':
        from .finetune import Finetune
        learner = Finetune(*args, **kwargs)
    elif method_name == 'icarl':
        from .icarl import iCaRL
        learner = iCaRL(*args, **kwargs)
    elif method_name == 'der':
        from .der.der import DER
        learner = DER(*args, **kwargs)
    elif method_name == 'tagfex':
        from .tagfex.tagfex import TagFex
        learner = TagFex(*args, **kwargs)
    else:
        raise ValueError(f'No method with name {method_name}')

    return learner