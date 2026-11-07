"""Audited original-HF lazy output-capture initialization, before runner setup.

Transformers4.57.1's first output_hidden_states=True call restores each decoder
forward as an instance-bound copy of the SAME class function. The first call
therefore changes Python bookkeeping from no instance override to an equal
bound method. Parameter bytes, class forward functions, module objects, modes
and parameter versions must remain unchanged. The frozen runtime's subsequent
strict state snapshot is taken only after this original-library initialization.
No model method or parameter is assigned/replaced by this helper.
"""
from __future__ import annotations
import time


def validate_lazy_forward_transition(before,after):
    parameters0,modules0=before;parameters1,modules1=after
    if parameters0!=parameters1 or set(modules0)!=set(modules1):
        raise RuntimeError('Original target parameter or module identity changed during HF initialization')
    changed=[]
    for name,a in modules0.items():
        b=modules1[name]
        if a==b:continue
        # Tuple fields: module ID, class, effective forward function,
        # instance forward override, training mode. Only an original bound
        # function being restored by HF's own output recorder is allowed.
        if (any(a[i]!=b[i] for i in (0,1,2,4)) or a[3] is not None or
                not name.startswith('model.layers.') or len(name.split('.'))!=3 or
                not name.rsplit('.',1)[1].isdigit() or
                getattr(b[3],'__func__',None) is not a[2] or
                id(getattr(b[3],'__self__',None))!=a[0]):
            raise RuntimeError('Unexpected original HF forward transition: '+name)
        changed.append(name)
    return changed


def initialize_feature_output_bookkeeping(target):
    import torch
    import transformers
    from feature_draft_proposer3b import _target_state
    from exact10_validation import hash_model_parameters
    if transformers.__version__!='4.57.1' or target.training or any(
            p.requires_grad or p.dtype!=torch.bfloat16 for p in target.parameters()):
        raise ValueError('Use the pinned original frozen all-BF16 target')
    began=time.perf_counter(); before=_target_state(target); before_hash=hash_model_parameters(target)
    device=target.get_input_embeddings().weight.device
    # A public fixed token, no evaluation prompt/output and no retained cache.
    with torch.inference_mode(False),torch.no_grad():
        ids=torch.tensor([[151644]],device=device,dtype=torch.long)
        output=target(ids,use_cache=False,logits_to_keep=1,output_hidden_states=True,
                      output_attentions=False,return_dict=True)
        if len(output.hidden_states)!=37:raise RuntimeError('Original target feature output contract changed')
        del output,ids
    if device.type=='cuda':torch.cuda.synchronize(device)
    changed=validate_lazy_forward_transition(before,_target_state(target))
    after_hash=hash_model_parameters(target)
    if before_hash!=after_hash:raise RuntimeError('Original target bytes changed during HF initialization')
    return {'original_HF_output_capture_initialized_before_runtime_snapshot':True,
        'original_HF_forward_called_with_hidden_states':True,'fixed_public_setup_token':151644,
        'changed_original_bound_forward_bookkeeping':changed,
        'original_target_parameter_versions_and_bytes_unchanged':True,
        'original_class_forward_functions_and_modules_unchanged':True,
        'target_parameter_sha256':after_hash,'setup_s':time.perf_counter()-began,
        'latency_scope':'one-time constructor setup, excluded from warmed generation'}
