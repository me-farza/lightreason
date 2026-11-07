"""Separate feature-conditioned PARD2 proposals, original target controller.

The frozen marginal-tree builder, acceptance, native EOS and compaction loop
are inherited unchanged. A narrow delegate supplies only the verifier's current
committed Feature4 prefix to each auxiliary request. Target weights, equations,
full vocabulary, template and reasoning budget are unchanged. Batched floating
point order/quality and any speed gain still require separate measurements.
"""
from __future__ import annotations

from pathlib import Path

import torch

from bucketed3b_tree_runner import BucketedNumerical3BTreeRunner
from feature4_3b_tree_verifier import Feature4BTreeVerifier
from feature_target_setup3b import initialize_feature_output_bookkeeping
from conditioned_pard2_proposer3b_v1 import ConditionedPARD2LatticeProposer
from pard2_proposer3b_v1 import PARD2Proposer,file_sha,require


class _VerifiedFeatureBoundProposer:
    """No model call/controller logic; inject the CURRENT verified prefix only."""
    def __init__(self,proposer,verifier):self.conditioned,self.verifier=proposer,verifier
    def __getattr__(self,name):return getattr(self.conditioned,name)
    def propose_lattice(self,visible_ids,num_draft_tokens):
        p=self.verifier.current_position
        if visible_ids.shape[1]!=p+1:
            raise RuntimeError("Target cache/features must exclude exactly the current visible anchor")
        history=self.verifier.committed_feature_history[:,:p]
        return self.conditioned.propose_lattice(visible_ids,history,p,num_draft_tokens)


class ConditionedPARD2Tree3BRunner(BucketedNumerical3BTreeRunner):
    def __init__(self,model,tokenizer,draft_path,*,projector_checkpoint_path,max_len=2048,
                 node_budget=64,draft_tokens=15,top_k=8,compile_static=False,compile_draft=False,
                 cuda_graph=True,share_pool=True,warmups=3,validate_topology=True,
                 commit_engine="triton",tree_policy="best-first",attention_buckets=(256,512,1024,2048),
                 diagnostic_test_mode=False):
        from exact10_validation import hash_model_parameters
        require(type(node_budget)is int and 2<=node_budget<=128 and type(draft_tokens)is int and
                1<=draft_tokens<=32 and tree_policy in("best-first","trunk"),"Use valid public tree budgets/policy")
        require(type(top_k)is int and 1<=top_k<=64,"Use fixed public auxiliary topK1..64")
        # A fixed prepared target shape requires enough unique possible paths.
        possible=1;level=1
        for _ in range(draft_tokens):
            level*=top_k;possible+=level
            if possible>=node_budget:break
        require(possible>=node_budget,"Public B/K cannot fill this fixed target node budget")
        setup=initialize_feature_output_bookkeeping(model)
        target_sha=hash_model_parameters(model)
        self.model,self.tokenizer,self.max_len=model,tokenizer,max_len
        self.node_budget,self.draft_tokens,self.top_k=node_budget,draft_tokens,top_k
        self.tree_policy=tree_policy
        buckets=tuple(sorted({int(v)for v in attention_buckets if 1<=int(v)<=max_len}|{max_len}))
        self.verifier=Feature4BTreeVerifier(model,max_len=max_len,node_budget=node_budget,buckets=buckets,
            cuda_graph=cuda_graph,compile_static=compile_static,share_pool=share_pool,warmups=warmups,
            validate_topology=validate_topology,commit_engine=commit_engine)
        base=PARD2Proposer(draft_path,target_tokenizer=tokenizer,device=model.device,
            cache_mode="incremental",max_cache_len=max_len,attn_implementation="sdpa")
        conditioned=ConditionedPARD2LatticeProposer(base,projector_checkpoint_path,
            expected_target_sha256=target_sha,max_len=max_len,attention_buckets=buckets,
            top_k=top_k,compile_model=compile_draft,share_pool=share_pool,warmups=warmups,
            diagnostic_test_mode=diagnostic_test_mode)
        self.proposer=_VerifiedFeatureBoundProposer(conditioned,self.verifier)
        with torch.inference_mode(False),torch.no_grad():
            self.visible=torch.empty((1,max_len),device=model.device,dtype=torch.long)
        self.setup_s,self.prepared=0.,False
        self.metadata={"variant":"original_BF16_target_Feature4_conditioned_PARD2_projector_tree_v1",
            "target_weights_and_equations_unchanged":True,"original_target_live_sha256":target_sha,
            "target_verifier":self.verifier.metadata,"proposer":conditioned.metadata,
            "projector_checkpoint_path":str(Path(projector_checkpoint_path).resolve()),
            "node_budget":node_budget,"draft_tokens":draft_tokens,"top_k":top_k,
            "tree_policy":tree_policy,"compile_static":compile_static,"compile_draft":compile_draft,
            "max_len":max_len,"attention_buckets":list(buckets),"original_HF_feature_setup":setup,
            "target_controller_inherited_unchanged":True,
            "conditioning_inputs":"Current target committed features only, excluding unprocessed anchor",
            "proposal_scores":"Conditioned position-wise PARD2 marginals; not branch-conditioned probabilities",
            "all_online_conditional_prefill_projection_cache_tree_verification_and_cpu_costs_timed":True,
            "full_target_vocabulary_and_original_native_EOS_authoritative":True,
            "strict_token_matching_required":False,"exact_token_agreement_must_be_reported":True,
            "speed_output_agreement_and_quality_unestablished":True,
            "runner_source_sha256":file_sha(__file__)}

    def audit_integrity(self,*,verify_bytes=True,verify_files=True):
        self.verifier.assert_target_unchanged()
        result=self.proposer.audit_integrity(verify_bytes=verify_bytes,verify_files=verify_files)
        if verify_bytes:
            from exact10_validation import hash_model_parameters
            require(hash_model_parameters(self.model)==self.metadata["original_target_live_sha256"],
                    "Original authoritative target live bytes changed")
        return {**result,"original_target_identity_checked":True,"target_live_bytes_verified":bool(verify_bytes)}


def make_quality_runner(model,tokenizer,args):
    from numerical3b_runner import NumericalGenerationAdapter
    names=("node_budget","draft_tokens","top_k","compile_static","compile_draft","cuda_graph",
        "share_pool","warmups","validate_topology","commit_engine","tree_policy","attention_buckets","diagnostic_test_mode")
    runner=ConditionedPARD2Tree3BRunner(model,tokenizer,args.draft_path,
        projector_checkpoint_path=args.projector_checkpoint_path,max_len=args.max_len,
        **{name:getattr(args,name)for name in names if hasattr(args,name)})
    configured=model.generation_config.eos_token_id
    eos=[configured]if isinstance(configured,int)else list(configured or [])
    if tokenizer.eos_token_id is not None:eos.append(tokenizer.eos_token_id)
    return NumericalGenerationAdapter(runner,tokenizer,sorted(set(eos)))
