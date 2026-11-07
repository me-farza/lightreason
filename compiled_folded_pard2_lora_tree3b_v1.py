"""Compiled full-private-query adapted AUXILIARY; original target/controller retained."""
from __future__ import annotations

from pathlib import Path

import torch

from conditioned_pard2_tree3b_v1 import ConditionedPARD2Tree3BRunner,_VerifiedFeatureBoundProposer
from feature4_3b_tree_verifier import Feature4BTreeVerifier
from feature_target_setup3b import initialize_feature_output_bookkeeping
from pard2_proposer3b_v1 import PARD2Proposer,file_sha,require
from folded_pard2_lora_proposer3b_v1 import _assert_no_alias
from compiled_folded_pard2_lora_proposer3b_v1 import CompiledFoldedPARD2LoRALatticeProposer


class CompiledFoldedPARD2LoRATree3BRunner(ConditionedPARD2Tree3BRunner):
    def __init__(self,model,tokenizer,draft_path,*,adaptation_checkpoint_path,max_len=2048,
                 node_budget=64,draft_tokens=15,top_k=8,compile_static=False,compile_draft=True,
                 cuda_graph=True,share_pool=True,warmups=3,validate_topology=True,
                 commit_engine="triton",tree_policy="best-first",attention_buckets=(256,512,1024,2048)):
        from exact10_validation import hash_model_parameters
        require(type(node_budget)is int and 2<=node_budget<=128 and type(draft_tokens)is int and
                1<=draft_tokens<=32 and type(top_k)is int and 1<=top_k<=64 and
                tree_policy in("best-first","trunk"),"Use valid public tree sizes/policy")
        require(compile_draft is True,"Select this compiled folded auxiliary variant with compile_draft=True")
        possible=1;level=1
        for _ in range(draft_tokens):
            level*=top_k;possible+=level
            if possible>=node_budget:break
        require(possible>=node_budget,"Public B/K cannot fill this fixed target shape")
        setup=initialize_feature_output_bookkeeping(model);target_sha=hash_model_parameters(model)
        self.model,self.tokenizer,self.max_len=model,tokenizer,max_len
        self.node_budget,self.draft_tokens,self.top_k=node_budget,draft_tokens,top_k;self.tree_policy=tree_policy
        buckets=tuple(sorted({int(v)for v in attention_buckets if 1<=int(v)<=max_len}|{max_len}))
        self.verifier=Feature4BTreeVerifier(model,max_len=max_len,node_budget=node_budget,buckets=buckets,
            cuda_graph=cuda_graph,compile_static=compile_static,share_pool=share_pool,warmups=warmups,
            validate_topology=validate_topology,commit_engine=commit_engine)
        reference=PARD2Proposer(draft_path,target_tokenizer=tokenizer,device=model.device,
            cache_mode="incremental",max_cache_len=max_len,attn_implementation="sdpa")
        _assert_no_alias(reference.model,model.parameters())
        folded=CompiledFoldedPARD2LoRALatticeProposer(reference,adaptation_checkpoint_path,
            expected_target_sha256=target_sha,max_len=max_len,attention_buckets=buckets,top_k=top_k,
            compile_model=compile_draft,share_pool=share_pool,warmups=warmups)
        _assert_no_alias(folded.model,model.parameters())
        self.proposer=_VerifiedFeatureBoundProposer(folded,self.verifier)
        with torch.inference_mode(False),torch.no_grad():
            self.visible=torch.empty((1,max_len),device=model.device,dtype=torch.long)
        self.setup_s,self.prepared=0.,False
        self.metadata={"variant":"original_BF16_target_Feature4_folded_LoRA_PARD2_FULL_private_query_compile_tree_v1",
            "original_target_live_sha256":target_sha,"target_weights_and_equations_unchanged":True,
            "target_verifier":self.verifier.metadata,"proposer":folded.metadata,
            "adaptation_checkpoint_path":str(Path(adaptation_checkpoint_path).resolve()),
            "effective_auxiliary_is_intentionally_adapted":True,"original_pinned_reference_kept_unchanged":True,
            "effective_auxiliary_equal_original_checkpoint_claimed":False,
            "node_budget":node_budget,"draft_tokens":draft_tokens,"top_k":top_k,"tree_policy":tree_policy,
            "compile_static":compile_static,"compile_draft":compile_draft,"max_len":max_len,
            "attention_buckets":list(buckets),"original_HF_feature_setup":setup,
            "target_controller_and_online_conditioning_math_inherited_unchanged":True,
            "all_online_feature_prefill_projection_cache_tree_and_verification_costs_timed":True,
            "folding_and_full_integrity_audits_are_constructor_setup":True,
            "strict_token_matching_required":False,"exact_token_agreement_must_be_reported":True,
            "unfolded_vs_folded_auxiliary_proposal_agreement_not_measured_here":True,
            "acceptance_speed_and_target_quality_require_separate_measurement":True,
            "runner_source_sha256":file_sha(__file__)}


def make_quality_runner(model,tokenizer,args):
    """Runtime factory only; a collection wrapper must bracket full audits."""
    from numerical3b_runner import NumericalGenerationAdapter
    names=("node_budget","draft_tokens","top_k","compile_static","compile_draft","cuda_graph","share_pool",
        "warmups","validate_topology","commit_engine","tree_policy","attention_buckets")
    runner=CompiledFoldedPARD2LoRATree3BRunner(model,tokenizer,args.draft_path,
        adaptation_checkpoint_path=args.adaptation_checkpoint_path,max_len=args.max_len,
        **{name:getattr(args,name)for name in names if hasattr(args,name)})
    configured=model.generation_config.eos_token_id
    eos=[configured]if isinstance(configured,int)else list(configured or [])
    if tokenizer.eos_token_id is not None:eos.append(tokenizer.eos_token_id)
    return NumericalGenerationAdapter(runner,tokenizer,sorted(set(eos)))
