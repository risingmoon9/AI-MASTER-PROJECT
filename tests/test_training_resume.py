"""Crash recovery must preserve actual Adam, scheduler, RNG, best model and logs."""
from pathlib import Path
import copy
import json
import os
import random
import sys
import tempfile
import unittest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
import hqa_orl.train as training


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)


def fake_evaluate(actor,**kwargs):
    random.random();np.random.rand();torch.rand(2)
    actor.eval()
    value=float(next(actor.parameters()).detach().sum().cpu())
    return {'return_mean':value,'normalized_score_mean':None,'per_episode':[{'env_seed':12000,'return':value}]}


class ResumeTests(unittest.TestCase):
    def test_all_algorithms(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        original_eval=training.evaluate_policy
        original_save=training._atomic_save
        cwd=Path.cwd()
        training.evaluate_policy=fake_evaluate
        try:
            with tempfile.TemporaryDirectory(prefix='hqa-followup-resume-') as directory:
                temp=Path(directory);rng=np.random.default_rng(21)
                obs=rng.normal(size=(29,5)).astype('float32')
                np.savez(temp/'data.npz',obs=obs,next_obs=obs+.01,actions=rng.uniform(-.8,.8,(29,2)).astype('float32'),
                         rewards=rng.normal(size=(29,1)).astype('float32'),dones=np.zeros((29,1),'float32'),
                         quality_weight=rng.uniform(.1,2,(29,1)).astype('float32'),
                         support=rng.uniform(0,1,(29,1)).astype('float32'),uncertainty=rng.uniform(0,1,(29,1)).astype('float32'))
                for algorithm,mode in [('iql','adaptive'),('iql','constant'),('bc','adaptive'),('reference_iql','adaptive')]:
                    with self.subTest(algorithm=algorithm,mode=mode):
                        device=os.environ.get('AUTODL_TEST_DEVICE','cpu')
                        cfg=dict(seed=13,device=device,algorithm=algorithm,
                                 training=dict(updates=6,batch_size=7,hidden_dim=12,hidden_depth=1,
                                               eval_interval=2,eval_episodes=1,log_interval=2,save_interval=4,
                                               normalize_obs=algorithm!='reference_iql',temperature=.5 if algorithm=='reference_iql' else 3.),
                                 hqa=dict(use_quality_weights=True,use_adaptive_bc_regularization=True,regularization_mode=mode,lambda_constant=.123),
                                 paths=dict(checkpoint_dir='checkpoints',result_dir='results'))
                        config=temp/'config.yaml';config.write_text(yaml.safe_dump(cfg))
                        data=temp/'data.npz'
                        def run(resume=False):return training.train_iql(str(config),str(data),'test-env','case',resume=resume)
                        def load():return torch.load('checkpoints/case.pt',map_location='cpu',weights_only=False)
                        full=temp/f'{algorithm}_{mode}_full';full.mkdir();os.chdir(full);run()
                        expected=load();expectedlog=Path('results/case.jsonl').read_bytes()
                        expectedbest=Path('checkpoints/case_best.pt').read_bytes()
                        interrupted=temp/f'{algorithm}_{mode}_interrupted';interrupted.mkdir();os.chdir(interrupted)
                        def crash(payload,path):
                            if Path(path).name=='case.pt' and payload['step']==4:
                                Path('checkpoints/case_best.pt').write_bytes(b'incomplete best')
                                with open('results/case.jsonl','ab') as f:f.write(b'incomplete log')
                                raise InterruptedError('intentional test crash')
                            original_save(payload,path)
                        training._atomic_save=crash
                        with self.assertRaises(InterruptedError):run()
                        training._atomic_save=original_save
                        self.assertEqual(load()['step'],2)
                        run(True);equal(expected,load())
                        self.assertEqual(expectedlog,Path('results/case.jsonl').read_bytes())
                        self.assertEqual(expectedbest,Path('checkpoints/case_best.pt').read_bytes())
                        latest=Path('checkpoints/case.pt').read_bytes();rng_before=training._rng_state()
                        Path('checkpoints/case_best.pt').unlink()
                        with open('results/case.jsonl','ab') as f:f.write(b'uncommitted tail')
                        run(True)
                        self.assertEqual(latest,Path('checkpoints/case.pt').read_bytes())
                        self.assertEqual(expectedlog,Path('results/case.jsonl').read_bytes())
                        self.assertEqual(expectedbest,Path('checkpoints/case_best.pt').read_bytes())
                        equal(rng_before,training._rng_state())
                        changed=copy.deepcopy(cfg);changed['training']['batch_size']=9
                        config.write_text(yaml.safe_dump(changed))
                        with self.assertRaisesRegex(ValueError,'configuration'):run(True)
                        os.chdir(cwd)
        finally:
            training.evaluate_policy=original_eval;training._atomic_save=original_save;os.chdir(cwd)


if __name__=='__main__':unittest.main()
