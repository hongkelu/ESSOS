import json
import numpy as np
import pytest
from essos.topology_optimizer import DesignProblem, Evaluation, AcceptedState, optimize_topology


def problem():
    return DesignProblem('manufactured-v1',np.zeros(2),np.array([2.,0.5]),np.array([-3.,-3.]),np.array([3.,3.]))


def test_scaled_optimizer_refreshes_each_accepted_design_and_restarts(tmp_path):
    target=np.array([1.,-0.4]); refreshes=[]
    def evaluate(c,snapshot):
        snapshot.setdefault('visits',[]).append(c.tolist())
        return Evaluation(c-target,snapshot,jacobian=np.eye(2))
    def refresh(c,s):
        refreshes.append(c.copy());return evaluate(c,s)
    checkpoint=tmp_path/'checkpoint.json'
    result=optimize_topology(problem(),[0,0],evaluate,refresh,checkpoint=checkpoint)
    np.testing.assert_allclose(problem().physical(result.state.q),target,atol=1e-7)
    assert result.state.objective<1e-14
    assert len(refreshes)==len(result.trials)+1
    loaded=AcceptedState.load(checkpoint,problem_id=problem().fingerprint)
    np.testing.assert_equal(loaded.q,result.state.q)
    external=loaded.snapshot;external['visits'].append('mutated')
    assert 'mutated' not in loaded.snapshot['visits']
    with pytest.raises(ValueError):loaded.q[0]=8
    with pytest.raises(ValueError):AcceptedState.load(checkpoint,problem_id='new-target')
    restarted=optimize_topology(problem(),[0,0],evaluate,refresh,restart=loaded)
    np.testing.assert_allclose(restarted.state.q,loaded.q,atol=1e-7)


def test_topology_valid_but_refreshed_merit_worse_is_rejected():
    def model(c,s):return Evaluation(c-np.ones(2),{'untouched':True},jacobian=np.eye(2))
    def physical(c,s):
        s['bad_trial']=True
        return Evaluation(np.array([1+np.dot(c,c)]),{'accepted_only':c.tolist()})
    result=optimize_topology(problem(),[0,0],model,physical,max_iterations=4)
    assert all(not t['accepted'] and t['status']=='merit_rejected' for t in result.trials)
    np.testing.assert_equal(result.state.q,[0,0])
    assert result.state.snapshot=={'accepted_only':[0.,0.]}


def test_hard_constraint_cannot_be_traded_for_objective():
    def model(c,s):
        return Evaluation(c-np.array([2.,1.]),s,jacobian=np.eye(2),
                          inequalities=np.array([c[0]-0.5]),inequality_jacobian=np.array([[1.,0.]]))
    def refresh(c,s):return model(c,s)
    result=optimize_topology(problem(),[0,0],model,refresh,max_iterations=20)
    physical=problem().physical(result.state.q)
    assert result.status=="constrained_stationarity"
    assert physical[0]<=0.5
    assert result.state.objective<2.5
    assert all(problem().physical(t['q'])[0]<=0.5 for t in result.trials if t.get('accepted'))


def test_invalid_initial_design_and_backend_exceptions_are_distinct():
    invalid=lambda c,s:Evaluation(c,s,valid=False,status='first_hit_changed')
    with pytest.raises(ValueError,match='Initial design'):
        optimize_topology(problem(),[0,0],invalid,invalid)
    def backend(c,s):raise RuntimeError('native backend failure')
    with pytest.raises(RuntimeError,match='native backend'):
        optimize_topology(problem(),[0,0],backend,backend)


def test_checkpoint_rejects_changed_scales_and_preserves_trial_history(tmp_path):
    def model(c,s):return Evaluation(c-np.ones(2),s,jacobian=np.eye(2))
    checkpoint=tmp_path/'restart.json'
    first=optimize_topology(problem(),[0,0],model,model,max_iterations=1,checkpoint=checkpoint)
    loaded=AcceptedState.load(checkpoint,problem_id=problem().fingerprint)
    assert loaded.radius==first.state.radius
    assert loaded.history==list(first.trials)
    changed=DesignProblem(problem().identity,[0,0],[1,1],[-3,-3],[3,3])
    with pytest.raises(ValueError,match='identity'):
        optimize_topology(changed,[0,0],model,model,restart=loaded)
    continued=optimize_topology(problem(),[0,0],model,model,restart=loaded)
    uninterrupted=optimize_topology(problem(),[0,0],model,model)
    np.testing.assert_allclose(continued.state.q,uninterrupted.state.q,atol=1e-12)
    assert continued.state.history==uninterrupted.state.history


def test_nonfinite_rejection_diagnostics_are_recorded_without_acceptance():
    calls=[]
    def model(c,s):return Evaluation(c-np.ones(2),s,jacobian=np.eye(2))
    def refresh(c,s):
        calls.append(c)
        if len(calls)>1:return Evaluation(c,s,valid=False,status='singular',diagnostics={'condition':np.inf})
        return model(c,s)
    result=optimize_topology(problem(),[0,0],model,refresh,max_iterations=1)
    assert not result.trials[0]['accepted']
    assert result.trials[0]['diagnostics']['condition']=='inf'
    np.testing.assert_equal(result.state.q,[0,0])
    with pytest.raises(ValueError):result.state.q.shape=(1,2)
