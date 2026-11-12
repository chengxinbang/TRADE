import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "SAC_TRADE"), str(ROOT / "DDQN_TRADE")]
from trade_scheduler import TRADEScheduler
from ddqn_agent import DDQNAgent
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer as SACReplayBuffer
from atari_wrappers import MaxAndSkipEnv, make_atari_env
from envs import make_env
import gymnasium as gym


def load_config(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SchedulerTests(unittest.TestCase):
    def test_identical_standalone_schedulers(self):
        self.assertEqual((ROOT/"DDQN_TRADE/trade_scheduler.py").read_bytes(),
                         (ROOT/"SAC_TRADE/trade_scheduler.py").read_bytes())

    def test_ema_initial_reference_and_floor(self):
        s = TRADEScheduler(100)
        self.assertFalse(s.update(10, 50_001))
        self.assertFalse(s.update(2, 50_005))
        self.assertEqual(s.initial_loss, 10)
        self.assertAlmostEqual(s.smoothed_loss, 9.92)
        self.assertEqual(s.last_diagnostic["interval"], 5)
        self.assertIsNone(s.last_diagnostic["score"])

    def test_environment_clock_not_gradient_clock(self):
        s = TRADEScheduler(100)
        for step in range(1, 102, 4):
            self.assertFalse(s.update(1, step))
            if step < 100:
                self.assertIsNone(s.last_diagnostic["score"])
        self.assertEqual(s.last_diagnostic["interval"], 101)
        self.assertEqual(s.last_diagnostic["score"], 0)

    def test_strict_rebound_and_full_reset(self):
        s = TRADEScheduler(2)
        s.update(1, 1)
        s.update(.5, 2)
        best = s.best_score
        self.assertTrue(s.update(2, 3))
        self.assertGreater(s.history[-1]["score"], best)
        self.assertEqual(s.history[-1]["interval"], 3)
        self.assertEqual(s.cycle, 1)
        self.assertIsNone(s.initial_loss)
        self.assertIsNone(s.smoothed_loss)
        self.assertEqual(s.best_score, math.inf)
        self.assertFalse(s.update(.01, 4))
        self.assertEqual(s.initial_loss, .01)
        self.assertEqual(s.last_diagnostic["interval"], 1)

    def test_constant_signal_does_not_force_a_refresh(self):
        s = TRADEScheduler(2)
        for step in range(1, 1000):
            self.assertFalse(s.update(1, step))
        self.assertEqual(s.history, [])

    def test_checkpoint_and_configuration_validation(self):
        a, b = TRADEScheduler(3), TRADEScheduler(3)
        a.update(3, 101)
        a.update(1, 105)
        b.load_state_dict(a.state_dict())
        for step, loss in [(109, 1), (113, 7), (117, 2)]:
            self.assertEqual(a.update(loss, step), b.update(loss, step))
            self.assertEqual(a.state_dict(), b.state_dict())
        with self.assertRaises(ValueError):
            TRADEScheduler(4).load_state_dict(a.state_dict())
        with self.assertRaises(ValueError):
            a.update(float("nan"), 118)


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ddqn_loss_and_frozen_target(self):
        agent = DDQNAgent((4,84,84), 3, "cpu", use_adaptive_update=True, C_min=100)
        self.assertIsInstance(agent.loss_fn, torch.nn.MSELoss)
        target = copy.deepcopy(agent.target_net.state_dict())
        batch = (torch.randint(0,256,(2,4,84,84),dtype=torch.uint8),
                 torch.tensor([0,1]), torch.tensor([1.,-1.]),
                 torch.randint(0,256,(2,4,84,84),dtype=torch.uint8), torch.zeros(2))
        loss = agent.update(batch, 50_001)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(agent.scheduler.last_diagnostic["interval"],1)
        self.assertTrue(all(torch.equal(v,agent.target_net.state_dict()[k]) for k,v in target.items()))
        clone = DDQNAgent((4,84,84),3,"cpu",use_adaptive_update=True,C_min=100)
        clone.load_checkpoint(copy.deepcopy(agent.get_checkpoint()))
        self.assertEqual(agent.scheduler.state_dict(),clone.scheduler.state_dict())
        a,b=agent.update(batch,50_005),clone.update(batch,50_005)
        self.assertAlmostEqual(a,b,places=7)

    def test_sac_hard_copy_soft_mixing_and_trigger(self):
        for mode in ["soft","hard_adaptive"]:
            agent=SACAgent(3,1,"cpu",target_update_mode=mode,c_min=2,hidden_dim=8)
            before=[p.detach().clone() for p in agent.critic_target.parameters()]
            with torch.no_grad():
                for p in agent.critic.parameters(): p.add_(1)
            agent._update_target_network(1,1)
            if mode=="soft":
                for a,b in zip(before,agent.critic_target.parameters()):
                    torch.testing.assert_close(b,a+.005)
            else:
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,agent.critic_target.parameters())))
                agent._update_target_network(.5,2)
                self.assertEqual(agent.target_refresh_count,0)
                agent._update_target_network(2,3)
                self.assertEqual(agent.target_refresh_count,1)
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(agent.critic.parameters(),agent.critic_target.parameters())))

    def test_sac_update_and_action_bounds(self):
        agent=SACAgent(3,1,"cpu",c_min=100,batch_size=2,hidden_dim=8)
        replay=SACReplayBuffer((3,),(1,),8,"cpu")
        for _ in range(4): replay.add(np.zeros(3),np.zeros(1),1,np.ones(3),False,False)
        loss=agent.update(replay,1001)
        self.assertTrue(math.isfinite(loss))
        action=agent.act(np.zeros(3))
        self.assertEqual(action.shape,(1,))
        self.assertTrue(np.all(np.abs(action)<=1))
        self.assertAlmostEqual(agent.scheduler.initial_loss,loss)


class ConfigurationTests(unittest.TestCase):
    def test_paper_defaults(self):
        ddqn=load_config("ddqn_config_test",ROOT/"DDQN_TRADE/config.py")
        sac=load_config("sac_config_test",ROOT/"SAC_TRADE/config.py")
        self.assertEqual(len(ddqn.ENV_IDS),30)
        self.assertEqual(ddqn.SEEDS,[0,1,2,3,4])
        self.assertEqual((ddqn.TOTAL_STEPS,ddqn.REPLAY_CAPACITY,ddqn.EVAL_EPISODES),(10_000_000,100_000,100))
        self.assertEqual(ddqn.FIXED_C,30000)
        self.assertEqual(ddqn.ADAPTIVE_PARAMS["adaptive_C_min"],500)
        self.assertEqual(ddqn.EVAL_EPSILON,.05)
        self.assertEqual(sac.configuration("dmc","cheetah_run","hard_adaptive",0).c_min,100)
        self.assertEqual(sac.MODES,["soft","hard_adaptive"])
        for suite,name in [("dmc","humanoid_run"),("metaworld","door-open-v2")]:
            cfg=sac.configuration(suite,name,"hard_adaptive",0,c_min=100)
            self.assertEqual(cfg.c_min,100)
            self.assertEqual((cfg.batch_size,cfg.replay_capacity,cfg.discount),(256,1_000_000,.99))
            self.assertEqual(cfg.warmup_steps,1000 if suite=="dmc" else 10000)


class EnvironmentTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("ale_py"),"ALE dependency is not installed")
    def test_all_paper_atari_task_ids(self):
        from atari_wrappers import _normalize_atari_env_id
        cfg=load_config("ddqn_config_registry_test",ROOT/"DDQN_TRADE/config.py")
        for task in cfg.ENV_IDS:
            self.assertIsNotNone(gym.spec(_normalize_atari_env_id(task)))

    def test_max_pool_early_terminal_uses_current_frames(self):
        class FrameEnv(gym.Env):
            observation_space=gym.spaces.Box(0,255,(2,2,3),np.uint8)
            action_space=gym.spaces.Discrete(1)
            def step(self,action):
                self.count+=1
                return np.full((2,2,3),self.count,np.uint8),1,self.count==1,False,{}
            def reset(self,**kwargs):
                self.count=0
                return np.zeros((2,2,3),np.uint8),{}
        env=MaxAndSkipEnv(FrameEnv())
        env.reset()
        obs,reward,terminated,_,info=env.step(0)
        self.assertTrue(terminated)
        self.assertTrue((obs==1).all())
        self.assertEqual(info["raw_frames"],1)

    @unittest.skipUnless(importlib.util.find_spec("ale_py"),"ALE dependency is not installed")
    def test_real_atari_skip_and_sticky_configuration(self):
        env=make_atari_env("BreakoutNoFrameskip-v4",0,episode_life=False,clip_rewards=False)
        try:
            obs,_=env.reset(seed=0)
            self.assertEqual(obs.shape,(4,84,84))
            self.assertEqual(env.unwrapped._frameskip,1)
            self.assertEqual(env.unwrapped.ale.getFloat("repeat_action_probability"),0)
            before=env.unwrapped.ale.getFrameNumber()
            _,_,_,_,info=env.step(0)
            self.assertEqual(env.unwrapped.ale.getFrameNumber()-before,4)
            self.assertEqual(info["raw_frames"],4)
        finally: env.close()

    @unittest.skipUnless(importlib.util.find_spec("dm_control"),"DMC dependency is not installed")
    def test_real_dmc_adapter(self):
        env=make_env("dmc","cartpole_balance",0)
        try:
            obs,_=env.reset()
            self.assertEqual(obs.shape,env.observation_space.shape)
            obs,reward,_,_,_=env.step(env.action_space.sample())
            self.assertTrue(np.isfinite(obs).all() and math.isfinite(reward))
        finally: env.close()

    @unittest.skipUnless(importlib.util.find_spec("metaworld"),"Meta-World dependency is not installed")
    def test_real_metaworld_v2_adapter(self):
        env=make_env("metaworld","door-open-v2",0)
        try:
            obs,_=env.reset()
            self.assertEqual(obs.shape,(39,))
            _,_,_,_,info=env.step(env.action_space.sample())
            self.assertIn("success",info)
        finally: env.close()


if __name__=="__main__": unittest.main()
