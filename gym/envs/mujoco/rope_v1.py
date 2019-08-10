import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from scipy.stats import linregress
from gym import spaces

class RopeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'rope_v1.xml', 4)
        utils.EzPickle.__init__(self)
        # self.action_space=spaces.Box(low=np.array([0.0, 0.0]), high=np.array([2, 2]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-2.0, -2.0]), high=np.array([2, 2]), dtype=np.float32)
        # self.action_space=self.sim.data.xfrc_applied[21,:2]




    def l2_norm_dist_2d(self):
        data_point=np.asarray(self.sim.data.geom_xpos[1:])
        _,_,_,_,error=linregress(data_point[:,0],data_point[:,1])
        return error



    # def step(self, a):
    #     ctrl_cost_coeff = 0.0001
    #     xposbefore = self.sim.data.qpos[0]
    #     self.do_simulation(a, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0]
    #     reward_fwd = (xposafter - xposbefore) / self.dt
    #     reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
    #     reward = reward_fwd + reward_ctrl
    #     ob = self._get_obs()
    #     return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)
    def step(self,action):
        # ctrl_cost_coeff = 0.0001
        # dist_before=self.l2_norm_dist_2d()
        # notdone =-dist_before/self.dt*100 < 0
        self.do_simulation_external(action,self.frame_skip)
        # dist_after=self.l2_norm_dist_2d()
        # reward_dist=(-dist_after)*10000
        pos = self.sim.data.geom_xpos
        reward_dist = 1000*np.linalg.norm(pos[11] - pos[21])
        # reward_ctrl=ctrl_cost_coeff*np.square(action).sum()
        reward=reward_dist
        # print(dist_after)
        # print(reward_dist)
        # notdone =(-dist_after/self.dt*100 )<-1
        # done=not notdone
        done=False
        # reward=reward_dist+reward_ctrl
        ob=self._get_obs()
        return ob,reward,done,dict(reward_dist=reward_dist)
        # return ob,reward, False, dict(reward_dist=reward_dist,reward_ctrl=reward_ctrl)




    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])
    # def _get_obs(self):
    #     qpos_head=self.sim.data.qpos('B0')
    #     qpos_tail=self.sim.data.qpos('B20')
    #     return np.concatenate([qpos_head,qpos_tail])

    def reset_model(self):
        # self.init_qpos[1:10]=self.init_qpos[11:20]
        self.init_qpos=self.init_qpos + self.np_random.uniform(low=-.01, high=.01, size=self.model.nq)
        # print(self.sim.data.geom_xpos[1:])

        while self.l2_norm_dist_2d()>4:
            # print(self.l2_norm_dist_2d())
            self.init_qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self.set_state(
            self.init_qpos,
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
