# Maml-Reptile-RL
This is the project from the course "Advanced Deep Learning in Robotics"

In this project, we basically reimplemented the algorithm from paper "Meta Reinforcement Learning for Sim-to-real Domain Adaptation".

To be more specific, we :

• Wrote PPO algorithm from scratch

• Wrote Reptile algorithm from scratch

• Wrote Pseudo MAML algorithm proposed in the literature from scratch

• Modify some classical Pybullet-Gym environments to conduct a series of experiments for model evaluation

To train your agent using pseudo MAML/PPO, please check the code in `main.py`

To train your agent using Reptile, please check the code in `reptile_rl`


Halfcheetah Environments:


Randomized Parameters: 

*Joints Coef: offset 30%

*Dynamics Coef: offset 30%

![This is an image](/img/halfcheetah.gif)



DoubleInvertedPendulum Environments:


Randomized Parameters: 

*Gravity: 1 -- 20

*Torque Factor: 50 -- 500

![This is an image](/img/doublependulum.gif)
