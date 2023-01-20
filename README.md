
[![DOI](https://img.shields.io/badge/DOI-10.7910%2FDVN%2FXWIWVS-orange?style=plastic)](https://doi.org/10.7910/DVN/XWIWVS)
![GitHub](https://img.shields.io/github/license/amirhosseinzlf/STARLA?style=plastic)
[![arXiv](https://img.shields.io/badge/arXiv%20-2206.07813-red?style=plastic&logo=arxiv)](https://arxiv.org/abs/2206.07813)
# STARLA: Search-Based Testing Approach for Deep Reinforcement Learning Agents

## Table of Contents
- [Introduction](#introduction)
- [Publication](#publication)
- [Description of the Approach](#description-of-the-approach)
- [Use cases](#use-case-1-cartpole)
  * [Use case 1](#use-case-1-cartpole)
  * [Use case 2](#use-case-2-mountain-car)
- [Code breakdown](#code-breakdown)
  * [Requirements](#requirements)
  * [Getting started](#getting-started)
  * [Repository Structure](#repository-structure)
  * [Dataset Structure](#dataset-structure)
- [Research Questions](#research-questions)
  * [RQ1](#rq1-do-we-find-more-faults-than-random-testing-with-the-same-testing-budget)
  * [RQ2](#rq2-can-we-rely-on-ml-models-to-predict-faulty-episodes)
  * [RQ3](#rq3-can-we-learn-accurate-rules-to-characterize-the-faulty-episodes-of-rl-agents)
- [Acknowledgements](#acknowledgements)


## Introduction

In this project, we propose a Search-based Testing Approach for Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent. This approach effectively searches for failing executions of the agent where we have a limited testing budget. To achieve this, we rely on machine learning models to guide our search and we use a dedicated genetic algorithm to narrow the search toward faulty episodes. These episodes are sequences of states and actions produced by the DRL agent. We apply STARLA on a DQN agent trained in a Cartpole environment for 50K time steps.



## Publication
This repository is a companion page for the following paper 
> "Search-Based Testing Approach for Deep Reinforcement Learning Agents".

> Amirhossein Zolfagharian (uOttawa, Canada), Manel Abdellatif (uOttawa, Canada), Lionel Briand (uOttawa, Canada), Ramesh S (General Motors, USA), and Mojtaba Bagherzadeh (uOttawa, Canada)

[arXiv:2206.07813](https://arxiv.org/abs/2206.07813)


## Description of the Approach

STARLA Requires a DRL agent and its training data as input as tries to effectively and efficiently generate failing test cases (episodes) to reveal the faults of agents policy. Detailed approach is depicted in the following diagram:


![Approach4_page-0001](https://user-images.githubusercontent.com/23516995/168500802-50486e30-2c5d-43c2-a080-9cc01d964e30.jpg)


As depicted, the main objective of STARLA is to generate and find episodes with high fault probabilities in order to assess whether an RL agent can be safely deployed. 
The algorithm uses the data from the Agent to build ML models that predict the probabilities of fault (to which extent episodes are similar to faulty episodes). The outputs of these models are combined with the reward of the agent and certainty level. They are meant to guide the Genetic search toward faulty episodes. 

In the Genetic search, we use specific crossover and mutation functions. Also as we have multiple fitness functions, we are using MOSA Algorithm[3]. For more explanations please see our paper. [arXiv:2206.07813](https://arxiv.org/abs/2206.07813)


# Use cases

## Use Case 1: Cartpole

We use the Cartpole environment from the OpenAI Gym library[2] as first case study. Cartpole environment is an open-source and widely used environment for RL agents

In the Cart-Pole (also known as invert pendulum), a pole is attached to a cart, which moves along a track. The movement of the cart is bidirectional so the available actions are pushing the cart to the left and right. However, the movement of the cart is restricted and the maximum rage is 2.4 from the central point. 
The pole starts upright, and the goal is to balance it by moving the cart left or right.


<p align="center" width="100%">
    <img width="45%" src="https://user-images.githubusercontent.com/23516995/168501958-b4e278ab-dce6-419c-bb35-4dc1f85b6d99.jpg"> 
</p>

As depicted in the figure, the state of the system is characterized by four elements:

• The position of the cart.

• The velocity of the cart.

• The angle of the pole.

• The angular velocity of the pole.


We provide a reward of +1 for each time step when the pole is still upright. 
The episodes end in three cases: 
1. The cart is away from the center with a distance more than 2.4 units
2. The pole’s angle is more than 12 degrees from vertical
3. The pole remains upright during 200 time-steps.

Consider a situation in which we are trying to reach a reward above 70.

We define functional faults in the Cart-Pole problem as follows:

- **Functional fault:** If in a given episode, the cart moves away from the center with a distance above 2.4 units, regardless of the accumulated reward, we consider that there is a functional fault in that episode.

## Use Case 2: Mountain Car

In the second case study, we have a DQN agent (impelented by stable baselines[1]) in Mountain Car environment from the OpenAI Gym library[2]. Mountain car environment is an open-source and another widely used environment for RL agents

In the Mountain Car problem, an under-powered car is located in a valley between two hills. 
Since the gravity is stronger than the engine of the car, the car cannot climb up the steep slope even with full throttle. The objective is to control the car and strategically use its momentum to reach the goal state on top of the right hill as soon as possible. The agent is penalized by -1 for each time step until termination. 


<p align="center" width="100%">
    <img width="45%" src="https://user-images.githubusercontent.com/23516995/212111530-f4f0f644-f946-495a-80ce-97d720910032.JPG"> 
</p>

The state of the agent is defined based on:

1. the location of the car along the x-axis.
2. the velocity of the car.

There are three discrete actions that can be used to control the car:

• Accelerate to the left.

• Accelerate to the right.

• Do not accelerate.


Episodes can have three termination scenarios: 

1. reaching the goal state,
2. crossing the left border, or 
3. exceeding the limit of200 time steps.

In our custom version of the Mountain Car, climbing the left hill is considered an unsafe situation. Consequently, reaching to the leftmost position in the environment results in a termination with the lowest reward. 

We define functional faults as follows:

- **Functional fault:** If in an episode, the car climbs the left hill and passes the left border of the environment, we consider that there is a functional fault and the reward is equal to the minimum reward (-200).


## Code Breakdown
This project is implemented in python with GoogleColab (Jupyter-notebook).


We have two main notebook files the first one is `STARLA.ipynb` which contains the implementation of our search-based testing approach. The second one `Execute_Results.ipynb` is the final step to execute the results as is meant to prepare data required for answering RQ1 & RQ3.

`STARLA.ipynb` contains the implementation of our search-based testing approach. The results are stored as files. 

`Execute_Results.ipynb` removed the duplicated episodes in the results and executed the final set of episodes. This is to keep only valid and consistent failing episodes. Thes results are saved as files.

Mountain Car folder contains the implementation of STARLA on Mountain Car problem. Files follow the same structore as well. 
`RUN_STARLA_MTC.ipynb` contains the implementation of our search-based testing approach for Mountain Car enrironment, `RE_EXECUTE_MTC.ipynb` is the final step to execute the results.

As our algorithm is randomized to have a fair comparison we need to run our algorithm and the baseline many times and compare the results. 


## Requirements

This project is implemented using the following Libraries:
 - stable-baselines==2.10.2
 - pymoo==0.4.2.2

To install dependencies: 
  - `!pip install stable-baselines==2.10.2`
  - `!pip install pymoo==0.4.2.2`

The code was developed and tested based on the following packages:

- python 3.7
- Tensorflow 1.15.2
- matplotlib 3.2.2
- sklearn 1.0.2
- gym 0.17.3
- numpy 1.21.6
- pandas 1.3.5
---------------
Change the version of TensorFlow using the line below:

`%tensorflow_version 1.x`

Here is the documentation on how to use this replication package.


### Getting Started

1. Clone the repo on your Google drive and run the codes using Google Colab https://colab.research.google.com/.
2. Download the Dataset of replication package from Harvard Dataverse [here](https://doi.org/10.7910/DVN/XWIWVS) and upload it to you Google drive (if you change the location of the files you need to update their path in notebooks).
3. To generate test episodes: open `STARLA.ipynb` Mount your drive and run the code.
4. To execute the final results run `Execute_Results.ipynb`.

The code to generate the results of research questions are in the `RQ` folder 



### Repository Structure

This is the root directory of the repository. The directory is structured as follows:

    Replication package of STARLA
     .
     |
     |STARLA/
     |
     |--- Cart-Pole/                               Cart-Pole use case
     |
     |---------- /RQ/                              Codes to replicate RQ1 - RQ2 and RQ3
     |
     |---------- STARLA.ipynb                      Implementation of the algorithm on Cart-Pole problem
     |
     |---------- Execute_Results.ipynb             Execution of the result (required for RQ1 and RQ3)            
     |
     |
     |--- Mountain_Car/                            Mountain Car use case
     |
     |---------- /RQ/                              Codes to replicate RQ1 - RQ2 and RQ3
     |
     |---------- RUN_STARLA_MTC.ipynb                      Implementation of the algorithm on Mountain Car problem
     |
     |---------- RE_EXECUTE_MTC.ipynb             Execution of the result (required for RQ1 and RQ3)          
  
### Dataset Structure 

  A Dataset is provided to reproduce the results. This dataset contains our DRL agent, training data of the agent, episodes of random testing of the agent, episodes generated STARLA, execution data of generated episodes as well as the data required to compare the similarities of states and answer RQs for both use cases. Thus the dataset is devided into two parts:
  1. Dataset_Cart_Pole 
  2. Dataset_MTC
  
  below is the structure of the dataset:

    Dataset
     .
     |Dataset_Cart_pole/
     |
     |--- /dqn-cartpole-50000-with127-GA-Mut-2.pkl             Trained DQN agent 50k steps in Cartpole environment 
     |
     |--- /dict_GA_Mut_10-09-2020.csv                          Training data of ML models 
     |
     |--- /Abstract_unique1_for_d=1.pickle                     Abstract states data     
     |
     |--- /mutation_number_t.pickle                            Number of Mutations that happened during the search
     |
     |--- /random_test_data.pickle                             Random tests episodes representing the final policy of the agent. This also provides the data as a baseline for comparison.
     |
     |--- /random_test_data_start_state.pickle                 Initial states of random episodes
     |
     |--- /Results/                                            Generated episodes as a result of running STARLA.ipynb. This folder contains results of 20 executions
     |
     |--- /Executions/                                         Executed results 
     |
     |--- /Execution-Similarity/                               Executed results + similarity of states
     |
     |
     |
     |
     |Dataset_MTC/
     |
     |--- /dqn-4-1-6-89946.zip                                 Trained DQN agent 90k steps in Mountain Car environment 
     |
     |--- /Final_episodes_trainand_Test_2062_FIXED2.pickle     Training data of ML models 
     |
     |--- /newly_seen_abs.pickle                               Newly seen abstract states during the search     
     |
     |--- /ToTalMutationNumber0.pickle                         Number of Mutations that happened during the search
     |
     |--- /RandomDataset/                                      Random tests episodes representing the final policy of the agent. This also provides the data as a baseline for comparison.
     |
     |--- /Abstraction/                                        Abstract class data
     |
     |--- /Results/                                            Results of running RUN_STARLA_MTC.ipynb. This folder contains results of 20 executions
     |
     |--- /Generation/                                         Generated episodes in 20 executions 
     |
     |--- /Exe_Sim/                                            Executed results + similarity of states
                 
     
----------------
     
  

# Research Questions


Our experimental evaluation answers the research questions below.

## RQ1: Do we find more faults than Random Testing with the same testing budget?

*In this research question we want to study the effectiveness of STARLA in finding more faults than Random Testing when we consider the same testing budget, measured as the number of executed episodes.*

To do so, we consider two practical testing scenarios:
In both scenarios training episodes of the RL agent are given.
### Scenario1: Randomly executed episodes are available or inexpensive: 
In this scenario, we can consider that episodes of random executions of the agent are available. 
One example is when the agent is tested to some extent. However, before final deployment, we want further test the agent using STARLA. 
Another situation is when the RL agent is trained and tested using both a simulator and hardware in the loop [4].

In this situation, an agent is trained and tested on a simulator in order to have a ”warm-start” learning on real hardware [4]. Since STARLA produces episodes with a high fault probability, we can use it to test the agent when executed on real hardware to further assess the reliability of the agent. In such situation, STARLA uses episodes that are generated with a simulator and executes the newly generated episodes on the hardware.

More precisely, the total testing budget in this scenario is equal to:

Mutated episodes that have been executed during the search + Faulty episodes generated by STARLA (executed after the search)


### Scenario2: Randomly executed episodes are generated with STARLA and should be accounted for in the testing budget: 
In the second scenario, we assume that the agent is trained but not tested so far and we want to test the agent using STARLA. Therefore, we need to use part of our testing budget for random executions, to generate the required episodes. 

More precisely, the total testing budget in this scenario is equal to:

The number of episodes in the initial population (generated through random executions of the agent) + Mutated episodes that have been executed during the search + Faulty episodes generated by STARLA (executed after the search)




<p align="center" width="100%">
   <img width="45%" alt="CartPole" src="https://user-images.githubusercontent.com/38301008/212175197-bba293be-da5b-4ea0-b894-e7803b262bf3.png">
 </p>
 <p align="center" width="50%">
   Number of detected functional faults in the Cartpole case study
</p>


<p align="center" width="100%">
    <img width="45%" src="https://user-images.githubusercontent.com/38301008/212175170-6ba47d61-2260-44ec-b18c-fdc3d80d28a6.png" > 
</p>
<p align="center" width="50%">
   Number of detected functional faults in the Mountain Car case study
</p>


**Answer:** For both scenarios and in both case studies, we find significantly more functional faults with STARLA than with Random Testing using the same testing budget. 


## RQ2: Can we rely on ML models to predict faulty episodes?

*In this research question we investigate the accuracy of ML classifiers in predicting faulty episodes of the RL agent.*

We use Random Forest to predict the probabilities of reward and functional faults in a given episode.
To build our training dataset, we sampled episodes from both episodes generated through random executions of the agent and episodes from the training phase of the agent. Episodes are encoded based on the presence or absence of their abstract states. We have two different ML models, one for predicting the probability of a reward fault and the other one for predicting the probability of a functional fault. We considered 70 % of data for training and 30% for testing.

**Answer:** Using the mentioned ML classifier and feature representation, we can accurately classify the episodes of RL agents as having functional faults or no fault at all.



## RQ3. Can we learn accurate rules to characterize the faulty episodes of RL agents?

*Here, we investigate the learning of interpretable rules that characterize faulty episodes to understand the conditions under which the RL agent can be expected to fail.*

For this reason, we need to rely on an interpretable ML model, in this case, a Decision Tree model, to learn such rules.
We assess the accuracy of decision trees and therefore our ability to learn accurate rules based on the faulty episodes that we identify with STARLA. 
In practice, engineers will need to use such an approach to assess the safety of using an RL agent and understand the reasons of faults.
In this part, we assess the accuracy of trained models that extract the rules of functional and reward faults based on k-fold cross-validation.

<!-- <p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/23516995/169616496-bebacddf-cb97-4ab3-bcf9-cfb8b654a4ee.png"> 
</p> -->

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/38301008/212181441-1f3237dd-f95e-4a77-9e6a-01717ce33f6c.png" > 
</p>
<p align="center" width="50%">
   Accuracy of the fault detection rules in the Cartpole case study
</p>

<p align="center" width="100%">
   <img width="50%" alt="CartPole" src="https://user-images.githubusercontent.com/38301008/212180368-aad9d8f4-c7ea-464f-8274-77bbade2c3e1.png">
 </p>
 <p align="center" width="50%">
   Accuracy of the fault detection rulesin the Mountain Car case study
</p>


Such highly accurate rules can help developers understand the conditions under which the agent fails. One can analyze, the concrete states that correspond to abstract states leading to faults to extract real-world conditions of failure. 
For example, we extracted the following faulty rule in the Cartpole problem $Not(S^\phi_{5})$ and $S^\phi_{12}$ and $S^\phi_{23}$ from our decision tree. First we extract all faulty episodes following this rule. Then, we extract from these episodes all concrete states belonging to the abstract states with the condition of presence in **R1**, i.e., $S^\phi_{12}$ and $S^\phi_{23}$.
For abstract states $S^\phi_5$ where the rule states they should be absent, we extract the set of all corresponding concrete states from all episodes in the final dataset.
Finally, for each abstract state in the rule, we analyze the distribution of each characteristic of the corresponding concrete states (i.e., the position of the cart, the velocity, the angle of the pole and the angular velocity) to interpret the situations under which the agent fails. below you see the boxplots of the mentioned distributions.

<p align="center" width="100%">
    <img width="100%" src="https://user-images.githubusercontent.com/23516995/171912017-75548e5b-9151-42f1-afbc-ffafbd8d163e.png"> 
</p>



<p align="center" width="100%">
    <img width="100%" src="https://user-images.githubusercontent.com/23516995/171912194-b58beb6f-7cdd-402a-99d2-193b1c2f1664.png"> 
</p>


<p align="center" width="100%">
    <img width="100%" src="https://user-images.githubusercontent.com/23516995/171912131-db832638-ea55-4754-a297-a8ae05a98b64.png"> 
</p>


Moreover, we rely on the median values of the distribution of the states' characteristics to illustrate each abstract state and hence the failing conditions. 
We illustrate in the following figure an interpretation of such conditions.


<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/23516995/171913026-a1713863-4ac8-46d9-b930-6a20b7ba1a53.png"> 
</p>

Each cart represents one abstract state. The Gray cart depicts the state of the system in abstract state $S^\phi_5$, which should be absent in the episode. The black carts represent the presence of abstract states $S^\phi_{12}$ and $S^\phi_{23}$, respectively. Having both states of the cart shown in the right as and not having the state at the left indicate a fault.

We realized that the presence of abstract states $S^\phi_{12}$ and $S^\phi_{23}$ represent situations where the cart is close to the right border of the track and the angle of the pole is towards the right. To compensate for the large angle of the pole, as you can see in the figure, the agent has no choice but to push the cart to the right, which results in a fault because of passing the border. Moreover, abstract state $S^\phi_{5}$ represents a situation where the angle of the pole is not large, and the position of the cart is toward the right but not close to the border. In such situation, the agent will be able to control the pole in the remaining area and keep the pole upright without crossing the border, which justifies why such abstract state should be absent in faulty episodes that satisfy rule **R1**.

**Answer:** By using STARLA and interpretable ML models, such as Decision Trees, we can accurately learn rules that characterize the faulty episodes of RL agents.

NOTE:TensorFlow 1.X is no longer supported by Google Colab. It is recommended that you create your own virtual environment with Python 3.7 and install the necessary requirements.

Acknowledgements
--------------
This work was supported by a research grant from General Motors as well as the Canada Research Chair and Discovery Grant programs of the Natural Sciences and Engineering Research Council of Canada (NSERC).


References
-----
1- [stable-baselines](https://github.com/hill-a/stable-baselines)

2- [gym](https://github.com/openai/gym)

3- [MOSA](https://ieeexplore.ieee.org/document/7840029)

4- [virtual vs. real:Trading off simulations and physical experiments in reinforcement learning with Bayesian optimization](https://dl.acm.org/doi/abs/10.1109/ICRA.2017.7989186)


Cite our paper:
--------

If you've found STARLA useful for your research, please cite our paper as follows:

    @misc{https://doi.org/10.48550/arxiv.2206.07813,
    doi = {10.48550/ARXIV.2206.07813},
    url = {https://arxiv.org/abs/2206.07813},
    author = {Zolfagharian, Amirhossein and Abdellatif, Manel and Briand, Lionel and Bagherzadeh, Mojtaba and S, Ramesh},
    keywords = {Software Engineering (cs.SE), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {A Search-Based Testing Approach for Deep Reinforcement Learning Agents},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
    }

