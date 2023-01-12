# STARLA: Search-Based Testing Approach for Deep Reinforcement Learning Agents

## Table of Contents
- [Publication](#publication)
- [Use case](#use-case)
- [Code breakdown](#code-breakdown)
  * [Requirements](#requirements)
  * [Getting started](#getting-started)
  * [Dataset Structure](#dataset-structure)
- [Research Questions](#research-questions)
  * [RQ1](#rq1-do-we-find-more-faults-than-random-testing-with-the-same-testing-budget)
  * [RQ2](#rq2-can-we-rely-on-ml-models-to-predict-faulty-episodes)
  * [RQ3](#rq3-can-we-learn-accurate-rules-to-characterize-the-faulty-episodes-of-rl-agents)
- [Acknowledgements](#acknowledgements)




## Publication
This repository is a companion page for the following paper 
> "Search-Based Testing Approach for Deep Reinforcement Learning Agents".
> Amirhossein Zolfagharian (uOttawa, Canada), Manel Abdellatif (uOttawa, Canada), Lionel Briand (uOttawa, Canada), Ramesh S (General Motors, USA), and Mojtaba Bagherzadeh (uOttawa, Canada)

[arXiv:2206.07813](https://arxiv.org/abs/2206.07813)



## Use Case 

In the second case study we have a DQN agent (impelented by stable baselines[1]) in Mountain Car environment from the OpenAI Gym library[2]. Mountain car environment is an open-source and another widely used environment for RL agents

In the Mountain Car problem, an under-powered car is located in a valley between two hills. 
Since the gravity is stronger than the engine of the car, the car cannot climb up the steep slope even with full throttle. The objective is to control the car and strategically use its momentum to reach the goal state on top of the right hill as soon as possible. The agent is penalized by -1 for each time step until termination. 


<p align="center" width="100%">
    <img width="45%" src="https://user-images.githubusercontent.com/23516995/212111530-f4f0f644-f946-495a-80ce-97d720910032.JPG"> 
</p>



As illustrated in Figure~\ref{fig:MountainCarExample}, the state of the agent is defined based on:

1. the location of the car along the x-axis, and 

2. the velocity of the car

There are three discrete actions that can be used to
control the car:

• Accelerate to the left.

• Accelerate to the right.

• Do not accelerate


Episodes can have three termination scenarios: 

1. reaching the goal state,
2. crossing the left border, or 
3. exceeding the limit of200 time steps.

In our custom version of the Mountain Car, climbing the left hill is considered an unsafe situation. Consequently, reaching to the leftmost position in the environment results in a termination with the lowest reward. 


Consider a situation in which we are trying to reach to the goal within 180 time steps.

We define reward and functional faults as follows:

- **Reward fault:** If the accumulative time steps of an episode is more than 180 (i.e., a reward below -180) then we consider that there is a reward fault in this episode (as the agent failed to reach the expected reward in the episode).

- **Functional fault:** If in an episode, the car climbs the left hill and passes the left border of the environment,  we consider that there is a functional fault and the reward is equal to the minimum reward (-200).


## Code Breakdown
This project is implemented in python with GoogleColab (Jupyter-notebook).


We have two main notebook files the first one is `RUN_STARLA_MTC.ipynb` which contains the implementation of our search-based testing approach for Mountain Car enrironment. The second one `RE_EXECUTE_MTC.ipynb` is the final step to execute the results as is meant to prepare data required for answering RQ1 & RQ3.

`RUN_STARLA_MTC.ipynb` contains the implementation of our search-based testing approach for Mountain Car environment. The results are stored as files. 

`RE_EXECUTE_MTC.ipynb` removed the duplicated episodes in the results and executed the final set of episodes in Mountain Car enrivonment. This is to keep only valid and consistent failing episodes. Thes results are saved as files.

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
2. Download the Dataset of replication package from [here](https://drive.google.com/drive/folders/16ALL0MuDw2bIDJenD12VLny_4vY23qDE?usp=sharing) and upload it to you Google drive (if you change the location of the files you need to update their path in notebooks).
3. To generate test episodes: open `RUN_STARLA_MTC.ipynb` Mount your drive and run the code.
4. To execute the final results run `RE_EXECUTE_MTC.ipynb`.

The code to generate the results of research questions are in seperate files


       
  
### Dataset Structure 

  A Dataset is provided to reproduce the results. This dataset contains our DRL agent, training data of the agent, episodes of random testing of the agent, episodes generated STARLA, execution data of generated episodes as well as the data required to compare the similarities of states and answer RQs.
  

    STARLA-MTC-dataset
     .
     |
     |--- /dqn-cartpole-50000-with127-GA-Mut-2.pkl             Trained DQN agent 50k steps in Cartpole environment 
     |
     |--- /dict_GA_Mut_10-09-2020.csv                          Training data of ML models 
     |
     |--- /Abstract_unique1_for_d=1.pickle                     Abstract states data     
     |
     |--- /mutation_numbers.pickle                             Number of Mutations that happened during the search
     |
     |--- /random_test_data.pickle                             Random tests episodes representing the final policy of the agent. This also provides the data as a baseline for comparison.
     |
     |--- /Results/                                            Generated episodes as a result of running STARLA.ipynb. This folder contains results of 20 executions
     |
     |--- /Executions/                                         Executed results 
     |
     |--- /Execution-Similarity/                               Executed results + similarity of states
                 
     
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


### Senario2: Randomly executed episodes are generated with STARLA and should be accounted for in the testing budget: 
In the second scenario, we assume that the agent is trained but not tested so far and we want to test the agent using STARLA. Therefore, we need to use part of our testing budget for random executions, to generate the required episodes. 

More precisely, the total testing budget in this scenario is equal to:

The number of episodes in the initial population (generated through random executions of the agent) + Mutated episodes that have been executed during the search + Faulty episodes generated by STARLA (executed after the search)

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/23516995/212109191-cb71399a-1c07-4da4-a71e-e277631c8f3d.png"> 
</p>



**Answer:** For both scenarios, we find significantly more functional faults with STARLA than with Random Testing using the same testing budget. 


## RQ2: Can we rely on ML models to predict faulty episodes?

*In this research question we investigate the accuracy of ML classifiers in predicting faulty episodes of the RL agent.*

We use Random Forest to predict the probabilities of reward and functional faults in a given episode.
To build our training dataset, we sampled episodes from both episodes generated through random executions of the agent and episodes from the training phase of the agent. Episodes are encoded based on the presence or absence of their abstract states. We have two different ML models, one for predicting the probability of a reward fault and the other one for predicting the probability of a functional fault. We considered 70 % of data for training and 30% for testing.

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/23516995/212109623-ab708ee0-5ea6-4058-9cef-3d46c671253a.png"> 
</p>



**Answer:** Using the mentioned ML classifier and feature representation, we can accurately classify the episodes of RL agents as having functional faults, reward faults, or no fault at all.



## RQ3. Can we learn accurate rules to characterize the faulty episodes of RL agents?

*Here, we investigate the learning of interpretable rules that characterize faulty episodes to understand the conditions under which the RL agent can be expected to fail.*


For this reason, we need to rely on an interpretable ML model, in this case, a Decision Tree model, to learn such rules.
We assess the accuracy of decision trees and therefore our ability to learn accurate rules based on the faulty episodes that we identify with STARLA. 
In practice, engineers will need to use such an approach to assess the safety of using an RL agent and understand the reasons of faults.
In this part, we assess the accuracy of trained models that extract the rules of functional and reward faults based on k-fold cross-validation.



<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/23516995/212110023-71f9b0f8-267b-42c8-858f-90b25556c1a4.png"> 
</p>



Such highly accurate rules can help developers understand the conditions under which the agent fails. One can analyze, the concrete states that correspond to abstract states leading to faults to extract real-world conditions of failure. 

**Answer:** By using STARLA and interpretable ML models, such as Decision Trees, we can accurately learn rules that characterize the faulty episodes of RL agents.



Acknowledgements
--------------
This work was supported by a research grant from General Motors as well as the Canada Research Chair and Discovery Grant programs of the Natural Sciences and Engineering Research Council of Canada (NSERC).


References
-----
1- [stable-baselines](https://github.com/hill-a/stable-baselines)

2- [gym](https://github.com/openai/gym)

3- [MOSA](https://ieeexplore.ieee.org/document/7840029)

4- [virtual vs. real:Trading off simulations and physical experiments in reinforcement learning with Bayesian optimization](https://dl.acm.org/doi/abs/10.1109/ICRA.2017.7989186)

