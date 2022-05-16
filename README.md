# STARLA: Search-Based Testing Approach for Deep Reinforcement Learning Agents

## Table of Contents
- [Introduction](#introduction)
- [Publication](#publication)
- [Description of the Approach](#description-of-the-approach)
- [Use case](#use-case)
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

Cite


## Description of the Approach

STARLA Requires a DRL agent and its training data as input as tries to effectively and efficiently generate failing test cases (episodes) to reveal the faults of agents policy. Detailed approach is depicted in the following diagram:


![Approach4_page-0001](https://user-images.githubusercontent.com/23516995/168500802-50486e30-2c5d-43c2-a080-9cc01d964e30.jpg)



As depicted, the main objective of STARLA is to generate and find episodes with high fault probabilities in order to assess whether an RL agent can be safely deployed. 
The algorithm uses the data from the Agent to build ML models that predict the probabilities of fault (to which extent episodes are similar to faulty episodes). The outputs of these models are combined with the reward of the agent and certainty level. They are meant to guide the Genetic search toward faulty episodes. 

In the Genetic search, we use specific crossover and mutation functions. Also as we have multiple fitness functions, we are using MOSA Algorithm. For more explanations please see our paper \cite


## Use case 

This project is implemented in the Cartpole environment for the OpenAI Gym library. Cartpole environment is an open-source and widely used environment for RL agents

In the Cart-Pole (also known as invert pendulum), a pole is attached to a cart, which moves along a track. The movement of the cart is bidirectional so the available actions are pushing the cart to the left and right. However, the movement of the cart is restricted and the maximum rage is 2.4 from the central point. 
The pole starts upright, and the goal is to balance it by moving the cart left or right.


<p align="center" width="100%">
    <img width="45%" src="https://user-images.githubusercontent.com/23516995/168501958-b4e278ab-dce6-419c-bb35-4dc1f85b6d99.jpg"> 
</p>

As depicted in the figure, the state of the system is characterized by four elements:
• The position of the cart
• The velocity of the cart
• The angle of the pole
• The angular velocity of the pole

We provide a reward of +1 for each time step when the pole is still upright. 
The episodes end in three cases: 
1. The cart is away from the center with a distance more than 2.4 units
2. The pole’s angle is more than 12 degrees from vertical
3. The pole remains upright during 200 time-steps.

Consider a situation in which we are trying to reach a reward above 70.

We define reward and functional faults in the Cart-Pole problem as follows:

- **Reward fault:** If the accumulative time steps of an episode is less than 70 then we consider that there is a reward fault in this episode (as the agent failed to reach the expected reward in the episode).

- **Functional fault:** If in a given episode, the cart moves away from the center with a distance above 2.4 units, regardless of the accumulated reward, we consider that there is a functional fault in that episode.


## Code breakdown
This project is implemented in python with GoogleColab (Jupyter-notebook).


We have two main notebook files the first one is `STARLA.ipynb` which contains the implementation of our search-based testing approach. The second one `Execute_Results.ipynb` is the final step to execute the results as is meant to prepare data required for answering RQ1 & RQ3.

`STARLA.ipynb` contains the implementation of our search-based testing approach. The results are stored as files. 

`Execute_Results.ipynb` removed the duplicated episodes in the results and executed the final set of episodes. This is to keep only valid and consistent failing episodes. Thes results are saved as files

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


### Getting started

1. Clone the repo on your Google drive and run the codes using Google Colab https://colab.research.google.com/.
2. Download the Dataset of replication package from [here](https://drive.google.com/drive/folders/16ALL0MuDw2bIDJenD12VLny_4vY23qDE?usp=sharing) and upload it to you Google drive( if you change the location of the files you need to update their path in notebooks
3. To generate test episodes: open `STARLA.ipynb` Mount your drive and run the code
4. To execute the final results run `Execute_Results.ipynb`

The code to generate the results of research questions are in the `RQ` folder 



### Repository Structure

This is the root directory of the repository. The directory is structured as follows:

    Replication package of STARLA
     .
     |
     |--- STARLA/RQ/                        Codes to replicate RQ1 - RQ2 and RQ3
     |
     |--- STARLA.ipynb                      Implementation of algorithm
     |
     |--- Execute_Results.ipynb             Execution of the result (required for RQ1 and RQ3)             
  
### Dataset Structure 

  A Dataset is provided to reproduce the results. This dataset contains our DRL agent, training data of the agent, episodes of random testing of the agent, episodes generated STARLA, execution data of generated episodes as well as the data required to compare the similarities of states and answer RQs.
  

    STARLA-dataset
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
     |--- /Results/                                            Generated episodes as a result of running STARLA.ipynb . this folder contains results of 14 executions
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
In both scenarios training episodes of the RL agent are given
### Scenario1: Randomly executed episodes are available or inexpensive: 
In this scenario, we can consider that episodes of random executions of the agent are available. 
One example is when the agent is tested to some extent. but before final deployment, we want further test the agent using STARLA 
Another situation is when the RL agent is trained and tested using both a simulator and hardware in the loop [56].

In this situation, an agent is trained and tested on a simulator in order to have a ”warm-start” learning on real hardware [56], [57]. Since STARLA produces episodes with a high fault probability, we can use it to test the agent when executed on real hardware to further assess the reliability of the agent. In such situation, STARLA uses simulator-generated episodes and executes the newly generated episodes on the hardware.



### Senario2: Randomly executed episodes are generated with STARLA and should be accounted for in the testing budget: 
In the second scenario, we assume that the agent is trained but not tested so far and we want to test the agent using STARLA. Therefore, we need to use part of our testing budget for random executions, to generate the required episodes. 

More precisely, the total testing budget in this scenario is equal to:

The number of episodes in the initial population (generated through random executions of the agent) + Mutated episodes that have been executed during the search + Faulty episodes generated by STARLA (executed after the search)

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/23516995/168507459-62e14ed5-9ed0-445e-9f60-a28d7dc74999.png"> 
</p>


**Answer:** We find significantly more functional faults with STARLA than with Random Testing using the same testing budget. 


## RQ2: Can we rely on ML models to predict faulty episodes?

*In this research question we investigate the accuracy of ML classifiers in predicting faulty episodes of the RL agent. *

we use Random Forest to predict the probabilities of reward and functional faults in a given episode.
To build our training dataset, we sampled episodes from both episodes generated through random executions of the agent and episodes from the training phase of the agent. Episodes are encoded based on the presence or absence of their abstract states. We have 2 different ML models, one for predicting the probability of a reward fault and the other one for predicting the probability of a functional fault. we considered 70 % of data for training and 30% for testing.

**Answer:** Using the mentioned ML classifier and feature representation, we can accurately classify the episodes of RL agents as having functional faults, reward faults, or no fault at all.



## RQ3. Can we learn accurate rules to characterize the faulty episodes of RL agents?

*Here, we investigate the learning of interpretable rules that characterize faulty episodes to understand the conditions under which the RL agent can be expected to fail. *
For this reason, we need to rely on an interpretable ML model, in this case, a Decision Tree, to learn such rules.
We assess the accuracy of decision trees and therefore our ability to learn accurate rules based on the faulty episodes that we identify with STARLA. 
In practice, engineers will need to use such an approach to assess the safety of using an RL agent and understand the reasons of faults.
In this part, we assess the accuracy of trained models that extract the rules of functional and reward faults based on k-fold cross-validation.




![image](https://user-images.githubusercontent.com/23516995/168505819-7c835496-6a8c-400d-a85c-ce5367b603b5.png)

**Answer:** By using STARLA and interpretable ML models, such as Decision Trees, we can accurately learn rules that characterize the faulty episodes of RL agents.




Acknowledgements
--------------
This work was supported by a research grant from General Motors as well as the Canada Research Chair and Discovery Grant programs of the Natural Sciences and Engineering Research Council of Canada (NSERC).

Notes
-----



References
-----
1- [stable-baselines](https://github.com/hill-a/stable-baselines)

2- [gym](https://github.com/openai/gym)

3- [MOSA](https://ieeexplore.ieee.org/document/7840029)

4- [virtual vs. real:Trading off simulations and physical experiments in reinforcement learning with Bayesian optimization](https://dl.acm.org/doi/abs/10.1109/ICRA.2017.7989186)

