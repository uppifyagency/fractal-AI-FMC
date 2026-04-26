
Fractal AI
A Fragile Theory of Intelligence

Sergio Hernández Cerezo
Guillem Duran Ballester

FrAGIle

BOOK #2
“AGI Structure”
Version: V0.2

















Main researchers
Sergio Hernández Cerezo (@EntropyFarmer)
Guillem Duran Ballester (@Miau_DB)

Special thanks
Researchers’ families for suffering us in all the ‘eureka’ moments.
HCSoft for their unconditional support.

Reviewers



Contents

1 - Introduction	3
2 - Revisiting Fractal Monte Carlo	4
2.1 - FMC external functions	4
2.2 - AlphaZero approach	5
2.3 - General approach	7
3 - The Badger structure	9
3.1 - Initial approach	9
3.2 - General structure	10
3.3 - Expert deliverations	10
3.3.1 - Loop functions	11
3.3.2 - Expert functions	11
3.3.3 - Pooling the experts	12
3.4 - Workflow	12
3.4.1 - Inner loop communications	15
3.5 - Intrinsic reward	15
3.6 - Final overview	15
3.7 - Loop properties	16
3.8 - The loop structure of FMC	16
4 - Building the intuition	18
4.1 - Darwinian evolution example	18
4.2 - Bacteria example	19
5 - Entropic first principles in AGI	22
5.1 - Second Law of Thermodynamics	22
5.2 - Least Action principle	22
5.3 - Causal Entropic Forces	23
5.4. - Fractal AI principle	23
5.5 - Dissipation maximization	23
5.6 - Free energy principle (*)	24
5.7 - Empowerment	24
5.8 - Surprise minimization	24
5.9 - Algorithmic complexity minimization	24
6 - Research directions	25
Bibliography (*)	26

1 - Introduction
We can roughly define a general intelligence, artificial or not, as any mechanism that can automatically learn to control any kind of agent and adapt its behaviour to better fit in any evolving environment. Equivalently, we could also define AGI as any mechanism that learns to maximise the agent probability of surviving in any evolving environment.

From a pure algorithmic perspective, AGI is a black-box mechanism where inputs are a time series of observation vectors, and the outputs are the corresponding series of action the agent should take at each time step to maximise his chances of survival in the long term.

Such an algorithm would be one that, when left alone in any place, embodied in any agent, and under any kind of changing conditions, will struggle to learn how to survive and dynamically adapt to the changing conditions of the environment by its own, just as a human baby learns from pure observations to incrementally adapt to a wider range of situations autonomously.

The goal of this second book is to present the general internal structure of such an algorithm as a special Badger structure, composed of different nested loops with a precise structure that we will define, each one solving a particular sub problem,, and actively collaborating one with each other by sharing information in the form of functions.

As such, this book is not aimed to present any new algorithm in detail, but at building an intuition in the reader about the modules needed, its many similarities, and their internal interactions. In this reward, some leaps of faith will be required for the reader to accept the intuition that the proposed structure will, in fact, produce the general intelligent behaviour we expect. We will try to support the claims by inspecting the assumptions we made in the planning algorithm presented in book #1, and also by describing some simple but intelligent agents from biology.
2 - Revisiting Fractal Monte Carlo
In book #1 we define a planning algorithm, FMC, based on first principles of entropy maximisation, and inspired in thermodynamics. At the end of the day, we had just an efficient planning algorithm that, for no means, could qualify for AGI. 

For the sake of simplicity and for testing of the planning algorithm itself, a number of externally defined functions were needed. The most evident one for RL practitioners were the use of a perfect model of the world, a perfect “simulation” for the next state, as in the Atari game examples, that allowed the agent to know the consequences of actions in a deterministic and perfect way. In a more general setup, this model of the world needs to be learned, and the simulation replaced with a guessing of the most probable next states, producing a probability distribution over the available next states of the agent.

For every external function used in FMC, an equivalent learning process must be defined instead, and only then, the FMC could had evolved into a full AGI, one that need no human defined functions as it learns all the needed functions just from its observations.
2.1 - FMC external functions
In this regard, it is interesting to start by enumerating every external function used by FMC:

Observation(): A function that outputs an state vector from the actual agent sensors. For instance, in a 2D physic setup like the rocket examples, the observation was a small vector with (x, y) position, (vx, vy) velocities, etc. In the general setup, we do not directly observe those numbers, but some raw sensor values like RGB pixels from video frames and, from this, we need to build a meaningful vector -an internal representation- like the vector <x, y, vx, vy, … >.
Distance(state1, state2): As our states were already small and meaningful, almost any distance calculated on them was really informative. In the general case, a distance over the observation vectors can be almost meaningless, so again, the agent have to learn and build a distance between states that is as informative as possible.
Simulation(state, action, dt): As in simulation software, we didn’t predict but calculate the ground-truth next state. In the general case, we need to learn a world-model, a predictor of the probability distribution over the available next states.
Reward(state): A function that scores states so we can compare two of them and know how beeter one is in relation to the other. This is by far the most complex function to be replaced by learning, and the one with the highest impact in the resulting agent behaviour.

As we enumerate the external function in FMC, a pattern appears:

Unsupervised learning tasks: much like in the AlphaZero approach:
Observation: it corresponds with learning an embedding of the observation that serves as an internal state in a latent space. A variational autoencoder can do this unsupervised: given an observation, a bottle neck NN compress it and produce an embedding, then, a second NN decompress the embedding producing an expected observation.  Both NNs are trained to minimize the differences between actual and predicted observations.
Distance: once the observations are replaced with embeddings, and due to the property of embeddings that similar states correspond to near points, using the distance between embeddings instead of raw observations is equivalent to building the most informative distance.
Simulation: as the agent randomly moves on the environment, datasets of rollouts consisting in (initial state, action made), (next state visited, next action made), etc. is continuously produced. It is then a matter of using a form of LSTM structure that learns to predict the next state following in the series.
Planning task:
Planning: actually performed by FMC:
Efficient scanning of of the future state space.
Automatic balance between exploration and exploitation.
Producing a probability distribution over the agent available actions.
Prior over the actions: In FMC all available actions were equally probable. Learning to build priors over those actions would be necessary to boost the efficiency of the search. 
Reward function generator:
Reward: score one state as more or less rewarding and redefines the reward function itself over time to maximize adaptation.

This is the reason why this book will propose dividing the AGI into three functional modules or loops: learning, planning and rewarding.
2.2 - AlphaZero approach
Learning, planning are rewarding are already familiar to Reinforced Learning (RL) researchers:

Learning is related to artificial neural networks architectures (ANN).
Planning is related to algorithms like Monte Carlo Tree Search (MCTS).
Reward is related to expected reward like q-values and actor critics in RL.

The RL architectures AlphaZero and AlphaStar by Deep mind, the ones that could beat games like Go, chess or StarCraft II, are actually a stack of modules doing basically this:

Uses VAE to build an embedding of the observation.
Uses LSTM to build a predictor of the next embedded observation.
Uses RL concepts like Actor critic to estimate the expected increase in reward for each action given the actual state, normalised into a prior over the actions.
Refine this prior using the planning algorithm MCTS: look some moves ahead to make sure what the priors suggest is actually a good idea.

Training all those parts is done in a separate process and using different techniques we will not cover here.

Is then AlphaZero considered as an AGI? No, the results are impressive, but it doesn't qualify for being “general”, not efficient:

A version that can learn to play chess can not learn to play StarCraft II or GTA-V, a lot of human effort is needed in this conversion, and most of the time, after a lot of trial and error,  you simple can’t find the way.
Once you find the way, training the model to actually learn consumes huge amounts of computing power (the amount for training StarCraft 2 is measured in weeks of running the whole google apps ecosystem -search, gmail, docs, meet, youtube, etc- on the world). In fact, each iteration of the trial and error previous process also consume quite a lot.
A version that had learn to play a game is not capable of learning a second game without losing its ability to play the first, and needs the same amount of time to learn it. The knowledge acquired in one game doesn’t “transfer” to the next one.

This document aims at presenting a general structure for an algorithm, similar to AlphaZero architecture, but where all the internal algorithms (learning, planning, etc) are similar to FMC (cellular automaton based, thermodynamics inspired), are self similar between them, and sefl adapt to the problem and reuse what it learns when learning new tasks.
2.3 - General approach (*)
By inspecting the AlphaZero architecture, a nice unifying patterns emerges: if we consider the state of our agent as being not only the observation, but a concatenation of all the vectors involved, the “full state” of the agent, we have that:

Full_state = <Observation, Embedding, Next embedding, Action, Expected reward>

Observation:
Ini: receives a full state from the last step.
New: builds a new observation of the environment.
Update: the “observation” portion is replaced with the new one.
Embedding:
Ini: receives a full state from the previous module.
New: builds a new “embedding” from the previous portions (observation).
Update: the “embedding” portion is replaced with the new one.
Predicting next state:
Ini: receives a full state from the previous module.
New: predicts the next embedding from the previous portions (embedding).
Update: the “next embedding” portion is replaced with the new one.
Planning:
Ini: receives a full state from the previous module.
New: selects the best action from previous portions (embedding, next embedding,).
Update: the “action” portion is replaced with the new action.
Reward:
Ini: receives a full state from the previous module.
New: gets an expected reward from the previous portions (  ).
Update: the “reward” portion is replaced with the new reward.

The pattern is clear: each module updates its portion of the full state with a better version and passes it to the following module in a closed loop, while learning to do it better over time. Again, the heart of each module, the “new” and “train” parts, are different for each level. Our intention is then to define a single “new + train” schema that could be used in all modules. 
3 - The Badger structure
In order to give this AGI modules a more convenient general structure, one that encode not only its internal structure but also the information flow, we will consider a special case of the Badger schema as the most natural fit. In the Badger approach, a series of nested loops are defined, each assigned a specific purpose and, eventually, populated with experts. By defining rules of communication at three levels (between adjacent loops, between experts and between its inner loops), the structure and working internals of an AGI can be clearly stated.
3.1 - Initial approach
Elon Musk wants to build the perfect car, but an experiment said the grip was not ok. Elon asks an expert car driver to test the car and confirmed a problem, the grip was not ok. His hypothesis is that dumpers are too soft, and it would be ok if they were +10% harder.

Elon ask an engineer expert in dumpers to design one to the driver specification. He comes to a good design that uses a liquid that is +10% more viscous and 5% more compressible than actual. His hypothesis is that his design, with this liquid, will fit the needs.

Elon ask a chemistry professor to design a chemical master recipe to get a liquid with the desired properties. He sends a recipe, his hypothesis is the formula will produce a liquid that fits. Elon produces the liquid, test it, and it doesn’t fit. Send the result to the expert, it updates its priors, and send a new formula that, finally, delivers.

Elon produces the new dumper, add the liquid, test it, and doesn’t deliver. He send the results to the engineer who updates his priors and produce a new design, with a different liquid properties, so Elon goes back to the chemistry professor who sends the correct formula in the first try. Elon tests the new dumper and deliver, replace the dumpers and call the driver again.

A new experiment on the new car finds that the grip is still bad. The driver then update its priors and suggest using dumpers +5% harder. Elon repeat this same process over and over until, months later, the driver is satisfied and the experiment find grip is ok. All the experts gained expertise by updating the priors they had in the process.

Elon realises that having many experts of each kind, he could had this done in parallel and have a perfect car in days instead of months. Elon also sees a pattern in the process and writes an internal procedure for getting perfect cars: he noticed how the problem, detected by experiments, was defined from a topmost expert in driving down to the lower level experts in chemistry, while the solution was built from the bottom level, up to the topmost level. The solutions were checked against reality by an expertised tester in a lab. When an expert fails, he sends their negative result to the circle of experts working for him in the lower level, and they update theirs prior. When it is positive, he send it to his upper level expert to continue. Finally, Elon can take a rest.

3.2 - General structure
As in our case we consider our full state to be <observation, embedding, next embedding, action, reward> -that we will name as portions P0 to P4, so the full state is <P0, P1, P2, P3, P4>- we will need many experts on all those tasks, and structure them in levels ranging from 0 to 4 representing the tasks of “Observe”,  “Embed”, “Predict”, “Plan” and “Reward” respectively.

The Badger structure proposed has a single innermost loop of level #0 loop (the tester), an “Observation” loop representing the sensorimotor system of the agent, and a single outermost loop of topmost level #4 -the driver- populated with a nested structure of loops of intermediate levels experts. Both loops interconnects theirs inputs and outputs states forming a closed loop:
 


3.3 - Expert deliverations
How the loops and experts in this structure communicates, and the effect of this communications, is the most important concept in a Badger structure, so let’s go for this.
3.3.1 - Loop functions
The sensorimotor loop #0 has no internal structure, instead, when the system receives an “Input state” (a new car), it just replaces the observation portion P0 with the readings from the sensors, leaving the rest intact and producing a new full state. We could say that the sensorimotor loop has a “function” for this:

loop.function(S = <P0, P1, P2, P3, P4>)  = <Sensor readings, P1, P2, P3,P4>

We will consider the function as being a composition of five functions LF0 to LF4 for future clearness:

	loop.function(S) = <LF0(S), LF1(S), LF2(S), LF3(S), LF4(S)>

Then, for the sensorimotor loop, function LF0  has no inputs and returns the sensor readings, while functions LF1 to LF4 are just the identity, so we will abbreviate it:

loop.function(S) = <Sensor_ readings, Id , Id ,Id ,Id >

All loops will have a loop function that transform one full observation to another. Loops communicate by passing their corresponding functions. The idea is that a loop of a given level N has expert in how to update the portion PN of the full state, but have no idea on how to update all the other portions. The loop function, received from its outer loop expert of level N+1, are instructions from experts of  higher levels about how to properly update PN+1 to P#Levels.

It is then clear that the loop function of the outermost loop #4 has to be the identity: there are no higher levels expert to ask!
3.3.2 - Expert functions
Experts in a loop #N are specialised in updating PN portions (let’s call them “expert #N”), so they take this inherited expertise -the loop function- and add its own function FN() as the ideal candidate for the task of updating PN: experts haves “candidate” functions that, when used in the input state, produce different “candidate” values for PN.

For instance, if the problem were about interpolating a curve given some points, each expert would prefer a different candidate method (linear interpolation, splines, etc) that would produce a different “candidate” value (2.7, 3.5, etc).

The experts in loop #4 are supposed to decide on which reward function to use. A generic reward function will input the first 4 portions of a full observation <P0, P1, P2, P3> and output a valid portion P4, in this case, an scalar “reward”.  We will say that the “internal” function of the expert is:

reward_function(<P0, P1, P2, P3>)

We will then say that an expert of a level #4 has an internal function he represents (the reward function) and a general function identical to the loop function except for the 4th component.

function(S)  = <LF0(S), LF1(S), LF2(S), LF3(S), reward_function(P0, P1, P2, P3)>

In general, the internal function of an expert of level N will input the first N component of the state and output the N+1-esim component PN and its general function will have the first N component of the loop function, the N+1-esim components is its internal function, and the identity in the rest of component.

function(S)  = <LF0(S), ... LFN-1(S), internal_function(P0, ... PN-1), Id, … , Id>

Note: we could assume that, for instance, the reward functions are ANN of the appropriate shape, but any other family of functions, as far as its parameters can be vectorized, will be ok. 
3.3.3 - Pooling the experts
The loop will decide which of its experts advise is the better by running a pool: each expert propose a function and a value, and have a population of inner loops with experts of lower levels that use his function -interpolation methods in the previous example- because, for their use cases, worked better than others, or because they were assigned to this specific expert.

The internal loop then represents the process of deliberating: inner loops, experts in lower levels, solve similar problems with the function his expert gave them, and communicate their results. Eventually, an follower of expert A will find that a follower of expert B is getting better results, and change side, adopting the B teachings an becoming an expert of the B loop.

After the deliberations, experts in the loop will have different relative populations: the most populars are followed by a higher portion of the lower level experts, so the loop knows how to proceed to generate its output state: average the experts options weighted with their relative populations. 
3.4 - Workflow
Let’s have a look a the outermost loop and then comment on the general workflow:



As viewed from this topmost level, the workflow, properly divided into functions, is as follows:

First, we initialize the whole structure by initializing the outermost loop #4:

Initialize():
Each expert in the loop:
Initializes its internal function’s parameters randomly.
Builds its function by concatenating loop function and internal function.
Each of the inner loops:
Assign it to a randomly chosen expert.
This set its loop function to the expert function.
Call its initialize function.
Recursively initialize all loops.
Recursion stops at level #1.

Once initialized, the loop receives a first observations (a car with a low grip) and activate and endless loop:

Activate(new state):
Update “Input State” with “New state” (with a new observation P0 from the sensors).
For each of the experts:
Apply function() to Input State and store the output as the expert “option”.
For each of the inner loops:
Set inner loop input state to the expert option.
Set inner loop function to expert function.
Until a new observation arrives:
Do a step of the internal loop.
Send “Output state” to the motors, so the action chosen is performed.
Restart the loop with Activate(new input).

Each time we do a step on the internal loop:

Internal_loop_step():
Each inner loop call its internal loop step.
Updates its  output state.
Calculates its intrinsic reward.
Each inner loops do a communication step:
Read some other inner loops’ output states and intrinsic rewards.
With some probability, clone the state of one of the other state.
The inner loop receives a new initial state, so it will reactivate.
As a side effect, the inner loop can change of expert, altering relative populations of experts and updating the inner loop function.
Update “Output state”:
My experts know nothing about the first 4 portions P0 to P3, so update them with the average of the inner loops output states that used my loop function for this.
My experts deals with the reward portion P4, so it is averaged from the experts’ options, weighted by the experts’ relative population.
If there were more portions left, use identity function.
Update its intrinsic reward (to 1 in the outermost loop).

This algorithm will recursively use the whole stack of layers and all their loops of experts to perform all the magic, but first we need to generalize to the intermediate level loops, as they are different from the outermost layer:

Outermost loop doesn’t belong to a higher level loop or experts:
It doesn’t need to communicate with other loops in his level.
So it doesn't need any intrinsic reward.
It continues looping until a new observation comes (a new car).
As it never clones, only initializes when a new observation arrives.
Its output state is directly sent to sensorimotor system at every reactivation.

If we inspect one of the level #3 loops in the image, we would find a very similar structure, and the pseudo-code above is same, just generalizing level #4 to level #N.
3.4.1 - Inner loop communications
The communication procedure, as stated in the pseudo-code, will allow an inner loop A to read the states of a fixed number of states from other inner loops, not necessarily in the same expert loop, and alter its state by cloning the state of some of those other loops, with the possible  effect of A leaving its expert and following another, modifying the relative population of both experts.

As a matter of fact, we will always use the idea of virtual reward and clone probability from FMC but, for the sake of generality, here we allowed for any other method you may find of interest, including the MCTS approach, for instance, as long as it produces the cloning of states.
3.5 - Intrinsic reward
Each level must be given an intrinsic reward the loops will use to score themselves after evolving. When this reward is aligned with the purpose of the level, the result of the workflow of the Badger will be modify the output state so it maximizes the intrinsic reward (for Elon, this is that the experiment confirms the properties predicted by the expert, the cross entropy of expected and confirmed properties). Which intrinsic reward fits each level purpose is the cornerstone of the whole idea.
3.6 - Final overview
If we should reduce the AGI problem to find a function of the sensor inputs that output the most intelligent activations on the motor (a function that test the car and gives you the right modifications in all parts of the car to make it a perfect car), we would find that the search space is really huge and, even if we decide to try, we have no practical way of measuring how intelligent the behaviour is.

In this approach, we substitute this space into a much bigger one by considering the state as not only the observation but the full state of the agent, a composition of several vectors (in the Elon example, the chemical structure of the liquid, the design of the dumper, and the opinion of driver), each one representing one aspect to be solved: find the best embedding for the observation, find the best predictor of the next embedding, find the best action sampling function and, finally, find the best reward function.

For each of the tasks, we then define a intrinsic reward function that must be maximized, like maximizing the cross entropy between predicted next embedding and the actual one.

Once we had this, we used a nested Badger structure were each level of loops were dedicated to find the best function for its corresponding portion of the full state. By allowing the functions used to be perturbed at each step, and using the intrinsic reward to score them, we applied a FMC algorithm to maximize the intrinsic reward. The function improved as in a Evolutive Approach. As we could use as perturbation as a train over a batch of examples, gradient descent can be added to the mix if desired. 

This process is made as the Badger structure infers the correct function values for the observation received. This process of inferring is based on the existence of experts at different layers that can communicate, so a single topmost level loop was presented with the task of filling with appropriate values not only its portion of the full state, but the whole.

This loop send the task of filling all the portions he doesn’t know how to fill to the inner level loops, that in turns, pass it to even lower loops until the level #1 loops receive the task and, and fill the last portion of the vector (the chemical formula for the fluid), passing it back to layer #2 for filling the second portion (the design of the dumper), recursively until the topmost layer have all portions filled, but the last one. He fills it by averaging the opinions of its experts weighted by its relative populations.

The learning then occurs first in the lowest level loops working on the embedding function, then, once the embeddings makes sense, the second layer loops start to learn good next embedding predictor up to the outermost level, were the single top level loop finally learns good reward functions.

The difficulty is then in finding the right intrinsic reward associated with all the layer we should define, as the rest is managed by the structure.
3.7 - Loop properties
In order to totally understand each possible level of loops, we should define all of the properties commented so far:

Generic loop, Level N
Purpose
What the level is supposed to be for.
Portion
Portion of the agent full state being updated.
Expert function
Function used by expert for update the “Portion”.
Intrinsic reward
Reward of the loop aligned with its purpose.
Inner loops
Which kind of internal worker it uses.


3.8 - The loop structure of FMC
Now that we are introduced into the Badger structure, we can try to express the planning algorithm FMC presented in book #1, and internally used in the communication procedure on all layers, as a loop structure:


FMC loop
Purpose
Define a good probability prior over the actions.
Portion
Probability distribution over the available actions.
Expert function
Relative population of walkers.
Intrinsic reward
Cross entropy of distribution of reward and walkers.
Inner loops
Walkers that moves by predicting the next state.


4 - Building the intuition
But just enumerating the modules, its properties and their interconnections in a badger structure will probably not  be enough for the reader to build an internal intuition of the AGI processes. We will use here an evolutive approach: we will describe some of the simplest intelligent agents and try to identify all the modules, tasks and loops we commented before.
4.1 - Darwinian evolution example
The most basic of the intelligences we could think of is the intelligence present in the darwinian evolution of especies. At this level of abstraction, we won’t care about the individual in the population but on the especie as a whole, that correspond to our “agent”, and its evolution in time. As evolution is a slow process, the time step usually considered to appreciate the dynamics of evolutions is the average lifespan of a generation of individuals.

If we consider the DNA of each individual as a proper encoding of it, and compare it among the population, we would find that only a portion of the DNA is common to all individuals, while other small parts can vary. If we simplify the setup as to consider that only one small portion of the DNA changes, and that it presents two possible mutations, our population can be divided into two separate groups, with the DNA being essentially the same in each group. Each group is represented by one expert in the structure.

The level here corresponds to the evolution of the especie itself: its full state portion is the DNA, the Input step is the most common DNA among the population, and the Output state is the same common DNA in the next generation.

Happily, evolution doesn’t need to know the agent state (the predominant DNA), nor the predictor of the next state: we just sit and wait some years until the next generation, sample the DNA on the population, and this is your next state.

If we now try to incorporate this darwinian evolution into our badger representation, we would need a new outermost layer with the following properties:

Evolution loop #5
Purpose
Define a good DNA for the population.
Portion
DNA vector.
Expert function
Total offsprings of an individual with this DNA.
Intrinsic reward
Population entropy ≃ population size x diversity.
Inner loops
Individuals of the population.


Experts are subgroups of individuals, each one corresponding to the original concept of agent that we initially represented with a 3 levels loop structure:



4.2 - Bacteria example
A cell needs some sensors and actuators before it can do any better than random, so we will fast forward evolution to alter the DNA part of the agent full state, until we find the first cells with some sensor and a motor: the flagellum bacteria.


In this image we will be interested in the following aspects:

There are some long light sensors -in the form of yellow hairs- in the head section of the cell.
There are also some shorter light sensors around its body.
There is flagellum in the back that, when activated, will push the bacteria ahead.

In order to simplify the example to a minimum, we will consider all sensor in the head as a single sensor s1, and the rest of the sensors in the body as a second sensor s2. The head sensor s1 will produce signals ranging from 1 (direct sunlight) to 0 (total darkness), while sensor s2 will be set to range from 1.2 to 0.2, for good reasons we will discover soon.

The “wiring” of the two sensors toward the actuator is simple and direct: if signal from head sensor s1 is higher than the reading from body sensor s2, flagellum will activate. If the bacteria internals were coded in Python, the code should look like this:

f.active = (s1.signal > s2.signal) 

This bacteria will show intelligence in the individual level: it will move, on average, from dark areas toward sunny ones, increasing the individual chances to survive. Also, in the extreme cases of total darkness or full exposure to direct sunlight, signal will be double at s2 and the flagellum will not activate in vain, shaving precious energy:

Total darkness environment:
Head sensor s1 signal will be in its lower bound, so s1.signal = 0.0
Body sensor s2 signal will be in its lower bound, so s2.signal = 0.2
Flagellum active = (s1.signal > s2.signal) = False.
Direct sunlight environment:
Head sensor s1 signal will be in its upper bound, so s1.signal = 1.0
Body sensor s2 signal will be in its upper bound, so s2.signal = 1.2
Flagellum active = (s1.signal > s2.signal) = False.
Head receives direct sunlight, body receives a half:
Head sensor s1 signal will be in its upper bound: s1.signal = 1.0
Body sensor s2 signal will be in its mid state: s2.signal = 0.6
Flagellum active = (s1.signal > s2.signal) = True.

5 - Entropic first principles in AGI
All of the levels previously defined corresponded to a particular entropic intrinsic reward, ultimately representing some kind of entropic “principle” the AGI seems to be following.

Lets then visit and comment on a number of different first principles proposed in the fields of physics, neuroscience, biology and artificial intelligence, in a desperate search for insights of new levels we haven’t discovered.
5.1 - Second Law of Thermodynamics
This law states that any closed system evolves in such a way that maximizes the entropy of the system. As to say, the system has a list of available next states, each representing an action that can be applied to the system, like a force, then, somehow scan its future and get informed about the entropy gain each of those options represent, and then choose the action that will take the system to the available next state with the highest entropy. More precisely, we would say that the system sample one of its available next state with a probability proportional to the entropy gain they represent.

In some sense, it is saying nothing, a tautology: if we consider the states as macro states, each one representing a varying number of microstates, where all the micro states are considered equiprobable -or, equivalently, that our space is ergodic- then, all it is saying is that, next available macrostates will be visited by the system with a probability proportional to their number of microstates. So that more probable macro states are visited with an equally higher frequency by the system.

In the other hand, it is a very powerful mechanism, and it is in fact present in the FMC algorithm used as communication procedure on every layer we defined, so this one is already present in the levels.
5.2 - Least Action principle
In a somehow mirrored version of the entropy maximization principle of the second law of thermodynamics, it also holds that any closed system evolves in such a way that, the “action” applied to the system is minimized. In the framework of physics, this action is defined as the Lagrangian of the system, the difference between the increments of the kinetic potential energy of the system. When a system evolves in a given trajectory, it chose the one with the least integral of the Lagrangian, the one with the least action.

Basically it means nature is lazy and will not do things in the hard way if there is another way of doing it more easily, it wants to do things in the less energy consuming way.

Both principles are equivalents, so no need to worry. If they produce the same trajectories on the systems, they are the same thing in different words. 
5.3 - Causal Entropic Forces
This forces push the system to states with high causal entropy, where the cone of trajectories, of the different things that can happen to the agent, is diverse and rewarding.

Causal Entropic Forces are in the heart of the FMC algorithm and as such is present of the planning loop and the communication procedures. 
5.4. - Fractal AI principle
Is the principle stated in the book #1 and used in FMC, stating that the swarm of walkers evolves in such a way that the density of visits is proportional to the density of reward, so the mutual information is maximized. It is equivalent to the entropy maximization principle.
5.5 - Dissipation maximization
For some biased reason, we placed evolution based in DNA as the first level in the list, but what about the evolution of complex structures that don’t have DNA? What about the structures at a molecular level, that grow in complexity and form stable structures?

Jeremy England took care of them and proposed that, those structures evolves into more complex forms that, for being able to keep its structure over time, are able to dissipate more and more heat to the environment.

I think this a brilliant idea, but not a genuine first principle. I think the first principle is that complex structures need to keep its internal entropy in a low and stable level to be stable, like humans need to keep a stable corporal temperature, and a few more grades can kill us, destroying our precious complexity into dust. The only way to keep internal entropy low is by exporting it out, augmented. We need to do a lot of internal work, use a lot of energy, and dissipate a big portion of it, so the entropy in the outside grows far more that is not growing in the interior. Dissipation is literally the smoke that points to the fire.

But a genuine new level can be produced out of it, one even outer that the evolution level, and one that is still perfectly visible on us: a human dissipates far more than a mice, this more than an insect, and this much than a unicellular organism. When you burn a fire to keep your temperature stable in the cold night, you do so at the cost of dissipating a lot more heat than usual, but this allows you complexity to survive one more day. A human driving his car or flying on a plane dissipates even way more, but this allows our complex society be this complex.

The intrinsic reward associated with such a level would sound as “minimize the changes on your internal entropy”. It can be reformulated to sound better this way: maximize the mutual information of your ideal internal state and the actual one. 
5.6 - Free energy principle (*)
In the same scenario as in the evolution of complex structures, a different principle has been proposed by Karl Friston, basically stating that complex system evolves so that the free energy available to the structure, the one that can be used to do work, is maximized.

In some way it is equivalent to the previous one: in the dissipation maximization, we wanted to keep our structure stable, and did a lot of work and dissipated a lot of heat for this. If the free energy depletes, no more work can be done, so the structure can not be stabilized and, eventually, collapses.

This is why we all prefer an electric car with a big battery, plenty of power to drive for longer, so we don’t get stuck in the road in the drive to work and our officine collapse without us. We want plenty of free energy to do more things and keep the structure going.

The theory also predicts that, those complex structures, instead of finding a state of maximum free energy and stay there forever, …

More text here...
5.7 - Empowerment

5.8 - Surprise minimization

5.9 - Algorithmic complexity minimization

6 - Research directions
A layer can be split into two or more layers, each one representing a subspace of the previous subspace, each using its own intrinsic reward that, in some way, combine to the one in the initial layer. We saw this when the learning layer was divided into two (embedding and prediction). Also, new layers can be added in the topmost level, as we did with evolution of species with DNA.

This rises some important questions:

Are there more intermediate layers we could have missed?
Are there more levels below the observation level?
Are there more levels above the evolution level?
Are there missing purposes in the schema?
What about memory?
What about abstract reasoning?
Are them new levels or just emergent properties?
What is Consciousness?
Is it a level we haven’t explored?
Is it an emergent behaviour from other layers?
Or is it a complex mixture of both things?
Are there other entropic intrinsic rewards we haven’t used?
Do they show the existence of levels we didn’t noticed?
Which purposes do they serve for?
Is the AGI presented a counter example of the “Free Lunch Theorem”?
Given that humans seems to perform better that random in most of the problems I can think of makes me confident that it could be.
In the other hand, if a totally alien civilization interacted with the humans, would our interactions with them be any better than random?
Given the fact that the most probable leader in the human side would be a given president, I strongly advocate for the reign of the mentioned theorem. 

Bibliography (*)
