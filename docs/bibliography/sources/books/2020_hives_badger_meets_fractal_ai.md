

Mixing Badger and Fractal AI







Honey Badger meets Fractal AI Hives
Index

Introduction
Generalised Badger Structure
Breaking the learning problem into “modules”
Loop levels
Tasks distribution
Expertise levels
Internal structure of the Badger
Internal structure of the different levels
Loops internal structure
Expert levels internal structure
Expert internal structure
Message structure
Learning as a structural collapse
Example of a Generalised Badger
Rewards
Initialization
Training step
Expert
Experts level
Level 1 loop
Level 2 and 3 loops
Training phase termination
Inference step
Enhancements
Architecture optimisation
Initialization optimisation
Matrices parameters optimisation



Introduction
The Badger structure is a general purpose RL structure where many of its part are intentionally not totally defined in order to accommodate different approaches in one general schema. 

Fractal AI, in the other hand, provides a general purpose planning algorithm based on first thermodynamic principles of entropy -mutual information- maximization that is general and efficient but lack of a more general and wider structure where to accommodate other aspects of intelligence like learning.

Both approaches can benefit from each other if mixed properly, giving a more concrete, yet general method for doing RL and, more generally, getting us nearer to a true general AGI in a longer term.

The present document proposes a generalization of the Badger structure to accommodate all aspects of solving a general problem. We will use a single outer layer, acting as a  main loop controller, plus a varying number of inner loop levels, each one focusing on optimizing one specific aspect of the general problem, working in a onion-like set of stacked layers, each one sporting a Fractal AI swarm responsible of the optimisation part. In the lowest level, the experts will use all the above layers to build a full-fledged version of the planning algorithm as proposed in the standard Fractal AI setup.

We will also define a topology for the communication between levels and experts and the effects those messages should produce in them by mimicking the internals of the Fractal AI working principles.

The resulting structure will still leave a great degree of freedom for choosing different algorithmic approaches for any of the inner loop levels.
Generalised Badger Structure
We will consider a nested structure or inner-loops controlled by a single outer loop , where each loop contains not one but several lower level loops, each one containing its own collection of even lower level loops recursively, down to the lowest level, populated by experts that, in turn, use a Fractal AI swarm of walkers for planning and selecting their next best actions.

The presented structure extend the standard Badger structure in those aspects:

Any loop will contain a collection of lower level inner loops.
The loops are divided into levels, forming a tree of loops.
The lowest level loops are populated by experts, representing “virtual agents”.
Inside experts we find a final layer of swarms of walkers, representing “virtual experts”.



Each level is assigned the task of optimizing one particular aspect of the decision processes the experts perform inside the Fractal AI algorithm, namely their expert policies.

Using Fractal AI as the planning algorithm for the experts is only possible if they can accurately predict the next environment observation, so experts will need access to a learned world model.


The overall goals of this structure is to build, on an semi-supervised way (an external goal is defined, like the score in a Atari game or, in the absence of score, keep alive as long as possible),  a reliable world model and a convenient reward function that, together with the planning algorithm, makes the agent to efficiently solve any emerging short term tasks needed in order to maximize the long-term, intrinsic or extrinsic, general goal.


In this document we will be considering the case where the expert policies are ANN based, but actually they could be replaced by any equivalent method, as far as the input-output meaning is kept.
Breaking the learning problem into “modules”
In order to properly distribute the training problem, it is highly convenient to break the problem into 3 different modules, each one representing an ANN. At the expert level, all those modules will be used as functions inside the Fractal AI planning algorithm.

Embedding module: transform observations into embeddings in a latent space (VAE).
Prediction module: predicts the next embedding from a time series of them (LSTM).
Reward module: defines a reward function over the latent space of embeddings.
Loop levels
The different levels the loops can fall into and their corresponding aspects to be optimized could be laid out at this:

Outer loop: The main controller level.
Inner level 4: Optimises the NN architecture of the three modules (optional).
Inner level 3: Optimises the parameters of the Embedding module.
Inner level 2: Optimises the parameters of the Prediction module.
Inner level 1: Optimises the parameters of the Reward module.
Expert level: Experts plan on the next action using Fractal AI.

Interestingly, we could consider the Fractal AI used by the experts as an additional level without breaking the Badger structure of the whole:

Fractal AI level: Walkers randomly perturb and clone following Fractal AI algorithm.

In most of the real world use-cases, some of the upper levels may be unnecessary: In many cases, the architecture of the ANN can be considered fixed, in others, the reward function could be given (like in a trading bot where reward is basically money earned, or in an Atari game where reward is the displayed score), in other use cases, embedding and prediction can be learned separately, etc.

Also, the first inner level dealing with the architecture could be broken into 3 different levels, each dealing with the architecture of one of the modules. This, along with the swarm level commented, makes for a beautiful overall structure of four pairs of levels working together:

Outer loop: The main controller level.
Inner level 3.A: Optimises the NN architecture of the Embedding module.
Inner level 3.B: Optimises the parameters of the Embedding module.
Inner level 2.A: Optimises the NN architecture of the Prediction module.
Inner level 2.B: Optimises the parameters of the Prediction module.
Inner level 1.A: Optimises the NN architecture of the Reward module.
Inner level 1.B: Optimises theparameters of the Reward module.
Expert level A: Experts plan on the next action using Fractal AI.
Expert level B: Walkers randomly perturb and clone following Fractal AI algorithm.
Tasks distribution
Despite the big number of loops and the varying number of levels we can arrange them into, the Badger structure is a container of it most basic elements: the experts.

Having loops of loops of experts is a very convenient way to break the problem of training into modules, and those into loops, so each loop can train on a separated GPU unit, making it extremely easy to distribute both the training and the inference phases, also allowing for adding or taking loops and agents from the structure, at any moment, a seamless task.

But at is core, all those loops can be thought as just being containers of a number of experts and, any property you could inspect from a loop, will ultimately be an average of the same property over the collection of experts inside this particular loop.

So the expert is the “atom” of the decision taking process and, as such, it will sport a full set of tools that will make of it an “independent” expert, one that can deal with the problem, better or worst, by its own.

Loops, in the other hand, are the atoms of learning, the units of training, working to serve their experts the best trained ANN they can collectively craft.

Finally, in the deepest level, walkers in the swarms used by experts to decide are the atoms of exploration. Their task is to scan all the possible outcomes of a potential action our expert could make by walking the causal path of its future consequences.

Loops: atoms of learning a world model and a reward function.
Expert: atoms of decision making by selecting the best actions.
Walkers: atoms of exploration of the future outcomes of action.
Expertise levels
The agent could ask the Badger structure for the best action by asking any of the loops from any of the levels present in the structure. Each loop will produce a prior over the actions as the average of the N priors obtained from its N lower level inner loops. Recursively, it will be finally equivalent to the averaged actions from all the experts contained in its lowest loops.

In this regard, each loop of any level can be seen as an expert itself, as we can use it to make a decision over the available actions. The highest the level, the more expertise will be averaged into the decision, so the more robust it will be. Also, the parameters of the NN at the different modules can also be averaged at any level, should we ever need to use a single expert instead of a badger of them in a simplified inference case.

In training phase, all loops from all levels are used in order to optimise all the corresponding modules, making the process to be expensive as we consider more levels and higher per-loop populations.
Internal structure of the Badger
Everything in the resulting Badger structure, including outer loop, inners loops, expert levels and experts themselves, can be seen as special instances of a swarm, while loops, experts levels and experts are each of a different kind.
Internal structure of the different levels
We can then divide the different levels of the Badger structure into 3 families, each one doing a different internal process: loops do learning, agent levels do planning in a federated form, and experts do planning to find the best trajectory, a sequence of (state, action) ending on a high rewarding final state.
Loops internal structure
Loops, both outer and inner ones, are a mixture of ANN standard training and a swarm of inner loops that manages the population.

For instance, a loop in level 3 correspond to a particular matrix of parameters for the embedding module, plus a collection of lower level loops, each one sporting this particular embedding parameters plus a different version of the predicting module ANN parameters.

Thus, a level 3 loop will be responsible of two main tasks:

Train its embedding ANN using a batch of N rollouts containing the best trajectories found by each of its N inner loops population of experts. At each level, a different portion of the trajectory is used, depending on the module being trained.
Consider those N inner loops as a swarm, where the reward assigned to a inner loop is the best reward found by their corresponding population of experts, and the distance between two inner loops is the distance between the parameters in their prediction modules ANN.

This allow the loop to both train its embedding and also do an evolutive strategy over their population of inner loops using standard Fractal AI methods.
Expert levels internal structure
A level of experts can be considered as a special kind of loop, in the sense that they do not have to make any learning (all the experts in the level share a common, fixed set of modules ANN parameters), but they have a collection of experts that can be using different internal parameters (for instance, a different exploration vs exploitation ratio).

The vector of internal parameters plays here the roll of ANN parameters in the upper levels, but you don’t have to learn on this. Then, this level is just a swarm of experts, where an expert reward is the final reward obtained at the final state each expert reach using the planning algorithm, and the distance between two experts is the norm of the difference between their parameter vectors.
Expert internal structure
An expert is basically a Fractal AI swarm generating a high reward trajectory by using a FMC version of Fractal AI where each action is calculated over a fixed time horizon, thus, the path generated will consist on a time series of (initial state, action taken, predicted next state).

The pairs (predicted vs actual next state) on the trajectory will be used in level 3 loops for training the embedding modules and, in level 2, for training the prediction modules.
Message structure
The fact that a swarm is used, one way or another at each existing level actually define the way messages are structured.

Each swarm enforce its walker members (inner loops, experts, etc) to contact with two randomly choose ones at each step. The message passed is the state of the other two walkers (the parameter matrix on a loop, the parameters vector on a expert, etc), and the calling unit always use them to:

Define a distance to one of the other walkers.
Read the Virtual Reward of the second walker.
Clone to the second walker’s state if VR is higher than mine.

The topology of the connections is then totally randomized, as each node, at each step, is pulling the exact same message (state, reward, virtual reward) for two randomly chosen other nodes.
Learning as a structural collapse
Apart from having each loop training its own ANN parameters independently of the other, it makes for a really interesting global learning approach to consider learning globally this way: as the training process evolves, we could consider one of the modules -or levels- to be already learned when the averaged parameter matrix coming from the different inner loops are not evolving over time anymore. For instance, if the embedding NN freeze its parameters after some training, we can decide embedding is already properly learnt, and “collapse” all inner loops into a single one sporting the averaged parameter, tuhs terminating with the training process of the loop.

After enough training time, the whole multi-level structure will naturally collapse into a single inner loop of experts, each one using a different vector of internal parameters, but all using the same “distilled” parameters for the three modules (embedding, prediction and reward), defined as the averaged parameters over the whole initial Badger structure. We will end up with a classical Badger structure!

Steeping deeper into the simplicity limit, we could also collapse this expert level by using a single expert with the averaged parameter vector, ending up with a standard Fractal AI setup.

So, in inference time, we could use any semi-trained (not fully collapsed) structure by averaging the actions from all its experts, or wait until a total collapse and use a single inner loop of experts, or just a single averaged expert.
Example of a Generalised Badger
In order to make the whole working structure more clear, let say we want to use a Generalised Badger structure to learn playing a game like “Space Engineering”.

As the inputs and their structure are known, we will handcraft the modules architectures: let say the embedding will consist on some convolution layers applied on the frames of the game and a VAE generating an embedding, etc. so we won’t need a level 4 loop to decide on the architecture.

Outer loop: The main controller level.
Inner level 3: Optimises the parameters of the Embedding module.
Inner level 2: Optimises the parameters of the Prediction module.
Inner level 1: Optimises the parameters of the Reward module.
Expert level: Experts plan on the next action using Fractal AI.

Also, we will consider that our agent receives a continuous flow of observations that he will sample at regular time intervals, thus forming a discrete time series of observations. Let say we will sample on screenshot every 0.5 seconds, independently of the actual frame rate the game is capable of showing.

The output of any of the experts will be an action to be taken in the game, chosen from a continuous or discrete decision space. For the continuous case, we can work with mouse movement plus key pressures, so we predict forces applied to both mouse x and y position puls keys. For the discrete case, we could consider the list of available discrete actions (move 1 meter south, or press this specific key). This won’t change the algorithm at all as Fractal AI can deal with both cases transparently.

Experts step will consist on playing a randomly initiated game for some maximum of 4 hours.
Rewards
The tricky part is defining a proper reward function: we will be using a mixture of two rewards: a handcrafted goal of surviving for longer, plus a learned reward function.

The first reward R1 will be the portion of the assigned play time (4 hours) the expert have survived, so a 2 hours game will receive a reward R1 of 0.5). The second part of the reward R2, will be freely assigned by the reward module for the final state reached, so, when the expert don’t survive for the full 4 hours, we will consider the expert trajectory to end on a death state, thus assigning it a R2 reward of 0. Thus, our final reward function would be like this:

Reward = R1 + if(R1=1.0; R2; 0.0)

This will make the training process to first focus on building a R2 reward function generating a safe behaviour on the experts (like rewarding running away from dangerous creatures) and, once they manage to keep them alive for the full 4 hours, it will start to focus on behaving in a way that maximises this secondary reward.

Finally, and just for the sake of simplicity, we will also consider that each loop contains the same number N of inner loops.

We will illustrate 3 main processes by using some comments and good old pseudo-code: the initialization, a learning step, and a decision or inference step. All code should be highly recursive due to the fact that the structure is basically the same at any level, but in the showed pseudo-code we will not go so far as to make it difficult to read.
Initialization
Basically all the initialization process is a recursive process of creating all the loops in the Badger structure down to the lowest level of experts.
 
// Controller outer loop:
self.count=50
// Choose an architecture for each module
embedding_str = my_VAE
prediction_str = my_LSTM
reward_str = my_FULLY_CONNECTED
structures = [embedding_str, prediction_str, reward_str]
// Initialize and populate the inner loops recursively
loops=[]
for n in range(count):
	loops.add(new_loop(level=3, count=self.count, fixed_NN=None)

// Recursively create an inner loop down to the expert level: 
new_loop(level: int, count: int, fixed_params: list of NN): loop
	NN = new_NN(structure=structures[level], params=random)
if self.level > 1:
	// Add N inner loops
	loops=[]
for n in range(count):
			loops.add(new_loop(level=self.level-1, count=count, fixed_NN=self.fixed_NN + NN)
	else:
		// Add population of experts
experts=[]
for n in range(count):
			experts.add(fragile.swarm(n_walkers=self.count, modules=fixed_NN)

Training step
Pseudo-code can be hard to follow in this step, so we will just comment on the processes happening in the correct order they will occur, and which information will each loop pass to the next.
Expert
During a training step, each expert will play a game for, at most, 4 hours, building a trajectory, and assign it a reward as previously commented.
Experts level
Each expert level will use its experts reward and parameter vector distances to perform a Fractal AI step, thus cloning some params from one expert to another.

Then, the expert level selects the better trajectory and final reward from its experts population, and expose it to the level 1 loop it belongs, along with the dataset of all trajectories as a training batch.

Now we average the vector of parameters from each of the expert and keep a list of the last M resulting vectors.

Finally, if we detect that the averaged vector of parameters haven’t substantially change for the last M steps, we could decide to collapse this level by reducing the number of experts down to one, sporting as parameter vector this average.
Level 1 loop
A level 1 loop will contain a population of experts level, each one using a different parameter matrix for its reward moule ANN. Once they have made an step, we also have access to a reward for each experts level (from its best trajectory) plus a collection of N trajectories from each expert level.

First, we will perform a Fractal AI step using those reward and the distances from the reward module used by each of the experts levels, cloning the lowest rewarding ones into the most rewarding ones while keeping high diversity on the population of ANN parameter matrices.

Then, we will use the training dataset from the best experts level (the one producing the best among the best trajectories of their experts) and use it as a training batch over the reward module ANN assigned to this level 1 loop.

Now we average the parameter matrices from each of the experts level and keep a list of the last M resulting matrices.

Finally, if we detect that the averaged parameter matrices haven’t substantially change for the last M steps, we could decide to collapse this loop by reducing the number of experts levels down to one, sporting as its embedding module ANN parameters this average.
Level 2 and 3 loops
The process here is almost identical to level 1 loops except that:

“Experts level” is changed into “lower level loop”.
The training is done over predicted state vs actual next state.
The training dataset is the union of all training dataset from all its inner loops.

Due to the fact that those levels train on pairs of predicted vs actual next state and are not affected by the agent ability to survive,  they will rapidly tend to collapsing, making all the remaining experts to share the same prediction modules thus building a consensus world model.

Once those levels collapse, the level 1 loops will start to converge, first to rewards that force experts to be cautious to maximize their survival chances, and then, toward rewards to maximize the secondary goals.
Training phase termination
If all the loops finally collapse and we reach the situation in which we have one single experts level, or a single expert if the experts level is allowed to collapse, the training phase can be terminated.

A sweet point could be to continue the training process until some minimum number of experts is reached due to eventual collapses in the structure.

Even once this collapse is complete, we still have a set of modules with their corresponding ANN that, if decided, can be trained over and over using new datasets of rollouts built during inference.
Inference step
Inference is actually pretty easy and can be done over the full initial structure, a partially collapsed one, or a totally collapsed one. In all the cases, when the Badger is asked to choose the next action, it will use all its experts, many or one, to decide on the next action using their particular modules, and the averaged action over all the experts will be output as the agent final decision.
Enhancements
The algorithm sketched is pretty complete, general and self-contained, but a number of enhancements could be added.
Architecture optimisation
As already mentioned, additional layers can be added with loops optimising over the space of possible architectures for each of the three modules.

The loops would behave exactly like level 2 and 3, except that:

Would optimise a vector of parameters defining the architecture.
The reward would be divided into the complexity of the architecture (number of layer and/or number of units, etc).

 Those levels, when present, would be the first candidates to an early collapse, as they have a low number of parameters and some specific combination makes more sense than the others.
Initialization optimisation
Each time an expert is asked to take a step, a random initial state is chosen for a new game. Ideally, those initial states will tend to cover the entire space of possible initial states for a game but, if the space is big enough, enforcing this can be really useful.

To achieve that, the system can keep track of all the visited initial states in a fixed length list, so each time a new state is visited by an expert, we would add it to the and pop out the last element, so initial states visited long before the modules were properly trained, are eventually forgotten and the states visited again after some time.

The states in the list at each moment define a density function over the space of initial states, and a reward can be assigned to any initial state candidate by using the inverse of this density, so less visited states are more rewarding to use.

Then, in order to have a set of many good initial states where the experts can sample when they need a random initial state, we would build a “pool” of many good initial states: we populate the pool with some thousands of initial states, assign each a reward, and use distance between theirs embedding to form a swarm of walkers: before a new step, we perturb this initial states, and perform a cloning phase on them, The resulting population is the new pool where the experts sample initial conditions for the games.

Matrices parameters optimisation
In the same spirit, the set of all parameter matrices for a given module present in the whole Badger at a given time can also be optimised for greater diversity.

Here we don’t need to score them with a custom reward based on visit density, in fact we want them to converge, so we will consider reward to be a constant value of one. Then, the only effect of running a swarm over ANN is to avoid too much clustering of similar ANN, and promote the diversity of the solutions.
