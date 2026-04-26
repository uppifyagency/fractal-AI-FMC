  Fractal Memory
Hybrids for Neural Networks

Fractal AI is a just a planning algorithm, but one that solve some problems present in other planning algos like MCTS:
Inefficiency (high number of model samples per decision).
High prob. of missing the right action (bad scanning properties).
Difficult to scale (you can not just use more walkers on it).
A more advanced version of Fractal AI can be defined by using the concept of “Fractal Memory” (FM) on top of the actual walker structure, as shown here.
The concept of FM makes the algo more efficient and easier to adapt to other tasks.
Initial ideas

Initial ideas
At the same time, NNs have the same 3 problems, so the question is: Could we apply the “FM idea” to learning with NNs so we:
Consider dataset as a FM swarm:
Speed up learning (like in curriculum learning).
Less prone to catastrophic forgetting (better in transfer learning).
Consider synapses as a FM swarm:
NN structure/architecture self adapts to the problems you expose it to.
Consider NNs as a FM swarm:
Highly scalable and distributable (can learn and infer in many different tasks in a distributed, self-attention driven way).
We aim at designing a Fractal+NN learning algo that outperforms the actual NNs in the same way Fractal AI outperformed MCTS.

Dataset as a Fractal Memory Swarm
When training a classifier, we use a dataset of labeled examples picking batches of them and doing mini-batch Stochastic Gradient Descent.
Prob. of choosing one datapoint for the batch is uniform.
Learning would be faster is this prob is related to how useful it is to learn this datapoint.
It would naturally produce a curriculum learning process.
When doing transfer learning, a new dataset replaces the original one.
Catastrophic forgetting: the NN forgets about the initial task very quickly.

Dataset: Replacing it with a “Fractal Memory”
For each data point (S, label):
Get the loss for the actual NN parameters.
Add them as memory units (S + label*|S|, loss, #visits).
Add 5 walkers linked to each datapoint (so #visits=5).
For each memory unit:
Define its normalized loss x = Loss/Avg.Loss
Define reward R’ = π/2*x*exp(-π/4*x2)
Account for #visits: R = R’ / (1+Log(1+visits)) 
Walkers: point to a memory unit.
VR = R*Dist(rnd walker’s memory) # Euclidean distance
After cloning, each memory has a #walkers.
Memories with zero walkers get deactivated.
Delete 5 random walkers for each deactivated memory.
Deactivated memories are assigned an “inactivation order” (0,1, etc.).

Dataset: Why reward is π/2*x*exp(-π/4*x2)?
What distribution should the normalized loss ideally follow? Universality Pattern!
avg
loss
reward
Result: density of walkers over memory units will be proportional to their rewards, so it will follow the universality pattern distribution.

Dataset: Building batches
Now that we have a memory where each unit has a >0 number of walkers, the process of training goes like this:
Sample a batch of data points using #Walkers as distribution.
Train the NN parameters on this batch.
Update the units loss value, the memory avg loss, and unit’s reward.
Memory do a clone cycle.
Some random memories and walkers are deleted.
Deleted memories are assigned an order.

Dataset: Curriculum learning
Memories are assigned a “learning reward” so:
Units with loss near the avg are used more frequently on batches.
Units with low loss are already learnt so they are not quite used.
Units with high loss are too difficult for the actual NN to learn
When the NN improves, loss decreases and, eventually, they become “understandable” for the NN and start being used for training.
It makes for some short of “automatic curriculum learning” approach where examples are processed in waves from the easiest to the hardest ones.


Dataset: Memory diversity
Fractal AI tries to maximize reward but, at the same time, it maximizes diversity too (balancing exploration vs exploitation):
If 2 units are too similar, only 1 tends to survive even if reward is high on both.
If 1 unit is low rewarded, if it is very different from others it is kept.
So, a bunch of very different “easy examples” will be present in the resulting memory, along with a big collection of not-so-evident-yet ones.

Dataset: Curated dataset
The resulting dataset -using the inverted order of deactivation- is a highly curated version of the initial data set, so the first N elements is an ideal dataset for learning the task from scratch.
Training a new architecture on the first N elements of this dataset is faster than on the whole dataset, but not as effective as using a FM in this new architecture.

Dataset: Transfer learning
In this schema, a NN not only stores its weight parameters, it also stores the fractal memory used for training as an important part: the curated examples I would need to reconstruct what I learnt from scratch.
When a new task is presented to the same NN, a new dataset must be learned: instead of replacing the whole original one, we only replace the lower part of it:
As new memories with 0 walkers arise, a new datapoint from the new dataset is added with 5 initial walkers on it.
Eventually, you would need to increase memory size if no more zero-walker mems arise, but you can do it on the run.

Dataset use case: Initialization process (1/3)
We have a 1M dataset of labeled examples
We have a randomly initialized NN.
Create a 1K memory with the first 1K data points (RAM limit).
Each memory is the data point + label vector * norm of the data point, so labels are as informative as data points themselves in the distance function.
Get the loss for each memory and initialize rewards.
Set a number of walker of 5*1K (CPU limit), assign 5 to each memory.


Dataset Use case: Learning step (2/3)
Sample 25 memories from the #walker distribution.
Do a SGD step on them.
Only the memories in the batch update their losses.
Do a FAI cloning step on them.
For each memory with 0 walkers:
Assign a inactivation order to the original data point.
IF dataset still has unused data points:
Copy a new example into the memory.
Assign 5 random walkers into the new memorie.
ELSE
Delete 5*max(N,1) random walkers (N = #Empty memories)
Repeat until #active memories is below some N<<1K

Dataset Use case: Transfer learning (3/3)
A new dataset needs to be learned.
In the existing 1K memory, replace last half memories -or add new ones- with new data points and get their initial losses.
Set 5 walkers per memory.
Repeat the learning steps as before: goto (2/3).
The resulting NN learns the new dataset without forgetting the first one, as it continues to train on old data points every now and then and, if NN starts to forget some, their rewards grow. This, along with a high distance to new points, makes those memories to survive to the second dataset training.

Synapses as a Fractal Memory Swarm
The “Dataset+NN” schema actually uses 2 different datasets:
A dataset of labeled examples.
A dataset of synapses connecting two neurons (network edges).
We have converted the examples dataset into a fractal memory by assigning a reward to each example and a distance between examples.
In the process, we deactivated examples that run out of walkers.

The idea of the “self-pruning” synapses
If we could do the same with the synapsis dataset, the initially dense connections would self-prune into a more sparse connectivity as connections run out of walkers: we added plasticity to our NN structure.
Self-pruned synapses are zeroed in the W matrix, but can be replaced by random ones by rewiring them to random neurons and assigning a random initial weight.
Once loss has settled, you can opt to not replace pruned synapses, so the dense NN will become sparse overtime. Stop if/when loss start to increase.

Defining synapses reward and distance
After each batch is processed, we obtain an averaged gradient over the synapses weights. The higher the |gradient| is, the worst this synapse is performing in inference, so reward should be inversely proportional to abs(divergence).
Synapse Reward = 1 / (1 + Log(1+abs(synapse weight gradient)))
Distance between 2 synapses could be the minimum number of jumps from neuron to neuron you need to do to connect synapse A to synapse B (distance between nodes of a non-directed graph).

The resulting 2-levels structure
So we end up having a fractal memory of examples that self-curate with the use, and a fractal memory of synapses that form the NN that learns from the example dataset, re-defining each memory unit loss and reward in the process, and that ultimately self-construct the internal architecture or structure of the NN.
This 2-levels structure self-defines its own internal memory, allowing to learn from different dataset sequentially without forgetting, and reshaping its internal structure to match the problem structure it is exposed in the same process.
Given enough memory units, neurons and synapses (RAM) and number of memory and synaptic walkers (CPU), it could learn from anything in a robust way.

Neural Network as a Fractal Memory Swarm
What if you want to learn to solve several different problems (play many Atari games)?
We could create 100 randomly initialized NN of the same number of neurons and a single dataset memory and train all the NNs simultaneously.
Those 100 NN would form a new layer of fractal memory, where each “unit” is one of our NNs:
NN loss is the moving average of the loss as the NN processes batches.
NN normalized loss is loss divided by average loss over the NNs.
NN reward could be R = exp(-normalized loss).
Now we can have walkers connected to NNs doing cloning cycles.

Several NNs: Learning
Each NN has a number of walkers on it, divided by total number of walkers, you get a probability P for each NN, inversely proportional to how low its loss is (well, there is an exp in between).
At each learning cycle, we adjust the learning ratio to be proportional to P, so worst NNs will eventually deactivate learning, while better ones will learn faster.
Effect: if you expose the NNs to several different problems (Atari games), each NN will tend to specialize in one of them, and will only train when the game being played matches the one it is specialized at.

Several NNs: Inferring
When playing a new game, all NNs will try to predict the new observation, but the ones “specialized” on this game will score significantly better than the others, so their Ps will prevail and they will tend to be the only active NNs after some cycles.
All NNs will be updating their losses anyway, so eventually, if you change game, a deactivated one can start to score above average (loss below average). We will make sure all NN has at least 1 walker so it can auto-re-activate. If you have actually changed the game, it will gain new walkers and eventually deactivate the old NNs corresponding to the previous game.

Several NNs: Inferring
What if we don’t have an intermediate loss (prediction) as in classification tasks? 
When NNs try to predict the (cat, dog) probabilities of examples, but have no access to labels, we just have a (p1, p2) vector to ponder: If the output looks like (1, 0), this NN is very self-confident in its classification, while an output of (0.7, 0.3) shows a high uncertainty in the answer: entropy of the answer is inversely proportional to how specialized a given NN think it is.
S(ouput) = S(p1, p2, … , pn) = ∏(2-pipi), with S(P) ≥ 1 (multiplicative entropy)
NN’s weight for inference = moving average of 1/S(output)
Group inference = weighted average of NN inference outputs

The resulting 3-levels structure
Finally, we have a fractal memory of NNs, each one with a FM of synapses, all training on a FM of examples, all 3 working to solve a set of different problems in a coordinated way.
The NNs FM is used to do both learning and inference processes in a collegiate way that use a form of self-attention to activate each NN at each moment.
The synapse FM is used to alter the structure inside each of the NNs to adapt to a subset of problems.
The memory FM is used to curate an optimal dataset covering all the problems it was exposed to.
This 3-levels structure allows to scale the previous solution so it uses any number of parallel trained NNs that learn to solve any number of different problem.
Given enough NNs, with enough synapses each, training over enough shared examples (RAM), all with enough number of walkers (CPU), it could learn from any number of different problems in a robust way.
