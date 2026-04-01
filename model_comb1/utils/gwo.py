import numpy as np
"""
grey wolf optimisation works by sending aplha, bedta and delta solutions in random directions. 
when aplha finds the better sollution, it leads the pack to that solution. then the beta and delta
explore around the aplha solution to find better solutions. It continues until max iterations is reached.

Number of wolves should be at least 3 (alpha, beta, delta). In this case 8 wolves are used.
which means 5 other wolves explore around the 3 best solutions. Also, the more the number of wolves,
the better the exploration of the search space. 3 best solutions are always used to guide the rest of the wolves.
# best solutions correspond to the lowest objective function values.

objective function should be minimization problem. It should take a 1D numpy array as input and return a scalar value.
It can be used for hyperparameter tuning of machine learning models or other optimization tasks.
It is based on the paper "Grey Wolf Optimizer" by Mirjalili et al., 2014.
It is basically an implementation of the algorithm described in the paper.
It works with alpha ( best solution), beta (2nd best solution) and delta (3rd best solution) by updating the positions of the wolves
based on their distances to these three best solutions.
"""

"""
In addition to the social hierarchy of wolves, group hunting is another interesting social behavior of grey
wolves. According to Muro et al. [47] the main phases of grey wolf hunting are as follows:
 Tracking, chasing, and approaching the prey
 Pursuing, encircling, and harassing the prey until it stops moving
 Attack towards the prey

In order to mathematically model the social hierarchy of wolves when designing GWO, we consider the
fittest solution as the alpha ( ). Consequently, the second and third best solutions are named beta ( ) and delta
( ) respectively. The rest of the candidate solutions are assumed to be omega ( ). In the GWO algorithm the
hunting (optimization) is guided by , , and . The wolves follow these three wolves. 

During the hunting process, wolves tend to encircle the victim at first, which can be formulated by
D=|C.Xp(t)-X(t)|
X(t+1)=Xp(t)-A.D
A=2.ar1-a
C=2.r2
t is the current iteration number,
 Xp indicates the position vector of the prey, 
 and X denotes the current position of an individual wolf.
A and C are coefficient vectors, by which wolves can reach different situations around the prey. 
D is distance from prey location.
r1 and r2 are random vectors inside [0,1].
The elements of a are linearly decreased from 2 to 0
over the course of iterations and used to coordinate the exploration and exploitation ability.

Although the hunting process is mainly guided by α , β and δ , 
the actual position of the prey is unknown in an abstract search space.
To imitate the hunting process, it is assumed that α , β and δ can guesstimate the possible location of the prey.
Therefore, the three best individuals on the decision level are saved and guide the others to update their positions in each iteration,
which can be formulated by
Dα=|C1.Xα(t)-X(t)|
Dβ=|C2.Xβ(t)-X(t)|
Dδ=|C3.Xδ(t)-X(t)|
X1=Xα(t)-A1.Dα
X2=Xβ(t)-A2.Dβ
X3=Xδ(t)-A3.Dδ
X(t+1)=(X1+X2+X3)/3
Where Xα , Xβ and Xδ are the position vectors of α , β and δ
"""
class GreyWolfOptimizer:
    """
    Simple Grey Wolf Optimizer implementation for continuous search spaces.
    Mimics the hunting behavior of grey wolves: alpha (best), beta (2nd best), delta (3rd best)
    lead the pack toward the optimal solution.
    
    Usage:
      gwo = GreyWolfOptimizer(num_wolves=8, max_iter=20, verbose=True)
      best_pos, best_score = gwo.optimize(objective_fn, lb, ub)
    
    objective_fn should accept a 1D numpy array and return a scalar (lower is better).
    """
    def __init__(self, num_wolves=8, max_iter=20, verbose=False):
        """
        Initialize the Grey Wolf Optimizer.
        
        Parameters:
          num_wolves: number of candidate solutions (wolves) in the population
          max_iter: maximum number of iterations (generations)
          verbose: if True, print progress after each iteration
        """
        # ensure at least 3 wolves (need alpha, beta, delta)
        self.num_wolves = max(3, int(num_wolves))
        # number of iterations to run
        self.max_iter = int(max_iter)
        # flag to control console output
        self.verbose = bool(verbose)

    def optimize(self, obj_fn, lb, ub):
        """
        Run the Grey Wolf Optimizer.
        
        Parameters:
          obj_fn: objective function to minimize; takes 1D numpy array, returns scalar
          lb: lower bounds (list or array of length = search space dimension)
          ub: upper bounds (list or array of length = search space dimension)
        
        Returns:
          best_pos: best solution found (1D numpy array)
          best_score: objective value of best solution
        """
        # convert bounds to numpy arrays of type float for math operations
        lb = np.array(lb, dtype=float)
        ub = np.array(ub, dtype=float)
        # dimension of the search space (number of hyperparameters to tune)
        dim = lb.size

        # randomly initialize wolf positions uniformly within bounds
        # shape: (num_wolves, dimension)
        X = np.random.uniform(lb, ub, (self.num_wolves, dim))
        
        # evaluate the objective function for each initial wolf position
        # lower value = better solution
        fitness = np.array([obj_fn(x) for x in X], dtype=float)

        # sort wolves by fitness: idx[0] is best, idx[1] is 2nd best, etc.
        idx = np.argsort(fitness)
        
        # alpha_pos: best solution found so far (leader of the pack)
        alpha_pos = X[idx[0]].copy()
        # beta_pos: 2nd best solution (second-in-command)
        beta_pos = X[idx[1]].copy()
        # delta_pos: 3rd best solution (third-in-command)
        delta_pos = X[idx[2]].copy()
        
        # store their objective values
        alpha_score = float(fitness[idx[0]])  # best (lowest) score
        beta_score = float(fitness[idx[1]])   # 2nd best score
        delta_score = float(fitness[idx[2]])  # 3rd best score

        # optionally print initial state
        if self.verbose:
            preview_len = min(5, dim)
            a_preview = ','.join(f"{v:.4f}" for v in alpha_pos[:preview_len])
            print(f"[GWO] Init best={alpha_score:.6f} beta={beta_score:.6f} delta={delta_score:.6f} | alpha_pos[:{preview_len}]={a_preview}")

        # main optimization loop over iterations
        for t in range(self.max_iter):
            # linearly decrease 'a' from 2.0 to 0.0 over iterations
            # controls exploration (high a) vs. exploitation (low a)
            a = 2 * (1 - t / float(self.max_iter))

            # update each wolf's position based on alpha, beta, delta
            for i in range(self.num_wolves):
                # current wolf position
                x = X[i].copy()
                
                # helper function to compute position update toward a leader
                def _calc(leader):
                    """
                    Compute new position by moving toward a leader (alpha/beta/delta).
                    Uses random coefficients to balance exploration and exploitation.
                    """
                    # random vectors in [0, 1] for stochasticity
                    r1 = np.random.rand(dim)
                    r2 = np.random.rand(dim)
                    
                    # coefficient A controls how much to move; decreases with iterations
                    A = 2 * a * r1 - a
                    # coefficient C adds randomness; always in [0, 2]
                    C = 2 * r2
                    
                    # distance between current wolf and leader
                    D = np.abs(C * leader - x)
                    
                    # new position: move from leader back by distance D scaled by A
                    return leader - A * D

                # compute candidate positions influenced by the three best wolves
                X1 = _calc(alpha_pos)  # move toward alpha (best)
                X2 = _calc(beta_pos)   # move toward beta (2nd best)
                X3 = _calc(delta_pos)  # move toward delta (3rd best)
                
                # new position is the average of moves toward all three leaders
                new_x = (X1 + X2 + X3) / 3.0
                
                # clip new position to stay within bounds [lb, ub]
                X[i] = np.minimum(np.maximum(new_x, lb), ub)

            # evaluate fitness of all wolves at their new positions
            fitness = np.array([obj_fn(x) for x in X], dtype=float)
            
            # sort wolves again to identify new alpha, beta, delta
            idx = np.argsort(fitness)

            # update leaders and their scores using current population
            alpha_pos = X[idx[0]].copy()
            alpha_score = float(fitness[idx[0]])
            beta_pos = X[idx[1]].copy()
            beta_score = float(fitness[idx[1]])
            delta_pos = X[idx[2]].copy()
            delta_score = float(fitness[idx[2]])

            # print iteration summary when verbose
            if self.verbose:
                # show first few dimensions of positions for readability
                preview_len = min(5, dim)
                a_preview = ','.join(f"{v:.4f}" for v in alpha_pos[:preview_len])
                b_preview = ','.join(f"{v:.4f}" for v in beta_pos[:preview_len])
                d_preview = ','.join(f"{v:.4f}" for v in delta_pos[:preview_len])
                pop_mean = float(np.mean(fitness))
                pop_std = float(np.std(fitness))
                print(f"[GWO] Iter {t+1}/{self.max_iter} | best={alpha_score:.6f} beta={beta_score:.6f} delta={delta_score:.6f} | mean={pop_mean:.6f} std={pop_std:.6f}")
                print(f"       alpha_pos[:{preview_len}]={a_preview}")
                print(f"       beta_pos[:{preview_len}]={b_preview}")
                print(f"       delta_pos[:{preview_len}]={d_preview}")

        # return the best solution and its objective value
        return alpha_pos, alpha_score