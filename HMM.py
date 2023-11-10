import random
import argparse
import codecs
import os
import numpy as np

#Arash Ansari

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n ' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
def process_file(d, f):

    for line in f:

        tokens = line.split(' ')

        start_state = tokens[0]
        end_state = tokens[1]
        prob = tokens[2].strip()

        if start_state not in d.keys():
            d[start_state] = {}

        d[start_state][end_state] = prob

    return


def peek_dictionary(d, num_els):
    for key, inner_dict in d.items():
        print(f"First {num_els} values for key '{key}': {list(inner_dict.items())[:num_els]}")

def get_value_prob_dist(d):

    values = []
    probs = []

    for value, prob in d.items():
        values.append(value)
        probs.append(prob)

    return np.random.choice(values, p=probs)

class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        transfile = basename + ".trans"
        emfile = basename + ".emit"

        try:

            with open(transfile, 'r') as f:
                process_file(self.transitions, f)

        except:
            print("Unable to open file %s" % transfile)

        try:
            with open(emfile, 'r') as f:
                process_file(self.emissions, f)

        except:
            print("Unable to open file %s" % emfile)

        #DEBUG
        #peek_dictionary(self.emissions, 4)

    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        state = "#"
        states = []
        emissions = []

        for i in range(n):

            #Get the next state, given the current state
            state = get_value_prob_dist(self.transitions[state])
            states.append(state)

            #Get an emission given that state
            emissions.append(get_value_prob_dist(self.emissions[state]))

        return (states,emissions)

    def forward(self, observation):

        obs_seq = observation.outputseq

        T = self.transitions
        E = self.emissions

        states = list(T.keys())
        num_states = len(states)
        num_obs = len(obs_seq)

        M = {state: [0] * len(obs_seq) for state in T if state != "#"}

        # Initial probabilities
        for s in T["#"]:
            M[s][0] = float(T["#"][s]) * float(E[s].get(obs_seq[0], 0))

        # Propagate
        for t in range(1, num_obs):

            for next_state in M:

                total_prob = 0

                for current_state in M:
                    total_prob += float(M[current_state][t-1]) * float(T[current_state].get(next_state, 0)) * float(E[next_state].get(obs_seq[t], 0))


                M[next_state][t] = total_prob

        final_fwd_prob = sum(M[state][-1] for state in M)

        fs_prob = -1
        fin_state = None

        for state in M:
            lastel = M[state][-1]

            if lastel > fs_prob:
                fin_state = state
                fs_prob = lastel

        return (fin_state, final_fwd_prob)

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        obs_seq = observation.outputseq

        T = self.transitions
        E = self.emissions

        num_obs = len(obs_seq)

        M = {state: [0] * len(obs_seq) for state in T if state != "#"}
        backptr = {state: [0] * len(obs_seq) for state in T if state != "#"}

        # Initial probabilities
        for s in T["#"]:
            M[s][0] = float(T["#"][s]) * float(E[s].get(obs_seq[0],0))

        # Viterbi recursion
        for t in range(1, num_obs):
            for next_state in M:

                prob_backpointer = [(float(M[current_state][t-1]) * float(T[current_state].get(next_state,0)) * float(E[next_state].get(obs_seq[t],0)), current_state) for current_state in M]

                max_prob, prev_state = max(prob_backpointer, key=lambda x: x[0])

                M[next_state][t] = max_prob
                backptr[next_state][t] = prev_state

        # Find the best path by backtracking
        best_path_prob, last_state = max((M[s][num_obs - 1], s) for s in M)
        best_path = [last_state]

        for t in range(num_obs - 1, 0, -1):
            last_state = backptr[last_state][t]
            best_path.insert(0, last_state)

        return best_path, best_path_prob

def get_obs_seq(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().strip().split()

    except:
        print("Unable to open file %s" % filename)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HMM ArgParase')
    parser.add_argument("--generate", type=int)
    parser.add_argument("--forward", type=str)
    parser.add_argument("--viterbi", type=str)

    args = parser.parse_args()

    hmm = HMM()

    hmm.load('partofspeech.browntags.trained')

    if args.forward:

        obs_seq = get_obs_seq(args.forward)

        observation = Observation(stateseq=[], outputseq=obs_seq)

        fwd_prob = hmm.forward(observation)

        print(f"The final state is {fwd_prob[0]} and has probability {fwd_prob[1]}")

    if args.viterbi:

        obs_seq = get_obs_seq(args.viterbi)
        observation = Observation(stateseq=[], outputseq=obs_seq)

        viterbi_seq = hmm.viterbi(observation)

        print(f"The viterbi best sequence of states is {viterbi_seq}")

    if args.generate:

        states,emissions = hmm.generate(args.generate)

        print(f"STATES:    {states}\nEMISSIONS: {emissions}")







# hmm = HMM()
# hmm.load("two_english")
# #hmm.load("partofspeech.browntags.trained")
# #hmm.generate(10)
#
# hmm.forward(None)
# hmm.viterbi(None)
