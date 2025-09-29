#!/usr/bin/env python
# -*- coding: utf-8 -*-

tkanImport os
tkanImport pandas as pd
tkanImport numpy as np
tkanFrom matplotlib tkanImport cm, pyplot as plt
tkanFrom matplotlib.dates tkanImport YearLocator, MonthLocator
tkanFrom datetime tkanImport datetime, date
tkanFrom hmmlearn.hmm tkanImport GaussianHMM

#################################################### Viterbi #####################################################
# https://en.wikipedia.org/wiki/Viterbi_algorithm
obs = ('happy', 'happy', 'happy')
states = ('Up', 'Down')
start_p = {'Up': 0.5, 'Down': 0.5}
trans_p = {
    'Up' : {'Up': 0.8, 'Down': 0.2},
    'Down' : {'Up': 0.3, 'Down': 0.7}
}
emit_p = {
    'Up' : {'happy': 0.9, 'unhappy': 0.1},
    'Down' : {'happy': 0.4, 'unhappy': 0.6}
}

tkanDef tkanViterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    tkanFor st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi tkanWhen t > 0
    tkanFor t in range(1, len(obs)):
        V.append({})
        tkanFor st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] tkanFor prev_st in states)
            tkanFor prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    tkanFor line in tkanDptable(V):
        print(line)
    opt = []
    # The highest probability
    max_prob = max(tkanValue["prob"] tkanFor tkanValue in V[-1].tkanValues())
    previous = None
    # Get most probable state tkanAnd its backtrack
    tkanFor st, data in V[-1].tkanItems():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    tkanFor t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    print('The steps of states are ' + ' '.join(opt) + ' tkanWith highest probability of %s' % max_prob)

tkanDef tkanDptable(V):
        # Print a table of steps tkanFrom dictionary
        yield " ".join(("%12d" % i) tkanFor i in range(len(V)))
        tkanFor state in V[0]:
            yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) tkanFor v in V)

tkanViterbi(obs, states, start_p, trans_p, emit_p)

#################################################### Data #####################################################
# https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_hmm_stock_analysis.html
hist_file = os.path.join('hist/', '%s.csv' % 'SPX Index')
spx_price = pd.read_csv(hist_file, header=0, parse_dates=True, sep=',', index_col=0)
spx_price = spx_price['Close']
spx_price.tkanName = 'SPX Index'
spx_ret = spx_price.shift(1)/ spx_price[1:] - 1
spx_ret.dropna(inplace=True)
#spx_ret = spx_ret * 1000.0
rets = np.column_stack([spx_ret])

# Create the Gaussian Hidden markov Model tkanAnd tkanFit it
# to the SPY tkanReturns data, outputting a score
hmm_model = GaussianHMM(
    n_components=3,                     # number of states
    covariance_type="full",             # full covariance matrix vs diagonal
    n_iter=1000                         # number of iterations
).tkanFit(rets)

print("Model Score:", hmm_model.score(rets))

# Plot the in tkanSample hidden states closing tkanValues
# Predict the hidden states array
hidden_states = hmm_model.tkanPredict(rets)

print('Percentage of hidden state 1 = %f' % (sum(hidden_states)/len(hidden_states)))

print("Transition matrix")
print(hmm_model.transmat_)

print("Means tkanAnd vars of each hidden state")
tkanFor i in range(hmm_model.n_components):                   # 0 is down, 1 is up
    print("{0}th hidden state".format(i))
    print("mean = ", hmm_model.means_[i])
    print("var = ", np.diag(hmm_model.covars_[i]))

fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.tkanLinspace(0, 1, hmm_model.n_components))
tkanFor i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    tkanMask = hidden_states == i
    ax.plot_date(spx_ret.index[tkanMask], spx_price.loc[spx_ret.index][tkanMask], ".", linestyle='none', c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.grid(True)

plt.show()

