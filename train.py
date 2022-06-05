import numpy as np
import numpy.random as npr
import torch
import itertools
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import cycle
from model import Network, Task, Algorithm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.stats import bernoulli
from scipy.ndimage.filters import uniform_filter1d


def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


def td(r, v, v_old, h_old, s_old, ws, wc, wa, gamma, var, lam, lr_ws, lr_wc, lr_wa):
    delta = r + gamma * v - v_old
    ws += lr_ws * delta * np.tile(wc * h_old, [2, 1]).T * 2 * (s_old - ws) / var
    wc += lr_wc * delta * h_old
    wa += (lr_wa / lam) * delta * h_old
    return delta, ws, wc, wa


def gain(delta, gains):
    return delta * gains[0] if delta > 0 else delta * gains[1]


def train_pfc(arch='PFC', N=128, S=2, R=2, task='beads-belief', maxsamples=10, context=(.65, .85),
              algo='Adam', Nepochs=10000, lr=1e-3, learningsites=('J', 'wr'), seed=1):

    # instantiate model
    net = Network(arch, N, S, R, fb_type='aligned', seed=seed)
    task = Task(task, maxsamples=maxsamples, context=context)
    algo = Algorithm(algo, Nepochs, lr)

    # convert to tensor
    sites = ('ws', 'J', 'wr')
    for site in sites:
        if site in learningsites:
            setattr(net, site, torch.tensor(getattr(net, site), requires_grad=True))
        else:
            setattr(net, site, torch.tensor(getattr(net, site), requires_grad=False))

    # optimizer
    opt = torch.optim.Adam([getattr(net, site) for site in sites], lr=algo.lr)
    lr_ = algo.lr

    # frequently used vars
    dt, NT, NT_sample, N, S, R = net.dt, task.NT, task.NT_sample, net.N, net.S, net.R

    # track variables during learning
    learning = {'epoch': [], 'lr': [], 'mses': [], 'contexts': [], 'jars': [], 'beads': []}

    # random initialization of hidden state
    z0 = npr.randn(N, 1)    # hidden state (potential)
    net.z0 = z0  # save

    contexts, jars = [], []
    for ei in range(algo.Nepochs):

        # select context and jar
        if 'beads' in task.name:
            # pick a context (proportion of majority beads)
            context = npr.choice(task.context)
            # pick a jar (majority red or majority blue)
            jar = npr.random_integers(0, 1)
            # probability of drawing blue
            q = context if jar else 1 - context
            contexts.append(context), jars.append(jar)

        # initialize activity
        z0 = net.z0     # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)

        # save tensors for plotting
        sa = torch.zeros(NT, S)  # save the inputs for each time bin for plotting
        ha = torch.zeros(NT, N)  # save the hidden states for each time bin for plotting
        ua = torch.zeros(NT, R)  # save the beliefs

        # errors
        err = torch.zeros(NT, R)     # error in beliefs

        beads = []
        for ti in range(NT):

            # draw a sample
            if ti % NT_sample == 0:
                s = 2 * bernoulli.rvs(q) - 1
                beads.append(s)
                b = 1 / (1 + (q / (1-q)) ** (len(beads) - 2 * np.sum(np.array(beads) == (2*jar - 1))))
                task.ustar[ti:ti+NT_sample] = [b, 1-b]  # actual belief state

            # network update
            Iin = net.ws.mm(torch.as_tensor(np.array([s, context])[:, None]))   # input current
            Irec = net.J.mm(h)                      # recurrent current
            z = Iin + Irec                          # potential
            h = (1 - dt) * h + dt * (net.f(z))      # activity
            u = net.wr.mm(h)                        # output

            # save values for plotting
            sa[ti], ha[ti], ua[ti] = torch.as_tensor(np.array([s, context])), h.T, u.T

            # error
            err[ti] = torch.tensor(task.ustar[ti]) - u.flatten() #if ti % NT_sample > 80 else torch.zeros(1, R)

        # print loss
        loss = task.loss(err)
        print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(loss.item()), end='')

        # save mse list and cond list
        learning['mses'].append(loss.item())
        learning['beads'].append(beads)

        # update learning rate if needed
        if ei >= algo.Nstart_anneal:
            lr_ *= np.exp(np.log(algo.annealed_lr / algo.lr) / (algo.Nepochs - algo.Nstart_anneal))
            opt.param_groups[0]['lr'] = lr_
            learning['lr'].append(lr_)

        # do BPTT
        loss.backward()
        opt.step()
        opt.zero_grad()

    # save contexts and jars
    learning['contexts'].append(contexts)
    learning['jars'].append(jars)

    return net, task, algo, learning


def train_bg(arch='BG', N=64, S=2, Rc=1, Ra=3, task='beads-choice', maxsamples=10, context=(.65, .85),
             rewards=(20, -400, -1), algo='TD', Nepochs=10000, lr=(5e-7, 5e-4, 5e-4), seed=1):

    # instantiate model
    net = Network(arch, N, S, Rc=Rc, Ra=Ra, seed=seed)
    task = Task(task, maxsamples=maxsamples, context=context, rewards=rewards)
    algo = Algorithm(algo, Nepochs, lr)

    # frequently used vars
    N, S, Rc, Ra = net.N, net.S, net.Rc, net.Ra
    lr_ws, lr_wc, lr_wa = algo.lr

    # track variables during learning
    learning = {'epoch': [], 'lr': [], 'contexts': [], 'jars': [], 'beads': [],
                'beliefs': [], 'values': [], 'actions': [], 'samples': [], 'rewards': []}

    contexts, jars = [], []
    for ei in range(algo.Nepochs):

        # select context and jar
        if 'beads' in task.name:
            # pick a context (proportion of majority beads)
            context = npr.choice(task.context)
            # pick a jar (majority red or majority blue)
            jar = npr.random_integers(0, 1)
            # probability of drawing blue
            q = context if jar else 1 - context
            contexts.append(context), jars.append(jar)

        # save tensors for plotting
        sa = np.zeros((maxsamples, S))      # save the inputs
        ha = np.zeros((maxsamples, N))      # save the hidden states
        ca = np.zeros((maxsamples, Rc))     # save the values
        aa = np.zeros((maxsamples, Ra))     # save the actions
        ra = np.zeros((maxsamples, 1))      # save the rewards

        # errors
        delta = np.zeros((maxsamples, 1))   # reward prediction error

        beads = []
        for sample in range(maxsamples):

            # draw a sample
            d = 2 * bernoulli.rvs(q) - 1
            beads.append(d)
            b = 1 / (1 + (q / (1-q)) ** (len(beads) - 2 * np.sum(np.array(beads) == (2*jar - 1))))
            s = np.array([b, 1-b])

            # striatal activity
            h = np.exp(- np.sum((s - net.ws) ** 2, axis=1) / net.var)           # striatum
            c = np.matmul(net.wc, h)                                            # critic (ventral striatum)
            a = np.matmul(net.wa, h)                                            # actor (dorsal striatum)

            # choose an action
            p_a = np.exp(a / net.lam) / np.sum(np.exp(a / net.lam))             # softmax policy
            choice = np.nonzero(npr.multinomial(1, p_a))[0][0]

            # collect reward
            if choice == 2:
                r = task.rewards[-1]    # sample
            elif choice == jar:
                r = task.rewards[0]     # correct
            else:
                r = task.rewards[1]     # incorrect

            # save values
            sa[sample], ha[sample], ca[sample], aa[sample], ra[sample] = s.T, h.T, c.T, a.T, r

            # weight updates
            if sample > 0:
                delta[sample] = r + net.gam * c - ca[sample-1]
                net.ws += lr_ws * delta[sample] * np.tile(net.wc * ha[sample-1], [2, 1]).T * 2 * (sa[sample-1] - net.ws) / net.var
                net.wc += lr_wc * delta[sample] * ha[sample-1]
                net.wa[choice] += (lr_wa / net.lam) * delta[sample] * ha[sample-1]

            if choice != 2:
                break

        # save trial
        learning['beliefs'].append(b)
        learning['values'].append(c)
        learning['actions'].append(choice)
        learning['beads'].append(beads)
        learning['samples'].append(len(beads))
        learning['rewards'].append(np.sum(ra))
        print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Reward:' + str(np.sum(ra)), end='')

    # save contexts and jars
    learning['contexts'].append(contexts)
    learning['jars'].append(jars)

    return net, task, algo, learning


def combine_pfc_bg(pfc, bg, task, algo, Nepochs=1000, maxsamples=10, context=(.65, .85), noise=0, circular=False,
                   rewards=(20, -200, -2), gains=(1, 1), amplification=1, seed=1):

    npr.seed(seed=seed)
    # frequently used vars
    dt, NT, NT_sample, N_pfc, N_bg, S, R, Rc, Ra = pfc.dt, task.NT, task.NT_sample, pfc.N, bg.N, pfc.S, pfc.R, bg.Rc, bg.Ra
    lr_ws, lr_wc, lr_wa = algo.lr

    # track variables during performance
    performance = {'epoch': [], 'lr': [], 'contexts': [], 'jars': [], 'beads': [],
                   'beliefs': [], 'values': [], 'actions': [], 'samples': [], 'rewards': [], 'rpes': []}

    # convert to tensor
    sites = ('ws', 'J', 'wr')
    for site in sites:
        setattr(pfc, site, getattr(pfc, site).detach().numpy())
    pfc.J *= amplification

    # variables to save
    contexts, jars, beliefs = [], [], []
    stim = np.zeros((Nepochs, NT, S))         # save the inputs
    resp = np.zeros((Nepochs, NT, R))         # save the beliefs
    stria = np.zeros((Nepochs, NT, N_bg))     # save the striatal activity
    vals = np.zeros((Nepochs, NT))            # save the instantaneous values

    for ei in range(Nepochs):

        # select context and jar
        if 'beads' in task.name:
            # pick a context (proportion of majority beads)
            context = npr.choice(task.context)
            # pick a jar (majority red or majority blue)
            jar = npr.random_integers(0, 1)
            # probability of drawing blue
            q = context if jar else 1 - context
            contexts.append(context), jars.append(jar)

        # initialize activity
        z0 = pfc.z0     # hidden state (potential)
        h0 = pfc.f(z0)  # hidden state (rate)
        z, h = z0, h0

        # errors
        delta = np.zeros((maxsamples, 1))   # reward prediction error

        # save tensors for plotting
        sa = np.zeros((NT, S))              # save the inputs for each time bin for plotting
        ua = np.zeros((NT, R))              # save the beliefs
        ya = np.zeros((maxsamples, R))      # save the inputs
        ma = np.zeros((maxsamples, N_bg))   # save the striatal activity
        ca = np.zeros((maxsamples, Rc))     # save the values
        aa = np.zeros((maxsamples, Ra))     # save the actions
        ra = np.zeros((maxsamples, 1))      # save the rewards

        beads, sample = [], 0
        for ti in range(NT):

            if ti % NT_sample == 0:
                # save true and model beliefs
                if ti == 0:
                    # draw a new sample
                    s = 2 * bernoulli.rvs(q) - 1
                    beads.append(s)
                else:
                    b_model = ua[ti-int(0.8 * NT_sample):ti, 0].mean()
                    y = np.array([b_model, 1 - b_model])

                    # striatal activity
                    m = np.exp(- np.sum((y - bg.ws) ** 2, axis=1) / bg.var)  # striatum
                    c = np.matmul(bg.wc, m)  # critic (ventral striatum)
                    a = np.matmul(bg.wa, m)  # actor (dorsal striatum)

                    # choose an action
                    p_a = np.exp(a / bg.lam) / np.sum(np.exp(a / bg.lam))  # softmax policy
                    choice = np.nonzero(npr.multinomial(1, p_a))[0][0]

                    # collect reward
                    if choice == 2:
                        r = rewards[-1]  # sample
                    elif choice == jar:
                        r = rewards[0]  # correct
                    else:
                        r = rewards[1]  # incorrect

                    # save values
                    ya[sample], ma[sample], ca[sample], aa[sample], ra[sample] = y.T, m.T, c.T, a.T, r

                    # weight updates
                    if sample > 0:
                        delta[sample] = gain(r + bg.gam * c - ca[sample - 1], gains)
                        bg.ws += lr_ws * delta[sample] * np.tile(bg.wc * ma[sample - 1], [2, 1]).T * 2 * (
                                    sa[sample - 1] - bg.ws) / bg.var
                        bg.wc += lr_wc * delta[sample] * ma[sample - 1]
                        bg.wa[choice] += (lr_wa / bg.lam) * delta[sample] * ma[sample - 1]

                    # update sample count
                    if choice == 2:
                        # draw a new sample
                        s = 2 * bernoulli.rvs(q) - 1
                        beads.append(s)
                        sample += 1
                    else:
                        break

            # pfc update
            ctx = np.max(ua[ti-1]) if (circular and ti > 0) else context
            Iin = np.matmul(pfc.ws, np.array([s, ctx])[:, None])     # input current
            Irec = np.matmul(pfc.J, h)                          # recurrent current
            z = Iin + Irec                                      # potential
            h = (1 - dt) * h + dt * (pfc.f(z)) + noise * npr.rand(N_pfc, 1)                  # activity
            u = np.matmul(pfc.wr, h)                            # output

            # striatal update
            m = np.exp(- np.sum((u.flatten() - bg.ws) ** 2, axis=1) / bg.var)  # striatum
            c = np.matmul(bg.wc, m)  # critic (ventral striatum)

            ua[ti] = u.T

            # save values for plotting
            stim[ei, ti], resp[ei, ti], stria[ei, ti], vals[ei, ti] = \
                np.array([s, context]), u.flatten(), m, c

            if ti == NT-1:
                b_model = ua[ti - int(0.8 * NT_sample):ti, 0].mean()
                y = np.array([b_model, 1 - b_model])

                # striatal activity
                m = np.exp(- np.sum((y - bg.ws) ** 2, axis=1) / bg.var)  # striatum
                c = np.matmul(bg.wc, m)  # critic (ventral striatum)
                a = np.matmul(bg.wa, m)  # actor (dorsal striatum)

                # choose an action
                p_a = np.exp(a / bg.lam) / np.sum(np.exp(a / bg.lam))  # softmax policy
                choice = np.nonzero(npr.multinomial(1, p_a))[0][0]

                # collect reward
                if choice == 2:
                    r = rewards[-1]  # sample
                elif choice == jar:
                    r = rewards[0]  # correct
                else:
                    r = rewards[1]  # incorrect

                # save values
                ya[sample], ma[sample], ca[sample], aa[sample], ra[sample] = y.T, m.T, c.T, a.T, r

                # weight updates
                if sample > 0:
                    delta[sample] = gain(r + bg.gam * c - ca[sample - 1], gains)
                    bg.ws += lr_ws * delta[sample] * np.tile(bg.wc * ma[sample - 1], [2, 1]).T * 2 * (
                            sa[sample - 1] - bg.ws) / bg.var
                    bg.wc += lr_wc * delta[sample] * ma[sample - 1]
                    bg.wa[choice] += (lr_wa / bg.lam) * delta[sample] * ma[sample - 1]

        # save epoch
        performance['contexts'].append(context)
        performance['jars'].append(jar)
        performance['beads'].append(beads)
        performance['actions'].append(choice)
        performance['samples'].append(sample + 1)
        performance['rpes'].append(delta)
        performance['rewards'].append(ra)
        print('\r' + str(ei + 1) + '/' + str(Nepochs), end='')

    return pfc, bg, stim, resp, stria, vals, performance


def plot_beliefnet(data, Nepochs=1000, seed=1):

    net, task, algo, learning = data['net'], data['task'], data['algo'], data['learning']

    # frequently used vars
    dt, NT, NT_sample, N, S, R = net.dt, task.NT, task.NT_sample, net.N, net.S, net.R

    # variables to save
    contexts, jars, beliefs = [], [], []
    rand_trl = npr.random_integers(0, Nepochs - 1, size=4)  # random trial
    stim = np.zeros((4, NT, S))  # save the inputs
    resp = np.zeros((4, NT, R))  # save the beliefs
    targ = np.zeros((4, NT, R))  # save the target

    for ei in range(Nepochs):

        # select context and jar
        if 'beads' in task.name:
            # pick a context (proportion of majority beads)
            context = npr.choice(task.context)
            # pick a jar (majority red or majority blue)
            jar = npr.random_integers(0, 1)
            # probability of drawing blue
            q = context if jar else 1 - context
            contexts.append(context), jars.append(jar)

        # initialize activity
        z0 = net.z0     # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)

        # save tensors for plotting
        sa = torch.zeros(NT, S)  # save the inputs for each time bin for plotting
        ua = torch.zeros(NT, R)  # save the beliefs

        beads = []
        for ti in range(NT):

            if ti % NT_sample == 0:
                # save true and model beliefs
                if ti > 0:
                    b_model = ua.detach().numpy()[ti-int(0.8 * NT_sample):ti, 0].mean()
                    beliefs.append([context, len(beads), b, b_model])   # context, nsamples, b_true, b_model

                # draw a new sample
                s = 2 * bernoulli.rvs(q) - 1
                beads.append(s)
                b = 1 / (1 + (q / (1-q)) ** (len(beads) - 2 * np.sum(np.array(beads) == (2*jar - 1))))
                task.ustar[ti:ti+NT_sample] = [b, 1-b]  # actual belief state

            # network update
            Iin = net.ws.mm(torch.as_tensor(np.array([s, context])[:, None]))   # input current
            Irec = net.J.mm(h)                      # recurrent current
            z = Iin + Irec                          # potential
            h = (1 - dt) * h + dt * (net.f(z))      # activity
            u = net.wr.mm(h)                        # output

            sa[ti], ua[ti] = torch.as_tensor(np.array([s, context])), u.T

            # save values for plotting
            if ei in rand_trl:
                example = np.nonzero(rand_trl == ei)[0][0]
                stim[example, ti], resp[example, ti], targ[example, ti] = \
                    np.array([s, context]), u.detach().numpy().flatten(), np.array([b, 1-b])

        # print epoch
        print('\r' + str(ei + 1) + '/' + str(Nepochs), end='')

        # save mse list and cond list
        learning['beads'].append(beads)

    # save contexts and jars
    learning['contexts'].append(contexts)
    learning['jars'].append(jars)

    # plot
    fig = plt.figure()
    for example in range(4):
        plt.subplot(2, 5, example+1)
        plt.title('Example ' + str(example+1))
        plt.plot(stim[example, :, 0]), plt.plot(stim[example, :, 1])
        plt.ylim((-1.2, 1.2)), plt.yticks(ticks=[-1, 0, 1])
        if example == 0: plt.ylabel('Inputs')
        plt.subplot(2, 5, example+6)
        plt.plot(targ[example, :, 0], color='xkcd:sage', alpha=.5), plt.plot(resp[example, :, 0], color='xkcd:sage')
        plt.plot(targ[example, :, 1], color='xkcd:coral', alpha=.5), plt.plot(resp[example, :, 1], color='xkcd:coral')
        plt.ylim((-0.2, 1.2)), plt.yticks(ticks=[0, 1])
        if example == 0:
            plt.ylabel('Outputs'), plt.legend(['P(A|data)', '_nolegend_', 'P(B|data)', '_nolegend_'], loc='best')
        plt.xlabel('Time')
    plt.subplot(2, 5, 5)
    plt.title('Model performance')
    context_easy = np.nonzero(np.array(beliefs)[:, 0] == 0.85)[0]
    context_diff = np.nonzero(np.array(beliefs)[:, 0] == 0.65)[0]
    plt.scatter(np.array(beliefs)[context_easy, 2], np.array(beliefs)[context_easy, 3],
                0.1 * np.array(beliefs)[context_easy, 1].astype(int), alpha=.1, color='k')
    plt.xticks(ticks=[0, 1]), plt.yticks(ticks=[0, 1])
    plt.ylabel('Model belief', labelpad=-160)
    plt.subplot(2, 5, 10)
    plt.scatter(np.array(beliefs)[context_diff, 2], np.array(beliefs)[context_diff, 3],
                0.1 * np.array(beliefs)[context_diff, 1].astype(int), alpha=.1, color='k')
    plt.xticks(ticks=[0, 1]), plt.yticks(ticks=[0, 1])
    plt.xlabel('True belief'), plt.ylabel('Model belief', labelpad=-160)


def plot_actorcritic(data, seed=1):

    net, task, algo, learning = data['net'], data['task'], data['algo'], data['learning']

    N, maxsamples = net.N, task.maxsamples

    # generate possible outcomes
    x = ["".join(seq) for seq in itertools.product("01", repeat=maxsamples)]
    x = np.array([np.fromstring(x[idx], 'u1') - ord('0') for idx in range(2 ** maxsamples)])

    # compute possible beliefs
    bvec = []
    k = np.cumsum(x, axis=1)
    n = np.tile(range(maxsamples), (2 ** maxsamples, 1))
    for q in task.context:
        bvec.append(np.unique(1. / (1 + (q / (1 - q)) ** (n - 2 * k))))
    bvec = np.unique(bvec)

    #
    ha = np.array([np.exp(-np.sum((np.array([bvec, 1 - bvec]) - net.ws[idx][:, None]) ** 2, axis=0) / net.var)
                   for idx in range(N)])
    ca = np.matmul(net.wc, ha)

    plt.figure()
    plt.subplot(231)
    plt.plot(uniform_filter1d(np.array(learning['rewards'])[(np.array(learning['contexts']) == 0.65).flatten()], size=500))
    plt.plot(uniform_filter1d(np.array(learning['rewards'])[(np.array(learning['contexts']) == 0.85).flatten()], size=500))
    plt.legend(['difficult context', 'easy context'], loc='best')
    plt.title('Actor-Critic learning')
    plt.xlabel('Epoch'), plt.ylabel('Reward')
    plt.subplot(232)
    plt.plot(bvec, ca.flatten(), color='k')
    plt.title('Critic')
    plt.xlabel('Belief in B'), plt.ylabel('Value')
    plt.subplot(233)
    plt.hist(np.array(learning['beliefs'])[(np.array(learning['actions']) == 2).flatten()], 5, density=True, color='xkcd:light grey')
    plt.hist(np.array(learning['beliefs'])[(np.array(learning['actions']) == 0).flatten()], 10, density=True, histtype='step', color='xkcd:sage')
    plt.hist(np.array(learning['beliefs'])[(np.array(learning['actions']) == 1).flatten()], 10, density=True, histtype='step', color='xkcd:coral')
    plt.legend(['choose A', 'choose B', 'sample'], loc='best')
    plt.title('Actor')
    plt.xlabel('Belief in B'), plt.ylabel('Probability density')
    plt.subplot(234)
    trl = 10000
    correct_A = np.sum(np.array(learning['actions'])[trl:][np.bitwise_and(np.array(learning['contexts'])[0, trl:] == 0.65,
                                                                          np.array(learning['jars'])[0, trl:] == 0)] == 0)
    correct_B = np.sum(np.array(learning['actions'])[trl:][np.bitwise_and(np.array(learning['contexts'])[0, trl:] == 0.65,
                                                                          np.array(learning['jars'])[0, trl:] == 1)] == 1)
    incorrect_A = np.sum(np.array(learning['actions'])[trl:][np.bitwise_and(np.array(learning['contexts'])[0, trl:] == 0.65,
                                                                            np.array(learning['jars'])[0, trl:] == 0)] == 1)
    incorrect_B = np.sum(np.array(learning['actions'])[trl:][np.bitwise_and(np.array(learning['contexts'])[0, trl:] == 0.65,
                                                                            np.array(learning['jars'])[0, trl:] == 1)] == 0)
    nochoice = np.sum(np.array(learning['actions'])[trl:][np.array(learning['contexts'])[0, trl:] == 0.65] == 2)
    correct, incorrect = correct_A + correct_B, incorrect_A + incorrect_B
    total = correct + incorrect + nochoice
    plt.bar([0, 1, 2], np.array([correct / total, incorrect / total, nochoice / total]), width=0.7)
    plt.ylim((0, 1)), plt.xticks(ticks=[0, 1, 2], labels=['correct', 'incorrect', 'no response'])
    plt.legend(['difficult context'])
    plt.ylabel('Probability')
    plt.subplot(235)
    trl = 10000
    correct_A = np.sum(np.array(learning['actions'])[trl:][np.bitwise_and(np.array(learning['contexts'])[0, trl:] == 0.85,
                                                                          np.array(learning['jars'])[0, trl:] == 0)] == 0)
    correct_B = np.sum(np.array(learning['actions'])[trl:][np.bitwise_and(np.array(learning['contexts'])[0, trl:] == 0.85,
                                                                          np.array(learning['jars'])[0, trl:] == 1)] == 1)
    incorrect_A = np.sum(np.array(learning['actions'])[trl:][np.bitwise_and(np.array(learning['contexts'])[0, trl:] == 0.85,
                                                                            np.array(learning['jars'])[0, trl:] == 0)] == 1)
    incorrect_B = np.sum(np.array(learning['actions'])[trl:][np.bitwise_and(np.array(learning['contexts'])[0, trl:] == 0.85,
                                                                            np.array(learning['jars'])[0, trl:] == 1)] == 0)
    nochoice = np.sum(np.array(learning['actions'])[trl:][np.array(learning['contexts'])[0, trl:] == 0.85] == 2)
    correct, incorrect = correct_A + correct_B, incorrect_A + incorrect_B
    total = correct + incorrect + nochoice
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    next(colors)
    plt.bar([0, 1, 2], np.array([correct / total, incorrect / total, nochoice / total]), width=0.7, color=next(colors))
    plt.ylim((0, 1)), plt.xticks(ticks=[0, 1, 2], labels=['correct', 'incorrect', 'no response'])
    plt.legend(['easy context'])
    plt.ylabel('Probability')
    plt.subplot(236)
    x1, y1 = ecdf(np.array(learning['samples'])[(np.array(learning['contexts']) == 0.65).flatten()])
    x1 = np.insert(x1, 0, x1[0])
    y1 = np.insert(y1, 0, 0.)
    x2, y2 = ecdf(np.array(learning['samples'])[(np.array(learning['contexts']) == 0.85).flatten()])
    x2 = np.insert(x2, 0, x2[0])
    y2 = np.insert(y2, 0, 0.)
    plt.plot(x1, y1, drawstyle='steps-post')
    plt.plot(x2, y2, drawstyle='steps-post')
    plt.legend(['difficult context', 'easy context'], loc='best')
    plt.xlabel('Number of samples'), plt.ylabel('Cumulative prob.')
    plt.show()


def plot_belief_actorcritic(data, seed=1):

    # load performance variables
    contexts, jars, actions, samples = data['performance']['contexts'], data['performance']['jars'], \
                                       data['performance']['actions'], data['performance']['samples']

    # load activity variables
    pfc, striatum, dopamine = data['pfc'], data['stria'], np.array(data['performance']['rpes'])[:, :, 0]
    Nepochs = np.shape(samples)[0]
    NT_samples = 100
    N_bg = data['actorcritic'].N

    # trials types
    trl_diff = (np.array(contexts) == 0.65).flatten()
    trl_easy = (np.array(contexts) == 0.85).flatten()
    trl_A = (np.array(jars) == 0).flatten()
    trl_B = (np.array(jars) == 1).flatten()

    # plot performance
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 6, figure=fig)

    # accuracy in difficult context
    correct_A = np.sum(np.array(actions)[np.bitwise_and(trl_diff, np.array(jars) == 0)] == 0)
    correct_B = np.sum(np.array(actions)[np.bitwise_and(trl_diff, np.array(jars) == 1)] == 1)
    incorrect_A = np.sum(np.array(actions)[np.bitwise_and(trl_diff, np.array(jars) == 0)] == 1)
    incorrect_B = np.sum(np.array(actions)[np.bitwise_and(trl_diff, np.array(jars) == 1)] == 0)
    nochoice = np.sum(np.array(actions)[trl_diff] == 2)
    correct, incorrect = correct_A + correct_B, incorrect_A + incorrect_B
    total = correct + incorrect + nochoice
    ax1 = fig.add_subplot(gs[0, :2])
    plt.bar([0, 1, 2], np.array([correct / total, incorrect / total, nochoice / total]), width=0.7)
    plt.ylim((0, 1)), plt.xticks(ticks=[0, 1, 2], labels=['correct', 'incorrect', 'no response'])
    plt.legend(['difficult context'])
    plt.ylabel('Probability')

    # accuracy in easy context
    correct_A = np.sum(np.array(actions)[np.bitwise_and(trl_easy, np.array(jars) == 0)] == 0)
    correct_B = np.sum(np.array(actions)[np.bitwise_and(trl_easy, np.array(jars) == 1)] == 1)
    incorrect_A = np.sum(np.array(actions)[np.bitwise_and(trl_easy, np.array(jars) == 0)] == 1)
    incorrect_B = np.sum(np.array(actions)[np.bitwise_and(trl_easy, np.array(jars) == 1)] == 0)
    nochoice = np.sum(np.array(actions)[trl_easy] == 2)
    correct, incorrect = correct_A + correct_B, incorrect_A + incorrect_B
    total = correct + incorrect + nochoice
    ax2 = fig.add_subplot(gs[0, 2:4])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    next(colors)
    plt.bar([0, 1, 2], np.array([correct / total, incorrect / total, nochoice / total]), width=0.7, color=next(colors))
    plt.ylim((0, 1)), plt.xticks(ticks=[0, 1, 2], labels=['correct', 'incorrect', 'no response'])
    plt.legend(['easy context'])
    plt.ylabel('Probability')

    # number of samples to decision
    ax3 = fig.add_subplot(gs[0, 4:])
    x1, y1 = ecdf(np.array(samples)[trl_diff])
    x1 = np.insert(x1, 0, x1[0])
    y1 = np.insert(y1, 0, 0.)
    x2, y2 = ecdf(np.array(samples)[trl_easy])
    x2 = np.insert(x2, 0, x2[0])
    y2 = np.insert(y2, 0, 0.)
    plt.plot(x1, y1, drawstyle='steps-post')
    plt.plot(x2, y2, drawstyle='steps-post')
    plt.legend(['difficult context', 'easy context'], loc='best')
    plt.xlabel('Number of samples'), plt.ylabel('Cumulative prob.')

    # pfc output activity
    samples_diff = np.median(np.array(samples)[trl_diff])
    samples_easy = np.median(np.array(samples)[trl_easy])
    trl_diffmedianA = np.bitwise_and.reduce((trl_diff, np.array(samples) <= samples_diff, trl_A))
    pfc_diffA = pfc[trl_diffmedianA, :(NT_samples * int(samples_diff))]
    pfc_diffA[pfc_diffA == 0] = 'nan'
    trl_diffmedianB = np.bitwise_and.reduce((trl_diff, np.array(samples) <= samples_diff, trl_B))
    pfc_diffB = pfc[trl_diffmedianB, :(NT_samples * int(samples_diff))]
    pfc_diffB[pfc_diffB == 0] = 'nan'
    pfc_diff = np.array([np.nanmean(pfc_diffA, axis=0), 1 - np.nanmean(pfc_diffB, axis=0)]).mean(axis=0)

    trl_easymedianA = np.bitwise_and.reduce((trl_easy, np.array(samples) <= samples_easy, trl_A))
    pfc_easyA = pfc[trl_easymedianA, :(NT_samples * int(samples_easy + 1))]
    pfc_easyA[pfc_easyA == 0] = 'nan'
    trl_easymedianB = np.bitwise_and.reduce((trl_easy, np.array(samples) <= samples_easy, trl_B))
    pfc_easyB = pfc[trl_easymedianB, :(NT_samples * int(samples_easy + 1))]
    pfc_easyB[pfc_easyB == 0] = 'nan'
    pfc_easy = np.array([np.nanmean(pfc_easyA, axis=0), 1 - np.nanmean(pfc_easyB, axis=0)]).mean(axis=0)

    ax4 = fig.add_subplot(gs[1, :2])
    plt.plot(pfc_diff[:, 1]), plt.plot(pfc_easy[:, 1])
    plt.gca().set_prop_cycle(None)
    plt.plot(pfc_diff[:, 0], alpha=0.5), plt.plot(pfc_easy[:, 0], alpha=0.5)
    plt.xlabel('Time'), plt.ylabel('Average output')

    # striatal activity
    b_pref = np.array(data['actorcritic'].ws)[:, 0] - np.array(data['actorcritic'].ws)[:, 1]
    sorted_by_pref = np.argsort(np.abs(b_pref))

    thresh_resp = 0.32
    trl_diffmedianA = np.bitwise_and.reduce((trl_diff, np.array(samples) == samples_diff, trl_A))
    trl_diffmedianB = np.bitwise_and.reduce((trl_diff, np.array(samples) == samples_diff, trl_B))
    trl_diffmedian = np.bitwise_or(trl_diffmedianA, trl_diffmedianB)
    stria_diff = striatum[trl_diffmedian, (NT_samples * int(samples_diff-3)):(NT_samples * int(samples_diff)), :]
    stria_diff[stria_diff == 0] = 'nan'
    stria_diff = np.nanmean(stria_diff[:, :, sorted_by_pref], axis=0)
    largeresponse = np.max(stria_diff, axis=0) > thresh_resp
    stria_diff = stria_diff[:, largeresponse]
    stria_diff /= np.tile(np.max(stria_diff, axis=0), [NT_samples * int(3), 1])
    ax4 = fig.add_subplot(gs[1, 2])
    plt.imshow(stria_diff.T, cmap='Blues'), plt.axis('tight')
    plt.xticks(ticks=[0, 100, 200], labels=['300', '200', '100']), plt.xlabel('Time to decision'), plt.ylabel('Neuron')

    thresh_resp = 0.34      # only plot neurons with peak activity above threshold
    trl_easymedianA = np.bitwise_and.reduce((trl_easy, np.array(samples) == samples_easy, trl_A))
    trl_easymedianB = np.bitwise_and.reduce((trl_easy, np.array(samples) == samples_easy, trl_B))
    trl_easymedian = np.bitwise_or(trl_easymedianA, trl_easymedianB)
    stria_easy = striatum[trl_easymedian, :(NT_samples * int(samples_easy)), :]
    stria_easy = np.nanmean(stria_easy[:, :, sorted_by_pref], axis=0)
    largeresponse = np.max(stria_easy, axis=0) > thresh_resp
    stria_easy = stria_easy[:, largeresponse]
    stria_easy /= np.tile(np.max(stria_easy, axis=0), [NT_samples * int(samples_easy), 1])
    ax4 = fig.add_subplot(gs[1, 3])
    plt.imshow(stria_easy.T, cmap='Oranges'), plt.axis('tight')
    plt.xticks(ticks=[0, 100], labels=['200', '100']), plt.xlabel('Time to decision'), plt.ylabel('Neuron')

    # dopamine activity aligned to start
    da_easy = dopamine[np.bitwise_and(np.array(samples) > 1, trl_easy), :]
    da_easy = np.nanmean(da_easy, axis=0)
    da_diff = dopamine[np.bitwise_and(np.array(samples) > 1, trl_diff), :]
    da_diff = np.nanmean(da_diff, axis=0)
    ax4 = fig.add_subplot(gs[1, 4])
    plt.plot(da_diff), plt.plot(da_easy)
    plt.xlabel('Sample'), plt.ylabel('Reward Prediction Error')

    # dopamine activity aligned to reward
    correct = np.bitwise_xor(jars, actions) == 0
    idx_easy = np.nonzero(np.bitwise_and.reduce((np.array(samples) > 1, trl_easy, correct)))[0]
    da_easy = np.empty((Nepochs, np.shape(dopamine)[1]))
    da_easy[:] = 'nan'
    for idx in idx_easy:
        sample = np.array(samples)[idx]
        da_easy[idx, -sample:] = dopamine[idx, :sample]
    idx_diff = np.nonzero(np.bitwise_and.reduce((np.array(samples) > 1, trl_diff, correct)))[0]
    da_diff = np.empty((Nepochs, np.shape(dopamine)[1]))
    da_diff[:] = 'nan'
    for idx in idx_diff:
        sample = np.array(samples)[idx]
        da_diff[idx, -sample:] = dopamine[idx, :sample]
    ax4 = fig.add_subplot(gs[1, 5])
    plt.plot(np.nanmean(da_diff, axis=0)), plt.plot(np.nanmean(da_easy, axis=0))

    incorrect = np.bitwise_xor(jars, actions) == 1
    idx_easy = np.nonzero(np.bitwise_and.reduce((np.array(samples) > 1, trl_easy, incorrect)))[0]
    da_easy = np.empty((Nepochs, np.shape(dopamine)[1]))
    da_easy[:] = 'nan'
    for idx in idx_easy:
        sample = np.array(samples)[idx]
        da_easy[idx, -sample:] = dopamine[idx, :sample]
    idx_diff = np.nonzero(np.bitwise_and.reduce((np.array(samples) > 1, trl_diff, incorrect)))[0]
    da_diff = np.empty((Nepochs, np.shape(dopamine)[1]))
    da_diff[:] = 'nan'
    for idx in idx_diff:
        sample = np.array(samples)[idx]
        da_diff[idx, -sample:] = dopamine[idx, :sample]
    ax4 = fig.add_subplot(gs[1, 5])
    plt.gca().set_prop_cycle(None)
    plt.plot(np.nanmean(da_diff, axis=0), alpha=0.5), plt.plot(np.nanmean(da_easy, axis=0), alpha=0.5)
    plt.xlim((5, 10))
    plt.xticks(ticks=[5, 6, 7, 8], labels=['4', '3', '2', '1']),
    plt.xlabel('Samples to decision')
    plt.legend(['diff_corr', 'easy_corr', 'diff_incorr', 'easy_incorr'], loc='lower left')
    plt.show()


def plot_schizo(control, schizo):

    # load performance variables
    contexts1, jars1, actions1, samples1 = control['performance']['contexts'], control['performance']['jars'], \
                                       control['performance']['actions'], control['performance']['samples']
    contexts2, jars2, actions2, samples2 = schizo['performance']['contexts'], schizo['performance']['jars'], \
                                       schizo['performance']['actions'], schizo['performance']['samples']

    # trial types
    trl_diff1 = (np.array(contexts1) == 0.65).flatten()
    trl_easy1 = (np.array(contexts1) == 0.85).flatten()
    trl_diff2 = (np.array(contexts2) == 0.65).flatten()
    trl_easy2 = (np.array(contexts2) == 0.85).flatten()

    # plot performance
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig)

    # accuracy in difficult context
    # controls
    correct_A = np.sum(np.array(actions1)[np.bitwise_and(trl_diff1, np.array(jars1) == 0)] == 0)
    correct_B = np.sum(np.array(actions1)[np.bitwise_and(trl_diff1, np.array(jars1) == 1)] == 1)
    incorrect_A = np.sum(np.array(actions1)[np.bitwise_and(trl_diff1, np.array(jars1) == 0)] == 1)
    incorrect_B = np.sum(np.array(actions1)[np.bitwise_and(trl_diff1, np.array(jars1) == 1)] == 0)
    nochoice1 = np.sum(np.array(actions1)[trl_diff1] == 2)
    correct1, incorrect1 = correct_A + correct_B, incorrect_A + incorrect_B
    total1 = correct1 + incorrect1 + nochoice1
    control_means = np.array([correct1 / total1, incorrect1 / total1, nochoice1 / total1])
    # schizo
    correct_A = np.sum(np.array(actions2)[np.bitwise_and(trl_diff2, np.array(jars2) == 0)] == 0)
    correct_B = np.sum(np.array(actions2)[np.bitwise_and(trl_diff2, np.array(jars2) == 1)] == 1)
    incorrect_A = np.sum(np.array(actions2)[np.bitwise_and(trl_diff2, np.array(jars2) == 0)] == 1)
    incorrect_B = np.sum(np.array(actions2)[np.bitwise_and(trl_diff2, np.array(jars2) == 1)] == 0)
    nochoice2 = np.sum(np.array(actions2)[trl_diff2] == 2)
    correct2, incorrect2 = correct_A + correct_B, incorrect_A + incorrect_B
    total2 = correct2 + incorrect2 + nochoice2
    schizo_means = np.array([correct2 / total2, incorrect2 / total2, nochoice2 / total2])

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(3)  # the label locations
    width = 0.35  # the width of the bars
    plt.bar(x - width / 2, control_means, width, label='Control')
    plt.gca().set_prop_cycle(None)
    plt.bar(x + width / 2, schizo_means, width, label='Schizo', alpha=.5)
    plt.legend(fontsize=16)
    plt.ylim((0, 1)), plt.xticks(ticks=[0, 1, 2], labels=['correct', 'incorrect', 'no response'])
    plt.title('Difficult context')
    plt.ylabel('Probability')

    # accuracy in easy context
    # controls
    correct_A = np.sum(np.array(actions1)[np.bitwise_and(trl_easy1, np.array(jars1) == 0)] == 0)
    correct_B = np.sum(np.array(actions1)[np.bitwise_and(trl_easy1, np.array(jars1) == 1)] == 1)
    incorrect_A = np.sum(np.array(actions1)[np.bitwise_and(trl_easy1, np.array(jars1) == 0)] == 1)
    incorrect_B = np.sum(np.array(actions1)[np.bitwise_and(trl_easy1, np.array(jars1) == 1)] == 0)
    nochoice1 = np.sum(np.array(actions1)[trl_easy1] == 2)
    correct1, incorrect1 = correct_A + correct_B, incorrect_A + incorrect_B
    total1 = correct1 + incorrect1 + nochoice1
    control_means = np.array([correct1 / total1, incorrect1 / total1, nochoice1 / total1])
    # schizo
    correct_A = np.sum(np.array(actions2)[np.bitwise_and(trl_easy2, np.array(jars2) == 0)] == 0)
    correct_B = np.sum(np.array(actions2)[np.bitwise_and(trl_easy2, np.array(jars2) == 1)] == 1)
    incorrect_A = np.sum(np.array(actions2)[np.bitwise_and(trl_easy2, np.array(jars2) == 0)] == 1)
    incorrect_B = np.sum(np.array(actions2)[np.bitwise_and(trl_easy2, np.array(jars2) == 1)] == 0)
    nochoice2 = np.sum(np.array(actions2)[trl_easy2] == 2)
    correct2, incorrect2 = correct_A + correct_B, incorrect_A + incorrect_B
    total2 = correct2 + incorrect2 + nochoice2
    schizo_means = np.array([correct2 / total2, incorrect2 / total2, nochoice2 / total2])

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(3)  # the label locations
    width = 0.35  # the width of the bars
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    next(colors)
    plt.bar(x - width / 2, control_means, width, label='Control', color=next(colors))
    plt.gca().set_prop_cycle(None)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    next(colors)
    plt.bar(x + width / 2, schizo_means, width, label='Schizo', alpha=.5, color=next(colors))
    plt.legend(fontsize=16)
    plt.ylim((0, 1)), plt.xticks(ticks=[0, 1, 2], labels=['correct', 'incorrect', 'no response'])
    plt.title('Easy context')
    plt.ylabel('Probability')

    # number of samples to decision
    ax1 = fig.add_subplot(gs[0, 2])
    # control
    x1, y1 = ecdf(np.array(samples1)[trl_diff1])
    x1 = np.insert(x1, 0, x1[0])
    y1 = np.insert(y1, 0, 0.)
    x2, y2 = ecdf(np.array(samples1)[trl_easy1])
    x2 = np.insert(x2, 0, x2[0])
    y2 = np.insert(y2, 0, 0.)
    plt.plot(x1, y1, drawstyle='steps-post')
    plt.plot(x2, y2, drawstyle='steps-post')
    # schizo
    x1, y1 = ecdf(np.array(samples2)[trl_diff2])
    x1 = np.insert(x1, 0, x1[0])
    y1 = np.insert(y1, 0, 0.)
    x2, y2 = ecdf(np.array(samples2)[trl_easy2])
    x2 = np.insert(x2, 0, x2[0])
    y2 = np.insert(y2, 0, 0.)
    plt.gca().set_prop_cycle(None)
    plt.plot(x1, y1, drawstyle='steps-post', linestyle='dashed')
    plt.plot(x2, y2, drawstyle='steps-post', linestyle='dashed')
    plt.legend(['difficult context', 'easy context'], loc='best')
    plt.xlabel('Number of samples'), plt.ylabel('Cumulative prob.')
    plt.suptitle('Circular inference')
    plt.show()