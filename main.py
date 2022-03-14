from train import train_pfc, train_bg, combine_pfc_bg, plot_beliefnet, plot_actorcritic, plot_belief_actorcritic
import torch

train_beliefnet, train_actorcritic, belief_actorcritic, plot_models = False, False, True, True

if train_beliefnet:
    # initialize
    print('\r Training PFC', end='\n')
    # train using bptt
    learningsites = ('ws', 'J', 'wr')                             # 'ws', 'J', 'wr'
    pfc, task, algo, learning = \
        train_pfc(arch='PFC', N=128, S=2, R=2, task='beads-belief', maxsamples=10, context=(.65, .85),
                  algo='Adam', Nepochs=10000, lr=1e-4, learningsites=learningsites, seed=1)
    # save
    torch.save({'net': pfc, 'task': task, 'algo': algo, 'learning': learning},
               '..//Data//beliefnet.pt')

if train_actorcritic:
    # initialize
    print('\r Training Actor-Critic', end='\n')
    bg, task, algo, learning = \
        train_bg(arch='BG', N=128, S=2, Rc=1, Ra=3, task='beads-choice', maxsamples=10,
                 context=(.65, .85), rewards=(20, -200, -2), algo='TD', Nepochs=20000, lr=(1e-8, 1e-5, 1e-5), seed=1)
    # save lr=(1e-7, 1e-4, 1e-4), rew=(20, -400, -1), Nepochs>20K, maxsamples=50
    torch.save({'net': bg, 'task': task, 'algo': algo, 'learning': learning},
               '..//Data//actorcritic.pt')

if belief_actorcritic:
    # initialize
    print('\r Running Combined models', end='\n')
    data = torch.load('..//Data//beliefnet.pt')
    pfc, task = data['net'], data['task']
    data = torch.load('..//Data//actorcritic.pt')
    bg, algo = data['net'], data['algo']
    pfc, bg, stim, resp, stria, vals, performance = combine_pfc_bg(pfc, bg, task, algo, Nepochs=1000, maxsamples=10,
                                                          context=(.65, .85), rewards=(20, -200, -2), gains=(1, 1), seed=1)
    torch.save({'beliefnet': pfc, 'actorcritic': bg, 'stim': stim, 'pfc': resp, 'stria': stria,
                'vals': vals, 'performance': performance}, '..//Data//belief_actorcritic.pt')

if plot_models:
    # initialize
    #print('\r Plotting beliefnet', end='\n')
    #data_pfc = torch.load('..//Data//beliefnet.pt')
    #plot_beliefnet(data_pfc, seed=1)
    print('\r Plotting actorcritic', end='\n')
    data_bg = torch.load('..//Data//actorcritic.pt')
    plot_actorcritic(data_bg, seed=1)
    data_belief_actorcritic = torch.load('..//Data//belief_actorcritic.pt')
    plot_belief_actorcritic(data_belief_actorcritic, seed=1)
