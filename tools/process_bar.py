def myprogress(current, whole=1, n=30, bars=u'▕▏▎▍▌▋▊▉', full='▉', empty='▕'): 
    """ current and whole can be an element of a list being iterated, or just two numbers """
    p = (whole.index(current))/len(whole)+1e-9 if type(whole)==list else current/whole+1e-9 
    return f"{full*int(p*n)}{bars[int(len(bars)*((p*n)%1))]}{empty*int((1-p)*n)} {p*100:>6.2f}%" 

def pbiter(it, *, total = None, width = 60, _cfg = {'idx': -1, 'pbs': {}, 'lline': 0}):
    try:
        total = total or len(it)
    except:
        total = None
    
    _cfg['idx'] += 1
    idx = _cfg['idx']
    pbs = _cfg['pbs']
    pbs[idx] = [0, total, 0]
    
    def Show():
        line2 = ' '.join([
            myprogress(e[1][0], max(e[1][0], e[1][1] or
                max(1, e[1][0]) / max(.1, e[1][2])), width // len(pbs))
            for e in sorted(pbs.items(), key = lambda e: e[0])
        ])
        line = line2 + ' ' * (max(0, _cfg['lline'] - len(line2)) + 0)
        print(line, end = '\r', flush = True)
        _cfg['lline'] = len(line2)
    
    try:
        Show()
        for e in it:
            yield e
            pbs[idx][0] += 1
            pbs[idx][2] += (1. - pbs[idx][2]) * .1
            Show()
        pbs[idx][2] = 1.
        Show()
    finally:
        del pbs[idx]