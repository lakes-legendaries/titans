"""Get initial win frac, against random"""

from titans.ai import Trainer


trainer = Trainer()
win_frac = []
for _ in range(20):
    win_frac.append(trainer.play(use_random=True))
    print(win_frac)
    trainer.train()
    print(win_frac)
