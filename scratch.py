"""Get initial win frac, against random"""

from titans.ai import Trainer


trainer = Trainer()
win_frac = []
for _ in range(20):
    win_frac.append(trainer.play())
    trainer.train()
    print(win_frac)
