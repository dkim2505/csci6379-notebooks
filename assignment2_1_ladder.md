| step | what changed | validation Dice (same hospital) | hidden hospitals | grade |
|---|---|---|---|---|
| starter | the small U-Net from class, 10 epochs, LR 1e-4 | 0.54 | **0.15** | D |
| + FIX 1 | LR 1e-3, 40 epochs, one-cycle schedule | 0.73 | **0.32** | C |
| + FIX 2 | width 32, four levels (7.8M parameters) | 0.79 | **0.37** | C |
| + FIX 3 | flips, 90° turns, small rotations and zoom; 60 epochs | 0.84 | **0.42** | C |
| + FIX 4 | mild colour changes (brightness, contrast, saturation) | 0.83 | **0.44** | C |
| + FIX 5 | BCE + soft Dice loss | 0.85 | **0.54** | B |

Two things to notice. Validation Dice and the hidden score are far apart at
every step. And FIX 4 did almost nothing for validation Dice, because the
validation images have the training hospital's colours; whether it helped at
all only shows up on the hidden hospitals.
