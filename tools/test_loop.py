import os

for epoch_id in range(100, 160, 10):
    print(f'epoch: {epoch_id}')
    os.system(f'python test.py --config ../configs/my_configs/vit_s_GID.py ' +
              f'--checkpoint results/EXP20220925_vits_GID/epoch_{epoch_id}.pth')


