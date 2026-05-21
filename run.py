import subprocess
# 设置CUDA环境变量的命令
cmd = "python tkbc/learner.py --dataset ICEWS14 --model testmodel --rank 1594 --emb_reg 1e-1 --time_reg 1e-4"
# cmd = "python tkbc/learner.py --dataset yago15k --model testmodel --rank 364 --max_epoch 10 --learning_rate 10e-1 --emb_reg 1e-1 --time_reg 1e-4"
# cmd = "python tkbc/learner.py --dataset gdelt --model testmodel --rank 1256 --max_epoch 1000 --learning_rate 3e-1 --emb_reg 1e-5 --time_reg 1e-2"
for i in range(0, 50):
    subprocess.run(cmd, shell=True)