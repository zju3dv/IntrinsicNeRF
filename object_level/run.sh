mkdir logs/chair
touch logs/chair/out_message
CUDA_VISIBLE_DEVICES=0 nohup python run_nerf.py --config configs/chair.txt --exp chair --datadir YourData_Dir/chair > logs/chair/out_message 2>&1 &

mkdir logs/drums
touch logs/drums/out_message
CUDA_VISIBLE_DEVICES=0 nohup python run_nerf.py --config configs/drums.txt --exp drums --datadir YourData_Dir/drums > logs/drums/out_message 2>&1 &

mkdir logs/ficus
touch logs/ficus/out_message
CUDA_VISIBLE_DEVICES=0 nohup python run_nerf.py --config configs/ficus.txt --exp ficus --datadir YourData_Dir/ficus > logs/ficus/out_message 2>&1 &

mkdir logs/lego
touch logs/lego/out_message
CUDA_VISIBLE_DEVICES=0 nohup python run_nerf.py --config configs/lego.txt --exp lego --datadir YourData_Dir/lego > logs/lego/out_message 2>&1 &

mkdir logs/chair2
touch logs/chair2/out_message
CUDA_VISIBLE_DEVICES=0 nohup python run_nerf.py --config configs/chair2.txt --exp chair2 --datadir YourData_Dir/chair2 > logs/chair2/out_message 2>&1 &

mkdir logs/jugs
touch logs/jugs/out_message
CUDA_VISIBLE_DEVICES=0 nohup python run_nerf.py --config configs/jugs.txt --exp jugs --datadir YourData_Dir/jugs > logs/jugs/out_message 2>&1 &

mkdir logs/air_baloons
touch logs/air_baloons/out_message
CUDA_VISIBLE_DEVICES=0 nohup python run_nerf.py --config configs/air_baloons.txt --exp air_baloons --w_s 10.0 --w_f 0.01 --datadir YourData_Dir/air_baloons > logs/air_baloons/out_message 2>&1 &

mkdir logs/hotdog
touch logs/hotdog/out_message
CUDA_VISIBLE_DEVICES=0 nohup python run_nerf.py --config configs/hotdog.txt --exp hotdog --w_i1 0.1 --w_f 0.01 --datadir YourData_Dir/hotdog > logs/hotdog/out_message 2>&1 &


