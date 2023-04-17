CUDA_VISIBLE_DEVICES=1,2,3 python main_supcon.py --batch_size 512 --learning_rate 0.5 --temp 0.1 --cosine

CUDA_VISIBLE_DEVICES=1,2,3 python main_linear.py --batch_size 512 --learning_rate 1 --ckpt /path/to/model.pth 