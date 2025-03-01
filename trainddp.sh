GPUS=$1
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))
PORT=$((50000 + $RANDOM % 4000))

shift 1
echo "Launching exp on $GPUS... PORT $PORT"
MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPUS} torchrun --master_port ${PORT} --nproc_per_node=${NB_GPUS} main.py train $@