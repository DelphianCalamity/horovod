#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -pp|--profiling-path)
    PROFILING="$2"
    shift # past argument
    shift # past value
    ;;
    -e|--example)
    EXAMPLE="$2"
    shift # past argument
    shift # past value
    ;;
    -u|--user)
    USER="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--port)
    PORT="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
   shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "PROFILING PATH = ${PROFILING}"
echo "EXAMPLE        = ${EXAMPLE}"
echo "USER           = ${USER}"
echo "PORT           = ${PORT}"

if [ -z "${PROFILING_PATH}" ]
then
        profiling="-x HOROVOD_TIMELINE=${PROFILING}"
fi

cmd=\
"
docker run --rm -it --name ${USER} --runtime=nvidia \
--network=host --privileged \
-v /home/ubuntu/${USER}:/home/ubuntu/ \
-v /mnt/scratch:/mnt/scratch \
-v /home/ubuntu/.ssh/shared:/root/.ssh \
-v /sys/bus/pci/drivers:/sys/bus/pci/drivers \
-v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
-v /sys/devices/system/node:/sys/devices/system/node \
-v /dev:/dev \
-v /lib/modules:/lib/modules \
-w /home/ubuntu/${USER} \
kelkost/horovod:bloom-filter \
mpirun \
-x CUDA_VISIBLE_DEVICES=1 --allow-run-as-root \
-wdir /home/ubuntu/kelly/ \
--mca orte_base_help_aggregate 0 \
-x NCCL_IB_DISABLE=1 \
-x NCCL_SOCKET_IFNAME=enp1s0f1 \
-x HOROVOD_GLOO=0 \
-x HOROVOD_FUSION_THRESHOLD=0 \
-x LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64 \
-np 2 -H 10.0.0.131:1,10.0.0.132:1 \
${profiling} \
--display-map --display-allocation -map-by slot -bind-to none -nooversubscribe -mca plm_rsh_args \
-p ${PORT}\
--mca pml ob1 --mca btl ^openib --mca btl_tcp_if_exclude docker0,eno2,ens4f0,ens4f1,lo,virbr0,ib0,ib1, \
--tag-output \
python ${EXAMPLE}
"

echo ${cmd}
${cmd}

