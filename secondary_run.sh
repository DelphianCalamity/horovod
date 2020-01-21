#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
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

echo "USER           = ${USER}"
echo "PORT           = ${PORT}"

cmd=\
"
docker run --rm -it --name ${USER} --runtime=nvidia --network=host --privileged \
-v /data/scratch/:/data/scratch/ \
-v /home/ubuntu/${USER}:/home/ubuntu \
-v /mnt/scratch:/mnt/scratch \
-v /home/ubuntu/.ssh/shared:/root/.ssh \
-v /sys/bus/pci/drivers:/sys/bus/pci/drivers \
-v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
-v /sys/devices/system/node:/sys/devices/system/node \
-v /dev:/dev \
-v /lib/modules:/lib/modules \
-w /home/ubuntu/${USER} \
kelkost/horovod:bloom-filter \
bash -c '/usr/sbin/sshd -p ${PORT}; sleep infinity'
"
echo ${cmd}
${cmd}