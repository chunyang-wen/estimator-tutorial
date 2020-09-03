#!/bin/sh

killed_exit=$1
file=main_dist.py

mkdir -p logs
FILE=logs/pid.file
if [ -f ${FILE} ]
then
    for i in `awk '{print $NF}' ${FILE}`
    do
        kill -9 $i
    done
fi

[[ ! -z ${killed_exit} ]] && exit 0


\rm -rf logs/*

function get_port() {
    local avaiable_port=$(python -c \
        'from __future__ import print_function;\
        import socket; s = socket.socket(); s.bind(("", 0)); \
        print(s.getsockname()[1])')
    echo $avaiable_port
}

function get_host() {
    size=$1
    hosts=""
    PORT=$(get_port)
    for i in `seq ${size}`
    do
        if [ -z "${hosts}" ]
        then
            hosts="localhost:"${PORT}
        else
            hosts=${hosts}";localhost:"${PORT}
        fi
        PORT=$(get_port)
    done

    echo ${hosts}
}

function start_tasks() {
    type=$1
    size=$2
    echo "Start ${type}, number: ${size}"
    ((size-=1))
    for i in `seq 0 ${size}`
    do
        index=$i
        python ${file} \
            --chief-hosts ${chief_hosts} \
            --evaluator-hosts ${evaluator_hosts} \
            --ps-hosts ${ps_hosts} \
            --worker-hosts ${worker_hosts} \
            --worker-type ${type} --worker-index ${index} &> logs/${type}.log.$i &
        echo "${type}: "${i}" pid= "$! >> logs/pid.file
    done

}

PS_SIZE=1
WORKER_SIZE=2
CHIEF_SIZE=1
EVALUATOR_SIZE=1
ps_hosts=$(get_host ${PS_SIZE})

worker_hosts=$(get_host ${WORKER_SIZE})
chief_hosts=$(get_host ${CHIEF_SIZE})

echo "ps = "${ps_hosts}
echo "worker = "${worker_hosts}
echo "chief = "${chief_hosts}
start_tasks "ps" ${PS_SIZE}

echo "Sleep 3s before start worker"
sleep 3s

start_tasks "worker" ${WORKER_SIZE}

type="chief"
index=0

python ${file} \
    --chief-hosts ${chief_hosts} \
    --evaluator-hosts ${evaluator_hosts} \
    --ps-hosts ${ps_hosts} \
    --worker-hosts ${worker_hosts} \
    --worker-type ${type} --worker-index ${index} &> logs/chief.log.$i
