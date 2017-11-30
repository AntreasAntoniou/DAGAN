#! /bin/bash

#####
# Updated: 2017-06-01
# Reason: Fix bug introduced in last update.
#####

SCRIPT_NAME='GPULockScript'

# If CUDA_VISIBLE_DEVICES has already been set (for example, by the cluster in the prolog)
# do not attempt to reset it:
if [[ -n "$CUDA_VISIBLE_DEVICES" ]] ; then
    exit 0
fi

# Set the directory to store lockfiles in.
LOCK_PATH=/disk/scratch/gpu_locks
#LOCK_PATH=/disk/scratch/

if [ ! -d "$LOCK_PATH" ] ; then
    mkdir "$LOCK_PATH"
    chmod a+w "$LOCK_PATH"
fi

# Get the current node's name
shortname=${HOSTNAME%%.*}

# Set up lockfile to stop another job locking a GPU while we are in the process of doing so.
lockfile -3 -l 60 $LOCK_PATH/LOCK

# Get a list of running jobs, gpus in the current node and existing lock files.
runningjobs=$(/opt/sge/bin/lx-amd64/qstat -u '*'  | grep "$shortname"| awk '{printf("%d.%d\n",$1,$10)}'| paste -s -d'\|')
gpuids=$(nvidia-smi --query-gpu=gpu_uuid,memory.used --format=csv,noheader,nounits | sed 's/, /,/' |  paste -s -d' ' -)
locks=$(ls $LOCK_PATH/GPU-*@* 2>/dev/null | xargs basename -a 2>/dev/null | paste -s -d' ' -)

# Find the number of gpus requested by this job.
gpuNum=$(/opt/sge/bin/lx-amd64/qstat -j $JOB_ID | grep -P "hard resource_list:.*\bgpu=\d+\b" | sed -r 's/.*\bgpu=([0-9]+)\b.*/\1/')

if [[ ! ${SGE_TASK_ID} == "undefined" ]] ; then
    JOB_TASK_ID="${SGE_TASK_ID}"
else
    JOB_TASK_ID="0"
fi

# If no gpus were explicitly requested, assume one was wanted.
if [ -z "$gpuNum" ] ; then
    gpuNum="1"
fi

# Choose the gpus that are not already locked, and which have the least memory used.
freegpu=$(python2.7 -c "
import os
import sys
gpuids={gId : int(mem) for gId, mem in map(lambda x: x.split(','), '$gpuids'.split(' '))}
locks=set('$locks'.split())
runningjobs='$runningjobs'.split('|')
unlockedjobs=runningjobs[:]
unlockedjobs.remove('${JOB_ID}.${JOB_TASK_ID}')
for lock in locks:
  sys.stderr.write('${SCRIPT_NAME}: Lock file: %s'%lock)
  gpid, jobid = lock.split('@')
  # For backwards compatability
  if jobid.count('.') == 0:
    jobid = jobid+'.0'
  if jobid.split('.')[1] == 'undefined':
    jobid = '.'.join([jobid[:-10],'0'])
  if jobid in runningjobs:
    sys.stderr.write(' -- job still running.\n')
    try:
      unlockedjobs.remove(jobid)
    except ValueError:
      pass
    if gpid in gpuids:
      del gpuids[gpid]
  else:
    # If a job isn't running, it shouldn't have a lock file.
    try:
      sys.stderr.write(' -- removed.\n')
      os.remove('$LOCK_PATH/'+lock)
    except :
      sys.stderr.write(' -- ERROR: Old lock file could not be removed.\n')
      sys.stderr.write(' '.join(sys.exc_info()[:2]))
      sys.stderr.write('\n')
if len(unlockedjobs) > 0:
    sys.stderr.write('${SCRIPT_NAME}: Running jobs without locks: %s \n'%str(unlockedjobs))
    sys.stderr.write('${SCRIPT_NAME}: (The lock file may have been removed by an old version of the lock script.)\n')
gNum = '$gpuNum'
try:
  gNum = int(gNum)
except:
  gNum = 0
if len(gpuids) == 0 or len(gpuids) < gNum:
  # If we can't get as many gpus as we wanted, register an error.
  # this should not happen
  sys.stderr.write('${SCRIPT_NAME}: ERROR: Job requested %d gpus, only %d are free on this node.\n'%(gNum,len(gpuids)))
  sys.exit(1)
else:
  print ' '.join(map(lambda x: x[0], sorted(gpuids.items(), key=lambda (k,v): v)[:gNum]))
")

# If we couldn't lock as many gpus as were requested, stop the job.
# (To get to this situation, the scheduler messed up, or another job has locked more gpus
# than it asked for.)
EXIT_CODE=$?
if [ ! $EXIT_CODE -eq 0 ] ; then
    rm  $LOCK_PATH/LOCK
    exit $EXIT_CODE
fi

# Create a lock file.
OLDMASK=`umask`
[[ ! -z $freegpu ]] && for gpuId in $freegpu; do (umask 113; touch "${LOCK_PATH}/${gpuId}@${JOB_ID}.${JOB_TASK_ID}" ); done;
umask $OLDMASK
rm  $LOCK_PATH/LOCK

export CUDA_VISIBLE_DEVICES=`echo "$freegpu" | sed 's/ /,/g'`

mkdir -p /disk/scratch/s1473470

export TMPDIR=/disk/scratch/s1473470/
export TMP=/disk/scratch/s1473470/