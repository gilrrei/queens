#!/bin/bash
##########################################
#                                        #
#  Specify your PBS directives           #
#                                        #
##########################################
# Job name:
#PBS -N {job_name}
# Number of nodes and processors per node (ppn)
#PBS -l nodes={pbs_nodes}:ppn={pbs_ppn}
# Walltime: (hours:minutes:seconds)
#PBS -l walltime={walltime}
# Executing queue
#PBS -q {pbs_queue}
###########################################

##########################################
#                                        #
#  Specify your paths                    #
#                                        #
##########################################
WORKDIR=/scratch/PBS_$PBS_JOBID
DESTDIR={DESTDIR}  # output directory for run
EXE={EXE} # either CAE excutable or singularity image
INPUT='{INPUT}'  # either input file or, for singularity, list of arguments specifying run
OUTPUTPREFIX={OUTPUTPREFIX}
##########################################
#                                        #
#       RESTART SPECIFICATION            #
RESTART=0                                #
RESTART_FROM_PREFIX=xxx                  #
##########################################

##########################################
#                                        #
#     POST- AND POST-POST-PROCESSING     #
#     SPECIFICATION                      #
#                                        #
##########################################
DoPostprocess={POSTPROCESSFLAG} # post-processing flag for non-singularity run
POSTEXE={POSTEXE} # post-processing executable for non-singularity run
DoPostpostprocess={POSTPOSTPROCESSFLAG} # post- and post-post-processing flag for singularity run

#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################
# This is not a suggestion, this is a rule.
# Talk to admin before touching this section.
source {CLUSTERSCRIPT}
trap 'EarlyTermination; StageOut' 2 9 15 18
DoChecks
StageIn
RunProgram
wait
#RunPostprocessor
# Post-processing for non-singularity run
# (only for post-processor without additional options so far) 
if [ $DoPostprocess = true ]
then
  if [ $RESTART -le 0 ]
  then
    $MPI_RUN $MPIFLAGS -np {nposttasks} $POSTEXE --file=$WORKDIR/$OUTPUTPREFIX
  else
    echo Attention! You are postprocessing files from a restarted simulation. Only the new data is postprocessed, as only this data is available.
    echo
    $MPI_RUN $MPIFLAGS -np {nposttasks} $POSTEXE --file=$WORKDIR/$OUTPUTPREFIX
  fi
fi
wait
StageOut
#Show
echo
echo "Job finished with exit code $? at: `date`"
# ------- FINISH AND CLEAN SINGULARITY JOB (DONE ON MASTER/LOGIN NODE!) -------
wait
# Post- and post-post-processing for singularity run
# (cd back into home since pwd does not exist anymore)
if [ $DoPostpostprocess = true ]
then
  $MPI_RUN $MPIFLAGS -np {nposttasks} $EXE $INPUT $WORKDIR '--post=true'
fi
# END ################## DO NOT TOUCH ME #########################
