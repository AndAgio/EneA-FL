#!/usr/bin/env bash
#
# script to preprocess data

# --------------------
# parse arguments

NAME="sent140" # name of the dataset, equivalent to directory name
SAMPLEMODE="" # -s tag, iid or niid
IUSER="" # --iu tag, # of users if iid sampling
SPW="" # maximum number of sample per user
SFRAC="" # --sf tag, fraction of data to sample
TFRAC="" # --tf tag, fraction of data in training set
SAMPLING_SEED="" # --smplseed, seed specified for sampling of data
SPLIT_SEED="" # --spltseed, seed specified for train/test data split
VERIFICATION_FILE="" # --verify <fname>, check if JSON files' MD5 matches given digest

META_DIR='meta'

while [[ $# -gt 0 ]]
do
key="$1"

# TODO: Use getopts instead of creating cases!
case $key in
    --name)
    NAME="$2"
    shift # past argument
    if [ ${NAME:0:1} = "-" ]; then
        NAME="sent140"
    else
        shift # past value
    fi
    ;;
    -s)
    SAMPLEMODE="$2"
    shift # past argument
    if [ ${SAMPLEMODE:0:1} = "-" ]; then
        SAMPLEMODE=""
    else
        shift # past value
    fi
    ;;
    --iu)
    IUSER="$2"
    shift # past argument
    if [ ${IUSER:0:1} = "-" ]; then
        IUSER=""
    else
        shift # past value
    fi
    ;;
    --spw)
    SPW="$2"
    shift # past argument
    if [ ${SPW:0:1} = "-" ]; then
        SPW=""
    else
        shift # past value
    fi
    ;;
    --sf)
    SFRAC="$2"
    shift # past argument
    if [ ${SFRAC:0:1} = "-" ]; then
        SFRAC=""
    else
        shift # past value
    fi
    ;;
    --tf)
    TFRAC="$2"
    shift # past argument
    if [ ${TFRAC:0:1} = "-" ]; then
        TFRAC=""
    else
        shift # past value
    fi
    ;;
    --smplseed)
    SAMPLING_SEED="$2"
    shift # past argument
    ;;
    --spltseed)
    SPLIT_SEED="$2"
    shift # past argument
    ;;
    --verify)
    VERIFICATION_FILE="$2"
    shift # past argument
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

# --------------------
# check if running in verification mode

if [ -n "${VERIFICATION_FILE}" ]; then
    pushd ../$NAME >/dev/null 2>/dev/null
        TMP_FILE=`mktemp /tmp/dir-checksum.XXXXXXX`
        find 'data/' -type f -name '*.json' -exec md5sum {} + | sort -k 2 > ${TMP_FILE}
        DIFF=`diff --brief ${VERIFICATION_FILE} ${TMP_FILE}`
        if [ $? -ne 0 ]; then
            echo "${DIFF}"
            diff ${TMP_FILE} ${VERIFICATION_FILE} 
            echo "Differing checksums found - please verify"
        else
            echo "Matching JSON files and checksums found!"
        fi
    popd >/dev/null 2>/dev/null
    exit 0
fi

# --------------------
# preprocess data

CONT_SCRIPT=true
cd ../$NAME

# setup meta directory if doesn't exist
if [ ! -d ${META_DIR} ]; then
    mkdir -p ${META_DIR}
fi
META_DIR=`realpath ${META_DIR}`

# download data and convert to .json format

if [ ! -d "data/all_data" ]; then
    cd preprocess
    ./data_to_json.sh
    cd ..
fi

NAMETAG="--name $NAME"

# sample data
IUSERTAG=""
if [ ! -z $IUSER ]; then
    IUSERTAG="--u $IUSER"
fi
SAMPLEMODETAG=""
if [ ! -z $SAMPLEMODE ]; then
  SAMPLEMODETAG="--sampling_mode $SAMPLEMODE"
fi
NUSERTAG=""
if [ ! -z $IUSER ]; then
    NUSERTAG="--n_workers $IUSER"
fi
SAMPPERWORKER=""
if [ ! -z $SPW ]; then
    SAMPPERWORKER="--spw $SPW"
fi
SFRACTAG=""
if [ ! -z $SFRAC ]; then
    SFRACTAG="--fraction $SFRAC"
fi

if [ ! -d "data/${IUSER}_workers" ]; then
    mkdir data/"${IUSER}"_workers
fi
if [ ! -d "data/${IUSER}_workers/spw=${SPW}" ]; then
    mkdir data/"${IUSER}"_workers/spw=${SPW}
fi
if [ ! -d "data/${IUSER}_workers/spw=${SPW}/mode=${SAMPLEMODE}" ]; then
    mkdir data/"${IUSER}"_workers/spw=${SPW}/mode=${SAMPLEMODE}
fi


if [ "$CONT_SCRIPT" = true ] && [ ! $SAMPLEMODE = "na" ]; then
    if [ -d "data/${IUSER}_workers/spw=${SPW}/mode=${SAMPLEMODE}/sampled_data" ] && [ "$(ls -A data/"${IUSER}"_workers/spw=${SPW}/mode=${SAMPLEMODE}/sampled_data)" ]; then
        CONT_SCRIPT=false
    else
        if [ ! -d "data/${IUSER}_workers/spw=${SPW}/mode=${SAMPLEMODE}/sampled_data" ]; then
            mkdir data/"${IUSER}"_workers/spw=${SPW}/mode=${SAMPLEMODE}/sampled_data
        fi

        cd ../utils

        # Defaults to -1 if not specified, causes script to randomly generate seed
        SEED_ARGUMENT="${SAMPLING_SEED:--1}" 

        LEAF_DATA_META_DIR=${META_DIR} python3 sample.py $NAMETAG $SAMPLEMODETAG $IUSERTAG $SAMPPERWORKER $SFRACTAG --seed ${SEED_ARGUMENT}

        cd ../$NAME
    fi
fi


# create train-test split
TFRACTAG=""
if [ ! -z $TFRAC ]; then
    TFRACTAG="--frac $TFRAC"
fi

if [ "$CONT_SCRIPT" = true ]; then
    if [ -d "data/${IUSER}_workers/spw=${SPW}/mode=${SAMPLEMODE}/train" ] && [ "$(ls -A data/"${IUSER}"_workers/spw=${SPW}/mode=${SAMPLEMODE}/train)" ]; then
        CONT_SCRIPT=false
    else
        if [ ! -d "data/${IUSER}_workers/spw=${SPW}/mode=${SAMPLEMODE}/train" ]; then
            mkdir data/"${IUSER}"_workers/spw=${SPW}/mode=${SAMPLEMODE}/train
        fi
        if [ ! -d "data/${IUSER}_workers/spw=${SPW}/mode=${SAMPLEMODE}/test" ]; then
            mkdir data/"${IUSER}"_workers/spw=${SPW}/mode=${SAMPLEMODE}/test
        fi

        cd ../utils

        # Defaults to -1 if not specified, causes script to randomly generate seed
        SEED_ARGUMENT="${SPLIT_SEED:--1}"

        LEAF_DATA_META_DIR=${META_DIR} python3 split_data.py $NAMETAG $NUSERTAG $SAMPLEMODETAG $SAMPPERWORKER $TFRACTAG --seed ${SEED_ARGUMENT}

        cd ../$NAME
    fi
fi

if [ "$CONT_SCRIPT" = false ]; then
    echo "Data for one of the specified preprocessing tasks has already been"
    echo "generated. If you would like to re-generate data for this directory,"
    echo "please delete the existing one. Otherwise, please remove the"
    echo "respective tag(s) from the preprocessing command."
fi
