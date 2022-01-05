#!/usr/bin/env bash

###########################################################
RAW_DATA_DIR=data             # where the un-preprocessed data files are
DATA_DIR=data_preprocessed    # where the preprocessed data files are written to
                              # (value must be different thant RAW_DATA_DIR!)
DATASET_NAME=java-small       # name of dataset
MAX_CONTEXTS=200              # max data contexts to use in training
TOKEN_VOCAB_SIZE=186277       # the number of tokens and target words to keep ..
TARGET_VOCAB_SIZE=26347       # .. in the vocabulary (the top occurring words and paths will be kept).
PYTHON=python3
###########################################################

TRAIN_DATA_FILE=${RAW_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.train.c2s
VAL_DATA_FILE=${RAW_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.val.c2s
TEST_DATA_FILE=${RAW_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.test.c2s

mkdir -p ${DATA_DIR}
mkdir -p ${DATA_DIR}/${DATASET_NAME}

TARGET_HISTOGRAM_FILE=${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.tgt.c2s
SOURCE_TOKEN_HISTOGRAM=${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.ori.c2s
NODE_HISTOGRAM_FILE=${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.node.c2s

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${SOURCE_TOKEN_HISTOGRAM}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${NODE_HISTOGRAM_FILE}

echo "Running Preprocess.py (Creating Dictionary and format files)"
${PYTHON} preprocess.py --train_data ${TRAIN_DATA_FILE} --test_data ${TEST_DATA_FILE} --val_data ${VAL_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --token_vocab_size ${TOKEN_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --token_histogram ${SOURCE_TOKEN_HISTOGRAM} \
  --node_histogram ${NODE_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} \
  --output_name ${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}
