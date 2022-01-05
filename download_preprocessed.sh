mkdir data

wget https://s3.amazonaws.com/code2seq/datasets/java-small-preprocessed.tar.gz
#wget https://s3.amazonaws.com/code2seq/datasets/java-large-preprocessed.tar.gz

tar -xvzf java-small-preprocessed.tar.gz
#tar -xvzf java-large-preprocessed.tar.gz

mv java-small data
#mv java-large data
