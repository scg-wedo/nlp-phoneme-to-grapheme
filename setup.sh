wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz

mkdir ./kenlm/build

cd ./kenlm/build

cmake ..

make -j2

cd ../..

# Trainning phrase (pre-setting)
cp -av './data/lm.txt' './kenlm/build'

bin/lmplz -o 4 --text ./kenlm/build/lm.txt --arpa ./model/p2g.arpa --discount_fallback

pip install gdown

mkdir ./dict

cd ./dict

gdown -- 1E3YALgqICZvPszcOJrj9dKMM8QBM3hRn




