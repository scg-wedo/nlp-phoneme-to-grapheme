wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz

mkdir ./kenlm/build

cd ./kenlm/build

cmake ..

make -j2

# Trainning phrase (pre-setting)
cp -av '../../data/lm.txt' '.'

bin/lmplz -o 4 --text ./lm.txt --arpa ../../model/p2g_1.arpa --discount_fallback

pip install gdown

mkdir ./dict

cd ./dict

gdown -- 1E3YALgqICZvPszcOJrj9dKMM8QBM3hRn




