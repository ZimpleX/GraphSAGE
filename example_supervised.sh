#python -m graphsage.supervised_train --train_prefix ./data.ignore/reddit/reddit --model graphsage_mean --mean
python3 -m graphsage.supervised_train --train_prefix ./example_data/ppi --model graphsage_mean --sigmoid --epochs 1
