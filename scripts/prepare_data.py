import csv
import random
import os
import argparse

current_path = os.getcwd()

def get_all_uttr_name(read_path):
    utterances_name = []
    for f in os.listdir(read_path):
        if os.path.isfile(os.path.join(read_path, f)):
            utterances_name.append(f[:f.rfind('.')]+'|\n')
        elif os.path.isdir(os.path.join(read_path, f)):
            utterances_name.extend(get_all_uttr_name(os.path.join(read_path, f)))
    return utterances_name
    
def split_train_val(utterances_name, val_size):
    random.shuffle(utterances_name)
    split_index = val_size
    val_uttrs = utterances_name[:split_index]
    train_uttrs = utterances_name[split_index:]
    return train_uttrs, val_uttrs

def write_uttr_name(wav_path, save_path, train_uttrs, val_uttrs):
    os.makedirs(save_path, exist_ok=True)
    
    f = open(os.path.join(save_path, 'train.txt'), 'w', encoding='UTF-8')
    f.writelines(train_uttrs)
    f.close()
    
    f = open(os.path.join(save_path, 'val.txt'), 'w', encoding='UTF-8')
    f.writelines(val_uttrs)
    f.close()
    
    wav_link_path = os.path.join(save_path, 'wavs')
    if not os.path.exists(wav_link_path):
        os.symlink(wav_path, wav_link_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_path', default=os.path.join(current_path, 'wavs'), help="path to wav files")
    parser.add_argument('--output_path', default=os.path.join(current_path, os.pardir, 'dataset'), help="path to saved data")
    parser.add_argument('--val_size', default=100, help="number of validation wav")
    args = parser.parse_args()
    
    print("reading data...")
    utterances_name = get_all_uttr_name(args.input_wavs_path)
    train_uttrs, val_uttrs = split_train_val(utterances_name, int(args.val_size))
    print("number of train wavs: " + str(len(train_uttrs)))
    print("number of validation wavs: " + str(len(val_uttrs)))
    write_uttr_name(args.input_wavs_path, args.output_path, train_uttrs, val_uttrs)
    print("data is saved.")

if __name__ == '__main__':
    # python scripts/prepare_data.py --input_wavs_path "/content/drive/MyDrive/Projects/TTS/multi speaker/main_files/processed_data/train_wav" --output_path "dataset" --val_size 100
    main()
