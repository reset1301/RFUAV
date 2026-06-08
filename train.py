# A script to train the detect model and classification model on spectrogram image.
from utils.trainer import CustomTrainer
from utils.trainer import DetTrainer


def main():

    # classification Trainer
    model = CustomTrainer(cfg='configs/exp0_test.yaml')
    model.train()

    # Detection Trainer
    save_dir = ''
    # todo configurable
    model = DetTrainer(model_name='yolo', dataset_dir = 'c:/111/detectors/RFUAV/dataset')
    model.train(save_dir=save_dir)


if __name__ == '__main__':
    main()