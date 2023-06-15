import fire
import logging
import tensorflow as tf
import numpy as np
from keras import layers, models
logging.basicConfig(level=logging.DEBUG, filename='../data/log_file.log', filemode='a')


class Model:
    def __init__(self):
        self.model = tf.keras.saving.load_model('../models/my_model_v3.h5')
        self.autotune = tf.data.AUTOTUNE
        self.commands = [1, 0]
        self.spectrogram_ds = None
        self.model_structure = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_dataset(self):
        """

        Creating train, validate and test data for dataset
        For storing information used json file which contains files and their labels

        """

        import json
        json_file = open('../dataset/dataset_json.json')
        json_str = json_file.read()
        data = json.loads(json_str)
        dataset_dict = data
        filenames = [k for k, v in dataset_dict.items()]
        train_files = filenames[:16000]
        val_files = filenames[16000: 16000 + 2000]
        test_files = filenames[16000 + 2000:]
        files_ds = tf.data.Dataset.from_tensor_slices(train_files)

        waveform_ds = files_ds.map(
            map_func=self.__get_waveform_and_label,
            num_parallel_calls=self.autotune)
        spectrogram_ds = waveform_ds.map(
            map_func=self.__get_spectrogram_and_label_id,
            num_parallel_calls=self.autotune)
        self.train_ds = spectrogram_ds
        self.val_ds = self.__preprocess_dataset(val_files)
        self.test_ds = self.__preprocess_dataset(test_files)

        batch_size = 128
        self.train_ds = self.train_ds.batch(batch_size)
        self.val_ds = self.val_ds.batch(batch_size)

        self.train_ds = self.train_ds.cache().prefetch(self.autotune)
        self.val_ds = self.val_ds.cache().prefetch(self.autotune)

    def predict_from_file(self, file_path):
        """

        Makes a prediction for an audio file
        :param file_path: path to file
        :return: 1 or 0

        """

        sample_ds = self.__preprocess_dataset([str(file_path)])
        for spectrogram, label in sample_ds.batch(1):
            prediction = self.model(spectrogram)

        return 1 if prediction[0][0] > 0.6 else 0

    def predict_from_data(self, data):
        """
        Makes a forecast from pure data
        :param data: array-like
        :return: 1 or 0
        """
        prediction = self.model(self.__get_spectrogram(
            tf.convert_to_tensor(list(np.array(data) / 32767.0), dtype=tf.float32)
        )[None, :, :, :])

        return 1 if prediction[0][0] > 0.6 else 0

    def __build_model(self):
        """
        Preparing the model structure
        """
        for spectrogram, _ in self.spectrogram_ds.take(1):
            input_shape = spectrogram.shape
        num_labels = len(self.commands)

        norm_layer = layers.Normalization()
        norm_layer.adapt(data=self.spectrogram_ds.map(map_func=lambda spec, label: spec))

        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Resizing(32, 32),
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        self.model_structure = model

    def train(self,  epochs):
        """
        Trains the model
        :param epochs: number of epochs
        :return:
        """
        self.prepare_dataset()
        self.__build_model()
        self.model_structure.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        history = self.model_structure.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )
        self.model_structure.save('../models/my_model.h5')

    def __preprocess_dataset(self, files):
        """

        Dataset preprocessing
        :param files: dataset files

        """

        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=self.__get_waveform_and_label,
            num_parallel_calls=self.autotune)
        output_ds = output_ds.map(
            map_func=self.__get_spectrogram_and_label_id,
            num_parallel_calls=self.autotune)
        return output_ds

    def __get_waveform_and_label(self, file_path):
        """

        Getting waveform and file label
        :param file_path: path to file

        """

        label = self.__get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = self.__decode_audio(audio_binary)
        return waveform, label

    def __get_spectrogram(self, waveform):
        """

        Creating spectrogram from waveform data
        :param waveform: waveform data

        """

        input_len = 16000
        waveform = waveform[:input_len]
        zero_padding = tf.zeros(
            [16000] - tf.shape(waveform),
            dtype=tf.float32)
        waveform = tf.cast(waveform, dtype=tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def __decode_audio(self, audio_binary):
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
        return tf.squeeze(audio, axis=-1)

    def __get_label(self, file_path):
        print(1, file_path)
        parts = tf.strings.split(
            input=file_path,
            sep='_')
        return int(parts[-2])

    def __get_spectrogram_and_label_id(self, audio, label):
        spectrogram = self.__get_spectrogram(audio)
        label_id = tf.argmax(label == self.commands)
        return spectrogram, label_id


if __name__ == '__main__':
    fire.Fire(Model)
