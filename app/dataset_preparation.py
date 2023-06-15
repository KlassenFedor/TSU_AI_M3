from pydub import AudioSegment
import numpy as np
import json


class DatasetCreator:
    def prepare_dataset(self):
        self.__prepare_background_clips()
        self.__prepare_and_augment_clips()

    def __prepare_background_clips(self):
        start_time = 0
        wav_file = AudioSegment.from_wav('../dataset/stream_test.wav')
        t1 = start_time
        t2 = t1 + 1000
        for i in range(20000):
            newAudio = wav_file[t1:t2]
            t1 = t2
            t2 += 1000
            newAudio.export('../dataset/back_clips/clip' + str(i) + '.wav', format="wav")

    def __prepare_and_augment_clips(self):
        insert_change_rate = 0.2
        stones_number = 16
        dataset_size = 20011
        dataset_dict = {}

        for i in range(dataset_size):
            background_clip = AudioSegment.from_wav('../dataset/back_clips/clip' + str(i) + '.wav')
            if np.random.rand() <= insert_change_rate:
                file_path = '../dataset/stones/' + str(np.random.randint(1, 17)) + '.wav'
                stone_clip = self.__match_target_amplitude(file_path, np.random.uniform(-10, 5))
                stone_clip = stone_clip.speedup(np.random.uniform(1.0, 1.3))
                sign = np.random.rand()
                val = int(np.random.uniform(0, 2000))
                if sign >= 0.5:
                    data = stone_clip.get_array_of_samples()[:-val]
                else:
                    data = stone_clip.get_array_of_samples()[val:]
                stone_clip = AudioSegment(data.tobytes(), frame_rate=stone_clip.frame_rate,
                                          sample_width=stone_clip.sample_width, channels=1)
                join = background_clip.overlay(stone_clip)
                join.export('../dataset/dataset_join/clip' + str(i) + '_1_.wav',
                            format="wav")
                dataset_dict['../dataset/dataset_join/clip' + str(i) + '_1_.wav'] = 1
            else:
                background_clip = self.__match_target_amplitude(
                    '../dataset/clip' + str(i) + '.wav', np.random.uniform(-5, 10))
                background_clip.export('../dataset/dataset_join/clip' + str(i) + '_0_.wav',
                                       format="wav")
                dataset_dict['../dataset/dataset_join/clip' + str(i) + '_0_.wav'] = 0

        with open('../dataset/dataset_json.json', 'w') as file:
            json.dump(dataset_dict, file, ensure_ascii=False)

    def __match_target_amplitude(self, sound, target_dBFS):
        clip = AudioSegment.from_wav(sound)
        clip += target_dBFS
        return clip