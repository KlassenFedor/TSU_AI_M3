import requests
import time
from pydub import AudioSegment
from array import array
from app.model import Model
import fire


class StreamAnalyzer:
    def __init__(self):
        self.attempts_number = 10
        self.waiting_time = 5
        self.samplerate = 16000
        self.window = 4000
        self.model = Model()

    def analyze_stream(self, url):
        flag = False
        word_num = 0
        blocks = []
        current_position = 0
        j = 0

        for i in range(self.attempts_number):
            print(i)
            flag = True
            try:
                r = requests.get(url, stream=True)
                print(r.status_code)
            except:
                flag = False
                time.sleep(self.waiting_time)

            if flag:

                for block in r.iter_content(16384, decode_unicode=True):
                    j += 1
                    with open('D:/TSU/HITS/TSU_AI_M3/notebooks/test/myfile_3_.mp3', 'wb') as f:
                        f.write(block)
                    sound = AudioSegment.from_mp3('D:/TSU/HITS/TSU_AI_M3/notebooks/test/myfile_3_.mp3')
                    sound = sound.set_frame_rate(self.samplerate)
                    data = sound.get_array_of_samples()
                    blocks.extend(data)
                    if len(blocks) >= 40000:
                        current_clip = blocks[current_position:current_position + self.samplerate]
                        if self.model.predict_from_data(current_clip):
                            print('word detected', j)
                            print(current_position, current_position + self.samplerate)
                            word_num += 1
                            secret_start = current_position + 3200 + self.samplerate
                            secret_end = current_position + self.samplerate * 2 + 3200
                            secret_clip = blocks[secret_start:secret_end]
                            AudioSegment(array('h', secret_clip), frame_rate=self.samplerate,
                                         sample_width=sound.sample_width, channels=1) \
                                .export('D:/TSU/HITS/TSU_AI_M3/notebooks/test/secret' + str(word_num) + '_5_.wav',
                                        format='wav')
                            current_position += self.samplerate * 2 + 3200
                        else:
                            current_position += self.window

        while current_position + 16000 < len(blocks):
            j += 1
            current_clip = blocks[current_position:current_position + self.samplerate]
            if self.model.predict_from_data(current_clip):
                print('word detected', j)
                print(current_position, current_position + self.samplerate)
                word_num += 1
                secret_start = current_position + 3200 + self.samplerate
                secret_end = current_position + self.samplerate * 2 + 3200
                secret_clip = blocks[secret_start:secret_end]
                AudioSegment(array('h', secret_clip), frame_rate=self.samplerate, sample_width=sound.sample_width,
                             channels=1) \
                    .export('D:/TSU/HITS/TSU_AI_M3/notebooks/test/secret' + str(word_num) + '_5_.wav', format='wav')
                current_position += self.samplerate * 2 + 3200
            else:
                current_position += self.window


if __name__ == '__main__':
    fire.Fire(StreamAnalyzer)